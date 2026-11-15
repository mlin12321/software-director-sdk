"""The director's state for one server: an org tree and its team_memory."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID, uuid4

from openhands.agent_server.director.boundary import (
    hand_over_refusal,
    nearest_team,
    write_refusal,
    written_paths,
)
from openhands.agent_server.director.git import (
    is_git_repository,
)
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.models import (
    Boundary,
    CreateOrgRequest,
    OrgNode,
    OrgNodeInput,
    RoutedMessage,
    RouteMessageRequest,
    StartAgainRequest,
)
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
)
from openhands.agent_server.director.pods import PodLauncher
from openhands.agent_server.director.prompts import (
    NEED_WRITE,
    POD_PROTOCOL,
)
from openhands.agent_server.director.recovery import (
    has_work,
)
from openhands.agent_server.director.run_state import (
    Intent,
)
from openhands.agent_server.director.runtime import bind_director
from openhands.agent_server.director.service.state import (
    AGENT_PAUSED,
    DIRECTOR_TOOL_NAMES,
    HAND_OVER_TOOLS,
    RESTART_NOTE,
    ROOT_ID,
    TOOLS_OF,
    _agent_title,
    _base_of,
    _branch_of,
    _checkout_root,
    _is_in,
    _new_pod,
    _summary_of,
    agent_conversation,
    hand_over_of,
)
from openhands.agent_server.director.service.sync import Sync
from openhands.sdk.conversation.request import AgentLaunchAdditions
from openhands.sdk.errors import (
    Conflict,
    NotFound,
    Unavailable,
    Unprocessable,
)
from openhands.sdk.event import ActionEvent
from openhands.sdk.tool.spec import Tool


logger = logging.getLogger(__name__)


__all__ = [
    "AGENT_PAUSED",
    "DIRECTOR_TOOL_NAMES",
    "HAND_OVER_TOOLS",
    "RESTART_NOTE",
    "TOOLS_OF",
    "DirectorService",
    "agent_conversation",
    "hand_over_of",
]


class DirectorService(Sync):
    """Owns the org tree and the team memory of one repository.

    An org lives in the repository its first conversation was opened in,
    which need not be the directory the server was started for; the director
    remembers it across restarts. Until there is an org, the director looks
    for a saved one in the server's directory. A repository that has no org
    and is a submodule of one that has stands for the nearest such
    repository above it, in both cases.

    Nothing is read from disk until the tree or the team memory is first used,
    so creating the service is free for servers that never use the director.

    The tree can be read and taught without :meth:`start`. Running agents
    needs it: starting them, sending them messages and saving the tree all
    happen on worker tasks, so nothing that asks (a route, an agent's tool
    on its own thread) waits on a conversation. What is asked for goes into
    a lane, usually its team's: a lane's work is carried out in the order it
    was asked for, and lanes do not wait for each other.
    """

    async def start(self, launcher: PodLauncher) -> None:
        """Begin running agents through ``launcher``, on the running loop."""
        self._loop = asyncio.get_running_loop()
        self._launcher = launcher
        self._lanes = {}
        self._unsettled = 0
        self._quiet.set()
        self._stop_status = launcher.on_status(self._on_status)
        self._stop_guard = launcher.on_action(self.guard_action)
        self._bwrap = await asyncio.to_thread(self._find_sandbox)
        if self._bwrap is None:
            logger.info("Director cannot confine shells here; it checks file tools")
        bind_director(self)
        self._stall_watch = asyncio.create_task(
            self._watch_for_stalls(), name="director-stall-watch"
        )
        if self._store is not None:
            self._enqueue(self._came_back)
        elif await asyncio.to_thread(self._remembered_repo) is not None:
            # An org that was left in the middle of its work is looked at
            # now, not when somebody first asks for it.
            await asyncio.to_thread(lambda: self.store)

    async def aclose(self) -> None:
        """Stop running agents and save the tree if a save was pending.

        An org that is in the middle of its work is left paused, so that the
        next start finds what to take up again.
        """
        bind_director(None)
        store = self._store
        if store is not None and store.root_id is not None and not self._fixture:
            run = self._runs()
            if self._paused is None and has_work(run):
                self._hold("shutdown")
            run.put("clean", "1")
        if self._stop_status is not None:
            self._stop_status()
            self._stop_status = None
        if self._stop_guard is not None:
            self._stop_guard()
            self._stop_guard = None
        workers = list(self._workers)
        if self._stall_watch is not None:
            workers.append(self._stall_watch)
            self._stall_watch = None
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        # A save that was cut off goes on in its thread: nothing of this
        # director may write the tree after it has closed.
        await asyncio.to_thread(self._wait_for_save)
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
            await self._save()
        self._launcher = None
        self._loop = None
        with self._lock:
            if self._run is not None:
                self._run.close()
                self._run = None

    async def create_org(self, request: CreateOrgRequest) -> OrgNode:
        """Start the root team of a new org and brief its IC with the task.

        Raises:
            Conflict: The director already has an org, or serves the fixture.
            Unavailable: The director is not running, or the conversation's
                workspace is not in a git repository.
            Unprocessable: The conversation cannot lead an org.
        """
        launcher = self._require_launcher()
        if self._fixture:
            raise Conflict("The director serves a fixture tree; it cannot run an org")
        store = await asyncio.to_thread(lambda: self.store)
        conversation = request.conversation
        workspace = Path(conversation.workspace.working_dir).resolve()
        if not await asyncio.to_thread(is_git_repository, workspace):
            raise Unavailable(f"An org needs a git repository; {workspace} is not one")
        repository = await asyncio.to_thread(_checkout_root, str(workspace))
        if conversation.conversation_runtime == "docker":
            raise Unprocessable(
                "An org's agents run in the server's own process, not in Docker"
            )

        ic_id = conversation.conversation_id or uuid4()
        named = provisional_summary(request.task)
        title = request.title or named.title
        additions = conversation.agent_launch_additions or AgentLaunchAdditions()
        suffix = "\n\n".join(
            part
            for part in (additions.system_message_suffix_append, POD_PROTOCOL)
            if part
        )
        async with self._create_lock:
            if store.root_id is not None:
                raise Conflict("The director already has an org")
            configured, self._repo_dir = self._repo_dir, repository
            store.upsert(
                OrgNodeInput(
                    id=ROOT_ID,
                    parent_id=None,
                    title=title,
                    boundary=Boundary(path="", writable=["**"]),
                    isolation="worktree" if conversation.worktree else "shared",
                    worktree_path=str(workspace),
                    branch="",
                    agents=_new_pod(ic_id),
                    status="idle",
                    headline=named.headline,
                    spec_summary=_summary_of(request.task),
                )
            )
            try:
                tool_params = await asyncio.to_thread(self._confine, ROOT_ID, "ic")
                started = await launcher.start_root(
                    conversation.model_copy(
                        update={
                            "conversation_id": ic_id,
                            "initial_message": None,
                            # Named like every other agent of the org.
                            "autotitle": False,
                            "agent_launch_additions": additions.model_copy(
                                update={"system_message_suffix_append": suffix}
                            ),
                        }
                    ),
                    [Tool(name=name) for name in TOOLS_OF["ic"]],
                    title=_agent_title(title, "ic"),
                    tool_params=tool_params,
                    adapt_agent=await asyncio.to_thread(
                        self._adapter, str(ic_id), TOOLS_OF["ic"]
                    ),
                )
            except BaseException:
                store.remove(ROOT_ID)
                self._repo_dir = configured
                raise
            await asyncio.to_thread(self._remember, repo_path=str(self._repo_dir))
            await asyncio.to_thread(self._working)
            store.set_agent(
                ROOT_ID, "ic", kind=started.kind, acp_server=started.acp_server
            )
            root = store.update_node(
                ROOT_ID,
                worktree_path=started.workspace,
                isolation=started.isolation,
                branch=await asyncio.to_thread(_branch_of, started.workspace),
                base_commit=await asyncio.to_thread(_base_of, started.workspace),
                boundary_enforcement=self._enforcement(started.kind),
            )
        # First, so that both agents are told the name the team keeps.
        self._enqueue(
            lambda: self._summarize(
                ROOT_ID, request.task, rename=request.title is None, first=True
            ),
            ROOT_ID,
        )
        self._begin("brief", ROOT_ID, role="ic", task=request.task)
        return root

    async def route_message(self, request: RouteMessageRequest) -> RoutedMessage:
        """Hand what the human says to the team nearest to where they say it.

        Without an org, the message joins the one saved in the repository of
        the conversation it comes with (or above it), and failing that
        starts one there and becomes its task.

        Raises:
            Unprocessable: The path is outside the org's repository, or there
                is no org and the request does not say how to start one.
            Conflict: No team of the org can take the message.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = await asyncio.to_thread(lambda: self.store)
        async with self._first_message_lock:
            conversation = request.conversation
            if store.root_id is None and conversation is None:
                raise Unprocessable(
                    "There is no org yet. Send `conversation` with the message "
                    "so that it can start one."
                )
            if store.root_id is None and conversation is not None:
                joined = not self._fixture and await asyncio.to_thread(
                    self._join, Path(conversation.workspace.working_dir)
                )
                if not joined:
                    root = await self.create_org(
                        CreateOrgRequest(task=request.text, conversation=conversation)
                    )
                    return RoutedMessage(
                        node_id=root.id,
                        conversation_id=agent_conversation(root, "ic") or "",
                        started_org=True,
                    )
                await self._came_back()
        path = await asyncio.to_thread(
            lambda: (
                self._home()
                if request.path is None
                else self._repo_relative(request.path)
            )
        )
        node = nearest_team(store.all(), path, lost=AGENT_LOST)
        if node is None and request.conversation is not None:
            # Nobody here can take it: the nearest team whose agents have no
            # conversation on this server is started again to take it.
            gone = nearest_team(store.all(), path, lost="")
            if gone is not None and gone.status != "done":
                node = await self.start_again(
                    gone.id, StartAgainRequest(conversation=request.conversation)
                )
        if node is None:
            raise Conflict(
                "No team of the org can take a message about "
                f"{path or 'the repository'!r}"
            )
        text = request.text
        if path != node.boundary.path:
            text = f"{text}\n\n(The human sent this from {path}.)"
        conversation_id = self.tell_ic(node.id, text, human=True) or ""
        return RoutedMessage(
            node_id=node.id, conversation_id=conversation_id, started_org=False
        )

    def guard_action(self, conversation_id: UUID, event: ActionEvent) -> str | None:
        """Why an agent of the org may not carry out an action, if it may not.

        A file tool is refused a write outside the boundary of the agent's
        team. Called on the agent's own thread, before the action runs.
        """
        store = self._store
        if store is None or event.action is None:
            return None
        found = store.find_by_conversation(str(conversation_id))
        paths = written_paths(event.tool_name, event.action.model_dump())
        if not paths:
            return None
        if found is None:
            in_hand_over = store.find_hand_over(str(conversation_id)) is not None
            return hand_over_refusal(paths[0]) if in_hand_over else None
        node, role = found
        # A team that has not started holds no one out of its directory:
        # the IC above may finish what it has there.
        sub_teams = [team for team in store.children(node.id) if team.agents]
        top = _checkout_root(node.worktree_path)
        for named in paths:
            target = Path(os.path.realpath(Path(node.worktree_path) / named))
            if _is_in(target, top):
                relative = target.relative_to(top).as_posix()
                refusal = write_refusal(
                    node, role, sub_teams, "" if relative == "." else relative
                )
                if refusal is not None:
                    if role == "ic":
                        self.record_need(node, relative, NEED_WRITE)
                    return refusal
            elif any(_is_in(target, checkout) for checkout in self._checkouts()):
                return (
                    f"{named!r} is in a checkout of the org that is not your "
                    "team's, so it was not changed."
                )
        return None

    def answer_escalation(
        self, node_id: str, escalation_id: str, answer: str
    ) -> OrgNode:
        """Settle an escalation that waits on the human with their answer.

        The answer goes to the agent that raised it.

        Raises:
            NotFound: The node or the escalation does not exist.
            Conflict: The escalation is still held by a team.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = self.store
        node = store.get(node_id)
        escalation = next(
            (e for e in node.escalations if e.id == escalation_id)
            if node
            else iter(()),
            None,
        )
        if node is None or escalation is None:
            raise NotFound(f"Node {node_id} has no escalation {escalation_id}")
        if escalation.held_by is not None:
            raise Conflict(
                f"Escalation {escalation_id} is still with team "
                f"{escalation.held_by}; it does not wait on the human"
            )
        store.settle_escalation(node_id, escalation_id)
        self.notify(
            agent_conversation(node, escalation.raised_by),
            f"The human answered your escalation {escalation.title!r}:\n{answer}",
            at_once=True,
        )
        return store.get(node_id) or node

    def _steps(self) -> dict[str, Callable[[Intent], Awaitable[None]]]:
        return {
            "brief": self._do_brief,
            "launch_manager": self._do_launch_manager,
            "split": self._carry_out_split,
            "start_team": self._do_start_team,
            "finish_split": self._do_finish_split,
            "hand_over": self._do_hand_over,
            "request_review": self._do_request_review,
            "accept": self._do_accept,
            "prune": self._prune,
        }
