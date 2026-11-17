"""The director's state for one server: an org tree and its team_memory."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID

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
from openhands.agent_server.director.merge_checkout import MergeCheckout
from openhands.agent_server.director.models import (
    Boundary,
    ChosenBy,
    CreateOrgRequest,
    OpenedSession,
    OpenSessionRequest,
    OrgNodeInput,
    RoutedMessage,
    RouteMessageRequest,
    SessionTeam,
)
from openhands.agent_server.director.org_store import (
    reviews_only,
)
from openhands.agent_server.director.pods import PodLauncher
from openhands.agent_server.director.prompts import (
    NEED_WRITE,
)
from openhands.agent_server.director.recovery import (
    has_work,
)
from openhands.agent_server.director.run_state import (
    MERGING,
    Intent,
)
from openhands.agent_server.director.runtime import bind_director
from openhands.agent_server.director.service.state import (
    AGENT_PAUSED,
    BASE_LANE,
    DIRECTOR_TOOL_NAMES,
    HAND_OVER_TOOLS,
    MERGE_WAITS,
    ORG_WAITS,
    RESTART_NOTE,
    ROOT_ID,
    TOOLS_OF,
    _checkout_root,
    _instant,
    _is_in,
    _summary_of,
    agent_conversation,
    hand_over_of,
    tools_of,
)
from openhands.agent_server.director.service.sync import Sync
from openhands.sdk.errors import (
    Conflict,
    NotFound,
    Unavailable,
    Unprocessable,
)
from openhands.sdk.event import ActionEvent


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
    "tools_of",
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
        if store is not None and store.root_id is not None:
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

    async def create_org(self, request: CreateOrgRequest) -> OpenedSession:
        """Start a new org: its top team, and the session that works on
        the task, whose IC is the conversation the task came in.

        Raises:
            Conflict: The director already has an org.
            Unavailable: The director is not running, or the conversation's
                workspace is not in a git repository.
            Unprocessable: The conversation cannot lead an org.
        """
        self._require_launcher()
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
        named = provisional_summary(request.task)
        title = request.title or named.title
        async with self._create_lock:
            if store.root_id is not None:
                raise Conflict("The director already has an org")
            configured, self._repo_dir = self._repo_dir, repository
            # The top team as the org keeps it: what it is for and where.
            # Nobody works at it but in a session.
            store.upsert(
                OrgNodeInput(
                    id=ROOT_ID,
                    parent_id=None,
                    title=title,
                    boundary=Boundary(path="", writable=["**"]),
                    isolation="shared",
                    headline=named.headline,
                    spec_summary=_summary_of(request.task),
                    since=_instant(),
                )
            )
            try:
                opened = await self.open_session(
                    OpenSessionRequest(
                        text=request.task, title=request.title or "", node_id=ROOT_ID
                    ),
                    conversation,
                    # An org without a name is named after its first task.
                    rename=request.title is None,
                )
            except BaseException:
                store.remove(ROOT_ID)
                self._repo_dir = configured
                raise
            await asyncio.to_thread(self._remember, repo_path=str(self._repo_dir))
            await asyncio.to_thread(self._working)
        if request.title is None:
            # The org is named after what its first session is for.
            top = opened.session.teams[0].id
            self._enqueue(lambda: self._name_org(top), BASE_LANE)
        return opened

    async def _name_org(self, top_id: str) -> None:
        """Give the top team the name a model made of the first task."""
        top = self.store.place(top_id)
        if top is not None and self.store.root_id is not None:
            self.store.update_team(
                self.store.root_id, title=top.title, headline=top.headline
            )

    async def _say_in_session(self, session_id: str, text: str) -> RoutedMessage:
        """Give what the human says in a session's conversation to the IC
        of one of the session's teams: the one it was opened at, unless the
        director is sure of one below.

        Raises:
            NotFound: The session is not open.
        """
        store = self.store
        places = [
            place for place in store.session(session_id) if not reviews_only(place)
        ]
        if not places:
            raise NotFound(f"Unknown session: {session_id}")
        top = place = places[0]
        chosen_by: ChosenBy = "path"
        reason = ""
        if self.settings.route_requests and len(places) > 1:
            choice = await self.choose_team(text, top, within=session_id)
            chosen_by, reason = choice.by, choice.reason
            place = store.instance(choice.node_id, session_id) or top
        conversation_id = await asyncio.to_thread(
            lambda: self.tell_ic(place.id, text, human=True)
        )
        if place.id != top.id:
            # Its IC may say it is not the team's own: the IC above has it then.
            await asyncio.to_thread(
                self.keep_routed, "asked", place, text, chosen_by, reason
            )
        return RoutedMessage(
            node_id=place.id,
            team=place.title,
            conversation_id=conversation_id or "",
            started_org=False,
            chosen_by=chosen_by,
            reason=reason,
        )

    async def route_message(self, request: RouteMessageRequest) -> RoutedMessage:
        """Open a session for what the human asks, or with ``session_id``
        say it in the session that is open.

        With ``auto`` the director chooses the team the request belongs to.
        Without it the request goes to the team nearest to where the human
        said it, and no further: the human wrote to that team.

        Without an org, the message joins the one saved in the repository of
        the conversation it comes with (or above it), and failing that
        starts one there and becomes its task.

        Raises:
            Unprocessable: The path is outside the org's repository, or there
                is no org and the request does not say how to start one.
            Conflict: No session can be opened for the message.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = await asyncio.to_thread(lambda: self.store)
        if request.session_id is not None:
            return await self._say_in_session(request.session_id, request.text)
        async with self._first_message_lock:
            conversation = request.conversation
            if store.root_id is None and conversation is None:
                raise Unprocessable(
                    "There is no org yet. Send `conversation` with the message "
                    "so that it can start one."
                )
            if store.root_id is None and conversation is not None:
                joined = await asyncio.to_thread(
                    self._join, Path(conversation.workspace.working_dir)
                )
                if not joined:
                    opened = await self.create_org(
                        CreateOrgRequest(task=request.text, conversation=conversation)
                    )
                    return RoutedMessage(
                        node_id=opened.session.teams[0].id,
                        team=opened.session.team,
                        conversation_id=opened.conversation_id,
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
        node = nearest_team(store.teams(), path)
        if node is None:
            raise Conflict(
                "No team of the org can take a message about "
                f"{path or 'the repository'!r}"
            )
        text = request.text
        chosen_by: ChosenBy = "path"
        reason = ""
        if request.auto and self.settings.route_requests:
            choice = await self.choose_team(text, node)
            chosen_by, reason = choice.by, choice.reason
            node = store.team(choice.node_id) or node
        elif path != node.boundary.path:
            text = f"{text}\n\n(The human sent this from {path}.)"
        opened = await self.open_session(
            OpenSessionRequest(text=text, node_id=node.id), request.conversation
        )
        top = opened.session.teams[0]
        if chosen_by not in ("path", "top"):
            await asyncio.to_thread(
                self.keep_routed, "asked", top, request.text, chosen_by, reason
            )
        return RoutedMessage(
            node_id=top.id,
            team=top.title,
            conversation_id=opened.conversation_id,
            started_org=False,
            chosen_by=chosen_by,
            reason=reason,
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
        sub_teams = [team for team in store.below(node.id) if team.agents]
        top = _checkout_root(node.worktree_path)
        merges = {
            team_id: MergeCheckout.model_validate_json(kept)
            for team_id, kept in self._runs().marked(MERGING).items()
        }
        for named in paths:
            target = Path(os.path.realpath(Path(node.worktree_path) / named))
            merging = next(
                (
                    (team_id, merge)
                    for team_id, merge in merges.items()
                    if _is_in(target, Path(os.path.realpath(merge.where)))
                ),
                None,
            )
            if merging is not None:
                # Only what does not merge is its reviewer's to write.
                team = store.place(merging[0])
                if role != "ic" or team is None or team.parent_id != node.id:
                    return (
                        f"{named!r} is in a merge another team's reviewer "
                        "finishes, so it was not changed."
                    )
                files = merging[1].files
                relative = target.relative_to(os.path.realpath(merging[1].where))
                if relative.as_posix() not in files:
                    return (
                        f"{named!r} merged by itself, so it was not changed. "
                        f"Yours to edit in that merge: {', '.join(files)}"
                    )
            elif _is_in(target, top):
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
    ) -> SessionTeam:
        """Settle an escalation that waits on the human with their answer.

        The answer goes to the agent that raised it.

        Raises:
            NotFound: The node or the escalation does not exist.
            Conflict: The escalation is still held by a team.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = self.store
        node = store.place(node_id)
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
        if escalation_id == MERGE_WAITS:
            # The way into the checkout was cleared: the accepted work is
            # merged now, with nobody asked again.
            self._begin("accept", node_id, needed_above="", reconciled="")
            return store.place(node_id) or node
        if escalation_id == ORG_WAITS:
            # The human goes on without the Manager.
            self._begin("land_org", node_id, org_holds=True)
            return store.place(node_id) or node
        self.notify(
            agent_conversation(node, escalation.raised_by),
            f"The human answered your escalation {escalation.title!r}:\n{answer}",
            at_once=True,
        )
        return store.place(node_id) or node

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
            "land_org": self._do_land_org,
            "prune": self._prune,
            "close_session": self._close_session,
        }
