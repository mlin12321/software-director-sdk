"""The director's state for one server: an org tree and its team_memory."""

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from uuid import UUID

from openhands.agent_server.director.boundary import (
    hand_over_refusal,
    named_paths,
    nearest_team,
    searched_folder,
    write_refusal,
    written_paths,
)
from openhands.agent_server.director.git import (
    common_dir,
    is_git_repository,
)
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.launch import TeamLauncher
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
    StartAgainRequest,
)
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    is_division_lead,
)
from openhands.agent_server.director.prompts import (
    NEED_WRITE,
)
from openhands.agent_server.director.recovery import (
    has_work,
)
from openhands.agent_server.director.run_state import (
    CLEARING,
    MERGING,
    Intent,
)
from openhands.agent_server.director.runtime import bind_director
from openhands.agent_server.director.service.state import (
    AGENT_PAUSED,
    DIRECTOR_TOOL_NAMES,
    HAND_OVER_TOOLS,
    LIMIT_WAITS,
    MERGE_WAITS,
    ORG_WAITS,
    RESTART_NOTE,
    ROOT_ID,
    STALL_WAITS,
    TOOLS_OF,
    _checkout_root,
    _instant,
    _is_in,
    _summary_of,
    agent_conversation,
    failure_waits,
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

    async def start(self, launcher: TeamLauncher) -> None:
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
            logger.warning(
                "Director cannot confine shells here (no bubblewrap): it checks "
                "file tools, and nothing holds the process of an ACP agent"
            )
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
        workspace = await asyncio.to_thread(
            Path(conversation.workspace.working_dir).resolve
        )
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
            if await asyncio.to_thread(self._join, workspace):
                # The repository holds an org already, which this server
                # was not pointed at: it is taken up, never written over.
                await self._came_back()
                return await self.open_session(
                    OpenSessionRequest(text=request.task, title=request.title or ""),
                    conversation,
                )
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
            # In the lane of that team, after the model that names it.
            self._enqueue(lambda: self._in_base(lambda: self._name_org(top)), top)
        return opened

    async def _say_in_session(self, session_id: str, text: str) -> RoutedMessage:
        """Give what the human says in a session's conversation to the IC
        of one of the session's teams: the one it was opened at, unless the
        director is sure of one below.

        Raises:
            NotFound: The session is not open.
        """
        store = self.store
        places = [
            place for place in store.session(session_id) if not is_division_lead(place)
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

    def is_agent(self, conversation_id: UUID) -> bool:
        """Whether a conversation is one of an agent of the org."""
        store = self._store
        return store is not None and (
            store.find_by_conversation(str(conversation_id)) is not None
            or store.find_hand_over(str(conversation_id)) is not None
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
        action = event.action.model_dump()
        secret = self._remembered and os.path.realpath(self._remembered)
        workspace = found[0].worktree_path if found else ""
        for named in named_paths(event.tool_name, action):
            # Read, the key in it lets an agent act as any other.
            if os.path.realpath(os.path.join(workspace, named)) == secret:
                return f"{named!r} is the director's own, and is not yours to open."
        searched = searched_folder(event.tool_name, action)
        if secret and searched is not None:
            folder = os.path.realpath(os.path.join(workspace, searched))
            # What a search finds in it tells of it, a character at a time.
            if secret.startswith(folder.rstrip(os.sep) + os.sep):
                return (
                    f"{searched or 'The workspace'!r} holds a file of the "
                    "director's own: search a folder inside it instead."
                )
        paths = written_paths(event.tool_name, action)
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
                        # The refusal stands whether or not it could be kept.
                        with suppress(Exception):
                            self.record_need(node, relative, NEED_WRITE)
                    return refusal
            elif any(_is_in(target, checkout) for checkout in self._checkouts()):
                return (
                    f"{named!r} is in a checkout of the org that is not your "
                    "team's, so it was not changed."
                )
            elif _is_in(target, Path(os.path.realpath(common_dir(self._repo_dir)))):
                # A git directory that is not inside its checkout.
                return (
                    f"{named!r} is in the repository's git directory, so it "
                    "was not changed. Use git."
                )
        return None

    def answer_escalation(
        self, place_id: str, escalation_id: str, answer: str
    ) -> SessionTeam:
        """Settle an escalation that waits on the human with their answer.

        The answer goes to the agent that raised it. An escalation the
        director raised itself for a merge or a Manager that waits has
        nobody to send it to: answering only lets the work go on. One it
        raised for an agent that failed or a team without progress goes to
        that agent, which runs again; a team whose agent has no conversation
        here is started again.

        Raises:
            NotFound: The team or the escalation does not exist.
            Conflict: The escalation is still held by a team.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = self.store
        place = store.place(place_id)
        escalation = next(
            (e for e in place.escalations if e.id == escalation_id)
            if place
            else iter(()),
            None,
        )
        if place is None or escalation is None:
            raise NotFound(f"Team {place_id} has no escalation {escalation_id}")
        if escalation.held_by is not None:
            raise Conflict(
                f"Escalation {escalation_id} is still with team "
                f"{escalation.held_by}; it does not wait on the human"
            )
        role = escalation.raised_by
        lost = escalation_id == failure_waits(role) and any(
            a.role == role and a.status == AGENT_LOST for a in place.agents
        )
        if lost and role == "ic":
            # It has no conversation to answer in: the team starts again
            # with the answer, and is asked of no longer once it has.
            async def again() -> None:
                await self.start_again(place_id, StartAgainRequest(), said=answer)
                with suppress(NotFound):
                    store.settle_escalation(place_id, escalation_id)

            self._enqueue(again, place_id)
            return place
        store.settle_escalation(place_id, escalation_id)
        if escalation_id == MERGE_WAITS:
            # The way into the checkout was cleared: the accepted work is
            # merged now as its reviewer decided, with nobody asked again.
            decided = json.loads(self._runs().marked(CLEARING).get(place_id) or "{}")
            self.accept_review(place_id, "", **decided)
            return store.place(place_id) or place
        if escalation_id == ORG_WAITS:
            # The human goes on without the Manager.
            self._begin("land_org", place_id, org_holds=True)
            return store.place(place_id) or place
        if escalation_id == LIMIT_WAITS:
            self.go_on_past_limit(place_id)
        if lost:
            # A new Manager starts with the answer; the IC goes on as it was.
            self.tell_manager(
                place_id,
                "The Manager before you could not be started. The human was "
                f"told, and says:\n{answer}",
                division_lead=is_division_lead(place),
            )
            return store.place(place_id) or place
        told = (
            "The human was told that your conversation failed, and says"
            if escalation_id == failure_waits(role)
            else "The human was told that your team shows no progress, and says"
            if escalation_id == STALL_WAITS
            else "Your session had reached its cost limit and its agents were "
            "put to rest. The human lets it go on, and says"
            if escalation_id == LIMIT_WAITS
            else f"The human answered your escalation {escalation.title!r}"
        )
        self.notify(agent_conversation(place, role), f"{told}:\n{answer}", at_once=True)
        return store.place(place_id) or place

    def _steps(self) -> dict[str, Callable[[Intent], Awaitable[None]]]:
        return {
            "brief": self._do_brief,
            "launch_manager": self._do_launch_manager,
            "split": self._do_split,
            "start_team": self._do_start_team,
            "finish_split": self._do_finish_split,
            "hand_over": self._do_hand_over,
            "request_review": self._do_request_review,
            "accept": self._do_accept,
            "land_org": self._do_land_org,
            "prune": self._do_prune,
            "close_session": self._do_close_session,
        }
