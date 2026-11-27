"""The director's state for one server: an org tree and its team_memory."""

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import UUID

from openhands.agent_server.director.boundary import nearest_team, worded
from openhands.agent_server.director.git import is_git_repository
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.intents import LandOrg
from openhands.agent_server.director.launch import TeamLauncher
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
from openhands.agent_server.director.run_state import CLEARING
from openhands.agent_server.director.runtime import bind_director
from openhands.agent_server.director.scope import Scope, named_paths, searched_folder
from openhands.agent_server.director.service.stall_watch_service import (
    STALL_WAITS,
)
from openhands.agent_server.director.service.state import (
    AGENT_PAUSED,
    BASE_LANE,
    LIMIT_WAITS,
    MERGE_WAITS,
    ORG_WAITS,
    RESTART_NOTE,
    ROOT_ID,
    _instant,
    _summary_of,
    checkout_root,
    failure_waits,
    hand_over_of,
)
from openhands.agent_server.director.service.sync import Sync
from openhands.agent_server.director.service.team_type_service import (
    DIRECTOR_TOOL_NAMES,
)
from openhands.sdk.errors import (
    Conflict,
    NotFound,
    Unavailable,
    Unprocessable,
)
from openhands.sdk.event import ActionEvent
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


__all__ = [
    "AGENT_PAUSED",
    "DIRECTOR_TOOL_NAMES",
    "RESTART_NOTE",
    "DirectorService",
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Takes what :class:`ServiceState` takes."""
        super().__init__(*args, **kwargs)
        for kind, step in {
            "brief": self._do_brief,
            "launch_member": self._do_launch_member,
            "split": self._do_split,
            "start_team": self._do_start_team,
            "finish_split": self._do_finish_split,
            "hand_over": self._do_hand_over,
            "request_review": self._do_request_review,
            "accept": self._do_accept,
            "land_org": self._do_land_org,
            "prune": self._do_prune,
            "close_session": self._do_close_session,
        }.items():
            self.resumable_work.register(kind, step)

    async def start(self, launcher: TeamLauncher) -> None:
        """Begin running agents through ``launcher``, on the running loop."""
        self._loop = asyncio.get_running_loop()
        self._launcher = launcher
        await self.work_queue.__aenter__()
        await self.org_save.__aenter__()
        self._stop_status = launcher.on_status(self._on_status)
        self._stop_guard = launcher.on_action(self.guard_action)
        self._stop_resume = launcher.on_resume(self.confinement.with_tools_again)
        await asyncio.to_thread(self.confinement.look_for_sandbox)
        bind_director(self)
        # What reviewers check work by is fetched now, off the loop, so
        # that no review request is the first to wait for it.
        self._review_instructions_ready = asyncio.create_task(
            asyncio.to_thread(self._load_review_instructions),
            name="director-review-instructions",
        )
        self._stall_timer = asyncio.create_task(
            self._watch_for_stalls(), name="director-stall-watch"
        )
        if self._store is not None:
            self.work_queue.enqueue(self._came_back)
        elif await asyncio.to_thread(self._remembered_repo) is not None:
            # An org that was left in the middle of its work is looked at
            # now, not when somebody first asks for it.
            await asyncio.to_thread(lambda: self.org)

    async def aclose(self) -> None:
        """Stop running agents and save the tree if a save was pending.

        An org that is in the middle of its work is left paused, so that the
        next start finds what to take up again.
        """
        bind_director(None)
        store = self._store
        if store is not None and store.root_id is not None:
            run = self.org_state.run_state()
            if self.org_state.paused is None and has_work(run):
                self.org_state.hold("shutdown")
                self.start_line.org_held()
            run.put("clean", "1")
        if self._stop_status is not None:
            self._stop_status()
            self._stop_status = None
        if self._stop_guard is not None:
            self._stop_guard()
            self._stop_guard = None
        if self._stop_resume is not None:
            self._stop_resume()
            self._stop_resume = None
        await self.work_queue.__aexit__(None, None, None)
        timers = [
            task
            for task in (self._stall_timer, self._review_instructions_ready)
            if task is not None
        ]
        self._stall_timer = self._review_instructions_ready = None
        for task in timers:
            task.cancel()
        await asyncio.gather(*timers, return_exceptions=True)
        await self.org_save.__aexit__(None, None, None)
        self._launcher = None
        self._loop = None
        self.org_state.close()

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
        store = await asyncio.to_thread(lambda: self.org)
        conversation = request.conversation
        workspace = await asyncio.to_thread(
            Path(conversation.workspace.working_dir).resolve
        )
        owns_git = await asyncio.to_thread(lambda: (workspace / ".git").exists())
        if request.at_workspace:
            if not owns_git:
                raise Unavailable(
                    f"An org needs a git repository; {workspace} is not one"
                )
        elif not await asyncio.to_thread(is_git_repository, workspace):
            raise Unavailable(f"An org needs a git repository; {workspace} is not one")
        repository = await asyncio.to_thread(checkout_root, str(workspace))
        if conversation.conversation_runtime == "docker":
            raise Unprocessable(
                "An org's agents run in the server's own process, not in Docker"
            )
        root_type = request.agent_type or self.org_settings.settings.default_agent_type
        allowed = self.org_settings.settings.allowed_agent_types
        if root_type not in await asyncio.to_thread(
            lambda: self.team_types.agent_types
        ):
            raise Unprocessable(
                f"The director finds no agent type named {root_type!r} for the "
                "org's top team"
            )
        if allowed and root_type not in allowed:
            raise Unprocessable(
                f"The org's top team cannot be of the type {root_type!r}. "
                f"The settings allow: {', '.join(allowed)}."
            )
        named = provisional_summary(request.task)
        title = request.title or named.title
        async with self._create_lock:
            if store.root_id is not None:
                raise Conflict("The director already has an org")
            if await asyncio.to_thread(self._join, workspace, not request.at_workspace):
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
                    agent_type=root_type,
                    headline=named.headline,
                    summary=_summary_of(request.task),
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
            await asyncio.to_thread(self.remembered.keep, repo_path=str(self._repo_dir))
            await asyncio.to_thread(self._working)
        if request.title is None:
            # The org is named after what its first session is for.
            top = opened.session.teams[0].id
            # In the lane of that team, after the model that names it.
            self.work_queue.enqueue(
                lambda: self.work_queue.run_in(BASE_LANE, lambda: self._name_org(top)),
                top,
            )
        return opened

    async def _say_in_session(self, session_id: str, text: str) -> RoutedMessage:
        """Give what the human says in a session's conversation to the IC
        of one of the session's teams: the one it was opened at, unless the
        director is sure of one below.

        Raises:
            NotFound: The session is not open.
        """
        store = self.org
        places = [
            place for place in store.session(session_id) if not is_division_lead(place)
        ]
        if not places:
            raise NotFound(f"Unknown session: {session_id}")
        top = place = places[0]
        chosen_by: ChosenBy = "path"
        reason = ""
        if self.org_settings.settings.route_requests and len(places) > 1:
            choice = await self.request_routing.choose_team(
                text, top, within=session_id
            )
            chosen_by, reason = choice.by, choice.reason
            place = store.instance(choice.node_id, session_id) or top
        conversation_id = await asyncio.to_thread(
            lambda: self.tell_writer(place.id, text, human=True)
        )
        if place.id != top.id:
            # Its IC may say it is not the team's own: the IC above has it then.
            await asyncio.to_thread(
                self.team_history.keep_routed, "asked", place, text, chosen_by, reason
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
        the conversation it comes with (or above it, unless ``at_workspace``),
        and failing that starts one there and becomes its task.

        Raises:
            Unprocessable: The path is outside the org's repository, or there
                is no org and the request does not say how to start one.
            Conflict: No session can be opened for the message.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = await asyncio.to_thread(lambda: self.org)
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
                    self._join,
                    Path(conversation.workspace.working_dir),
                    not request.at_workspace,
                )
                if not joined:
                    opened = await self.create_org(
                        CreateOrgRequest(
                            task=request.text,
                            agent_type=request.agent_type,
                            conversation=conversation,
                            at_workspace=request.at_workspace,
                        )
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
        if request.auto and self.org_settings.settings.route_requests:
            choice = await self.request_routing.choose_team(text, node)
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
                self.team_history.keep_routed,
                "asked",
                top,
                request.text,
                chosen_by,
                reason,
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
        action = event.action.model_dump()
        if (
            not named_paths(event.tool_name, action)
            and searched_folder(event.tool_name, action) is None
        ):
            # Nothing a scope is asked about: most actions.
            return None
        found = store.find_by_conversation(str(conversation_id))
        if found is not None:
            scope = self.confinement.scope_of(*found)
        else:
            aside = store.find_hand_over(str(conversation_id))
            scope = (
                self.confinement.scope_of(
                    aside, self.team_types.team(aside).writer, str(conversation_id)
                )
                if aside
                else None
            )
        denied = self.confinement.check(scope, event.tool_name, action)
        if denied is None:
            return None
        works = found is not None and found[1] == self.team_types.team(found[0]).writer
        if found is not None and works and denied.relative is not None:
            # The refusal stands whether or not it could be kept.
            with suppress(Exception):
                self.team_history.record_need(
                    found[0], denied.relative or ".", NEED_WRITE
                )
        return worded(
            denied,
            scope or Scope(checkout="", directory=""),
            names=self.team_types.names_at(found[0].id if found else ""),
        )

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
        names = self.team_types.names_at(place_id)
        self._require_launcher()
        store = self.org
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
        if lost and role == self.team_types.team(place).writer:
            # It has no conversation to answer in: the team starts again
            # with the answer, and is asked of no longer once it has.
            async def again() -> None:
                await self.start_again(place_id, StartAgainRequest(), said=answer)
                with suppress(NotFound):
                    store.settle_escalation(place_id, escalation_id)

            self.work_queue.enqueue(again, place_id)
            return place
        store.settle_escalation(place_id, escalation_id)
        if escalation_id == MERGE_WAITS:
            # The way into the checkout was cleared: the accepted work is
            # merged now as its reviewer decided, with nobody asked again.
            decided = json.loads(
                self.org_state.run_state().marked(CLEARING).get(place_id) or "{}"
            )
            self.accept_review(place_id, "", **decided)
            return store.place(place_id) or place
        if escalation_id == ORG_WAITS:
            # The human goes on without the Manager.
            self.resumable_work.begin(place_id, LandOrg(org_holds=True))
            return store.place(place_id) or place
        if escalation_id == LIMIT_WAITS:
            self.go_on_past_limit(place_id)
        if lost:
            # A new Manager starts with the answer; the IC goes on as it was.
            self.tell_manager(
                place_id,
                f"The {names.manager} before you could not be started. The human was "
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
        self.message_delivery.notify(
            self.team_types.team(place).conversation(role),
            f"{told}:\n{answer}",
            at_once=True,
        )
        return store.place(place_id) or place
