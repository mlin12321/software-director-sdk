"""Opening and closing the human's sessions, and handing a request on
to the team it is for."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from uuid import UUID, uuid4

from openhands.agent_server.director.environment import EnvironmentFailed
from openhands.agent_server.director.environment.checkouts import Checkouts, time_stamp
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.intents import Brief, CloseSession
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import (
    ChosenBy,
    OpenedSession,
    OpenSessionRequest,
    OrgNode,
    RoutedRequest,
    SessionInput,
    SessionTeam,
    SessionTeamInput,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    OrgStoreError,
    conversations_of,
    division_lead_id,
    is_division_lead,
    new_agents,
    session_team_id,
)
from openhands.agent_server.director.prompts import (
    handed_up,
    misrouted_notice,
    session_brief,
)
from openhands.agent_server.director.run_state import (
    CHECKS,
    PROGRESS,
    TASK,
    TOLD,
    Intent,
)
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.agent_start_service import (
    AgentStartService,
    agent_title,
)
from openhands.agent_server.director.service.check_count_service import (
    CheckCountService,
)
from openhands.agent_server.director.service.message_delivery_service import (
    MessageDeliveryService,
)
from openhands.agent_server.director.service.model_call_service import ModelCallService
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.request_routing_service import (
    RequestRoutingService,
)
from openhands.agent_server.director.service.resumable_work_service import (
    ResumableWorkService,
)
from openhands.agent_server.director.service.session_document_service import (
    SessionDocumentService,
)
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.team_history_service import (
    ASKED_KEPT,
    TeamHistoryService,
)
from openhands.agent_server.director.service.team_messaging_service import (
    TeamMessagingService,
)
from openhands.agent_server.director.service.team_split_service import TeamSplitService
from openhands.agent_server.director.service.team_summary_service import (
    TeamSummaryService,
)
from openhands.agent_server.director.service.team_type_service import TeamTypeService
from openhands.agent_server.director.service.work_queue_service import (
    BASE_LANE,
    WorkQueueService,
)
from openhands.agent_server.director.split import shape_of
from openhands.sdk.conversation.request import (
    ConversationWorktreeOptions,
    StartConversationRequest,
)
from openhands.sdk.errors import (
    Conflict,
    NotFound,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.spec import Tool


logger = get_logger(__name__)

# Where the last commit of a session is kept when the session is closed
# with work that was not merged.
SESSION_REFS = "refs/agents/sessions"
_TITLE_LENGTH = 60


@dataclass
class SessionLifecycleService:
    """Opens and closes the human's sessions.

    All work of the org is done in sessions. Opening one gives it a place
    at the team it is opened at and at every team below, and starts it
    with the human's request; a request that is for a team further down
    is handed on to it. Closing one is a step of resumable work
    (`do_close_session`): its agents are stopped, what it kept to itself
    goes with it, and work that was never merged is kept under a ref.
    """

    org: Callable[[], OrgStore]
    """The org the sessions work in."""
    checkouts: Checkouts
    """Where each team's work is and how it moves."""
    repo_dir: Callable[[], Path]
    """The repository the org is in."""
    org_state: OrgStateService
    org_save: OrgSaveService
    org_settings: OrgSettingsService
    team_types: TeamTypeService
    team_history: TeamHistoryService
    team_summaries: TeamSummaryService
    team_messaging: TeamMessagingService
    message_delivery: MessageDeliveryService
    agent_start: AgentStartService
    check_counts: CheckCountService
    confinement: AgentConfinementService
    session_documents: SessionDocumentService
    session_drafts: SessionDraftService
    team_splits: TeamSplitService
    model_calls: ModelCallService
    resumable_work: ResumableWorkService
    work_queue: WorkQueueService
    request_routing: Callable[[], RequestRoutingService]
    """What chooses the team a request goes straight to; it is given
    :meth:`hand_on`, so it is asked for when it is needed."""
    launcher: Callable[[], TeamLauncher]
    """What reaches the conversations; raises where there is none."""
    checkout_for_session: Callable[[str], None]
    """Gives the team a session is opened at the checkout it works in."""
    settle_on_close: Callable[[SessionTeam], None]
    """Settles what waits for a review of a session that is closed."""
    org_turn: Callable[[], None]
    """Has the next session whose work waits take its turn to land."""
    tell_turns: Callable[[], None]
    """Tells the sessions that wait where they stand in line."""

    async def open_session(
        self,
        request: OpenSessionRequest,
        conversation: StartConversationRequest | None = None,
        *,
        rename: bool = False,
    ) -> OpenedSession:
        """Open a session at a team and start it with the human's request.

        All work of the org is done in sessions. One has a place of its own
        for the team it is opened at and for every team the org has below
        that one, and a Division Lead place: the team above, or
        the top team itself for a session opened there. Only the team it is
        opened at starts; a team below starts when that IC gives it a task,
        and the reviewer when the session submits its work.

        Args:
            conversation: The conversation the human started the request
                in, which becomes the session's IC. Without it the IC is
                derived from one the org has already.

        Raises:
            NotFound: The team does not exist.
            Conflict: There is no org, it is paused, or the org has no
                conversation to derive the session's IC from.
            Unavailable: The director is not running.
        """
        launcher = self.launcher()
        store = await asyncio.to_thread(lambda: self.org())
        root = store.team(store.root_id) if store.root_id else None
        if root is None:
            raise Conflict("There is no org to open a session in")
        if self.org_state.paused is not None:
            raise Conflict("The org is paused; resume it before opening a session")
        most = self.org_settings.settings.max_open_sessions
        if most and len(store.sessions()) >= most:
            raise Conflict(
                f"The org's settings allow {most} open "
                f"{'session' if most == 1 else 'sessions'} at once. Close "
                "one, or raise the limit in the settings."
            )
        text = request.text.strip()
        chosen_by: ChosenBy = "path"
        reason = ""
        if request.node_id is not None:
            team = store.team(request.node_id)
            if team is None:
                raise NotFound(f"Unknown team: {request.node_id}")
        elif self.org_settings.settings.route_requests and store.sub_teams(root.id):
            choice = await self.request_routing().choose_team(text, root)
            chosen_by, reason = choice.by, choice.reason
            team = store.team(choice.node_id) or root
        else:
            team = root
        above = store.team(team.parent_id) if team.parent_id else team
        if above is None:
            raise NotFound(f"Unknown team: {team.parent_id}")
        if conversation is None and await self.agent_start.template() is None:
            raise Conflict(
                "No conversation of the org runs on this server to start the "
                f"session's {self.team_types.type_of(team).names.writer} from. Send "
                "the request from a new conversation."
            )
        shape = [team]
        for current in shape:
            shape.extend(store.sub_teams(current.id))
        session_id = f"s{uuid4().hex[:6]}"
        title = request.title.strip() or provisional_summary(text).title
        title = title[:_TITLE_LENGTH]
        others = sorted(
            {
                store.session_title(other)
                for other in store.places()
                if other.team_id == team.id and other.agents
            }
        )
        run = self.org_state.run_state()
        checkout = str(self.repo_dir())
        branch = await asyncio.to_thread(self.checkouts.branch, checkout)
        lead_id = division_lead_id(above.id, session_id)
        ids = {current.id: session_team_id(current.id, session_id) for current in shape}
        top = store.open_session(
            SessionInput(
                id=session_id,
                title=title,
                team_id=team.id,
                above_id=above.id,
                request=text[:ASKED_KEPT],
                opened_at=datetime.now(UTC).isoformat(),
                branch="" if branch == "HEAD" else branch,
                cost_allowance=self.org_settings.settings.session_cost_limit,
                shape=shape_of(shape),
            ),
            [
                SessionTeamInput(
                    id=lead_id,
                    session=session_id,
                    team_id=above.id,
                    division_lead=True,
                    parent_id=None,
                    title=above.title,
                    # It reads the session's work and writes nothing.
                    boundary=above.boundary.model_copy(update={"writable": []}),
                    isolation="shared",
                    agent_profile=above.agent_profile,
                    agent_type=above.agent_type,
                    worktree_path=checkout,
                    branch=branch,
                    agents=[],
                    status="idle",
                    summary=above.summary,
                    since=above.since,
                ),
                *(
                    SessionTeamInput(
                        id=ids[current.id],
                        session=session_id,
                        team_id=current.id,
                        parent_id=lead_id
                        if current.id == team.id
                        else ids[current.parent_id or ""],
                        title=current.title,
                        boundary=current.boundary,
                        # Its work is kept apart from the repository's
                        # checkout until it is reviewed.
                        isolation="worktree"
                        if current.id == team.id
                        else current.isolation,
                        agent_profile=current.agent_profile,
                        agent_type=current.agent_type,
                        worktree_path=checkout,
                        branch=branch,
                        agents=[],
                        status="idle",
                        headline=provisional_summary(text).headline
                        if current.id == team.id
                        else current.headline,
                        summary=current.summary,
                        leftover_documents=current.leftover_documents,
                        since=current.since,
                    )
                    for current in shape
                ),
            ],
        )
        for current in shape:
            run.mark(TASK, ids[current.id], self.team_messaging.purpose(current))
        await asyncio.to_thread(
            self.team_history.record_event,
            above.id,
            "session_opened",
            top,
            text[:ASKED_KEPT],
        )
        brief = session_brief(
            top,
            title,
            others,
            names=self.team_types.team(top).names,
            kept=self.team_types.kept_documents(top),
        )
        if conversation is None:
            conversation_id = (
                self.team_messaging.tell_writer(top.id, text + brief, human=True) or ""
            )
        else:
            try:
                conversation_id = await self._start_with(
                    top, conversation, text, brief, launcher, rename
                )
            except BaseException:
                # Nothing of a session that never began stays in the record.
                self.session_documents.leave(top)
                for place in store.session(session_id):
                    run.unmark(TASK, place.id)
                store.close_session(session_id)
                raise
        if not rename and not request.title.strip():
            self.work_queue.enqueue(
                lambda: self.team_summaries.name_session(top.id, text), top.id
            )
        self.org_save.save_soon()
        return OpenedSession(
            session=next(found for found in store.sessions() if found.id == session_id),
            conversation_id=conversation_id,
            chosen_by=chosen_by,
            reason=reason,
        )

    async def _start_with(
        self,
        top: SessionTeam,
        conversation: StartConversationRequest,
        text: str,
        brief: str,
        launcher: TeamLauncher,
        rename: bool = False,
    ) -> str:
        """Start a session's IC as the conversation the human began the
        request in, and brief it.

        Returns:
            Its conversation.
        """
        store = self.org()
        ic_id = conversation.conversation_id or uuid4()
        writer = self.team_types.team(top).writer
        store.update_place(top.id, agents=new_agents(ic_id, writer))
        apart = top.isolation == "worktree"
        tools = self.team_types.tools_for(top, writer)
        tool_params = await asyncio.to_thread(self.confinement.confine, top.id, writer)
        started = await launcher.start_root(
            conversation.model_copy(
                update={
                    "conversation_id": ic_id,
                    "initial_message": None,
                    # Named like every other agent of the org.
                    "autotitle": False,
                    **(
                        {
                            "worktree": True,
                            "worktree_options": ConversationWorktreeOptions(
                                base="head"
                            ),
                        }
                        if apart
                        else {}
                    ),
                }
            ),
            [Tool(name=name) for name in tools],
            title=agent_title(
                top.title,
                self.team_types.team(top).title(writer),
                store.session_title(top),
            ),
            tool_params=tool_params,
            adapt_agent=await asyncio.to_thread(
                self.confinement.adapter, str(ic_id), tools
            ),
        )
        store.set_agent(
            top.id,
            writer,
            kind=started.kind,
            acp_server=started.acp_server,
            model=started.model,
        )
        store.update_place(
            top.id,
            worktree_path=started.workspace,
            isolation=started.isolation if apart else "shared",
            branch=await asyncio.to_thread(self.checkouts.branch, started.workspace),
            base_commit=await asyncio.to_thread(
                self.checkouts.version, started.workspace
            )
            or "",
            boundary_enforcement=self.confinement.enforcement(started.kind),
        )
        self.checkout_for_session(top.id)
        self.team_messaging.progressed(top.id)
        self.message_delivery.joined(str(ic_id))
        self.work_queue.enqueue(
            lambda: self.team_summaries.summarize(
                top.id, text, first=True, rename=rename
            ),
            top.id,
        )
        self.resumable_work.begin(top.id, Brief(role=writer, task=text + brief))
        return str(ic_id)

    def hand_on(
        self, place: SessionTeam, above: SessionTeam, request: RoutedRequest, why: str
    ) -> str:
        """Give a request a team said is not its own to the IC above it.

        Nobody works above the team a session was opened at, so what that
        team hands back goes one team up the org, in a session of its own,
        and to the human only from the top."""
        names = self.team_types.team(place).names
        if not is_division_lead(above):
            self.team_messaging.tell_writer(
                above.id, misrouted_notice(place, request.text, why, names=names)
            )
            return (
                f"Handed to the {names.writer} of the team above ({above.title}). "
                "It is no longer yours; go on with your own work."
            )
        own = self.org().team(place.team_id)
        higher = self.org().team(own.parent_id or "") if own else None
        try:
            top = self._lift(place, higher) if higher else None
        except OrgStoreError:
            top = None
        if top is None:
            self._tell_human_misrouted(place, request.text, why)
            return (
                "The human, who asked, is told that it is not your team's. "
                "Wait for the answer."
            )
        # It may say the same, and the session then goes up once more.
        self.team_history.keep_routed("asked", top, request.text, "top", why)
        self.team_messaging.tell_writer(
            top.id,
            handed_up(place, request.text, why, names=names)
            + session_brief(
                top,
                self.org().session_title(top),
                [],
                names=names,
                kept=self.team_types.kept_documents(top),
            ),
        )
        self.org_save.save_soon()
        return (
            f"Handed to the {names.writer} of the team above ({top.title}), which this "
            "session is now opened at. Your team stays in it as a sub-team: "
            f"that {names.writer} gives you a task if it has one for you."
        )

    def _lift(self, node: SessionTeam, higher: OrgNode) -> SessionTeam:
        """Make the session opened at ``node`` one opened at ``higher``,
        the team above it in the org, in the checkout it has.

        Raises:
            OrgStoreError: Its work is in review already.
        """
        store = self.org()
        session = store.session_of(node)
        reviewed_from = store.place(node.parent_id or "")
        above = store.team(higher.parent_id) if higher.parent_id else higher
        if session is None or reviewed_from is None or above is None:
            raise OrgStoreError(f"{node.id} is not what a session was opened at")
        sid = session.id
        shape = [higher]
        for current in shape:
            shape.extend(store.sub_teams(current.id))
        # What it has below its own team stays as the session made it.
        has = {place.team_id for place in store.session(sid)}
        added: list[OrgNode] = []
        for current in shape[1:]:
            if current.id not in has and (
                current.parent_id == higher.id
                or any(current.parent_id == other.id for other in added)
            ):
                added.append(current)
        lead_id = division_lead_id(above.id, sid)
        top_id = session_team_id(higher.id, sid)
        top = store.lift_session(
            session.model_copy(
                update={
                    "team_id": higher.id,
                    "above_id": above.id,
                    "shape": shape_of(shape),
                }
            ),
            SessionTeamInput(
                id=top_id,
                session=sid,
                team_id=higher.id,
                parent_id=lead_id,
                title=higher.title,
                boundary=higher.boundary,
                # The session's own checkout is this team's from now on.
                isolation=node.isolation,
                agent_profile=higher.agent_profile,
                agent_type=higher.agent_type,
                worktree_path=node.worktree_path,
                branch=node.branch,
                base_commit=node.base_commit,
                agents=[],
                status="idle",
                headline=node.headline,
                summary=higher.summary,
                leftover_documents=higher.leftover_documents,
                since=higher.since,
            ),
            SessionTeamInput(
                id=lead_id,
                session=sid,
                team_id=above.id,
                division_lead=True,
                parent_id=None,
                title=above.title,
                boundary=above.boundary.model_copy(update={"writable": []}),
                isolation="shared",
                agent_profile=above.agent_profile,
                agent_type=above.agent_type,
                worktree_path=reviewed_from.worktree_path,
                branch=reviewed_from.branch,
                agents=[],
                status="idle",
                summary=above.summary,
                since=above.since,
            ),
            [
                SessionTeamInput(
                    id=session_team_id(current.id, sid),
                    session=sid,
                    team_id=current.id,
                    parent_id=session_team_id(current.parent_id or "", sid),
                    title=current.title,
                    boundary=current.boundary,
                    isolation=current.isolation,
                    agent_profile=current.agent_profile,
                    agent_type=current.agent_type,
                    worktree_path=node.worktree_path,
                    branch=node.branch,
                    agents=[],
                    status="idle",
                    headline=current.headline,
                    summary=current.summary,
                    leftover_documents=current.leftover_documents,
                    since=current.since,
                )
                for current in added
            ],
        )
        store.update_place(node.id, isolation="shared")
        run = self.org_state.run_state()
        for current in [higher, *added]:
            run.mark(
                TASK,
                session_team_id(current.id, sid),
                self.team_messaging.purpose(current),
            )
        return top

    def _tell_human_misrouted(self, place: SessionTeam, text: str, why: str) -> None:
        """Put a request no team took before the human, who asked."""
        taken = {escalation.id for escalation in place.escalations}
        self.org().raise_escalation(
            place.id,
            next(name for n in count(1) if (name := f"misrouted-{n}") not in taken),
            kind="design_decision",
            title="This request is not this team's",
            detail=f"{why}\n\nThe request:\n{text}",
        )
        self.org_save.save_soon()

    def close_session(self, session_id: str) -> None:
        """Close a session: its agents are stopped and removed with its
        teams. Work of it that was accepted stays where it was merged;
        the rest is kept under a ref and is in no checkout any more. Safe
        from any thread.

        Raises:
            NotFound: No such session is open.
        """
        teams = self.org().session(session_id)
        if not teams:
            raise NotFound(f"No session {session_id!r} is open")
        self.resumable_work.begin(teams[0].id, CloseSession())

    async def do_close_session(self, intent: Intent) -> None:
        store = self.org()
        run = self.org_state.run_state()
        place_id = intent.node_id
        launcher = self.launcher()
        if "planned" not in intent.steps:
            top = store.place(place_id)
            session = store.session_of(top) if top else None
            if top is None or session is None:
                return
            teams = store.session(session.id)
            # Fixed before anything is removed, so that closing again after
            # a restart removes the same things and records them once.
            intent = run.step_done(
                intent,
                "planned",
                session=session.id,
                title=session.title,
                team=top.title,
                parent=session.above_id,
                teams=[team.id for team in teams],
                conversations=conversations_of(teams),
                stamp=time_stamp(),
                # A session none of whose work was accepted is not on the
                # org's record, and so neither is its end. Decided now:
                # what the session kept to itself goes before its end is
                # recorded.
                known=all(
                    event.kind != "session_opened" or event.team_id != top.team_id
                    for event in self.session_drafts.events(
                        session.id, session.above_id
                    )
                ),
            )
        plan = intent.of(CloseSession)
        conversations = [UUID(agent) for agent in plan.conversations]
        if "stopped" not in intent.steps:
            self.message_delivery.wait_until_quiet(intent, plan.teams)
            for conversation_id in conversations:
                await launcher.stop(conversation_id)
            intent = run.step_done(intent, "stopped")
        if "kept" not in intent.steps:
            # Only teams that started have a checkout with work of theirs.
            started = [
                team
                for team in map(store.place, plan.teams)
                # A Division Lead place is in the human's own checkout.
                if team is not None and team.agents and not is_division_lead(team)
            ]
            top = store.place(place_id)
            # Its Division Lead place is in the repository's checkout.
            above = store.place(top.parent_id or "") if top else None
            kept = None
            # A session whose IC never started has no checkout of its own:
            # what is in the checkout above is not its work.
            if started and top and above and self.checkouts.is_apart(top, above):
                kept = await asyncio.to_thread(
                    self._keep_unmerged, top, above, plan.stamp
                )
                await asyncio.to_thread(
                    self.team_splits.discard_work, started, plan.stamp
                )
            intent = run.step_done(intent, "kept", kept=kept)
        if "deleted" not in intent.steps:
            for conversation_id in conversations:
                await launcher.delete(conversation_id)
                run.unmark(TOLD, str(conversation_id))
                run.drop_held(str(conversation_id))
            intent = run.step_done(intent, "deleted")
        if "removed" not in intent.steps:
            top = store.place(place_id)
            for team in map(store.place, plan.teams):
                if team is not None:
                    for kind in (TASK, PROGRESS, CHECKS):
                        run.unmark(kind, team.id)
                    self.check_counts.work_accepted(team)
            if top is not None:
                await self.work_queue.run_in(
                    BASE_LANE, lambda: asyncio.to_thread(self.settle_on_close, top)
                )
                gone = await asyncio.to_thread(self.session_documents.leave, top)
                # Once: a step cut off after it would record it again.
                if plan.known and "told" not in intent.steps:
                    await asyncio.to_thread(
                        self.team_history.land_event,
                        plan.parent,
                        self.team_history.event(
                            "session_closed",
                            top,
                            "; ".join(
                                said
                                for said in (
                                    "with work that was not merged"
                                    if intent.of(CloseSession).kept
                                    else "",
                                    "what it wrote and never had accepted went "
                                    f"with it ({gone})"
                                    if gone
                                    else "",
                                )
                                if said
                            ),
                        ),
                    )
                    intent = run.step_done(intent, "told")
            store.close_session(plan.session)
            await asyncio.to_thread(self.org_turn)
            self.tell_turns()
            await self.org_save.save()
            intent = run.step_done(intent, "removed")
        self.message_delivery.stop_waiting(intent.id)

    def _keep_unmerged(
        self, top: SessionTeam, above: SessionTeam, stamp: str
    ) -> str | None:
        """Keep the last commit of a session whose commits are not all in
        the checkout above.

        Returns:
            The ref it is kept under; None when everything was merged.
        """
        ref = f"{SESSION_REFS}/{top.session}/{stamp}"
        try:
            if not self.checkouts.keep_unmerged(
                top.worktree_path, above.worktree_path, ref
            ):
                return None
        except (EnvironmentFailed, OSError):
            logger.exception("Could not keep the work of session %s", top.session)
            return None
        return ref
