"""Running the org: pausing and resuming it, taking up what it had begun
before the server stopped, and following what its agents do."""

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

from openhands.agent_server.director.environment.checkouts import Checkouts
from openhands.agent_server.director.intents import Brief, LandOrg, StartTeam
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import (
    FileEdit,
    Recovery,
    SessionTeam,
    StartAgainRequest,
)
from openhands.agent_server.director.org_store import (
    AGENT_BLOCKED,
    AGENT_LOST,
    OrgStore,
    hand_over_of,
    is_division_lead,
    new_agents,
)
from openhands.agent_server.director.prompts import (
    changed_since_accept,
)
from openhands.agent_server.director.recovery import (
    STARTING,
    describe,
    has_work,
    put_aside_stale,
)
from openhands.agent_server.director.run_state import (
    CLEARING,
    HELD,
    RUNNING,
    WAITING,
    Intent,
)
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.message_delivery_service import (
    MessageDeliveryService,
)
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.resumable_work_service import (
    ResumableWorkService,
)
from openhands.agent_server.director.service.review_service import (
    MERGE_WAITS,
    ReviewService,
)
from openhands.agent_server.director.service.session_limit_service import (
    LIMIT_WAITS,
    SessionLimitService,
)
from openhands.agent_server.director.service.session_merge_service import (
    ORG_WAITS,
    SessionMergeService,
)
from openhands.agent_server.director.service.stall_watch_service import STALL_WAITS
from openhands.agent_server.director.service.start_line_service import StartLineService
from openhands.agent_server.director.service.team_member_start_service import (
    TeamMemberStartService,
)
from openhands.agent_server.director.service.team_messaging_service import (
    STARTED_WITH,
    TeamMessagingService,
    failure_waits,
)
from openhands.agent_server.director.service.team_split_service import TeamSplitService
from openhands.agent_server.director.service.team_summary_service import agent_title
from openhands.agent_server.director.service.team_type_service import TeamTypeService
from openhands.agent_server.director.service.work_queue_service import (
    ORG_LANE,
    WorkQueueService,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import (
    Conflict,
    NotFound,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.spec import Tool


logger = get_logger(__name__)

# An agent whose conversation is gone, or could not be started.
# An agent that was running when its org was paused, and runs again when it
# is resumed.
AGENT_PAUSED = "paused"
# What an agent is told when the server ended in the middle of its step.
RESTART_NOTE = (
    "The server was restarted while you were working. The command or edit "
    "you were in the middle of may not have finished: check what is there "
    "before you carry on with your task."
)
# The lane agents are paused in, so that nothing waits for their last step.
PAUSE_LANE = "pause"
_AGENT_STATUS: dict[ConversationExecutionStatus, str] = {
    ConversationExecutionStatus.RUNNING: "working",
    ConversationExecutionStatus.ERROR: "blocked",
    ConversationExecutionStatus.STUCK: "blocked",
}


@dataclass
class OrgRunService:
    """Pauses and resumes the org and follows the status of its agents.

    A pause stops every agent that runs and holds what the director was
    in the middle of; a resume runs those agents again and takes the held
    work up. After a restart the org is resumed where that is safe and
    otherwise shown to the human as what was left (`recovery`). The
    launcher reports each change of an agent's status to `on_status`.
    """

    org: Callable[[], OrgStore]
    """The org that is run."""
    checkouts: Checkouts
    """Where each team's work is and how it moves."""
    find_org: Callable[[], OrgStore | None]
    """The org if it was loaded already, for a status that arrives early."""
    repo_dir: Callable[[], Path]
    """The repository the org is in."""
    org_state: OrgStateService
    org_save: OrgSaveService
    org_settings: OrgSettingsService
    start_line: StartLineService
    message_delivery: MessageDeliveryService
    team_messaging: TeamMessagingService
    team_types: TeamTypeService
    member_start: TeamMemberStartService
    confinement: AgentConfinementService
    resumable_work: ResumableWorkService
    work_queue: WorkQueueService
    team_splits: TeamSplitService
    launcher: Callable[[], TeamLauncher]
    """What reaches the conversations; raises where there is none."""
    find_launcher: Callable[[], TeamLauncher | None]
    """The same, or None where the director runs no agents."""
    loop: Callable[[], asyncio.AbstractEventLoop | None]
    """The loop the director works on while it is started."""
    home: Callable[[], str]
    """The directory the server was started for, in the repository."""
    instance: Callable[[], str]
    """Names this server, to tell its run state from another's."""
    working: Callable[[], None]
    """Records that the org is run by this server."""
    lane_of: Callable[[Intent], str]
    """The lane a step is carried out in."""
    reviews: ReviewService
    session_merges: SessionMergeService
    session_limits: SessionLimitService
    """Lets a session that rests at its limit go on."""
    _resume_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def shown(self, conversation_id: str, status: ConversationExecutionStatus) -> str:
        """An agent's status as the org shows it. One that was running when
        the org was paused is paused, whatever stopping did to it."""
        held = (
            self.org_state.paused is not None
            and conversation_id in self.org_state.running
        )
        if held or conversation_id in self.org_state.resuming:
            if status != ConversationExecutionStatus.RUNNING:
                return AGENT_PAUSED
        return _AGENT_STATUS.get(status, "idle")

    async def pause(self) -> Recovery:
        """Hold the org: its agents stop after the step they are in, and
        the director's own steps and messages wait.

        What agents still report is recorded and acted on after
        :meth:`resume`.

        Raises:
            Conflict: There is no org.
            Unavailable: The director is not running.
        """
        self.launcher()
        store = await asyncio.to_thread(lambda: self.org())
        if store.root_id is None:
            raise Conflict("There is no org to pause")
        if self.org_state.paused is None:
            self.org_state.hold("human")
            self.start_line.org_held()
            self.work_queue.enqueue(self._pause_agents, PAUSE_LANE)
        return await self.recovery()

    async def resume(self) -> Recovery:
        """Take a paused org up again.

        Run state that no longer fits the org is put aside first. Then the
        teams that waited to start are started in the order they waited,
        the steps the director had begun are finished, held messages are
        delivered and the agents that were running run again.

        Returns:
            What was put aside, and nothing else.

        Raises:
            Unavailable: The director is not running.
        """
        launcher = self.launcher()
        store = await asyncio.to_thread(lambda: self.org())
        # One resume at a time: a second would run the same agents again.
        async with self._resume_lock:
            if self.org_state.paused is None:
                return await self.recovery()
            # A pause that is still stopping agents would stop them after this
            # has set them going again.
            await self.work_queue.after(PAUSE_LANE)
            run = self.org_state.run_state()
            await asyncio.to_thread(self._put_aside_stale)
            await asyncio.to_thread(
                self.checkouts.restore, self.repo_dir(), store.places(), self.home()
            )
            dropped = self.org_state.dropped()
            was_running = run.marked(RUNNING)
            waited = list(run.marked(WAITING))
            statuses = await launcher.statuses([UUID(agent) for agent in was_running])
            again: dict[str, str | None] = {}
            for conversation_id in was_running:
                status = statuses.get(UUID(conversation_id))
                if status == ConversationExecutionStatus.ERROR:
                    again[conversation_id] = RESTART_NOTE
                elif status == ConversationExecutionStatus.PAUSED:
                    again[conversation_id] = None
            self.org_state.resuming = set(again)
            self.org_state.release()
        await self.resync()
        for conversation_id in again:
            # Seen stopped, they were just taken for agents that do not run:
            # a server that ends before they are run again takes them up.
            run.mark(RUNNING, conversation_id, was_running[conversation_id])
        for place_id in waited:
            run.unmark(WAITING, place_id)
        intents = run.open_intents()
        starting = {intent.node_id for intent in intents}
        self.start_line.line_up(place_id for place_id in waited if place_id in starting)
        for intent in intents:
            self.resumable_work.take_up(intent.id, self.lane_of(intent))
        for held in run.held():
            self.work_queue.enqueue(
                lambda held=held: self.message_delivery.deliver(
                    held.seq, held.conversation_id, held.text
                ),
                held.node_id or ORG_LANE,
            )
        for place_id in run.marked(HELD):
            submitted = store.place(place_id)
            self.reviews.release_reviews(submitted.parent_id if submitted else None)
        for conversation_id, note in again.items():
            self.work_queue.enqueue(
                lambda conversation_id=conversation_id, note=note: (
                    self.team_messaging.run_again(conversation_id, note)
                ),
                was_running[conversation_id],
            )
        return Recovery(paused=None, items=dropped, messages_held=0)

    async def recovery(self) -> Recovery:
        """What resuming the org does, and what was put aside as stale."""
        store = await asyncio.to_thread(lambda: self.org())
        run = self.org_state.run_state()
        if self.org_state.paused is None:
            return Recovery(paused=None, items=[], messages_held=0)
        launcher = self.find_launcher()
        statuses = (
            await launcher.statuses(
                [
                    UUID(conversation_id)
                    for node in store.places()
                    for agent in node.agents
                    for conversation_id in (
                        agent.conversation_id,
                        # A hand-over is paused and run again as an agent is.
                        agent.hand_over.conversation_id if agent.hand_over else None,
                    )
                    if conversation_id is not None
                ]
            )
            if launcher is not None
            else {}
        )
        items = await asyncio.to_thread(
            describe, run, store, statuses, self.repo_dir(), self.checkouts
        )
        return Recovery(
            paused=self.org_state.paused,
            items=[*items, *self.org_state.dropped()],
            messages_held=len(run.held()),
        )

    async def edit_file(self, place_id: str, path: str, content: str) -> FileEdit:
        """Write one file of a team by hand, commit it and tell its IC.

        Raises:
            NotFound: There is no such team, or its checkout is not here.
            Conflict: An agent of the team is in the middle of a turn.
            Forbidden: The team's agents may not write the file either.
            Invalid: The path does not name a file that can be committed.
        """
        store = await asyncio.to_thread(lambda: self.org())
        place = store.place(place_id)
        if place is None:
            raise NotFound(f"Unknown team: {place_id}")
        sub_teams = store.below(place_id)
        self.checkouts.require_editable(place, sub_teams, path)
        agents = [
            UUID(agent.conversation_id)
            for agent in place.agents
            if agent.conversation_id is not None
        ]
        launcher = self.find_launcher()
        if agents and launcher is not None:
            statuses = await launcher.statuses(agents)
            if ConversationExecutionStatus.RUNNING in statuses.values():
                raise Conflict(
                    f"An agent of team {place.title!r} is working. Pause the "
                    "org or wait for it to finish, then save again."
                )
        edit = await asyncio.to_thread(
            self.checkouts.write_by_hand, place, sub_teams, path, content
        )
        ic = self.team_types.writer_conversation(place)
        if edit.commit is not None and ic is not None and self.loop() is not None:
            self.message_delivery.notify(
                ic,
                f"The human changed {edit.path} in your team's checkout by hand "
                f"and committed it as {edit.commit[:12]}. Read the file again "
                "before you change it, and keep what they wrote unless they "
                "tell you otherwise.",
            )
        return edit

    async def start_again(
        self, node_id: str, request: StartAgainRequest, said: str | None = None
    ) -> SessionTeam:
        """Start afresh a team whose agents have no conversation on this
        server, on the work its branch has.

        The team's IC starts in the team's own checkout and is briefed
        from what the org kept of the team and its last headline; its
        Manager is derived from it.
        The IC starts as the given conversation, or without one like the IC
        of the team above (from the team's own agent profile if it has one).
        What the human ``said`` when asked about the lost IC ends its briefing.

        Raises:
            NotFound: There is no such team.
            Conflict: The team is done, its IC has a conversation here, or
                no conversation was given and the team above has no IC here
                to start it like.
            Unavailable: The director is not running.
        """
        names = self.team_types.names_at(node_id)
        launcher = self.launcher()
        store = await asyncio.to_thread(lambda: self.org())
        place = store.place(node_id)
        if place is None:
            team = store.team(node_id)
            if team is None:
                raise NotFound(f"There is no team {node_id}")
            raise Conflict(
                f"Nobody works at {team.title!r} itself: open a session there."
            )
        # A session's reviewer starts like the first time: from the IC the
        # session works with, in the checkout it reviews in.
        leads = is_division_lead(place)
        if place.status == "done":
            raise Conflict(f"Team {place.title} is done")
        ic = self.team_types.writer_conversation(place)
        if ic is not None and await launcher.started(UUID(ic)) is not None:
            raise Conflict(
                f"The {names.writer} of team {place.title} has a conversation here"
            )
        conversation = request.conversation
        like = self.member_start.derived_from(place) if leads else place.parent_id
        above = store.place(like) if like else None
        # A session's IC that never started has no checkout yet: it gets
        # one of its own, never the one of the team it works beside.
        own_checkout = not leads and not (
            above is not None
            and store.opens_session(place)
            and not self.checkouts.is_apart(place, above)
        )
        if conversation is not None and not own_checkout:
            raise Conflict(
                f"The session {store.session_title(place)!r} has no checkout yet. "
                "Start it again without a conversation."
            )
        if (
            conversation is None
            and await self.member_start.started_ic(above) is None
            and await self.member_start.template() is None
        ):
            raise Conflict(
                "No conversation of the org runs on this server to start "
                f"the {names.writer} of {place.title!r} from. Start it again with a "
                "conversation."
            )
        await asyncio.to_thread(
            self.checkouts.restore, self.repo_dir(), [place], self.home()
        )
        ic_id = uuid4()
        task = await asyncio.to_thread(
            lambda: (
                self.team_messaging.purpose(place)
                if leads
                else self.member_start.task_of(place)
            )
        )
        if said:
            task = (
                f"{task}{STARTED_WITH}The human was told that the {names.writer} "
                "before "
                f"you was lost, and says:\n{said}"
            )
        # The team's other agents stay its own: only the IC is new.
        writer = self.team_types.team(place).writer
        tools = self.team_types.tools_for(place, writer)
        agents = [
            *new_agents(ic_id, writer),
            *(a for a in place.agents if a.role != writer),
        ]
        if conversation is None:
            store.update_place(node_id, agents=agents, status="idle")
            self.resumable_work.begin(
                node_id,
                StartTeam(
                    task=task,
                    parent=like,
                    checkout=place.worktree_path
                    if own_checkout or self.member_start.in_place(place)
                    else None,
                ),
            )
            return store.place(node_id) or place
        store.update_place(node_id, agents=agents, status="idle")
        try:
            started = await launcher.start_root(
                conversation.model_copy(
                    update={
                        "conversation_id": ic_id,
                        "initial_message": None,
                        "autotitle": False,
                        "worktree": False,
                        "workspace": conversation.workspace.model_copy(
                            update={"working_dir": place.worktree_path}
                        ),
                    }
                ),
                [Tool(name=name) for name in tools],
                title=agent_title(
                    place.title,
                    self.team_types.team(place).title(writer),
                    store.session_title(place),
                ),
                tool_params=await asyncio.to_thread(
                    self.confinement.confine, node_id, writer
                ),
                adapt_agent=await asyncio.to_thread(
                    self.confinement.adapter, str(ic_id), tools
                ),
            )
        except BaseException:
            store.update_place(node_id, agents=place.agents, status=place.status)
            raise
        store.set_agent(
            node_id,
            writer,
            kind=started.kind,
            acp_server=started.acp_server,
            model=started.model,
        )
        self.resumable_work.begin(node_id, Brief(role=writer, task=task))
        return store.place(node_id) or place

    def _put_aside_stale(self) -> None:
        run = self.org_state.run_state()
        dropped = put_aside_stale(run, self.org(), instance=self.instance())
        if run.get("paused") is None:
            # Nothing that was left is this server's to take up.
            self.org_state.refresh()
        if dropped:
            run.put(
                "dropped",
                json.dumps(
                    [
                        item.model_dump()
                        for item in [*self.org_state.dropped(), *dropped]
                    ]
                ),
            )

    async def came_back(self) -> None:
        """Look at what an org that was just loaded was left in the middle
        of. A server that ended without stopping its org leaves it paused."""
        await asyncio.to_thread(self.look_at_what_was_left)
        await self.resync()
        if (
            self.org_state.paused in ("shutdown", "crash")
            and self.org_settings.settings.resume_on_start
        ):
            await self.resume()

    def look_at_what_was_left(self) -> None:
        if self.org().root_id is None:
            return
        run = self.org_state.run_state()
        self._put_aside_stale()
        if self.org_state.paused is None and run.get("clean") == "0" and has_work(run):
            self.org_state.hold("crash")
            self.start_line.org_held()
        if self.org_state.paused is None:
            for item in self.org_state.dropped():
                logger.warning("Run state put aside: %s", item.detail)
            run.put("dropped", None)
        self.working()

    async def _pause_agents(self) -> None:
        store = self.org()
        run = self.org_state.run_state()
        launcher = self.launcher()
        agents: list[tuple[str, str, UUID]] = [
            (node.id, agent.role, UUID(agent.conversation_id))
            for node in store.places()
            for agent in node.agents
            if agent.conversation_id is not None
        ]
        statuses = await launcher.statuses([agent for _, _, agent in agents])
        stopping: list[UUID] = []
        for place_id, role, conversation_id in agents:
            if statuses.get(conversation_id) != ConversationExecutionStatus.RUNNING:
                continue
            run.mark(RUNNING, str(conversation_id), place_id)
            self.org_state.running.add(str(conversation_id))
            with suppress(NotFound):
                store.set_agent(place_id, role, status=AGENT_PAUSED)
            stopping.append(conversation_id)
        # An IC in hand-over mode is paused in that conversation too.
        modes = [
            (node.id, mode)
            for node in store.places()
            if (mode := hand_over_of(node)) is not None
        ]
        running = await launcher.statuses(
            [UUID(mode.conversation_id) for _, mode in modes]
        )
        for place_id, mode in modes:
            aside = UUID(mode.conversation_id)
            if running.get(aside) != ConversationExecutionStatus.RUNNING:
                continue
            run.mark(RUNNING, mode.conversation_id, place_id)
            self.org_state.running.add(mode.conversation_id)
            with suppress(NotFound):
                store.set_hand_over(
                    place_id, mode.model_copy(update={"status": AGENT_PAUSED})
                )
            stopping.append(aside)
        await asyncio.gather(*(launcher.pause(agent) for agent in stopping))

    def on_status(
        self, conversation_id: UUID, status: ConversationExecutionStatus
    ) -> None:
        loop = self.loop()
        if status != ConversationExecutionStatus.RUNNING and loop is not None:
            loop.call_soon_threadsafe(self.start_line.room_made)
        store = self.find_org()
        if store is None:
            return
        agent = str(conversation_id)
        found = store.find_by_conversation(agent)
        if found is None:
            self._on_hand_over_status(store, agent, status)
            return
        node, role = found
        if status == ConversationExecutionStatus.RUNNING:
            self.org_state.resuming.discard(agent)
        if self.org_state.paused is None:
            self.message_delivery.note_running(
                agent, node.id, status == ConversationExecutionStatus.RUNNING
            )
        was = node.status
        shown = self.shown(agent, status)
        failed = shown == AGENT_BLOCKED and all(
            a.status != AGENT_BLOCKED for a in node.agents if a.role == role
        )
        now = store.set_agent(node.id, role, status=shown)
        if self.org_state.paused is None and self.loop() is not None:
            if failed:
                self.work_queue.enqueue(
                    lambda: self._failed(node.id, role, conversation_id, status),
                    node.id,
                )
            if status == ConversationExecutionStatus.RUNNING:
                self._runs_again(now, role)
            if now.status == "blocked" and was != "blocked":
                # A blocked team is no longer waited for.
                self.reviews.release_reviews(node.parent_id)
            if (
                role == self.team_types.team(node).writer
                and status == ConversationExecutionStatus.RUNNING
            ):
                self.reviews.release_reviews(node.id)
            if status != ConversationExecutionStatus.RUNNING:
                self._remind_to_submit(node.session)

    async def _failed(
        self,
        place_id: str,
        role: str,
        conversation_id: UUID,
        status: ConversationExecutionStatus,
    ) -> None:
        """Find why an agent's conversation failed, keep it and tell of it."""
        said = await self.launcher().failure(conversation_id)
        if said is None:
            said = (
                "It repeated the same step without getting anywhere."
                if status == ConversationExecutionStatus.STUCK
                else "Its conversation ended with an error it gave no account of."
            )
        team = self.org().place(place_id)
        if team is None or all(
            a.status != AGENT_BLOCKED for a in team.agents if a.role == role
        ):
            # It runs again already.
            return
        self.team_messaging.tell_of_failure(place_id, role, said)

    def acts(self, place_id: str, role: str) -> None:
        """An agent that uses one of the director's tools is at work,
        whatever its conversation last said of itself. Safe from any thread.

        A conversation that fails and is started again by what runs it may
        never say that it runs: the failure the human was told of would
        then stay with them after the agent has gone on.
        """
        team = self.org().place(place_id)
        if team is None or all(
            a.status != AGENT_BLOCKED for a in team.agents if a.role == role
        ):
            return
        with suppress(NotFound):
            self._runs_again(
                self.org().set_agent(place_id, role, status="working"), role
            )

    def _runs_again(self, team: SessionTeam, role: str) -> None:
        """An agent that runs is no longer something the human is asked of."""
        waits = {failure_waits(role)} | (
            {STALL_WAITS} if role == self.team_types.team(team).writer else set()
        )
        settled = [e.id for e in team.escalations if e.id in waits]
        for escalation_id in settled:
            with suppress(NotFound):
                self.org().settle_escalation(team.id, escalation_id)
        if settled:
            self.org_save.save_soon()

    def _on_hand_over_status(
        self, store: OrgStore, conversation_id: str, status: ConversationExecutionStatus
    ) -> None:
        """Follow the hand-over conversation of an IC. When it stops
        working, the IC is told in its main conversation what it did."""
        node = store.find_hand_over(conversation_id)
        mode = hand_over_of(node) if node else None
        if node is None or mode is None:
            return
        if status == ConversationExecutionStatus.RUNNING:
            self.org_state.resuming.discard(conversation_id)
        if self.org_state.paused is None:
            self.message_delivery.note_running(
                conversation_id, node.id, status == ConversationExecutionStatus.RUNNING
            )
        shown = self.shown(conversation_id, status)
        store.set_hand_over(node.id, mode.model_copy(update={"status": shown}))
        # One that was held while it worked may have ended while it was held.
        at_work = ("working", AGENT_PAUSED)
        if mode.status in at_work and shown not in at_work:
            self.team_splits.hand_over_ended(node.id)
            self._remind_to_submit(node.session)

    def profile_exists(self, name: str) -> bool:
        """Whether the server has an agent profile of that name. Called
        from an agent's tool, off the loop; where it cannot be asked, the
        answer is yes and the team's start says otherwise."""
        launcher, loop = self.find_launcher(), self.loop()
        if launcher is None or loop is None:
            return True
        try:
            if asyncio.get_running_loop() is loop:
                return True
        except RuntimeError:
            pass
        asked = asyncio.run_coroutine_threadsafe(launcher.profile_kind(name), loop)
        return asked.result(timeout=30) is not None

    def _remind_to_submit(self, session: str) -> None:
        """Tell the IC of a session at rest that what it changed since its
        work was accepted is not the org's until it submits again. Once for
        each accept. Safe from any thread."""
        if (
            self.session_merges.changed_since_accept.get(session, True)
            or self.org_state.paused is not None
        ):
            return
        places = self.org().session(session)
        at_work = ("working", AGENT_PAUSED)
        if any(
            place.status == "working"
            or ((mode := hand_over_of(place)) is not None and mode.status in at_work)
            for place in places
        ):
            return
        self.session_merges.changed_since_accept[session] = True
        top = places[0] if places else None
        if top is None or top.review is None or top.review.state != "accepted":
            return
        self.message_delivery.notify(
            self.team_types.writer_conversation(top),
            changed_since_accept(self.team_types.kept_documents(top)),
        )

    async def resync(self) -> None:
        """Read every agent's status back from its conversation."""
        store = self.org()
        agents: list[tuple[str, str, UUID]] = [
            (node.id, agent.role, UUID(agent.conversation_id))
            for node in store.places()
            for agent in node.agents
            if agent.conversation_id is not None
        ]
        if not agents:
            return
        # Shells that were lost with the git directory come back for the
        # agents that still start theirs.
        await asyncio.to_thread(self.confinement.confine_all)
        statuses = await self.launcher().statuses(
            [conversation_id for _, _, conversation_id in agents]
        )
        starting = {
            intent.node_id
            for intent in self.org_state.run_state().open_intents()
            if intent.kind in STARTING
        }
        for place_id, role, conversation_id in agents:
            status = statuses.get(conversation_id)
            agent = str(conversation_id)
            if self.org_state.paused is None:
                self.message_delivery.note_running(
                    agent, place_id, status == ConversationExecutionStatus.RUNNING
                )
            if status is not None:
                shown = self.shown(agent, status)
            else:
                shown = "idle" if place_id in starting else AGENT_LOST
            with suppress(NotFound):
                store.set_agent(place_id, role, status=shown)
        await self._resync_hand_overs()

    async def _resync_hand_overs(self) -> None:
        """Read back how each IC's hand-over conversation stands. One that
        ended or was lost while nothing followed it is ended here."""
        store = self.org()
        modes = [
            (node.id, mode)
            for node in store.places()
            if (mode := hand_over_of(node)) is not None
        ]
        if not modes:
            return
        statuses = await self.launcher().statuses(
            [UUID(mode.conversation_id) for _, mode in modes]
        )
        starting = {
            intent.node_id
            for intent in self.org_state.run_state().open_intents()
            if intent.kind == "hand_over"
        }
        for place_id, mode in modes:
            status = statuses.get(UUID(mode.conversation_id))
            if status is not None:
                self._on_hand_over_status(store, mode.conversation_id, status)
            elif place_id not in starting and mode.status != AGENT_LOST:
                store.set_hand_over(
                    place_id, mode.model_copy(update={"status": AGENT_LOST})
                )
                self.team_splits.hand_over_ended(place_id)

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
        self.launcher()
        store = self.org()
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
            self.reviews.accept_review(place_id, "", **decided)
            return store.place(place_id) or place
        if escalation_id == ORG_WAITS:
            # The human goes on without the Manager.
            self.resumable_work.begin(place_id, LandOrg(org_holds=True))
            return store.place(place_id) or place
        if escalation_id == LIMIT_WAITS:
            self.session_limits.go_on_past_limit(place_id)
        if lost:
            # A new Manager starts with the answer; the IC goes on as it was.
            self.team_messaging.tell_manager(
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
