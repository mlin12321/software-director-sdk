"""What is said between teams: a message or a document from one team's
member to another's, what the director tells a team, and the teams that
are stopped and set going again."""

import json
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from openhands.agent_server.director.documents.store import DocumentRef, DocumentStore
from openhands.agent_server.director.intents import LaunchMember, StartTeam
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import (
    AgentRef,
    Escalation,
    OrgNode,
    SessionTeam,
    Stopped,
    ThreadMessage,
)
from openhands.agent_server.director.org import TeamMember
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    OrgStore,
    instant,
    is_division_lead,
    new_agents,
    team_ref,
    utc_now,
)
from openhands.agent_server.director.prompts import (
    failure_notice,
    manager_check,
)
from openhands.agent_server.director.run_state import (
    CHECKS,
    HANDING,
    PROGRESS,
    RUNNING,
    TASK,
)
from openhands.agent_server.director.service.check_count_service import (
    CheckCountService,
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
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
    thread_name,
)
from openhands.agent_server.director.service.stall_watch_service import (
    progress_of,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from openhands.agent_server.director.service.team_member_start_service import (
    TeamMemberStartService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import (
    Conflict,
    Invalid,
    NotFound,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# What follows a team's purpose in its task once something started it.
STARTED_WITH = "\n\nWhat your team is started with:\n"


def failure_waits(role: str) -> str:
    """The id of the escalation that tells the human an agent failed."""
    return f"{role}-failed"


@dataclass
class TeamMessagingService:
    """Carries what is said between teams and to them.

    A message to a member that has no conversation yet starts it; one to
    a team that is stopped waits. The service also counts what wakes a
    Manager, records that a team made progress, and stops a team and sets
    it going again.
    """

    org: Callable[[], OrgStore]
    """The org whose teams talk."""
    org_state: OrgStateService
    """The run state, and whether the org is paused."""
    org_settings: OrgSettingsService
    """The settings of the org, for when a Manager is woken."""
    org_save: OrgSaveService
    """Saves the tree a step that was begun refers to."""
    team_types: TeamTypeService
    """Who the members of a team are and what they are called."""
    team_history: TeamHistoryService
    """Where what became of a team is kept."""
    session_drafts: SessionDraftService
    """Where the threads and the documents passed between teams are kept."""
    documents: DocumentStore
    """Where a document that is sent is read."""
    check_counts: CheckCountService
    """What a Manager's checks count."""
    message_delivery: MessageDeliveryService
    """Sends an agent a message."""
    member_start: TeamMemberStartService
    """Says where a member that has no conversation yet is derived from."""
    resumable_work: ResumableWorkService
    """Begins the step that starts a member."""
    work_queue: WorkQueueService
    """What carries out setting a team going again."""
    launcher: Callable[[], TeamLauncher]
    """What reaches the conversations; raises where there is none."""
    joined: Callable[[str], None]
    """Notes that the human has written in a conversation."""
    clock: Callable[[], datetime]
    release_reviews: Callable[[str | None], None]
    """Hands the reviews that wait at a team to its IC
    (:meth:`ReviewService.release_reviews`); a function, because that
    service holds this one."""
    """The time, for when a team last made progress."""
    _adding_manager: threading.Lock = field(default_factory=threading.Lock, init=False)

    def send_ref(
        self,
        sender_place_id: str,
        sender: str,
        target_place_id: str,
        target: str,
        ref: DocumentRef,
    ) -> None:
        """Pass a document from one agent of a session to another, to read
        from now on in that session: the one way a document goes between
        agents that is not a matter of course. The reader's team is told
        of it, and of what it says, with its next message. Safe from any
        thread.

        Raises:
            NotFound: One of the two teams is no longer in the org, or
                there is no such document.
            Invalid: The org does not allow it.
        """
        store = self.org()
        sender_place = store.place(sender_place_id)
        target_place = store.place(target_place_id)
        if sender_place is None or target_place is None:
            raise NotFound("That team is no longer in the org")
        if self.documents.read(sender_place.session, ref) is None:
            raise NotFound(
                f"There is no {ref.name} of {ref.folder or 'the repository'}"
            )
        if not self.org_settings.org_kind.allows_ref(
            TeamMember(team_ref(sender_place), sender),
            TeamMember(team_ref(target_place), target),
            ref,
        ):
            raise Invalid(
                "This org does not let the "
                f"{self.team_types.team(sender_place).title(sender)} of "
                f"{sender_place.title!r} pass a document to {target_place.title!r}"
            )
        self.session_drafts.keep_passed(target_place_id, ref.folder, ref.name)

    def send_msg(
        self,
        sender_place_id: str,
        sender: str,
        target_place_id: str,
        target: str,
        text: str,
        note: str = "",
    ) -> bool:
        """Keep and deliver what one agent of a session writes to another:
        the one way every message between agents goes. Safe from any
        thread.

        A message from the agent that gives the target its work (an IC to
        the IC of the team directly below) is also a task: see ``_assign``.

        Args:
            sender_place_id: The team of the agent that writes.
            target_place_id: The team of the agent written to: the same
                one, or one directly below.
            note: What the director adds for the reader; it is not kept.

        Returns:
            Whether the target's team was stopped and goes on with this.

        Raises:
            NotFound: One of the two teams is no longer in the org.
            Invalid: The org does not allow it.
        """
        store = self.org()
        sender_place = store.place(sender_place_id)
        target_place = store.place(target_place_id)
        if sender_place is None or target_place is None:
            raise NotFound("That team is no longer in the org")
        if not self.org_settings.org_kind.allows_msg(
            TeamMember(team_ref(sender_place), sender),
            TeamMember(team_ref(target_place), target),
        ):
            raise Invalid(
                "This org does not let the "
                f"{self.team_types.team(sender_place).title(sender)} of "
                f"{sender_place.title!r} write to {target_place.title!r}"
            )
        # The member that works in a team's folder is the one the team is
        # written to through, and the one that is given work.
        to_writer = target == self.team_types.team(target_place).writer
        between_ics = to_writer and sender == self.team_types.team(sender_place).writer
        # A thread is named by its reader's team when an IC reads it, and
        # by its writer's when an IC writes to a Manager.
        if to_writer:
            first, second = sender_place_id, target_place_id
        else:
            first, second = target_place_id, sender_place_id
        self.session_drafts.keep_message(
            thread_name(first, second, between_ics),
            ThreadMessage(at=instant(), sender=sender, text=text),
        )
        stopped = self._assign(target_place_id) if between_ics else None
        writer = self.team_types.team(sender_place).title(sender)
        heading = (
            f"Your {writer} writes:"
            if sender_place_id == target_place_id or not to_writer
            else f"The {writer} of the team above ({sender_place.title}) writes:"
        )
        if stopped is not None:
            note += "\n\nIt had stopped your team; with this message your team goes on."
        self.tell(target_place_id, target, f"{heading}\n{text}{note}")
        return stopped is not None

    def _assign(self, place_id: str) -> Stopped | None:
        """What a message does to a team when it comes from the agent that
        gives the team its work: the team owes a review again, goes on if
        it was stopped, and its time without progress starts again.

        Returns:
            How the team was stopped, if it was.
        """
        store = self.org()
        team = store.place(place_id)
        if team is None:
            return None
        if team.review is not None and team.review.state == "accepted":
            # What it does with this is new work, which it submits again:
            # left accepted, it would go up with a sibling's, unreviewed.
            store.set_review(place_id, None)
        stopped = team.stopped
        if stopped is not None:
            store.set_stopped(place_id, None)
            self.work_queue.enqueue(
                lambda: self.go_on(place_id, stopped.running), place_id
            )
        self.progressed(place_id, by_above=True)
        return stopped

    def tell(
        self,
        place_id: str,
        member: str,
        text: str,
        *,
        human: bool = False,
        division_lead: bool = False,
    ) -> str | None:
        """Send a member of a team a message. A team has no agent for a
        member until something is sent to it: the first message starts one
        and is part of its briefing, or follows it. Safe from any thread.

        Args:
            human: The human's own words, which reach an agent at once and
                are not marked as the director's.
            division_lead: The text asks for what a session's Division Lead
                place is for: a review, or to settle the session's
                sub-teams. Nothing else starts or reaches an agent there.

        Returns:
            The member's conversation; None when the team is gone or the
            message is not for an agent of that place.
        """
        place = self.org().place(place_id)
        if place is None:
            return None
        if member == self.team_types.team(place).writer:
            return self._tell_writer(
                place_id, member, text, human=human, division_lead=division_lead
            )
        return self._tell_other(place_id, member, text, division_lead=division_lead)

    def tell_manager(
        self, place_id: str, text: str, *, division_lead: bool = False
    ) -> None:
        """Send a message from the director to the member of a team that
        settles what its sub-teams are. Safe from any thread."""
        place = self.org().place(place_id)
        if place is not None:
            self.tell(
                place_id,
                self.team_types.team(place).manager,
                text,
                division_lead=division_lead,
            )

    def _tell_other(
        self, place_id: str, member: str, text: str, *, division_lead: bool
    ) -> str | None:
        """Send a message to a member that does not work in the team's
        folder. The first message starts it, and follows its briefing."""
        store = self.org()
        with self._adding_manager:
            place = store.place(place_id)
            if place is None:
                return None
            if is_division_lead(place) and not division_lead:
                return None
            # One that could not be started is started afresh.
            lost = [
                agent
                for agent in place.agents
                if agent.role == member and agent.status == AGENT_LOST
            ]
            if lost or self.team_types.team(place).conversation(member) is None:
                added = AgentRef(
                    role=member, conversation_id=str(uuid4()), status="idle"
                )
                kept = [agent for agent in place.agents if agent not in lost]
                place = store.update_place(place_id, agents=[*kept, added])
                task = (
                    self.org_state.run_state().marked(TASK).get(place_id)
                    or place.summary
                )
                self.resumable_work.begin(
                    place_id, LaunchMember(task=task, role=member)
                )
        conversation_id = self.team_types.team(place).conversation(member)
        self.message_delivery.notify(conversation_id, text)
        return conversation_id

    def tell_above(
        self,
        team: SessionTeam,
        notice: str,
        *,
        waits: str,
        title: str,
        detail: str,
        raised_by: str | None = None,
    ) -> None:
        """Tell the IC above a team of something wrong with it. Where no IC
        works above it (the team a session was opened at, and the session's
        Division Lead place), the human is asked instead, once.

        Args:
            notice: What the IC above is told.
            waits: The id of the escalation the human is asked with.
            raised_by: The agent of the team the human's answer goes to;
                None for the one that works in its folder.
        """
        store = self.org()
        above = store.place(team.parent_id) if team.parent_id else None
        if above is not None and not is_division_lead(above):
            self.tell(above.id, self.team_types.team(above).reviewer, notice)
            return
        now = store.place(team.id)
        if now is None or any(e.id == waits for e in now.escalations):
            return
        store.raise_escalation(
            team.id,
            waits,
            kind="critical_issue",
            title=title,
            detail=detail,
            at=utc_now(),
            raised_by=raised_by or self.team_types.team(team).writer,
        )
        self.org_save.save_soon()

    def tell_of_failure(self, place_id: str, role: str, reason: str) -> None:
        """Keep why an agent of a team failed, and tell of it."""
        store = self.org()
        with suppress(NotFound):
            store.set_agent(place_id, role, reason=reason)
        team = store.place(place_id)
        if team is None:
            return
        name = self.team_types.team(team).title(role)
        self.team_history.record_event(
            team.parent_id, "failed", team, f"{name}: {reason}"
        )
        lost = any(a.role == role and a.status == AGENT_LOST for a in team.agents)
        self.tell_above(
            team,
            failure_notice(
                team,
                name,
                reason,
                works=role == self.team_types.team(team).writer,
                names=self.team_types.team(team).names,
            ),
            waits=failure_waits(role),
            title=f"The {name} of {team.title!r} failed",
            detail=(
                f"{reason}\n"
                + (
                    "It has no conversation on this server. Answer this to "
                    "start the team again."
                    if lost
                    else "It is at rest. Answer this to run it again: your "
                    "answer is sent to it."
                )
            ),
            raised_by=role,
        )

    def tell_writer(
        self,
        place_id: str,
        text: str,
        *,
        human: bool = False,
        division_lead: bool = False,
    ) -> str | None:
        """Send a message to the member of a team that works in its
        folder. Safe from any thread.

        Returns:
            Its conversation; None when the team is gone.
        """
        place = self.org().place(place_id)
        if place is None:
            return None
        return self.tell(
            place_id,
            self.team_types.team(place).writer,
            text,
            human=human,
            division_lead=division_lead,
        )

    def _tell_writer(
        self,
        place_id: str,
        member: str,
        text: str,
        *,
        human: bool,
        division_lead: bool,
    ) -> str | None:
        """Send a message to the member that works in the team's folder. A
        team that was only set up has no such agent until something is sent
        to it: the first message starts one and is part of its briefing."""
        store = self.org()
        with self._adding_manager:
            place = store.place(place_id)
            if place is None:
                return None
            conversation_id = self.team_types.team(place).conversation(member)
            leads = is_division_lead(place)
            if conversation_id is None and leads and not division_lead:
                return None
            if conversation_id is None:
                [ic] = new_agents(uuid4(), member)
                store.update_place(place_id, agents=[ic, *place.agents])
                purpose = self.purpose(place)
                self.resumable_work.begin(
                    place_id,
                    StartTeam(
                        task=f"{purpose}{STARTED_WITH}{text}",
                        purpose=purpose,
                        parent=self.member_start.derived_from(place),
                        checkout=place.worktree_path
                        if self.member_start.in_place(place)
                        else None,
                    ),
                )
                if human and ic.conversation_id:
                    self.joined(ic.conversation_id)
                return ic.conversation_id
        if human:
            self.message_delivery.post(conversation_id, text, at_once=True)
            self.joined(conversation_id)
        else:
            self.message_delivery.notify(conversation_id, text)
        return conversation_id

    def purpose(self, node: OrgNode) -> str:
        """What a team is for, without the message it was started with."""
        task = self.org_state.run_state().marked(TASK).get(node.id) or node.summary
        return task.split(STARTED_WITH)[0]

    def count_accepted_review(self, place_id: str) -> None:
        """Count a review a team's IC accepted, and wake the team's Manager
        to look at its teams when that makes as many as the org set. Safe
        from any thread."""
        names = self.team_types.names_at(place_id)
        limit = self.org_settings.settings.manager_check_accepted_reviews
        if not limit:
            return
        with self.check_counts.lock:
            counted = self.check_counts.counted(place_id)
            counted["accepted"] += 1
            due = counted["accepted"] >= limit
            if due:
                counted["accepted"] = 0
            self.org_state.run_state().mark(CHECKS, place_id, json.dumps(counted))
        if due:
            self.wake_manager(
                place_id,
                f"your {names.reviewer} has accepted {limit} reviews since the "
                "last look "
                "of this kind.",
                counted,
                after_successes=True,
            )

    def count_escalation_from_below(self, place_id: str, below_id: str) -> None:
        """Count an escalation the IC of a team directly below raised to a
        team's IC, and wake the team's Manager when that team has raised as
        many, on the work it has in hand, as the org set. Safe from any
        thread."""
        names = self.team_types.names_at(place_id)
        limit = self.org_settings.settings.manager_check_ic_setbacks
        team = self.org().place(below_id)
        if not limit or team is None:
            return
        with self.check_counts.lock:
            counted = self.check_counts.counted(place_id)
            raised = counted["escalations"].get(below_id, 0) + 1
            due = raised >= limit
            told = dict(counted["escalations"], **{below_id: raised})
            counted["escalations"][below_id] = 0 if due else raised
            self.org_state.run_state().mark(CHECKS, place_id, json.dumps(counted))
        if due:
            self.wake_manager(
                place_id,
                f"the {names.writer} of {team.title!r} has escalated {limit} times to "
                f"your {names.writer} on the work it has in hand.",
                {**counted, "escalations": told},
                team=team,
            )

    def wake_manager(
        self,
        place_id: str,
        reason: str,
        counted: dict[str, Any],
        *,
        team: SessionTeam | None = None,
        after_successes: bool = False,
    ) -> None:
        """Wake a team's Manager to look at its teams, and keep that it was.

        Args:
            team: The team whose escalations reached the limit, when that
                is why; it is kept with the event, so that the record says
                on which of a team's tasks the limit was reached.
        """
        store = self.org()
        now = self.clock()
        quiet = {}
        for team_id, kept in self.org_state.run_state().marked(PROGRESS).items():
            at = progress_of(kept).get("at")
            if at:
                since = now - datetime.fromisoformat(at)
                quiet[team_id] = int(since.total_seconds() // 60)
        teams = store.below(place_id)
        waiting = {}
        for below in teams:
            writer = self.team_types.team(below).writer
            raised = [e.raised_at for e in below.escalations if e.raised_by == writer]
            if raised:
                since = now - datetime.fromisoformat(min(raised))
                waiting[below.id] = int(since.total_seconds() // 60)
        self.team_history.record_event(place_id, "woken", team, reason)
        self.tell_manager(
            place_id,
            manager_check(
                reason,
                teams,
                quiet,
                waiting,
                counted["escalations"],
                self.team_history.history_of(place_id),
                after_successes=after_successes,
                escalation_limit=self.org_settings.settings.manager_check_ic_setbacks,
                names=self.team_types.names_at(place_id),
            ),
        )

    async def stop(self, place_id: str) -> None:
        team = self.org().place(place_id)
        if team is None or team.stopped is None or self.org_state.paused is not None:
            return
        launcher = self.launcher()
        for conversation_id in self.org().conversations_below(team):
            if conversation_id in self.org_state.running:
                with suppress(NotFound):
                    await launcher.pause(UUID(conversation_id))

    async def go_on(
        self, place_id: str, running: list[str], *, own_too: bool = False
    ) -> None:
        """Set a team that was stopped going again: what waited for its
        agents is delivered, and those cut off in a turn run again.

        Args:
            own_too: Run the team's own IC again as well, when no message
                to it does.
        """
        team = self.org().place(place_id)
        if team is None or team.stopped is not None:
            return
        own = None if own_too else self.team_types.writer_conversation(team)
        if self.org_state.paused is not None:
            # The org goes on later: those cut off are run again then, like
            # any agent that was in a turn when the org was paused.
            for conversation_id in running:
                found = self.org().find_by_conversation(conversation_id)
                if found is not None and conversation_id != own:
                    self.org_state.running.add(conversation_id)
                    self.org_state.run_state().mark(
                        RUNNING, conversation_id, found[0].id
                    )
            return
        conversations = self.org().conversations_below(team)
        # The time it was stopped is not time it showed no progress.
        for found in map(self.org().find_by_conversation, conversations):
            self.progressed(found[0].id if found else None)
        # In the order it was sent, and before the message that set the
        # team going, which waits behind this in the team's lane.
        for held in self.org_state.run_state().held():
            if held.conversation_id in conversations:
                await self.message_delivery.deliver(
                    held.seq, held.conversation_id, held.text, at_once=True
                )
        for conversation_id in conversations:
            # The team's own IC is run by the message that set it going.
            if conversation_id in running and conversation_id != own:
                await self.run_again(conversation_id, None)

    def progressed(self, place_id: str | None, *, by_above: bool = False) -> None:
        """Note that a team got somewhere just now: the time it may show no
        progress starts again. Safe from any thread.

        Args:
            by_above: The IC of the team above turned to it just now (gave
                it a task, wrote to it, sent its work back), so the time
                until that IC is next told how it stands starts again too.
        """
        if place_id is None:
            return
        run = self.org_state.run_state()
        seen = progress_of(run.marked(PROGRESS).get(place_id))
        now = self.clock().isoformat()
        checked = now if by_above else seen.get("checked") or now
        run.mark(
            PROGRESS,
            place_id,
            json.dumps({"at": now, "head": seen.get("head"), "checked": checked}),
        )

    async def run_again(self, conversation_id: str, note: str | None) -> None:
        """Have an agent that a restart cut off carry on. It stays one that
        is being taken up, and so shows paused, until it is seen running: its
        run starts a moment after it is told, and what it says until then is
        the error the restart left."""
        if self.org().is_stopped(conversation_id):
            # Its team was stopped meanwhile and goes on when it is told to.
            self.org_state.resuming.discard(conversation_id)
            return
        launcher = self.launcher()
        agent = UUID(conversation_id)
        try:
            await launcher.resume(agent, note)
        except NotFound:
            self.org_state.resuming.discard(conversation_id)
            return
        # One that ran already says nothing new of it.
        running = (await launcher.statuses([agent])).get(agent)
        if running == ConversationExecutionStatus.RUNNING:
            self.org_state.resuming.discard(conversation_id)

    def note_hand_over(
        self, place_id: str, *, team: str | None = None, wrote: str | None = None
    ) -> None:
        """Keep what a team's hand-over conversation did, for its IC to be
        told when it ends. Safe from any thread."""
        run = self.org_state.run_state()
        kept = run.marked(HANDING).get(place_id)
        if kept is None:
            return
        done = json.loads(kept)
        if team is not None and team not in done["given"]:
            done["given"].append(team)
        if wrote is not None and wrote not in done["written"]:
            done["written"].append(wrote)
        run.mark(HANDING, place_id, json.dumps(done))

    def hand_escalation(
        self,
        holder_id: str,
        raiser: SessionTeam,
        through: SessionTeam,
        escalation: Escalation,
        passed_up: list[str],
    ) -> None:
        """Tell the agent that now holds an escalation. Safe from any thread.

        Args:
            through: The team directly below the holder that it comes from:
                the one that raised it, or the one that passed it up.
        """
        holder = self.org().place(holder_id)
        if holder is None:
            return
        history = "".join(f"\nPassed up by {entry}" for entry in passed_up)
        if escalation.raised_by == self.team_types.team(raiser).writer:
            # It is woken now anyway, so it gets what waits for its review too.
            self.release_reviews(holder_id)
            self.count_escalation_from_below(holder_id, through.id)
        raised_by = escalation.raised_by
        text = (
            f"The {self.team_types.team(raiser).title(raised_by)} of team "
            f"{raiser.title!r} (id {raiser.id}) escalates {escalation.title!r} "
            f"(escalation {escalation.id}):\n"
            f"{escalation.detail or '(no detail)'}{history}\n"
            "Settle it with resolve_escalation if you can; pass it up only with "
            "the reason you cannot."
        )
        self.tell(
            holder_id, self.team_types.team(holder).answers(escalation.raised_by), text
        )

    def stop_team(self, place_id: str, reason: str) -> None:
        """Stop a team for the IC of the team above: its agents and those
        of every team below it end the step they are in and are sent
        nothing until that IC writes to the team. Safe from any thread.

        Raises:
            NotFound: The team does not exist.
            Conflict: It is stopped already.
        """
        names = self.team_types.names_at(place_id)
        store = self.org()
        team = store.place(place_id)
        if team is None:
            raise NotFound(f"Team {place_id} does not exist")
        if team.stopped is not None:
            raise Conflict(f"Team {place_id} is stopped already")
        running = [
            conversation_id
            for conversation_id in self.org().conversations_below(team)
            if conversation_id in self.org_state.running
        ]
        store.set_stopped(
            place_id, Stopped(at=utc_now(), reason=reason, running=sorted(running))
        )
        if self.work_queue.is_open:
            self.work_queue.enqueue(lambda: self.stop(place_id), place_id)
        self.org_save.save_soon()
        self.team_history.record_event(team.parent_id, "stopped", team, reason)
        # A stopped team is waited for no longer by the reviews beside it.
        self.release_reviews(team.parent_id)
        if team.parent_id is not None:
            self.tell_manager(
                team.parent_id,
                f"Your {names.writer} stopped the team {team.title!r} (id {team.id}): "
                f"{reason}\nIts agents are at rest and nothing is sent to "
                f"them. It goes on when your {names.writer} writes to it with "
                "msg_team. "
                "Nothing is asked of you. If it should not go on at all, you "
                f"may remove it with prune_team, or have your {names.writer} propose "
                "teams that take its place.",
            )
