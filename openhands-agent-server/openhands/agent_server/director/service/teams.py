"""Starting a team's agents and what is sent to them."""

import asyncio
import json
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from openhands.agent_server.director.context import (
    briefing_block,
    steady,
)
from openhands.agent_server.director.documents.store import DocumentRef
from openhands.agent_server.director.git import (
    GitError,
    bring_over,
    head_of,
    uncommitted,
)
from openhands.agent_server.director.intents import Brief, LaunchMember, StartTeam
from openhands.agent_server.director.launch import StartedAgent
from openhands.agent_server.director.models import (
    AgentRef,
    OrgNode,
    SessionTeam,
    Stopped,
    ThreadMessage,
)
from openhands.agent_server.director.org import TeamMember
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    is_division_lead,
    session_team_id,
    utc_now,
)
from openhands.agent_server.director.prompts import (
    brief,
    failure_notice,
    hand_over_brief,
    left_for_team,
    manager_check,
    role_text,
)
from openhands.agent_server.director.run_state import (
    CHECKS,
    HANDING,
    PROGRESS,
    RUNNING,
    SENT,
    TASK,
    TOLD,
    Intent,
)
from openhands.agent_server.director.service.delivery import Delivery
from openhands.agent_server.director.service.records import REPORT
from openhands.agent_server.director.service.state import (
    _REASON_KEPT,
    _STARTED_WITH,
    LEFT_REFS,
    _agent_title,
    _base_of,
    _branch_of,
    _Held,
    _instant,
    _new_agents,
    _progress_of,
    checkout_root,
    failure_waits,
    team_ref,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import (
    Conflict,
    Invalid,
    NotFound,
)
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.spec import Tool


logger = get_logger(__name__)


class Teams(Delivery):
    """Starts agents and carries what is said between teams."""

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
        store = self.org
        sender_place = store.place(sender_place_id)
        target_place = store.place(target_place_id)
        if sender_place is None or target_place is None:
            raise NotFound("That team is no longer in the org")
        if self.documents.read(sender_place.session, ref) is None:
            raise NotFound(
                f"There is no {ref.name} of {ref.folder or 'the repository'}"
            )
        if not self.org_kind.allows_ref(
            TeamMember(team_ref(sender_place), sender),
            TeamMember(team_ref(target_place), target),
            ref,
        ):
            raise Invalid(
                f"This org does not let the {self.team(sender_place).title(sender)} "
                f"of {sender_place.title!r} pass a document to {target_place.title!r}"
            )
        with self._keeping:
            run = self._runs()
            kept = run.marked(SENT).get(target_place_id)
            passed = json.loads(kept) if kept else []
            if [ref.folder, ref.name] not in passed:
                run.mark(
                    SENT, target_place_id, json.dumps([*passed, [ref.folder, ref.name]])
                )

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
        store = self.org
        sender_place = store.place(sender_place_id)
        target_place = store.place(target_place_id)
        if sender_place is None or target_place is None:
            raise NotFound("That team is no longer in the org")
        if not self.org_kind.allows_msg(
            TeamMember(team_ref(sender_place), sender),
            TeamMember(team_ref(target_place), target),
        ):
            raise Invalid(
                f"This org does not let the {self.team(sender_place).title(sender)} "
                f"of {sender_place.title!r} write to {target_place.title!r}"
            )
        # The member that works in a team's folder is the one the team is
        # written to through, and the one that is given work.
        to_writer = target == self.team(target_place).writer
        between_ics = to_writer and sender == self.team(sender_place).writer
        # A thread is named by its reader's team when an IC reads it, and
        # by its writer's when an IC writes to a Manager.
        if to_writer:
            first, second = sender_place_id, target_place_id
        else:
            first, second = target_place_id, sender_place_id
        self._keep_message(
            first,
            second,
            ThreadMessage(at=_instant(), sender=sender, text=text),
            ic_to_ic=between_ics,
        )
        stopped = self._assign(target_place_id) if between_ics else None
        writer = self.team(sender_place).title(sender)
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
        store = self.org
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
            self._enqueue(lambda: self._go_on(place_id, stopped.running), place_id)
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
        place = self.org.place(place_id)
        if place is None:
            return None
        if member == self.team(place).writer:
            return self._tell_writer(
                place_id, member, text, human=human, division_lead=division_lead
            )
        return self._tell_other(place_id, member, text, division_lead=division_lead)

    def tell_overseer(
        self, place_id: str, text: str, *, division_lead: bool = False
    ) -> None:
        """Send a message from the director to the member of a team that
        settles what its sub-teams are. Safe from any thread."""
        place = self.org.place(place_id)
        if place is not None:
            self.tell(
                place_id, self.team(place).overseer, text, division_lead=division_lead
            )

    def _tell_other(
        self, place_id: str, member: str, text: str, *, division_lead: bool
    ) -> str | None:
        """Send a message to a member that does not work in the team's
        folder. The first message starts it, and follows its briefing."""
        store = self.org
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
            if lost or self.team(place).conversation(member) is None:
                added = AgentRef(
                    role=member, conversation_id=str(uuid4()), status="idle"
                )
                kept = [agent for agent in place.agents if agent not in lost]
                place = store.update_place(place_id, agents=[*kept, added])
                task = self._runs().marked(TASK).get(place_id) or place.purpose
                self._begin(place_id, LaunchMember(task=task, role=member))
        conversation_id = self.team(place).conversation(member)
        self.notify(conversation_id, text)
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
        store = self.org
        above = store.place(team.parent_id) if team.parent_id else None
        if above is not None and not is_division_lead(above):
            self.tell(above.id, self.team(above).reviewer, notice)
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
            raised_by=raised_by or self.team(team).writer,
        )
        self._save_soon()

    def tell_of_failure(self, place_id: str, role: str, reason: str) -> None:
        """Keep why an agent of a team failed, and tell of it."""
        store = self.org
        with suppress(NotFound):
            store.set_agent(place_id, role, reason=reason)
        team = store.place(place_id)
        if team is None:
            return
        name = self.team(team).title(role)
        self.record_event(team.parent_id, "failed", team, f"{name}: {reason}")
        lost = any(a.role == role and a.status == AGENT_LOST for a in team.agents)
        self.tell_above(
            team,
            failure_notice(
                team,
                name,
                reason,
                works=role == self.team(team).writer,
                names=self.team(team).names,
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
        place = self.org.place(place_id)
        if place is None:
            return None
        return self.tell(
            place_id,
            self.team(place).writer,
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
        store = self.org
        with self._adding_manager:
            place = store.place(place_id)
            if place is None:
                return None
            conversation_id = self.team(place).conversation(member)
            leads = is_division_lead(place)
            if conversation_id is None and leads and not division_lead:
                return None
            if conversation_id is None:
                [ic] = _new_agents(uuid4(), member)
                store.update_place(place_id, agents=[ic, *place.agents])
                purpose = self._purpose(place)
                self._begin(
                    place_id,
                    StartTeam(
                        task=f"{purpose}{_STARTED_WITH}{text}",
                        purpose=purpose,
                        parent=self._derived_from(place),
                        checkout=place.worktree_path if self._in_place(place) else None,
                    ),
                )
                if human and ic.conversation_id:
                    self.joined(ic.conversation_id)
                return ic.conversation_id
        if human:
            self._post(conversation_id, text, at_once=True)
            self.joined(conversation_id)
        else:
            self.notify(conversation_id, text)
        return conversation_id

    def _in_place(self, place: SessionTeam) -> bool:
        """Whether a team that starts works in the checkout it has: a
        session's Division Lead place, and the team a session is opened
        at when sessions get no worktree or the session has one already."""
        return is_division_lead(place) or (
            self.org.opens_session(place)
            and (
                place.isolation != "worktree"
                or checkout_root(place.worktree_path)
                != checkout_root(str(self._repo_dir))
            )
        )

    def _derived_from(self, place: SessionTeam) -> str | None:
        """The place whose IC the IC of ``node`` is derived from when it
        starts: the team above, or for the team a session is opened at and
        its reviewer, who of the session works already."""
        store = self.org
        opened = store.session_of(place)
        if is_division_lead(place) and opened:
            return session_team_id(opened.team_id, opened.id)
        if store.opens_session(place):
            # A session that went a team higher has an IC below already.
            return next(
                (team.id for team in store.below(place.id) if team.agents),
                place.parent_id,
            )
        return place.parent_id

    async def _template(self) -> str | None:
        """A conversation of the org to derive an agent from when nothing
        above it has started: the IC a session started with, oldest first."""
        store, launcher = self.org, self._require_launcher()
        for session in store.sessions():
            ic = session.conversation_id
            if ic is not None and await launcher.started(UUID(ic)) is not None:
                return ic
        return None

    def _purpose(self, node: OrgNode) -> str:
        """What a team is for, without the message it was started with."""
        task = self._runs().marked(TASK).get(node.id) or node.purpose
        return task.split(_STARTED_WITH)[0]

    def count_accepted_review(self, place_id: str) -> None:
        """Count a review a team's IC accepted, and wake the team's Manager
        to look at its teams when that makes as many as the org set. Safe
        from any thread."""
        names = self.names_at(place_id)
        limit = self.settings.manager_check_accepted_reviews
        if not limit:
            return
        with self._counting:
            counted = self._counted(place_id)
            counted["accepted"] += 1
            due = counted["accepted"] >= limit
            if due:
                counted["accepted"] = 0
            self._runs().mark(CHECKS, place_id, json.dumps(counted))
        if due:
            self._wake_manager(
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
        names = self.names_at(place_id)
        limit = self.settings.manager_check_ic_setbacks
        team = self.org.place(below_id)
        if not limit or team is None:
            return
        with self._counting:
            counted = self._counted(place_id)
            raised = counted["escalations"].get(below_id, 0) + 1
            due = raised >= limit
            told = dict(counted["escalations"], **{below_id: raised})
            counted["escalations"][below_id] = 0 if due else raised
            self._runs().mark(CHECKS, place_id, json.dumps(counted))
        if due:
            self._wake_manager(
                place_id,
                f"the {names.writer} of {team.title!r} has escalated {limit} times to "
                f"your {names.writer} on the work it has in hand.",
                {**counted, "escalations": told},
                team=team,
            )

    def _wake_manager(
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
        store = self.org
        now = self._clock()
        quiet = {}
        for team_id, kept in self._runs().marked(PROGRESS).items():
            at = _progress_of(kept).get("at")
            if at:
                since = now - datetime.fromisoformat(at)
                quiet[team_id] = int(since.total_seconds() // 60)
        teams = store.below(place_id)
        waiting = {}
        for below in teams:
            writer = self.team(below).writer
            raised = [e.raised_at for e in below.escalations if e.raised_by == writer]
            if raised:
                since = now - datetime.fromisoformat(min(raised))
                waiting[below.id] = int(since.total_seconds() // 60)
        self.record_event(place_id, "woken", team, reason)
        self.tell_overseer(
            place_id,
            manager_check(
                reason,
                teams,
                quiet,
                waiting,
                counted["escalations"],
                self.history_of(place_id),
                after_successes=after_successes,
                escalation_limit=self.settings.manager_check_ic_setbacks,
                names=self.names_at(place_id),
            ),
        )

    async def _stop(self, place_id: str) -> None:
        team = self.org.place(place_id)
        if team is None or team.stopped is None or self._paused is not None:
            return
        launcher = self._require_launcher()
        for conversation_id in self._conversations_below(team):
            if conversation_id in self._running:
                with suppress(NotFound):
                    await launcher.pause(UUID(conversation_id))

    async def _go_on(
        self, place_id: str, running: list[str], *, own_too: bool = False
    ) -> None:
        """Set a team that was stopped going again: what waited for its
        agents is delivered, and those cut off in a turn run again.

        Args:
            own_too: Run the team's own IC again as well, when no message
                to it does.
        """
        team = self.org.place(place_id)
        if team is None or team.stopped is not None:
            return
        own = None if own_too else self.writer_conversation(team)
        if self._paused is not None:
            # The org goes on later: those cut off are run again then, like
            # any agent that was in a turn when the org was paused.
            for conversation_id in running:
                found = self.org.find_by_conversation(conversation_id)
                if found is not None and conversation_id != own:
                    self._running.add(conversation_id)
                    self._runs().mark(RUNNING, conversation_id, found[0].id)
            return
        conversations = self._conversations_below(team)
        # The time it was stopped is not time it showed no progress.
        for found in map(self.org.find_by_conversation, conversations):
            self.progressed(found[0].id if found else None)
        # In the order it was sent, and before the message that set the
        # team going, which waits behind this in the team's lane.
        for held in self._runs().held():
            if held.conversation_id in conversations:
                self._at_once.add(held.seq)
                await self._deliver(held.seq, held.conversation_id, held.text)
        for conversation_id in conversations:
            # The team's own IC is run by the message that set it going.
            if conversation_id in running and conversation_id != own:
                await self._run_again(conversation_id, None)

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
        run = self._runs()
        seen = _progress_of(run.marked(PROGRESS).get(place_id))
        now = self._clock().isoformat()
        checked = now if by_above else seen.get("checked") or now
        run.mark(
            PROGRESS,
            place_id,
            json.dumps({"at": now, "head": seen.get("head"), "checked": checked}),
        )

    def hand_over(self, team: SessionTeam, notes: str, contracts: str) -> None:
        """Give a team what its IC above passes on of the teams that were
        there before it: notes, kept with the team's reports, and the
        contracts its own sub-teams must keep. Both are the session's alone
        until its work is accepted. Starts nothing. Safe from any thread."""
        self.removed_team_handed_over(team, {REPORT: notes, "contracts": contracts})

    def _task_of(self, place: SessionTeam) -> str:
        """What a team that is started again is to do, from what the org
        kept of it."""
        specs = [spec for spec in self.specs_of(place) if not spec.old]
        task = specs[0].spec if specs else place.purpose
        if place.headline and place.headline not in task:
            task = f"{task}\n\nWhere the team last was: {place.headline}"
        return (
            f"{task}\n\nThis team was worked on before. "
            "What it committed is in your checkout: look at it first and "
            "carry on from there."
        )

    async def _run_again(self, conversation_id: str, note: str | None) -> None:
        """Have an agent that a restart cut off carry on. It stays one that
        is being taken up, and so shows paused, until it is seen running: its
        run starts a moment after it is told, and what it says until then is
        the error the restart left."""
        if self._is_stopped(conversation_id):
            # Its team was stopped meanwhile and goes on when it is told to.
            self._resuming.discard(conversation_id)
            return
        launcher = self._require_launcher()
        agent = UUID(conversation_id)
        try:
            await launcher.resume(agent, note)
        except NotFound:
            self._resuming.discard(conversation_id)
            return
        # One that ran already says nothing new of it.
        running = (await launcher.statuses([agent])).get(agent)
        if running == ConversationExecutionStatus.RUNNING:
            self._resuming.discard(conversation_id)

    def _briefing(self, place: SessionTeam, role: str, task: str) -> tuple[str, str]:
        """An agent's first message, and the context it tells, as kept to
        find later what changed."""
        contracts = self.contracts_of(place)
        context = self.context_of(place, role)
        text = brief(
            place,
            role,
            task,
            self.specs_of(place),
            contracts,
            self.reports_of(place),
            briefing_block(context),
            self.type_of(place).text_for(role),
            works=role == self.team(place).writer,
            titles=self.type_of(place).titles,
            role_said=self.role_text(place, role),
            names=self.team(place).names,
        )
        return text, json.dumps(steady(context))

    def _hand_over_briefing(self, place: SessionTeam, job: str) -> str:
        """The first message of an IC's hand-over conversation."""
        contracts = self.contracts_of(place)
        return hand_over_brief(
            place,
            job,
            self.specs_of(place),
            contracts,
            self.reports_of(place),
            briefing_block(self.context_of(place, self.team(place).writer)),
            titles=self.type_of(place).titles,
            role=role_text(self.team(place).names, works=True, hand_over=True),
            names=self.team(place).names,
        )

    async def _start_hand_over(
        self, place: SessionTeam, conversation_id: str, job: str
    ) -> bool:
        """Start the hand-over conversation of a team's IC, derived from its
        main one.

        Returns:
            Whether it runs.
        """
        names = self.team(place).names
        launcher = self._require_launcher()
        try:
            if await launcher.started(UUID(conversation_id)) is not None:
                return True
            parent = await self._started_ic(place)
            if parent is None:
                raise NotFound(f"Team {place.id} has no {names.writer} to hand over")
            text = await asyncio.to_thread(self._hand_over_briefing, place, job)
            tool_params = await asyncio.to_thread(
                self._confine, place.id, self.team(place).writer, conversation_id
            )
            adapt = await asyncio.to_thread(
                self._adapter, conversation_id, self.hand_over_tools
            )

            async def start() -> StartedAgent:
                return await launcher.start_derived(
                    UUID(parent),
                    UUID(conversation_id),
                    title=f"{place.title} ({names.writer}, hand-over)",
                    task=text,
                    isolation="shared",
                    tool_params=tool_params,
                    adapt_agent=adapt,
                )

            await self._when_there_is_room(place.id, start)
            if self._paused is not None:
                self._note_running(conversation_id, place.id, True)
                await launcher.pause(UUID(conversation_id))
        except _Held:
            raise
        except Exception:
            logger.exception("Could not start the hand-over of team %s", place.id)
            return False
        return True

    async def _brief(self, place_id: str, role: str, task: str) -> None:
        place = self.org.place(place_id)
        if place is None:
            return
        conversation_id = self.team(place).conversation(role)
        if conversation_id is not None:
            text, told = await asyncio.to_thread(self._briefing, place, role, task)
            self._runs().mark(TOLD, conversation_id, told)
            await self._send(conversation_id, text)

    async def _start_agent(
        self,
        place_id: str,
        role: str,
        parent_conversation: str | None,
        task: str,
        checkout: str | None = None,
    ) -> bool:
        """Start one agent of a team from ``parent_conversation``.

        Args:
            checkout: The team's checkout when it has one already and the
                agent is to work on in it.

        Returns:
            Whether it runs. A failure is recorded on the team.
        """
        store = self.org
        place = store.place(place_id)
        conversation_id = self.team(place).conversation(role) if place else None
        if place is None or conversation_id is None:
            return False
        launcher = self._require_launcher()
        try:
            started = await launcher.started(UUID(conversation_id))
            if started is None:
                started = await self._start_new(
                    place, role, conversation_id, parent_conversation, task, checkout
                )
                if self._paused is not None:
                    self._note_running(conversation_id, place_id, True)
                    await launcher.pause(UUID(conversation_id))
        except _Held:
            raise
        except Exception as error:
            logger.exception("Could not start the %s of team %s", role, place_id)
            store.set_agent(place_id, role, status=AGENT_LOST)
            self.tell_of_failure(
                place_id, role, f"It could not be started: {error}"[:_REASON_KEPT]
            )
            return False
        store.set_agent(
            place_id,
            role,
            kind=started.kind,
            acp_server=started.acp_server,
            model=started.model,
        )
        if role == self.team(place).writer:
            store.update_place(
                place_id,
                worktree_path=started.workspace,
                isolation=started.isolation,
                branch=await asyncio.to_thread(_branch_of, started.workspace),
                base_commit=place.base_commit
                or await asyncio.to_thread(_base_of, started.workspace),
                boundary_enforcement=self._enforcement(started.kind),
            )
        return True

    async def _start_new(
        self,
        place: SessionTeam,
        role: str,
        conversation_id: str,
        parent_conversation: str | None,
        task: str,
        checkout: str | None = None,
    ) -> StartedAgent:
        """Start the conversation of an agent that has none yet.

        Raises:
            NotFound: There is nobody to derive the agent from.
            Conflict: A step of the director is replacing or removing the team.
            _Held: The org was paused while the team waited for a run.
        """
        place_id = place.id
        launcher = self._require_launcher()
        if any(place_id in teams for teams, _ in self._settling.values()):
            # The step fixed which conversations it stops before this one.
            raise Conflict(
                f"Team {place.title!r} is being replaced or removed, so its "
                f"{role} was not started"
            )
        text, told = await asyncio.to_thread(self._briefing, place, role, task)
        tool_params = await asyncio.to_thread(self._confine, place_id, role)
        if parent_conversation is None:
            raise NotFound(f"Team {place_id} has nobody to derive its {role} from")
        parent_id = UUID(parent_conversation)
        # A team given a profile of its own starts its IC from it; its
        # Manager, like every Manager, is derived from the IC.
        works = role == self.team(place).writer
        profile = place.agent_profile if works else None
        kind = await (
            launcher.profile_kind(profile)
            if profile
            else launcher.agent_kind(parent_id)
        )
        isolation = place.isolation if works else "shared"
        leads = is_division_lead(place)
        if kind == "acp" and works and self._bwrap is None:
            # Nothing holds its process to the team's directory, so its
            # work is kept apart and checked when it is merged.
            isolation = "worktree"
        tools = self.tools_for(place, role)
        adapt = await asyncio.to_thread(self._adapter, conversation_id, tools)
        # A session starts from where the repository's checkout is.
        base = (
            await asyncio.to_thread(head_of, self._repo_dir)
            if works
            and not checkout
            and isolation == "worktree"
            and self.org.opens_session(place)
            else None
        )
        title = _agent_title(
            place.title,
            self.team(place).title(role),
            self.org.session_title(place),
            division_lead=leads,
        )

        async def start() -> StartedAgent:
            if profile:
                return await launcher.start_from_profile(
                    profile,
                    parent_id,
                    UUID(conversation_id),
                    [Tool(name=name) for name in tools],
                    title=title,
                    task=text,
                    isolation=isolation,
                    tool_params=tool_params,
                    adapt_agent=adapt,
                    checkout=checkout,
                    base=base,
                )
            return await launcher.start_derived(
                parent_id,
                UUID(conversation_id),
                title=title,
                task=text,
                isolation=isolation,
                tool_params=tool_params,
                adapt_agent=adapt,
                checkout=checkout,
                base=base,
            )

        started = await self._when_there_is_room(place_id, start)
        self._runs().mark(TOLD, conversation_id, told)
        return started

    async def _started_ic(self, place: SessionTeam | None) -> str | None:
        """The IC of ``node`` or, while that team has not started, of the
        nearest team above that has: the conversation a new agent is
        derived from."""
        store, launcher = self.org, self._require_launcher()
        while place is not None:
            ic = self.writer_conversation(place)
            # Never a Division Lead IC, whose role of reviewer the new agent
            # would inherit.
            if (
                ic is not None
                and not is_division_lead(place)
                and await launcher.started(UUID(ic)) is not None
            ):
                return ic
            place = store.place(place.parent_id) if place.parent_id else None
        return None

    async def _launch_member(self, place_id: str, task: str, member: str) -> None:
        place = self.org.place(place_id)
        if place is not None:
            # A Division Lead Manager is derived from the session's IC,
            # never from the Division Lead IC, whose role it would inherit.
            derived_from = await self._started_ic(
                self.org.place(self._derived_from(place) or "")
                if is_division_lead(place)
                else place
            )
            # A member that reviews a session's work without being the one
            # that works there still does it in the place's own checkout.
            reviews_here = (
                is_division_lead(place) and member == self.team(place).reviewer
            )
            await self._start_agent(
                place_id,
                member,
                derived_from,
                task,
                place.worktree_path if reviews_here else None,
            )

    async def _do_brief(self, intent: Intent) -> None:
        brief = intent.of(Brief)
        place = self.org.place(intent.node_id)
        if place and brief.role == self.team(place).writer:
            self._runs().mark(TASK, intent.node_id, brief.task)
        await self._brief(intent.node_id, brief.role, brief.task)

    async def _do_launch_member(self, intent: Intent) -> None:
        launch = intent.of(LaunchMember)
        await self._launch_member(intent.node_id, launch.task, launch.role)

    async def _do_start_team(self, intent: Intent) -> None:
        start = intent.of(StartTeam)
        place_id, task = intent.node_id, start.task
        above = self.org.place(start.parent or "")
        parent_ic = await self._started_ic(above) or await self._template()
        checkout = start.checkout
        self._runs().mark(TASK, place_id, task)
        self.progressed(place_id, by_above=True)
        started = self.org.place(place_id)
        left: list[str] = []
        # A session starts beside the team's own work: what is uncommitted
        # in the team's directory is not the session's.
        opens = started is not None and (
            self.org.opens_session(started) or is_division_lead(started)
        )
        if started is not None:
            # A team of the same name before it left nothing to count on.
            self._work_accepted(started)
            if above is not None and not opens:
                with suppress(GitError, OSError):
                    left = await asyncio.to_thread(
                        uncommitted,
                        Path(above.worktree_path),
                        started.boundary.path,
                    )
        if started is None:
            return
        await self._start_agent(
            place_id,
            self.team(started).writer,
            parent_ic,
            f"{task}\n\n{left_for_team(left, names=self.names_at(place_id))}"
            if left
            else task,
            checkout,
        )
        if above is not None and not opens:
            await asyncio.to_thread(self._pass_down, above, place_id)
        if opens and not is_division_lead(started):
            self._checkout_for_session(place_id)
        purpose = start.purpose or task
        self._enqueue(lambda: self._summarize(place_id, purpose, first=True), place_id)

    def _checkout_for_session(self, place_id: str) -> None:
        """Give the teams of a session that have not started the checkout
        of the team the session was opened at, as a team that is set up
        has the checkout of the team above."""
        store = self.org
        top = store.place(place_id)
        if top is None or not top.agents:
            return
        teams = store.below(place_id)
        for team in teams:
            if not team.agents:
                store.update_place(
                    team.id, worktree_path=top.worktree_path, branch=top.branch
                )
                teams.extend(store.below(team.id))

    def _pass_down(self, above: SessionTeam, place_id: str) -> None:
        """Close a started team's directory to the IC above, and bring what
        that IC left uncommitted there into the team's own checkout."""
        team = self.org.place(place_id)
        if team is None or not team.worktree_path:
            return
        with suppress(GitError, OSError):
            # The shell the IC above starts next cannot write there.
            self._confine(above.id, self.team(above).writer)
        source, target = Path(above.worktree_path), Path(team.worktree_path)
        if checkout_root(str(source)) == checkout_root(str(target)):
            # In one checkout the team finds the files where they are.
            return
        try:
            bring_over(source, target, team.boundary.path, f"{LEFT_REFS}/{team.id}")
        except (GitError, OSError):
            logger.exception("Could not pass what was left to team %s", team.id)

    def note_hand_over(
        self, place_id: str, *, team: str | None = None, spec: bool = False
    ) -> None:
        """Keep what a team's hand-over conversation did, for its IC to be
        told when it ends. Safe from any thread."""
        run = self._runs()
        kept = run.marked(HANDING).get(place_id)
        if kept is None:
            return
        done = json.loads(kept)
        if team is not None and team not in done["given"]:
            done["given"].append(team)
        done["spec"] = done["spec"] or spec
        run.mark(HANDING, place_id, json.dumps(done))
