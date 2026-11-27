"""What the director's timer watches of a running org: teams that show no
progress, and what the sessions' agents use."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from openhands.agent_server.director.environment.checkouts import Checkouts
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import (
    Session,
    SessionTeam,
    Stopped,
    Usage,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    is_division_lead,
    utc_now,
)
from openhands.agent_server.director.prompts import (
    STALL_QUESTION,
    check_in_notice,
    in_words,
    stall_notice,
    unstarted_notice,
)
from openhands.agent_server.director.run_state import PROGRESS, USE, RunState
from openhands.agent_server.director.service.model_call_service import (
    ModelCallService,
)
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import (
    OrgStateService,
    progress_of,
)
from openhands.agent_server.director.service.review_service import ReviewService
from openhands.agent_server.director.service.session_merge_service import (
    SessionMergeService,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from openhands.agent_server.director.service.team_messaging_service import (
    TeamMessagingService,
)
from openhands.agent_server.director.service.team_type_service import TeamTypeService
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.agent_server.director.team import Team


# The id of the escalation that asks the human what a team that shows no
# progress is to do, when no IC stands above it.
STALL_WAITS = "stalled"
# The escalation that asks the human whether a session that reached its
# token limit goes on.
LIMIT_WAITS = "limit-reached"
# The lane the director's timer does its work in.
STALL_LANE = "stalls"


@dataclass
class OrgWatchService:
    """Watches the running org on the director's one timer.

    Each look does two things. What the agents of each session used since
    the last look is added up (`count_use`), and a session that reached
    its limit is put to rest and the human asked (`LIMIT_WAITS`); the
    answer lets it go on (`go_on_past_limit`). And the teams that owe work
    and show no progress are found (`check_stalls`): the IC above each is
    told once for every stretch of time the org allows, and from time to
    time how each team below it stands.

    What it has seen is kept in the run state. `watch` is the timer;
    whoever starts the director runs it as a task.
    """

    org: Callable[[], OrgStore]
    """The org's tree, as it is when asked."""
    has_org: Callable[[], bool]
    """Whether an org is loaded: none is loaded only to look at it."""
    checkouts: Checkouts
    """Where each team's work is, for when it last changed."""
    launcher: Callable[[], TeamLauncher]
    """What runs the agents and knows what each has used."""
    poll_seconds: float
    """How often the timer looks."""
    org_state: OrgStateService
    org_settings: OrgSettingsService
    org_save: OrgSaveService
    team_types: TeamTypeService
    team_history: TeamHistoryService
    team_messaging: TeamMessagingService
    reviews: ReviewService
    session_merges: SessionMergeService
    model_calls: ModelCallService
    """Asks the IC of a team that shows no progress what it is doing."""
    work_queue: WorkQueueService

    def _writer_conversation(self, team: SessionTeam) -> str | None:
        of_team = Team(self.org(), team)
        return of_team.conversation(of_team.writer)

    def owes_work(self, team: SessionTeam) -> bool:
        """Whether a team has yet to submit, or to submit again, and nothing
        holds it."""
        return (
            team.parent_id is not None
            and team.status != "blocked"
            and not team.waiting_to_start
            and (team.review is None or team.review.state == "changes_requested")
            and self._writer_conversation(team) is not None
        )

    def may_stall(self, team: SessionTeam) -> bool:
        """Whether a team owes work nobody waits to give it: it has not
        submitted, or its work was sent back, and nothing holds it. A team
        that waits for one below it that owes work is not such a team: the
        wait is counted where the work is."""
        return (
            self.owes_work(team)
            and not any(above.stopped for above in self.org().above(team.id))
            and not any(
                below.stopped is None and self.owes_work(below)
                for below in self.org().below(team.id)
            )
        )

    def _waits_for_a_task(self, team: SessionTeam) -> bool:
        """Whether a team was set up and never given a task, with nothing
        that holds the team above from giving it one."""
        return (
            team.parent_id is not None
            # A team the org already had is there for the session's IC to
            # use or not; one the session set up itself is owed a task.
            and self.org().team(team.team_id) is None
            and not is_division_lead(team)
            and not team.agents
            and not team.waiting_to_start
            and team.stopped is None
            and not any(above.stopped for above in self.org().above(team.id))
        )

    def _may_be_checked_in_on(self, team: SessionTeam) -> bool:
        """Whether the IC above is told from time to time how a team stands:
        it owes work, is not stopped, and an IC stands above it."""
        store = self.org()
        above = store.place(team.parent_id) if team.parent_id else None
        return (
            above is not None
            and not is_division_lead(above)
            and self._writer_conversation(above) is not None
            and self.owes_work(team)
            and team.stopped is None
            and not any(each.stopped for each in store.above(team.id))
        )

    async def watch(self) -> None:
        """The director's one timer: has what the sessions use counted
        and the teams looked at, for as long as it is not cancelled."""
        while True:
            await asyncio.sleep(self.poll_seconds)
            if self.work_queue.is_open:
                self.work_queue.enqueue(self.count_use, STALL_LANE)
                self.work_queue.enqueue(self.check_stalls, STALL_LANE)

    async def check_stalls(self) -> None:
        """Tell the IC above of each team that has shown no progress for
        longer than the org allows, once for every such stretch of time, and
        of each team it has not turned to or been told of for the time
        between two check-ins. A team is never told of twice at once: being
        told that it shows no progress is a check-in too."""
        limit = self.org_settings.settings.soft_check_in_minutes
        check_in = self.org_settings.settings.hard_check_in_minutes
        if (
            (not limit and not check_in)
            or not self.has_org()
            or self.org_state.paused is not None
        ):
            return
        store = self.org()
        if limit:
            await asyncio.to_thread(self.session_merges.check_managers)
        run = self.org_state.run_state()
        marks = run.marked(PROGRESS)
        teams = {team.id: team for team in store.places()}
        for gone in marks.keys() - teams.keys():
            run.unmark(PROGRESS, gone)
        # What each IC is told of the teams below it, in one message.
        check_ins: dict[str, list[str]] = {}
        for team in teams.values():
            if team.parent_id is None:
                continue
            now = self.org_state.clock()
            seen = progress_of(marks.get(team.id))
            head = await asyncio.to_thread(
                self.checkouts.last_change, Path(team.worktree_path), team.boundary.path
            )
            if "at" not in seen:
                # Seen for the first time.
                first = {
                    "at": now.isoformat(),
                    "head": head,
                    "checked": now.isoformat(),
                }
                run.mark(PROGRESS, team.id, json.dumps(first))
                continue
            if head != seen.get("head"):
                # It has committed since.
                seen = {
                    "at": now.isoformat(),
                    "head": head,
                    "checked": seen.get("checked") or now.isoformat(),
                }
                run.mark(PROGRESS, team.id, json.dumps(seen))
            elif limit and self._tell_if_stalled(team, seen, now, limit, run):
                continue
            if not check_in or not self._may_be_checked_in_on(team):
                continue
            checked = datetime.fromisoformat(seen.get("checked") or seen["at"])
            if now - checked < timedelta(minutes=max(check_in, limit)):
                continue
            # Told of now, so it is not told of as showing no progress
            # before as long again.
            told = {"checked": now.isoformat(), "told": now.isoformat()}
            run.mark(PROGRESS, team.id, json.dumps({**seen, **told}))
            ic = self._writer_conversation(team)
            check_ins.setdefault(team.parent_id, []).append(
                check_in_notice(
                    team,
                    int((now - checked).total_seconds() // 60),
                    in_a_turn=ic in self.org_state.running,
                    head=head,
                    again=max(check_in, limit),
                    names=Team(self.org(), team).names,
                )
            )
        for above_id, texts in check_ins.items():
            self.team_messaging.tell_writer(above_id, "\n\n".join(texts))

    def _tell_if_stalled(
        self,
        team: SessionTeam,
        seen: dict[str, Any],
        now: datetime,
        limit: int,
        run: RunState,
    ) -> bool:
        """Have the IC above told of a team that has shown no progress for
        ``limit`` minutes since it last did, or since that IC was last told.

        Returns:
            Whether it is told of now.
        """
        unstarted = self._waits_for_a_task(team)
        if not unstarted and not self.may_stall(team):
            return False
        since = datetime.fromisoformat(seen["at"])
        told = seen.get("told")
        if unstarted and told is not None:
            # Said once: a team may be left without a task on purpose.
            return False
        if now - datetime.fromisoformat(told or seen["at"]) < timedelta(minutes=limit):
            return False
        again = {"told": now.isoformat(), "checked": now.isoformat()}
        run.mark(PROGRESS, team.id, json.dumps({**seen, **again}))
        if unstarted and team.parent_id is not None:
            self.team_messaging.tell_writer(
                team.parent_id,
                unstarted_notice(
                    team,
                    int((now - since).total_seconds() // 60),
                    names=Team(self.org(), team).names,
                ),
            )
            return True
        if told is None:
            # The IC above has this much time to act before it shows
            # no progress itself.
            self.team_messaging.progressed(team.parent_id)
        minutes = int((now - since).total_seconds() // 60)
        self.work_queue.enqueue(
            lambda team_id=team.id, minutes=minutes: self._tell_of_stall(
                team_id, minutes
            ),
            f"review:{team.id}",
        )
        return True

    async def _tell_of_stall(self, place_id: str, minutes: int) -> None:
        store = self.org()
        team = store.place(place_id)
        above = store.place(team.parent_id) if team and team.parent_id else None
        if team is None or above is None or self.org_state.paused is not None:
            return
        names = Team(store, team).names
        if not self.may_stall(team):
            return
        ic = self._writer_conversation(team)
        in_a_turn = ic in self.org_state.running
        account = None
        if ic is not None and not in_a_turn:
            account = await self.model_calls.ask_agent(
                ic, in_words(STALL_QUESTION, names)
            )
        self.team_history.record_event(above.id, "stalled", team, f"{minutes} min")
        # Its reviewer is woken now anyway, and no longer waits for it.
        self.reviews.release_held(above.id)
        said = (
            f"Its {names.writer} is in a turn right now, so it was not asked what "
            "it is doing."
            if in_a_turn
            else f"Asked just now, its {names.writer} says:\n{account.strip()}"
            if account
            else f"Asked just now what it is doing, its {names.writer} gave no answer."
        )
        self.team_messaging.tell_above(
            team,
            stall_notice(
                team, minutes, in_a_turn=in_a_turn, account=account, names=names
            ),
            waits=STALL_WAITS,
            title=f"{team.title!r} has shown no progress for {minutes} min",
            detail=(
                f"No commit in {team.boundary.path or 'the repository'}, "
                f"nothing submitted, no escalation.\n{said}\nAnswer this to "
                f"tell its {names.writer} what to do: your answer is sent to it."
            ),
        )

    async def count_use(self) -> None:
        """Add up what each session's agents have used since it was last
        looked at, and put a session that reached its limit to rest."""
        if not self.has_org():
            return
        store = self.org()
        launcher = self.launcher()
        run = self.org_state.run_state()
        marks = run.marked(USE)
        for session in store.sessions():
            conversations = {
                conversation_id
                for team in session.teams
                for conversation_id in self.org().conversations_below(team)
            }
            now = await launcher.usage([UUID(each) for each in sorted(conversations)])
            last: dict[str, list[float]] = json.loads(marks.get(session.id) or "{}")
            added = Usage()
            for conversation_id, used in now.items():
                before = last.get(str(conversation_id), [0, 0, 0.0, 0])
                # A count that went down began again, with a server.
                added = added.plus(
                    Usage(
                        input_tokens=max(0, used.input_tokens - int(before[0])),
                        output_tokens=max(0, used.output_tokens - int(before[1])),
                        cost=max(0.0, used.cost - before[2]),
                        unpriced_tokens=max(0, used.unpriced_tokens - int(before[3])),
                    )
                )
                last[str(conversation_id)] = [
                    used.input_tokens,
                    used.output_tokens,
                    used.cost,
                    used.unpriced_tokens,
                ]
            run.mark(USE, session.id, json.dumps(last))
            # As it is now: the human may have let it go on while this waited.
            session = next(
                (each for each in store.sessions() if each.id == session.id), None
            )
            if session is None:
                continue
            limit = (
                session.cost_allowance or self.org_settings.settings.session_cost_limit
            )
            kept = store.set_session_use(session.id, session.usage.plus(added), limit)
            if kept is not None:
                self._hold_at_limit(kept)

    def _hold_at_limit(self, session: Session) -> None:
        """Put the agents of a session that reached its limit to rest and
        ask the human whether it goes on."""
        store = self.org()
        top = store.instance(session.team_id, session.id)
        limit = session.cost_allowance
        if (
            not limit
            or session.usage.cost < limit
            or top is None
            or top.stopped is not None
            or self.org_state.paused is not None
        ):
            return
        running = [
            conversation_id
            for conversation_id in self.org().conversations_below(top)
            if conversation_id in self.org_state.running
        ]
        store.set_stopped(
            top.id,
            Stopped(
                at=utc_now(),
                reason=f"The session reached its limit of ${limit:,.2f}.",
                running=sorted(running),
            ),
        )
        self.work_queue.enqueue(lambda: self.team_messaging.stop(top.id), top.id)
        used = session.usage
        if all(e.id != LIMIT_WAITS for e in top.escalations):
            store.raise_escalation(
                top.id,
                LIMIT_WAITS,
                kind="critical_issue",
                title=f"{session.title!r} reached its limit of ${limit:,.2f}",
                detail=(
                    f"Its agents have used ${used.cost:,.2f} "
                    f"({used.input_tokens:,} tokens in, "
                    f"{used.output_tokens:,} out) and are at rest.\nAnswer "
                    "this to let it go on for as much again; your answer is "
                    f"sent to its {self.team_types.team(top).names.writer}. Close the "
                    "session to end it."
                ),
                at=utc_now(),
            )
        self.org_save.save_soon()

    def go_on_past_limit(self, place_id: str) -> None:
        """Let a session that was put to rest at its limit go on for as
        much again, or without a limit when the org has none now."""
        store = self.org()
        top = store.place(place_id)
        session = store.session_of(top) if top else None
        if top is None or session is None:
            return
        limit = self.org_settings.settings.session_cost_limit
        store.set_session_use(
            session.id, session.usage, session.usage.cost + limit if limit else 0.0
        )
        stopped = top.stopped
        if stopped is not None:
            store.set_stopped(top.id, None)
            self.work_queue.enqueue(
                lambda: self.team_messaging.go_on(
                    top.id, stopped.running, own_too=True
                ),
                top.id,
            )
        self.org_save.save_soon()
