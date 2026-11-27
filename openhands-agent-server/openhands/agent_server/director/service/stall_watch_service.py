"""Teams that show no progress: finding them and telling the IC above."""

import asyncio
import json
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from openhands.agent_server.director.environment.checkouts import Checkouts
from openhands.agent_server.director.models import DirectorSettings, SessionTeam
from openhands.agent_server.director.org_store import OrgStore, is_division_lead
from openhands.agent_server.director.prompts import (
    STALL_QUESTION,
    check_in_notice,
    in_words,
    stall_notice,
    unstarted_notice,
)
from openhands.agent_server.director.run_state import PROGRESS, RunState
from openhands.agent_server.director.service.model_call_service import (
    ModelCallService,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.agent_server.director.team import Team


# The id of the escalation that asks the human what a team that shows no
# progress is to do, when no IC stands above it.
STALL_WAITS = "stalled"


def progress_of(kept: str | None) -> dict[str, Any]:
    """What is known of a team's progress: when it last got anywhere
    (``at``), the last commit in its directory then (``head``), when the IC
    above was last told that it shows none (``told``), and when that IC last
    turned to the team or was told of it (``checked``)."""
    try:
        seen = json.loads(kept) if kept else {}
    except ValueError:
        return {}
    return seen if isinstance(seen, dict) else {}


@dataclass
class StallWatchService:
    """Finds the teams that owe work and show no progress, and has the IC
    above each told: once for every stretch of time the org allows, and
    from time to time how each team below it stands.

    It keeps what it has seen in the run state and sends nothing itself:
    what is said goes through the functions it is given. It runs no timer;
    whoever holds it has ``check_stalls`` run.
    """

    org: Callable[[], OrgStore]
    """The org's tree, as it is when asked."""
    checkouts: Checkouts
    """Where each team's work is and how it moves."""
    has_org: Callable[[], bool]
    """Whether an org is loaded: none is loaded only to look at it."""
    settings: Callable[[], DirectorSettings]
    """The org's settings, for the minutes a team may show no progress."""
    run_state: Callable[[], RunState]
    """Where what was seen of each team is kept."""
    clock: Callable[[], datetime]
    """The time now."""
    paused: Callable[[], bool]
    """Whether the org is paused, when nobody is told anything."""
    running: Callable[[], Collection[str]]
    """The conversations that are in a turn."""
    model_calls: ModelCallService
    """Asks the IC of a team that shows no progress what it is doing."""
    check_managers: Callable[[], None]
    """Looks at the Managers that were asked to settle teams and did not."""
    work_queue: WorkQueueService
    """What telling of a stall is carried out with."""
    progressed: Callable[[str | None], None]
    """Starts a team's time without progress again."""
    release_reviews: Callable[[str | None], None]
    """Sends a team's IC the reviews that wait for it."""
    record_event: Callable[..., None]
    """Keeps an event in the record of a team (``Records.record_event``)."""
    tell_writer: Callable[[str, str], object]
    """Sends a message to the IC of the team with this id."""
    tell_above: Callable[..., None]
    """Tells the IC above a team, or the human when none stands above it
    (``Teams.tell_above``)."""

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

    async def check_stalls(self) -> None:
        """Tell the IC above of each team that has shown no progress for
        longer than the org allows, once for every such stretch of time, and
        of each team it has not turned to or been told of for the time
        between two check-ins. A team is never told of twice at once: being
        told that it shows no progress is a check-in too."""
        limit = self.settings().soft_check_in_minutes
        check_in = self.settings().hard_check_in_minutes
        if (not limit and not check_in) or not self.has_org() or self.paused():
            return
        store = self.org()
        if limit:
            await asyncio.to_thread(self.check_managers)
        run = self.run_state()
        marks = run.marked(PROGRESS)
        teams = {team.id: team for team in store.places()}
        for gone in marks.keys() - teams.keys():
            run.unmark(PROGRESS, gone)
        # What each IC is told of the teams below it, in one message.
        check_ins: dict[str, list[str]] = {}
        for team in teams.values():
            if team.parent_id is None:
                continue
            now = self.clock()
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
                    in_a_turn=ic in self.running(),
                    head=head,
                    again=max(check_in, limit),
                    names=Team(self.org(), team).names,
                )
            )
        for above_id, texts in check_ins.items():
            self.tell_writer(above_id, "\n\n".join(texts))

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
            self.tell_writer(
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
            self.progressed(team.parent_id)
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
        if team is None or above is None or self.paused():
            return
        names = Team(store, team).names
        if not self.may_stall(team):
            return
        ic = self._writer_conversation(team)
        in_a_turn = ic in self.running()
        account = None
        if ic is not None and not in_a_turn:
            account = await self.model_calls.ask_agent(
                ic, in_words(STALL_QUESTION, names)
            )
        self.record_event(above.id, "stalled", team, f"{minutes} min")
        # Its reviewer is woken now anyway, and no longer waits for it.
        self.release_reviews(above.id)
        said = (
            f"Its {names.writer} is in a turn right now, so it was not asked what "
            "it is doing."
            if in_a_turn
            else f"Asked just now, its {names.writer} says:\n{account.strip()}"
            if account
            else f"Asked just now what it is doing, its {names.writer} gave no answer."
        )
        self.tell_above(
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
