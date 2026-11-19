"""Teams that show no progress, and teams that are stopped."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

from openhands.agent_server.director.git import (
    last_commit_in,
)
from openhands.agent_server.director.models import (
    Escalation,
    Session,
    SessionTeam,
    Stopped,
    Usage,
)
from openhands.agent_server.director.org_store import (
    is_division_lead,
    utc_now,
)
from openhands.agent_server.director.prompts import (
    STALL_QUESTION,
    stall_notice,
    unstarted_notice,
)
from openhands.agent_server.director.run_state import (
    PROGRESS,
    USE,
)
from openhands.agent_server.director.service.reviews import Reviews
from openhands.agent_server.director.service.state import (
    LIMIT_WAITS,
    ROLE_NAMES,
    STALL_LANE,
    STALL_WAITS,
    _progress_of,
    agent_conversation,
)
from openhands.sdk.errors import (
    Conflict,
    NotFound,
)


class Watch(Reviews):
    """Looks for teams without progress and stops teams, and carries what
    an agent cannot settle to the agent that can."""

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
        holder = self.store.place(holder_id)
        if holder is None:
            return
        history = "".join(f"\nPassed up by {entry}" for entry in passed_up)
        if escalation.raised_by == "ic":
            # It is woken now anyway, so it gets what waits for its review too.
            self.release_reviews(holder_id, now=True)
            self.count_ic_escalation(holder_id, through.id)
        text = (
            f"The {ROLE_NAMES[escalation.raised_by]} of team {raiser.title!r} "
            f"(id {raiser.id}) escalates {escalation.title!r} (escalation "
            f"{escalation.id}):\n{escalation.detail or '(no detail)'}{history}\n"
            "Settle it with resolve_escalation if you can; pass it up only with "
            "the reason you cannot."
        )
        if escalation.raised_by == "manager":
            self.tell_manager(holder_id, text)
        else:
            self.tell_ic(holder_id, text)

    def stop_team(self, place_id: str, reason: str) -> None:
        """Stop a team for the IC of the team above: its agents and those
        of every team below it end the step they are in and are sent
        nothing until that IC writes to the team. Safe from any thread.

        Raises:
            NotFound: The team does not exist.
            Conflict: It is stopped already.
        """
        store = self.store
        team = store.place(place_id)
        if team is None:
            raise NotFound(f"Team {place_id} does not exist")
        if team.stopped is not None:
            raise Conflict(f"Team {place_id} is stopped already")
        running = [
            conversation_id
            for conversation_id in self._conversations_below(team)
            if conversation_id in self._running
        ]
        store.set_stopped(
            place_id, Stopped(at=utc_now(), reason=reason, running=sorted(running))
        )
        if self._loop is not None:
            self._enqueue(lambda: self._stop(place_id), place_id)
        self._save_soon()
        self.record_event(team.parent_id, "stopped", team, reason)
        # A stopped team is waited for no longer by the reviews beside it.
        self.release_reviews(team.parent_id)
        if team.parent_id is not None:
            self.tell_manager(
                team.parent_id,
                f"Your IC stopped the team {team.title!r} (id {team.id}): "
                f"{reason}\nIts agents are at rest and nothing is sent to "
                "them. It goes on when your IC writes to it with msg_team. "
                "Nothing is asked of you. If it should not go on at all, you "
                "may remove it with prune_team, or have your IC propose "
                "teams that take its place.",
            )

    async def _watch_for_stalls(self) -> None:
        while True:
            await asyncio.sleep(self._stall_poll)
            if self._loop is not None:
                self._enqueue(self._count_use, STALL_LANE)
                self._enqueue(self._check_stalls, STALL_LANE)

    async def _count_use(self) -> None:
        """Add up what each session's agents have used since it was last
        looked at, and put a session that reached its limit to rest."""
        store = self._store
        if store is None:
            return
        launcher = self._require_launcher()
        run = self._runs()
        marks = run.marked(USE)
        for session in store.sessions():
            conversations = {
                conversation_id
                for team in session.teams
                for conversation_id in self._conversations_below(team)
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
            limit = session.cost_allowance or self.settings.session_cost_limit
            kept = store.set_session_use(session.id, session.usage.plus(added), limit)
            if kept is not None:
                self._hold_at_limit(kept)

    def _hold_at_limit(self, session: Session) -> None:
        """Put the agents of a session that reached its limit to rest and
        ask the human whether it goes on."""
        store = self.store
        top = store.instance(session.team_id, session.id)
        limit = session.cost_allowance
        if (
            not limit
            or session.usage.cost < limit
            or top is None
            or top.stopped is not None
            or self._paused is not None
        ):
            return
        running = [
            conversation_id
            for conversation_id in self._conversations_below(top)
            if conversation_id in self._running
        ]
        store.set_stopped(
            top.id,
            Stopped(
                at=utc_now(),
                reason=f"The session reached its limit of ${limit:,.2f}.",
                running=sorted(running),
            ),
        )
        self._enqueue(lambda: self._stop(top.id), top.id)
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
                    "sent to its IC. Close the session to end it."
                ),
                at=utc_now(),
            )
        self._save_soon()

    def go_on_past_limit(self, place_id: str) -> None:
        """Let a session that was put to rest at its limit go on for as
        much again, or without a limit when the org has none now."""
        store = self.store
        top = store.place(place_id)
        session = store.session_of(top) if top else None
        if top is None or session is None:
            return
        limit = self.settings.session_cost_limit
        store.set_session_use(
            session.id, session.usage, session.usage.cost + limit if limit else 0.0
        )
        stopped = top.stopped
        if stopped is not None:
            store.set_stopped(top.id, None)
            self._enqueue(
                lambda: self._go_on(top.id, stopped.running, own_too=True), top.id
            )
        self._save_soon()

    def _may_stall(self, team: SessionTeam) -> bool:
        """Whether a team owes work nobody waits to give it: it has not
        submitted, or its work was sent back, and nothing holds it."""
        return (
            team.parent_id is not None
            and team.status != "blocked"
            and not team.waiting_to_start
            and (team.review is None or team.review.state == "changes_requested")
            and agent_conversation(team, "ic") is not None
            and not any(above.stopped for above in self.store.above(team.id))
        )

    def _waits_for_a_task(self, team: SessionTeam) -> bool:
        """Whether a team was set up and never given a task, with nothing
        that holds the team above from giving it one."""
        return (
            team.parent_id is not None
            # A team the org already had is there for the session's IC to
            # use or not; one the session set up itself is owed a task.
            and self.store.team(team.team_id) is None
            and not is_division_lead(team)
            and not team.agents
            and not team.waiting_to_start
            and team.stopped is None
            and not any(above.stopped for above in self.store.above(team.id))
        )

    async def _check_stalls(self) -> None:
        """Tell the IC above of each team that has shown no progress for
        longer than the org allows, once for every such stretch of time."""
        limit = self.settings.stall_check_minutes
        store = self._store
        if not limit or store is None or self._paused is not None:
            return
        await asyncio.to_thread(self._check_managers)
        run = self._runs()
        marks = run.marked(PROGRESS)
        teams = {team.id: team for team in store.places()}
        for gone in marks.keys() - teams.keys():
            run.unmark(PROGRESS, gone)
        for team in teams.values():
            if team.parent_id is None:
                continue
            now = self._clock()
            seen = _progress_of(marks.get(team.id))
            head = await asyncio.to_thread(
                last_commit_in, Path(team.worktree_path), team.boundary.path
            )
            if "at" not in seen or head != seen.get("head"):
                # Seen for the first time, or it has committed since.
                run.mark(
                    PROGRESS, team.id, json.dumps({"at": now.isoformat(), "head": head})
                )
                continue
            unstarted = self._waits_for_a_task(team)
            if not unstarted and not self._may_stall(team):
                continue
            since = datetime.fromisoformat(seen["at"])
            told = seen.get("told")
            if unstarted and told is not None:
                # Said once: a team may be left without a task on purpose.
                continue
            if now - datetime.fromisoformat(told or seen["at"]) < timedelta(
                minutes=limit
            ):
                continue
            run.mark(PROGRESS, team.id, json.dumps({**seen, "told": now.isoformat()}))
            if unstarted:
                self.tell_ic(
                    team.parent_id,
                    unstarted_notice(team, int((now - since).total_seconds() // 60)),
                )
                continue
            if told is None:
                # The IC above has this much time to act before it shows
                # no progress itself.
                self.progressed(team.parent_id)
            minutes = int((now - since).total_seconds() // 60)
            self._enqueue(
                lambda team_id=team.id, minutes=minutes: self._tell_of_stall(
                    team_id, minutes
                ),
                f"review:{team.id}",
            )

    async def _tell_of_stall(self, place_id: str, minutes: int) -> None:
        store = self.store
        team = store.place(place_id)
        above = store.place(team.parent_id) if team and team.parent_id else None
        if team is None or above is None or self._paused is not None:
            return
        if not self._may_stall(team):
            return
        ic = agent_conversation(team, "ic")
        in_a_turn = ic in self._running
        account = None
        if ic is not None and not in_a_turn:
            account = await self._ask_agent(ic, STALL_QUESTION)
        self.record_event(above.id, "stalled", team, f"{minutes} min")
        # Its reviewer is woken now anyway, and no longer waits for it.
        self._release_reviews(above.id, now=True)
        said = (
            "Its IC is in a turn right now, so it was not asked what it is doing."
            if in_a_turn
            else f"Asked just now, its IC says:\n{account.strip()}"
            if account
            else "Asked just now what it is doing, its IC gave no answer."
        )
        self.tell_above(
            team,
            stall_notice(team, minutes, in_a_turn=in_a_turn, account=account),
            waits=STALL_WAITS,
            title=f"{team.title!r} has shown no progress for {minutes} min",
            detail=(
                f"No commit in {team.boundary.path or 'the repository'}, "
                f"nothing submitted, no escalation.\n{said}\nAnswer this to "
                "tell its IC what to do: your answer is sent to it."
            ),
        )
