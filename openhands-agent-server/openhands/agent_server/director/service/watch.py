"""Teams that show no progress, and teams that are stopped."""

import asyncio
import json
from functools import cached_property
from uuid import UUID

from openhands.agent_server.director.models import (
    Escalation,
    Session,
    SessionTeam,
    Stopped,
    Usage,
)
from openhands.agent_server.director.org_store import (
    utc_now,
)
from openhands.agent_server.director.run_state import (
    USE,
)
from openhands.agent_server.director.service.reviews import Reviews
from openhands.agent_server.director.service.stall_watch_service import (
    StallWatchService,
)
from openhands.agent_server.director.service.state import (
    LIMIT_WAITS,
    STALL_LANE,
)
from openhands.sdk.errors import (
    Conflict,
    NotFound,
)


class Watch(Reviews):
    """Stops teams, counts what sessions use, and carries what an agent
    cannot settle to the agent that can. What looks for teams without
    progress is a service it holds."""

    @cached_property
    def stall_watch(self) -> StallWatchService:
        """Finds teams that show no progress. Built here, where everything
        it is given is defined."""
        return StallWatchService(
            org=lambda: self.org,
            has_org=lambda: self._store is not None,
            settings=lambda: self.settings,
            run_state=self._runs,
            clock=self._clock,
            paused=lambda: self._paused is not None,
            running=lambda: self._running,
            model_calls=self.model_calls,
            check_managers=self._check_managers,
            work_queue=self.work_queue,
            progressed=self.progressed,
            release_reviews=self._release_reviews,
            record_event=self.record_event,
            tell_writer=self.tell_writer,
            tell_above=self.tell_above,
        )

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
        holder = self.org.place(holder_id)
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
        store = self.org
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
            self.work_queue.enqueue(lambda: self._stop(place_id), place_id)
        self.org_save.save_soon()
        self.record_event(team.parent_id, "stopped", team, reason)
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

    async def _watch_for_stalls(self) -> None:
        while True:
            await asyncio.sleep(self._stall_poll)
            if self._loop is not None:
                self.work_queue.enqueue(self._count_use, STALL_LANE)
                self.work_queue.enqueue(self.stall_watch.check_stalls, STALL_LANE)

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
        store = self.org
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
        self.work_queue.enqueue(lambda: self._stop(top.id), top.id)
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
        store = self.org
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
            self.work_queue.enqueue(
                lambda: self._go_on(top.id, stopped.running, own_too=True), top.id
            )
        self.org_save.save_soon()
