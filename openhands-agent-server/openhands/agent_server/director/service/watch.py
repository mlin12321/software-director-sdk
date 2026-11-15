"""Teams that show no progress, and teams that are stopped."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from openhands.agent_server.director.git import (
    last_commit_in,
)
from openhands.agent_server.director.models import (
    OrgNode,
    Stopped,
)
from openhands.agent_server.director.prompts import (
    STALL_QUESTION,
    stall_notice,
    unstarted_notice,
)
from openhands.agent_server.director.run_state import (
    PROGRESS,
)
from openhands.agent_server.director.service.reviews import Reviews
from openhands.agent_server.director.service.state import (
    STALL_LANE,
    _now,
    _progress_of,
    agent_conversation,
)
from openhands.sdk.errors import (
    Conflict,
    NotFound,
)


class Watch(Reviews):
    """Looks for teams without progress and stops teams."""

    def stop_team(self, node_id: str, reason: str) -> None:
        """Stop a team for the IC of the team above: its agents and those
        of every team below it end the step they are in and are sent
        nothing until that IC writes to the team. Safe from any thread.

        Raises:
            NotFound: The team does not exist.
            Conflict: It is stopped already.
        """
        store = self.store
        team = store.get(node_id)
        if team is None:
            raise NotFound(f"Node {node_id} does not exist")
        if team.stopped is not None:
            raise Conflict(f"Team {node_id} is stopped already")
        running = [
            conversation_id
            for conversation_id in self._conversations_below(team)
            if conversation_id in self._running
        ]
        store.set_stopped(
            node_id, Stopped(at=_now(), reason=reason, running=sorted(running))
        )
        if self._loop is not None:
            self._enqueue(lambda: self._stop(node_id), node_id)
        self._save_soon()
        self.record_event(team.parent_id, "stopped", team, reason)
        # A stopped team is waited for no longer by the reviews beside it.
        self.release_reviews(team.parent_id)
        if team.parent_id is not None:
            self.tell_manager(
                team.parent_id,
                f"Your IC stopped the team {team.title!r} (node {team.id}): "
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
                self._enqueue(self._check_stalls, STALL_LANE)

    def _may_stall(self, team: OrgNode) -> bool:
        """Whether a team owes work nobody waits to give it: it has not
        submitted, or its work was sent back, and nothing holds it."""
        return (
            team.parent_id is not None
            and team.status != "blocked"
            and not team.waiting_to_start
            and (team.review is None or team.review.state == "changes_requested")
            and agent_conversation(team, "ic") is not None
            and not any(above.stopped for above in self.store.ancestors(team.id))
        )

    def _waits_for_a_task(self, team: OrgNode) -> bool:
        """Whether a team was set up and never given a task, with nothing
        that holds the team above from giving it one."""
        return (
            team.parent_id is not None
            and not team.agents
            and not team.waiting_to_start
            and team.stopped is None
            and not any(above.stopped for above in self.store.ancestors(team.id))
        )

    async def _check_stalls(self) -> None:
        """Tell the IC above of each team that has shown no progress for
        longer than the org allows, once for every such stretch of time."""
        limit = self.settings.stall_check_minutes
        store = self._store
        if not limit or store is None or self._paused is not None or self._fixture:
            return
        run = self._runs()
        marks = run.marked(PROGRESS)
        teams = {team.id: team for team in store.all()}
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

    async def _tell_of_stall(self, node_id: str, minutes: int) -> None:
        store = self.store
        team = store.get(node_id)
        above = store.get(team.parent_id) if team and team.parent_id else None
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
        self.tell_ic(
            above.id,
            stall_notice(team, minutes, in_a_turn=in_a_turn, account=account),
        )
