"""Teams that show no progress, and teams that are stopped."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from openhands.agent_server.director import routing
from openhands.agent_server.director.git import (
    last_commit_in,
)
from openhands.agent_server.director.git_store import team_key
from openhands.agent_server.director.models import (
    Escalation,
    OrgNode,
    Stopped,
)
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    reviews_only,
    session_id_of,
)
from openhands.agent_server.director.prompts import (
    NEED_ASKED,
    PASSED_UP,
    SETTLED,
    STALL_QUESTION,
    UNANSWERED,
    asked_by_team,
    many_requests_notice,
    requests_between,
    stall_notice,
    unstarted_notice,
)
from openhands.agent_server.director.run_state import (
    PROGRESS,
)
from openhands.agent_server.director.service.requests import takes_requests
from openhands.agent_server.director.service.reviews import Reviews
from openhands.agent_server.director.service.state import (
    ROLE_NAMES,
    STALL_LANE,
    _now,
    _progress_of,
    agent_conversation,
    team_of,
)
from openhands.sdk.errors import (
    Conflict,
    NotFound,
)


# How many requests one team may have open with another it asked directly.
OPEN_REQUESTS_TO_ONE_TEAM = 3


class Watch(Reviews):
    """Looks for teams without progress and stops teams, and carries what
    an agent cannot settle to the agent that can."""

    def hand_escalation(
        self,
        holder_id: str,
        raiser: OrgNode,
        through: OrgNode,
        escalation: Escalation,
        passed_up: list[str],
    ) -> None:
        """Tell the agent that now holds an escalation. Safe from any thread.

        Args:
            through: The team directly below the holder that it comes from:
                the one that raised it, or the one that passed it up.
        """
        holder = self.store.get(holder_id)
        if holder is None:
            return
        history = "".join(f"\nPassed up by {entry}" for entry in passed_up)
        if escalation.raised_by == "ic":
            # It is woken now anyway, so it gets what waits for its review too.
            self.release_reviews(holder_id, now=True)
            self.count_ic_escalation(holder_id, through.id)
        text = (
            f"The {ROLE_NAMES[escalation.raised_by]} of team {raiser.title!r} "
            f"(node {raiser.id}) escalates {escalation.title!r} (escalation "
            f"{escalation.id}):\n{escalation.detail or '(no detail)'}{history}\n"
            "Settle it with resolve_escalation if you can; pass it up only with "
            "the reason you cannot."
        )
        if escalation.raised_by == "manager":
            self.tell_manager(holder_id, text)
        else:
            self.tell_ic(holder_id, text)

    def ask_team(self, node_id: str, escalation_id: str, team_for: str) -> None:
        """Take what a team's IC needs of another team straight to that
        team's IC if it is sure which team it is, and to the IC of the team
        above otherwise. Safe from any thread."""
        self._enqueue(lambda: self._ask_team(node_id, escalation_id, team_for), node_id)

    async def _ask_team(self, node_id: str, escalation_id: str, team_for: str) -> None:
        store = self.store
        raiser = store.get(node_id)
        escalation = next(
            (e for e in raiser.escalations if e.id == escalation_id)
            if raiser
            else iter(()),
            None,
        )
        root = store.get(store.root_id) if store.root_id else None
        if raiser is None or escalation is None or root is None:
            return
        if escalation.held_by is None or escalation.asked_of is not None:
            return
        asked = "\n".join(filter(None, (team_for, escalation.title, escalation.detail)))
        choice = await self.choose_team(asked, root, asker=raiser)
        # The team as it is in the session that asks: a request never
        # leaves the session it was made in.
        owner = store.instance(choice.node_id, session_id_of(raiser))
        if not choice.sure or owner is None or not self._may_ask(raiser, owner):
            self.hand_escalation(escalation.held_by, raiser, raiser, escalation, [])
            return
        store.ask_escalation(
            raiser.id, escalation.id, owner.id, self._clock().isoformat()
        )
        self._save_soon()
        parents = {node.id: node.parent_id for node in store.all()}
        above = store.get(
            routing.lowest_holding([raiser.id, owner.id], parents) or root.id
        )
        if above is not None:
            await asyncio.to_thread(
                self._note_request, above, raiser, owner, escalation
            )
        self.tell_ic(owner.id, asked_by_team(raiser, escalation, team_for))

    def _may_ask(self, raiser: OrgNode, owner: OrgNode) -> bool:
        """Whether a team's IC may be asked directly by another's: not a
        team above or below the one that asks, not one that cannot take a
        request now, not one that itself waits on the asking team, and not
        one the asking team already has several requests open with."""
        store = self.store
        parents = {node.id: node.parent_id for node in store.all()}
        if routing.on_one_line(raiser.id, owner.id, parents):
            return False
        if not takes_requests(owner):
            return False
        if any(escalation.asked_of == raiser.id for escalation in owner.escalations):
            return False
        open_with = sum(1 for e in raiser.escalations if e.asked_of == owner.id)
        return open_with < OPEN_REQUESTS_TO_ONE_TEAM

    def _note_request(
        self, above: OrgNode, raiser: OrgNode, owner: OrgNode, escalation: Escalation
    ) -> None:
        """Keep a direct request in the record of the team above both, count
        it between that team's two sub-teams it passed between, and wake
        that team's IC when one team has asked the other as often as the
        org's setbacks number."""
        store = self.store
        self.record_event(above.id, "asked_team", raiser, escalation.title, other=owner)
        sides = [
            next(
                (
                    team
                    for team in (*store.ancestors(node.id), node)
                    if team.parent_id == above.id
                ),
                None,
            )
            for node in (raiser, owner)
        ]
        if sides[0] is not None and sides[1] is not None:
            self.record_event(
                above.id,
                "needed",
                sides[0],
                NEED_ASKED,
                [owner.boundary.path],
                sides[1],
            )
        limit = self.settings.manager_check_ic_setbacks
        table = requests_between(self.team_memory.history(above.id))
        row = next(
            (
                r
                for r in table
                if (r["from_node"], r["to_node"]) == (team_of(raiser), team_of(owner))
            ),
            None,
        )
        if limit and row is not None and row["asked"] == limit:
            self.tell_ic(above.id, many_requests_notice(row, table))

    def answered_directly(
        self, raiser: OrgNode, escalation: Escalation, action: str, text: str
    ) -> str:
        """The IC of a team that was asked directly settles what was asked,
        or cannot. Safe from any thread."""
        owner = self.store.get(escalation.asked_of or "")
        title = owner.title if owner else "a team that is gone"
        if action != "settle":
            self._back_to_chain(raiser, escalation, title, PASSED_UP, text)
            return (
                "Passed up. It is now with the IC of the team above the one that asked."
            )
        self.store.settle_escalation(raiser.id, escalation.id)
        self._save_soon()
        self.progressed(raiser.id)
        self._note_answer(raiser, escalation, SETTLED)
        self.notify(
            agent_conversation(raiser, "ic"),
            f"The IC of {title!r}, which the runtime asked directly, settled "
            f"your escalation {escalation.title!r}:\n{text}",
        )
        return "Settled; the IC that asked has your answer."

    def _back_to_chain(
        self, raiser: OrgNode, escalation: Escalation, title: str, how: str, why: str
    ) -> None:
        """A team that was asked directly did not settle a request: it goes
        to the IC of the team above the one that asked, as it would have
        without the direct way."""
        self._note_answer(raiser, escalation, how)
        back = self.store.ask_escalation(raiser.id, escalation.id, None)
        self._save_soon()
        if back.held_by is None:
            return
        self.hand_escalation(
            back.held_by,
            raiser,
            raiser,
            back,
            [f"{title}, which was asked directly ({how}): {why}"],
        )

    def _note_answer(self, raiser: OrgNode, escalation: Escalation, how: str) -> None:
        """Keep how a direct request ended, in the record that has it."""
        asked = escalation.asked_of
        for above in self.store.ancestors(raiser.id):
            made = [
                event
                for event in self.team_memory.history(above.id)
                if event.kind == "asked_team"
                and (event.team_id, event.other_id)
                == (team_of(raiser), team_key(asked or ""))
            ]
            if made:
                self.record_event(
                    above.id,
                    "answered",
                    raiser,
                    how,
                    other_gone=(made[-1].other_id, made[-1].other),
                )
                return

    async def _check_asks(self) -> None:
        """Give back to the chain each request a team was asked directly
        that it can no longer answer, or has not answered for as long as a
        team may show no progress."""
        store = self._store
        if store is None or self._paused is not None or self._fixture:
            return
        limit = self.settings.stall_check_minutes
        now = self._clock()
        for raiser in store.all():
            for escalation in raiser.escalations:
                if escalation.asked_of is None:
                    continue
                owner = store.get(escalation.asked_of)
                ic = (
                    next((a for a in owner.agents if a.role == "ic"), None)
                    if owner
                    else None
                )
                gone = (
                    owner is None
                    or ic is None
                    or ic.status == AGENT_LOST
                    or owner.stopped is not None
                )
                late = bool(limit) and now - datetime.fromisoformat(
                    escalation.asked_at or escalation.raised_at
                ) >= timedelta(minutes=limit)
                if gone or late:
                    await asyncio.to_thread(
                        self._back_to_chain,
                        raiser,
                        escalation,
                        owner.title if owner else "a team that is gone",
                        UNANSWERED,
                        "it can take no request now"
                        if gone
                        else f"no answer in {limit} min",
                    )

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
                self._enqueue(self._check_asks, STALL_LANE)
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
            # A team the org already had is there for the session's IC to
            # use or not; one the session set up itself is owed a task.
            and self.store.team(team_of(team)) is None
            and not reviews_only(team)
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
