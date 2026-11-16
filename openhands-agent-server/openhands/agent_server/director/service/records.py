"""What the service reads and keeps of the teams: their context, their
records and what they needed of each other."""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openhands.agent_server.director.boundary import (
    owns,
)
from openhands.agent_server.director.git import (
    GitError,
    changed_together,
    tracked_files,
)
from openhands.agent_server.director.git_store import (
    CONTRACTS_FILE,
    GitStore,
    normalize_store_path,
)
from openhands.agent_server.director.models import (
    AGENT_ROLES,
    AgentRole,
    ConversationLink,
    ConversationPlace,
    ConversationRelation,
    FormerTeam,
    HistoryEvent,
    InheritedSpec,
    OldSpec,
    OrgHistory,
    OrgNode,
    Report,
    RoutedRequest,
    SplitProposal,
    TeamEvent,
    TeamEventKind,
    Thread,
)
from openhands.agent_server.director.org_store import session_id_of
from openhands.agent_server.director.prompts import (
    NEEDED_WORTH_SAYING,
    event_line,
    need_words,
    needs_between,
    proposal_facts,
    record_line,
    requests_between,
    track_records,
)
from openhands.agent_server.director.run_state import (
    CHECKS,
    JOINED,
)
from openhands.agent_server.director.service.state import (
    ASKED_KEPT,
    ASKED_TOLD,
    DECLINED_TOLD,
    HISTORY_TOLD,
    NOT_STARTED,
    TOGETHER_WORTH_SAYING,
    ServiceState,
    _now,
    _spec_file,
    team_of,
)
from openhands.agent_server.director.split import (
    is_inside,
)
from openhands.sdk.errors import NotFound


logger = logging.getLogger(__name__)


# How the org's own team is named to the sessions that work beside it.
OWN_SESSION = "the team's own work"


def _written_by(report: Report, session: str) -> str:
    """Who wrote a report, as the agents of a team in the session titled
    ``session`` are told ('' for the org's own)."""
    if report.handed_over:
        return "ic_of_the_team_above"
    if report.session == session:
        return report.role
    return f"{report.role} in {report.session or OWN_SESSION!r}"


class Records(ServiceState):
    """Reads and keeps what is known of the teams."""

    def record_event(
        self,
        above_id: str | None,
        kind: TeamEventKind,
        team: OrgNode | None = None,
        detail: str = "",
        paths: Sequence[str] = (),
        other: OrgNode | None = None,
        *,
        other_gone: tuple[str, str] | None = None,
    ) -> None:
        """Keep, for the IC and the Manager of a team, what became of a team
        directly below it. Safe from any thread.

        Args:
            above_id: The team whose record it is; nothing is kept for None.
            team: The team it is about, or None when it is about none.
            paths: The directories a proposal was for.
            other: The team beside it whose directory it needed, or the
                team it asked directly.
            other_gone: The id and the title of that team when it is no
                longer in the org.
        """
        if above_id is None:
            return
        event = TeamEvent(
            at=_now(),
            kind=kind,
            # The team of the org, whatever session it was the work of.
            team_id=team_of(team) if team else "",
            team=team.title if team else "",
            session=self.store.session_title(team) if team else "",
            detail=detail,
            paths=list(paths),
            other_id=team_of(other) if other else (other_gone or ("", ""))[0],
            other=other.title if other else (other_gone or ("", ""))[1],
        )
        try:
            self.team_memory.append_history(above_id, event)
        except (GitError, OSError):
            logger.warning("Could not record %s for node %s", kind, above_id)

    def record_need(self, team: OrgNode, path: str, how: str) -> None:
        """Keep that a team needed a file of a team beside it; nothing is
        kept when ``path`` is no such team's. The same file needed the same
        way is kept once. Safe from any thread.

        Args:
            path: Relative to the repository.
            how: `NEED_WRITE` or `NEED_FINDING`.
        """
        if team.parent_id is None:
            return
        beside = next(
            (
                other
                for other in self.store.children(team.parent_id)
                if other.id != team.id and owns(other, path)
            ),
            None,
        )
        if beside is None:
            return
        with self._keeping_needs:
            kept = False
            for event in self.team_memory.history(team.parent_id):
                if event.kind == "set_up" and event.team_id in (team.id, beside.id):
                    # A new team in that place is counted from its start.
                    kept = False
                elif event.kind == "needed":
                    kept = kept or (
                        event.team_id,
                        event.other_id,
                        event.paths,
                        event.detail,
                    ) == (team.id, beside.id, [path], how)
            if not kept:
                self.record_event(team.parent_id, "needed", team, how, [path], beside)

    def routed_requests(self) -> list[RoutedRequest]:
        """What the human said that went straight to a team, oldest first.
        Read once for each state of team memory. Safe from any thread."""
        team_memory = self._memory()
        head = team_memory.head()
        with self._lock:
            if self._routed_read is not None and self._routed_read[0] == head:
                return self._routed_read[1]
        routed = team_memory.routed()
        with self._lock:
            self._routed_read = (head, routed)
        return routed

    def keep_routed(
        self, kind: str, team: OrgNode, text: str, chosen_by: str, reason: str
    ) -> None:
        """Keep a request that went straight to a team, or that the team
        said was not its own: for the whole org, and as an event in the
        record of the team above. Safe from any thread."""
        self._memory().append_routed(
            RoutedRequest.model_validate(
                {
                    "at": _now(),
                    "kind": kind,
                    "team_id": team.id,
                    "team": team.title,
                    "text": text,
                    "chosen_by": chosen_by,
                    "reason": reason,
                }
            )
        )
        self.record_event(
            team.parent_id,
            "asked" if kind == "asked" else "misrouted",
            team,
            text[:ASKED_KEPT],
        )

    def asked_below(self, node: OrgNode) -> list[dict[str, str]]:
        """The latest of what the human asked directly of the teams below
        a team, oldest first."""
        store = self.store
        below = {
            team.id
            for team in store.all()
            if any(above.id == node.id for above in store.ancestors(team.id))
        }
        return [
            {
                "node_id": request.team_id,
                "team": request.team,
                "at": request.at,
                "text": request.text,
            }
            for request in self.routed_requests()
            if request.kind == "asked" and request.team_id in below
        ][-ASKED_TOLD:]

    def asked_of(self, team: OrgNode) -> list[str]:
        """What the human asked a team directly since its work was last
        accepted, by the order of the record of the team above."""
        if team.parent_id is None:
            return []
        asked: list[str] = []
        for event in self.team_memory.history(team.parent_id):
            if event.team_id != team.id:
                continue
            if event.kind in ("accepted", "set_up"):
                asked = []
            elif event.kind == "asked":
                asked.append(event.detail)
        return asked

    def needs_below(self, node_id: str) -> list[dict[str, Any]]:
        """The pairs of teams directly below a node that needed each
        other's directory often enough to say so."""
        return [
            pair
            for pair in needs_between(self.team_memory.history(node_id))
            if pair["writes_refused"] + pair["review_findings"] + pair["requests"]
            >= NEEDED_WORTH_SAYING
        ]

    def _so_far(self, team: OrgNode) -> str:
        """What became of a team's work before, in words, as the team above
        has it on record."""
        if team.parent_id is None:
            return ""
        records = track_records(self.team_memory.history(team.parent_id))
        counted = next((record_line(r) for r in records if r["node_id"] == team.id), "")
        needed = [
            need_words(pair, team.id)
            for pair in self.needs_below(team.parent_id)
            if team.id in pair["node_ids"]
        ]
        return "; ".join(part for part in (counted, *needed) if part)

    def facts_of_proposal(self, node: OrgNode, proposal: SplitProposal) -> str:
        """What the director knows of the sub-teams a team's IC proposes:
        how much is in each directory, which were changed together, what
        became of teams that had them before, and what was declined for
        them before."""
        checkout = Path(node.worktree_path)
        children = proposal.children
        sizes = [(child, tracked_files(checkout, child.path)) for child in children]
        together = [
            (one.path, other.path, count)
            for index, one in enumerate(children)
            for other in children[index + 1 :]
            if (count := changed_together(checkout, one.path, other.path))
            >= TOGETHER_WORTH_SAYING
        ]
        history = self.team_memory.history(node.id)
        why = {
            (event.team_id, event.at): event.detail
            for event in history
            if event.kind == "pruned"
        }
        paths = {child.path for child in children}
        before = []
        for former in self.former_teams_of(node):
            if former.path not in paths:
                continue
            reason = next(
                (
                    detail
                    for (team_id, _), detail in reversed(list(why.items()))
                    if team_id == former.id and detail
                ),
                "",
            )
            before.append(
                f"A team {former.title!r} had {former.path} before and was "
                f"{former.ended}"
                + (f": {reason}" if former.ended == "pruned" and reason else ".")
            )
        # Latest first, and no more than a Manager will weigh.
        declined = [
            f"Sub-teams for {', '.join(shared)} were proposed before and "
            f"declined: {event.detail}"
            for event in reversed(history)
            if event.kind == "declined"
            and (shared := sorted(paths.intersection(event.paths)))
        ][:DECLINED_TOLD]
        # Teams that one proposal replaces together are being merged or
        # redrawn: any time they needed each other is worth the Manager's eye.
        needed = [
            f"{need_words(pair)}."
            for pair in needs_between(history)
            if set(pair["node_ids"]) <= set(proposal.replaces)
        ]
        return proposal_facts(
            sizes, together, before + declined + needed, proposal.reason
        )

    def _work_accepted(self, team: OrgNode) -> None:
        """A team's work was accepted: what its IC escalated on it is
        counted no further, since what it does next is other work."""
        if team.parent_id is None:
            return
        with self._counting:
            counted = self._counted(team.parent_id)
            if counted["escalations"].pop(team.id, None) is not None:
                self._runs().mark(CHECKS, team.parent_id, json.dumps(counted))

    def _counted(self, node_id: str) -> dict[str, Any]:
        """What a Manager's checks have counted: the reviews its IC
        accepted, and the escalations of each team below on its work."""
        try:
            kept = json.loads(self._runs().marked(CHECKS).get(node_id) or "{}")
        except ValueError:
            kept = {}
        if not isinstance(kept, dict):
            kept = {}
        below = {team.id for team in self.store.below(node_id)}
        return {
            "accepted": int(kept.get("accepted", 0)),
            "escalations": {
                team_id: int(count)
                for team_id, count in dict(kept.get("escalations") or {}).items()
                if team_id in below
            },
        }

    def place_of(self, conversation_id: str) -> ConversationPlace:
        """Where an agent's conversation is in the org, with the
        conversations one level from it.

        Raises:
            NotFound: No agent of the org has this conversation.
        """
        store = self.store
        found = store.find_by_conversation(conversation_id)
        if found is None:
            raise NotFound(f"No agent of the org has conversation {conversation_id}")
        node, role = found
        above = store.get(node.parent_id) if node.parent_id else None
        if above is not None and above.id == node.id:
            above = None
        near: list[tuple[ConversationRelation, OrgNode]] = [
            *([("above", above)] if above else []),
            ("beside", node),
            *(("below", team) for team in store.children(node.id)),
            *(("session", team) for team in store.opened_under(node.id)),
        ]
        return ConversationPlace(
            conversation_id=conversation_id,
            node_id=team_of(node),
            team=node.title,
            role=role,
            session_id=session_id_of(node),
            session=store.session_title(node),
            human_joined=conversation_id in self._runs().marked(JOINED),
            links=[
                ConversationLink(
                    conversation_id=agent.conversation_id,
                    relation=relation,
                    node_id=team_of(team),
                    team=team.title,
                    role=agent.role,
                    session_id=session_id_of(team),
                    session=store.session_title(team),
                    status=agent.status,
                )
                for relation, team in near
                for agent in team.agents
                if agent.conversation_id
                and agent.conversation_id != conversation_id
                # Below a team, its IC is the one that is written to.
                and (relation in ("above", "beside") or agent.role == "ic")
            ],
        )

    def threads_of(self, node: OrgNode) -> list[Thread]:
        """What the agents of a team have said to Managers and ICs: its own
        two to each other, its Manager and its IC to the ICs directly
        below, and the Manager and the IC above to its IC. Threads nobody
        wrote in are left out."""
        team_memory = self._memory()
        down = [(node.id, team.id) for team in self.store.below(node.id)]
        up = [(node.parent_id, node.id)] if node.parent_id is not None else []
        return [
            Thread(
                manager_node_id=writer,
                ic_node_id=ic,
                ic_to_ic=ic_to_ic,
                messages=messages,
            )
            for ic_to_ic, pairs in (
                (False, [(node.id, node.id), *down, *up]),
                (True, [*down, *up]),
            )
            for writer, ic in pairs
            if (messages := team_memory.thread(writer, ic, ic_to_ic=ic_to_ic))
        ]

    def context_of(
        self, node: OrgNode, role: AgentRole, *, in_full: bool = False
    ) -> dict[str, Any]:
        """Who an agent is and everything of the org that concerns its
        team. Safe from any thread.

        Args:
            in_full: Also what the teams removed from below an IC's team
                left, which is otherwise only named.
        """
        store = self.store
        team_memory = self.team_memory
        above = store.get(node.parent_id) if node.parent_id else None
        _, contracts = team_memory.applies_to(
            node.boundary.path, above.boundary.path if above else None
        )
        specs = self.specs_of(node)
        own_spec = next((spec for spec in specs if not spec.old), None)
        to_settle = [
            {
                "node_id": other.id,
                "team": other.title,
                "escalation_id": escalation.id,
                "kind": escalation.kind,
                "title": escalation.title,
                "detail": escalation.detail,
                "passed_up": [
                    {"by": handoff.node_id, "reason": handoff.reason}
                    for handoff in escalation.handoffs
                ],
            }
            | ({"asked_of_you_directly": True} if escalation.asked_of else {})
            for other in store.all()
            for escalation in other.escalations
            if escalation.raised_by == role
            and (escalation.asked_of or escalation.held_by) == node.id
        ]
        others = [
            other
            for other in store.all()
            if team_of(other) == team_of(node) and other.id != node.id and other.agents
        ]
        title = store.session_title(node)
        context: dict[str, Any] = {
            "you": {"role": role, "node_id": node.id, "team": node.title}
            | ({"session": title} if title else {}),
            "team_above": node.parent_id,
            "status": node.status,
            "boundary": node.boundary.model_dump(),
            "workspace": {
                "path": node.worktree_path,
                "branch": node.branch,
                "isolation": node.isolation,
            },
            "spec": own_spec.spec if own_spec else None,
            "specs_left_by_teams_before_you": [
                {"directory": spec.path or ".", "team": spec.left_by, "spec": spec.spec}
                for spec in specs
                if spec.old
            ],
            "contracts_with_the_teams_beside_you": [
                agreed.contracts for agreed in contracts if not agreed.own
            ],
            "contracts_between_your_sub_teams": [
                agreed.contracts for agreed in contracts if agreed.own
            ],
            "reports_of_your_team": [
                {
                    "by": _written_by(report, title),
                    "at": report.at,
                    "text": report.text,
                }
                for report in self.reports_of(node)
            ],
            "sub_teams": [
                {
                    "node_id": child.id,
                    "team": child.title,
                    "path": child.boundary.path,
                    "status": child.status if child.agents else NOT_STARTED,
                    "headline": child.headline,
                    "summary": child.spec_summary,
                    "review": child.review.state if child.review else None,
                    "stopped": child.stopped.reason if child.stopped else None,
                }
                for child in store.children(node.id)
            ],
            "review_of_your_work": node.review.model_dump() if node.review else None,
            "agent_profiles_a_sub_team_may_run_on": self.settings.allowed_profiles,
            "pending_split": (
                node.pending_split.model_dump() if node.pending_split else None
            ),
            "escalations_you_raised": [
                {
                    "escalation_id": escalation.id,
                    "title": escalation.title,
                    "with": escalation.asked_of or escalation.held_by or "the human",
                }
                for escalation in node.escalations
                if escalation.raised_by == role
            ],
            "escalations_for_you_to_settle": to_settle,
        }
        if others:
            # They work on the same directory at the same time, each in a
            # checkout of its own, and share the team's reports.
            context["other_sessions_of_your_team"] = [
                {
                    "node_id": other.id,
                    "session": store.session_title(other) or OWN_SESSION,
                    "status": other.status,
                    "headline": other.headline,
                    "review": other.review.state if other.review else None,
                    "branch": other.branch,
                }
                for other in others
            ]
        opened = store.opened_under(node.id)
        if role == "ic" and opened:
            context["sessions_whose_work_you_review"] = [
                {
                    "node_id": top.id,
                    "team": top.title,
                    "session": store.session_title(top),
                    "status": top.status if top.agents else NOT_STARTED,
                    "review": top.review.state if top.review else None,
                }
                for top in opened
            ]
        if role == "ic" and store.children(node.id):
            context["asked_directly_of_teams_below"] = self.asked_below(node)
            context["what_teams_below_asked_of_each_other_directly"] = requests_between(
                team_memory.history(node.id)
            )
        if role == "ic":
            # The IC is the one that passes on what removed teams left.
            context["former_teams"] = [
                {
                    "node_id": former.id,
                    "team": former.title,
                    "directory": former.path or ".",
                    "ended": former.ended,
                    "at": former.at,
                }
                | (
                    {
                        "was_for": former.purpose,
                        "reports_as_one": former.compacted_report,
                        "contracts_between_its_sub_teams": former.contracts,
                    }
                    if in_full
                    else {}
                )
                for former in self._former_below(team_memory, node.id)
            ]
        if in_full:
            # What became of the teams below so far, also of the ones that
            # are gone: for both agents to weigh what they do next by.
            events = team_memory.history(node.id)
            context["track_record_of_the_teams_below"] = track_records(events)
            needed = [need_words(pair) for pair in self.needs_below(node.id)]
            if needed:
                context["teams_below_that_needed_each_other"] = needed
            context["what_became_of_them_lately"] = [
                event_line(event) for event in events[-HISTORY_TOLD:]
            ]
        return context

    def _former_below(self, team_memory: GitStore, node_id: str) -> list[FormerTeam]:
        """The records of the teams removed from below a node. They are
        read once for each state of team memory, since a context is built
        for every message."""
        head = team_memory.head()
        with self._lock:
            if self._former_read is None or self._former_read[0] != head:
                self._former_read = (head, {})
            read = self._former_read[1]
        if node_id not in read:
            read[node_id] = team_memory.former_below(node_id)
        return read[node_id]

    def former_teams_of(self, node: OrgNode) -> list[FormerTeam]:
        """What the teams removed from directly below a node left."""
        return self._former_below(self.team_memory, node.id)

    def org_history(self, limit: int, before: str | None = None) -> OrgHistory:
        """What became of every team of the org, newest first.

        Teams needing each other's files are left out: they are counted
        for an IC and are no event of the org.

        Args:
            before: Only events before this time (ISO 8601).
        """
        team_memory = self.team_memory
        events = [
            HistoryEvent(**event.model_dump(), above_id=node.id, above=node.title)
            for node in self.store.all()
            for event in team_memory.history(node.id)
            if event.kind != "needed" and (before is None or event.at < before)
        ]
        # Times are kept to the second: of events in one second the later
        # recorded comes first, and a page never ends inside a second, so
        # that reading on from its last time misses none.
        events.reverse()
        events.sort(key=lambda event: event.at, reverse=True)
        while limit < len(events) and events[limit].at == events[limit - 1].at:
            limit += 1
        return OrgHistory(events=events[:limit], earlier=len(events) > limit)

    def reports_of(self, node: OrgNode) -> list[Report]:
        """What the two agents of a team recorded, oldest first. Reports
        are the team's: an agent started again reads what it wrote before."""
        team_memory = self._memory()
        reports = [
            report
            for role in AGENT_ROLES
            for report in team_memory.reports(team_of(node), role)
        ]
        return sorted(reports, key=lambda report: report.at)

    def specs_of(self, node: OrgNode) -> list[InheritedSpec]:
        """The spec of a team's own boundary and the ones teams before it
        left in its directory, which are old."""
        team_memory = self._memory()
        specs, _ = team_memory.applies_to(node.boundary.path, None)
        left = {old.path: old for old in node.old_specs}
        found = [
            spec.model_copy(
                update={"old": spec.path in left, "left_by": left[spec.path].team}
            )
            if spec.path in left
            else spec
            for spec in specs
        ]
        for old in node.old_specs:
            if old.path == normalize_store_path(node.boundary.path):
                continue
            text = team_memory.read(_spec_file(old.path))
            if text:
                found.append(
                    InheritedSpec(
                        path=old.path,
                        spec=text,
                        readonly=True,
                        old=True,
                        left_by=old.team,
                    )
                )
        return found

    def _specs_left_in(self, path: str, removed: list[OrgNode]) -> list[OldSpec]:
        """The specs a team that is given ``path`` finds there: the one of
        the directory itself and those of the ``removed`` teams whose
        directories hold it or lie in it. None of them is its own."""
        team_memory = self._memory()
        own = normalize_store_path(path)
        left: dict[str, str | None] = {}

        def concerns(directory: str) -> bool:
            return (
                directory == own
                or is_inside(own, directory)
                or is_inside(directory, own)
            )

        for team in removed:
            # What a removed team was left and never rewrote is passed on.
            for old in team.old_specs:
                if concerns(old.path):
                    left.setdefault(old.path, old.team)
            theirs = normalize_store_path(team.boundary.path)
            if concerns(theirs):
                left[theirs] = team.title
        # A spec in the directory itself that no team here is known to have
        # left: a team pruned long ago, or one written by hand.
        left.setdefault(own, None)
        return [
            OldSpec(path=directory, team=team)
            for directory, team in left.items()
            if team_memory.read(_spec_file(directory))
        ]

    def _record_contracts(self, node: OrgNode, contracts: str, replace: bool) -> None:
        """Add the contracts of a split to those of the node's boundary,
        or put them in their place when the split replaced teams."""
        team_memory = self._memory()
        path = node.boundary.path
        earlier = team_memory.read(
            f"{path}/{CONTRACTS_FILE}" if path else CONTRACTS_FILE
        )
        text = f"{contracts.strip()}\n"
        if replace:
            if earlier != text:
                team_memory.write(path, CONTRACTS_FILE, text)
            return
        if earlier and text.strip() in earlier:
            return
        team_memory.write(
            path,
            CONTRACTS_FILE,
            f"{earlier.rstrip()}\n\n{text}" if earlier and earlier.strip() else text,
        )
