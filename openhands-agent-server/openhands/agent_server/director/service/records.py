"""What the service reads and keeps of the teams: their context, their
records and what they needed of each other."""

import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

from pydantic import TypeAdapter

from openhands.agent_server.director.boundary import (
    owns,
)
from openhands.agent_server.director.git import (
    changed_together,
    committed_text,
    tracked_files,
)
from openhands.agent_server.director.git_store import (
    SPEC_FILE,
    GitStore,
    Landing,
    normalize_store_path,
    team_key,
)
from openhands.agent_server.director.models import (
    BoundaryContracts,
    ConversationLink,
    ConversationPlace,
    ConversationRelation,
    FormerTeam,
    FormerThread,
    HistoryEvent,
    InheritedSpec,
    OldSpec,
    OrgHistory,
    OrgNode,
    Report,
    RoutedRequest,
    SessionTeam,
    SplitProposal,
    TeamEvent,
    TeamEventKind,
    Thread,
    ThreadMessage,
)
from openhands.agent_server.director.org_store import (
    is_division_lead,
    team_id_of,
    utc_now,
)
from openhands.agent_server.director.prompts import (
    NEEDED_WORTH_SAYING,
    event_line,
    need_words,
    needs_between,
    proposal_facts,
    record_line,
    track_records,
)
from openhands.agent_server.director.recovery import STEP_NAMES
from openhands.agent_server.director.run_state import (
    CHECKS,
    EVENTS,
    FILES,
    FORMER,
    JOINED,
    REPORTS,
    ROUTED,
    TASK,
    THREADS,
    USE,
    Intent,
)
from openhands.agent_server.director.service.state import (
    _REASON_KEPT,
    ASKED_KEPT,
    ASKED_TOLD,
    DECLINED_TOLD,
    HISTORY_TOLD,
    NOT_STARTED,
    TOGETHER_WORTH_SAYING,
    ServiceState,
    _contracts_file,
    _spec_file,
)
from openhands.agent_server.director.split import (
    home_of_lost,
    is_inside,
)
from openhands.sdk.errors import NotFound
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)
# How much of one document of another team a context carries.
DOCUMENT_TOLD = 20_000
_REPORTS = TypeAdapter(list[Report])
_EVENTS = TypeAdapter(list[TeamEvent])
_MESSAGES = TypeAdapter(list[ThreadMessage])


def _directory_of(file: str) -> str:
    """The directory a spec or contracts file of team memory is of."""
    return file.rpartition("/")[0]


def session_of(node: OrgNode) -> str:
    """The session a place belongs to; '' for a team of the org."""
    return node.session if isinstance(node, SessionTeam) else ""


def session_in(node_id: str) -> str:
    """The session the id of a place names; '' for the id of a team of the
    org."""
    return node_id.split("@")[1] if "@" in node_id else ""


def thread_name(writer_node_id: str, ic_node_id: str, ic_to_ic: bool) -> str:
    """What names the messages of one agent to another."""
    kind = ".threads-ic" if ic_to_ic else ".threads"
    return f"{kind}/{writer_node_id}/{ic_node_id}.json"


class Taken(NamedTuple):
    """What became of a session's records when its work was accepted."""

    grown: list[tuple[str, str]]
    # The files that were not written: someone else changed them meanwhile.
    left: list[str]


def _written_by(report: Report, session: str) -> str:
    """Who wrote a report, as the agents of a team in the session with the
    id ``session`` are told."""
    if report.handed_over:
        return "ic_of_the_team_above"
    if report.session_id == session:
        return report.role
    return f"{report.role} in {report.session!r}"


# The events of a session that change what the org has of its teams.
_CHANGES_THE_ORG: frozenset[TeamEventKind] = frozenset(
    {"set_up", "replaced", "pruned", "moved"}
)


class Records(ServiceState):
    """Reads and keeps what is known of the teams."""

    def _keep_failure(self, intent: Intent, error: Exception) -> None:
        """Keep a step that failed where the team above and the human read
        what became of a team."""
        team = self.org.place(intent.node_id)
        if team is None:
            return
        what = STEP_NAMES.get(intent.kind, intent.kind)
        self.record_event(
            team.parent_id or team.id,
            "failed",
            team,
            f"{what}: not done. {type(error).__name__}: {error}"[:_REASON_KEPT],
        )

    def record_event(
        self,
        above_id: str | None,
        kind: TeamEventKind,
        team: SessionTeam | None = None,
        detail: str = "",
        paths: Sequence[str] = (),
        other: SessionTeam | None = None,
        *,
        other_gone: tuple[str, str] | None = None,
        between: tuple[str, str] = ("", ""),
    ) -> None:
        """Keep, for the IC and the Manager of a team, what became of a team
        directly below it. It is its session's alone until the session's
        work is accepted. Safe from any thread.

        Args:
            above_id: The team whose record it is; nothing is kept for None.
            team: The team it is about, or None when it is about none.
            paths: The directories a proposal was for.
            other: The team beside it whose directory it needed, or the
                team it asked directly.
            other_gone: The id and the title of that team when it is no
                longer in the org.
            between: The two commits an accept took the team's work between.
        """
        session = session_in(above_id or "") or (team.session if team else "")
        if above_id is None or not session:
            return
        event = self._event(kind, team, detail, paths, other, other_gone)
        event = event.model_copy(update={"base": between[0], "head": between[1]})
        key = f"{session}\n{team_key(above_id)}"
        with self._keeping:
            events = [*self.own_events(session, team_key(above_id)), event]
            self._runs().mark(EVENTS, key, _EVENTS.dump_json(events).decode())
        if kind in _CHANGES_THE_ORG:
            self._note_change(session)

    def _note_change(self, session: str) -> None:
        """Remember that a session whose work is accepted has changed
        something since: the org takes it only when it is submitted again."""
        top = next(iter(self.org.session(session)), None)
        if top is not None and top.review is not None:
            if top.review.state == "accepted" and self.org.opens_session(top):
                self._changed_since_accept.setdefault(session, False)

    def _event(
        self,
        kind: TeamEventKind,
        team: SessionTeam | None,
        detail: str = "",
        paths: Sequence[str] = (),
        other: SessionTeam | None = None,
        other_gone: tuple[str, str] | None = None,
    ) -> TeamEvent:
        return TeamEvent(
            at=utc_now(),
            kind=kind,
            # The team of the org, whatever session it was the work of.
            team_id=team.team_id if team else "",
            team=team.title if team else "",
            session=self.org.session_title(team) if team else "",
            detail=detail,
            paths=list(paths),
            other_id=other.team_id if other else (other_gone or ("", ""))[0],
            other=other.title if other else (other_gone or ("", ""))[1],
            id=uuid4().hex,
            unaccepted=True,
        )

    def _land_event(self, team_id: str, event: TeamEvent) -> None:
        """Add to the org's record of a team what is no session's to keep."""
        self._memory().take_in(
            Landing(
                files={},
                seen={},
                reports={},
                former=[],
                forget=[],
                events={team_id: [event.model_copy(update={"unaccepted": False})]},
                routed=[],
                members=self.member_names(),
            ),
            f"memory: history of {team_id}",
        )

    def own_events(self, session: str, team: str) -> list[TeamEvent]:
        """What a session kept of what became of the teams below a team
        since its work was last accepted."""
        kept = self._runs().marked(EVENTS).get(f"{session}\n{team}")
        return _EVENTS.validate_json(kept) if kept else []

    def history_of(self, node_id: str) -> list[TeamEvent]:
        """What became of the teams directly below a team, oldest first:
        the org's record, and for a place what its session kept since."""
        node = self.org.get(node_id)
        org = (
            self._memory().history(node_id)
            if node is None or self.is_orgs(node)
            else []
        )
        session = session_in(node_id)
        own = self.own_events(session, team_key(node_id)) if session else []
        return [*org, *own]

    def record_need(self, team: SessionTeam, path: str, how: str) -> None:
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
                for other in self.org.below(team.parent_id)
                if other.id != team.id and owns(other, path)
            ),
            None,
        )
        if beside is None:
            return
        with self._keeping_needs:
            kept = False
            for event in self.history_of(team.parent_id):
                if event.kind == "set_up" and event.team_id in (
                    team.team_id,
                    beside.team_id,
                ):
                    # A new team in that place is counted from its start.
                    kept = False
                elif event.kind == "needed":
                    kept = kept or (
                        event.team_id,
                        event.other_id,
                        event.paths,
                        event.detail,
                    ) == (team.team_id, beside.team_id, [path], how)
            if not kept:
                self.record_event(team.parent_id, "needed", team, how, [path], beside)

    def routed_requests(self) -> list[RoutedRequest]:
        """What the human said that went straight to a team, oldest first:
        what the org has on record, and what the sessions open here kept
        since their work was last accepted. Safe from any thread."""
        team_memory = self._memory()
        head = team_memory.head()
        with self._lock:
            read = self._routed_read
        if read is None or read[0] != head:
            read = (head, team_memory.routed())
            with self._lock:
                self._routed_read = read
        own = [
            RoutedRequest.model_validate_json(raw)
            for raw in self._runs().marked(ROUTED).values()
        ]
        return sorted([*read[1], *own], key=lambda request: request.at)

    def keep_routed(
        self, kind: str, team: SessionTeam, text: str, chosen_by: str, reason: str
    ) -> None:
        """Keep a request that went straight to a team, or that the team
        said was not its own, and as an event in the record of the team
        above. Both are its session's alone until the session's work is
        accepted. Safe from any thread."""
        request = RoutedRequest.model_validate(
            {
                "at": utc_now(),
                "kind": kind,
                "team_id": team.id,
                "team": team.title,
                "text": text,
                "chosen_by": chosen_by,
                "reason": reason,
                "id": uuid4().hex,
            }
        )
        self._runs().mark(
            ROUTED, f"{team.session}\n{request.id}", request.model_dump_json()
        )
        self.record_event(
            team.parent_id,
            "asked" if kind == "asked" else "misrouted",
            team,
            text[:ASKED_KEPT],
        )

    def asked_below(self, place: SessionTeam) -> list[dict[str, str]]:
        """The latest of what the human asked directly of the teams below
        a team, oldest first."""
        store = self.org
        below = {
            team.id
            for team in store.places()
            if any(above.id == place.id for above in store.above(team.id))
        }
        return [
            {
                "team_id": request.team_id,
                "team": request.team,
                "at": request.at,
                "text": request.text,
            }
            for request in self.routed_requests()
            if request.kind == "asked" and request.team_id in below
        ][-ASKED_TOLD:]

    def asked_of(self, team: SessionTeam) -> list[str]:
        """What the human asked a team directly since its work was last
        accepted, by the order of the record of the team above."""
        if team.parent_id is None:
            return []
        asked: list[str] = []
        for event in self.history_of(team.parent_id):
            if event.team_id != team.team_id:
                continue
            if event.kind in ("accepted", "set_up"):
                asked = []
            elif event.kind == "asked":
                asked.append(event.detail)
        return asked

    def needs_below(self, node_id: str) -> list[dict[str, Any]]:
        """The pairs of teams directly below a team that needed each
        other's directory often enough to say so."""
        return [
            pair
            for pair in needs_between(self.history_of(node_id))
            if pair["writes_refused"] + pair["review_findings"] >= NEEDED_WORTH_SAYING
        ]

    def _so_far(self, team: SessionTeam) -> str:
        """What became of a team's work before, in words, as the team above
        has it on record."""
        if team.parent_id is None:
            return ""
        records = track_records(self.history_of(team.parent_id))
        own = team.team_id
        counted = next((record_line(r) for r in records if r["team_id"] == own), "")
        needed = [
            need_words(pair, own)
            for pair in self.needs_below(team.parent_id)
            if own in pair["team_ids"]
        ]
        return "; ".join(part for part in (counted, *needed) if part)

    def facts_of_proposal(self, place: SessionTeam, proposal: SplitProposal) -> str:
        """What the director knows of the sub-teams a team's IC proposes:
        how much is in each directory, which were changed together, what
        became of teams that had them before, and what was declined for
        them before."""
        checkout = Path(place.worktree_path)
        children = proposal.children
        sizes = [(child, tracked_files(checkout, child.path)) for child in children]
        together = [
            (one.path, other.path, count)
            for index, one in enumerate(children)
            for other in children[index + 1 :]
            if (count := changed_together(checkout, one.path, other.path))
            >= TOGETHER_WORTH_SAYING
        ]
        history = self.history_of(place.id)
        why = {
            (event.team_id, event.at): event.detail
            for event in history
            if event.kind == "pruned"
        }
        paths = {child.path for child in children}
        before = []
        for former in self.former_teams_of(place):
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
            if set(pair["team_ids"]) <= {team_key(r) for r in proposal.replaces}
        ]
        return proposal_facts(
            sizes,
            together,
            before + declined + needed,
            proposal.reason,
            names=self.team(place).names,
        )

    def _work_accepted(self, team: SessionTeam) -> None:
        """A team's work was accepted: what its IC escalated on it is
        counted no further, since what it does next is other work."""
        if team.parent_id is None:
            return
        with self._counting:
            counted = self._counted(team.parent_id)
            if counted["escalations"].pop(team.id, None) is not None:
                self._runs().mark(CHECKS, team.parent_id, json.dumps(counted))

    def _counted(self, place_id: str) -> dict[str, Any]:
        """What a Manager's checks have counted: the reviews its IC
        accepted, and the escalations of each team below on its work."""
        try:
            kept = json.loads(self._runs().marked(CHECKS).get(place_id) or "{}")
        except ValueError:
            kept = {}
        if not isinstance(kept, dict):
            kept = {}
        below = {team.id for team in self.org.below(place_id)}
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
        store = self.org
        found = store.find_by_conversation(conversation_id)
        if found is None:
            raise NotFound(f"No agent of the org has conversation {conversation_id}")
        node, role = found
        above = store.place(node.parent_id) if node.parent_id else None
        near: list[tuple[ConversationRelation, SessionTeam]] = [
            *([("above", above)] if above else []),
            ("beside", node),
            *(("below", team) for team in store.below(node.id)),
        ]
        return ConversationPlace(
            conversation_id=conversation_id,
            node_id=node.team_id,
            team=node.title,
            role=role,
            session_id=node.session,
            session=store.session_title(node),
            human_joined=conversation_id in self._runs().marked(JOINED),
            links=[
                ConversationLink(
                    conversation_id=agent.conversation_id,
                    relation=relation,
                    node_id=team.team_id,
                    team=team.title,
                    role=agent.role,
                    session_id=team.session,
                    session=store.session_title(team),
                    status=agent.status,
                )
                for relation, team in near
                for agent in team.agents
                if agent.conversation_id
                and agent.conversation_id != conversation_id
                # Below a team, its IC is the one that is written to.
                and (
                    relation in ("above", "beside")
                    or agent.role == self.team(team).writer
                )
            ],
        )

    def threads_of(self, place: SessionTeam) -> list[Thread]:
        """What the agents of a team have said to Managers and ICs: its own
        two to each other, its Manager and its IC to the ICs directly
        below, and the Manager and the IC above to its IC. Threads nobody
        wrote in are left out."""
        down = [(place.id, team.id) for team in self.org.below(place.id)]
        up = [(place.parent_id, place.id)] if place.parent_id is not None else []
        return [
            Thread(
                manager_node_id=writer,
                ic_node_id=ic,
                ic_to_ic=ic_to_ic,
                messages=messages,
            )
            for ic_to_ic, pairs in (
                (False, [(place.id, place.id), *down, *up]),
                (True, [*down, *up]),
            )
            for writer, ic in pairs
            if (messages := self._thread(writer, ic, ic_to_ic))
        ]

    def _thread(
        self, writer_node_id: str, ic_node_id: str, ic_to_ic: bool
    ) -> list[ThreadMessage]:
        """What a Manager and an IC have said to each other, oldest first;
        with ``ic_to_ic``, what the IC of the team above wrote to an IC.
        It is kept with the session and is never the org's."""
        name = thread_name(writer_node_id, ic_node_id, ic_to_ic)
        kept = self._runs().marked(THREADS).get(name)
        return _MESSAGES.validate_json(kept) if kept else []

    def _keep_message(
        self,
        writer_node_id: str,
        ic_node_id: str,
        message: ThreadMessage,
        *,
        ic_to_ic: bool = False,
    ) -> None:
        name = thread_name(writer_node_id, ic_node_id, ic_to_ic)
        with self._keeping:
            messages = [*self._thread(writer_node_id, ic_node_id, ic_to_ic), message]
            self._runs().mark(THREADS, name, _MESSAGES.dump_json(messages).decode())

    def _take_threads(self, place: str) -> list[FormerThread]:
        """Take out what the agents at a session's place wrote and were
        written, so that a later team at the same place does not find it."""
        run = self._runs()
        taken = [
            FormerThread(file=name, messages=_MESSAGES.validate_json(raw))
            for name, raw in run.marked(THREADS).items()
            if place in (Path(name).parent.name, Path(name).stem)
        ]
        for thread in taken:
            run.unmark(THREADS, thread.file)
        return taken

    def context_of(
        self, place: SessionTeam, role: str, *, in_full: bool = False
    ) -> dict[str, Any]:
        """Who an agent is and everything of the org that concerns its
        team. Safe from any thread.

        Args:
            in_full: Also what the teams removed from below an IC's team
                left, which is otherwise only named.
        """
        store = self.org
        contracts = self.contracts_of(place)
        specs = self.specs_of(place)
        own_spec = next((spec for spec in specs if not spec.old), None)
        to_settle = [
            {
                "team_id": other.id,
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
            for other in store.places()
            for escalation in other.escalations
            if escalation.raised_by == role and escalation.held_by == place.id
        ]
        others = [
            other
            for other in store.places()
            if other.team_id == place.team_id
            and other.id != place.id
            and other.agents
            and not is_division_lead(other)
        ]
        title = store.session_title(place)
        context: dict[str, Any] = {
            "you": {
                "role": role,
                "team_id": place.id,
                "team": place.title,
                "type": place.agent_type,
            }
            | ({"session": title} if title else {}),
            "team_above": place.parent_id,
            "status": place.status,
            "boundary": place.boundary.model_dump(),
            "workspace": {
                "path": place.worktree_path,
                "branch": place.branch,
                "isolation": place.isolation,
            },
            "spec": own_spec.spec if own_spec else None,
            "specs_left_by_teams_before_you": [
                {
                    "directory": spec.path or "the repository",
                    "team": spec.left_by,
                    "spec": spec.spec,
                }
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
                    "by": _written_by(report, session_of(place)),
                    "at": report.at,
                    "text": report.text,
                }
                for report in self.reports_of(place)
            ],
            "sub_teams": [
                {
                    "team_id": child.id,
                    "team": child.title,
                    "path": child.boundary.path,
                    "status": child.status if child.agents else NOT_STARTED,
                    "headline": child.headline,
                    "summary": child.spec_summary,
                    "review": child.review.state if child.review else None,
                    "stopped": child.stopped.reason if child.stopped else None,
                }
                for child in store.below(place.id)
            ],
            # What it wrote of its work and what was found in it reach it in
            # the messages of the review, not a second time here.
            "review_of_your_work": (
                {"state": place.review.state, "round": place.review.round}
                if place.review
                else None
            ),
            "agent_profiles_a_sub_team_may_run_on": self.settings.allowed_profiles,
            "types_a_sub_team_may_be": {
                kind.name: kind.description
                + (
                    f" Runs on the agent profile {kind.sas_profile!r} unless "
                    "you name another."
                    if kind.sas_profile
                    else ""
                )
                for kind in self.types_allowed()
            },
            "type_of_a_sub_team_that_names_none": self.default_type(),
            # The proposal itself is in the message that made it.
            "pending_split": (
                None
                if place.pending_split is None
                else place.pending_split.model_dump()
                if in_full
                else {
                    "status": place.pending_split.status,
                    "teams": [child.title for child in place.pending_split.children],
                }
            ),
            "escalations_you_raised": [
                {
                    "escalation_id": escalation.id,
                    "title": escalation.title,
                    "with": escalation.held_by or "the human",
                }
                for escalation in place.escalations
                if escalation.raised_by == role
            ],
            "escalations_for_you_to_settle": to_settle,
        }
        if others:
            # They work on the same directory at the same time, each in a
            # checkout of its own, and share the team's reports.
            context["other_sessions_of_your_team"] = [
                {
                    "team_id": other.id,
                    "session": store.session_title(other),
                    "status": other.status,
                    "headline": other.headline,
                    "review": other.review.state if other.review else None,
                    "branch": other.branch,
                }
                for other in others
            ]
        if documents := self.documents_for(place):
            context["documents_of_other_teams"] = documents
        works = role == self.team(place).writer
        if works and store.below(place.id):
            context["asked_directly_of_teams_below"] = self.asked_below(place)
        if works:
            # The IC is the one that passes on what removed teams left.
            context["former_teams"] = [
                {
                    "team_id": former.id,
                    "team": former.title,
                    "directory": former.path or "the repository",
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
                for former in self.former_teams_of(place)
            ]
        if in_full:
            # What became of the teams below so far, also of the ones that
            # are gone: for both agents to weigh what they do next by.
            events = self.history_of(place.id)
            context["track_record_of_the_teams_below"] = track_records(events)
            needed = [need_words(pair) for pair in self.needs_below(place.id)]
            if needed:
                context["teams_below_that_needed_each_other"] = needed
            context["what_became_of_them_lately"] = [
                event_line(event) for event in events[-HISTORY_TOLD:]
            ]
        return context

    def documents_for(self, place: SessionTeam) -> list[dict[str, Any]]:
        """The documents other teams keep by their types that the team at
        ``place`` reads: of the team above, the teams below and the teams
        beside it, each where its type names that reader.

        What goes up or beside is given as the checkout of the team above
        its keeper has it committed, which is where accepted work is
        merged, so what a team wrote and nobody accepted yet is not passed
        on. A team that works in the directory of the team above
        (`shared`) commits into that checkout, so there it is what was
        committed. What a team keeps for its sub-teams is its own to say,
        as contracts are: it is given as the team's own checkout has it.
        Nobody works at a session's Division Lead place, so it keeps none.
        """
        store = self.org
        above = store.place(place.parent_id) if place.parent_id else None
        kept_by: list[tuple[SessionTeam, str]] = [
            (child, "above") for child in store.below(place.id)
        ]
        if above is not None:
            kept_by.append((above, "below"))
            kept_by += [
                (other, "beside")
                for other in store.below(above.id)
                if other.id != place.id
            ]
        found: list[dict[str, Any]] = []
        for keeper, reader in kept_by:
            if is_division_lead(keeper):
                continue
            over = store.place(keeper.parent_id) if keeper.parent_id else None
            checkout = (
                over.worktree_path
                if over is not None and reader != "below"
                else keeper.worktree_path
            )
            folder = normalize_store_path(keeper.boundary.path)
            for name, document in self.type_of(keeper).own_documents.items():
                if reader not in document.readers or not checkout:
                    continue
                path = f"{folder}/{document.path}" if folder else document.path
                text = committed_text(Path(checkout), path)
                if text is None:
                    continue
                found.append(
                    {
                        "document_id": f"{keeper.id}/{name}",
                        "team_id": keeper.id,
                        "team": keeper.title,
                        "document": name,
                        "holds": document.holds.strip(),
                        "path": path,
                        "text": text
                        if len(text) <= DOCUMENT_TOLD
                        else text[:DOCUMENT_TOLD] + f" … (cut; the rest is in {path})",
                    }
                )
        return found

    def _former_below(self, team_memory: GitStore, node_id: str) -> list[FormerTeam]:
        """The records of the teams removed from below a team. They are
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
        """What the teams removed from directly below a team left, with
        the ones its own session removed and the org does not know of yet."""
        own = [
            former
            for former in self.former_drafts(session_of(node))
            if former.above == team_key(node.id)
        ]
        return [*self._former_below(self.team_memory, node.id), *own]

    def org_history(self, limit: int, before: str | None = None) -> OrgHistory:
        """What became of every team of the org, newest first.

        Teams needing each other's files are left out: they are counted
        for an IC and are no event of the org.

        Args:
            before: Only events before this time (ISO 8601).
        """
        team_memory = self.team_memory
        # A record is the team's, whichever sessions have a place at it.
        teams: dict[str, str] = {}
        for node in (*self.org.teams(), *self.org.places()):
            teams.setdefault(team_id_of(node), node.title)
        events = [
            HistoryEvent(**event.model_dump(), above_id=team_id, above=title)
            for team_id, title in teams.items()
            for event in team_memory.history(team_id)
            if event.kind != "needed" and (before is None or event.at < before)
        ]
        # Times are kept to the second: of events in one second the later
        # recorded comes first, and a page never ends inside a second, so
        # that reading on from its last time misses none.
        for key, raw in self._runs().marked(EVENTS).items():
            team_id = key.partition("\n")[2]
            events += [
                HistoryEvent(
                    **event.model_dump(),
                    above_id=team_id,
                    above=teams.get(team_id, ""),
                )
                for event in _EVENTS.validate_json(raw)
                if event.kind != "needed" and (before is None or event.at < before)
            ]
        events.reverse()
        events.sort(key=lambda event: event.at, reverse=True)
        while limit < len(events) and events[limit].at == events[limit - 1].at:
            limit += 1
        return OrgHistory(events=events[:limit], earlier=len(events) > limit)

    def is_orgs(self, node: OrgNode) -> bool:
        """Whether a place is of the team the org has by its id now. An id
        comes from the directory, so the org may have another team there."""
        if not isinstance(node, SessionTeam):
            return True
        team = self.org.team(node.team_id)
        return team is not None and team.since == node.since

    def reports_of(self, node: OrgNode) -> list[Report]:
        """What the two agents of a team recorded, oldest first. Reports
        are the team's: an agent started again reads what it wrote before."""
        team_memory = self._memory()
        reports = [
            report
            for role in self.type_of(node).agents
            for report in team_memory.reports(team_id_of(node), role)
            if self.is_orgs(node)
        ]
        return sorted([*reports, *self.own_reports(node)], key=lambda report: report.at)

    def own_reports(self, node: OrgNode) -> list[Report]:
        """What the agents of a session's place recorded since the session's
        work was last accepted. It is the session's alone until its work
        is accepted again, and the team's from then on. Nothing for a team
        of the org."""
        kept = self._runs().marked(REPORTS).get(node.id)
        return _REPORTS.validate_json(kept) if kept else []

    def _keep_report(self, place: SessionTeam, report: Report) -> None:
        named = report.model_copy(update={"id": report.id or uuid4().hex})
        with self._keeping:
            reports = [*self.own_reports(place), named]
            self._runs().mark(REPORTS, place.id, _REPORTS.dump_json(reports).decode())

    def drafts(self, session: str) -> dict[str, tuple[str | None, str | None]]:
        """The files of team memory a session wrote, which are its own
        until its work is accepted.

        Returns:
            For each file, its text (None where the session removed it)
            and the text it was written over (None where there was none).
        """
        if not session:
            return {}
        prefix = f"{session}\n"
        found = {}
        for key, raw in self._runs().marked(FILES).items():
            if key.startswith(prefix):
                draft = json.loads(raw)
                found[key.removeprefix(prefix)] = (draft["text"], draft["over"])
        return found

    def _draft(self, session: str, file: str, text: str | None) -> None:
        """Keep, for a session alone, what it makes of a file of team
        memory. The text it is written over is the one the session first
        wrote over, so that what the org made of it since is not lost
        sight of."""
        with self._keeping:
            earlier = self.drafts(session).get(file)
            over = earlier[1] if earlier else self.team_memory.read(file)
            self._runs().mark(
                FILES, f"{session}\n{file}", json.dumps({"text": text, "over": over})
            )
        self._note_change(session)

    def _undraft(self, session: str, file: str) -> None:
        self._runs().unmark(FILES, f"{session}\n{file}")

    def read_file(self, node: OrgNode, file: str) -> str | None:
        """A file of team memory as the agents of ``node`` have it: what
        their session wrote there, else the org's."""
        draft = self.drafts(session_of(node)).get(file)
        return draft[0] if draft else self._memory().read(file)

    def contracts_of(self, node: OrgNode) -> list[BoundaryContracts]:
        """The contracts a team works to: the ones between it and the teams
        beside it, set by the team above, and the ones between its own
        sub-teams."""
        above = self.org.get(node.parent_id) if node.parent_id else None
        own = normalize_store_path(node.boundary.path)
        paths = [
            normalize_store_path(place.boundary.path)
            for place in (above, node)
            if place
        ]
        return [
            BoundaryContracts(path=path, contracts=text, own=path == own)
            # A session opened at the top team has that team's own place above.
            for path in dict.fromkeys(paths)
            if (text := self.read_file(node, _contracts_file(path)))
        ]

    def specs_of(self, node: OrgNode) -> list[InheritedSpec]:
        """The spec of a team's own boundary and the ones teams before it
        left in its directory, which are old."""
        own = normalize_store_path(node.boundary.path)
        text = self.read_file(node, _spec_file(own))
        specs = (
            [] if text is None else [InheritedSpec(path=own, spec=text, readonly=False)]
        )
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
            if old.path == own:
                continue
            text = self.read_file(node, _spec_file(old.path))
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

    def spec_draft(self, node: OrgNode) -> tuple[str, str | None] | None:
        """The spec a session wrote for one of its teams, which is its own
        until its work is accepted.

        Returns:
            The spec and the team's spec it was written over, None when
            the team had none; None when the session wrote no spec there.
        """
        file = _spec_file(normalize_store_path(node.boundary.path))
        draft = self.drafts(session_of(node)).get(file)
        if draft is None or draft[0] is None:
            return None
        return draft[0], draft[1]

    def write_spec(self, place: SessionTeam, content: str) -> None:
        """Write the spec of a place's directory. It is kept for the
        session alone until its work is accepted."""
        file = _spec_file(normalize_store_path(place.boundary.path))
        self._draft(place.session, file, content)

    def specs_inherited(self, top: SessionTeam) -> list[tuple[str, str, str, str]]:
        """The specs a session wrote for teams the org had when it began
        and has no more.

        Returns:
            For each, the directory of the team of the org that holds the
            lost team's directory now, that team's spec, what the lost team
            was called and the spec the session wrote for it.
        """
        store = self.org
        session = store.session_of(top)
        if session is None:
            return []
        began = {team for team, *_ in session.shape}
        found = []
        for place in store.session(top.session):
            if (
                is_division_lead(place)
                or place.team_id not in began
                or store.team(place.team_id) is not None
            ):
                continue
            draft = self.spec_draft(place)
            home = store.team(self._home_of(top, place.boundary.path))
            if draft is None or home is None:
                continue
            path = normalize_store_path(home.boundary.path)
            now = self._memory().read(_spec_file(path)) or ""
            found.append((path, now, place.title, draft[0]))
        return found

    def specs_to_reconcile(self, top: SessionTeam) -> list[tuple[str, str, str]]:
        """The specs a session wrote while the team's spec in the same
        directory was changed by someone else.

        Returns:
            For each, the directory, the team's spec now and the session's.
        """
        team_memory = self._memory()
        found = []
        for file, (text, over) in sorted(self.drafts(top.session).items()):
            if not file.endswith(SPEC_FILE) or text is None:
                continue
            now = team_memory.read(file)
            if now != over and now != text:
                found.append((_directory_of(file), now or "", text))
        return found

    def _home_of(self, top: SessionTeam, path: str) -> str:
        """The team of the org that has what a session's team the org does
        not have left: the nearest one that holds its directory."""
        session = self.org.session_of(top)
        return home_of_lost(
            path, self.org.teams(), session.above_id if session else top.team_id
        )

    def former_drafts(self, session: str) -> list[FormerTeam]:
        """The records of the teams a session removed, which are its own
        until its work is accepted."""
        if not session:
            return []
        return [
            FormerTeam.model_validate_json(raw)
            for key, raw in self._runs().marked(FORMER).items()
            if key.startswith(f"{session}\n")
        ]

    def _keep_former(self, session: str, record: FormerTeam) -> None:
        self._runs().mark(
            FORMER, f"{session}\n{record.id}\n{record.at}", record.model_dump_json()
        )

    def _land_records(self, top: SessionTeam, *args: Any, **kwargs: Any) -> Taken:
        """:meth:`_land_kept`, with nothing written to the session's records
        between reading them and clearing them: it would be lost."""
        with self._keeping:
            return self._land_kept(top, *args, **kwargs)

    def _land_kept(
        self,
        top: SessionTeam,
        seen: dict[str, str | None] | None = None,
        chosen: dict[str, str | None] | None = None,
        *,
        waits: Collection[str] = (),
        held: Collection[str] = (),
        tree: bool = False,
    ) -> Taken:
        """Make what a session kept to itself the org's, now that its work
        is accepted: the specs and contracts it wrote, what its agents
        recorded and the records of the teams it removed. All of it is one
        commit of team memory.

        What is of a team that waits stays the session's until then. What
        is of a team the org does not have, or has another team in place
        of, is kept as that team's record with the nearest team of the org
        that holds its directory. A spec is put in place of one that
        changed only where its reviewer was shown that one.

        Args:
            seen: The specs the reviewer was shown, by file.
            chosen: The contracts that were settled, by file: the text
                that holds.
            waits: The teams whose records stay the session's for now.
            held: The files that stay the session's for now.
            tree: Whether the org tree as it is now lands with it.

        Returns:
            The records that grew and the files that were not written.
        """
        store = self.org
        run = self._runs()
        session = top.session
        review = store.place(top.parent_id or "")
        held = set(held) - set(chosen or {})
        places = [
            place for place in store.session(session) if not is_division_lead(place)
        ]
        by_path = {normalize_store_path(p.boundary.path): p for p in places}
        stamp = utc_now()
        left = {
            place.id: place
            for place in places
            if place.team_id not in waits and not self.is_orgs(place)
        }
        files: dict[str, tuple[str | None, str | None]] = {}
        passed_on: dict[str, SessionTeam] = {}
        of_left: dict[str, dict[str, str]] = {}
        drafts = self.drafts(session)
        for file, text in (chosen or {}).items():
            if file not in drafts:
                # Contracts that came with a team: there is no draft of them.
                files[file] = (text, self._memory().read(file))
        for file, (text, over) in drafts.items():
            owner = by_path.get(_directory_of(file))
            if file in held or (owner is not None and owner.team_id in waits):
                continue
            if owner is not None and owner.id in left:
                kind = "spec" if file.endswith(SPEC_FILE) else "contracts"
                of_left.setdefault(owner.id, {})[kind] = text or ""
                if (
                    kind == "spec"
                    and text
                    and store.team(owner.team_id) is None
                    and all(
                        normalize_store_path(team.boundary.path) != _directory_of(file)
                        for team in store.teams()
                    )
                ):
                    # No team has the directory: the spec stays there as an
                    # old one of the team that holds it now.
                    files[file] = (text, over)
                    passed_on[file] = owner
                continue
            if chosen is not None and file in chosen:
                files[file] = (chosen[file], self._memory().read(file))
            else:
                files[file] = (text, over)
        reports: dict[str, list[Report]] = {}
        for place in places:
            if place.team_id not in waits and place.id not in left:
                reports[place.team_id] = self.own_reports(place)
        if review is not None:
            # What its reviewers recorded is the team's they review from.
            at = (
                review.team_id
                if self.is_orgs(review)
                else self._home_of(top, review.boundary.path)
            )
            reports[at] = sorted(
                [*reports.get(at, []), *self.own_reports(review)],
                key=lambda report: report.at,
            )
        not_taken = [
            FormerTeam(
                id=place.team_id,
                title=place.title,
                above=self._home_of(top, place.boundary.path),
                path=place.boundary.path,
                purpose=run.marked(TASK).get(place.id, place.spec_summary),
                ended="not_taken",
                at=stamp,
                review=place.review.state if place.review else None,
                reports=self.own_reports(place),
                compacted_report="\n\n".join(r.text for r in self.own_reports(place)),
                contracts=of_left.get(place.id, {}).get("contracts", ""),
                spec=of_left.get(place.id, {}).get("spec", ""),
                session=store.session_title(place),
                since=place.since,
            )
            for place in left.values()
            if self.own_reports(place) or of_left.get(place.id)
        ]
        removed = [
            record for record in self.former_drafts(session) if record.id not in waits
        ]
        former, forget = [], []
        for record in removed:
            team = store.team(record.id)
            if team is not None and team.since == record.since:
                # The org kept the team the session removed.
                former.append(record.model_copy(update={"ended": "not_taken"}))
            else:
                former.append(record)
                ours = next((p for p in places if p.team_id == record.id), None)
                if team is None or (ours is not None and ours.since == team.since):
                    forget.append(record.id)
        of_team = {place.team_id: place for place in places}
        events: dict[str, list[TeamEvent]] = {}
        kept_events: list[str] = []
        for key, raw in run.marked(EVENTS).items():
            if not key.startswith(f"{session}\n"):
                continue
            team_id = key.partition("\n")[2]
            place = of_team.get(team_id)
            if team_id in waits:
                continue
            if place is not None and place.id in left:
                team_id = self._home_of(top, place.boundary.path)
            elif store.team(team_id) is None:
                team_id = self._home_of(top, top.boundary.path)
            events.setdefault(team_id, []).extend(
                event.model_copy(update={"unaccepted": False})
                for event in _EVENTS.validate_json(raw)
            )
            kept_events.append(key)
        for record in not_taken:
            place = next(p for p in left.values() if p.team_id == record.id)
            events.setdefault(record.above, []).append(
                self._event(
                    "not_taken",
                    place,
                    "what its session wrote for it is kept in its record",
                ).model_copy(update={"unaccepted": False})
            )
        routed = {
            key: RoutedRequest.model_validate_json(raw)
            for key, raw in run.marked(ROUTED).items()
            if key.startswith(f"{session}\n")
        }
        conflicts = self._commit_landing(
            Landing(
                files=files,
                seen=seen or {},
                reports={team: kept for team, kept in reports.items() if kept},
                former=[*former, *not_taken],
                forget=forget,
                events=events,
                routed=list(routed.values()),
                members=self.member_names(),
            ),
            f"memory: the accepted work of session {session}",
            tree,
        )
        for key in kept_events:
            run.unmark(EVENTS, key)
        for key in routed:
            run.unmark(ROUTED, key)
        if review is not None:
            run.unmark(REPORTS, review.id)
        for file in files:
            if file in conflicts:
                continue
            self._undraft(session, file)
            if file in passed_on:
                lost = passed_on[file]
                home = store.team(self._home_of(top, lost.boundary.path))
                rewritten = (
                    _spec_file(normalize_store_path(home.boundary.path)) if home else ""
                )
                directory = _directory_of(file)
                if (
                    home is not None
                    and not (
                        files.get(rewritten, (None,))[0] and rewritten not in conflicts
                    )
                    and all(old.path != directory for old in home.old_specs)
                ):
                    store.update_team(
                        home.id,
                        old_specs=[
                            *home.old_specs,
                            OldSpec(path=directory, team=lost.title),
                        ],
                    )
                continue
            owner = by_path.get(_directory_of(file))
            if (
                owner is not None
                and file.endswith(SPEC_FILE)
                and files[file][0] is not None
            ):
                store.update_team(
                    owner.team_id, spec_summary=owner.spec_summary, old_specs=[]
                )
        for place in left.values():
            for file in self.drafts(session):
                # What waits for the org's part of the landing stays a draft.
                if by_path.get(_directory_of(file)) is place and file not in held:
                    self._undraft(session, file)
        for place in places:
            if place.team_id not in waits:
                run.unmark(REPORTS, place.id)
        for record in removed:
            run.unmark(FORMER, f"{session}\n{record.id}\n{record.at}")
        grown: list[tuple[str, str]] = [
            (team, role)
            for team, kept in reports.items()
            for role in dict.fromkeys(report.role for report in kept)
        ]
        return Taken(grown, sorted(conflicts))

    def _leave(self, top: SessionTeam) -> str:
        """Drop what a session that is being closed kept to itself: what
        it wrote for work that was never accepted goes with it.

        Returns:
            What went with the session, in words; '' when nothing did.
        """
        store = self.org
        run = self._runs()
        session = top.session
        places = store.session(session)
        reports = sum(len(self.own_reports(place)) for place in places)
        files = sorted(self.drafts(session))
        records = self.former_drafts(session)
        for place in places:
            run.unmark(REPORTS, place.id)
            self._drop_merge_checkout(place.id)
        for file in files:
            self._undraft(session, file)
        for record in records:
            run.unmark(FORMER, f"{session}\n{record.id}\n{record.at}")
        for kind in (EVENTS, ROUTED):
            for key in run.marked(kind):
                if key.startswith(f"{session}\n"):
                    run.unmark(kind, key)
        for place in places:
            self._take_threads(place.id)
        run.unmark(USE, session)
        return ", ".join(
            said
            for said in (
                f"{reports} reports" if reports else "",
                ", ".join(files),
                f"the records of {len(records)} teams it removed" if records else "",
            )
            if said
        )

    def _specs_left_in(
        self, place: SessionTeam, path: str, removed: list[SessionTeam]
    ) -> list[OldSpec]:
        """The specs a team that is given ``path`` in the session of
        ``node`` finds there: the one of the directory itself and those of
        the ``removed`` teams whose directories hold it or lie in it. None
        of them is its own."""
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
            if self.read_file(place, _spec_file(directory))
        ]

    def add_to_contracts(self, place: SessionTeam, text: str) -> None:
        """Add to the contracts between the sub-teams of ``place`` what its
        IC found they may all rely on. It is the session's own until its
        work is accepted, and the sub-teams read it from then."""
        self._record_contracts(place, text, replace=False)

    def own_contracts(self, place: SessionTeam) -> str:
        """The contracts between the sub-teams of ``place``, as its session
        has them; '' when there are none."""
        file = _contracts_file(normalize_store_path(place.boundary.path))
        return self.read_file(place, file) or ""

    def set_contracts(self, place: SessionTeam, text: str) -> None:
        """Put ``text`` in the place of the contracts between the sub-teams
        of ``place``, for its session until its work is accepted."""
        self._record_contracts(place, text, replace=True)

    def _record_contracts(
        self, place: SessionTeam, contracts: str, replace: bool
    ) -> None:
        """Add the contracts of a split to those of the team's boundary,
        or put them in their place when the split replaced teams. They are
        the session's alone until its work is accepted."""
        file = _contracts_file(normalize_store_path(place.boundary.path))
        earlier = self.read_file(place, file)
        text = f"{contracts.strip()}\n"
        if replace:
            if earlier != text:
                self._draft(place.session, file, text)
            return
        if earlier and text.strip() in earlier:
            return
        self._draft(
            place.session,
            file,
            f"{earlier.rstrip()}\n\n{text}" if earlier and earlier.strip() else text,
        )
