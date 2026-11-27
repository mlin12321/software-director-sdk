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
from openhands.agent_server.director.documents.events import (
    SubTeamsProposed,
    TeamRemoved,
)
from openhands.agent_server.director.documents.protocol import (
    DocumentProtocolBase,
    DocumentsBriefed,
    ShownDocument,
)
from openhands.agent_server.director.documents.store import DocumentRef, TeamRef
from openhands.agent_server.director.git import (
    changed_together,
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
    LeftoverDocument,
    OrgHistory,
    OrgNode,
    Report,
    RoutedRequest,
    SessionTeam,
    SplitProposal,
    TeamDocument,
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
    left_document,
    need_words,
    needs_between,
    proposal_facts,
    record_line,
    rules_said,
    track_records,
)
from openhands.agent_server.director.recovery import STEP_NAMES
from openhands.agent_server.director.run_state import (
    CHECKS,
    EVENTS,
    FORMER,
    HANDED,
    JOINED,
    REPORTS,
    ROUTED,
    SENT,
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
    _instant,
    _spec_file,
    team_ref,
)
from openhands.agent_server.director.split import (
    home_of_lost,
    is_inside,
)
from openhands.sdk.errors import NotFound, NotSupported
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)
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


# The events of a session that change what the org has of its teams.
_CHANGES_THE_ORG: frozenset[TeamEventKind] = frozenset(
    {"set_up", "replaced", "pruned", "moved"}
)


# The document a team's ``leftover_documents`` are what was left of, until what is
# left behind carries the name of its document.
LEFT_BEHIND = "spec"
# The one log the director keeps: what a team's members recorded.
REPORT = "report"
# The name of the document the wire still calls a team's contracts.
CONTRACTS = "contracts"


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
                forget_reports=[],
                events={team_id: [event.model_copy(update={"unaccepted": False})]},
                routed=[],
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
        specs = self.specs_of(place)
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
            **self.documents_told(place),
            "documents_passed_to_you": [
                {
                    "document": passed.name,
                    "of": passed.folder or "the repository",
                    "text": passed.text,
                }
                for passed in self.documents_passed_to(place)
            ],
            "specs_left_by_teams_before_you": [
                {
                    "directory": spec.path or "the repository",
                    "team": spec.left_by,
                    "spec": spec.spec,
                }
                for spec in specs
                if spec.old
            ],
            "sub_teams": [
                {
                    "team_id": child.id,
                    "team": child.title,
                    "path": child.boundary.path,
                    "status": child.status if child.agents else NOT_STARTED,
                    "headline": child.headline,
                    "summary": child.summary,
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
        works = role == self.team(place).writer
        if works and store.below(place.id):
            context["asked_directly_of_teams_below"] = self.asked_below(place)
        if works and (handed := self.handed_over_to(place)):
            context["handed_over_to_your_team"] = handed
            # It is the team's to write into its own documents.
            todo = rules_said(
                self.document_rules(place),
                "handed_over",
                self.team(place).names.words(),
            )
            if todo:
                context["what_to_do_with_what_was_handed_over"] = " ".join(todo)
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
                        "documents": former.documents,
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
        reports = team_memory.reports(node.boundary.path) if self.is_orgs(node) else []
        return sorted([*reports, *self.own_reports(node)], key=lambda report: report.at)

    def own_reports(self, node: OrgNode) -> list[Report]:
        """What the agents of a session's place recorded since the session's
        work was last accepted. It is the session's alone until its work
        is accepted again, and the team's from then on. Nothing for a team
        of the org."""
        kept = self._runs().marked(REPORTS).get(node.id)
        return _REPORTS.validate_json(kept) if kept else []

    def entries(self, team: TeamRef, name: str) -> list[Report]:
        """The one log the director keeps for a team, as the store of
        documents is asked for it: what its members recorded."""
        place = self.org.place(team.id)
        return self.reports_of(place) if place and name == REPORT else []

    def append(self, team: TeamRef, name: str, entry: Report) -> None:
        """Add to what the members of a team recorded, as the store of
        documents is asked to.

        Raises:
            NotSupported: It is another log than the one the director keeps.
            NotFound: The team is no place of a session.
        """
        if name != REPORT:
            raise NotSupported(f"The director keeps no log named {name!r}")
        place = self.org.place(team.id)
        if place is None:
            raise NotFound(f"There is no team {team.id}")
        said_where = {
            "session": self.org.session_title(place),
            "session_id": session_of(place),
        }
        # Its session's alone until the session's work is accepted.
        self._keep_report(
            place, entry.model_copy(update={"at": entry.at or _instant(), **said_where})
        )

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
        return {
            ref.path: (draft.text, draft.over)
            for ref, draft in self.documents.drafts(session).items()
        }

    def _draft(self, session: str, file: str, text: str | None) -> None:
        """Keep, for a session alone, what it makes of a file of team
        memory."""
        _ = self.team_memory  # refuses where the org has no team memory
        ref = DocumentRef.at(file)
        if text is None:
            self.documents.remove(session, ref)
        else:
            self.documents.write(session, ref, text)

    def _document_changed(self, session: str) -> None:
        self._note_change(session)

    def _undraft(self, session: str, file: str) -> None:
        self.documents.discard(session, DocumentRef.at(file))

    def read_file(self, node: OrgNode, file: str) -> str | None:
        """A file of team memory as the agents of ``node`` have it: what
        their session wrote there, else the org's."""
        return self.documents.read(session_of(node), DocumentRef.at(file))

    def contracts_of(self, node: OrgNode) -> list[BoundaryContracts]:
        """The contracts a team works to: the ones between it and the teams
        beside it, set by the team above, and the ones between its own
        sub-teams."""
        return [
            BoundaryContracts(path=shown.folder, contracts=shown.text, own=shown.own)
            for shown in self.documents_shown(node)
            if shown.name == CONTRACTS
        ]

    def documents_told(self, node: OrgNode) -> dict[str, Any]:
        """What the context of a team's agents says of the documents it
        reads, as each document has it said: the ones its type keeps, in
        the order the type lists them, and any other it is shown one of."""
        shown = self.documents_shown(node)
        kept = self.protocols_of(node)
        told: dict[str, Any] = {}
        for protocol in self._protocols_for(node):
            name = protocol.document.name
            of_it = [found for found in shown if found.name == name]
            if of_it or protocol in kept:
                told |= protocol.told_to_reader(team_ref(node), of_it)
        return told

    def documents_briefed(self, place: SessionTeam) -> list[tuple[str, str]]:
        """What the first message of a team's agents says of the documents
        it reads, as each document has it said, and after each what teams
        before it left of that document in its folder."""
        shown = self.documents_shown(place)
        kept = self.protocols_of(place)
        team = self.team(place)
        briefed = [
            DocumentsBriefed(
                reader=team_ref(place),
                shown=[found for found in shown if found.name == name],
                titles=team.kind.titles,
            )
            for name in (
                protocol.document.name for protocol in self._protocols_for(place)
            )
        ]
        left = [found for found in self.documents_of(place) if found.left]
        parts: list[tuple[str, str]] = []
        for protocol, told in zip(self._protocols_for(place), briefed):
            document = protocol.document
            if told.shown or protocol in kept:
                parts += [
                    (part.heading, part.text)
                    for part in protocol.told_in_briefing(told)
                ]
            parts += [
                (
                    left_document(
                        document.name, old.path, old.left_by, document.written_with
                    ),
                    old.text,
                )
                for old in left
                if old.name == document.name
            ]
        return parts

    def handed_over_to(self, node: OrgNode) -> list[dict[str, str]]:
        """What the team above said to this team of the teams that were
        there before it: who said it, when, and the text."""
        kept = self._runs().marked(HANDED).get(node.id)
        return json.loads(kept) if kept else []

    def documents_passed_to(self, node: OrgNode) -> list[ShownDocument]:
        """The documents another team passed to this one to read, as they
        are now in the session they were passed in."""
        kept = self._runs().marked(SENT).get(node.id)
        found = []
        for folder, name in json.loads(kept) if kept else []:
            text = self.documents.read(session_of(node), DocumentRef(folder, name))
            if text:
                found.append(
                    ShownDocument(folder=folder, name=name, text=text, own=False)
                )
        return found

    def _protocols_for(self, node: OrgNode) -> list[DocumentProtocolBase]:
        """What every document of the org does: first the ones the team's
        type keeps, in the order the type lists them."""
        kept = self.protocols_of(node)
        return [
            *kept,
            *(found for found in self.protocols.values() if found not in kept),
        ]

    def documents_of(self, node: OrgNode) -> list[TeamDocument]:
        """The documents a team reads, for whoever looks at the team: the
        ones it is shown, and with them what teams before it left in its
        folder."""
        by_type = self.type_of(node).documents
        shown = self.documents_shown(node)
        left = [spec for spec in self.specs_of(node) if spec.old]
        found: list[TeamDocument] = []
        for protocol in self._protocols_for(node):
            document = protocol.document
            said = by_type.get(document.name) or document.description
            found += [
                TeamDocument(
                    name=one.name,
                    description=said,
                    path=one.folder,
                    text=one.text,
                    own=one.own,
                )
                for one in shown
                if one.name == document.name
            ]
            if document.name == LEFT_BEHIND:
                found += [
                    TeamDocument(
                        name=document.name,
                        description=said,
                        path=spec.path,
                        text=spec.spec,
                        own=False,
                        left=True,
                        left_by=spec.left_by,
                    )
                    for spec in left
                ]
        return found

    def documents_shown(self, node: OrgNode) -> list[ShownDocument]:
        """Every document a team reads, as its session has it: of each
        kind of document, the ones of the teams its protocol names, where
        the org allows it."""
        reader = team_ref(node)
        left = {normalize_store_path(old.path) for old in node.leftover_documents}
        found: dict[DocumentRef, ShownDocument] = {}
        for protocol in self._protocols_for(node):
            name = protocol.document.name
            for owner in protocol.read_from(reader):
                ref = owner.keeps(name)
                # A session opened at the top team has that team's own
                # place above it: one folder, shown once.
                if ref in found or not self.org_kind.allows_read(reader, owner, name):
                    continue
                # What a team before this one left in the folder is not
                # this team's own until it writes one itself.
                if name == LEFT_BEHIND and ref.folder in left:
                    continue
                text = self.documents.read(reader.session, ref)
                if text:
                    found[ref] = ShownDocument(
                        folder=ref.folder,
                        name=name,
                        text=text,
                        own=ref.folder == reader.folder,
                    )
        return list(found.values())

    def specs_of(self, node: OrgNode) -> list[InheritedSpec]:
        """The spec of a team's own boundary and the ones teams before it
        left in its directory, which are old."""
        own = normalize_store_path(node.boundary.path)
        text = self.read_file(node, _spec_file(own))
        specs = (
            [] if text is None else [InheritedSpec(path=own, spec=text, readonly=False)]
        )
        left = {old.path: old for old in node.leftover_documents}
        found = [
            spec.model_copy(
                update={"old": spec.path in left, "left_by": left[spec.path].team}
            )
            if spec.path in left
            else spec
            for spec in specs
        ]
        for old in node.leftover_documents:
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
                kind = DocumentRef.at(file).name
                kept_of_it = of_left.setdefault(owner.id, {})
                if text:
                    kept_of_it[kind] = text
                if (
                    kind == LEFT_BEHIND
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
        # By the folder of each team, where what it recorded is kept.
        reports: dict[str, list[Report]] = {}
        for place in places:
            if place.team_id not in waits and place.id not in left:
                at = normalize_store_path(place.boundary.path)
                reports[at] = [*reports.get(at, []), *self.own_reports(place)]
        if review is not None:
            # What its reviewers recorded is the team's they review from.
            home = (
                review
                if self.is_orgs(review)
                else store.team(self._home_of(top, review.boundary.path))
            )
            at = normalize_store_path((home or review).boundary.path)
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
                purpose=run.marked(TASK).get(place.id, place.summary),
                ended="not_taken",
                at=stamp,
                review=place.review.state if place.review else None,
                reports=self.own_reports(place),
                compacted_report="\n\n".join(r.text for r in self.own_reports(place)),
                documents=of_left.get(place.id, {}),
                session=store.session_title(place),
                since=place.since,
            )
            for place in left.values()
            if self.own_reports(place) or place.id in of_left
        ]
        removed = [
            record for record in self.former_drafts(session) if record.id not in waits
        ]
        former, forget, forget_reports = [], [], []
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
                    forget_reports.append(record.path)
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
                forget_reports=forget_reports,
                events=events,
                routed=list(routed.values()),
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
            run.unmark(SENT, review.id)
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
                    and all(old.path != directory for old in home.leftover_documents)
                ):
                    store.update_team(
                        home.id,
                        leftover_documents=[
                            *home.leftover_documents,
                            LeftoverDocument(
                                name=LEFT_BEHIND, path=directory, team=lost.title
                            ),
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
                    owner.team_id, summary=owner.summary, leftover_documents=[]
                )
        for place in left.values():
            for file in self.drafts(session):
                # What waits for the org's part of the landing stays a draft.
                if by_path.get(_directory_of(file)) is place and file not in held:
                    self._undraft(session, file)
        for place in places:
            if place.team_id not in waits:
                run.unmark(REPORTS, place.id)
                run.unmark(SENT, place.id)
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
            run.unmark(SENT, place.id)
            run.unmark(HANDED, place.id)
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
    ) -> list[LeftoverDocument]:
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
            for old in team.leftover_documents:
                if concerns(old.path):
                    left.setdefault(old.path, old.team)
            theirs = normalize_store_path(team.boundary.path)
            if concerns(theirs):
                left[theirs] = team.title
        # A spec in the directory itself that no team here is known to have
        # left: a team pruned long ago, or one written by hand.
        left.setdefault(own, None)
        return [
            LeftoverDocument(name=LEFT_BEHIND, path=directory, team=team)
            for directory, team in left.items()
            if self.read_file(place, _spec_file(directory))
        ]

    def documents_sub_teams_read(self, place: SessionTeam) -> list[tuple[str, str]]:
        """The documents of ``place`` that every team directly below it
        reads, as its session has them: the name and the text of each
        that says something."""
        team = team_ref(place)
        found = []
        for protocol in self.protocols_of(place):
            if not protocol.read_by_sub_teams:
                continue
            name = protocol.document.name
            text = self.documents.read(team.session, team.keeps(name))
            if text and text.strip():
                found.append((name, text.strip()))
        return found

    def before_sub_teams_are_proposed(self, place: SessionTeam) -> None:
        """Ask the documents of ``place`` whether its writer may propose
        sub-teams now.

        Raises:
            Invalid: One of them is not ready, with what to do first.
        """
        proposed = SubTeamsProposed(team_ref(place))
        for protocol in self.protocols_of(place):
            protocol.before_sub_teams_are_proposed(proposed)

    def removed_team_handed_over(
        self, place: SessionTeam, above: SessionTeam, member: str, text: str
    ) -> None:
        """Keep what a member of the team above says to ``place`` of the
        teams that were there before it. The team is told with its next
        message, or in its briefing, and no document is written by this.
        Safe from any thread."""
        with self._keeping:
            run = self._runs()
            kept = run.marked(HANDED).get(place.id)
            said = json.loads(kept) if kept else []
            entry = {
                "by": f"the {self.team(above).title(member)} of {above.title}",
                "at": utc_now(),
                "text": text,
            }
            run.mark(HANDED, place.id, json.dumps([*said, entry]))

    def team_removed(self, place: SessionTeam) -> dict[str, str]:
        """Tell the documents of ``place`` that it is gone.

        Returns:
            What is kept of each in the team's record, by its name.
        """
        removed = TeamRemoved(team_ref(place))
        kept = {
            protocol.document.name: protocol.when_team_is_removed(removed)
            for protocol in self.protocols_of(place)
        }
        return {name: text for name, text in kept.items() if text}
