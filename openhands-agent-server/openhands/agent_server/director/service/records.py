"""What the service reads and keeps of the teams: their context, their
records and what they needed of each other."""

import json
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

from openhands.agent_server.director.documents.schema import (
    ShownDocument,
)
from openhands.agent_server.director.documents.store import REPORT, DocumentRef, TeamRef
from openhands.agent_server.director.documents.team_documents import RemovedTeam
from openhands.agent_server.director.git import (
    changed_together,
    tracked_files,
)
from openhands.agent_server.director.git_store import (
    normalize_store_path,
    team_key,
)
from openhands.agent_server.director.models import (
    ConversationLink,
    ConversationPlace,
    ConversationRelation,
    LeftoverDocument,
    OrgNode,
    Report,
    SessionTeam,
    SplitProposal,
    Thread,
)
from openhands.agent_server.director.org_store import (
    is_division_lead,
    session_of,
    team_ref,
    utc_now,
)
from openhands.agent_server.director.prompts import (
    NEEDED_WORTH_SAYING,
    event_line,
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
    JOINED,
    USE,
    Intent,
)
from openhands.agent_server.director.service.session_draft_service import (
    thread_name,
)
from openhands.agent_server.director.service.session_landing_service import (
    SessionLandingService,
)
from openhands.agent_server.director.service.state import (
    _REASON_KEPT,
    ASKED_TOLD,
    DECLINED_TOLD,
    HISTORY_TOLD,
    NOT_STARTED,
    TOGETHER_WORTH_SAYING,
    ServiceState,
    _instant,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from openhands.sdk.errors import NotFound, NotSupported
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


def _directory_of(file: str) -> str:
    """The folder a document's file of team memory is of."""
    return file.rpartition("/")[0]


class Inherited(NamedTuple):
    """A document a session wrote for a team the org no longer has."""

    name: str
    # The folder of the team of the org that holds the lost team's now.
    home: str
    # That team's own document of the name, as the org has it.
    now: str
    # What the lost team was called.
    lost: str
    written: str


class Records(ServiceState):
    """Reads and keeps what is known of the teams."""

    @cached_property
    def session_landing(self) -> SessionLandingService:
        """What makes a session's records the org's once its work is
        accepted."""
        return SessionLandingService(
            org=lambda: self.org,
            run_state=self.org_state.run_state,
            team_memory=self._memory,
            documents=self.documents,
            team_types=self.team_types,
            session_drafts=self.session_drafts,
            org_save=self.org_save,
            event=self.team_history.event,
        )

    @cached_property
    def team_history(self) -> TeamHistoryService:
        """The history of the org's teams."""
        return TeamHistoryService(
            org=lambda: self.org,
            team_memory=self._memory,
            require_team_memory=lambda: self.team_memory,
            session_drafts=self.session_drafts,
            org_changed=self._note_change,
        )

    def _keep_failure(self, intent: Intent, error: Exception) -> None:
        """Keep a step that failed where the team above and the human read
        what became of a team."""
        team = self.org.place(intent.node_id)
        if team is None:
            return
        what = STEP_NAMES.get(intent.kind, intent.kind)
        self.team_history.record_event(
            team.parent_id or team.id,
            "failed",
            team,
            f"{what}: not done. {type(error).__name__}: {error}"[:_REASON_KEPT],
        )

    def _note_change(self, session: str) -> None:
        """Remember that a session whose work is accepted has changed
        something since: the org takes it only when it is submitted again."""
        top = next(iter(self.org.session(session)), None)
        if top is not None and top.review is not None:
            if top.review.state == "accepted" and self.org.opens_session(top):
                self._changed_since_accept.setdefault(session, False)

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
            for request in self.team_history.routed_requests()
            if request.kind == "asked" and request.team_id in below
        ][-ASKED_TOLD:]

    def asked_of(self, team: SessionTeam) -> list[str]:
        """What the human asked a team directly since its work was last
        accepted, by the order of the record of the team above."""
        if team.parent_id is None:
            return []
        asked: list[str] = []
        for event in self.team_history.history_of(team.parent_id):
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
            for pair in needs_between(self.team_history.history_of(node_id))
            if pair["writes_refused"] + pair["review_findings"] >= NEEDED_WORTH_SAYING
        ]

    def _so_far(self, team: SessionTeam) -> str:
        """What became of a team's work before, in words, as the team above
        has it on record."""
        if team.parent_id is None:
            return ""
        records = track_records(self.team_history.history_of(team.parent_id))
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
        history = self.team_history.history_of(place.id)
        why = {
            (event.team_id, event.at): event.detail
            for event in history
            if event.kind == "pruned"
        }
        paths = {child.path for child in children}
        before = []
        for former in self.team_history.former_teams_of(place):
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
            names=self.team_types.team(place).names,
        )

    def _work_accepted(self, team: SessionTeam) -> None:
        """A team's work was accepted: what its IC escalated on it is
        counted no further, since what it does next is other work."""
        if team.parent_id is None:
            return
        with self._counting:
            counted = self._counted(team.parent_id)
            if counted["escalations"].pop(team.id, None) is not None:
                self.org_state.run_state().mark(
                    CHECKS, team.parent_id, json.dumps(counted)
                )

    def _counted(self, place_id: str) -> dict[str, Any]:
        """What a Manager's checks have counted: the reviews its IC
        accepted, and the escalations of each team below on its work."""
        try:
            kept = json.loads(
                self.org_state.run_state().marked(CHECKS).get(place_id) or "{}"
            )
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
            human_joined=conversation_id in self.org_state.run_state().marked(JOINED),
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
                    or agent.role == self.team_types.team(team).writer
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
            if (
                messages := self.session_drafts.thread(
                    thread_name(writer, ic, ic_to_ic)
                )
            )
        ]

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
            **self.team_documents.told(self.team_types.reader_of(place)),
            "documents_passed_to_you": [
                {
                    "document": passed.name,
                    "of": passed.folder or "the repository",
                    "text": passed.text,
                }
                for passed in self.documents_passed_to(place)
            ],
            "documents_left_by_teams_before_you": [
                {
                    "document": old.name,
                    "directory": old.path or "the repository",
                    "team": old.left_by,
                    "text": old.text,
                }
                for old in self.team_documents.left_to(self.team_types.reader_of(place))
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
            "agent_profiles_a_sub_team_may_run_on": (
                self.org_settings.settings.allowed_profiles
            ),
            "types_a_sub_team_may_be": {
                kind.name: kind.description
                + (
                    f" Runs on the agent profile {kind.sas_profile!r} unless "
                    "you name another."
                    if kind.sas_profile
                    else ""
                )
                for kind in self.org_settings.types_allowed()
            },
            "type_of_a_sub_team_that_names_none": self.org_settings.default_type(),
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
        works = role == self.team_types.team(place).writer
        if works and store.below(place.id):
            context["asked_directly_of_teams_below"] = self.asked_below(place)
        if works and (handed := self.session_drafts.handed_over_to(place.id)):
            context["handed_over_to_your_team"] = handed
            # It is the team's to write into its own documents.
            todo = rules_said(
                self.team_types.document_rules(place),
                "handed_over",
                self.team_types.team(place).names.words(),
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
                for former in self.team_history.former_teams_of(place)
            ]
        if in_full:
            # What became of the teams below so far, also of the ones that
            # are gone: for both agents to weigh what they do next by.
            events = self.team_history.history_of(place.id)
            context["track_record_of_the_teams_below"] = track_records(events)
            needed = [need_words(pair) for pair in self.needs_below(place.id)]
            if needed:
                context["teams_below_that_needed_each_other"] = needed
            context["what_became_of_them_lately"] = [
                event_line(event) for event in events[-HISTORY_TOLD:]
            ]
        return context

    def reports_of(self, node: OrgNode) -> list[Report]:
        """What the two agents of a team recorded, oldest first. Reports
        are the team's: an agent started again reads what it wrote before."""
        team_memory = self._memory()
        reports = (
            team_memory.reports(node.boundary.path) if self.org.is_orgs(node) else []
        )
        own = self.session_drafts.reports(node.id)
        return sorted([*reports, *own], key=lambda report: report.at)

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
        self.session_drafts.keep_report(
            place.id,
            entry.model_copy(update={"at": entry.at or _instant(), **said_where}),
        )

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

    def documents_passed_to(self, node: OrgNode) -> list[ShownDocument]:
        """The documents another team passed to this one to read, as they
        are now in the session they were passed in."""
        found = []
        for folder, name in self.session_drafts.passed_to(node.id):
            text = self.documents.read(session_of(node), DocumentRef(folder, name))
            if text:
                found.append(
                    ShownDocument(folder=folder, name=name, text=text, own=False)
                )
        return found

    def _documents_left_in(
        self, place: SessionTeam, path: str, removed: list[SessionTeam]
    ) -> list[LeftoverDocument]:
        """The documents a team that is given ``path`` in the session of
        ``place`` finds there, with the ``removed`` teams gone."""
        return self.team_documents.left_in(
            place.session,
            path,
            [
                RemovedTeam(team.boundary.path, team.title, team.leftover_documents)
                for team in removed
            ],
        )

    def documents_inherited(self, top: SessionTeam) -> list[Inherited]:
        """The documents a session wrote for teams the org had when it
        began and has no more, of the kind that stays when a team is
        removed: each is left with the team that holds the folder now."""
        store = self.org
        session = store.session_of(top)
        if session is None:
            return []
        began = {team for team, *_ in session.shape}
        memory = self._memory()
        found = []
        for place in store.session(top.session):
            if (
                is_division_lead(place)
                or place.team_id not in began
                or store.team(place.team_id) is not None
            ):
                continue
            home = store.team(self.session_landing.home_of(top, place.boundary.path))
            if home is None:
                continue
            path = normalize_store_path(home.boundary.path)
            documents = self.team_documents
            for name, protocol in documents.protocols.items():
                draft = documents.draft(team_ref(place), name)
                if draft is None or not protocol.stays_when_team_is_removed:
                    continue
                now = memory.read(DocumentRef(path, name).path) or ""
                found.append(Inherited(name, path, now, place.title, draft[0]))
        return found

    def _leave(self, top: SessionTeam) -> str:
        """Drop what a session that is being closed kept to itself: what
        it wrote for work that was never accepted goes with it.

        Returns:
            What went with the session, in words; '' when nothing did.
        """
        store = self.org
        run = self.org_state.run_state()
        session = top.session
        places = store.session(session)
        reports = sum(len(self.session_drafts.reports(place.id)) for place in places)
        files = sorted(self.drafts(session))
        records = self.session_drafts.former(session)
        for place in places:
            self.session_drafts.forget_reports(place.id)
            self.session_drafts.forget_passed(place.id)
            self.session_drafts.forget_handed_over(place.id)
            self._drop_merge_checkout(place.id)
        for file in files:
            self._undraft(session, file)
        self.session_drafts.forget_session(session)
        for place in places:
            self.session_drafts.take_threads(place.id)
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

    def removed_team_handed_over(
        self, place: SessionTeam, above: SessionTeam, member: str, text: str
    ) -> None:
        """Keep what a member of the team above says to ``place`` of the
        teams that were there before it. The team is told with its next
        message, or in its briefing, and no document is written by this.
        Safe from any thread."""
        self.session_drafts.keep_handed_over(
            place.id,
            {
                "by": (
                    f"the {self.team_types.team(above).title(member)} of {above.title}"
                ),
                "at": utc_now(),
                "text": text,
            },
        )
