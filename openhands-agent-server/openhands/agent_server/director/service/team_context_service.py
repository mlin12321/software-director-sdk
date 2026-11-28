"""What an agent reads of its place: who it is, everything of the org
that concerns its team, and the facts a proposal is weighed by."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openhands.agent_server.director.documents.schema import (
    ShownDocument,
)
from openhands.agent_server.director.documents.store import DocumentRef, DocumentStore
from openhands.agent_server.director.environment.checkouts import Checkouts
from openhands.agent_server.director.environment.records import (
    team_key,
)
from openhands.agent_server.director.models import (
    ConversationLink,
    ConversationPlace,
    ConversationRelation,
    OrgNode,
    SessionTeam,
    SplitProposal,
    Thread,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    is_division_lead,
    session_of,
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
from openhands.agent_server.director.run_state import (
    JOINED,
)
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
    thread_name,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.sdk.errors import NotFound


# How a team that was set up and given no task reads to the IC above.
NOT_STARTED = "not_started"
# How many of the human's direct requests to the teams below an IC is told.
ASKED_TOLD = 10
# How many of the latest events of its sub-teams an agent reads at once.
HISTORY_TOLD = 30
# How many commits must have changed two proposed directories together
# before a Manager is told: a first commit of everything says nothing.
TOGETHER_WORTH_SAYING = 3
# How many earlier declines for the same directories a proposal is shown with.
DECLINED_TOLD = 3


@dataclass
class TeamContextService:
    """Builds what each agent is told of its team and the teams around it.

    It only reads: the tree, the documents, the history of the teams and
    what the sessions keep. All of it is safe from any thread.
    """

    org: Callable[[], OrgStore]
    """The org whose teams are read."""
    checkouts: Checkouts
    """Where each team's work is and how it moves."""
    org_state: OrgStateService
    """The run state, for what the human joined."""
    org_settings: OrgSettingsService
    """The settings of the org and the types its teams may be."""
    team_types: TeamTypeService
    """Who the members of a team are and which documents it keeps."""
    team_history: TeamHistoryService
    """What became of the teams below a team."""
    session_drafts: SessionDraftService
    """What a session keeps until its work is accepted."""
    documents: DocumentStore
    """Where documents are read."""

    def asked_below(self, place: SessionTeam) -> list[dict[str, str]]:
        """The latest of what the human asked directly of the teams below
        a team, oldest first."""
        store = self.org()
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

    def so_far(self, team: SessionTeam) -> str:
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
        sizes = [
            (child, self.checkouts.file_count(checkout, child.path))
            for child in children
        ]
        together = [
            (one.path, other.path, count)
            for index, one in enumerate(children)
            for other in children[index + 1 :]
            if (
                count := self.checkouts.changed_together(checkout, one.path, other.path)
            )
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

    def place_of(self, conversation_id: str) -> ConversationPlace:
        """Where an agent's conversation is in the org, with the
        conversations one level from it.

        Raises:
            NotFound: No agent of the org has this conversation.
        """
        store = self.org()
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
        down = [(place.id, team.id) for team in self.org().below(place.id)]
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
        store = self.org()
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
            **self.org_settings.team_documents().told(self.team_types.reader_of(place)),
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
                for old in self.org_settings.team_documents().left_to(
                    self.team_types.reader_of(place)
                )
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
