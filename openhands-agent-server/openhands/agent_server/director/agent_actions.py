"""What the agents of an org can do to it, behind their tools.

Every function names the agent that acts and refuses what its role or its
team does not allow, so a tool cannot reach past its own team.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel

from openhands.agent_server.director.boundary import owns
from openhands.agent_server.director.documents.schema import DocumentToolCall
from openhands.agent_server.director.environment.records import team_key
from openhands.agent_server.director.models import (
    DocumentChoice,
    EscalationKind,
    Review,
    ReviewFinding,
    SessionTeam,
    SplitProposal,
    TeamChoice,
)
from openhands.agent_server.director.org_store import (
    is_division_lead,
    team_ref,
    utc_now,
)
from openhands.agent_server.director.prompts import role_text
from openhands.agent_server.director.service import (
    DirectorService,
)
from openhands.agent_server.director.split import (
    Proposed,
    ProposedTeam,
    as_written,
    check_limits,
    normalize_split,
    proposed_teams,
)
from openhands.sdk.errors import Conflict, Invalid, NotFound


@dataclass(frozen=True)
class Caller:
    """The agent that uses a tool."""

    node: SessionTeam
    role: str
    # Whether it is an IC that calls from its hand-over conversation.
    hand_over: bool = False


def caller_of(service: DirectorService, conversation_id: UUID) -> Caller:
    """The agent whose conversation is ``conversation_id``, as it is now.

    Raises:
        NotFound: The conversation is not an agent of the org.
    """
    found = service.org_load.org.find_by_conversation(str(conversation_id))
    if found is not None:
        return Caller(node=found[0], role=found[1])
    node = service.org_load.org.find_hand_over(str(conversation_id))
    if node is None:
        raise NotFound("This conversation is not an agent of the org")
    return Caller(node=node, role=service.team_types.team(node).writer, hand_over=True)


def acting(service: DirectorService, conversation_id: UUID) -> Caller:
    """The agent that uses a tool now, known to be at work from here on.

    Raises:
        NotFound: The conversation is not an agent of the org.
    """
    caller = caller_of(service, conversation_id)
    if not caller.hand_over:
        service.org_run.acts(caller.node.id, caller.role)
    return caller


_ASIDE = (
    "You are in hand-over mode, and this conversation is for the hand-over "
    "alone: of the director's "
    "tools only `read_team_context`, `hand_over` and the ones your first "
    "message here named work. Leave the rest to your main conversation."
)


def _not_aside(caller: Caller) -> None:
    if caller.hand_over:
        raise Invalid(_ASIDE)


def _title(service: DirectorService, caller: Caller, member: str) -> str:
    """A member of the caller's team as the agents are told of it."""
    return service.team_types.type_of(caller.node).title_of(member)


def _has(
    service: DirectorService,
    caller: Caller,
    tool: str,
    what: str,
    *,
    aside: bool = False,
) -> None:
    """Refuse an agent whose team's type does not give it ``tool``.

    Args:
        what: What the tool does, for the refusal.
        aside: Whether it may be used in hand-over mode.
    """
    kind = service.team_types.type_of(caller.node)
    if kind.has(caller.role, tool):
        if not (caller.hand_over and aside):
            _not_aside(caller)
        return
    _not_aside(caller)
    holder = kind.member_with(tool) or kind.member_with(tool, lead=True)
    raise Invalid(
        f"Only the {_title(service, caller, holder or tool)} of a team can {what}"
    )


def _writes(service: DirectorService, caller: Caller, what: str) -> None:
    """Refuse an agent that is not the one that works in its team's
    folder."""
    _not_aside(caller)
    writer = service.team_types.type_of(caller.node).writer
    if caller.role != writer:
        raise Invalid(
            f"Only the {_title(service, caller, writer)} of a team can {what}"
        )


def node_context(service: DirectorService, caller: Caller) -> str:
    """Who the agent is and everything of the org that concerns its team."""
    context = service.team_context.context_of(caller.node, caller.role, in_full=True)
    names = service.team_types.team(caller.node).names
    kind = service.team_types.type_of(caller.node).text_for(caller.role)
    # The briefing said these once; they are given again only when asked for.
    context["how_the_org_works"] = (
        role_text(
            names,
            works=True,
            hand_over=True,
            rules=service.team_types.document_rules(caller.node),
        )
        if caller.hand_over
        else service.team_types.role_text(caller.node, caller.role)
    )
    if kind:
        context["what_your_teams_type_says"] = kind
    return json.dumps(context, indent=2)


def write_document(
    service: DirectorService, caller: Caller, name: str, arguments: BaseModel
) -> str:
    """Carry out a call of the tool of the document named ``name``: the
    document's protocol does it, for the team of the caller."""
    protocol = service.team_types.protocols.get(name)
    if protocol is None:
        raise Invalid(f"The org keeps no document named {name!r}")
    _has(
        service,
        caller,
        protocol.document.written_with,
        f"write its {name}",
        aside=protocol.usable_in_hand_over,
    )
    node = caller.node
    written = protocol.writer(
        DocumentToolCall(
            team=team_ref(node),
            member=caller.role,
            arguments=arguments,
            hand_over=caller.hand_over,
            leads_session=is_division_lead(node),
            words=asdict(service.team_types.team(node).names),
        )
    )
    if written.summary is not None:
        # With a document of its own that says what the team is for, what
        # teams before it left is no longer its guide.
        service.org_load.org.update_place(
            node.id, summary=written.summary, leftover_documents=[]
        )
    if caller.hand_over:
        service.team_messaging.note_hand_over(node.id, wrote=name)
    else:
        service.message_delivery.knows(node.id, caller.role, name)
    return written.answer


def _checked(
    service: DirectorService, place: SessionTeam, proposed: Proposed
) -> SplitProposal:
    """Proposed sub-teams, held to what the org allows."""
    store = service.org_load.org
    allowed = service.org_settings.settings.allowed_profiles
    for child in proposed.children:
        if child.agent_profile and child.agent_profile not in allowed:
            raise Invalid(
                f"Sub-team {child.title!r} cannot run on the model "
                f"{child.agent_profile!r}. "
                + (
                    f"The org allows: {', '.join(allowed)}."
                    if allowed
                    else "The org allows none; leave `model` out."
                )
            )
    types = {kind.name: kind for kind in service.org_settings.types_allowed()}
    children = []
    for child in proposed.children:
        kind = types.get(child.agent_type)
        if kind is None:
            raise Invalid(
                f"Sub-team {child.title!r} cannot be of the type "
                f"{child.agent_type!r}. The types are: {', '.join(types)}."
            )
        # A type may name the profile its teams run on. The proposal's own
        # choice holds over it. It need not be among the profiles an IC
        # may choose: whoever runs the server wrote the type.
        if not child.agent_profile and kind.sas_profile:
            if not service.org_run.profile_exists(kind.sas_profile):
                raise Invalid(
                    f"Sub-team {child.title!r} cannot be of the type "
                    f"{kind.name!r}: that type runs on the agent profile "
                    f"{kind.sas_profile!r}, and this server has none of that "
                    "name. Choose another type, or raise it with "
                    "raise_escalation."
                )
            child = child.model_copy(update={"agent_profile": kind.sas_profile})
        children.append(child)
    proposed = proposed._replace(children=children)
    split = normalize_split(store, place, proposed.children, proposed.keep)
    check_limits(
        store,
        place,
        len(split.children) - len(split.replaces),
        service.org_settings.settings,
    )
    return SplitProposal(
        children=split.children,
        reason=proposed.reason,
        replaces=[team.id for team in split.replaces],
        keeps=[team.id for team in split.keeps],
        proposed_at=utc_now(),
    )


def _named(proposal: SplitProposal, replaced: list[str]) -> str:
    """The sub-teams of a proposal, as the record of proposals names them."""
    teams = ", ".join(f"{child.title} ({child.path})" for child in proposal.children)
    return teams + (f"; in place of {', '.join(replaced)}" if replaced else "")


def _paths(proposal: SplitProposal) -> list[str]:
    return [child.path for child in proposal.children]


def _waiting(place: SessionTeam) -> SplitProposal | None:
    """The sub-teams a team's IC proposed that still wait on its Manager."""
    proposal = place.pending_split
    return proposal if proposal and proposal.status == "proposed" else None


def msg_manager(service: DirectorService, caller: Caller, text: str) -> str:
    """Send the caller's Manager a message."""
    names = service.team_types.team(caller.node).names
    _has(service, caller, "msg_manager", f"message its {names.manager}")
    node = caller.node
    text = text.strip()
    if not text:
        raise Invalid("A message cannot be empty")
    if not service.org_load.org.below(node.id):
        raise Invalid(
            f"A team without sub-teams has nothing to tell its {names.manager} but a "
            "proposal: use propose_teams. Raise anything else with "
            "raise_escalation."
        )
    service.team_messaging.send_msg(
        node.id, caller.role, node.id, service.team_types.team(node).manager, text
    )
    return f"Sent to your {names.manager}."


def propose_teams(
    service: DirectorService,
    caller: Caller,
    teams: Sequence[ProposedTeam],
    keep: Sequence[str],
    text: str,
) -> str:
    """Put the sub-teams the caller proposes before its Manager."""
    names = service.team_types.team(caller.node).names
    _has(service, caller, "propose_teams", "propose sub-teams")
    node = caller.node
    if node.pending_split and node.pending_split.status == "approved":
        raise Conflict(
            f"The sub-teams your {names.manager} approved are still being set up"
        )
    service.org_settings.team_documents().before_sub_teams_are_proposed(
        service.team_types.reader_of(node)
    )
    proposed = proposed_teams(teams, keep, text, service.org_settings.default_type())
    proposal = _checked(service, node, proposed)
    store = service.org_load.org
    store.update_place(node.id, pending_split=proposal)
    service.message_delivery.knows(node.id, caller.role, "pending_split")
    replaced = [team.title for team in map(store.place, proposal.replaces) if team]
    kept = [team.title for team in map(store.place, proposal.keeps) if team]
    note = (
        "\n\n[director] These sub-teams passed the org's checks."
        + (
            f" They take the place of: {', '.join(replaced)}, whose own sub-teams stay."
            if replaced
            else ""
        )
        + (f" Kept as they are: {', '.join(kept)}." if kept else "")
        + " approve_teams sets them up exactly as written; decline_teams "
        "declines them, with your reason. They wait on you until you call "
        f"one of the two; msg_ic asks your {names.writer} something first."
    )
    facts = service.team_context.facts_of_proposal(node, proposal)
    if facts:
        note += f"\n\n{facts}"
    read = service.org_settings.team_documents().read_by_sub_teams(
        service.team_types.reader_of(node)
    )
    service.team_history.record_event(
        node.id, "proposed", None, _named(proposal, replaced), _paths(proposal)
    )
    said = proposed.reason or "I propose these sub-teams."
    service.team_messaging.send_msg(
        node.id,
        caller.role,
        node.id,
        service.team_types.team(node).manager,
        f"{said}\n\nProposed sub-teams:\n{as_written(proposed)}"
        + "".join(
            f"\n\nYour team's {name}, which each sub-team reads:\n{text}"
            for name, text in read
        ),
        note,
    )
    return (
        f"Sent to your {names.manager}. Nothing is set up until it approves; you "
        "will hear either way."
    )


def msg_ic(service: DirectorService, caller: Caller, text: str, team: str) -> str:
    """Send a message to the caller's own IC, or to the IC of a team
    directly below."""
    names = service.team_types.team(caller.node).names
    _has(service, caller, "msg_ic", f"message {names.a_writer}")
    node = caller.node
    text = text.strip()
    if not text:
        raise Invalid("A message cannot be empty")
    target = node
    if team.strip() and team.strip() != node.id:
        found = _team_named(service, caller, team)
        if found is None or found.parent_id != node.id:
            raise Invalid(f"{team.strip()} is not your team or one directly below it")
        target = found
    # A message is no decision: what the IC proposed goes on waiting.
    waiting = target.id == node.id and _waiting(node) is not None
    note = (
        "\n\n[director] The sub-teams you proposed still wait on your "
        f"{names.manager}. A new proposal takes their place."
        if waiting
        else ""
    )
    service.team_messaging.send_msg(
        node.id,
        caller.role,
        target.id,
        service.team_types.team(target).writer,
        text,
        note,
    )
    return f"Sent to the {names.writer} of {target.title!r}." + (
        " The sub-teams it proposed still wait on you: approve_teams or decline_teams."
        if waiting
        else ""
    )


def approve_teams(service: DirectorService, caller: Caller) -> str:
    """Set up the sub-teams the caller's IC proposed, as written."""
    names = service.team_types.team(caller.node).names
    _has(service, caller, "approve_teams", "approve sub-teams")
    node = caller.node
    proposal = _waiting(node)
    if proposal is None:
        raise Conflict(f"Your {names.writer} has no proposed sub-teams waiting for you")
    # The org may have changed since they were proposed.
    split = normalize_split(service.org_load.org, node, proposal.children)
    check_limits(
        service.org_load.org,
        node,
        len(split.children) - len(split.replaces),
        service.org_settings.settings,
    )
    service.org_load.org.update_place(
        node.id,
        pending_split=proposal.model_copy(
            update={
                "status": "approved",
                "replaces": [team.id for team in split.replaces],
            }
        ),
    )
    service.team_history.record_event(
        node.id,
        "approved",
        None,
        _named(proposal, [team.title for team in split.replaces]),
        _paths(proposal),
    )
    service.team_splits.carry_out_split(node.id)
    waits = (
        " The teams they replace are removed first, once their agents have "
        "finished what they are doing."
        if split.replaces
        else ""
    )
    return (
        f"Approved. {len(proposal.children)} sub-teams are being set up.{waits} "
        f"None of them starts until your {names.writer} gives it its task. Your "
        f"{names.writer} is "
        "told what is its to do now"
        + (" and to pass on what the replaced teams left" if split.replaces else "")
        + "; write to it with msg_ic if the change needs explaining."
    )


def decline_teams(service: DirectorService, caller: Caller, reason: str) -> str:
    """Decline the sub-teams the caller's IC proposed, and keep why."""
    names = service.team_types.team(caller.node).names
    _has(service, caller, "decline_teams", "decline sub-teams")
    node = caller.node
    reason = reason.strip()
    if not reason:
        raise Invalid(f"Say why: your {names.writer} is told, and the reason is kept")
    proposal = _waiting(node)
    if proposal is None:
        raise Conflict(f"Your {names.writer} has no proposed sub-teams waiting for you")
    store = service.org_load.org
    replaced = [team.title for team in map(store.place, proposal.replaces) if team]
    named = _named(proposal, replaced)
    store.update_place(node.id, pending_split=None)
    service.team_history.record_event(
        node.id, "declined", None, f"{named}. Reason: {reason}", _paths(proposal)
    )
    service.team_messaging.send_msg(
        node.id,
        caller.role,
        node.id,
        service.team_types.team(node).writer,
        reason,
        f"\n\n[director] Your {names.manager} declined the sub-teams you proposed "
        f"({named}); the above is its reason. Nothing was set up. Propose "
        "others if the reason leaves room for it; the same ones only if "
        "something has changed.",
    )
    return (
        f"Declined. Your {names.writer} is told your reason, and it is kept in "
        "the record."
    )


def raise_escalation(
    service: DirectorService,
    caller: Caller,
    kind: EscalationKind | Literal["misrouted"],
    title: str,
    detail: str,
) -> str:
    """Send something the caller cannot settle to the agent of its own role
    in the team above. `misrouted` hands back what the human's words
    brought the team directly, and is no escalation that waits."""
    _not_aside(caller)
    if not title.strip():
        raise Invalid("An escalation needs a title")
    if kind == "misrouted":
        _writes(service, caller, "say a request is not its team's")
        return service.session_lifecycle.not_ours(
            caller.node, ": ".join(filter(None, (title.strip(), detail.strip())))
        )
    store = service.org_load.org
    node = caller.node
    escalation = store.raise_escalation(
        node.id,
        uuid4().hex[:8],
        kind=kind,
        title=title.strip(),
        detail=detail.strip(),
        raised_by=caller.role,
    )
    service.org_state.progressed(node.id)
    if escalation.held_by is None:
        return (
            f"Raised as {escalation.id}. Nobody is above your team: it waits "
            "on the human."
        )
    service.team_messaging.hand_escalation(
        escalation.held_by, node, node, escalation, []
    )
    mine = _title(service, caller, caller.role)
    return (
        f"Raised as {escalation.id}. It is with the {mine} of the team above; "
        "you will be told the answer."
    )


def resolve_escalation(
    service: DirectorService,
    caller: Caller,
    node_id: str,
    escalation_id: str,
    action: Literal["settle", "pass_up"],
    text: str,
) -> str:
    """Settle an escalation the caller holds, or pass it one team up.

    An IC holds what the ICs below raised, a Manager what the Managers did.
    """
    _not_aside(caller)
    if not text.strip():
        raise Invalid("Say what you decided, or why you cannot")
    store = service.org_load.org
    raiser = _team_named(service, caller, node_id)
    escalation = next(
        (e for e in raiser.escalations if e.id == escalation_id)
        if raiser
        else iter(()),
        None,
    )
    if raiser is None or escalation is None:
        raise NotFound(f"Team {node_id} has no escalation {escalation_id}")
    if escalation.held_by != caller.node.id:
        raise Invalid(f"Escalation {escalation_id} is not with your team")
    holder = service.team_types.team(caller.node).answers(escalation.raised_by)
    if holder != caller.role:
        raise Invalid(
            f"Escalation {escalation_id} was raised by the "
            f"{service.team_types.team(raiser).title(escalation.raised_by)} of its "
            f"team: it is for your team's {_title(service, caller, holder)} to settle"
        )
    mine = _title(service, caller, caller.role)
    if action == "settle":
        store.settle_escalation(raiser.id, escalation_id)
        service.org_state.progressed(raiser.id)
        service.message_delivery.notify(
            service.team_types.team(raiser).conversation(escalation.raised_by),
            f"The {mine} of {caller.node.title!r} settled "
            f"your escalation {escalation.title!r}:\n{text.strip()}",
        )
        return "Settled; the agent that raised it has your answer."
    passed = store.pass_up_escalation(raiser.id, escalation_id, reason=text)
    if passed.held_by is None:
        return "Passed up. Nobody is above your team: it now waits on the human."
    service.team_messaging.hand_escalation(
        passed.held_by,
        raiser,
        caller.node,
        passed,
        [f"{handoff.node_id}: {handoff.reason}" for handoff in passed.handoffs],
    )
    return f"Passed up to the {mine} of the team above."


def submit_review(service: DirectorService, caller: Caller, summary: str) -> str:
    """Hand the work of the caller's team to the IC of the team above for
    review."""
    names = service.team_types.team(caller.node).names
    _has(service, caller, "submit_review", "submit its team's work")
    store = service.org_load.org
    node = caller.node
    summary = summary.strip()
    if not summary:
        raise Invalid("Say what your team delivered")
    review = node.review
    if review is not None and review.state == "pending":
        raise Conflict("Your team's work already waits for its review")
    if node.parent_id is None:
        raise Conflict("A session's work is reviewed from here; nothing is submitted")
    unfinished = [
        team
        for team in store.below(node.id)
        # A team that is blocked or has no agent holds nothing up.
        if team.agents
        and team.status != "blocked"
        and (
            # A team whose work was accepted is finished, also in the turn
            # that tells it so: a new task takes its review away.
            (team.review is None and team.status == "working")
            or (team.review is not None and team.review.state != "accepted")
        )
    ]
    if unfinished:
        raise Conflict(
            "Your team's work goes up whole. Not finished below you: "
            + "; ".join(
                f"{team.title!r} "
                + (
                    "is working"
                    if team.review is None
                    else "waits for your review"
                    if team.review.state == "pending"
                    else "was sent back and has not submitted again"
                )
                for team in unfinished
            )
            + ". Review what waits with review_team, wait for a team that "
            "works or write to it with msg_team, or remove one you no "
            "longer need with prune_team. Then submit."
        )
    left = service.checkouts.uncommitted(Path(node.worktree_path), node.boundary.path)
    for team in store.below(node.id):
        theirs = [path for path in left if owns(team, path)]
        if theirs:
            raise Conflict(
                f"{', '.join(theirs[:10])}"
                + (f" and {len(theirs) - 10} more" if len(theirs) > 10 else "")
                + f" in the folder of your sub-team {team.title!r} "
                "is not committed, and is that team's to commit. "
                + (
                    "Wait until it has submitted its work and you have accepted it."
                    if team.agents
                    else "Give the team its task with msg_team; it "
                    "commits and submits what is there, and you review "
                    "that."
                )
            )
    if left:
        raise Conflict(
            "Commit your team's work before you submit it. Not "
            f"committed: {', '.join(left[:10])}"
            + (f" and {len(left) - 10} more" if len(left) > 10 else "")
        )
    astray = service.checkouts.astray(
        node, store.below(node.id), service.org_load.repo_dir
    )
    if astray is not None:
        raise Conflict(astray)
    again = review is not None and review.state == "changes_requested"
    store.set_review(
        node.id,
        Review(
            state="pending",
            round=review.round + 1 if review is not None and again else 1,
            summary=summary,
            at=utc_now(),
            seq=service.session_merges.next_submission(),
            commit=service.checkouts.version(Path(node.worktree_path)) or "",
        ),
    )
    service.org_state.progressed(node.id)
    service.reviews.rest(node.id)
    service.reviews.request_review(node.id)
    service.team_summaries.refresh_headline(
        node.id, f"it submitted its work. {summary}"
    )
    return (
        f"Submitted. The {names.above_reviewer} of the team above reviews it. "
        "Your turn ends "
        "here: you are woken when it is accepted or sent back, or when "
        "something else is sent to you."
    )


def review_team(
    service: DirectorService,
    caller: Caller,
    place_id: str,
    decision: Literal["accept", "reject"],
    findings: list[ReviewFinding],
    needed_above: str,
    reconciled: str = "",
    choices: Sequence[DocumentChoice] = (),
) -> str:
    """Accept the work a team directly below submitted, or send it back.

    Args:
        choices: For the documents the work and the org each made
            something else of: which holds.
    """
    _has(service, caller, "review_team", "review the work of a team")
    team = _team_named(service, caller, place_id)
    if team is None or team.parent_id != caller.node.id:
        raise Invalid(f"{place_id} is not a team directly below yours")
    place_id = team.id
    if team.review is None or team.review.state != "pending":
        raise Conflict(f"Team {place_id} has no work waiting for your review")
    if decision == "reject":
        if not findings:
            raise Invalid("Say what must change before you can accept the work")
        service.reviews.request_changes(place_id, findings)
        return f"Sent back to {team.title!r} with what must change."
    chosen = service.session_merges.document_choices(place_id, choices)
    service.reviews.check_accept(place_id, reconciled.strip())
    service.reviews.accept_review(
        place_id, needed_above.strip(), reconciled.strip(), chosen
    )
    return (
        f"Accepting the work of {team.title!r}. You are told if it cannot be "
        "brought into your checkout."
    )


def settle_teams(
    service: DirectorService,
    caller: Caller,
    teams: list[TeamChoice],
    note: str,
    choices: Sequence[DocumentChoice] = (),
) -> str:
    """Settle what waits for the caller of a session's teams.

    Args:
        choices: For the documents that wait with the teams: which holds.
    """
    _has(service, caller, "settle_teams", "settle the sub-teams of a session")
    return service.session_merges.settle_teams(
        caller.node.id,
        [
            choice.model_copy(
                update={
                    "team": team_key(choice.team.strip()),
                    "under": team_key(choice.under.strip()) if choice.under else None,
                    "title": choice.title.strip() if choice.title else None,
                }
            )
            for choice in teams
        ],
        note.strip(),
        choices,
    )


def _team_named(
    service: DirectorService, caller: Caller, name: str
) -> SessionTeam | None:
    """The team an agent names: by the id of its place in the agent's own
    session, or by the team's id in the org, which is then its place in
    that session."""
    name = name.strip()
    store = service.org_load.org
    if "@" in name:
        return store.place(name)
    return store.instance(name, caller.node.session)


def send_document(
    service: DirectorService, caller: Caller, team: str, document: str
) -> str:
    """Pass a document of the caller's team to another team of its
    session, to read."""
    _has(service, caller, "send_document", "pass a document to another team")
    target = _team_named(service, caller, team)
    if target is None or target.session != caller.node.session:
        raise Invalid(f"{team.strip()} is not a team of your session")
    name = document.strip()
    if name not in service.team_types.protocols:
        raise Invalid(f"The org keeps no document named {name!r}")
    service.team_messaging.send_ref(
        caller.node.id,
        caller.role,
        target.id,
        service.team_types.team(target).writer,
        team_ref(caller.node).keeps(name),
    )
    return (
        f"{target.title!r} reads your team's {name} from now on. It is told "
        "with its next message."
    )


def _team_below(service: DirectorService, caller: Caller, node_id: str) -> SessionTeam:
    team = _team_named(service, caller, node_id)
    if team is None or team.parent_id != caller.node.id:
        raise Invalid(f"{node_id.strip()} is not a team directly below yours")
    return team


def msg_team(service: DirectorService, caller: Caller, team: str, text: str) -> str:
    """Send a message to the IC of a team directly below the caller's."""
    names = service.team_types.team(caller.node).names
    _has(service, caller, "msg_team", f"message the {names.writer} of a team below")
    text = text.strip()
    if not text:
        raise Invalid("A message cannot be empty")
    target = _team_below(service, caller, team)
    new = service.team_types.writer_conversation(target) is None
    left = (
        service.checkouts.uncommitted(
            Path(caller.node.worktree_path), target.boundary.path
        )
        if new
        else []
    )
    if service.team_messaging.send_msg(
        caller.node.id,
        caller.role,
        target.id,
        service.team_types.team(target).writer,
        text,
    ):
        return (
            f"Sent to the {names.writer} of {target.title!r}. Its team was stopped "
            "and goes on now."
        )
    if new:
        return (
            f"{target.title!r} starts with what you wrote as its task. Its "
            f"{names.writer} "
            "does not answer in a message: it submits its work for your "
            "review, or raises an escalation."
            + (
                f" The {len(left)} file(s) you left uncommitted in its "
                "folder are the team's now, and it is told of them. From "
                "here on a write of yours there is refused."
                if left
                else ""
            )
        )
    return (
        f"Sent to the {names.writer} of {target.title!r}. It does not answer in a "
        "message: it acts on what you wrote, or raises an escalation."
    )


def hand_over(service: DirectorService, caller: Caller, team: str, text: str) -> str:
    """Pass on to a team directly below what the teams before it left.

    Args:
        text: What the team should know of it, in the caller's words. It
            is kept for the team, which is told with its next message and
            writes what should last into its own documents. It starts
            nothing.
    """
    _has(service, caller, "hand_over", "hand over what removed teams left", aside=True)
    if not text.strip():
        raise Invalid("Say what the team should know: there is nothing to hand over")
    target = _team_below(service, caller, team)
    service.session_records.removed_team_handed_over(
        target, caller.node, caller.role, text.strip()
    )
    if caller.hand_over:
        service.team_messaging.note_hand_over(caller.node.id, team=target.title)
    return f"Handed over to {target.title!r}. " + (
        "Its agents read it with the next message they get."
        if target.agents
        else "It has not started; it reads this in its briefing when you "
        "give it its task with msg_team."
    )


def stop_team(service: DirectorService, caller: Caller, team: str, reason: str) -> str:
    """Stop a team directly below the caller's."""
    names = service.team_types.team(caller.node).names
    _has(service, caller, "stop_team", "stop a team")
    if not reason.strip():
        raise Invalid(f"Say why the team is stopped; your {names.manager} is told")
    target = _team_below(service, caller, team)
    service.team_messaging.stop_team(target.id, reason.strip())
    return (
        f"Stopped {target.title!r} and every team below it: their agents "
        f"end the step they are in and are sent nothing. Your {names.manager} is "
        "told. The team goes on when you write to it with msg_team; only "
        f"your {names.manager} can remove it."
    )


def prune_team(
    service: DirectorService,
    caller: Caller,
    place_id: str,
    reason: str,
) -> str:
    """Remove a team directly below the caller's."""
    names = service.team_types.team(caller.node).names
    _has(service, caller, "prune_team", "prune a team")
    if not reason.strip():
        raise Invalid(
            f"Say why the team is removed; your team's {names.writer} is told"
        )
    team = _team_named(service, caller, place_id)
    if team is None or team.parent_id != caller.node.id:
        raise Invalid(f"{place_id} is not a team directly below yours")
    place_id = team.id
    service.team_splits.prune(place_id, reason.strip())
    return (
        f"Pruning {team.title!r}. Its folder goes back to your team's {names.writer} "
        f"when it is gone. Your {names.writer} is given what the team left and told "
        "what is its to do now; write to it with msg_ic if the change "
        "needs explaining."
    )
