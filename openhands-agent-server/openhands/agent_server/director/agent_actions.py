"""What the agents of an org can do to it, behind their tools.

Every function names the agent that acts and refuses what its role or its
team does not allow, so a tool cannot reach past its own team.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from openhands.agent_server.director.boundary import owns
from openhands.agent_server.director.git import uncommitted
from openhands.agent_server.director.git_store import team_key
from openhands.agent_server.director.memory_edit import EditCommand, edited
from openhands.agent_server.director.models import (
    ContractsChoice,
    EscalationKind,
    Review,
    ReviewFinding,
    SessionTeam,
    SpecChoice,
    SplitProposal,
    TeamChoice,
)
from openhands.agent_server.director.org_store import is_division_lead, utc_now
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
    found = service.store.find_by_conversation(str(conversation_id))
    if found is not None:
        return Caller(node=found[0], role=found[1])
    node = service.store.find_hand_over(str(conversation_id))
    if node is None:
        raise NotFound("This conversation is not an agent of the org")
    return Caller(node=node, role=service.team(node).writer, hand_over=True)


_ASIDE = (
    "You are in hand-over mode, and this conversation is for the hand-over "
    "alone: of the director's "
    "tools only `read_team_context`, `hand_over` and `write_spec` work here. "
    "Leave the rest to your main conversation."
)


def _not_aside(caller: Caller) -> None:
    if caller.hand_over:
        raise Invalid(_ASIDE)


def _title(service: DirectorService, caller: Caller, member: str) -> str:
    """A member of the caller's team as the agents are told of it."""
    return service.type_of(caller.node).title_of(member)


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
    kind = service.type_of(caller.node)
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
    writer = service.type_of(caller.node).writer
    if caller.role != writer:
        raise Invalid(
            f"Only the {_title(service, caller, writer)} of a team can {what}"
        )


def node_context(service: DirectorService, caller: Caller) -> str:
    """Who the agent is and everything of the org that concerns its team."""
    context = service.context_of(caller.node, caller.role, in_full=True)
    names = service.team(caller.node).names
    kind = service.type_of(caller.node).text_for(caller.role)
    # The briefing said these once; they are given again only when asked for.
    context["how_the_org_works"] = (
        role_text(names, works=True, hand_over=True)
        if caller.hand_over
        else service.role_text(caller.node, caller.role)
    )
    if kind:
        context["what_your_teams_type_says"] = kind
    return json.dumps(context, indent=2)


def write_spec(
    service: DirectorService,
    caller: Caller,
    content: str,
    *,
    command: Literal["create", "str_replace", "insert"] = "create",
    old_str: str = "",
    new_str: str = "",
    insert_line: int | None = None,
) -> str:
    """Write the spec of the caller's boundary, or change one place in it,
    as the file editor's command of that name does.

    Args:
        content: The whole spec, for ``create``.
    """
    _has(service, caller, "write_spec", "write its spec", aside=True)
    if command != "create":
        if content.strip():
            raise Invalid(
                f"`file_text` is the whole spec and goes with `create`; "
                f"`{command}` takes `old_str`, `new_str` and `insert_line`"
            )
        own = next(
            (spec for spec in service.specs_of(caller.node) if not spec.old), None
        )
        if own is None:
            raise Invalid("Your team has no spec to change yet: write it with `create`")
        content = edited(
            own.spec,
            command,
            old_str=old_str,
            new_str=new_str,
            insert_line=insert_line,
            name="the spec",
        )
    elif old_str or new_str or insert_line is not None:
        raise Invalid(
            "`create` takes the whole spec in `file_text`; to change one "
            "place use `str_replace` or `insert`"
        )
    if not content.strip():
        raise Invalid("A spec cannot be empty")
    path = caller.node.boundary.path
    service.write_spec(caller.node, content)
    summary = content.strip().splitlines()[0].lstrip("# ").strip()[:200]
    # With a spec of its own, what teams before it left is no longer its guide.
    service.store.update_place(caller.node.id, spec_summary=summary, old_specs=[])
    if caller.hand_over:
        service.note_hand_over(caller.node.id, spec=True)
    else:
        service.knows(caller.node.id, caller.role, "spec")
    return f"Saved the spec of {path or 'the repository'}."


def _checked(
    service: DirectorService, place: SessionTeam, proposed: Proposed
) -> SplitProposal:
    """Proposed sub-teams, held to what the org allows."""
    store = service.store
    allowed = service.settings.allowed_profiles
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
    types = {kind.name: kind for kind in service.types_allowed()}
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
            if not service.profile_exists(kind.sas_profile):
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
        store, place, len(split.children) - len(split.replaces), service.settings
    )
    return SplitProposal(
        children=split.children,
        contracts=proposed.contracts,
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
    names = service.team(caller.node).names
    _has(service, caller, "msg_manager", f"message its {names.overseer}")
    node = caller.node
    text = text.strip()
    if not text:
        raise Invalid("A message cannot be empty")
    if not service.store.below(node.id):
        raise Invalid(
            f"A team without sub-teams has nothing to tell its {names.overseer} but a "
            "proposal: use propose_teams. Raise anything else with "
            "raise_escalation."
        )
    service.send(node.id, caller.role, node.id, service.team(node).overseer, text)
    return f"Sent to your {names.overseer}."


def propose_teams(
    service: DirectorService,
    caller: Caller,
    teams: Sequence[ProposedTeam],
    contracts: str,
    keep: Sequence[str],
    text: str,
) -> str:
    """Put the sub-teams the caller proposes before its Manager."""
    names = service.team(caller.node).names
    _has(service, caller, "propose_teams", "propose sub-teams")
    node = caller.node
    if node.pending_split and node.pending_split.status == "approved":
        raise Conflict(
            f"The sub-teams your {names.overseer} approved are still being set up"
        )
    proposed = proposed_teams(teams, contracts, keep, text, service.default_type())
    proposal = _checked(service, node, proposed)
    store = service.store
    store.update_place(node.id, pending_split=proposal)
    service.knows(node.id, caller.role, "pending_split")
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
    facts = service.facts_of_proposal(node, proposal)
    if facts:
        note += f"\n\n{facts}"
    service.record_event(
        node.id, "proposed", None, _named(proposal, replaced), _paths(proposal)
    )
    said = proposed.reason or "I propose these sub-teams."
    service.send(
        node.id,
        caller.role,
        node.id,
        service.team(node).overseer,
        f"{said}\n\nProposed sub-teams:\n{as_written(proposed)}",
        note,
    )
    return (
        f"Sent to your {names.overseer}. Nothing is set up until it approves; you "
        "will hear either way."
    )


def msg_ic(service: DirectorService, caller: Caller, text: str, team: str) -> str:
    """Send a message to the caller's own IC, or to the IC of a team
    directly below."""
    names = service.team(caller.node).names
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
        f"{names.overseer}. A new proposal takes their place."
        if waiting
        else ""
    )
    service.send(
        node.id, caller.role, target.id, service.team(target).writer, text, note
    )
    return f"Sent to the {names.writer} of {target.title!r}." + (
        " The sub-teams it proposed still wait on you: approve_teams or decline_teams."
        if waiting
        else ""
    )


def approve_teams(service: DirectorService, caller: Caller) -> str:
    """Set up the sub-teams the caller's IC proposed, as written."""
    names = service.team(caller.node).names
    _has(service, caller, "approve_teams", "approve sub-teams")
    node = caller.node
    proposal = _waiting(node)
    if proposal is None:
        raise Conflict(f"Your {names.writer} has no proposed sub-teams waiting for you")
    # The org may have changed since they were proposed.
    split = normalize_split(service.store, node, proposal.children)
    check_limits(
        service.store,
        node,
        len(split.children) - len(split.replaces),
        service.settings,
    )
    service.store.update_place(
        node.id,
        pending_split=proposal.model_copy(
            update={
                "status": "approved",
                "replaces": [team.id for team in split.replaces],
            }
        ),
    )
    service.record_event(
        node.id,
        "approved",
        None,
        _named(proposal, [team.title for team in split.replaces]),
        _paths(proposal),
    )
    service.carry_out_split(node.id)
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
        "told to rewrite the team's spec"
        + (" and to pass on what the replaced teams left" if split.replaces else "")
        + "; write to it with msg_ic if the change needs explaining."
    )


def decline_teams(service: DirectorService, caller: Caller, reason: str) -> str:
    """Decline the sub-teams the caller's IC proposed, and keep why."""
    names = service.team(caller.node).names
    _has(service, caller, "decline_teams", "decline sub-teams")
    node = caller.node
    reason = reason.strip()
    if not reason:
        raise Invalid(f"Say why: your {names.writer} is told, and the reason is kept")
    proposal = _waiting(node)
    if proposal is None:
        raise Conflict(f"Your {names.writer} has no proposed sub-teams waiting for you")
    store = service.store
    replaced = [team.title for team in map(store.place, proposal.replaces) if team]
    named = _named(proposal, replaced)
    store.update_place(node.id, pending_split=None)
    service.record_event(
        node.id, "declined", None, f"{named}. Reason: {reason}", _paths(proposal)
    )
    service.send(
        node.id,
        caller.role,
        node.id,
        service.team(node).writer,
        reason,
        f"\n\n[director] Your {names.overseer} declined the sub-teams you proposed "
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
        return service.not_ours(
            caller.node, ": ".join(filter(None, (title.strip(), detail.strip())))
        )
    store = service.store
    node = caller.node
    escalation = store.raise_escalation(
        node.id,
        uuid4().hex[:8],
        kind=kind,
        title=title.strip(),
        detail=detail.strip(),
        raised_by=caller.role,
    )
    service.progressed(node.id)
    if escalation.held_by is None:
        return (
            f"Raised as {escalation.id}. Nobody is above your team: it waits "
            "on the human."
        )
    service.hand_escalation(escalation.held_by, node, node, escalation, [])
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
    store = service.store
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
    holder = service.team(caller.node).answers(escalation.raised_by)
    if holder != caller.role:
        raise Invalid(
            f"Escalation {escalation_id} was raised by the "
            f"{service.team(raiser).title(escalation.raised_by)} of its team: it "
            f"is for your team's {_title(service, caller, holder)} to settle"
        )
    mine = _title(service, caller, caller.role)
    if action == "settle":
        store.settle_escalation(raiser.id, escalation_id)
        service.progressed(raiser.id)
        service.notify(
            service.team(raiser).conversation(escalation.raised_by),
            f"The {mine} of {caller.node.title!r} settled "
            f"your escalation {escalation.title!r}:\n{text.strip()}",
        )
        return "Settled; the agent that raised it has your answer."
    passed = store.pass_up_escalation(raiser.id, escalation_id, reason=text)
    if passed.held_by is None:
        return "Passed up. Nobody is above your team: it now waits on the human."
    service.hand_escalation(
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
    names = service.team(caller.node).names
    _has(service, caller, "submit_review", "submit its team's work")
    store = service.store
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
    left = uncommitted(Path(node.worktree_path), node.boundary.path)
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
    again = review is not None and review.state == "changes_requested"
    store.set_review(
        node.id,
        Review(
            state="pending",
            round=review.round + 1 if review is not None and again else 1,
            summary=summary,
            at=utc_now(),
            seq=service.next_submission(),
        ),
    )
    service.progressed(node.id)
    service.rest(node.id)
    service.request_review(node.id)
    service.refresh_headline(node.id, f"it submitted its work. {summary}")
    return (
        f"Submitted. The {names.reviewer} of the team above reviews it. Your turn ends "
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
    specs: list[SpecChoice] | None = None,
    contracts: str = "",
) -> str:
    """Accept the work a team directly below submitted, or send it back.

    Args:
        contracts: What the accepted work establishes for every team below
            the caller's, added to the contracts between them.
    """
    names = service.team(caller.node).names
    _has(service, caller, "review_team", "review the work of a team")
    team = _team_named(service, caller, place_id)
    if team is None or team.parent_id != caller.node.id:
        raise Invalid(f"{place_id} is not a team directly below yours")
    place_id = team.id
    if team.review is None or team.review.state != "pending":
        raise Conflict(f"Team {place_id} has no work waiting for your review")
    contracts = contracts.strip()
    if contracts and decision == "reject":
        raise Invalid(
            "`contracts` records what accepted work establishes; leave it "
            "empty when you reject"
        )
    if contracts and is_division_lead(caller.node):
        raise Invalid(
            f"Contracts are not yours as the Division Lead {names.reviewer}: the "
            "Division "
            f"Lead {names.overseer} settles them"
        )
    if decision == "reject":
        if not findings:
            raise Invalid("Say what must change before you can accept the work")
        service.request_changes(place_id, findings)
        return f"Sent back to {team.title!r} with what must change."
    chosen = service.spec_choices(place_id, specs or [])
    service.check_accept(place_id, reconciled.strip())
    service.accept_review(place_id, needed_above.strip(), reconciled.strip(), chosen)
    if contracts:
        service.add_to_contracts(caller.node, contracts)
    return (
        f"Accepting the work of {team.title!r}. You are told if it cannot be "
        "brought into your checkout."
        + (
            " What you wrote was added to the contracts between your "
            "sub-teams; each is told with its next message."
            if contracts
            else ""
        )
    )


def settle_teams(
    service: DirectorService,
    caller: Caller,
    teams: list[TeamChoice],
    note: str,
    contracts: list[ContractsChoice] | None = None,
) -> str:
    """Settle what waits for the caller of a session's teams."""
    _has(service, caller, "settle_teams", "settle the sub-teams of a session")
    return service.settle_teams(
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
        contracts,
    )


def edit_contracts(
    service: DirectorService,
    caller: Caller,
    command: EditCommand,
    *,
    old_str: str = "",
    new_str: str = "",
    insert_line: int | None = None,
) -> str:
    """Change one place in the contracts between the caller's sub-teams, or
    add to them, as the file editor's command of that name does."""
    names = service.team(caller.node).names
    _has(
        service, caller, "edit_contracts", "change the contracts between its sub-teams"
    )
    node = caller.node
    if is_division_lead(node):
        raise Invalid(
            f"Contracts are not yours as the Division Lead {names.reviewer}: the "
            "Division "
            f"Lead {names.overseer} settles them"
        )
    if not service.store.below(node.id):
        raise Invalid(
            "Your team has no sub-teams, so it sets no contracts. The ones "
            "you keep are the team's above; raise what must change in them "
            "with raise_escalation."
        )
    now = service.own_contracts(node)
    service.set_contracts(
        node,
        edited(
            now,
            command,
            old_str=old_str,
            new_str=new_str,
            insert_line=insert_line,
            name="the contracts",
        ),
    )
    return (
        "The contracts between your sub-teams are changed. Each sub-team is "
        "told with its next message; write to one with msg_team if it must "
        "know at once."
    )


def report(service: DirectorService, caller: Caller, text: str) -> str:
    """Record something for the two agents of the caller's team."""
    _not_aside(caller)
    service.report(caller.node, caller.role, text)
    return "Recorded for you and the other agent of your team."


def _team_named(
    service: DirectorService, caller: Caller, name: str
) -> SessionTeam | None:
    """The team an agent names: by the id of its place in the agent's own
    session, or by the team's id in the org, which is then its place in
    that session."""
    name = name.strip()
    store = service.store
    if "@" in name:
        return store.place(name)
    return store.instance(name, caller.node.session)


def _team_below(service: DirectorService, caller: Caller, node_id: str) -> SessionTeam:
    team = _team_named(service, caller, node_id)
    if team is None or team.parent_id != caller.node.id:
        raise Invalid(f"{node_id.strip()} is not a team directly below yours")
    return team


def msg_team(service: DirectorService, caller: Caller, team: str, text: str) -> str:
    """Send a message to the IC of a team directly below the caller's."""
    names = service.team(caller.node).names
    _has(service, caller, "msg_team", f"message the {names.writer} of a team below")
    text = text.strip()
    if not text:
        raise Invalid("A message cannot be empty")
    target = _team_below(service, caller, team)
    new = service.writer_conversation(target) is None
    left = (
        uncommitted(Path(caller.node.worktree_path), target.boundary.path)
        if new
        else []
    )
    if service.send(
        caller.node.id, caller.role, target.id, service.team(target).writer, text
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


def hand_over(
    service: DirectorService, caller: Caller, team: str, notes: str, contracts: str
) -> str:
    """Pass on to a team directly below what the teams before it left."""
    _has(service, caller, "hand_over", "hand over what removed teams left", aside=True)
    if not notes.strip() and not contracts.strip():
        raise Invalid("Give notes, contracts, or both: there is nothing to hand over")
    target = _team_below(service, caller, team)
    service.hand_over(target, notes, contracts)
    if caller.hand_over:
        service.note_hand_over(caller.node.id, team=target.title)
    given = [
        part
        for part, text in (
            ("your notes are with its reports, marked as handed over", notes),
            ("the contracts between its sub-teams are written", contracts),
        )
        if text.strip()
    ]
    return f"Handed over to {target.title!r}: {' and '.join(given)}. " + (
        "Its agents read it with the next message they get."
        if target.agents
        else "It has not started; it reads this in its briefing when you "
        "give it its task with msg_team."
    )


def stop_team(service: DirectorService, caller: Caller, team: str, reason: str) -> str:
    """Stop a team directly below the caller's."""
    names = service.team(caller.node).names
    _has(service, caller, "stop_team", "stop a team")
    if not reason.strip():
        raise Invalid(f"Say why the team is stopped; your {names.overseer} is told")
    target = _team_below(service, caller, team)
    service.stop_team(target.id, reason.strip())
    return (
        f"Stopped {target.title!r} and every team below it: their agents "
        f"end the step they are in and are sent nothing. Your {names.overseer} is "
        "told. The team goes on when you write to it with msg_team; only "
        f"your {names.overseer} can remove it."
    )


def prune_team(
    service: DirectorService,
    caller: Caller,
    place_id: str,
    reason: str,
) -> str:
    """Remove a team directly below the caller's."""
    names = service.team(caller.node).names
    _has(service, caller, "prune_team", "prune a team")
    if not reason.strip():
        raise Invalid(
            f"Say why the team is removed; your team's {names.writer} is told"
        )
    team = _team_named(service, caller, place_id)
    if team is None or team.parent_id != caller.node.id:
        raise Invalid(f"{place_id} is not a team directly below yours")
    place_id = team.id
    service.prune(place_id, reason.strip())
    return (
        f"Pruning {team.title!r}. Its folder goes back to your team's {names.writer} "
        f"when it is gone. Your {names.writer} is given what the team left and told to "
        "rewrite the team's spec; write to it with msg_ic if the change "
        "needs explaining."
    )
