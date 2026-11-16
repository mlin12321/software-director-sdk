"""What the agents of an org can do to it, behind their tools.

Every function names the agent that acts and refuses what its role or its
node does not allow, so a tool cannot reach past its own team.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from openhands.agent_server.director.boundary import owns
from openhands.agent_server.director.git import uncommitted
from openhands.agent_server.director.models import (
    ROLE_NAMES,
    AgentRole,
    EscalationKind,
    OrgNode,
    Review,
    ReviewFinding,
    SplitProposal,
)
from openhands.agent_server.director.org_store import session_id_of
from openhands.agent_server.director.service import (
    DirectorService,
    agent_conversation,
)
from openhands.agent_server.director.split import (
    TEAMS_FENCE,
    Proposed,
    check_limits,
    normalize_split,
    proposed_teams,
)
from openhands.sdk.errors import Conflict, Invalid, NotFound


@dataclass(frozen=True)
class Caller:
    """The agent that uses a tool."""

    node: OrgNode
    role: AgentRole
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
    return Caller(node=node, role="ic", hand_over=True)


_ASIDE = (
    "You are in hand-over mode, and this conversation is for the hand-over "
    "alone: of the director's "
    "tools only `read_node_context`, `hand_over` and `write_spec` work here. "
    "Leave the rest to your main conversation."
)


def _not_aside(caller: Caller) -> None:
    if caller.hand_over:
        raise Invalid(_ASIDE)


def _only(caller: Caller, role: AgentRole, what: str, *, aside: bool = False) -> None:
    """Refuse an agent that is not in ``role``.

    Args:
        aside: Whether an IC may do this in hand-over mode.
    """
    if caller.hand_over and role == "ic" and aside:
        return
    _not_aside(caller)
    if caller.role != role:
        raise Invalid(f"Only the {ROLE_NAMES[role]} of a team can {what}")


def node_context(service: DirectorService, caller: Caller) -> str:
    """Who the agent is and everything of the org that concerns its team."""
    return json.dumps(
        service.context_of(caller.node, caller.role, in_full=True), indent=2
    )


def write_spec(service: DirectorService, caller: Caller, content: str) -> str:
    """Replace the spec of the caller's boundary."""
    _only(caller, "ic", "write its spec", aside=True)
    if not content.strip():
        raise Invalid("A spec cannot be empty")
    path = caller.node.boundary.path
    service.write_spec(caller.node, content)
    summary = content.strip().splitlines()[0].lstrip("# ").strip()[:200]
    # With a spec of its own, what teams before it left is no longer its guide.
    service.store.update_node(caller.node.id, spec_summary=summary, old_specs=[])
    if caller.hand_over:
        service.note_hand_over(caller.node.id, spec=True)
    return f"Saved the spec of {path or 'the repository'}."


def _checked(
    service: DirectorService, node: OrgNode, proposed: Proposed
) -> SplitProposal:
    """The sub-teams a message proposes, held to what the org allows."""
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
    split = normalize_split(store, node, proposed.children, proposed.keep)
    check_limits(
        store, node, len(split.children) - len(split.replaces), service.settings
    )
    return SplitProposal(
        children=split.children,
        contracts=proposed.contracts,
        reason=proposed.reason,
        replaces=[team.id for team in split.replaces],
        keeps=[team.id for team in split.keeps],
        proposed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def _named(proposal: SplitProposal, replaced: list[str]) -> str:
    """The sub-teams of a proposal, as the record of proposals names them."""
    teams = ", ".join(f"{child.title} ({child.path})" for child in proposal.children)
    return teams + (f"; in place of {', '.join(replaced)}" if replaced else "")


def _paths(proposal: SplitProposal) -> list[str]:
    return [child.path for child in proposal.children]


def _waiting(node: OrgNode) -> SplitProposal | None:
    """The sub-teams a team's IC proposed that still wait on its Manager."""
    proposal = node.pending_split
    return proposal if proposal and proposal.status == "proposed" else None


def msg_manager(service: DirectorService, caller: Caller, text: str) -> str:
    """Send the caller's Manager a message, and put before it the sub-teams
    the message proposes, if any."""
    _only(caller, "ic", "message its Manager")
    node = caller.node
    text = text.strip()
    if not text:
        raise Invalid("A message cannot be empty")
    proposed = proposed_teams(text)
    if proposed is None and not service.store.children(node.id):
        raise Invalid(
            "A team without sub-teams messages its Manager only to propose "
            f"some, in a ```{TEAMS_FENCE} block. Raise anything else with "
            "raise_escalation."
        )
    note = ""
    if proposed is not None:
        if node.pending_split and node.pending_split.status == "approved":
            raise Conflict("The sub-teams your Manager approved are still being set up")
        proposal = _checked(service, node, proposed)
        service.store.update_node(node.id, pending_split=proposal)
        store = service.store
        replaced = [team.title for team in map(store.get, proposal.replaces) if team]
        kept = [team.title for team in map(store.get, proposal.keeps) if team]
        note = (
            "\n\n[director] The sub-teams in this message passed the org's "
            "checks."
            + (
                f" They take the place of: {', '.join(replaced)}, whose own "
                "sub-teams stay."
                if replaced
                else ""
            )
            + (f" Kept as they are: {', '.join(kept)}." if kept else "")
            + " approve_teams sets them up exactly as written; decline_teams "
            "declines them, with your reason. They wait on you until you call "
            "one of the two; msg_ic asks your IC something first."
        )
        facts = service.facts_of_proposal(node, proposal)
        if facts:
            note += f"\n\n{facts}"
        service.record_event(
            node.id, "proposed", None, _named(proposal, replaced), _paths(proposal)
        )
    service.message(node.id, node.id, "ic", text, note)
    if proposed is None:
        return "Sent to your Manager."
    return (
        "Sent to your Manager with the sub-teams you proposed. Nothing "
        "is set up until it approves; you will hear either way."
    )


def msg_ic(service: DirectorService, caller: Caller, text: str, team: str) -> str:
    """Send a message to the caller's own IC, or to the IC of a team
    directly below."""
    _only(caller, "manager", "message an IC")
    node = caller.node
    text = text.strip()
    if not text:
        raise Invalid("A message cannot be empty")
    target = node
    if team.strip() and team.strip() != node.id:
        found = service.store.get(team.strip())
        if found is None or found.parent_id != node.id:
            raise Invalid(f"{team.strip()} is not your team or one directly below it")
        target = found
    # A message is no decision: what the IC proposed goes on waiting.
    waiting = target.id == node.id and _waiting(node) is not None
    note = (
        "\n\n[director] The sub-teams you proposed still wait on your "
        "Manager. A new proposal takes their place."
        if waiting
        else ""
    )
    service.message(node.id, target.id, "manager", text, note)
    return f"Sent to the IC of {target.title!r}." + (
        " The sub-teams it proposed still wait on you: approve_teams or decline_teams."
        if waiting
        else ""
    )


def approve_teams(service: DirectorService, caller: Caller) -> str:
    """Set up the sub-teams the caller's IC proposed, as written."""
    _only(caller, "manager", "approve sub-teams")
    node = caller.node
    proposal = _waiting(node)
    if proposal is None:
        raise Conflict("Your IC has no proposed sub-teams waiting for you")
    # The org may have changed since they were proposed.
    split = normalize_split(service.store, node, proposal.children)
    check_limits(
        service.store,
        node,
        len(split.children) - len(split.replaces),
        service.settings,
    )
    service.store.update_node(
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
        "None of them starts until your IC gives it its task. Your IC is "
        "told to rewrite the team's spec"
        + (" and to pass on what the replaced teams left" if split.replaces else "")
        + "; write to it with msg_ic if the change needs explaining."
    )


def decline_teams(service: DirectorService, caller: Caller, reason: str) -> str:
    """Decline the sub-teams the caller's IC proposed, and keep why."""
    _only(caller, "manager", "decline sub-teams")
    node = caller.node
    reason = reason.strip()
    if not reason:
        raise Invalid("Say why: your IC is told, and the reason is kept")
    proposal = _waiting(node)
    if proposal is None:
        raise Conflict("Your IC has no proposed sub-teams waiting for you")
    store = service.store
    replaced = [team.title for team in map(store.get, proposal.replaces) if team]
    named = _named(proposal, replaced)
    store.update_node(node.id, pending_split=None)
    service.record_event(
        node.id, "declined", None, f"{named}. Reason: {reason}", _paths(proposal)
    )
    service.message(
        node.id,
        node.id,
        "manager",
        reason,
        f"\n\n[director] Your Manager declined the sub-teams you proposed "
        f"({named}); the above is its reason. Nothing was set up. Propose "
        "others if the reason leaves room for it; the same ones only if "
        "something has changed.",
    )
    return "Declined. Your IC is told your reason, and it is kept in the record."


def _role_name(role: AgentRole) -> str:
    return ROLE_NAMES[role]


def raise_escalation(
    service: DirectorService,
    caller: Caller,
    kind: EscalationKind | Literal["misrouted"],
    title: str,
    detail: str,
    team_for: str = "",
) -> str:
    """Send something the caller cannot settle to the agent of its own role
    in the team above. `misrouted` hands back what the human's words
    brought the team directly, and is no escalation that waits."""
    _not_aside(caller)
    if not title.strip():
        raise Invalid("An escalation needs a title")
    if kind == "misrouted":
        _only(caller, "ic", "say a request is not its team's")
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
    if team_for.strip() and caller.role == "ic" and service.settings.route_requests:
        service.ask_team(node.id, escalation.id, team_for.strip())
        return (
            f"Raised as {escalation.id}. The runtime takes it to the IC of the "
            "team it concerns if it is sure which that is, and otherwise to "
            "the IC of the team above; you will be told the answer."
        )
    service.hand_escalation(escalation.held_by, node, node, escalation, [])
    return (
        f"Raised as {escalation.id}. It is with the {_role_name(caller.role)} "
        "of the team above; you will be told the answer."
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
    raiser = store.get(node_id)
    escalation = next(
        (e for e in raiser.escalations if e.id == escalation_id)
        if raiser
        else iter(()),
        None,
    )
    if raiser is None or escalation is None:
        raise NotFound(f"Node {node_id} has no escalation {escalation_id}")
    direct = escalation.asked_of == caller.node.id
    if not direct and (
        escalation.asked_of is not None or escalation.held_by != caller.node.id
    ):
        raise Invalid(f"Escalation {escalation_id} is not with your team")
    if escalation.raised_by != caller.role:
        raise Invalid(
            f"Escalation {escalation_id} was raised by "
            f"{'an IC' if escalation.raised_by == 'ic' else 'a Manager'}: it is "
            f"for your team's {_role_name(escalation.raised_by)} to settle"
        )
    if direct:
        return service.answered_directly(raiser, escalation, action, text.strip())
    if action == "settle":
        store.settle_escalation(node_id, escalation_id)
        service.progressed(node_id)
        service.notify(
            agent_conversation(raiser, escalation.raised_by),
            f"The {_role_name(caller.role)} of {caller.node.title!r} settled "
            f"your escalation {escalation.title!r}:\n{text.strip()}",
        )
        return "Settled; the agent that raised it has your answer."
    passed = store.pass_up_escalation(node_id, escalation_id, reason=text)
    if passed.held_by is None:
        return "Passed up. Nobody is above your team: it now waits on the human."
    service.hand_escalation(
        passed.held_by,
        raiser,
        caller.node,
        passed,
        [f"{handoff.node_id}: {handoff.reason}" for handoff in passed.handoffs],
    )
    return f"Passed up to the {_role_name(caller.role)} of the team above."


def submit_review(service: DirectorService, caller: Caller, summary: str) -> str:
    """Hand the work of the caller's team to the IC of the team above for
    review; the top team's is shown to the human."""
    _only(caller, "ic", "submit its team's work")
    store = service.store
    node = caller.node
    summary = summary.strip()
    if not summary:
        raise Invalid("Say what your team delivered")
    review = node.review
    if review is not None and review.state == "pending":
        raise Conflict("Your team's work already waits for its review")
    parent = store.get(node.parent_id) if node.parent_id else None
    if parent is not None:
        left = uncommitted(Path(node.worktree_path), node.boundary.path)
        for team in store.children(node.id):
            theirs = [path for path in left if owns(team, path)]
            if theirs:
                raise Conflict(
                    f"{', '.join(theirs[:10])}"
                    + (f" and {len(theirs) - 10} more" if len(theirs) > 10 else "")
                    + f" in the directory of your sub-team {team.title!r} "
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
            state="unreviewed" if parent is None else "pending",
            round=review.round + 1 if review is not None and again else 1,
            summary=summary,
            at=datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )
    service.progressed(node.id)
    service.rest(node.id)
    service.request_review(node.id)
    service.refresh_headline(node.id, f"it submitted its work. {summary}")
    if parent is None:
        return (
            "Submitted. Nobody is above your team: the human sees its work. "
            "Your turn ends here."
        )
    return (
        "Submitted. The IC of the team above reviews it. Your turn ends "
        "here: you are woken when it is accepted or sent back, or when "
        "something else is sent to you."
    )


def review_team(
    service: DirectorService,
    caller: Caller,
    node_id: str,
    decision: Literal["accept", "reject"],
    findings: list[ReviewFinding],
    needed_above: str,
    reconciled: str = "",
) -> str:
    """Accept the work a team directly below submitted, or send it back."""
    _only(caller, "ic", "review the work of a team")
    team = service.store.get(node_id)
    if team is None or team.parent_id != caller.node.id:
        raise Invalid(f"{node_id} is not a team directly below yours")
    if team.review is None or team.review.state != "pending":
        raise Conflict(f"Team {node_id} has no work waiting for your review")
    if decision == "reject":
        if not findings:
            raise Invalid("Say what must change before you can accept the work")
        service.request_changes(node_id, findings)
        return f"Sent back to {team.title!r} with what must change."
    service.check_accept(node_id, reconciled.strip())
    service.accept_review(node_id, needed_above.strip(), reconciled.strip())
    return (
        f"Accepting the work of {team.title!r}. You are told if it cannot be "
        "brought into your checkout."
    )


def report(service: DirectorService, caller: Caller, text: str) -> str:
    """Record something for the two agents of the caller's team."""
    _not_aside(caller)
    service.report(caller.node, caller.role, text)
    return "Recorded for you and the other agent of your team."


def _team_below(service: DirectorService, caller: Caller, node_id: str) -> OrgNode:
    team = service.store.get(node_id.strip())
    if team is None or team.parent_id != caller.node.id:
        raise Invalid(f"{node_id.strip()} is not a team directly below yours")
    return team


def msg_team(service: DirectorService, caller: Caller, team: str, text: str) -> str:
    """Send a message to the IC of a team directly below the caller's."""
    _only(caller, "ic", "message the IC of a team below")
    text = text.strip()
    if not text:
        raise Invalid("A message cannot be empty")
    target = _team_below(service, caller, team)
    new = agent_conversation(target, "ic") is None
    left = (
        uncommitted(Path(caller.node.worktree_path), target.boundary.path)
        if new
        else []
    )
    if service.message_team(caller.node.id, target.id, text):
        return (
            f"Sent to the IC of {target.title!r}. Its team was stopped and goes on now."
        )
    if new:
        return (
            f"{target.title!r} starts with what you wrote as its task. Its IC "
            "does not answer in a message: it submits its work for your "
            "review, or raises an escalation."
            + (
                f" The {len(left)} file(s) you left uncommitted in its "
                "directory are the team's now, and it is told of them. From "
                "here on a write of yours there is refused."
                if left
                else ""
            )
        )
    return (
        f"Sent to the IC of {target.title!r}. It does not answer in a "
        "message: it acts on what you wrote, or raises an escalation."
    )


def hand_over(
    service: DirectorService, caller: Caller, team: str, notes: str, contracts: str
) -> str:
    """Pass on to a team directly below what the teams before it left."""
    _only(caller, "ic", "hand over what removed teams left", aside=True)
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
    _only(caller, "ic", "stop a team")
    if not reason.strip():
        raise Invalid("Say why the team is stopped; your Manager is told")
    target = _team_below(service, caller, team)
    service.stop_team(target.id, reason.strip())
    return (
        f"Stopped {target.title!r} and every team below it: their agents "
        "end the step they are in and are sent nothing. Your Manager is "
        "told. The team goes on when you write to it with msg_team; only "
        "your Manager can remove it."
    )


def prune_team(
    service: DirectorService,
    caller: Caller,
    node_id: str,
    reason: str,
) -> str:
    """Remove a team directly below the caller's."""
    _only(caller, "manager", "prune a team")
    if not reason.strip():
        raise Invalid("Say why the team is removed; your team's IC is told")
    team = service.store.get(node_id)
    if team is None or team.parent_id != caller.node.id:
        raise Invalid(f"{node_id} is not a team directly below yours")
    if session_id_of(team):
        raise Invalid(
            f"{team.title!r} works in the session "
            f"{service.store.session_title(team)!r}, "
            "which the human opened. Only the human closes a session, and "
            "its teams are not pruned one by one."
        )
    sessions = service.store.sessions_at(team.id)
    if sessions:
        raise Conflict(
            f"{team.title!r} cannot be pruned while the human's sessions "
            f"({', '.join(sessions)}) work in it or below it."
        )
    service.prune(node_id, reason.strip())
    return (
        f"Pruning {team.title!r}. Its directory goes back to your team's IC "
        "when it is gone. Your IC is given what the team left and told to "
        "rewrite the team's spec; write to it with msg_ic if the change "
        "needs explaining."
    )
