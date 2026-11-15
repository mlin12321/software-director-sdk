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

from openhands.agent_server.director.git import uncommitted
from openhands.agent_server.director.git_store import SPEC_FILE
from openhands.agent_server.director.models import (
    AgentRole,
    Escalation,
    EscalationKind,
    OrgNode,
    Review,
    ReviewFinding,
    SplitProposal,
)
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


def caller_of(service: DirectorService, conversation_id: UUID) -> Caller:
    """The agent whose conversation is ``conversation_id``, as it is now.

    Raises:
        NotFound: The conversation is not an agent of the org.
    """
    found = service.store.find_by_conversation(str(conversation_id))
    if found is None:
        raise NotFound("This conversation is not an agent of the org")
    return Caller(node=found[0], role=found[1])


def _only(caller: Caller, role: AgentRole, what: str) -> None:
    if caller.role != role:
        name = "IC" if role == "ic" else "Manager"
        raise Invalid(f"Only the {name} of a team can {what}")


def node_context(service: DirectorService, caller: Caller) -> str:
    """Who the agent is and everything of the org that concerns its team."""
    return json.dumps(service.context_of(caller.node, caller.role), indent=2)


def write_spec(service: DirectorService, caller: Caller, content: str) -> str:
    """Replace the spec of the caller's boundary."""
    _only(caller, "ic", "write its spec")
    if not content.strip():
        raise Invalid("A spec cannot be empty")
    path = caller.node.boundary.path
    service.team_memory.write(path, SPEC_FILE, content)
    summary = content.strip().splitlines()[0].lstrip("# ").strip()[:200]
    service.store.update_node(caller.node.id, spec_summary=summary)
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
        replaces=[team.id for team in split.replaces],
        keeps=[team.id for team in split.keeps],
        proposed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def msg_manager(service: DirectorService, caller: Caller, text: str) -> str:
    """Send the caller's Manager a message, and put before it the sub-teams
    the message proposes, if any."""
    _only(caller, "ic", "message its Manager")
    node = caller.node
    text = text.strip()
    if not text:
        raise Invalid("A message cannot be empty")
    proposed = proposed_teams(text)
    if proposed is None and not node.child_ids:
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
            + " approve_teams sets them up exactly as written; to ask for "
            "changes or decline, answer with msg_ic."
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
    note = ""
    if (
        target.id == node.id
        and node.pending_split
        and node.pending_split.status == "proposed"
    ):
        # An answer other than approval puts the proposal aside.
        service.store.update_node(node.id, pending_split=None)
        note = (
            "\n\n[director] The sub-teams you proposed were not approved. "
            "Propose again in a new message if you still want them."
        )
    service.message(node.id, target.id, "manager", text, note)
    return f"Sent to the IC of {target.title!r}."


def approve_teams(service: DirectorService, caller: Caller) -> str:
    """Set up the sub-teams the caller's IC proposed, as written."""
    _only(caller, "manager", "approve sub-teams")
    node = caller.node
    proposal = node.pending_split
    if proposal is None or proposal.status != "proposed":
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
    service.carry_out_split(node.id)
    waits = (
        " The teams they replace are removed first, once their agents have "
        "finished what they are doing."
        if split.replaces
        else ""
    )
    return (
        f"Approved. {len(proposal.children)} sub-teams are being set up.{waits} "
        "None of them starts until your IC gives it its task."
    )


def _role_name(role: AgentRole) -> str:
    return "IC" if role == "ic" else "Manager"


def raise_escalation(
    service: DirectorService,
    caller: Caller,
    kind: EscalationKind,
    title: str,
    detail: str,
) -> str:
    """Send something the caller cannot settle to the agent of its own role
    in the team above."""
    if not title.strip():
        raise Invalid("An escalation needs a title")
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
    _hand_to(service, escalation.held_by, node, node, escalation, [])
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
    if escalation.held_by != caller.node.id:
        raise Invalid(f"Escalation {escalation_id} is not with your team")
    if escalation.raised_by != caller.role:
        raise Invalid(
            f"Escalation {escalation_id} was raised by "
            f"{'an IC' if escalation.raised_by == 'ic' else 'a Manager'}: it is "
            f"for your team's {_role_name(escalation.raised_by)} to settle"
        )
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
    _hand_to(
        service,
        passed.held_by,
        raiser,
        caller.node,
        passed,
        [f"{handoff.node_id}: {handoff.reason}" for handoff in passed.handoffs],
    )
    return f"Passed up to the {_role_name(caller.role)} of the team above."


def _hand_to(
    service: DirectorService,
    holder_id: str,
    raiser: OrgNode,
    through: OrgNode,
    escalation: Escalation,
    passed_up: list[str],
) -> None:
    """Tell the agent that now holds an escalation.

    Args:
        through: The team directly below the holder that it comes from:
            the one that raised it, or the one that passed it up.
    """
    holder = service.store.get(holder_id)
    if holder is None:
        return
    history = "".join(f"\nPassed up by {entry}" for entry in passed_up)
    if escalation.raised_by == "ic":
        # It is woken now anyway, so it gets what waits for its review too.
        service.release_reviews(holder_id, now=True)
        service.count_ic_escalation(holder_id, through.id)
    text = (
        f"The {_role_name(escalation.raised_by)} of team {raiser.title!r} "
        f"(node {raiser.id}) escalates {escalation.title!r} (escalation "
        f"{escalation.id}):\n{escalation.detail or '(no detail)'}{history}\n"
        "Settle it with resolve_escalation if you can; pass it up only with "
        "the reason you cannot."
    )
    if escalation.raised_by == "manager":
        service.tell_manager(holder_id, text)
    else:
        service.tell_ic(holder_id, text)


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
    service.accept_review(node_id, needed_above.strip())
    return (
        f"Accepting the work of {team.title!r}. You are told if it cannot be "
        "brought into your checkout."
    )


def report(service: DirectorService, caller: Caller, text: str) -> str:
    """Record something for the two agents of the caller's team."""
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
    if service.message_team(caller.node.id, target.id, text):
        return (
            f"Sent to the IC of {target.title!r}. Its team was stopped and goes on now."
        )
    if new:
        return (
            f"{target.title!r} starts with what you wrote as its task. Its IC "
            "does not answer in a message: it submits its work for your "
            "review, or raises an escalation."
        )
    return (
        f"Sent to the IC of {target.title!r}. It does not answer in a "
        "message: it acts on what you wrote, or raises an escalation."
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
    service.prune(node_id, reason.strip())
    return (
        f"Pruning {team.title!r}. Its directory goes back to your team's IC "
        "when it is gone."
    )
