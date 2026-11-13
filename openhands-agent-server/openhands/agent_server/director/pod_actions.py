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
from openhands.agent_server.director.git_blackboard import SPEC_FILE
from openhands.agent_server.director.models import (
    AgentRole,
    EscalationKind,
    OrgNode,
    Review,
    ReviewFinding,
    SplitChild,
    SplitProposal,
)
from openhands.agent_server.director.service import (
    DirectorService,
    agent_conversation,
)
from openhands.agent_server.director.split import normalize_split
from openhands.sdk.errors import Conflict, Invalid, NotFound


DeclaredStatus = Literal["working", "done", "blocked"]


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
    store = service.store
    node = caller.node
    blackboard = service.blackboard
    above = store.get(node.parent_id) if node.parent_id else None
    specs, contracts = blackboard.applies_to(
        node.boundary.path, above.boundary.path if above else None
    )
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
        for other in store.all()
        for escalation in other.escalations
        if escalation.held_by == node.id
    ]
    context = {
        "you": {"role": caller.role, "node_id": node.id, "team": node.title},
        "team_above": node.parent_id,
        "status": node.status,
        "boundary": node.boundary.model_dump(),
        "workspace": {
            "path": node.worktree_path,
            "branch": node.branch,
            "isolation": node.isolation,
        },
        "spec": specs[0].spec if specs else None,
        "contracts_with_the_teams_beside_you": [
            agreed.contracts for agreed in contracts if not agreed.own
        ],
        "contracts_between_your_sub_teams": [
            agreed.contracts for agreed in contracts if agreed.own
        ],
        "rules": blackboard.inherited_rules(node.boundary.path),
        "failed_here_before": [
            {"approach": autopsy.approach, "why_failed": autopsy.why_failed}
            for autopsy in blackboard.autopsies(node.boundary.path)
        ],
        "sub_teams": [
            {
                "node_id": child.id,
                "team": child.title,
                "path": child.boundary.path,
                "status": child.status,
                "headline": child.headline,
                "summary": child.spec_summary,
                "review": child.review.state if child.review else None,
            }
            for child in store.children(node.id)
        ],
        "review_of_your_work": node.review.model_dump() if node.review else None,
        "pending_split": (
            node.pending_split.model_dump() if node.pending_split else None
        ),
        "escalations_you_raised": [
            {
                "escalation_id": escalation.id,
                "title": escalation.title,
                "with": escalation.held_by or "the human",
            }
            for escalation in node.escalations
        ],
        "escalations_for_your_manager_to_settle": to_settle,
    }
    return json.dumps(context, indent=2)


def write_spec(service: DirectorService, caller: Caller, content: str) -> str:
    """Replace the spec of the caller's boundary."""
    _only(caller, "ic", "write its spec")
    if not content.strip():
        raise Invalid("A spec cannot be empty")
    path = caller.node.boundary.path
    service.blackboard.write(path, SPEC_FILE, content)
    summary = content.strip().splitlines()[0].lstrip("# ").strip()[:200]
    service.store.update_node(caller.node.id, spec_summary=summary)
    return f"Saved the spec of {path or 'the repository'}."


def propose_split(
    service: DirectorService,
    caller: Caller,
    children: list[SplitChild],
    contracts: str,
) -> str:
    """Put a split of the caller's boundary before its Manager."""
    _only(caller, "ic", "propose a split")
    node = caller.node
    if node.pending_split and node.pending_split.status != "refine_requested":
        raise Conflict(
            "Your team already has a split waiting; it must be decided or "
            "carried out first"
        )
    store = service.store
    proposal = SplitProposal(
        children=normalize_split(node, store.children(node.id), children),
        contracts=contracts,
        proposed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    store.update_node(node.id, pending_split=proposal)
    listing = "\n".join(
        f"- {child.title}: {child.path} ({child.isolation}). Task: {child.task}"
        for child in proposal.children
    )
    service.notify(
        agent_conversation(node, "manager"),
        f"Your IC proposes to split {node.boundary.path or 'the repository'} "
        f"into these sub-teams:\n{listing}\n\nContracts between them:\n"
        f"{contracts.strip() or '(none given)'}\n\n"
        "Decide with decide_split: approve, or ask for a refinement.",
    )
    return "Proposed. Your Manager decides; you will be told the outcome."


def decide_split(
    service: DirectorService,
    caller: Caller,
    decision: Literal["approve", "refine"],
    feedback: str,
) -> str:
    """Approve the split the caller's IC proposed, or ask for a refinement."""
    _only(caller, "manager", "decide a split")
    node = caller.node
    proposal = node.pending_split
    if proposal is None or proposal.status != "proposed":
        raise Conflict("Your IC has no split waiting for a decision")
    if decision == "refine":
        if not feedback.strip():
            raise Invalid("Asking for a refinement needs feedback on what to change")
        service.store.update_node(
            node.id,
            pending_split=proposal.model_copy(
                update={"status": "refine_requested", "feedback": feedback.strip()}
            ),
        )
        service.notify(
            agent_conversation(node, "ic"),
            "Your Manager asks you to refine the split before it is approved:\n"
            f"{feedback.strip()}\nPropose it again, or do the work yourself.",
        )
        return "Asked the IC to refine the split."
    service.store.update_node(
        node.id, pending_split=proposal.model_copy(update={"status": "approved"})
    )
    service.carry_out_split(node.id)
    return (
        f"Approved. {len(proposal.children)} sub-teams are being started; they "
        "appear under read_node_context."
    )


def raise_escalation(
    service: DirectorService,
    caller: Caller,
    kind: EscalationKind,
    title: str,
    detail: str,
) -> str:
    """Send something the caller's team cannot settle to the team above."""
    if not title.strip():
        raise Invalid("An escalation needs a title")
    store = service.store
    node = caller.node
    escalation = store.raise_escalation(
        node.id, uuid4().hex[:8], kind=kind, title=title.strip(), detail=detail.strip()
    )
    if escalation.held_by is None:
        return (
            f"Raised as {escalation.id}. Nobody is above your team: it waits "
            "on the human."
        )
    _hand_to(service, escalation.held_by, node, escalation.id, title, detail, [])
    return (
        f"Raised as {escalation.id}. It is with the Manager of the team above; "
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
    """Settle an escalation the caller's team holds, or pass it one team up."""
    _only(caller, "manager", "resolve an escalation")
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
    if action == "settle":
        store.settle_escalation(node_id, escalation_id)
        service.notify(
            agent_conversation(raiser, "ic"),
            f"The Manager of {caller.node.title!r} settled your escalation "
            f"{escalation.title!r}:\n{text.strip()}",
        )
        return "Settled; the team that raised it has your answer."
    passed = store.pass_up_escalation(node_id, escalation_id, reason=text)
    if passed.held_by is None:
        return "Passed up. Nobody is above your team: it now waits on the human."
    _hand_to(
        service,
        passed.held_by,
        raiser,
        escalation_id,
        escalation.title,
        escalation.detail,
        [f"{handoff.node_id}: {handoff.reason}" for handoff in passed.handoffs],
    )
    return "Passed up to the Manager of the team above."


def _hand_to(
    service: DirectorService,
    holder_id: str,
    raiser: OrgNode,
    escalation_id: str,
    title: str,
    detail: str,
    passed_up: list[str],
) -> None:
    holder = service.store.get(holder_id)
    if holder is None:
        return
    history = "".join(f"\nPassed up by {entry}" for entry in passed_up)
    service.notify(
        agent_conversation(holder, "manager"),
        f"Team {raiser.title!r} (node {raiser.id}) escalates {title.strip()!r} "
        f"(escalation {escalation_id}):\n{detail.strip() or '(no detail)'}{history}\n"
        "Settle it with resolve_escalation if you can; pass it up only with "
        "the reason you cannot.",
    )


def set_status(
    service: DirectorService, caller: Caller, status: DeclaredStatus, summary: str
) -> str:
    """Declare the state of the caller's team.

    Done work goes to the IC of the team above for review; a block is told
    to the Manager there.
    """
    store = service.store
    node = caller.node
    parent = store.get(node.parent_id) if node.parent_id else None
    summary = summary.strip()
    review = node.review
    if status == "done":
        if parent is not None:
            left = uncommitted(Path(node.worktree_path), node.boundary.path)
            if left:
                raise Conflict(
                    "Commit your team's work before you report it done. Not "
                    f"committed: {', '.join(left[:10])}"
                    + (f" and {len(left) - 10} more" if len(left) > 10 else "")
                )
        again = review is not None and review.state == "changes_requested"
        review = Review(
            state="unreviewed" if parent is None else "pending",
            round=review.round + 1 if review is not None and again else 1,
            summary=summary,
            at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    elif review is not None and review.state == "pending":
        review = None
    node = store.update_node(node.id, status=status, review=review)
    if status == "done":
        service.request_review(node.id)
    if parent is not None and status == "blocked":
        service.notify(
            agent_conversation(parent, "manager"),
            f"Team {node.title!r} (node {node.id}) reports it is blocked."
            + (f"\n{summary}" if summary else ""),
        )
    if summary:
        service.refresh_headline(node.id, f"it is {status}. {summary}")
    if status == "done" and parent is not None:
        return (
            "Your team is now done. The IC of the team above reviews the work; "
            "you are told if it wants changes."
        )
    return f"Your team is now {status}."


def review_team(
    service: DirectorService,
    caller: Caller,
    node_id: str,
    decision: Literal["accept", "reject"],
    findings: list[ReviewFinding],
    needed_above: str,
) -> str:
    """Accept the work a team directly below reported done, or send it back."""
    _only(caller, "ic", "review the work of a team")
    team = service.store.get(node_id)
    if team is None or team.parent_id != caller.node.id:
        raise Invalid(f"{node_id} is not a team directly below yours")
    if team.review is None or team.review.state != "pending":
        raise Conflict(f"Team {node_id} has no work waiting for your review")
    if decision == "reject":
        if not findings:
            raise Invalid("Say what must change before you can accept the work")
        if service.request_changes(node_id, findings):
            return (
                f"Sent back for the last time: {team.title!r} did not get its "
                "work accepted and is now stopped and blocked."
            )
        return f"Sent back to {team.title!r} with what must change."
    service.accept_review(node_id, needed_above.strip())
    return (
        f"Accepting the work of {team.title!r}. You are told if it cannot be "
        "brought into your checkout."
    )


def prune_team(
    service: DirectorService,
    caller: Caller,
    node_id: str,
    approach: str,
    why_failed: str,
    replacement_task: str,
) -> str:
    """Remove a team directly below the caller's, keeping why it failed."""
    _only(caller, "manager", "prune a team")
    if not why_failed.strip():
        raise Invalid("Say why the team failed; whoever works there next reads it")
    team = service.store.get(node_id)
    if team is None or team.parent_id != caller.node.id:
        raise Invalid(f"{node_id} is not a team directly below yours")
    service.prune(
        node_id,
        approach=approach.strip(),
        why_failed=why_failed.strip(),
        replacement_task=replacement_task.strip() or None,
    )
    if replacement_task.strip():
        return f"Pruning {team.title!r}; a fresh team takes over its directory."
    return (
        f"Pruning {team.title!r}. Its directory goes back to your team's IC "
        "when it is gone."
    )
