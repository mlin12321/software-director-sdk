"""Wire models for the director API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


AgentRole = Literal["lead_ic", "manager", "reviewer", "worker"]

NodeHealth = Literal[
    "converging",
    "negotiating_split",
    "collision",
    "needs_attention",
    "pruned",
    "done",
]

IsolationMode = Literal["worktree", "shared"]

EscalationKind = Literal["critical_issue", "design_decision"]


class Boundary(BaseModel):
    """A codeowner boundary. All paths and globs are repo-relative."""

    path: str = Field(
        description="Directory the boundary is rooted at; '' is the repository root."
    )
    writable: list[str] = Field(
        default_factory=list, description="Globs the node's agents may write."
    )
    readonly: list[str] = Field(
        default_factory=list, description="Globs the node's agents may only read."
    )


class AgentRef(BaseModel):
    """One agent in a node's pod."""

    role: AgentRole
    conversation_id: str | None = Field(
        default=None,
        description="None until a conversation has been started for this agent.",
    )
    status: str


class EscalationHandoff(BaseModel):
    """A node that tried to settle an escalation, could not, and passed it up."""

    node_id: str
    reason: str = Field(min_length=1, description="Why this node could not settle it.")
    at: str = Field(description="ISO 8601 timestamp.")


class Escalation(BaseModel):
    """Something a node cannot settle itself.

    It goes to the node above, which settles it or passes it up in turn. It
    waits on the human only once every node above the one that raised it has
    passed it up.
    """

    id: str = Field(description="Unique within the node that raised it.")
    kind: EscalationKind
    title: str
    detail: str = ""
    raised_at: str = Field(description="ISO 8601 timestamp.")
    held_by: str | None = Field(
        description=(
            "The node whose agents are working on it now: the parent of the "
            "last node that passed it up. None when it waits on the human."
        )
    )
    handoffs: list[EscalationHandoff] = Field(
        default_factory=list,
        description=(
            "The nodes that passed it up, nearest to the node that raised it "
            "first. Always the unbroken chain from that node's parent."
        ),
    )


class HealthCounts(BaseModel):
    """Number of nodes in each health state."""

    converging: int = 0
    negotiating_split: int = 0
    collision: int = 0
    needs_attention: int = 0
    pruned: int = 0
    done: int = 0


class OrgNodeInput(BaseModel):
    """What a caller supplies for a node; the store derives the rest."""

    id: str
    parent_id: str | None = Field(description="None for the root node.")
    title: str
    boundary: Boundary
    isolation: IsolationMode = Field(
        description=(
            "'worktree': the node has its own git worktree. "
            "'shared': it works in its parent's directory."
        )
    )
    worktree_path: str
    branch: str
    agents: list[AgentRef] = Field(default_factory=list)
    health: NodeHealth
    spec_summary: str = ""
    escalations: list[Escalation] = Field(
        default_factory=list,
        description=(
            "Open escalations this node raised, oldest first, wherever in the "
            "chain above they are held."
        ),
    )


class OrgNode(OrgNodeInput):
    """A node of the org tree, including the fields the store derives."""

    child_ids: list[str] = Field(default_factory=list)
    descendant_counts: HealthCounts = Field(
        default_factory=HealthCounts,
        description=(
            "Health of every node below this one, so a parent can summarize its "
            "whole subtree without the client loading it."
        ),
    )
    escalation_count: int = Field(
        default=0,
        description=(
            "Escalations raised by this node or a node below it that wait on "
            "the human. Those still held by a node are not counted."
        ),
    )


class OrgNodeDetail(BaseModel):
    """A node with the nodes around it."""

    node: OrgNode
    ancestors: list[OrgNode] = Field(description="Root first, excluding the node.")
    children: list[OrgNode]


class Autopsy(BaseModel):
    """Why a pruned attempt failed, kept so successors do not repeat it."""

    node_id: str
    at: str = Field(description="ISO 8601 timestamp of the prune.")
    approach: str
    why_failed: str


class BoundaryMemory(BaseModel):
    """Plain-English rules that every agent working in a boundary inherits."""

    rules: list[str] = Field(default_factory=list)


class InheritedSpec(BaseModel):
    """A spec that applies to a boundary."""

    path: str
    spec: str
    readonly: bool = Field(
        description=(
            "Ancestor specs are invariants; only the boundary's own spec is writable."
        )
    )


class NodeBlackboard(BaseModel):
    """Everything on the blackboard that applies to one node."""

    specs: list[InheritedSpec]
    memory: BoundaryMemory
    autopsies: list[Autopsy]


class TeachRequest(BaseModel):
    """A rule to add to a boundary's permanent memory."""

    rule: str = Field(min_length=1, description="The rule, in plain English.")


class EscalationPathStep(BaseModel):
    """One node on the way from the root to an escalation."""

    id: str
    title: str


class OpenEscalation(BaseModel):
    """An escalation that waits on the human, with the nodes it came up through."""

    node_id: str = Field(description="The node that raised it.")
    escalation: Escalation
    path: list[EscalationPathStep] = Field(
        description="Root first, ending at the node that raised it."
    )


class SnapshotEvent(BaseModel):
    """The part of the tree one subscriber follows.

    ``nodes`` holds the focused node, its ancestors, its direct reports and
    the direct reports of every expanded node; or, for a full subscription,
    the focused node, its ancestors and every node below it.
    """

    type: Literal["snapshot"] = "snapshot"
    focus_id: str
    nodes: list[OrgNode]
    escalations: list[OpenEscalation] = Field(
        default_factory=list,
        description=(
            "Every escalation in the tree that waits on the human, wherever "
            "the subscriber is focused: critical issues first, then longest "
            "waiting first."
        ),
    )


class NodeUpsertEvent(BaseModel):
    """A node was added or changed."""

    type: Literal["node_upsert"] = "node_upsert"
    node: OrgNode


class NodeRemovedEvent(BaseModel):
    """A node was removed."""

    type: Literal["node_removed"] = "node_removed"
    id: str


class EscalationsEvent(BaseModel):
    """The escalations that wait on the human changed."""

    type: Literal["escalations"] = "escalations"
    escalations: list[OpenEscalation] = Field(
        description="The complete new list, in the snapshot's order."
    )


OrgEvent = Annotated[
    SnapshotEvent | NodeUpsertEvent | NodeRemovedEvent | EscalationsEvent,
    Field(discriminator="type"),
]
