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


class SnapshotEvent(BaseModel):
    """The focused node, its ancestors and its direct reports."""

    type: Literal["snapshot"] = "snapshot"
    focus_id: str
    nodes: list[OrgNode]


class NodeUpsertEvent(BaseModel):
    """A node was added or changed."""

    type: Literal["node_upsert"] = "node_upsert"
    node: OrgNode


class NodeRemovedEvent(BaseModel):
    """A node was removed."""

    type: Literal["node_removed"] = "node_removed"
    id: str


OrgEvent = Annotated[
    SnapshotEvent | NodeUpsertEvent | NodeRemovedEvent, Field(discriminator="type")
]
