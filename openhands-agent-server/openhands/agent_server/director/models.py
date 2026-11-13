"""Wire models for the director API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from openhands.agent_server.models import StartConversationRequest


# A pod has an IC and a Manager. The IC of a team leads it, is a report of
# the team above, and does the code work in its boundary that no sub-team
# has taken on.
AgentRole = Literal["ic", "manager"]

NodeStatus = Literal["idle", "working", "done", "blocked"]

IsolationMode = Literal["worktree", "shared"]

EscalationKind = Literal["critical_issue", "design_decision"]

SplitStatus = Literal["proposed", "refine_requested", "approved"]

# 'sandbox': an agent's shell can write only its team's directory, and its
# file tools are checked. 'checks': only its file tools are checked.
BoundaryEnforcement = Literal["sandbox", "checks"]


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


class SplitChild(BaseModel):
    """A sub-team an IC proposes to hand part of its boundary to."""

    title: str = Field(min_length=1)
    path: str = Field(
        description=(
            "Directory the sub-team owns, repo-relative and strictly inside "
            "the proposing node's boundary."
        )
    )
    writable: list[str] = Field(
        default_factory=list,
        description="Globs under `path` the sub-team may write. Empty: all of it.",
    )
    readonly: list[str] = Field(default_factory=list)
    isolation: IsolationMode = "shared"
    task: str = Field(min_length=1, description="What the sub-team is to deliver.")


class SplitProposal(BaseModel):
    """An IC's proposal to split its boundary, for its Manager to decide."""

    children: list[SplitChild] = Field(min_length=1)
    contracts: str = Field(
        default="",
        description="The interfaces the sub-teams must keep between them.",
    )
    status: SplitStatus = "proposed"
    feedback: str = Field(
        default="", description="What the Manager wants changed, when it asked."
    )
    proposed_at: str = Field(description="ISO 8601 timestamp.")


class StatusCounts(BaseModel):
    """Number of nodes in each status."""

    idle: int = 0
    working: int = 0
    done: int = 0
    blocked: int = 0


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
    status: NodeStatus
    headline: str = Field(
        default="",
        description=(
            "One line on what the team is delivering, written by a model from "
            "the team's task and its latest report."
        ),
    )
    spec_summary: str = ""
    escalations: list[Escalation] = Field(
        default_factory=list,
        description=(
            "Open escalations this node raised, oldest first, wherever in the "
            "chain above they are held."
        ),
    )
    pending_split: SplitProposal | None = Field(
        default=None,
        description="A split of this node's boundary that is not carried out yet.",
    )


class OrgNode(OrgNodeInput):
    """A node of the org tree, including the fields the store derives."""

    child_ids: list[str] = Field(default_factory=list)
    descendant_counts: StatusCounts = Field(
        default_factory=StatusCounts,
        description=(
            "Status of every node below this one, so a parent can summarize its "
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


class OrgSummary(BaseModel):
    """Whether the director has an org, and where it would keep one."""

    root_id: str | None = Field(description="None when no org has been started.")
    node_count: int
    repo_path: str = Field(
        description=(
            "The directory the org works in. When the server was started in a "
            "git submodule that has no org, this is the nearest repository "
            "above it that has one."
        )
    )
    directory: str = Field(
        description=(
            "The directory the server was started for, relative to "
            "`repo_path`; '' when they are the same. A message to the org "
            "that names no path is taken to come from here."
        )
    )
    is_git_repository: bool = Field(
        description="An org can only be started in a git repository."
    )
    fixture: bool = Field(
        description="The tree is a fixed sample; its agents are not running."
    )
    boundary_enforcement: BoundaryEnforcement = Field(
        description=(
            "How far a team is held to its boundary. 'sandbox': the shell of "
            "an agent can write only the directory its team owns, and its "
            "file tools are refused anything else. 'checks': only the file "
            "tools are refused; the server cannot confine a shell."
        )
    )


class CreateOrgRequest(BaseModel):
    """Start an org: a root team that works on ``task``."""

    title: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The root team's name. Left out, the team is first named after "
            "the task's first line and then by a model that summarizes it."
        ),
    )
    task: str = Field(min_length=1, description="What the org is to deliver.")
    conversation: StartConversationRequest = Field(
        description=(
            "How to start the root team's IC. Every other agent of the org is "
            "derived from it. Its workspace must be the director's directory, "
            "and its initial message is ignored: the IC is briefed with `task`."
        )
    )


class RouteMessageRequest(BaseModel):
    """Something the human says to the org, from somewhere in the repository."""

    text: str = Field(min_length=1)
    path: str | None = Field(
        default=None,
        description=(
            "The directory the message is about: relative to the org's "
            "`repo_path`, or absolute inside it (a submodule's directory is "
            "inside the repository that has the submodule). Defaults to the "
            "org summary's `directory`."
        ),
    )
    conversation: StartConversationRequest | None = Field(
        default=None,
        description=(
            "How to start the root team's IC when there is no org yet; the "
            "message then becomes the org's task. See `CreateOrgRequest`. "
            "Ignored when an org exists."
        ),
    )


class RoutedMessage(BaseModel):
    """Where a message to the org went."""

    node_id: str = Field(description="The team that received it.")
    conversation_id: str = Field(description="The conversation of that team's IC.")
    started_org: bool = Field(
        description="There was no org, and the message started one."
    )


class AnswerEscalationRequest(BaseModel):
    """The human's answer to an escalation that waits on them."""

    answer: str = Field(min_length=1)


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
