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

# 'pending': the IC of the team above has the team's work to review.
# 'unreviewed': the root team's, which has nobody above.
ReviewState = Literal["pending", "changes_requested", "accepted", "unreviewed"]

# 'sandbox': an agent's shell can write only its team's directory, and its
# file tools are checked. 'checks': only its file tools are checked.
BoundaryEnforcement = Literal["sandbox", "checks"]
# How one team is held to its boundary. 'merge': its agent runs its own tools
# and no sandbox holds it, so what it commits outside its directory is
# refused only when its work is merged into the team above.
TeamEnforcement = Literal["sandbox", "checks", "merge"]
# 'acp': an external agent, such as Claude Code or Codex, that runs its own
# tools and reaches the director's through an MCP server.
AgentKind = Literal["openhands", "acp"]
# Why an org's agents are held. 'shutdown': the server was stopped while the
# org worked. 'crash': the server ended without stopping the org first.
PauseReason = Literal["human", "shutdown", "crash"]
# 'run_again': an agent that was running runs again. 'start': a team that
# waited to start is started. 'continue': a step the director had begun is
# finished. 'message': messages are delivered. 'dropped': run state that no
# longer fits the org was put aside. 'cannot_resume': a team that cannot be
# taken up as it is.
Side = Literal["mine", "theirs"]
ConflictKind = Literal["field", "team", "file"]
SyncResult = Literal["up_to_date", "fast_forward", "merged", "conflicts", "pushed"]
BranchResult = Literal["created", "fast_forwarded", "diverged", "pushed"]

RecoveryKind = Literal[
    "run_again", "start", "continue", "message", "dropped", "cannot_resume"
]


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
    kind: AgentKind = "openhands"
    acp_server: str | None = Field(
        default=None,
        description=(
            "Which external agent an `acp` agent is, such as `claude-code` or "
            "`codex`; None when the server could not tell."
        ),
    )


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
    agent_profile: str | None = Field(
        default=None,
        description=(
            "Name of the agent profile the sub-team's agents run on, from the "
            "ones `read_node_context` lists. Leave it out for the same agent "
            "as your own."
        ),
    )


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


class ReviewFinding(BaseModel):
    """One thing a review found wrong in a team's work."""

    path: str = Field(default="", description="The file, relative to the repository.")
    line: int | None = None
    issue: str = Field(min_length=1, description="What is wrong.")
    evidence: str = Field(
        default="", description="What shows it: the code, a failing check's output."
    )


class Review(BaseModel):
    """The review of the work a team reported as done.

    The IC of the team above reviews it. Before that, the team's own IC is
    asked to describe the changes and to review them, and both answers are
    handed to that IC.
    """

    state: ReviewState
    round: int = Field(
        default=1,
        description="1 for the first time this work is reviewed, one more "
        "each time it comes back after changes were requested.",
    )
    summary: str = Field(default="", description="What the team said it delivered.")
    description: str = Field(
        default="",
        description="The team's own account of what it changed, written as "
        "a pull request description; '' when it gave none.",
    )
    automated_ran: bool = Field(
        default=False,
        description="Whether the team's IC, asked to review the changes by "
        "the code review instructions, gave an answer.",
    )
    automated: list[ReviewFinding] = Field(
        default_factory=list, description="What it found."
    )
    findings: list[ReviewFinding] = Field(
        default_factory=list,
        description="What the reviewing IC wants changed, when it asked.",
    )
    at: str = Field(description="ISO 8601 timestamp of the last change.")


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
    base_commit: str = Field(
        default="",
        description=(
            "The commit the team's work is compared with when it is reviewed: "
            "where its checkout was when the team started, or when its work "
            "was last accepted."
        ),
    )
    review: Review | None = Field(
        default=None,
        description="The review of what the team last reported as done.",
    )
    agent_profile: str | None = Field(
        default=None,
        description=(
            "The agent profile the team's agents were started from; None "
            "when they were derived from the team above."
        ),
    )
    boundary_enforcement: TeamEnforcement | None = Field(
        default=None,
        description="How this team is held to its boundary; None until its IC ran.",
    )
    waiting_to_start: bool = Field(
        default=False,
        description=(
            "An agent of the team could not be started yet because the "
            "server runs as many conversations as it may; it starts when "
            "one of them ends."
        ),
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


class Report(BaseModel):
    """Something an agent recorded for itself and the other agent of its
    team: a design decision, a setback, a reflection."""

    at: str = Field(description="ISO 8601 timestamp.")
    role: AgentRole = Field(description="Which agent of the team wrote it.")
    text: str = Field(min_length=1)
    compacted: bool = Field(
        default=False,
        description=(
            "True for the one report a model wrote in place of the earlier "
            "ones, once they had grown long."
        ),
    )


class InheritedSpec(BaseModel):
    """The spec of a boundary."""

    path: str
    spec: str
    readonly: bool = Field(
        description=(
            "False for the spec of the node's own boundary, which its IC writes."
        )
    )


class BoundaryContracts(BaseModel):
    """What the sub-teams of one boundary must keep between them."""

    path: str = Field(description="The boundary that was split.")
    contracts: str
    own: bool = Field(
        description=(
            "True for the contracts between the node's own sub-teams. False "
            "for the ones the team above set, which the node must keep with "
            "the teams beside it."
        )
    )


class NodeMemory(BaseModel):
    """Everything in the team memory that concerns one node."""

    specs: list[InheritedSpec] = Field(
        description="The spec of the node's own boundary, when it has one."
    )
    contracts: list[BoundaryContracts] = Field(default_factory=list)
    reports: list[Report] = Field(
        default_factory=list,
        description="What the team's own two agents recorded, oldest first.",
    )


class BoundaryOwner(BaseModel):
    """A direct report, as the owner of a directory."""

    node_id: str
    title: str
    path: str = Field(description="The directory it owns, relative to the repository.")
    isolation: IsolationMode
    branch: str


class OwnedFile(BaseModel):
    path: str = Field(description="Relative to the repository.")
    owner: str | None = Field(
        description=(
            "The direct report whose boundary the file is in; None for a "
            "file the node kept for itself."
        )
    )


class Collision(BaseModel):
    """A file that the committed work of two teams changed in ways that do
    not merge."""

    path: str
    node_ids: list[str] = Field(
        description="The teams whose checkouts collide, the node itself included."
    )


ChangeKind = Literal["added", "modified", "deleted"]


class ChangedFile(BaseModel):
    """One file a team changed."""

    path: str = Field(description="Relative to the top of the repository.")
    kind: ChangeKind
    added: int | None = Field(description="Lines added; None when not text.")
    removed: int | None = Field(description="Lines removed; None when not text.")
    committed: bool = Field(
        description="False when the file has changes that are not committed."
    )


class TeamChanges(BaseModel):
    """What a team changed in its directory since its base commit."""

    node_id: str
    base: str = Field(
        description="The commit compared with; '' when the team has none yet."
    )
    files: list[ChangedFile] = Field(description="Sorted by path.")
    truncated: bool = Field(description="There were more files than are listed.")


class FileSides(BaseModel):
    """A file at a team's base commit and in its checkout now."""

    path: str
    before: str | None = Field(description="None when the file was not there.")
    after: str | None = Field(description="None when the file is not there.")
    unreadable: bool = Field(
        description="A side is not text or is too large, and is left out."
    )


class EditFileRequest(BaseModel):
    """The whole new content of one file."""

    content: str


class FileEdit(BaseModel):
    """What came of a change made by hand."""

    path: str
    commit: str | None = Field(
        description="The commit that holds it; None when nothing changed."
    )


class NodeOwnership(BaseModel):
    """How a node's boundary is divided among its direct reports."""

    node_id: str
    owners: list[BoundaryOwner]
    files: list[OwnedFile] = Field(
        description=(
            "The files of the node's boundary in its own checkout that git "
            "does not ignore, sorted by path."
        )
    )
    truncated: bool = Field(description="There were more files than are listed.")
    collisions: list[Collision] = Field(
        description=(
            "Found by merging the tips of the checkouts of the node and its "
            "direct reports in memory. Uncommitted changes are not compared."
        )
    )
    collisions_checked: bool = Field(
        description="False when the installed git is too old to look for them."
    )


class MemorySyncRequest(BaseModel):
    """Which remote to share the team memory with."""

    remote: str = Field(default="origin", min_length=1)


class MemorySync(BaseModel):
    """What a pull or a push of the team memory did."""

    remote: str
    head: str | None = Field(
        description="The local team memory's commit afterwards; None when empty."
    )
    changed: bool = Field(description="The local team memory took in remote changes.")
    kept_local: list[str] = Field(
        default_factory=list,
        description=(
            "Specs and contracts both sides had changed. The local text was "
            "kept; the remote text is in the team memory's history."
        ),
    )
    pushed: bool = Field(description="The remote now has the local team memory.")


class DirectorSettings(BaseModel):
    """Limits on the teams an org may still add, and how it starts again.

    The limits are checked when a team splits. Teams that exist are never
    removed for exceeding a limit that was lowered.
    """

    max_agents: int = Field(
        default=64,
        ge=2,
        description=(
            "The most agents a team may have together with every team "
            "below it. Each team has two."
        ),
    )
    max_depth: int = Field(
        default=8,
        ge=1,
        description="The most levels of teams; the root team is level 1.",
    )
    allowed_profiles: list[str] = Field(
        default_factory=list,
        description=(
            "Names of the agent profiles a team may give a sub-team in place "
            "of its own agent. Empty: every team runs on the agent of the "
            "team above."
        ),
    )
    resume_on_start: bool = Field(
        default=False,
        description=(
            "Whether an org that was stopped in the middle of its work takes "
            "it up again as soon as the server is back, without being asked."
        ),
    )
    report_compaction_chars: int = Field(
        default=24_000,
        ge=1_000,
        description=(
            "How many characters of reports one agent may have before a "
            "model rewrites them into one shorter report."
        ),
    )


class DirectorSettingsUpdate(BaseModel):
    """New values for some of the settings; what is left out stays."""

    max_agents: int | None = Field(default=None, ge=2)
    max_depth: int | None = Field(default=None, ge=1)
    allowed_profiles: list[str] | None = None
    resume_on_start: bool | None = None
    report_compaction_chars: int | None = Field(default=None, ge=1_000)


class DirectorSettingsState(DirectorSettings):
    """The settings, beside what the org has now."""

    agents: int = Field(description="How many agents the org has.")
    depth: int = Field(description="How many levels of teams the org has.")
    max_concurrent_runs: int | None = Field(
        description=(
            "How many conversations the server runs at once, the org's "
            "agents among them; teams beyond that wait to start. None when "
            "the director was not told."
        )
    )


class TeamMatch(BaseModel):
    """A team found by its title or its directory."""

    id: str
    title: str
    path: str = Field(description="The directory the team owns; '' at the root.")


class RecoveryItem(BaseModel):
    """One thing a paused org takes up again, or cannot."""

    node_id: str = Field(description="The team it is about; '' for the whole org.")
    title: str
    kind: RecoveryKind
    detail: str = Field(description="What it is, in words for the human.")


class Recovery(BaseModel):
    """What a paused org does when it is resumed."""

    paused: PauseReason | None = Field(
        description="Why the org is paused; None when it is not."
    )
    items: list[RecoveryItem] = Field(
        description=(
            "What resuming does, team by team, and what it cannot do. Empty "
            "when the org is not paused."
        )
    )
    messages_held: int = Field(
        description="How many messages to agents wait for the org to resume."
    )


class OrgConflict(BaseModel):
    """Something both sides of a pull changed, for the human to decide."""

    id: str = Field(description="Names the conflict in a resolve request.")
    kind: ConflictKind = Field(
        description=(
            "`field`: one field of a team was changed on both sides. `team`: "
            "one side removed a team the other changed. `file`: both sides "
            "changed the same lines of a spec or of contracts."
        )
    )
    node_id: str = Field(description="The team; '' for a file of the team_memory.")
    title: str
    field: str = Field(description="The field, or the file's path.")
    mine: str | None = Field(description="What is here; None when removed here.")
    theirs: str | None = Field(
        description="What the remote has; None when removed there."
    )
    choice: Side | None = Field(
        default=None, description="What was chosen; None while it is open."
    )


class OrgMerge(BaseModel):
    """A pull that stopped because both sides changed the same thing.

    Nothing of it has been written to the org. The org stays paused until
    every conflict has a choice, or the merge is abandoned.
    """

    remote: str
    theirs: str = Field(description="The remote's org commit being merged.")
    conflicts: list[OrgConflict]


class BranchSync(BaseModel):
    """What a pull or a push did with the branch of one team's work."""

    node_id: str
    title: str
    branch: str
    result: BranchResult = Field(
        description=(
            "`created`: the branch was not here and now is. `fast_forwarded`: "
            "it took in the other side's commits. `diverged`: both sides have "
            "commits the other lacks; on a pull the remote's are kept at "
            "`ref` and the team's IC is told to merge them, on a push the "
            "branch was not pushed. `pushed`: the remote now has it."
        )
    )
    ref: str | None = Field(
        default=None, description="Where the remote's commits are kept, if diverged."
    )


class OrgSyncRequest(BaseModel):
    """Which remote to exchange the org with."""

    remote: str = Field(default="origin", min_length=1)


class OrgSync(BaseModel):
    """What a pull or a push of the org did."""

    remote: str
    result: SyncResult = Field(
        description=(
            "`up_to_date`: nothing to take in. `fast_forward`: only the "
            "remote had changed. `merged`: both had, and nothing clashed. "
            "`conflicts`: both changed the same thing; see `merge`. "
            "`pushed`: the remote now has what is here."
        )
    )
    head: str | None = Field(description="The local org's commit afterwards.")
    merge: OrgMerge | None = Field(
        default=None, description="The merge waiting for choices, if any."
    )
    memory: MemorySync | None = Field(
        default=None, description="What happened to specs, contracts and reports."
    )
    branches: list[BranchSync] = Field(
        default_factory=list, description="The teams' branches that changed."
    )
    put_aside: list[RecoveryItem] = Field(
        default_factory=list,
        description="Unfinished work dropped because its team was changed.",
    )


class ResolveMergeRequest(BaseModel):
    """Choices for the conflicts of a merge."""

    choices: dict[str, Side] = Field(
        default_factory=dict, description="A side per conflict id."
    )
    rest: Side | None = Field(
        default=None, description="The side for every conflict not named."
    )


class StartAgainRequest(BaseModel):
    """How to start a team whose agents have no conversation here."""

    conversation: StartConversationRequest | None = Field(
        default=None,
        description=(
            "How to start the team's IC. Its workspace is replaced by the "
            "team's own checkout; the team's Manager is derived from it. "
            "Without one the IC is started like the IC of the team above, or "
            "from the team's own agent profile, which needs the team above "
            "to have its IC here."
        ),
    )


class OrgSummary(BaseModel):
    """Whether the director has an org, and where it would keep one."""

    root_id: str | None = Field(description="None when no org has been started.")
    node_count: int
    repo_path: str = Field(
        description=(
            "The repository the org works in: the one its first conversation "
            "was opened in, or the nearest repository above that one which "
            "has it as a submodule and already had an org. Without an org, "
            "the directory the server was started for."
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
        description=(
            "Whether `repo_path` is in a git repository. An org can only be "
            "started in one."
        )
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
    paused: PauseReason | None = Field(
        default=None,
        description=(
            "Why the org's agents are held and its steps and messages wait; "
            "None when the org runs. `GET /org/recovery` says what resuming "
            "does."
        ),
    )
    merging: bool = Field(
        default=False,
        description="A pull stopped on conflicts and waits for choices.",
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
            "derived from it. The org lives in the git repository of its "
            "workspace. Its initial message is ignored: the IC is briefed "
            "with `task`."
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
            "How to start the root team's IC when there is no org yet. If "
            "the git repository of its workspace already has a saved org, "
            "that org is taken up and the message routed in it; otherwise "
            "the message starts an org there and becomes its task. See "
            "`CreateOrgRequest`. Ignored when an org exists."
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
    paused: PauseReason | None = Field(
        default=None, description="Why the org is paused; null while it runs."
    )
    merging: bool = Field(
        default=False,
        description="A pull stopped on conflicts and waits for choices.",
    )
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


class OrgStateEvent(BaseModel):
    """Whether the org is paused, and whether a pull waits for choices.

    Sent whenever either changes; the snapshot has them as they were.
    """

    type: Literal["org_state"] = "org_state"
    paused: PauseReason | None = Field(
        description="Why the org is paused; null while it runs."
    )
    merging: bool = Field(
        description="A pull stopped on conflicts and waits for choices."
    )


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
