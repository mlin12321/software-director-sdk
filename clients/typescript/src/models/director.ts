/**
 * Wire types of the Agent Server director API (`/api/director`).
 *
 * Handwritten from `openhands.agent_server.director.models` because the
 * pinned Agent Server release, which the generated schema comes from, does
 * not have the director yet. Field names are the server's, unchanged.
 */

export type DirectorNodeId = string;

/**
 * A pod has an IC and a Manager. A team's IC leads that team, is a report
 * of the team above, and does the code work in its boundary that no
 * sub-team has taken on.
 */
export type DirectorAgentRole = 'ic' | 'manager';

export type DirectorNodeStatus = 'idle' | 'working' | 'done' | 'blocked';

/** `worktree`: own git worktree. `shared`: same directory as the parent. */
export type DirectorIsolationMode = 'worktree' | 'shared';

export type DirectorEscalationKind = 'critical_issue' | 'design_decision';

/** A codeowner boundary. All paths and globs are repo-relative. */
export interface DirectorBoundary {
  /** Directory the boundary is rooted at; `""` is the repository root. */
  path: string;
  writable: string[];
  readonly: string[];
}

export interface DirectorAgentRef {
  role: DirectorAgentRole;
  /** `null` until a conversation has been started for this agent. */
  conversation_id: string | null;
  status: string;
}

/** A node that tried to settle an escalation, could not, and passed it up. */
export interface DirectorEscalationHandoff {
  node_id: DirectorNodeId;
  /** Why this node could not settle it. */
  reason: string;
  /** ISO 8601 timestamp. */
  at: string;
}

/**
 * Something a node cannot settle itself. It goes to the node above, which
 * settles it or passes it up in turn; it waits on the human only once every
 * node above the one that raised it has passed it up.
 */
export interface DirectorEscalation {
  /** Unique within the node that raised it. */
  id: string;
  kind: DirectorEscalationKind;
  title: string;
  detail: string;
  /** ISO 8601 timestamp. */
  raised_at: string;
  /** The node working on it now; `null` when it waits on the human. */
  held_by: DirectorNodeId | null;
  /** The nodes that passed it up, nearest to the one that raised it first. */
  handoffs: DirectorEscalationHandoff[];
}

export type DirectorStatusCounts = Record<DirectorNodeStatus, number>;

export type DirectorSplitStatus = 'proposed' | 'refine_requested' | 'approved';

/** A sub-team an IC proposes to hand part of its boundary to. */
export interface DirectorSplitChild {
  title: string;
  /** Directory the sub-team owns, strictly inside the proposing node's boundary. */
  path: string;
  writable: string[];
  readonly: string[];
  isolation: DirectorIsolationMode;
  /** What the sub-team is to deliver. */
  task: string;
}

/** An IC's proposal to split its boundary, for its Manager to decide. */
export interface DirectorSplitProposal {
  children: DirectorSplitChild[];
  /** The interfaces the sub-teams must keep between them. */
  contracts: string;
  status: DirectorSplitStatus;
  /** What the Manager wants changed, when it asked. */
  feedback: string;
  /** ISO 8601 timestamp. */
  proposed_at: string;
}

/**
 * `pending`: the IC of the team above has the work to review. `unreviewed`:
 * the root team's, which has nobody above.
 */
export type DirectorReviewState = 'pending' | 'changes_requested' | 'accepted' | 'unreviewed';

/** One thing a review found wrong in a team's work. */
export interface DirectorReviewFinding {
  /** The file, relative to the repository; '' when it is about no one file. */
  path: string;
  line: number | null;
  issue: string;
  evidence: string;
}

/** The review of the work a team reported as done. */
export interface DirectorReview {
  state: DirectorReviewState;
  /** 1 the first time; one more each time the work comes back after changes. */
  round: number;
  /** What the team said it delivered. */
  summary: string;
  /** The team's own account of what it changed, written as a pull request
   * description; '' when it gave none. */
  description: string;
  /** Whether the team's IC, asked to review the changes by the code review
   * instructions, gave an answer. */
  automated_ran: boolean;
  /** What it found, for the reviewing IC to verify. */
  automated: DirectorReviewFinding[];
  /** What the reviewing IC wants changed, when it asked. */
  findings: DirectorReviewFinding[];
  /** ISO 8601 timestamp of the last change. */
  at: string;
}

export interface DirectorOrgNode {
  id: DirectorNodeId;
  parent_id: DirectorNodeId | null;
  child_ids: DirectorNodeId[];
  title: string;
  boundary: DirectorBoundary;
  isolation: DirectorIsolationMode;
  worktree_path: string;
  branch: string;
  agents: DirectorAgentRef[];
  status: DirectorNodeStatus;
  /**
   * One line on what the team is delivering, written by a model from the
   * team's task and its latest report.
   */
  headline: string;
  spec_summary: string;
  /** Open escalations this node raised, wherever above they are held. */
  escalations: DirectorEscalation[];
  /** A split of this node's boundary that is not carried out yet. */
  pending_split: DirectorSplitProposal | null;
  /** The commit the team's work is compared with when it is reviewed. */
  base_commit: string;
  /** The review of what the team last reported as done. */
  review: DirectorReview | null;
  /**
   * An agent of the team could not be started yet because the server runs
   * as many conversations as it may; it starts when one of them ends.
   */
  waiting_to_start: boolean;
  /** Status of every node below this one. */
  descendant_counts: DirectorStatusCounts;
  /** Escalations from this node or below that wait on the human. */
  escalation_count: number;
}

export interface DirectorOrgNodeDetail {
  node: DirectorOrgNode;
  /** Root first, excluding the node. */
  ancestors: DirectorOrgNode[];
  children: DirectorOrgNode[];
}

/** Whether the director has an org, and where it would keep one. */
export interface DirectorOrgSummary {
  /** `null` when no org has been started. */
  root_id: DirectorNodeId | null;
  node_count: number;
  /**
   * The repository the org works in: the one its first conversation was
   * opened in, or the nearest repository above that one which has it as a
   * submodule and already had an org. Without an org, the directory the
   * server was started for.
   */
  repo_path: string;
  /**
   * The directory the server was started for, relative to `repo_path`; `''`
   * when they are the same. A message to the org that names no path is taken
   * to come from here.
   */
  directory: string;
  /** Whether `repo_path` is in a git repository. An org needs one. */
  is_git_repository: boolean;
  /** The tree is a fixed sample; its agents are not running. */
  fixture: boolean;
  boundary_enforcement: DirectorBoundaryEnforcement;
}

/**
 * How far a team is held to its boundary. `sandbox`: the shell of an agent
 * can write only the directory its team owns, and its file tools are refused
 * anything else. `checks`: only the file tools are refused; the server
 * cannot confine a shell.
 */
export type DirectorBoundaryEnforcement = 'sandbox' | 'checks';

/** Where a message to the org went. */
export interface DirectorRoutedMessage {
  /** The team that received it. */
  node_id: DirectorNodeId;
  /** The conversation of that team's IC. */
  conversation_id: string;
  /** There was no org, and the message started one. */
  started_org: boolean;
}

export interface DirectorEscalationPathStep {
  id: DirectorNodeId;
  title: string;
}

/** An escalation that waits on the human, with the nodes it came up through. */
export interface DirectorOpenEscalation {
  /** The node that raised it. */
  node_id: DirectorNodeId;
  escalation: DirectorEscalation;
  /** Root first, ending at the node that raised it. */
  path: DirectorEscalationPathStep[];
}

export interface DirectorAutopsy {
  node_id: DirectorNodeId;
  at: string;
  approach: string;
  why_failed: string;
  /** The git ref that holds the uncommitted changes the prune discarded. */
  saved_ref?: string | null;
}

export interface DirectorBoundaryMemory {
  rules: string[];
}

export interface DirectorInheritedSpec {
  path: string;
  spec: string;
  /** False for the spec of the node's own boundary, which its IC writes. */
  readonly: boolean;
}

/** What the sub-teams of one boundary must keep between them. */
export interface DirectorBoundaryContracts {
  /** The boundary that was split. */
  path: string;
  contracts: string;
  /**
   * True for the contracts between the node's own sub-teams; false for the
   * ones the team above set, which the node keeps with the teams beside it.
   */
  own: boolean;
}

export interface DirectorNodeBlackboard {
  /** The spec of the node's own boundary, when it has one. */
  specs: DirectorInheritedSpec[];
  contracts?: DirectorBoundaryContracts[];
  memory: DirectorBoundaryMemory;
  autopsies: DirectorAutopsy[];
}

/** A direct report, as the owner of a directory. */
export interface DirectorBoundaryOwner {
  node_id: DirectorNodeId;
  title: string;
  /** The directory it owns, relative to the repository. */
  path: string;
  isolation: DirectorIsolationMode;
  branch: string;
}

export interface DirectorOwnedFile {
  /** Relative to the repository. */
  path: string;
  /** The direct report that owns the file; null when the node kept it. */
  owner: DirectorNodeId | null;
}

/** A file the committed work of two teams changed in ways that do not merge. */
export interface DirectorCollision {
  path: string;
  /** The teams whose checkouts collide, the node itself included. */
  node_ids: DirectorNodeId[];
}

/** How a node's boundary is divided among its direct reports. */
export interface DirectorNodeOwnership {
  node_id: DirectorNodeId;
  owners: DirectorBoundaryOwner[];
  /** The unignored files of the node's boundary in its own checkout, by path. */
  files: DirectorOwnedFile[];
  truncated: boolean;
  /** From the tips of the checkouts; uncommitted changes are not compared. */
  collisions: DirectorCollision[];
  /** False when the server's git is too old to look for collisions. */
  collisions_checked: boolean;
}

/**
 * What `GET /api/director/org/tree-stream` sends.
 *
 * A `snapshot` holds the focused node, its ancestors, its direct reports and
 * the direct reports of every expanded node (or, for a full stream, every
 * node below the focused one), plus every escalation in the tree that waits
 * on the human, wherever the stream is focused. `node_upsert` and `node_removed` are limited to
 * those nodes. `escalations` carries the whole new list.
 */
export type DirectorOrgEvent =
  | {
      type: 'snapshot';
      focus_id: DirectorNodeId;
      nodes: DirectorOrgNode[];
      escalations: DirectorOpenEscalation[];
    }
  | { type: 'node_upsert'; node: DirectorOrgNode }
  | { type: 'node_removed'; id: DirectorNodeId }
  | { type: 'escalations'; escalations: DirectorOpenEscalation[] };

/**
 * Limits on the teams an org may still add. They are checked when a team
 * splits; teams that exist are never removed for exceeding one.
 */
export interface DirectorSettings {
  /** The most agents a team may have together with every team below it. */
  max_agents: number;
  /** The most levels of teams; the root team is level 1. */
  max_depth: number;
}

/** The settings, beside what the org has now. */
export interface DirectorSettingsState extends DirectorSettings {
  agents: number;
  depth: number;
  /**
   * How many conversations the server runs at once; teams beyond that wait
   * to start. `null` when the director was not told.
   */
  max_concurrent_runs: number | null;
}

/** A team found by its title or its directory. */
export interface DirectorTeamMatch {
  id: DirectorNodeId;
  title: string;
  /** The directory the team owns; '' at the root. */
  path: string;
}

/** What a pull or a push of the blackboard did. */
export interface DirectorBlackboardSync {
  remote: string;
  /** The local blackboard's commit afterwards; `null` when it is empty. */
  head: string | null;
  /** The local blackboard took in remote changes. */
  changed: boolean;
  /**
   * Specs and contracts both sides had changed. The local text was kept;
   * the remote text is in the blackboard's history.
   */
  kept_local: string[];
  /** The remote now has the local blackboard. */
  pushed: boolean;
}
