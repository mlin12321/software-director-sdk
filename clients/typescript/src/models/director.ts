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
  /** The directory the org works in. */
  repo_path: string;
  /** An org can only be started in a git repository. */
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
}

export interface DirectorBoundaryMemory {
  rules: string[];
}

export interface DirectorInheritedSpec {
  path: string;
  spec: string;
  /** Ancestor specs are invariants; only the node's own spec is writable. */
  readonly: boolean;
}

export interface DirectorNodeBlackboard {
  specs: DirectorInheritedSpec[];
  memory: DirectorBoundaryMemory;
  autopsies: DirectorAutopsy[];
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
