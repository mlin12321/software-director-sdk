/**
 * Wire types of the Agent Server director API (`/api/director`).
 *
 * Handwritten from `openhands.agent_server.director.models` because the
 * pinned Agent Server release, which the generated schema comes from, does
 * not have the director yet. Field names are the server's, unchanged.
 */

export type DirectorNodeId = string;

export type DirectorAgentRole = 'lead_ic' | 'manager' | 'reviewer' | 'worker';

export type DirectorNodeHealth =
  'converging' | 'negotiating_split' | 'collision' | 'needs_attention' | 'pruned' | 'done';

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

export type DirectorHealthCounts = Record<DirectorNodeHealth, number>;

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
  health: DirectorNodeHealth;
  spec_summary: string;
  /** Open escalations this node raised, wherever above they are held. */
  escalations: DirectorEscalation[];
  /** Health of every node below this one. */
  descendant_counts: DirectorHealthCounts;
  /** Escalations from this node or below that wait on the human. */
  escalation_count: number;
}

export interface DirectorOrgNodeDetail {
  node: DirectorOrgNode;
  /** Root first, excluding the node. */
  ancestors: DirectorOrgNode[];
  children: DirectorOrgNode[];
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
