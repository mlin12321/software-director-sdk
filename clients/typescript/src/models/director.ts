/**
 * Wire types of the Agent Server director API (`/api/director`).
 *
 * Handwritten from `openhands.agent_server.director.models` because the
 * pinned Agent Server release, which the generated schema comes from, does
 * not have the director yet. Field names are the server's, unchanged.
 */

export type DirectorNodeId = string;

/**
 * The member of a team an agent is, by the name its team's type gives it.
 * The types the director ships have an IC (`ic`) and a Manager (`manager`).
 */
export type DirectorAgentRole = string;

/**
 * An IC in hand-over mode: beside its main conversation it has one the
 * director started to pass on what removed teams left, with a context of its
 * own.
 */
export interface DirectorHandOverMode {
  conversation_id: string;
  status: string;
}

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
  /**
   * Set on an IC that is, or last was, in hand-over mode.
   */
  hand_over: DirectorHandOverMode | null;
  /** `null` until a conversation has been started for this agent. */
  conversation_id: string | null;
  status: string;
  /**
   * Why the agent is `blocked` or `lost`, in words for the human; `null`
   * when it is neither or the server could not tell.
   */
  reason: string | null;
  kind: DirectorAgentKind;
  /**
   * Which external agent an `acp` agent is, such as `claude-code` or
   * `codex`; `null` when the server could not tell.
   */
  acp_server: string | null;
  /**
   * The model the agent runs on, as it is configured; `null` for an `acp`
   * agent left on its own default, and until the agent started.
   */
  model: string | null;
}

/**
 * `acp`: an external agent, such as Claude Code or Codex, that runs its own
 * tools and reaches the director's through an MCP server.
 */
export type DirectorAgentKind = 'openhands' | 'acp';

/**
 * How one team is held to its boundary. `merge`: its agent runs its own
 * tools and no sandbox holds it, so what it commits outside its directory is
 * refused only when its work is merged into the team above.
 */
export type DirectorTeamEnforcement = 'sandbox' | 'checks' | 'merge';

/** A team that tried to settle an escalation, could not, and passed it up. */
export interface DirectorEscalationHandoff {
  node_id: DirectorNodeId;
  /** Why this team could not settle it. */
  reason: string;
  /** ISO 8601 timestamp. */
  at: string;
}

/**
 * Something a team cannot settle itself. It goes to the team above, which
 * settles it or passes it up in turn; it waits on the human once every team
 * above the one that raised it in its session has passed it up, which for
 * the team the session was opened at is at once.
 */
export interface DirectorEscalation {
  /** Unique within the team that raised it. */
  id: string;
  kind: DirectorEscalationKind;
  title: string;
  detail: string;
  /** ISO 8601 timestamp. */
  raised_at: string;
  /**
   * Which agent of the team raised it. The same agent of each team above
   * handles it, and the answer goes back to this one.
   */
  raised_by: DirectorAgentRole;
  /** The team working on it now; `null` when it waits on the human. */
  held_by: DirectorNodeId | null;
  /** The teams that passed it up, nearest to the one that raised it first. */
  handoffs: DirectorEscalationHandoff[];
}

export type DirectorSplitStatus = 'proposed' | 'approved';

/** A sub-team an IC proposes to hand part of its boundary to. */
export interface DirectorSplitChild {
  title: string;
  /** Directory the sub-team owns, strictly inside the proposing team's boundary. */
  path: string;
  writable: string[];
  readonly: string[];
  isolation: DirectorIsolationMode;
  /** What the sub-team is to deliver. */
  task: string;
  /** Why the proposer wants this team, when it said. */
  reason?: string;
  /**
   * Name of the agent profile the sub-team's agents run on; `null` for the
   * same agent as the proposing team's.
   */
  agent_profile: string | null;
  /**
   * The type of the sub-team: the name of the YAML file its agents are told
   * from what the team is for.
   */
  agent_type: string;
}

/**
 * The sub-teams an IC proposed in a message to its Manager, which approves
 * them as written or answers with a message.
 */
export interface DirectorSplitProposal {
  children: DirectorSplitChild[];
  /** The interfaces the sub-teams must keep between them. */
  contracts: string;
  /** Why the proposer wants this make-up of teams, when it said. */
  reason?: string;
  status: DirectorSplitStatus;
  /**
   * The sub-teams the team has now that the proposed ones take the place of.
   * Their own sub-teams stay.
   */
  replaces?: DirectorNodeId[];
  /** The sub-teams the team has now that the IC said are to stay as they are. */
  keeps?: DirectorNodeId[];
  /** ISO 8601 timestamp. */
  proposed_at: string;
}

/** `pending`: the IC of the team above has the work to review. */
export type DirectorReviewState = 'pending' | 'changes_requested' | 'accepted';

/** One thing a review found wrong in a team's work. */
export interface DirectorReviewFinding {
  /** The file, relative to the repository; '' when it is about no one file. */
  path: string;
  line: number | null;
  issue: string;
  evidence: string;
}

/** The review of the work a team submitted. */
export interface DirectorReview {
  state: DirectorReviewState;
  /** Its place in the order work was submitted in; 0 when it has none. */
  seq?: number;
  /** 1 the first time; one more each time the work comes back after changes. */
  round: number;
  /** What the team said it delivered. */
  summary: string;
  /** The team's own account of what it changed, written as a pull request
   * description; '' when it gave none. */
  description: string;
  /** What the reviewing IC wants changed, when it asked. */
  findings: DirectorReviewFinding[];
  /** Whether the work went up without a review of its own: all of it was
   * accepted work of the team's sub-teams. */
  passed_up?: boolean;
  /** The commit the team's checkout was on when it submitted. */
  commit?: string;
  /** ISO 8601 timestamp of the last change. */
  at: string;
}

/**
 * A team of the org tree: what an agent that works on it is briefed from.
 * Nobody works at one; the work is at a session's place (`DirectorSessionTeam`).
 */
export interface DirectorOrgNode {
  id: DirectorNodeId;
  parent_id: DirectorNodeId | null;
  child_ids: DirectorNodeId[];
  title: string;
  boundary: DirectorBoundary;
  isolation: DirectorIsolationMode;
  /**
   * One line on what the team is delivering, written by a model from the
   * team's task and its latest report.
   */
  headline: string;
  spec_summary: string;
  /**
   * Specs that teams before this one left in its directory. They are old
   * until the team's IC writes its own spec.
   */
  old_specs: DirectorOldSpec[];
  /**
   * The agent profile the team's agents were started from; `null` when
   * they were derived from the team above.
   */
  agent_profile: string | null;
  /**
   * The type of the team: what its agents are told the team is for and how
   * its work is reviewed. It changes nothing the runtime enforces.
   */
  agent_type: string;
  /**
   * When the team was set up. Two teams with one id are the same team only
   * when this is the same: an id comes from the directory, so a team that
   * replaces another has its id.
   */
  since?: string;
}

export interface DirectorOrgNodeDetail {
  node: DirectorOrgNode;
  /** Root first, excluding the team. */
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
  boundary_enforcement: DirectorBoundaryEnforcement;
  /**
   * Why the org's agents are held and its steps and messages wait; `null`
   * when the org runs.
   */
  paused: DirectorPauseReason | null;
  /** A pull stopped on conflicts and waits for choices. */
  merging?: boolean;
}

/**
 * Why an org is paused. `shutdown`: the server was stopped while the org
 * worked. `crash`: the server ended without stopping the org first.
 */
export type DirectorPauseReason = 'human' | 'shutdown' | 'crash';

/**
 * `run_again`: an agent that was running runs again. `start`: a team that
 * waited to start is started. `continue`: a step the director had begun is
 * finished. `message`: messages are delivered. `dropped`: run state that no
 * longer fits the org was put aside. `cannot_resume`: a team that cannot be
 * taken up as it is.
 */
export type DirectorRecoveryKind =
  'run_again' | 'start' | 'continue' | 'message' | 'dropped' | 'cannot_resume';

/** One thing a paused org takes up again, or cannot. */
export interface DirectorRecoveryItem {
  /** The place it is about (a team in a session); `''` for the whole org. */
  node_id: DirectorNodeId;
  title: string;
  kind: DirectorRecoveryKind;
  /** What it is, in words for the human. */
  detail: string;
}

/** What a paused org does when it is resumed. */
export interface DirectorRecovery {
  /** Why the org is paused; `null` when it is not. */
  paused: DirectorPauseReason | null;
  /** Empty when the org is not paused. */
  items: DirectorRecoveryItem[];
  /** How many messages to agents wait for the org to resume. */
  messages_held: number;
}

export type DirectorSide = 'mine' | 'theirs';

/**
 * Something both sides of a pull changed. `field`: one field of a team.
 * `team`: one side removed a team the other changed. `file`: both changed the
 * same lines of a spec or of contracts.
 */
export interface DirectorOrgConflict {
  /** Names the conflict in a resolve request. */
  id: string;
  kind: 'field' | 'team' | 'file';
  /** The team; '' for a file of the team_memory. */
  node_id: string;
  title: string;
  /** The field, or the file's path. */
  field: string;
  /** What is here; `null` when removed here. */
  mine: string | null;
  /** What the remote has; `null` when removed there. */
  theirs: string | null;
  /** `null` while it is open. */
  choice: DirectorSide | null;
}

/** A pull that stopped because both sides changed the same thing. */
export interface DirectorOrgMerge {
  remote: string;
  theirs: string;
  conflicts: DirectorOrgConflict[];
}

/** What a pull or a push of the org did. */
export interface DirectorOrgSync {
  remote: string;
  result: 'up_to_date' | 'fast_forward' | 'merged' | 'conflicts' | 'pushed';
  /** The local org's commit afterwards. */
  head: string | null;
  /** The merge waiting for choices, when `result` is `conflicts`. */
  merge: DirectorOrgMerge | null;
  memory: DirectorMemorySync | null;
}

/** Choices for the conflicts of a merge. */
export interface DirectorResolveMerge {
  /** A side per conflict id. */
  choices?: Record<string, DirectorSide>;
  /** The side for every conflict not named. */
  rest?: DirectorSide | null;
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
  /** The place that received it (a team in a session). */
  node_id: DirectorNodeId;
  /** The title of that team. */
  team: string;
  /** The conversation of that team's IC. */
  conversation_id: string;
  /** There was no org, and the message started one. */
  started_org: boolean;
  /** How the team was found. */
  chosen_by: DirectorChosenBy;
  /** Why this team, in a sentence; may be empty. */
  reason: string;
}

/**
 * How the team a message went to was found. `path`: it owns the directory
 * the message came from. `named`: the director chose it by the files the
 * text names. `model`: the director chose it by what a model made of the
 * text. `top`: the director was asked to choose and was not sure of any team
 * below the one the directory gives.
 */
export type DirectorChosenBy = 'path' | 'named' | 'model' | 'top';

/**
 * What a session has at one team of the org: agents, a checkout, a review
 * and escalations of its own. It is no team of the org tree. `id` is its
 * own (`<team id>@<session id>`), `parent_id` what the session has at the
 * team above, or for the team it was opened at its Division Lead place,
 * and `child_ids` what it has at the team's sub-teams.
 */
export interface DirectorSessionTeam extends DirectorOrgNode {
  /** The session. */
  session: string;
  /** The team of the org. */
  team_id: DirectorNodeId;
  worktree_path: string;
  branch: string;
  /** Empty while the team was set up and has been given no task yet. */
  agents: DirectorAgentRef[];
  status: DirectorNodeStatus;
  /** Open escalations this place raised, wherever above they are held. */
  escalations: DirectorEscalation[];
  /** A split of this team's boundary that is not carried out yet. */
  pending_split: DirectorSplitProposal | null;
  /** The commit the team's work is compared with when it is reviewed. */
  base_commit: string;
  /** The review of what the team last submitted. */
  review: DirectorReview | null;
  /**
   * Set while the IC of the team above has stopped the team: its agents and
   * those of every team below it are at rest and nothing is sent to them.
   */
  stopped?: DirectorStopped | null;
  /**
   * An agent of the team could not be started yet because the server runs
   * as many conversations as it may; it starts when one of them ends.
   */
  waiting_to_start: boolean;
  /** How this team is held to its boundary; `null` until its IC ran. */
  boundary_enforcement: DirectorTeamEnforcement | null;
  /**
   * Whether this is the session's Division Lead place: the team
   * above the one the session was opened at (the top team itself for a
   * session opened there), whose IC only reviews here.
   */
  division_lead: boolean;
  /**
   * On a session's Division Lead place: how the work the session
   * submitted becomes the org's, with what waits to be settled first.
   */
  merge?: DirectorMerge | null;
}

/**
 * What a session and the org each made something else of. `code`: a file
 * both changed. `spec`, `contracts`: the text of one directory. `team`: a
 * sub-team.
 */
export type DirectorMergeKind = 'code' | 'spec' | 'contracts' | 'team';

/** Who settles it: the IC that reviews the work, or its Manager. */
export type DirectorMergeGate = 'ic' | 'manager';

/** One thing a session and the org each made something else of. */
export interface DirectorMergeItem {
  kind: DirectorMergeKind;
  /**
   * What it is about: a file for `code`, a directory for `spec` and
   * `contracts`, a team's id for `team`.
   */
  key: string;
  gate: DirectorMergeGate;
  /** What a team is called. */
  title?: string;
  /** What the session has: a text, or a team in words. */
  session?: string;
  /** What the org has there now. */
  org?: string;
  /**
   * What was decided: the org's stays, the session's holds, or `text` does.
   * `null` while nobody has decided.
   */
  take?: 'org' | 'session' | 'text' | null;
  /** What holds when `take` is `text`. */
  text?: string;
  /**
   * The team of the org a team would be put under, when the team the
   * session has above it is not in the org any more.
   */
  moved_to?: string | null;
  /**
   * The waiting team contracts come with: they are the session's only if
   * that team becomes the org's.
   */
  with_team?: string | null;
  /** The team of the org its Manager decided to put a team under. */
  under?: string | null;
  /** What its Manager decided to call a team. */
  named?: string | null;
}

/**
 * How the work a session submitted becomes the org's. Its code and specs go
 * in when the reviewing IC accepts; its teams and contracts when it is their
 * turn and the Division Lead Manager has settled what waits of them.
 */
export interface DirectorMerge {
  /** Its place in the order: work submitted earlier goes in first. */
  seq: number;
  items?: DirectorMergeItem[];
  /** Whether the code of its accepted work is in the repository. */
  code_landed?: boolean;
  /**
   * Whether its teams and contracts wait for those of a session that
   * submitted earlier. What waits of them is not known yet.
   */
  org_queued?: boolean;
  /** What names the state its Manager was last asked of. */
  asked?: string;
  /** When its Manager was last asked; '' when it was not. */
  asked_at?: string;
  /** Whether its Manager was asked again for not answering. */
  reminded?: boolean;
  /** What the Manager said in settling. */
  note?: string;
}

/** What agents have used of their models. */
export interface DirectorUsage {
  input_tokens: number;
  output_tokens: number;
  /**
   * In US dollars: what the agents' models reported and, where one reported
   * none, what its tokens cost by the public price list.
   */
  cost: number;
  /**
   * How many of the tokens are of models that reported no cost and have no
   * price in the list. They are not in `cost`, so a limit on cost does not
   * hold them.
   */
  unpriced_tokens: number;
}

/**
 * A line of work at a team. All of an org's work is done in sessions: a team
 * of the org tree is reference only, and a session has agents and a checkout
 * of its own at the team it was opened at and, once they are given a task
 * in it, at the teams below. Its work is reviewed by its Division Lead IC
 * and, once accepted, merged into the repository's
 * checkout, where it becomes the org's.
 */
export interface DirectorSession {
  id: string;
  title: string;
  /** The team of the org it was opened at. */
  team_id: DirectorNodeId;
  /** The title of that team. */
  team: string;
  /** The team whose IC reviews its work. */
  above_id: DirectorNodeId;
  /** What the human asked of it. */
  request: string;
  /** ISO 8601 timestamp. */
  opened_at: string;
  /**
   * The branch the repository's checkout was on when the session was
   * opened. Its accepted work is merged into that branch only; '' when it
   * was on none.
   */
  branch: string;
  /**
   * What its agents have used so far, with the director's own model calls
   * for them.
   */
  usage: DirectorUsage;
  /**
   * What it may cost in US dollars before its agents are put to rest and the
   * human is asked; 0 while nothing limits it.
   */
  cost_allowance: number;
  /**
   * The sub-teams the org had under the team when the session's work was
   * last made the org's: each team, the team above it, its directory and
   * what else makes it that team.
   */
  shape?: [string, string, string, string][];
  /**
   * Of the whole session: blocked while the team it was opened at is; else
   * working while any of its teams or its reviewer is; else blocked while
   * any of them is; else what the team it was opened at is.
   */
  status: DirectorNodeStatus;
  /** Where the review of its work stands. */
  review: DirectorReviewState | null;
  /** The conversation of its IC at the team it was opened at. */
  conversation_id: string | null;
  /**
   * What it has at each of its teams, the one it was opened at first, and
   * last its Division Lead place.
   */
  teams: DirectorSessionTeam[];
}

/**
 * Where an agent is from a conversation: `above` an agent of the team that
 * reviews its team's work, `beside` the other agent of its own team, `below`
 * the IC of a team directly below.
 */
export type DirectorConversationRelation = 'above' | 'beside' | 'below';

/** An agent's conversation, as it is reached from another one. */
export interface DirectorConversationLink {
  conversation_id: string;
  relation: DirectorConversationRelation;
  /** The team of the org the agent works for. */
  node_id: DirectorNodeId;
  team: string;
  role: DirectorAgentRole;
  /** The session the agent works in. */
  session_id: string;
  session: string;
  /** The agent's status. */
  status: string;
}

/**
 * Where an agent's conversation is in the org, and the conversations one
 * level from it.
 */
export interface DirectorConversationPlace {
  conversation_id: string;
  node_id: DirectorNodeId;
  team: string;
  role: DirectorAgentRole;
  session_id: string;
  session: string;
  /** Whether the human has written in this conversation. */
  human_joined: boolean;
  /** Above first, then beside and below. */
  links: DirectorConversationLink[];
}

/** A session that was just opened. */
export interface DirectorOpenedSession {
  session: DirectorSession;
  /** The conversation of the IC the session starts with. */
  conversation_id: string;
  chosen_by: DirectorChosenBy;
  reason: string;
}

export interface DirectorEscalationPathStep {
  id: DirectorNodeId;
  title: string;
}

/** An escalation that waits on the human, with the teams it came up through. */
export interface DirectorOpenEscalation {
  /** The team of the org that raised it. */
  node_id: DirectorNodeId;
  /** The session it was raised in. */
  session_id: string;
  /** That session's title. */
  session: string;
  escalation: DirectorEscalation;
  /** The team its session was opened at first, ending at the one that raised it. */
  path: DirectorEscalationPathStep[];
}

/** Something an agent recorded for itself and the other agent of its team. */
export interface DirectorReport {
  /** ISO 8601 timestamp. */
  at: string;
  /** Which agent of the team wrote it. */
  role: DirectorAgentRole;
  text: string;
  /**
   * The title of the session the agent that wrote it works in; `''` for a
   * report a model rewrote or the IC above handed over.
   */
  session?: string;
  /** That session's id: two sessions may have one title. */
  session_id?: string;
  /** True for the one report a model wrote in place of earlier ones. */
  compacted?: boolean;
  /**
   * True for what the IC of the team above passed on of the teams that were
   * there before this one. The team's own agents did not write it.
   */
  handed_over?: boolean;
  /**
   * Names the report, so that it is added to a team's record once however
   * often that is tried. `''` for one a model rewrote.
   */
  id?: string;
}

/** One message between a Manager and an IC. */
export interface DirectorThreadMessage {
  /** ISO 8601 timestamp. */
  at: string;
  /** Which of the two wrote it. */
  sender: DirectorAgentRole;
  text: string;
}

/**
 * What one Manager and one IC have said to each other, oldest first. The IC
 * is the Manager's own, or the IC of a team directly below it. With
 * `ic_to_ic`, what the IC of the team above wrote to an IC instead: those go
 * one way, and `manager_node_id` is then the team of the IC that writes.
 */
export interface DirectorThread {
  manager_node_id: DirectorNodeId;
  ic_node_id: DirectorNodeId;
  ic_to_ic?: boolean;
  messages: DirectorThreadMessage[];
}

/** That the IC of the team above stopped a team, and why. */
export interface DirectorStopped {
  /** ISO 8601 timestamp. */
  at: string;
  reason: string;
  /** The conversations that were in a turn when the team was stopped. */
  running: string[];
}

/** A document a team reads, as the team memory has it. */
export interface DirectorTeamDocument {
  /** The name of the document, as the org declares it. */
  name: string;
  /** What the document holds for a team of this type. */
  holds?: string;
  /** The folder of the team that keeps it. */
  path: string;
  text: string;
  /**
   * True for the one the team itself keeps and writes; false for one a team
   * above keeps for it, and for one a team before it left.
   */
  own: boolean;
  /**
   * True for one a team before this one left in its folder: it says how
   * that team worked and is to be rewritten.
   */
  left?: boolean;
  /** The title of the team that left it, when known. */
  left_by?: string | null;
}

/** A spec a team before left in a directory a team now owns. */
export interface DirectorOldSpec {
  /** The directory of the spec; '' at the root. */
  path: string;
  /** The title of the team that left it, when known. */
  team: string | null;
}

export type DirectorTeamEventKind =
  | 'set_up'
  | 'accepted'
  | 'sent_back'
  | 'reconciled'
  | 'stalled'
  | 'failed'
  | 'stopped'
  | 'replaced'
  | 'pruned'
  | 'woken'
  | 'proposed'
  | 'approved'
  | 'declined'
  | 'needed'
  | 'asked'
  | 'misrouted'
  | 'session_opened'
  | 'session_closed'
  | 'not_taken'
  | 'moved';

/** Something that became of a team directly below a team. */
export interface DirectorTeamEvent {
  /** What it is added to a record by, once; '' on events kept before ids. */
  id?: string;
  /**
   * Whether it is of a session whose work is not accepted yet: it is the
   * session's own and not in the org's record.
   */
  unaccepted?: boolean;
  at: string;
  kind: DirectorTeamEventKind;
  /**
   * Empty when the event is about no single team. A `woken` event names the
   * team whose escalations reached the limit that wakes the Manager.
   */
  team_id?: string;
  team?: string;
  /** The title of the session that team works in, if it is in one. */
  session?: string;
  detail?: string;
  /**
   * The directories a proposal was for, on `proposed`, `approved` and
   * `declined`; on `needed`, the file in the other team's directory; on
   * `accepted`, the team's directory (`''` is the whole repository).
   */
  paths?: string[];
  /**
   * On `accepted`, the commit the team's work was last accepted at or
   * started from; `''` when it is not known.
   */
  base?: string;
  /** On `accepted`, the commit that was accepted. */
  head?: string;
  /** On `needed`, the team beside this one whose directory it needed. */
  other_id?: string;
  other?: string;
}

/** One event of an org's history, with the team whose record holds it. */
export interface DirectorHistoryEvent extends DirectorTeamEvent {
  /** The team directly above the event's team. */
  above_id: DirectorNodeId;
  above: string;
}

/** What became of the teams of an org, newest first. */
export interface DirectorOrgHistory {
  events: DirectorHistoryEvent[];
  /** Whether there are events before these. */
  earlier: boolean;
}

/** A team that was removed from below a team. */
export interface DirectorFormerTeam {
  id: DirectorNodeId;
  title: string;
  path: string;
  purpose: string;
  /**
   * How it ended. `not_taken`: a session set it up or changed it, and when
   * the session's work was accepted the org did not make it its own.
   */
  ended: 'replaced' | 'pruned' | 'not_taken';
  at: string;
  compacted_report?: string;
  contracts?: string;
  /** The spec a session wrote for it that the org did not take. */
  spec?: string;
  /** The title of the session it was a team of; `''` when none. */
  session?: string;
}

export interface DirectorNodeMemory {
  /**
   * The documents the team reads: the ones it keeps, in the order its type
   * lists them, then any other it is shown.
   */
  documents?: DirectorTeamDocument[];
  /** What the team's own two agents recorded, oldest first. */
  reports?: DirectorReport[];
  /** The teams removed from directly below it, oldest first. */
  former_teams?: DirectorFormerTeam[];
  /** What became of the teams directly below it, oldest first. */
  history?: DirectorTeamEvent[];
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
  /** The direct report that owns the file; null when the team kept it. */
  owner: DirectorNodeId | null;
}

/** A file the committed work of two teams changed in ways that do not merge. */
export interface DirectorCollision {
  path: string;
  /** The teams whose checkouts collide, the team itself included. */
  node_ids: DirectorNodeId[];
}

/** One file a team changed. */
export interface DirectorChangedFile {
  /** Relative to the top of the repository. */
  path: string;
  kind: 'added' | 'modified' | 'deleted';
  /** `null` when the file is not text. */
  added: number | null;
  removed: number | null;
  /** False when the file has changes that are not committed. */
  committed: boolean;
}

/** What a team changed in its directory since its base commit. */
export interface DirectorTeamChanges {
  node_id: DirectorNodeId;
  /** The commit compared with; `''` when the team has none yet. */
  base: string;
  files: DirectorChangedFile[];
  truncated: boolean;
}

/** What changed in a directory from one commit to another. */
export interface DirectorAcceptedChanges {
  base: string;
  head: string;
  files: DirectorChangedFile[];
  truncated: boolean;
}

/**
 * A file before and after: at a team's base commit and in its checkout now,
 * or at the two commits of an accept.
 */
export interface DirectorFileSides {
  path: string;
  /** `null` when the file was not there. */
  before: string | null;
  /** `null` when the file is not there. */
  after: string | null;
  /** A side is not text or is too large, and is left out. */
  unreadable: boolean;
}

/** What came of a change made by hand. */
export interface DirectorFileEdit {
  path: string;
  /** `null` when nothing changed. */
  commit: string | null;
}

/** How a team's boundary is divided among its direct reports. */
export interface DirectorNodeOwnership {
  node_id: DirectorNodeId;
  owners: DirectorBoundaryOwner[];
  /**
   * The unignored files of the team's boundary in its own checkout, by path;
   * for a team of the org, in the repository's checkout.
   */
  files: DirectorOwnedFile[];
  truncated: boolean;
  /**
   * From the tips of the checkouts; uncommitted changes are not compared.
   * Always empty for a team of the org.
   */
  collisions: DirectorCollision[];
  /**
   * False when the server's git is too old to look for collisions, and for
   * a team of the org.
   */
  collisions_checked: boolean;
}

/**
 * What `GET /api/director/org/tree-stream` sends.
 *
 * A `snapshot` holds the focused team, its ancestors, its direct reports and
 * the direct reports of every expanded team (or, for a full stream, every
 * team below the focused one), plus the open sessions and every escalation
 * that waits on the human, wherever the stream is focused. `node_upsert` and
 * `node_removed` are limited to those teams. `sessions` and `escalations`
 * carry the whole new list. The snapshot says
 * whether the org is paused and whether a pull waits for choices; `org_state`
 * follows whenever either changes.
 */
export type DirectorOrgEvent =
  | {
      type: 'snapshot';
      focus_id: DirectorNodeId;
      nodes: DirectorOrgNode[];
      escalations: DirectorOpenEscalation[];
      /** The sessions that are open. They are no teams of the tree. */
      sessions: DirectorSession[];
      paused?: DirectorPauseReason | null;
      merging?: boolean;
    }
  | { type: 'org_state'; paused: DirectorPauseReason | null; merging: boolean }
  | { type: 'node_upsert'; node: DirectorOrgNode }
  | { type: 'node_removed'; id: DirectorNodeId }
  | { type: 'escalations'; escalations: DirectorOpenEscalation[] }
  | { type: 'sessions'; sessions: DirectorSession[] };

/**
 * Limits on the teams an org may still add. They are checked when a team
 * splits; teams that exist are never removed for exceeding one.
 */
export interface DirectorSettings {
  /**
   * The most teams a team may be together with every team below it. Every
   * team has an IC and a Manager; the Manager is inactive while the team has
   * no sub-teams.
   */
  max_teams: number;
  /** The most levels of teams; the root team is level 1. */
  max_depth: number;
  /**
   * Names of the agent profiles a team may give a sub-team in place of its
   * own agent. Empty: every team runs on the agent of the team above.
   */
  allowed_profiles: string[];
  /**
   * For every org that sets none of its own: the agent type of a new team
   * whose making names none.
   */
  default_agent_type: string;
  /**
   * For every org that sets none of its own: the agent types a new team may
   * be. Empty: every type the director finds.
   */
  allowed_agent_types: string[];
  /**
   * Whether an org that was stopped in the middle of its work takes it up
   * again as soon as the server is back, without being asked.
   */
  resume_on_start: boolean;
  /**
   * How many characters of reports one agent may have before a model
   * rewrites them into one shorter report.
   */
  report_compaction_chars: number;
  /**
   * How many minutes a team may show no progress, with nothing submitted or
   * with its work sent back, before the IC of the team above is told. 0
   * turns the check off.
   */
  soft_check_in_minutes: number;
  /**
   * After how many minutes the IC of the team above is told how a team that
   * has not submitted stands, whether or not it shows progress. Counted from
   * when that IC last wrote to the team, sent its work back, or was told of
   * it, and never shorter than soft_check_in_minutes. 0 turns the check-in off.
   */
  hard_check_in_minutes: number;
  /**
   * After how many reviews its IC accepted a Manager is woken to look at its
   * teams, without a message from its IC. 0 turns the check off.
   */
  manager_check_accepted_reviews: number;
  /**
   * After how many escalations the IC of one team directly below raised on
   * the work it has in hand, and after how many times that work was sent
   * back, a Manager is woken to look at its teams. 0 turns the check off.
   */
  manager_check_ic_setbacks: number;
  /**
   * Whether a message sent with `auto` may go straight to the team the
   * director chooses. Off, it goes to the team its folder gives.
   */
  route_requests: boolean;
  /**
   * What the agents of one session may cost in US dollars before they are
   * put to rest and the human is asked whether it goes on for as much again.
   * 0: no limit.
   */
  session_cost_limit: number;
  /** How many sessions may be open at once. 0: no limit. */
  max_open_sessions: number;
}

/** The settings, beside what the org has now. */
export interface DirectorSettingsState extends DirectorSettings {
  /** How many teams the org has. */
  teams: number;
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

/** What a pull of the org did to the team memory. */
export interface DirectorMemorySync {
  remote: string;
  /** The local team memory's commit afterwards; `null` when it is empty. */
  head: string | null;
  /** The local team memory took in remote changes. */
  changed: boolean;
  /**
   * Specs and contracts both sides had changed. The local text was kept;
   * the remote text is in the team memory's history.
   */
  kept_local: string[];
}

/**
 * What one org sets for itself, kept with the org in its repository. What it
 * leaves out is as the director's settings say for every org.
 */
export interface DirectorOrgSettings {
  /** The agent type of a new team whose making names none. `""` unsets it. */
  default_agent_type?: string | null;
  /**
   * The agent types a new team may be; empty allows every one. A team that
   * exists keeps its type.
   */
  allowed_agent_types?: string[] | null;
}

/** An agent type, as someone choosing one sees it. */
/** One member of a team of a type, as a page that shows teams reads it. */
export interface DirectorMemberInfo {
  /** What the type names it: an agent's `role`. */
  name: DirectorAgentRole;
  /** What it is called in words for a person. */
  title: string;
  /** Whether it works in the team's folder. */
  writes: boolean;
  /** Whether it reviews the work of the teams below. */
  reviews: boolean;
  /** Whether it settles what the team's sub-teams are. */
  oversees: boolean;
}

export interface DirectorAgentTypeInfo {
  name: string;
  description: string;
  /** Whether a new team of the org may be of it. */
  allowed: boolean;
  /**
   * The agent profile a new team of this type runs on when its making names
   * none.
   */
  sas_profile: string | null;
  /** The members of a team of this type, in the type's order. */
  members: DirectorMemberInfo[];
}

/** What holds for new teams of the org, and what it has to choose from. */
export interface DirectorOrgSettingsState {
  default_agent_type: string;
  allowed_agent_types: string[];
  /** What the org sets for itself. */
  own: DirectorOrgSettings;
  /** Every agent type the director finds now. */
  agent_types: DirectorAgentTypeInfo[];
}
