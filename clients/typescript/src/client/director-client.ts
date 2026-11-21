import { HttpClient, HttpError } from './http-client';
import type { CreateConversationPayload } from './conversation-client';
import type {
  DirectorAcceptedChanges,
  DirectorFileEdit,
  DirectorFileSides,
  DirectorNodeMemory,
  DirectorOrgHistory,
  DirectorNodeOwnership,
  DirectorNodeId,
  DirectorOrgEvent,
  DirectorOrgNodeDetail,
  DirectorOrgSummary,
  DirectorOpenedSession,
  DirectorRoutedMessage,
  DirectorSession,
  DirectorSessionTeam,
  DirectorConversationPlace,
  DirectorThread,
  DirectorOrgMerge,
  DirectorOrgSync,
  DirectorRecovery,
  DirectorResolveMerge,
  DirectorSettings,
  DirectorOrgSettings,
  DirectorOrgSettingsState,
  DirectorSettingsState,
  DirectorTeamChanges,
  DirectorTeamMatch,
} from '../models/director';

export interface DirectorClientOptions {
  host: string;
  apiKey?: string;
  timeout?: number;
}

export interface StreamOrgTreeOptions {
  /** Node to show as the root of the chart. Defaults to the tree's root. */
  node?: DirectorNodeId;
  /** Nodes whose direct reports to follow as well. */
  expanded?: readonly DirectorNodeId[];
  /** Follow every node below `node`, for drawing the whole chart. */
  full?: boolean;
  /** Abort to stop following. */
  signal?: AbortSignal;
}

/** Start an org: its top team, and the session that works on `task`. */
export interface CreateOrgRequest {
  /**
   * The root team's name. Left out, the team is first named after the
   * task's first line and then by a model that summarizes it.
   */
  title?: string;
  task: string;
  /**
   * How to start the IC of the org's first session, as for a normal
   * conversation. Every other agent of the org is derived from one that
   * runs already. The org lives in the git repository of its workspace. Its
   * initial message is ignored: the IC is briefed with `task`.
   */
  conversation: CreateConversationPayload;
  /**
   * The org is the git repository of the conversation's workspace itself,
   * and not one above it. A workspace with no `.git` of its own is not a
   * repository.
   */
  at_workspace?: boolean;
}

/** Something the human says to the org, from somewhere in the repository. */
export interface RouteMessageRequest {
  text: string;
  /**
   * The directory the message is about: relative to the org's `repo_path`,
   * or absolute inside it. Defaults to the org summary's `directory`.
   */
  path?: string;
  /**
   * Let the director choose a team below the one `path` gives, by the files
   * the text names and by what a model makes of it. It moves down only when
   * it is sure, and not at all while the org's `route_requests` is off.
   */
  auto?: boolean;
  /**
   * The session the human says it in, when the message continues that
   * session's conversation and begins no new one. It goes to the IC of one
   * of the session's own teams, those it set up itself included: the team it
   * was opened at unless the director is sure of one below. `path`, `auto`
   * and `conversation` are ignored.
   */
  session_id?: string;
  /**
   * How to start the IC of the session the message opens; without it that
   * IC is derived from one that runs already. Needed when there is no org
   * yet: if the git repository of its workspace already has a saved org,
   * that org is taken up and the message routed in it; otherwise the message
   * starts an org there and becomes its task.
   */
  conversation?: CreateConversationPayload;
  /**
   * The agent type of the top team of the org this message starts. Left
   * out, the default one. Ignored when the org exists.
   */
  agent_type?: string;
  /**
   * Join or start the org only in the conversation workspace's own checkout,
   * not in a repository above it. A workspace with no `.git` of its own is
   * not a repository.
   */
  at_workspace?: boolean;
}

/** A request the human wants worked on, in a session of its own. */
export interface OpenSessionRequest {
  /** What the session is to do. */
  text: string;
  /** What to call the session; taken from the text when empty. */
  title?: string;
  /**
   * The team the session is opened at. Without it the director chooses the
   * team as it does for a message sent with `auto`.
   */
  node_id?: DirectorNodeId;
}

const DIRECTOR_PATH = '/api/director';

function nodePath(nodeId: DirectorNodeId, action?: string): string {
  const path = `${DIRECTOR_PATH}/org/nodes/${encodeURIComponent(nodeId)}`;
  return action ? `${path}/${action}` : path;
}

/**
 * Splits a server-sent event stream into the `data` of each event. Comment
 * frames (the server's keep-alives) and frames without data are dropped.
 */
async function* readServerSentData(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, { stream: true }).replaceAll('\r\n', '\n');
      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = frame
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice('data:'.length).replace(/^ /, ''))
          .join('\n');
        if (data) yield data;
        boundary = buffer.indexOf('\n\n');
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
  }
}

/**
 * The org tree of agent teams and the team memory that applies to each node.
 *
 * Client-ahead of the pinned Agent Server release: these calls need a server
 * that provides the director API and answer 404 on one that does not.
 */
export class DirectorClient {
  public readonly host: string;
  public readonly apiKey?: string;
  private readonly client: HttpClient;

  constructor(options: DirectorClientOptions) {
    this.host = options.host.replace(/\/$/, '');
    this.apiKey = options.apiKey;
    this.client = new HttpClient({
      baseUrl: this.host,
      apiKey: this.apiKey,
      timeout: options.timeout || 60000,
    });
  }

  /**
   * Whether the director has an org. Without one `root_id` is null and
   * `streamOrgTree` is refused with 404 until `createOrg` starts one.
   */
  async getOrg(): Promise<DirectorOrgSummary> {
    const response = await this.client.get<DirectorOrgSummary>(`${DIRECTOR_PATH}/org`);
    return response.data;
  }

  /**
   * Start an org: its top team, and the session that works on the task.
   * Nobody works at a team of the org itself; the given conversation is the
   * IC of the first session.
   *
   * @returns The session that was opened. Follow the org with
   *   `streamOrgTree`.
   * @throws HttpError 409 when the director already has an org; 422 when the
   *   conversation cannot lead one (workspace outside the director's
   *   directory, Docker runtime, ACP agent); 503 when the directory is not a
   *   git repository.
   */
  async createOrg(request: CreateOrgRequest): Promise<DirectorOpenedSession> {
    const response = await this.client.post<DirectorOpenedSession>(`${DIRECTOR_PATH}/org`, request);
    return response.data;
  }

  /**
   * Say something to the org from a directory of the repository.
   *
   * The message opens a session at the nearest team: the search starts at
   * `path` and moves up one directory at a time until it reaches a directory
   * a team owns, skipping a team that may only read `path`. With
   * `session_id` it is said in that session instead. Without an org the
   * message starts one and becomes its task.
   *
   * @throws HttpError 409 when no team can take the message; 422 when the
   *   path is outside the repository, or there is no org and `conversation`
   *   is missing or cannot lead one.
   */
  async routeMessage(request: RouteMessageRequest): Promise<DirectorRoutedMessage> {
    const response = await this.client.post<DirectorRoutedMessage>(
      `${DIRECTOR_PATH}/org/messages`,
      request
    );
    return response.data;
  }

  /** The sessions that are open. */
  async listSessions(): Promise<DirectorSession[]> {
    const response = await this.client.get<DirectorSession[]>(`${DIRECTOR_PATH}/org/sessions`);
    return response.data;
  }

  /**
   * Open a session at a team and start it with a request: the team gets an
   * IC for it, in a git worktree of its own, whose work an IC at the team
   * above, started only to review, reviews. It adds no team to the org.
   *
   * @throws HttpError 404 when there is no such team; 409 when there is no
   *   org, it is paused, or no conversation of the org runs on the server
   *   to start the session's IC from.
   */
  async openSession(request: OpenSessionRequest): Promise<DirectorOpenedSession> {
    const response = await this.client.post<DirectorOpenedSession>(
      `${DIRECTOR_PATH}/org/sessions`,
      request
    );
    return response.data;
  }

  /**
   * Close a session: its agents and its checkout are removed.
   * Work of it that was accepted stays where it was merged.
   *
   * @throws HttpError 404 when no such session is open.
   */
  async closeSession(sessionId: string): Promise<void> {
    await this.client.delete(`${DIRECTOR_PATH}/org/sessions/${encodeURIComponent(sessionId)}`);
  }

  /** One open session with what it has at each of its teams. */
  async getSession(sessionId: string): Promise<DirectorSession> {
    const response = await this.client.get<DirectorSession>(
      `${DIRECTOR_PATH}/org/sessions/${encodeURIComponent(sessionId)}`
    );
    return response.data;
  }

  /**
   * Answer an escalation a team raised in a session and that waits on the
   * human. The answer is sent to the session's IC at that team.
   *
   * @throws HttpError 404 when there is no such session, team or escalation;
   *   409 when the escalation is still with a team.
   */
  async answerSessionEscalation(
    sessionId: string,
    nodeId: DirectorNodeId,
    escalationId: string,
    answer: string
  ): Promise<DirectorSessionTeam> {
    const response = await this.client.post<DirectorSessionTeam>(
      `${DIRECTOR_PATH}/org/sessions/${encodeURIComponent(sessionId)}/teams/${encodeURIComponent(
        nodeId
      )}/escalations/${encodeURIComponent(escalationId)}/answer`,
      { answer }
    );
    return response.data;
  }

  /**
   * Where an agent's conversation is in the org, with links to the
   * conversations one level from it.
   *
   * @throws HttpError 404 when no agent of the org has the conversation.
   */
  async getConversationPlace(conversationId: string): Promise<DirectorConversationPlace> {
    const response = await this.client.get<DirectorConversationPlace>(
      `${DIRECTOR_PATH}/org/conversations/${encodeURIComponent(conversationId)}`
    );
    return response.data;
  }

  /**
   * Note that the human has written in an agent's conversation, so that it
   * is tagged `humanjoined=true` and listed with the human's own. A message
   * sent through the director does this itself.
   */
  async joinConversation(conversationId: string): Promise<DirectorConversationPlace> {
    const response = await this.client.post<DirectorConversationPlace>(
      `${DIRECTOR_PATH}/org/conversations/${encodeURIComponent(conversationId)}/joined`
    );
    return response.data;
  }

  /** A team of the org tree with the teams above it and its sub-teams. */
  async getNode(nodeId: DirectorNodeId): Promise<DirectorOrgNodeDetail> {
    const response = await this.client.get<DirectorOrgNodeDetail>(nodePath(nodeId));
    return response.data;
  }

  /**
   * What became of the teams of the whole org, newest first. Pass the time
   * of the last event received as `before` to read the ones before it.
   */
  async getOrgHistory(limit = 100, before?: string): Promise<DirectorOrgHistory> {
    const response = await this.client.get<DirectorOrgHistory>(`${DIRECTOR_PATH}/org/history`, {
      params: before ? { limit, before } : { limit },
    });
    return response.data;
  }

  /** The spec and contracts a node works to, and what its agents reported. */
  async getNodeMemory(nodeId: DirectorNodeId): Promise<DirectorNodeMemory> {
    const response = await this.client.get<DirectorNodeMemory>(nodePath(nodeId, 'memory'));
    return response.data;
  }

  /**
   * The threads the agents of a place of a session take part in: its IC and
   * its Manager with each other, its Manager and its IC with the ICs directly
   * below, and the Manager and the IC above with its IC. Threads nobody wrote
   * in are left out. Empty for a team of the org tree.
   */
  async getNodeMessages(nodeId: DirectorNodeId): Promise<DirectorThread[]> {
    const response = await this.client.get<DirectorThread[]>(nodePath(nodeId, 'messages'));
    return response.data;
  }

  /**
   * Which direct report owns each file of a node's boundary, and the files
   * where the committed work of two of these teams does not merge. For a
   * team of the org tree no collisions are looked for.
   */
  async getNodeOwnership(nodeId: DirectorNodeId): Promise<DirectorNodeOwnership> {
    const response = await this.client.get<DirectorNodeOwnership>(nodePath(nodeId, 'ownership'));
    return response.data;
  }

  /**
   * The files a team changed in its directory since its work started or
   * was last accepted, uncommitted and new files included.
   *
   * @param nodeId A place of a session.
   * @throws HttpError 404 when the place or its checkout is not on the server.
   */
  async getNodeChanges(nodeId: DirectorNodeId): Promise<DirectorTeamChanges> {
    const response = await this.client.get<DirectorTeamChanges>(nodePath(nodeId, 'changes'));
    return response.data;
  }

  /**
   * One file as it was at the team's base commit and as it is now.
   *
   * @param path Relative to the top of the repository.
   */
  async getNodeChangedFile(nodeId: DirectorNodeId, path: string): Promise<DirectorFileSides> {
    const response = await this.client.get<DirectorFileSides>(nodePath(nodeId, 'changes/file'), {
      params: { path },
    });
    return response.data;
  }

  /**
   * The files an accept took: what differs under `path` between the `base`
   * and the `head` of an `accepted` event in the history. It can be read
   * after the team's session is closed.
   *
   * @param path The team's directory; `''` is the whole repository.
   * @throws HttpError 404 when the repository no longer has a commit.
   */
  async getAcceptedChanges(
    base: string,
    head: string,
    path: string
  ): Promise<DirectorAcceptedChanges> {
    const response = await this.client.get<DirectorAcceptedChanges>(
      `${DIRECTOR_PATH}/org/accepted/changes`,
      { params: { base, head, path } }
    );
    return response.data;
  }

  /**
   * One file as it was before an accept and as it was accepted.
   *
   * @param path Relative to the top of the repository.
   */
  async getAcceptedFile(base: string, head: string, path: string): Promise<DirectorFileSides> {
    const response = await this.client.get<DirectorFileSides>(
      `${DIRECTOR_PATH}/org/accepted/changes/file`,
      { params: { base, head, path } }
    );
    return response.data;
  }

  /**
   * Write one file of a team by hand. It is committed alone and the team's
   * IC is told.
   *
   * @throws HttpError 409 while an agent of the team is in the middle of a
   *   turn, 403 for a file the team's agents may not write either.
   */
  async editNodeFile(
    nodeId: DirectorNodeId,
    path: string,
    content: string
  ): Promise<DirectorFileEdit> {
    const response = await this.client.put<DirectorFileEdit>(
      nodePath(nodeId, 'files'),
      { content },
      { params: { path } }
    );
    return response.data;
  }

  /**
   * Take in the org a remote of the repository has, as a merge: the tree
   * and the team memory. Sessions stay on their machine. When both sides
   * changed the same thing the result is `conflicts` and nothing is written
   * until `resolveMerge` has a choice for each, or `abortMerge` is called.
   *
   * @throws HttpError 404 when the repository has no such remote, 409 when a
   *   merge already waits, 502 when the remote cannot be reached.
   */
  async pullOrg(remote = 'origin'): Promise<DirectorOrgSync> {
    const response = await this.client.post<DirectorOrgSync>(`${DIRECTOR_PATH}/org/pull`, {
      remote,
    });
    return response.data;
  }

  /**
   * Give a remote of the repository the org and the team memory.
   *
   * @throws HttpError 409 when there is no org, a merge waits, or the remote
   *   has changes that were not pulled.
   */
  async pushOrg(remote = 'origin'): Promise<DirectorOrgSync> {
    const response = await this.client.post<DirectorOrgSync>(`${DIRECTOR_PATH}/org/push`, {
      remote,
    });
    return response.data;
  }

  /**
   * Start afresh what a session has at a team when its agents have no
   * conversation on this server, on the work its branch has.
   *
   * @param nodeId The team of the org, or the id of the session's place.
   * @param conversation How to start the team's IC, where no conversation of
   *   the org runs here to start it from.
   * @throws HttpError 404 when the session has no such team, 409 when it is
   *   done, its IC has a conversation here, or no conversation of the org
   *   runs here to start it from and none is given.
   */
  async startAgain(
    sessionId: string,
    nodeId: DirectorNodeId,
    conversation?: CreateConversationPayload
  ): Promise<DirectorSessionTeam> {
    const response = await this.client.post<DirectorSessionTeam>(
      `${DIRECTOR_PATH}/org/sessions/${encodeURIComponent(sessionId)}/teams/${encodeURIComponent(
        nodeId
      )}/start-again`,
      conversation ? { conversation } : {}
    );
    return response.data;
  }

  /** The merge that waits for choices; `null` when none does. */
  async getMerge(): Promise<DirectorOrgMerge | null> {
    const response = await this.client.get<DirectorOrgMerge | null>(`${DIRECTOR_PATH}/org/merge`);
    return response.data;
  }

  /**
   * Choose a side for conflicts of the merge that waits. It is finished once
   * every conflict has a choice.
   *
   * @throws HttpError 409 when no merge waits.
   */
  async resolveMerge(request: DirectorResolveMerge): Promise<DirectorOrgSync> {
    const response = await this.client.post<DirectorOrgSync>(
      `${DIRECTOR_PATH}/org/merge/resolve`,
      request
    );
    return response.data;
  }

  /** Abandon the merge that waits; the org stays as it was. */
  async abortMerge(): Promise<DirectorOrgSync> {
    const response = await this.client.post<DirectorOrgSync>(`${DIRECTOR_PATH}/org/merge/abort`);
    return response.data;
  }

  /** The limits on new teams, beside how large the org is now. */
  async getSettings(): Promise<DirectorSettingsState> {
    const response = await this.client.get<DirectorSettingsState>(`${DIRECTOR_PATH}/settings`);
    return response.data;
  }

  /**
   * Change the limits on new teams. A limit is checked only when a team
   * splits, so lowering one below what the org has removes nothing.
   */
  async updateSettings(settings: Partial<DirectorSettings>): Promise<DirectorSettingsState> {
    const response = await this.client.patch<DirectorSettingsState>(
      `${DIRECTOR_PATH}/settings`,
      settings
    );
    return response.data;
  }

  /**
   * The agent type of a new team and the types one may be, for this org,
   * beside every agent type the director finds now. Without an org these
   * are the ones every new org starts with.
   */
  async getOrgSettings(): Promise<DirectorOrgSettingsState> {
    const response = await this.client.get<DirectorOrgSettingsState>(
      `${DIRECTOR_PATH}/org/settings`
    );
    return response.data;
  }

  /**
   * Set, for this org alone, the agent type of a new team and the types one
   * may be. It holds for teams made from now on.
   */
  async updateOrgSettings(settings: DirectorOrgSettings): Promise<DirectorOrgSettingsState> {
    const response = await this.client.patch<DirectorOrgSettingsState>(
      `${DIRECTOR_PATH}/org/settings`,
      settings
    );
    return response.data;
  }

  /**
   * Hold the org: its agents stop after the step they are in, and the
   * director's steps and the messages to agents wait. Returns what
   * `resumeOrg` would do.
   */
  async pauseOrg(): Promise<DirectorRecovery> {
    const response = await this.client.post<DirectorRecovery>(`${DIRECTOR_PATH}/org/pause`);
    return response.data;
  }

  /**
   * Take a paused org up again. Returns only what was put aside because it
   * no longer fits the org.
   */
  async resumeOrg(): Promise<DirectorRecovery> {
    const response = await this.client.post<DirectorRecovery>(`${DIRECTOR_PATH}/org/resume`);
    return response.data;
  }

  /** What resuming a paused org does, team by team. */
  async getRecovery(): Promise<DirectorRecovery> {
    const response = await this.client.get<DirectorRecovery>(`${DIRECTOR_PATH}/org/recovery`);
    return response.data;
  }

  /**
   * Teams whose title or directory has every word of `query`, parents
   * before their sub-teams.
   */
  async searchTeams(query: string, limit = 20): Promise<DirectorTeamMatch[]> {
    const response = await this.client.get<DirectorTeamMatch[]>(`${DIRECTOR_PATH}/org/search`, {
      params: { q: query, limit },
    });
    return response.data;
  }

  /**
   * Follow the part of the org tree a chart has open.
   *
   * The first event is a snapshot; changes follow until `signal` aborts or
   * the server closes the stream, at which point iteration ends. To expand
   * or collapse a node, stop and call again with the new `expanded` set; the
   * new snapshot replaces everything received before. Nothing is retried.
   *
   * @throws HttpError The server refused the stream, e.g. 404 for a node
   *   that does not exist or an org tree that is empty.
   */
  async *streamOrgTree(options: StreamOrgTreeOptions = {}): AsyncGenerator<DirectorOrgEvent> {
    const url = new URL(`${this.host}${DIRECTOR_PATH}/org/tree-stream`);
    if (options.node !== undefined) url.searchParams.set('node', options.node);
    for (const nodeId of options.expanded ?? []) url.searchParams.append('expanded', nodeId);
    if (options.full) url.searchParams.set('full', 'true');

    const headers: Record<string, string> = { Accept: 'text/event-stream' };
    if (this.apiKey) headers['X-Session-API-Key'] = this.apiKey;

    const response = await fetch(url.toString(), { headers, signal: options.signal });
    if (!response.ok || !response.body) {
      const body: unknown = await response.json().catch(() => undefined);
      throw new HttpError(response.status, response.statusText, body);
    }
    for await (const data of readServerSentData(response.body)) {
      yield JSON.parse(data) as DirectorOrgEvent;
    }
  }

  close(): void {
    this.client.close();
  }
}
