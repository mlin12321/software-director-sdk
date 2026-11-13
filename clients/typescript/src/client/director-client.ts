import { HttpClient, HttpError } from './http-client';
import type { CreateConversationPayload } from './conversation-client';
import type {
  DirectorBoundaryMemory,
  DirectorNodeBlackboard,
  DirectorNodeOwnership,
  DirectorNodeId,
  DirectorOrgEvent,
  DirectorOrgNode,
  DirectorOrgNodeDetail,
  DirectorOrgSummary,
  DirectorRoutedMessage,
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

/** Start an org: a root team that works on `task`. */
export interface CreateOrgRequest {
  /**
   * The root team's name. Left out, the team is first named after the
   * task's first line and then by a model that summarizes it.
   */
  title?: string;
  task: string;
  /**
   * How to start the root team's IC, as for a normal conversation. Every
   * other agent of the org is derived from it. The org lives in the git
   * repository of its workspace. Its initial message is ignored: the IC is
   * briefed with `task`.
   */
  conversation: CreateConversationPayload;
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
   * How to start the root team's IC when there is no org yet. If the git
   * repository of its workspace already has a saved org, that org is taken
   * up and the message routed in it; otherwise the message starts an org
   * there and becomes its task. Ignored when an org exists.
   */
  conversation?: CreateConversationPayload;
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
 * The org tree of agent teams and the blackboard that applies to each node.
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
   * Start an org and brief the IC of its root team with the task.
   *
   * @returns The root node. Follow the org with `streamOrgTree`.
   * @throws HttpError 409 when the director already has an org; 422 when the
   *   conversation cannot lead one (workspace outside the director's
   *   directory, Docker runtime, ACP agent); 503 when the directory is not a
   *   git repository.
   */
  async createOrg(request: CreateOrgRequest): Promise<DirectorOrgNode> {
    const response = await this.client.post<DirectorOrgNode>(`${DIRECTOR_PATH}/org`, request);
    return response.data;
  }

  /**
   * Say something to the org from a directory of the repository.
   *
   * The message goes to the IC of the nearest team: the search starts at
   * `path` and moves up one directory at a time until it reaches a directory
   * a team owns, skipping a team whose IC is gone or that may only read
   * `path`. Without an org the message starts one and becomes its task.
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

  /**
   * Answer an escalation that waits on the human, which settles it. The
   * answer is sent to the IC of the team that raised it.
   *
   * @param nodeId The node that raised the escalation.
   * @returns That node after the change.
   * @throws HttpError 409 when the escalation is still held by a team.
   */
  async answerEscalation(
    nodeId: DirectorNodeId,
    escalationId: string,
    answer: string
  ): Promise<DirectorOrgNode> {
    const response = await this.client.post<DirectorOrgNode>(
      nodePath(nodeId, `escalations/${encodeURIComponent(escalationId)}/answer`),
      { answer }
    );
    return response.data;
  }

  /** A node with its ancestors and its direct reports. */
  async getNode(nodeId: DirectorNodeId): Promise<DirectorOrgNodeDetail> {
    const response = await this.client.get<DirectorOrgNodeDetail>(nodePath(nodeId));
    return response.data;
  }

  /** The specs, memory and autopsies that apply to a node's boundary. */
  async getNodeBlackboard(nodeId: DirectorNodeId): Promise<DirectorNodeBlackboard> {
    const response = await this.client.get<DirectorNodeBlackboard>(nodePath(nodeId, 'blackboard'));
    return response.data;
  }

  /**
   * Which direct report owns each file of a node's boundary, and the files
   * where the committed work of two of these teams does not merge.
   */
  async getNodeOwnership(nodeId: DirectorNodeId): Promise<DirectorNodeOwnership> {
    const response = await this.client.get<DirectorNodeOwnership>(nodePath(nodeId, 'ownership'));
    return response.data;
  }

  /**
   * Add a rule to the permanent memory of a node's boundary.
   *
   * @returns The boundary's memory after the change.
   */
  async teach(nodeId: DirectorNodeId, rule: string): Promise<DirectorBoundaryMemory> {
    const response = await this.client.post<DirectorBoundaryMemory>(nodePath(nodeId, 'teach'), {
      rule,
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
