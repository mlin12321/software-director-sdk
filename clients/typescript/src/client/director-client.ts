import { HttpClient, HttpError } from './http-client';
import type {
  DirectorBoundaryMemory,
  DirectorNodeBlackboard,
  DirectorNodeId,
  DirectorOrgEvent,
  DirectorOrgNodeDetail,
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
  /** Abort to stop following. */
  signal?: AbortSignal;
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
