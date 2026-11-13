import { DirectorClient } from '../client/director-client';
import { isHttpError } from '../client/http-client';
import type { DirectorOrgEvent } from '../models/director';

const originalFetch = global.fetch;

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function streamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
      controller.close();
    },
  });
  return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
}

function mockFetch(response: () => Response) {
  const mock = vi.fn().mockImplementation(() => Promise.resolve(response()));
  global.fetch = mock as typeof fetch;
  return mock;
}

async function collect(stream: AsyncGenerator<DirectorOrgEvent>): Promise<DirectorOrgEvent[]> {
  const events: DirectorOrgEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

describe('DirectorClient', () => {
  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('reads whether there is an org and starts one', async () => {
    const summary = {
      root_id: null,
      node_count: 0,
      repo_path: '/repo',
      directory: '',
      is_git_repository: true,
      fixture: false,
      boundary_enforcement: 'sandbox',
    };
    const fetchMock = mockFetch(() => jsonResponse(summary));
    const client = new DirectorClient({ host: 'http://example.com' });
    const conversation = { workspace: { working_dir: '/repo' }, agent_profile_id: 'default' };

    const before = await client.getOrg();
    await client.createOrg({ title: 'Checkout', task: 'Build the checkout flow', conversation });

    const calls = fetchMock.mock.calls;
    expect(before).toEqual(summary);
    expect(calls.map(([url, init]) => `${init.method} ${url}`)).toEqual([
      'GET http://example.com/api/director/org',
      'POST http://example.com/api/director/org',
    ]);
    expect(JSON.parse(calls[1][1].body)).toEqual({
      title: 'Checkout',
      task: 'Build the checkout flow',
      conversation,
    });
  });

  it('sends a message to the org from a directory', async () => {
    const routed = { node_id: 'payments.api', conversation_id: 'c-1', started_org: false };
    const fetchMock = mockFetch(() => jsonResponse(routed));
    const client = new DirectorClient({ host: 'http://example.com' });

    const result = await client.routeMessage({ text: 'Add refunds', path: 'services/api' });

    const [url, init] = fetchMock.mock.calls[0];
    expect(`${init.method} ${url}`).toBe('POST http://example.com/api/director/org/messages');
    expect(JSON.parse(init.body)).toEqual({ text: 'Add refunds', path: 'services/api' });
    expect(result).toEqual(routed);
  });

  it('answers an escalation of the node that raised it', async () => {
    const fetchMock = mockFetch(() => jsonResponse({ id: 'payments.api', escalations: [] }));
    const client = new DirectorClient({ host: 'http://example.com' });

    const node = await client.answerEscalation('payments.api', 'jwt/1', 'Opaque tokens.');

    const [url, init] = fetchMock.mock.calls[0];
    expect(`${init.method} ${url}`).toBe(
      'POST http://example.com/api/director/org/nodes/payments.api/escalations/jwt%2F1/answer'
    );
    expect(JSON.parse(init.body)).toEqual({ answer: 'Opaque tokens.' });
    expect(node.escalations).toEqual([]);
  });

  it('reports a refused answer as an HTTP error', async () => {
    mockFetch(() => jsonResponse({ detail: 'still with team root' }, 409));
    const client = new DirectorClient({ host: 'http://example.com' });

    const refusal = await client.answerEscalation('api', 'jwt', 'x').catch((error) => error);

    expect(isHttpError(refusal) && refusal.status).toBe(409);
  });

  it('calls the node routes with the node id escaped and the session key', async () => {
    const fetchMock = mockFetch(() => jsonResponse({ rules: ['Never mock the clock'] }));
    const client = new DirectorClient({ host: 'http://example.com/', apiKey: 'secret' });

    await client.getNode('payments/api');
    await client.getNodeBlackboard('payments.api');
    await client.getNodeOwnership('payments.api');
    const memory = await client.teach('payments.api', 'Never mock the clock');

    const calls = fetchMock.mock.calls;
    expect(calls.map(([url, init]) => `${init.method} ${url}`)).toEqual([
      'GET http://example.com/api/director/org/nodes/payments%2Fapi',
      'GET http://example.com/api/director/org/nodes/payments.api/blackboard',
      'GET http://example.com/api/director/org/nodes/payments.api/ownership',
      'POST http://example.com/api/director/org/nodes/payments.api/teach',
    ]);
    expect(JSON.parse(calls[3][1].body)).toEqual({ rule: 'Never mock the clock' });
    expect(calls[0][1].headers['X-Session-API-Key']).toBe('secret');
    expect(memory).toEqual({ rules: ['Never mock the clock'] });
  });

  it('pulls and pushes the blackboard, from origin unless told otherwise', async () => {
    const sync = { remote: 'origin', head: 'abc', changed: true, kept_local: [], pushed: false };
    const fetchMock = mockFetch(() => jsonResponse(sync));
    const client = new DirectorClient({ host: 'http://example.com' });

    const pulled = await client.pullBlackboard();
    await client.pushBlackboard('upstream');

    const calls = fetchMock.mock.calls;
    expect(calls.map(([url, init]) => `${init.method} ${url}`)).toEqual([
      'POST http://example.com/api/director/org/blackboard/pull',
      'POST http://example.com/api/director/org/blackboard/push',
    ]);
    expect(calls.map(([, init]) => JSON.parse(init.body))).toEqual([
      { remote: 'origin' },
      { remote: 'upstream' },
    ]);
    expect(pulled).toEqual(sync);
  });

  it('streams events, across chunk boundaries and past keep-alives', async () => {
    const snapshot = { type: 'snapshot', focus_id: 'root', nodes: [], escalations: [] };
    const removed = { type: 'node_removed', id: 'a' };
    const wire =
      `event: snapshot\ndata: ${JSON.stringify(snapshot)}\n\n` +
      ': keep-alive\n\n' +
      `event: node_removed\r\ndata: ${JSON.stringify(removed)}\r\n\r\n`;
    const middle = Math.floor(wire.length / 2);
    mockFetch(() =>
      streamResponse([wire.slice(0, 20), wire.slice(20, middle), wire.slice(middle)])
    );
    const client = new DirectorClient({ host: 'http://example.com' });

    const events = await collect(client.streamOrgTree());

    expect(events).toEqual([snapshot, removed]);
  });

  it('asks for the focused node and every expanded node', async () => {
    const fetchMock = mockFetch(() => streamResponse([]));
    const client = new DirectorClient({ host: 'http://example.com', apiKey: 'secret' });
    const controller = new AbortController();

    await collect(
      client.streamOrgTree({
        node: 'payments',
        expanded: ['payments.api', 'a b'],
        signal: controller.signal,
      })
    );
    await collect(new DirectorClient({ host: 'http://example.com' }).streamOrgTree({ full: true }));

    const [[url, init], [rootUrl, rootInit]] = fetchMock.mock.calls;
    expect(url).toBe(
      'http://example.com/api/director/org/tree-stream?node=payments&expanded=payments.api&expanded=a+b'
    );
    expect(init.headers['X-Session-API-Key']).toBe('secret');
    expect(init.signal).toBe(controller.signal);
    expect(rootUrl).toBe('http://example.com/api/director/org/tree-stream?full=true');
    expect(rootInit.headers).not.toHaveProperty('X-Session-API-Key');
  });

  it('reports a refused stream as an HttpError with the server detail', async () => {
    mockFetch(() => jsonResponse({ detail: 'Unknown node: missing' }, 404));
    const client = new DirectorClient({ host: 'http://example.com' });

    const error = await collect(client.streamOrgTree({ node: 'missing' })).catch(
      (caught: unknown) => caught
    );

    expect(isHttpError(error)).toBe(true);
    expect(error).toMatchObject({ status: 404, detail: 'Unknown node: missing' });
  });
});
