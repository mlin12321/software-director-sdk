"""Director router for OpenHands Agent Server.

Serves the org tree of agent teams, a live stream of its changes, and the
blackboard that applies to each node. State lives in ``director/``.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import StreamingResponse

# The org's agents get their tools from this server, so it registers them.
import openhands.agent_server.director.tools  # noqa: F401
from openhands.agent_server.director import blackboard_sync
from openhands.agent_server.director.models import (
    AnswerEscalationRequest,
    BlackboardSync,
    BlackboardSyncRequest,
    BoundaryMemory,
    CreateOrgRequest,
    NodeBlackboard,
    NodeOwnership,
    OrgNode,
    OrgNodeDetail,
    OrgSummary,
    RoutedMessage,
    RouteMessageRequest,
    SnapshotEvent,
    TeachRequest,
)
from openhands.agent_server.director.org_store import OrgChange
from openhands.agent_server.director.ownership import node_ownership
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.errors import error_responses
from openhands.sdk.errors import (
    Conflict,
    DomainError,
    Invalid,
    NotFound,
    NotSupported,
    RateLimited,
    Unavailable,
    UpstreamFailed,
)


director_router = APIRouter(prefix="/director", tags=["Director"])

# How often an idle stream sends a comment, so proxies keep it open.
STREAM_HEARTBEAT_SECONDS = 15.0

_NODE_NOT_FOUND: dict[type[DomainError], str] = {
    NotFound: "No such node in the org tree"
}
_NO_BLACKBOARD: dict[type[DomainError], str] = {
    Unavailable: "The director's directory is not a git repository",
}


_NO_REMOTE: dict[type[DomainError], str] = {
    **_NO_BLACKBOARD,
    NotFound: "The repository has no remote of that name",
    UpstreamFailed: "The remote could not be reached, or refused",
    NotSupported: "The installed git is too old to merge blackboards",
}


def get_director_service(request: Request) -> DirectorService:
    return request.app.state.director_service


def _require_node(service: DirectorService, node_id: str) -> OrgNode:
    node = service.store.get(node_id)
    if node is None:
        raise NotFound(f"Unknown node: {node_id}")
    return node


def _sse(event: SnapshotEvent | OrgChange) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


@director_router.get(
    "/org/tree-stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "Server-sent events. The first is a `snapshot` of the focused "
                "node, its ancestors, its direct reports, the direct reports of "
                "every expanded node, and every escalation in the tree that "
                "waits on the human. `node_upsert` and "
                "`node_removed` events for those same nodes follow, and an "
                "`escalations` event carries the whole new list whenever it "
                "changes."
            ),
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        **error_responses(
            {NotFound: "The node does not exist, or the org tree is empty"}
        ),
    },
)
async def stream_org_tree(
    service: DirectorService = Depends(get_director_service),
    node: str | None = Query(
        default=None, description="Node to focus on. Defaults to the root."
    ),
    expanded: Annotated[
        list[str] | None,
        Query(
            description=(
                "Nodes whose direct reports to follow as well. Repeat the "
                "parameter for each one; unknown ids are ignored."
            ),
        ),
    ] = None,
    full: Annotated[
        bool,
        Query(
            description=(
                "Follow every node below the focused one, for a client that "
                "draws the whole chart."
            ),
        ),
    ] = False,
) -> StreamingResponse:
    """Stream the part of the org tree a client has open as server-sent events.

    A client subscribes to the node it shows as the root of its chart plus
    the nodes it has expanded, so its traffic grows with what is on screen
    and not with the size of the tree. To expand or collapse a node, open a
    new stream with the new set. `full` follows the whole subtree instead.
    """
    # The first use reads the tree from git.
    store = await asyncio.to_thread(lambda: service.store)
    focus_id = node if node is not None else store.root_id
    if focus_id is None:
        raise NotFound("The org tree is empty")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[OrgChange] = asyncio.Queue()

    def on_change(event: OrgChange) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    snapshot, unsubscribe = store.subscribe(
        focus_id, on_change, expanded or (), full=full
    )
    if snapshot is None:
        raise NotFound(f"Unknown node: {focus_id}")

    async def events() -> AsyncIterator[str]:
        try:
            yield _sse(snapshot)
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=STREAM_HEARTBEAT_SECONDS
                    )
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield _sse(event)
        finally:
            unsubscribe()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            # Tell buffering proxies to pass events through as they are written.
            "X-Accel-Buffering": "no",
        },
    )


@director_router.get("/org", response_model=OrgSummary)
def get_org(service: DirectorService = Depends(get_director_service)) -> OrgSummary:
    """Get whether an org exists and where one would work.

    Without an org `root_id` is null and the tree stream answers 404; start
    one with `POST /org`.
    """
    return service.summary()


@director_router.post(
    "/org",
    response_model=OrgNode,
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(
        {
            Conflict: "The director already has an org, or serves the fixture",
            Invalid: (
                "The conversation cannot lead an org: it runs in Docker, or "
                "its agent is an ACP agent"
            ),
            RateLimited: "The server cannot run another conversation now",
            Unavailable: (
                "The director is not running agents, or the conversation's "
                "workspace is not in a git repository"
            ),
        }
    ),
)
async def create_org(
    request: CreateOrgRequest,
    service: DirectorService = Depends(get_director_service),
) -> OrgNode:
    """Start an org: a root team whose IC is briefed with the task.

    The IC runs as the given conversation and its Manager, like every agent of
    the teams it later splits into, is derived from it. The org lives in the
    git repository of that conversation's workspace, which the server keeps
    for its next start. Follow the org on the tree stream.
    """
    return await service.create_org(request)


@director_router.post(
    "/org/messages",
    response_model=RoutedMessage,
    responses=error_responses(
        {
            Conflict: "No team of the org can take the message",
            Invalid: (
                "The path is outside the org's repository; or there is no org and "
                "the request has no `conversation` to start one with, or one "
                "that cannot lead an org"
            ),
            RateLimited: "The server cannot run another conversation now",
            Unavailable: (
                "The director is not running agents, or there is no org and "
                "the conversation's workspace is not in a git repository"
            ),
        }
    ),
)
async def route_message(
    request: RouteMessageRequest,
    service: DirectorService = Depends(get_director_service),
) -> RoutedMessage:
    """Say something to the org from a directory of the repository.

    The message goes to the IC of the nearest team: the search starts at
    `path` (by default the directory the server was started for) and moves
    up one directory at a time until it reaches a directory a team owns,
    skipping a team whose IC is gone or that may only read `path`. The root
    team takes what no team below it does. A server started in a git
    submodule that has no org of its own searches on through the repository
    that has the submodule.

    When there is no org yet, the message needs a `conversation`. If the
    repository of that conversation's workspace (or one above that has it as
    a submodule) has a saved org, the director takes that org up and routes
    the message in it. Otherwise the message starts an org there: the root
    team is briefed with it as its task and named by a model that summarizes
    it.
    """
    return await service.route_message(request)


@director_router.post(
    "/org/nodes/{node_id}/escalations/{escalation_id}/answer",
    response_model=OrgNode,
    responses=error_responses(
        {
            NotFound: "No such node, or it has no such escalation",
            Conflict: "The escalation is still with a team, not with the human",
            Unavailable: "The director is not running agents",
        }
    ),
)
def answer_escalation(
    node_id: str,
    escalation_id: str,
    request: AnswerEscalationRequest,
    service: DirectorService = Depends(get_director_service),
) -> OrgNode:
    """Answer an escalation that waits on the human, which settles it.

    The answer is sent to the IC of the team that raised it.
    """
    return service.answer_escalation(node_id, escalation_id, request.answer)


@director_router.get(
    "/org/nodes/{node_id}",
    response_model=OrgNodeDetail,
    responses=error_responses(_NODE_NOT_FOUND),
)
def get_org_node(
    node_id: str, service: DirectorService = Depends(get_director_service)
) -> OrgNodeDetail:
    """Get a node with its ancestors and its direct reports."""
    node = _require_node(service, node_id)
    return OrgNodeDetail(
        node=node,
        ancestors=service.store.ancestors(node_id),
        children=service.store.children(node_id),
    )


@director_router.get(
    "/org/nodes/{node_id}/blackboard",
    response_model=NodeBlackboard,
    responses=error_responses({**_NODE_NOT_FOUND, **_NO_BLACKBOARD}),
)
def get_node_blackboard(
    node_id: str, service: DirectorService = Depends(get_director_service)
) -> NodeBlackboard:
    """Get what a node works to: the spec of its boundary, the contracts
    between its sub-teams and the ones the team above set for it, its
    memory and its autopsies."""
    node = _require_node(service, node_id)
    path = node.boundary.path
    above = service.store.get(node.parent_id) if node.parent_id else None
    blackboard = service.blackboard
    specs, contracts = blackboard.applies_to(
        path, above.boundary.path if above else None
    )
    return NodeBlackboard(
        specs=specs,
        contracts=contracts,
        memory=blackboard.memory(path),
        autopsies=blackboard.autopsies(path),
    )


@director_router.get(
    "/org/nodes/{node_id}/ownership",
    response_model=NodeOwnership,
    responses=error_responses(_NODE_NOT_FOUND),
)
def get_node_ownership(
    node_id: str, service: DirectorService = Depends(get_director_service)
) -> NodeOwnership:
    """Get which direct report owns each file of a node's boundary, and the
    files where the committed work of two of these teams does not merge.

    The files are those of the node's own checkout. A team that works in a
    git worktree of its own is compared by the tip of that worktree.
    """
    node = _require_node(service, node_id)
    return node_ownership(node, service.store.children(node_id))


@director_router.post(
    "/org/nodes/{node_id}/teach",
    response_model=BoundaryMemory,
    responses=error_responses(
        {
            **_NODE_NOT_FOUND,
            **_NO_BLACKBOARD,
            Invalid: "The rule is blank",
        }
    ),
)
def teach_node(
    node_id: str,
    request: TeachRequest,
    service: DirectorService = Depends(get_director_service),
) -> BoundaryMemory:
    """Add a rule to the permanent memory of a node's boundary.

    Teaching a rule the boundary already has changes nothing.
    """
    path = _require_node(service, node_id).boundary.path
    blackboard = service.blackboard
    blackboard.teach(path, request.rule)
    return blackboard.memory(path)


@director_router.post(
    "/org/blackboard/pull",
    response_model=BlackboardSync,
    responses=error_responses(_NO_REMOTE),
)
def pull_blackboard(
    request: BlackboardSyncRequest | None = None,
    service: DirectorService = Depends(get_director_service),
) -> BlackboardSync:
    """Take in the blackboard a remote of the repository has.

    Rules and autopsies both sides added are kept from both. Where both
    changed a spec or contracts, the local text is kept and the file is
    named in `kept_local`.
    """
    remote = (request or BlackboardSyncRequest()).remote
    return blackboard_sync.pull(service.blackboard, remote)


@director_router.post(
    "/org/blackboard/push",
    response_model=BlackboardSync,
    responses=error_responses(
        {**_NO_REMOTE, Conflict: "There is no blackboard to push yet"}
    ),
)
def push_blackboard(
    request: BlackboardSyncRequest | None = None,
    service: DirectorService = Depends(get_director_service),
) -> BlackboardSync:
    """Give a remote of the repository the local blackboard.

    If the remote has changes the local blackboard lacks, they are pulled
    in first, as `pull` does, and the push is made once more.
    """
    remote = (request or BlackboardSyncRequest()).remote
    return blackboard_sync.push(service.blackboard, remote)
