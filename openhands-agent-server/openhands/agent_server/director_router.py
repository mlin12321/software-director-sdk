"""Director router for OpenHands Agent Server.

Serves the org tree of agent teams, a live stream of its changes, and the
blackboard that applies to each node. State lives in ``director/``.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from openhands.agent_server.director.models import (
    BoundaryMemory,
    NodeBlackboard,
    OrgNode,
    OrgNodeDetail,
    SnapshotEvent,
    TeachRequest,
)
from openhands.agent_server.director.org_store import OrgChange
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.errors import error_responses
from openhands.sdk.errors import DomainError, Invalid, NotFound, Unavailable


director_router = APIRouter(prefix="/director", tags=["Director"])

# How often an idle stream sends a comment, so proxies keep it open.
STREAM_HEARTBEAT_SECONDS = 15.0

_NODE_NOT_FOUND: dict[type[DomainError], str] = {
    NotFound: "No such node in the org tree"
}
_NO_BLACKBOARD: dict[type[DomainError], str] = {
    Unavailable: "The director's directory is not a git repository",
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
                "every expanded node, and the open escalations at or below the "
                "focused node. `node_upsert` and `node_removed` events for "
                "those same nodes follow, and an `escalations` event carries "
                "the whole new list whenever it changes."
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
) -> StreamingResponse:
    """Stream the part of the org tree a client has open as server-sent events.

    A client subscribes to the node it shows as the root of its chart plus
    the nodes it has expanded, so its traffic grows with what is on screen
    and not with the size of the tree. To expand or collapse a node, open a
    new stream with the new set.
    """
    store = service.store
    focus_id = node if node is not None else store.root_id
    if focus_id is None:
        raise NotFound("The org tree is empty")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[OrgChange] = asyncio.Queue()

    def on_change(event: OrgChange) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    snapshot, unsubscribe = store.subscribe(focus_id, on_change, expanded or ())
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
    """Get the specs, memory and autopsies that apply to a node's boundary."""
    path = _require_node(service, node_id).boundary.path
    blackboard = service.blackboard
    return NodeBlackboard(
        specs=blackboard.inherited(path),
        memory=blackboard.memory(path),
        autopsies=blackboard.autopsies(path),
    )


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
