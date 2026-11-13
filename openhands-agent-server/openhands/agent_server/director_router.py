"""Director router for OpenHands Agent Server.

Serves the org tree of agent teams, a live stream of its changes, and the
blackboard that applies to each node. State lives in ``director/``.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

# The org's agents get their tools from this server, so it registers them.
import openhands.agent_server.director.tools  # noqa: F401
from openhands.agent_server.director import blackboard_sync
from openhands.agent_server.director.acp import TOOLS_PATH
from openhands.agent_server.director.mcp_shim import TOKEN_HEADER
from openhands.agent_server.director.models import (
    AnswerEscalationRequest,
    BlackboardSync,
    BlackboardSyncRequest,
    BoundaryMemory,
    CreateOrgRequest,
    DirectorSettingsState,
    DirectorSettingsUpdate,
    NodeBlackboard,
    NodeOwnership,
    OrgMerge,
    OrgNode,
    OrgNodeDetail,
    OrgStateEvent,
    OrgSummary,
    OrgSync,
    OrgSyncRequest,
    Recovery,
    ResolveMergeRequest,
    RoutedMessage,
    RouteMessageRequest,
    SnapshotEvent,
    StartAgainRequest,
    TeachRequest,
    TeamMatch,
)
from openhands.agent_server.director.org_store import OrgChange
from openhands.agent_server.director.ownership import node_ownership
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools.remote import (
    RemoteTool,
    ToolAnswer,
    call_tool,
    list_tools,
)
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
# For agents that run their own tools. They have no session key of the
# server; each is known by its own token.
director_tools_router = APIRouter(prefix=TOOLS_PATH, tags=["Director"])

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
    service: DirectorService = request.app.state.director_service
    service.reached_at(str(request.base_url).rstrip("/"))
    return service


def _calling_agent(
    request: Request,
    token: Annotated[str | None, Header(alias=TOKEN_HEADER)] = None,
) -> UUID:
    """The conversation of the agent that shows its token."""
    service: DirectorService = request.app.state.director_service
    try:
        return service.agent_of(token or "")
    except NotFound as error:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(error)) from error


def _require_node(service: DirectorService, node_id: str) -> OrgNode:
    node = service.store.get(node_id)
    if node is None:
        raise NotFound(f"Unknown node: {node_id}")
    return node


def _sse(event: SnapshotEvent | OrgChange | OrgStateEvent) -> str:
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
                "changes. The snapshot says whether the org is paused and "
                "whether a pull waits for choices; an `org_state` event "
                "follows whenever either changes."
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
    queue: asyncio.Queue[OrgChange | OrgStateEvent] = asyncio.Queue()

    def on_change(event: OrgChange | OrgStateEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    snapshot, unsubscribe = store.subscribe(
        focus_id, on_change, expanded or (), full=full
    )
    if snapshot is None:
        raise NotFound(f"Unknown node: {focus_id}")
    unwatch = service.watch_state(on_change)
    state = await asyncio.to_thread(service.state)

    async def events() -> AsyncIterator[str]:
        try:
            yield _sse(
                snapshot.model_copy(
                    update={"paused": state.paused, "merging": state.merging}
                )
            )
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
            unwatch()

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


@director_router.get("/settings", response_model=DirectorSettingsState)
def get_settings(
    service: DirectorService = Depends(get_director_service),
) -> DirectorSettingsState:
    """Get the limits on new teams, beside how large the org is now."""
    return service.settings_state()


@director_router.patch("/settings", response_model=DirectorSettingsState)
def update_settings(
    request: DirectorSettingsUpdate,
    service: DirectorService = Depends(get_director_service),
) -> DirectorSettingsState:
    """Change the limits on new teams.

    A limit is checked only when a team splits. Lowering one below what the
    org already has removes nothing; it refuses the next split. The values
    are kept for the server's next start.
    """
    return service.update_settings(request)


@director_router.get("/org/search", response_model=list[TeamMatch])
def search_org(
    q: Annotated[
        str,
        Query(
            description=(
                "Words to find; a team matches when its title or its "
                "directory has all of them, in any case."
            ),
            max_length=200,
        ),
    ],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    service: DirectorService = Depends(get_director_service),
) -> list[TeamMatch]:
    """Find teams by title or directory, parents before their sub-teams."""
    return service.search(q, limit)


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
    "/org/pause",
    response_model=Recovery,
    responses=error_responses(
        {
            Conflict: "There is no org, or the director serves the fixture",
            Unavailable: "The director is not running agents",
        }
    ),
)
async def pause_org(
    service: DirectorService = Depends(get_director_service),
) -> Recovery:
    """Hold the org.

    Its agents stop after the step they are in. The steps the director
    takes for them and the messages to them, the human's included, wait and
    are kept across restarts. What agents still report is recorded. Pausing
    a paused org changes nothing. The answer says what `POST /org/resume`
    would do.
    """
    return await service.pause()


@director_router.post(
    "/org/resume",
    response_model=Recovery,
    responses=error_responses({Unavailable: "The director is not running agents"}),
)
async def resume_org(
    service: DirectorService = Depends(get_director_service),
) -> Recovery:
    """Take a paused org up again.

    Teams that waited to start are started in the order they waited, steps
    the director had begun are finished, held messages are delivered in
    order and the agents that were running run again. An agent the server
    ended in the middle of a step is told so first. The answer lists only
    what was put aside because it no longer fits the org.
    """
    return await service.resume()


@director_router.get("/org/recovery", response_model=Recovery)
async def get_recovery(
    service: DirectorService = Depends(get_director_service),
) -> Recovery:
    """Get what resuming a paused org does, team by team.

    An org is paused by `POST /org/pause`, and comes back paused when the
    server stopped or ended while it was in the middle of its work
    (`OrgSummary.paused` says which). Without a paused org the list is
    empty.
    """
    return await service.recovery()


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


@director_router.post(
    "/org/pull",
    response_model=OrgSync,
    responses=error_responses(
        {
            **_NO_REMOTE,
            Conflict: "A merge waits for choices, or this is the fixture",
            Unavailable: "The director is not running, or has no repository",
        }
    ),
)
async def pull_org(
    request: OrgSyncRequest | None = None,
    service: DirectorService = Depends(get_director_service),
) -> OrgSync:
    """Take in the org a remote of the repository has, as git merges a branch.

    The tree, the blackboard and the branches of the teams' work are
    fetched. A team changed on one side only is taken from that side, and so
    is a field of a team; a repository without an org takes the remote's.
    Where both sides changed the same thing the result is `conflicts`:
    nothing is written, the org stays paused, and `merge` lists what to
    choose with `POST /org/merge/resolve`.
    """
    return await service.pull_org((request or OrgSyncRequest()).remote)


@director_router.post(
    "/org/push",
    response_model=OrgSync,
    responses=error_responses(
        {
            **_NO_REMOTE,
            Conflict: (
                "There is no org, a merge waits for choices, or the remote has "
                "changes that were not pulled"
            ),
        }
    ),
)
async def push_org(
    request: OrgSyncRequest | None = None,
    service: DirectorService = Depends(get_director_service),
) -> OrgSync:
    """Give a remote of the repository the org, the blackboard and the
    branches of the teams' work.

    The branches are kept on the remote under `refs/agents/teams/`. A push
    is refused while the remote has changes of the org that are not here.
    """
    return await service.push_org((request or OrgSyncRequest()).remote)


@director_router.get("/org/merge", response_model=OrgMerge | None)
async def get_merge(
    service: DirectorService = Depends(get_director_service),
) -> OrgMerge | None:
    """The merge that waits for choices; null when none does."""
    return await service.merge()


@director_router.post(
    "/org/merge/resolve",
    response_model=OrgSync,
    responses=error_responses(
        {
            Conflict: "No merge waits for choices",
            Invalid: "A choice names no conflict of the merge",
        }
    ),
)
async def resolve_merge(
    request: ResolveMergeRequest,
    service: DirectorService = Depends(get_director_service),
) -> OrgSync:
    """Choose `mine` or `theirs` for conflicts of the merge that waits.

    When every conflict has a choice the merge is written, as a commit with
    both sides as parents, and the org is resumed if the pull had paused it.
    Until then the result is `conflicts` with what is still open.
    """
    return await service.resolve_merge(request)


@director_router.post(
    "/org/merge/abort",
    response_model=OrgSync,
    responses=error_responses({Conflict: "No merge waits for choices"}),
)
async def abort_merge(
    service: DirectorService = Depends(get_director_service),
) -> OrgSync:
    """Abandon the merge that waits. The org is as it was before the pull."""
    return await service.abort_merge()


@director_router.post(
    "/org/nodes/{node_id}/start_again",
    response_model=OrgNode,
    responses=error_responses(
        {
            NotFound: "There is no such team",
            Conflict: "The team is done, or its IC has a conversation here",
            Unavailable: "The director is not running agents",
        }
    ),
)
async def start_team_again(
    node_id: str,
    request: StartAgainRequest,
    service: DirectorService = Depends(get_director_service),
) -> OrgNode:
    """Start afresh a team whose agents have no conversation on this server.

    That is the state of every team after its org was pulled into another
    clone: the tree, the specs and the teams' branches travel, the agents'
    conversations do not. The team's IC starts as the given conversation in
    the team's own checkout and is briefed from the team's spec.
    """
    return await service.start_again(node_id, request)


@director_tools_router.get("", dependencies=[Depends(_calling_agent)])
def list_agent_tools() -> list[RemoteTool]:
    """The director's tools, for an agent that runs its own tools."""
    return list_tools()


@director_tools_router.post(
    "/{name}", responses=error_responses({NotFound: "The director has no such tool"})
)
async def call_agent_tool(
    name: str,
    arguments: dict[str, Any],
    request: Request,
    conversation_id: UUID = Depends(_calling_agent),
) -> ToolAnswer:
    """Run one of the director's tools for the agent that calls it.

    What the director refuses comes back in the answer, for the agent to
    read and act on.
    """
    service: DirectorService = request.app.state.director_service
    return await asyncio.to_thread(call_tool, service, conversation_id, name, arguments)
