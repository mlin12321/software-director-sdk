"""Director router for OpenHands Agent Server.

Serves the org tree of agent teams, a live stream of its changes, and the
team memory that applies to each team. State lives in ``director/``.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse

# The org's agents get their tools from this server, so it registers them.
import openhands.agent_server.director.tools  # noqa: F401
from openhands.agent_server.director.acp import TOOLS_PATH
from openhands.agent_server.director.changes import file_sides, team_changes
from openhands.agent_server.director.mcp_shim import TOKEN_HEADER
from openhands.agent_server.director.models import (
    AnswerEscalationRequest,
    ConversationPlace,
    CreateOrgRequest,
    DirectorSettingsState,
    DirectorSettingsUpdate,
    EditFileRequest,
    FileEdit,
    FileSides,
    FormerTeamSummary,
    NodeMemory,
    NodeOwnership,
    OpenedSession,
    OpenSessionRequest,
    OrgHistory,
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
    Session,
    SessionTeam,
    SnapshotEvent,
    StartAgainRequest,
    TeamChanges,
    TeamMatch,
    Thread,
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
    Forbidden,
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
    NotFound: "No such team in the org tree"
}
_NO_MEMORY: dict[type[DomainError], str] = {
    Unavailable: "The director's directory is not a git repository",
}


_NO_REMOTE: dict[type[DomainError], str] = {
    **_NO_MEMORY,
    NotFound: "The repository has no remote of that name",
    UpstreamFailed: "The remote could not be reached, or refused",
    NotSupported: "The installed git is too old to merge team memories",
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
    """A team of the org tree. What a session has at a team is no team."""
    node = service.store.team(node_id)
    if node is None:
        raise NotFound(f"Unknown team: {node_id}")
    return node


def _require_any(service: DirectorService, node_id: str) -> OrgNode:
    """A team of the org tree, or a place of a session, by its id."""
    node = service.store.get(node_id)
    if node is None:
        raise NotFound(f"Unknown team: {node_id}")
    return node


def _require_place(service: DirectorService, node_id: str) -> SessionTeam:
    """A place of a session: where a team is worked on."""
    node = service.store.place(node_id)
    if node is None:
        raise NotFound(f"Nobody works at {node_id}: name a team of a session")
    return node


def _require_session(service: DirectorService, session_id: str) -> Session:
    for session in service.sessions():
        if session.id == session_id:
            return session
    raise NotFound(f"No session {session_id!r} is open")


def _sse(event: SnapshotEvent | OrgChange | OrgStateEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


@director_router.get(
    "/org/tree-stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "Server-sent events. The first is a `snapshot` of the focused "
                "team, its ancestors, its direct reports, the direct reports of "
                "every expanded team, the open sessions, and every "
                "escalation that waits on the human. `node_upsert` and "
                "`node_removed` events for those same teams follow, and a "
                "`sessions` or an `escalations` event carries the whole new "
                "list whenever it changes. The snapshot says whether the org "
                "is paused and whether a pull waits for choices; an "
                "`org_state` event "
                "follows whenever either changes."
            ),
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        **error_responses(
            {NotFound: "The team does not exist, or the org tree is empty"}
        ),
    },
)
async def stream_org_tree(
    service: DirectorService = Depends(get_director_service),
    node: str | None = Query(
        default=None, description="Team to focus on. Defaults to the root."
    ),
    expanded: Annotated[
        list[str] | None,
        Query(
            description=(
                "Teams whose direct reports to follow as well. Repeat the "
                "parameter for each one; unknown ids are ignored."
            ),
        ),
    ] = None,
    full: Annotated[
        bool,
        Query(
            description=(
                "Follow every team below the focused one, for a client that "
                "draws the whole chart."
            ),
        ),
    ] = False,
) -> StreamingResponse:
    """Stream the part of the org tree a client has open as server-sent events.

    A client subscribes to the team it shows as the root of its chart plus
    the teams it has expanded, so its traffic grows with what is on screen
    and not with the size of the tree. To expand or collapse a team, open a
    new stream with the new set. `full` follows the whole subtree instead.
    The teams are the teams of the org; the work at them is in `sessions`.
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
        raise NotFound(f"Unknown team: {focus_id}")
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
    response_model=OpenedSession,
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(
        {
            Conflict: "The director already has an org",
            Invalid: ("The conversation cannot lead an org: it runs in Docker"),
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
) -> OpenedSession:
    """Start an org: its top team, and the session that works on the task.

    The session's IC runs as the given conversation, and every other agent
    of the org is derived from one that runs already. Nobody works at a
    team of the org itself: all work is done in sessions, and accepted work
    is merged into the checkout of the repository the conversation's
    workspace is in, which the server keeps for its next start. Follow the
    org on the tree stream.
    """
    return await service.create_org(request)


@director_router.post(
    "/org/pause",
    response_model=Recovery,
    responses=error_responses(
        {
            Conflict: "There is no org",
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

    The message opens a session at the nearest team, whose IC works on it:
    the search starts at `path` (by default the directory the server was
    started for) and moves up one directory at a time until it reaches a
    directory a team owns, skipping a team that may only read `path`. The
    root team takes what no team below it does. A server started in a git
    submodule that has no org of its own searches on through the repository
    that has the submodule.

    With `auto`, the director may open the session at a team below that
    one: the lowest team that holds every team the text concerns, by the
    files it names and by what a model makes of it. It does so only when it
    is sure, and the answer says how the team was chosen.

    With `session_id` the message continues that session and opens none: it
    goes to the IC of the team the session was opened at, or of a team the
    session has below it when the director is sure, a team the session set
    up itself included.

    When there is no org yet, the message needs a `conversation`. If the
    repository of that conversation's workspace (or one above that has it as
    a submodule) has a saved org, the director takes that org up and routes
    the message in it. Otherwise the message starts an org there: its first
    session is briefed with it as its task, and the root team is named by a
    model that summarizes it.
    """
    return await service.route_message(request)


@director_router.get(
    "/org/nodes/{node_id}",
    response_model=OrgNodeDetail,
    responses=error_responses(_NODE_NOT_FOUND),
)
def get_org_node(
    node_id: str, service: DirectorService = Depends(get_director_service)
) -> OrgNodeDetail:
    """Get a team of the org tree with the teams above it and its
    sub-teams. A place of a session is read with its session."""
    node = _require_node(service, node_id)
    return OrgNodeDetail(
        node=node,
        ancestors=service.store.teams_above(node_id),
        children=service.store.sub_teams(node_id),
    )


@director_router.get(
    "/org/conversations/{conversation_id}",
    response_model=ConversationPlace,
    responses=error_responses({NotFound: "No agent of the org has this conversation"}),
)
def get_conversation_place(
    conversation_id: str, service: DirectorService = Depends(get_director_service)
) -> ConversationPlace:
    """Say where an agent's conversation is in the org.

    With it come the conversations one level away: the agents of the team
    above, the other agent of the same team, and the ICs of the teams
    directly below; for an agent that reviews a session's work, the
    session's IC is the one below. A client shows them as links from one
    conversation to the next.
    """
    return service.place_of(conversation_id)


@director_router.post(
    "/org/conversations/{conversation_id}/joined",
    response_model=ConversationPlace,
    responses=error_responses(
        {
            NotFound: "No agent of the org has this conversation",
            Unavailable: "The director is not running agents",
        }
    ),
)
def join_conversation(
    conversation_id: str, service: DirectorService = Depends(get_director_service)
) -> ConversationPlace:
    """Note that the human has written in an agent's conversation.

    The conversation gets the tag `humanjoined=true` beside
    `startedby=director`, so that a client lists it with the conversations
    the human started. A message sent through the director, and the request
    a session is opened with, do this themselves; a client that writes to
    an agent's conversation directly calls this. Calling it again changes
    nothing.
    """
    service.joined(conversation_id)
    return service.place_of(conversation_id)


@director_router.get("/org/sessions", response_model=list[Session])
def list_sessions(
    service: DirectorService = Depends(get_director_service),
) -> list[Session]:
    """List the sessions that are open.

    A session is a line of work the human opened at a team; all work of an
    org is done in sessions. It is no part of the org tree: the tree is the
    org's structure, and a session has agents and a checkout of its own at the
    team it was opened at and, once they are given a task in it, at the
    teams below that one (`teams`).
    """
    return service.sessions()


@director_router.get(
    "/org/sessions/{session_id}",
    response_model=Session,
    responses=error_responses({NotFound: "No such session is open"}),
)
def get_session(
    session_id: str, service: DirectorService = Depends(get_director_service)
) -> Session:
    """Get one open session with what it has at each of its teams."""
    return _require_session(service, session_id)


@director_router.post(
    "/org/sessions/{session_id}/teams/{node_id}/escalations/{escalation_id}/answer",
    response_model=SessionTeam,
    responses=error_responses(
        {
            NotFound: "No such session or team, or it has no such escalation",
            Conflict: "The escalation is still with a team, not with the human",
            Unavailable: "The director is not running agents",
        }
    ),
)
def answer_session_escalation(
    session_id: str,
    node_id: str,
    escalation_id: str,
    request: AnswerEscalationRequest,
    service: DirectorService = Depends(get_director_service),
) -> OrgNode:
    """Answer an escalation a team raised in a session and that waits on
    the human, which settles it.

    The answer is sent to the agent that raised it. For a wait the
    director raised itself (a merge that could not go into the checkout,
    a Division Lead Manager that does not answer) it only lets the work
    go on.
    """
    session = _require_session(service, session_id)
    team = next((team for team in session.teams if team.team_id == node_id), None)
    if team is None:
        raise NotFound(f"Session {session_id!r} has no team {node_id}")
    return service.answer_escalation(team.id, escalation_id, request.answer)


@director_router.post(
    "/org/sessions",
    response_model=OpenedSession,
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(
        {
            NotFound: "There is no such team",
            Conflict: (
                "There is no org, it is paused, or no conversation of the "
                "org runs on this server to start the session's IC from"
            ),
            Unavailable: "The director is not running agents",
        }
    ),
)
async def open_session(
    request: OpenSessionRequest,
    service: DirectorService = Depends(get_director_service),
) -> OpenedSession:
    """Open a session at a team and start it with a request.

    The team gets an IC for the request, in a git worktree and on a branch
    of its own, so that it can work on
    the team's directory while other sessions do. The session has a place
    for each team below, which starts when that IC gives it a task. An IC
    at the team above, started for it and only reviewing, reviews the
    session's work like a sub-team's, and an accept merges it into the
    repository's checkout; the work of a session opened at the top team is
    reviewed from the top team itself.

    Without `node_id` the director chooses the team by what the text
    concerns, as for a message sent with `auto`. Opening a session adds no
    team to the org, so `max_teams` and `max_depth` do not count it.
    """
    return await service.open_session(request)


@director_router.delete(
    "/org/sessions/{session_id}",
    status_code=status.HTTP_202_ACCEPTED,
    # It answers with no body.
    response_class=Response,
    responses=error_responses({NotFound: "No such session is open"}),
)
def close_session(
    session_id: str, service: DirectorService = Depends(get_director_service)
) -> None:
    """Close a session.

    Its agents end the step they are in, and are removed with their
    conversations and the session's checkout. Work of the
    session that was accepted stays where it was merged. Uncommitted work
    is kept as a commit under `refs/agents/pruned/`, and commits that were
    never merged stay reachable from `refs/agents/sessions/<session id>/`.
    """
    service.close_session(session_id)


@director_router.get(
    "/org/history", response_model=OrgHistory, responses=error_responses(_NO_MEMORY)
)
def get_org_history(
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    before: Annotated[
        str | None, Query(description="Only events before this ISO 8601 time.")
    ] = None,
    service: DirectorService = Depends(get_director_service),
) -> OrgHistory:
    """Get what became of the teams of the whole org, newest first: set up,
    work accepted or sent back, stalled, stopped, replaced, pruned, and what
    was proposed, approved and declined."""
    return service.org_history(limit, before)


@director_router.get(
    "/org/nodes/{node_id}/memory",
    response_model=NodeMemory,
    responses=error_responses({**_NODE_NOT_FOUND, **_NO_MEMORY}),
)
def get_node_memory(
    node_id: str, service: DirectorService = Depends(get_director_service)
) -> NodeMemory:
    """Get what a team works to: the spec of its boundary, the contracts
    between its sub-teams and the ones the team above set for it, and
    what its two agents reported."""
    node = _require_any(service, node_id)
    return NodeMemory(
        specs=service.specs_of(node),
        contracts=service.contracts_of(node),
        reports=service.reports_of(node),
        former_teams=[
            FormerTeamSummary.model_validate(former.model_dump())
            for former in service.former_teams_of(node)
        ],
        history=service.history_of(node.id),
    )


@director_router.get(
    "/org/nodes/{node_id}/messages",
    response_model=list[Thread],
    responses=error_responses({**_NODE_NOT_FOUND, **_NO_MEMORY}),
)
def get_node_messages(
    node_id: str, service: DirectorService = Depends(get_director_service)
) -> list[Thread]:
    """Get what the agents of a place of a session have said to Managers
    and ICs: its IC and its Manager to each other, its Manager and its IC to
    the ICs directly below, and the Manager and the IC above to its IC.
    Empty for a team of the org tree."""
    place = service.store.place(node_id)
    if place is None:
        # Nobody works at a team of the org itself, so nothing was said there.
        _require_node(service, node_id)
        return []
    return service.threads_of(place)


@director_router.get(
    "/org/nodes/{node_id}/ownership",
    response_model=NodeOwnership,
    responses=error_responses(_NODE_NOT_FOUND),
)
def get_node_ownership(
    node_id: str, service: DirectorService = Depends(get_director_service)
) -> NodeOwnership:
    """Get which direct report owns each file of a team's boundary, and the
    files where the committed work of two of these teams does not merge.

    For a team of the org the files are those of the repository's checkout
    and no collisions are looked for (`collisions_checked` is false).
    For a place of a session they are those of its own checkout, and a
    place below it that works in a git worktree of its own is compared by
    the tip of that worktree.
    """
    store = service.store
    place = store.place(node_id)
    if place is not None:
        return node_ownership(place, store.below(node_id), Path(place.worktree_path))
    node = _require_node(service, node_id)
    return node_ownership(node, store.sub_teams(node_id), service.repo_dir)


_NO_CHECKOUT: dict[type[DomainError], str] = {
    NotFound: "No such place of a session, or its checkout is not on this server",
}


@director_router.get(
    "/org/nodes/{node_id}/changes",
    response_model=TeamChanges,
    responses=error_responses(_NO_CHECKOUT),
)
def get_node_changes(
    node_id: str, service: DirectorService = Depends(get_director_service)
) -> TeamChanges:
    """List the files a team changed in its directory since its work
    started or was last accepted, uncommitted and new files included.
    `node_id` is a place of a session."""
    return team_changes(_require_place(service, node_id))


@director_router.get(
    "/org/nodes/{node_id}/changes/file",
    response_model=FileSides,
    responses=error_responses(
        {
            **_NO_CHECKOUT,
            Invalid: "The path does not name a file of a checkout",
            Forbidden: "The file is not in the team's directory",
        }
    ),
)
def get_node_changed_file(
    node_id: str,
    path: Annotated[str, Query(description="Relative to the top of the repository.")],
    service: DirectorService = Depends(get_director_service),
) -> FileSides:
    """Get one file as it was at the team's base commit and as it is in the
    team's checkout now. `node_id` is a place of a session."""
    return file_sides(_require_place(service, node_id), path)


@director_router.put(
    "/org/nodes/{node_id}/files",
    response_model=FileEdit,
    responses=error_responses(
        {
            **_NO_CHECKOUT,
            Invalid: "The path does not name a file that can be committed",
            Forbidden: "The team's agents may not write the file either",
            Conflict: "An agent of the team is in the middle of a turn",
        }
    ),
)
async def edit_node_file(
    node_id: str,
    request: EditFileRequest,
    path: Annotated[str, Query(description="Relative to the top of the repository.")],
    service: DirectorService = Depends(get_director_service),
) -> FileEdit:
    """Write one file of a team by hand. `node_id` is a place of a session.

    The file is committed alone in the team's checkout and the team's IC is
    told. Refused while an agent of the team is in the middle of a turn, and
    for a file the team's own agents may not write.
    """
    _require_place(service, node_id)
    return await service.edit_file(node_id, path, request.content)


@director_router.post(
    "/org/pull",
    response_model=OrgSync,
    responses=error_responses(
        {
            **_NO_REMOTE,
            Conflict: "A merge waits for choices",
            Unavailable: "The director is not running, or has no repository",
        }
    ),
)
async def pull_org(
    request: OrgSyncRequest | None = None,
    service: DirectorService = Depends(get_director_service),
) -> OrgSync:
    """Take in the org a remote of the repository has, as git merges a branch.

    The tree and the team memory are fetched; sessions, with their
    checkouts and branches, stay on the machine that has them. A team
    changed on one side only is taken from that side, and so is a field of
    a team; a repository without an org takes the remote's.
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
    """Give a remote of the repository the org tree and the team memory.

    Sessions, with their checkouts and branches, stay on this machine. A
    push is refused while the remote has changes of the org that are not
    here.
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
    "/org/sessions/{session_id}/teams/{node_id}/start-again",
    response_model=SessionTeam,
    responses=error_responses(
        {
            NotFound: "There is no such team",
            Conflict: "The team is done, or its IC has a conversation here",
            Unavailable: "The director is not running agents",
        }
    ),
)
async def start_team_again(
    session_id: str,
    node_id: str,
    request: StartAgainRequest,
    service: DirectorService = Depends(get_director_service),
) -> OrgNode:
    """Start afresh a team of a session whose agents have no conversation
    on this server any more. The team's IC starts as the given conversation
    in the team's own checkout and is briefed from the team's spec and what
    the session recorded.
    """
    session = _require_session(service, session_id)
    team = next((team for team in session.teams if team.team_id == node_id), None)
    if team is None:
        raise NotFound(f"Session {session_id!r} has no team {node_id}")
    return await service.start_again(team.id, request)


@director_tools_router.get("")
def list_agent_tools(
    request: Request, conversation_id: UUID = Depends(_calling_agent)
) -> list[RemoteTool]:
    """The director's tools the calling agent is offered: its role's, or
    the few of a session's Division Lead."""
    service: DirectorService = request.app.state.director_service
    try:
        return list_tools(service, conversation_id)
    except NotFound as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error


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
