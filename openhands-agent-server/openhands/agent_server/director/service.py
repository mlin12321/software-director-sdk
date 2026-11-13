"""The director's state for one server: an org tree and its blackboard."""

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import threading
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from uuid import UUID, uuid4

from openhands.agent_server.config import Config
from openhands.agent_server.conversation_service import ToolParams
from openhands.agent_server.director import bugbot, memory
from openhands.agent_server.director.acp import (
    agent_token,
    as_org_agent,
    conversation_of,
    tools_server,
)
from openhands.agent_server.director.boundary import (
    nearest_team,
    write_refusal,
    written_paths,
)
from openhands.agent_server.director.fixture import build_fixture_nodes
from openhands.agent_server.director.git import (
    GitError,
    enclosing_repository,
    git,
    head_of,
    is_git_repository,
    save_and_discard,
)
from openhands.agent_server.director.git_blackboard import (
    CONTRACTS_FILE,
    GitBlackboard,
)
from openhands.agent_server.director.git_index import (
    excludes_file,
    git_script,
    rcfile,
)
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.merge_back import merge_back
from openhands.agent_server.director.models import (
    AgentKind,
    AgentRef,
    AgentRole,
    Autopsy,
    Boundary,
    CreateOrgRequest,
    DirectorSettings,
    DirectorSettingsState,
    DirectorSettingsUpdate,
    OrgNode,
    OrgNodeInput,
    OrgSummary,
    ReviewFinding,
    RoutedMessage,
    RouteMessageRequest,
    SplitChild,
    TeamEnforcement,
    TeamMatch,
)
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    load_org,
    save_org,
)
from openhands.agent_server.director.org_store import OrgStore, OrgStoreError
from openhands.agent_server.director.pods import PodLauncher, StartedAgent
from openhands.agent_server.director.prompts import (
    MANAGER_ROLE,
    POD_PROTOCOL,
    brief,
    changes_requested,
    from_director,
    review_request,
)
from openhands.agent_server.director.runtime import bind_director
from openhands.agent_server.director.sandbox import (
    command_script,
    find_bwrap,
    shell_script,
    write_shell,
)
from openhands.agent_server.director.split import (
    AGENTS_PER_TEAM,
    check_limits,
    child_boundary,
    clean_path,
)
from openhands.agent_server.event_service import ConversationRunLimitExceeded
from openhands.sdk.agent import ACPAgent
from openhands.sdk.agent.base import AgentBase
from openhands.sdk.conversation.request import AgentLaunchAdditions
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import (
    Conflict,
    DomainError,
    NotFound,
    Unavailable,
    Unprocessable,
)
from openhands.sdk.event import ActionEvent
from openhands.sdk.tool.spec import Tool
from openhands.sdk.utils.files import atomic_write_text


logger = logging.getLogger(__name__)

ROOT_ID = "root"
# The tools every agent of an org gets; they register in ``director.tools``.
DIRECTOR_TOOL_NAMES = (
    "read_node_context",
    "write_spec",
    "propose_boundary_split",
    "decide_split",
    "raise_escalation",
    "resolve_escalation",
    "set_status",
    "review_team",
    "record_rule",
    "prune_team",
)
# How long tree changes are gathered before they are saved as one commit.
SAVE_DELAY_SECONDS = 2.0
# How long to wait before starting an agent again when the server is full.
RUN_LIMIT_RETRY_SECONDS = 5.0
# How long a model may take to name a team before the team keeps the name
# it has; the root's agents are not briefed until then.
SUMMARY_TIMEOUT_SECONDS = 20.0
# How long a model may take over a team's diff, or over a boundary's rules.
MODEL_CALL_TIMEOUT_SECONDS = 120.0
# A team whose work is sent back this many times is stopped.
MAX_REVIEW_ROUNDS = 3
# How often the rules of the blackboard are looked over for consolidation.
CONSOLIDATION_SECONDS = 900.0
# Where, in the org's repository, the uncommitted work a prune discarded is
# kept.
PRUNED_REFS = "refs/agents/pruned"
# An agent whose conversation is gone, or could not be started.
AGENT_LOST = "lost"
# Where, in the repository's git directory, the director keeps the shells
# its agents' terminals start, and what those shells need.
_SHELLS_DIR = "director-shells"
# After a conversation id: the script an ACP agent's process is started by.
_ACP_LAUNCHER = ".acp.sh"
_REMEMBERED_FILE = "director.json"
# The lane of what concerns the org as a whole, such as saving its tree.
ORG_LANE = "org"
# The lane of the rules' consolidation.
MODELS_LANE = "models"
# How many of the director's own model calls run at once.
MODEL_CALLS_AT_ONCE = 2

_AGENT_STATUS: dict[ConversationExecutionStatus, str] = {
    ConversationExecutionStatus.RUNNING: "working",
    ConversationExecutionStatus.ERROR: "blocked",
    ConversationExecutionStatus.STUCK: "blocked",
}

type Effect = Callable[[], Awaitable[None]]


def agent_conversation(node: OrgNode, role: AgentRole) -> str | None:
    """The conversation of the node's agent in ``role``, if it has one."""
    for agent in node.agents:
        if agent.role == role:
            return agent.conversation_id
    return None


def _agent_title(team: str, role: AgentRole) -> str:
    return f"{team} ({'IC' if role == 'ic' else 'Manager'})"


def _summary_of(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line[:200]


@lru_cache(maxsize=4096)
def _checkout_root(workspace: str) -> Path:
    """The top of the git checkout ``workspace`` is in, symlinks resolved."""
    try:
        top = git(Path(workspace), ["rev-parse", "--show-toplevel"]).strip()
    except (GitError, OSError):
        top = workspace
    return Path(os.path.realpath(top))


def _is_in(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


class DirectorService:
    """Owns the org tree and the blackboard of one repository.

    An org lives in the repository its first conversation was opened in,
    which need not be the directory the server was started for; the director
    remembers it across restarts. Until there is an org, the director looks
    for a saved one in the server's directory. A repository that has no org
    and is a submodule of one that has stands for the nearest such
    repository above it, in both cases.

    Nothing is read from disk until the tree or the blackboard is first used,
    so creating the service is free for servers that never use the director.

    The tree can be read and taught without :meth:`start`. Running agents
    needs it: starting them, sending them messages and saving the tree all
    happen on worker tasks, so nothing that asks (a route, an agent's tool
    on its own thread) waits on a conversation. What is asked for goes into
    a lane, usually its team's: a lane's work is carried out in the order it
    was asked for, and lanes do not wait for each other.
    """

    def __init__(
        self,
        repo_dir: Path,
        *,
        fixture: bool = False,
        save_delay: float = SAVE_DELAY_SECONDS,
        run_limit_retry: float = RUN_LIMIT_RETRY_SECONDS,
        worktree_root: Path | None = None,
        find_sandbox: Callable[[], str | None] = lambda: None,
        summary_timeout: float = SUMMARY_TIMEOUT_SECONDS,
        remembered: Path | None = None,
        find_review_instructions: Callable[[], str | None] = lambda: None,
        model_call_timeout: float = MODEL_CALL_TIMEOUT_SECONDS,
        consolidate_every: float = 0,
        max_concurrent_runs: int | None = None,
        server_url: str | None = None,
    ) -> None:
        """
        Args:
            server_url: Where this server answers, for agents that reach the
                director's tools from a process of their own; None to take
                it from the requests the director gets.
            max_concurrent_runs: How many conversations the server runs at
                once, to tell the human; None when not known.
            find_review_instructions: Returns what a model is told when it
                reads a team's diff before the IC above reviews it, or None
                to leave that pass out.
            consolidate_every: Seconds between looks at the blackboard's
                rules for boundaries to consolidate; 0 for never.
            remembered: A file to keep the org's repository in, so that an
                org started outside ``repo_dir`` is found again after a
                restart.
            worktree_root: Where the server makes the git worktrees of
                conversations, so that no team writes into another's.
            find_sandbox: Returns the bubblewrap binary to confine the
                agents' shells with, or None to leave them unconfined.
        """
        self._directory = repo_dir
        # Where the org is; moves up when the store is loaded from above.
        self._repo_dir = repo_dir
        self._fixture = fixture
        self._worktree_root = worktree_root
        self._find_sandbox = find_sandbox
        self._summary_timeout = summary_timeout
        self._remembered = remembered
        self._find_review_instructions = find_review_instructions
        self._model_call_timeout = model_call_timeout
        # Teams that had reported done and are working again only to review.
        self._done_reviewers: set[str] = set()
        self._consolidate_every = consolidate_every
        self._consolidator: asyncio.Task[None] | None = None
        self._bwrap: str | None = None
        self._stop_guard: Callable[[], None] | None = None
        self._save_delay = save_delay
        self._run_limit_retry = run_limit_retry
        self._lock = threading.Lock()
        self._store: OrgStore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._launcher: PodLauncher | None = None
        self._max_concurrent_runs = max_concurrent_runs
        self._settings: DirectorSettings | None = None
        self._configured_url = server_url
        self._server_url = server_url
        self._tool_key: bytes | None = None
        # What each lane still has to carry out; a lane exists while it works.
        self._lanes: dict[str, deque[Effect]] = {}
        self._workers: set[asyncio.Task[None]] = set()
        # Effects asked for and not carried out yet.
        self._unsettled = 0
        self._quiet = asyncio.Event()
        self._quiet.set()
        self._model_calls = asyncio.Semaphore(MODEL_CALLS_AT_ONCE)
        # Set when a conversation stopped running, so a start may fit now.
        self._room = asyncio.Event()
        # Held by the start that waits for room, so that starts are served
        # in the order they began to wait.
        self._start_gate = asyncio.Lock()
        self._save_timer: asyncio.TimerHandle | None = None
        self._stop_status: Callable[[], None] | None = None
        self._create_lock = asyncio.Lock()
        self._first_message_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: Config) -> "DirectorService":
        return cls(
            config.director_repo_path or config.workspace_path,
            fixture=config.director_fixture,
            worktree_root=config.conversation_worktree_root,
            find_sandbox=find_bwrap,
            remembered=config.conversations_path.parent / _REMEMBERED_FILE,
            find_review_instructions=bugbot.review_instructions,
            consolidate_every=config.director_consolidation_seconds,
            max_concurrent_runs=config.max_concurrent_runs,
            server_url=config.director_server_url,
        )

    @property
    def store(self) -> OrgStore:
        """The org tree, loaded on first use."""
        with self._lock:
            if self._store is None:
                self._store = self._load_store()
            return self._store

    @property
    def blackboard(self) -> GitBlackboard:
        """The blackboard of the director's repository.

        Raises:
            Unavailable: The director's directory is not a git repository.
        """
        _ = self.store  # settles which repository the org is in
        if not is_git_repository(self._repo_dir):
            raise Unavailable(
                f"The director has no blackboard: {self._repo_dir} is not a git "
                "repository. Point OH_DIRECTOR_REPO_PATH at one."
            )
        return GitBlackboard(self._repo_dir)

    def summary(self) -> OrgSummary:
        store = self.store
        return OrgSummary(
            root_id=store.root_id,
            node_count=len(store),
            repo_path=str(self._repo_dir),
            directory=self._home(),
            is_git_repository=is_git_repository(self._repo_dir),
            fixture=self._fixture,
            boundary_enforcement="sandbox" if self._bwrap else "checks",
        )

    @property
    def settings(self) -> DirectorSettings:
        """The limits on new teams, as last saved."""
        with self._lock:
            if self._settings is None:
                saved = self._read_remembered().get("settings")
                try:
                    self._settings = DirectorSettings.model_validate(saved or {})
                except ValueError:
                    logger.warning("Ignoring the saved director settings: %r", saved)
                    self._settings = DirectorSettings()
            return self._settings

    def settings_state(self) -> DirectorSettingsState:
        """The settings, with how large the org is now."""
        levels: dict[str, int] = {}
        for node in self.store.all():
            levels[node.id] = levels.get(node.parent_id or "", 0) + 1
        return DirectorSettingsState(
            **self.settings.model_dump(),
            agents=len(levels) * AGENTS_PER_TEAM,
            depth=max(levels.values(), default=0),
            max_concurrent_runs=self._max_concurrent_runs,
        )

    def update_settings(self, update: DirectorSettingsUpdate) -> DirectorSettingsState:
        """Change the limits on new teams. Existing teams are left alone."""
        changed = self.settings.model_copy(update=update.model_dump(exclude_none=True))
        with self._lock:
            self._settings = changed
        self._remember(settings=changed.model_dump())
        return self.settings_state()

    def reached_at(self, url: str) -> None:
        """Note the address a request reached this server at.

        An agent that runs its own tools calls the director's from its own
        process and is given this address to do so.
        """
        if self._configured_url is not None or url == self._url():
            return
        self._server_url = url
        self._remember(server_url=url)

    def agent_of(self, token: str) -> UUID:
        """The conversation of the agent a tool call comes from.

        Raises:
            NotFound: The token was not given out by this director.
        """
        conversation_id = conversation_of(self._key(), token)
        if conversation_id is None:
            raise NotFound("That is not the token of an agent of this org")
        return UUID(conversation_id)

    def _url(self) -> str | None:
        if self._server_url is None:
            kept = self._read_remembered().get("server_url")
            self._server_url = kept if isinstance(kept, str) else None
        return self._server_url

    def _key(self) -> bytes:
        """What the tokens of the org's agents are signed with. It is kept
        beside the server's conversations, never in the repository."""
        with self._lock:
            if self._tool_key is None:
                kept = self._read_remembered().get("tool_key")
                if not isinstance(kept, str):
                    kept = secrets.token_hex(32)
                    self._remember(tool_key=kept)
                self._tool_key = kept.encode()
            return self._tool_key

    def _adapter(self, conversation_id: str) -> Callable[[AgentBase], AgentBase]:
        """What is changed on an agent as its conversation starts.

        An agent that runs its own tools gets the director's through an MCP
        server and is started by the script that holds its process to its
        team's directory. Any other agent is left as it is.
        """
        launchers = self._shells()
        launcher = launchers / f"{conversation_id}{_ACP_LAUNCHER}"

        def adapt(agent: AgentBase) -> AgentBase:
            if not isinstance(agent, ACPAgent):
                return agent
            url = self._url()
            if url is None:
                raise Unavailable(
                    "The director does not know the address of its own "
                    "server, which an ACP agent needs to reach its tools"
                )
            return as_org_agent(
                agent,
                tools=tools_server(url, agent_token(self._key(), conversation_id)),
                launcher=launcher if launcher.exists() else None,
                launchers=launchers,
            )

        return adapt

    def _shells(self) -> Path:
        git_dir = git(
            self._repo_dir, ["rev-parse", "--path-format=absolute", "--git-common-dir"]
        ).strip()
        return Path(git_dir) / _SHELLS_DIR

    def _enforcement(self, kind: AgentKind) -> TeamEnforcement:
        if self._bwrap is not None:
            return "sandbox"
        return "merge" if kind == "acp" else "checks"

    def search(self, query: str, limit: int) -> list[TeamMatch]:
        """Teams whose title or directory has every word of ``query``."""
        words = query.lower().split()
        found: list[TeamMatch] = []
        if not words:
            return found
        for node in self.store.all():
            text = f"{node.title}\n{node.boundary.path}".lower()
            if all(word in text for word in words):
                found.append(
                    TeamMatch(id=node.id, title=node.title, path=node.boundary.path)
                )
                if len(found) == limit:
                    break
        return found

    async def start(self, launcher: PodLauncher) -> None:
        """Begin running agents through ``launcher``, on the running loop."""
        self._loop = asyncio.get_running_loop()
        self._launcher = launcher
        self._lanes = {}
        self._unsettled = 0
        self._quiet.set()
        self._stop_status = launcher.on_status(self._on_status)
        self._stop_guard = launcher.on_action(self.guard_action)
        self._bwrap = await asyncio.to_thread(self._find_sandbox)
        if self._bwrap is None:
            logger.info("Director cannot confine shells here; it checks file tools")
        bind_director(self)
        if self._consolidate_every > 0 and not self._fixture:
            self._consolidator = asyncio.create_task(
                self._consolidate_regularly(), name="director-consolidation"
            )
        if self._store is not None:
            self._enqueue(self._resync)

    async def aclose(self) -> None:
        """Stop running agents and save the tree if a save was pending."""
        bind_director(None)
        if self._stop_status is not None:
            self._stop_status()
            self._stop_status = None
        if self._stop_guard is not None:
            self._stop_guard()
            self._stop_guard = None
        if self._consolidator is not None:
            self._consolidator.cancel()
            await asyncio.gather(self._consolidator, return_exceptions=True)
            self._consolidator = None
        workers = list(self._workers)
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
            await self._save()
        self._launcher = None
        self._loop = None

    async def settled(self) -> None:
        """Wait until everything asked for so far has been carried out."""
        # Let what other threads asked for reach the queue first.
        await asyncio.sleep(0)
        # What is carried out may ask for more.
        while self._unsettled:
            await self._quiet.wait()
            await asyncio.sleep(0)

    async def create_org(self, request: CreateOrgRequest) -> OrgNode:
        """Start the root team of a new org and brief its IC with the task.

        Raises:
            Conflict: The director already has an org, or serves the fixture.
            Unavailable: The director is not running, or the conversation's
                workspace is not in a git repository.
            Unprocessable: The conversation cannot lead an org.
        """
        launcher = self._require_launcher()
        if self._fixture:
            raise Conflict("The director serves a fixture tree; it cannot run an org")
        store = await asyncio.to_thread(lambda: self.store)
        conversation = request.conversation
        workspace = Path(conversation.workspace.working_dir).resolve()
        if not await asyncio.to_thread(is_git_repository, workspace):
            raise Unavailable(f"An org needs a git repository; {workspace} is not one")
        repository = await asyncio.to_thread(_checkout_root, str(workspace))
        if conversation.conversation_runtime == "docker":
            raise Unprocessable(
                "An org's agents run in the server's own process, not in Docker"
            )

        ic_id = conversation.conversation_id or uuid4()
        named = provisional_summary(request.task)
        title = request.title or named.title
        additions = conversation.agent_launch_additions or AgentLaunchAdditions()
        suffix = "\n\n".join(
            part
            for part in (additions.system_message_suffix_append, POD_PROTOCOL)
            if part
        )
        async with self._create_lock:
            if store.root_id is not None:
                raise Conflict("The director already has an org")
            configured, self._repo_dir = self._repo_dir, repository
            store.upsert(
                OrgNodeInput(
                    id=ROOT_ID,
                    parent_id=None,
                    title=title,
                    boundary=Boundary(path="", writable=["**"]),
                    isolation="worktree" if conversation.worktree else "shared",
                    worktree_path=str(workspace),
                    branch="",
                    agents=_new_pod(ic_id),
                    status="idle",
                    headline=named.headline,
                    spec_summary=_summary_of(request.task),
                )
            )
            try:
                tool_params = await asyncio.to_thread(self._confine, ROOT_ID, "ic")
                started = await launcher.start_root(
                    conversation.model_copy(
                        update={
                            "conversation_id": ic_id,
                            "initial_message": None,
                            # Named like every other agent of the org.
                            "autotitle": False,
                            "agent_launch_additions": additions.model_copy(
                                update={"system_message_suffix_append": suffix}
                            ),
                        }
                    ),
                    [Tool(name=name) for name in DIRECTOR_TOOL_NAMES],
                    title=_agent_title(title, "ic"),
                    tool_params=tool_params,
                    adapt_agent=await asyncio.to_thread(self._adapter, str(ic_id)),
                )
            except BaseException:
                store.remove(ROOT_ID)
                self._repo_dir = configured
                raise
            await asyncio.to_thread(self._remember, repo_path=str(self._repo_dir))
            store.set_agent(
                ROOT_ID, "ic", kind=started.kind, acp_server=started.acp_server
            )
            root = store.update_node(
                ROOT_ID,
                worktree_path=started.workspace,
                isolation=started.isolation,
                branch=await asyncio.to_thread(_branch_of, started.workspace),
                base_commit=await asyncio.to_thread(_base_of, started.workspace),
                boundary_enforcement=self._enforcement(started.kind),
            )
        # First, so that both agents are told the name the team keeps.
        self._enqueue(
            lambda: self._summarize(
                ROOT_ID, request.task, rename=request.title is None, first=True
            ),
            ROOT_ID,
        )
        self._enqueue(lambda: self._brief(ROOT_ID, "ic", request.task), ROOT_ID)
        self._enqueue(lambda: self._launch_manager(ROOT_ID, request.task), ROOT_ID)
        return root

    async def route_message(self, request: RouteMessageRequest) -> RoutedMessage:
        """Hand what the human says to the team nearest to where they say it.

        Without an org, the message joins the one saved in the repository of
        the conversation it comes with (or above it), and failing that
        starts one there and becomes its task.

        Raises:
            Unprocessable: The path is outside the org's repository, or there
                is no org and the request does not say how to start one.
            Conflict: No team of the org can take the message.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = await asyncio.to_thread(lambda: self.store)
        async with self._first_message_lock:
            conversation = request.conversation
            if store.root_id is None and conversation is None:
                raise Unprocessable(
                    "There is no org yet. Send `conversation` with the message "
                    "so that it can start one."
                )
            if store.root_id is None and conversation is not None:
                joined = not self._fixture and await asyncio.to_thread(
                    self._join, Path(conversation.workspace.working_dir)
                )
                if not joined:
                    root = await self.create_org(
                        CreateOrgRequest(task=request.text, conversation=conversation)
                    )
                    return RoutedMessage(
                        node_id=root.id,
                        conversation_id=agent_conversation(root, "ic") or "",
                        started_org=True,
                    )
                await self._resync()
        path = await asyncio.to_thread(
            lambda: (
                self._home()
                if request.path is None
                else self._repo_relative(request.path)
            )
        )
        node = nearest_team(store.all(), path, lost=AGENT_LOST)
        conversation_id = agent_conversation(node, "ic") if node else None
        if node is None or conversation_id is None:
            raise Conflict(
                "No team of the org can take a message about "
                f"{path or 'the repository'!r}"
            )
        text = request.text
        if path != node.boundary.path:
            text = f"{text}\n\n(The human sent this from {path}.)"
        self._enqueue(lambda: self._send(conversation_id, text), node.id)
        return RoutedMessage(
            node_id=node.id, conversation_id=conversation_id, started_org=False
        )

    def refresh_headline(self, node_id: str, report: str) -> None:
        """Have a node's headline rewritten after it reported. Any thread."""
        node = self.store.get(node_id)
        if node is not None:
            task = f"{node.spec_summary}\n\nLatest report of the team: {report}"
            self._enqueue(lambda: self._summarize(node_id, task), f"headline:{node_id}")

    def guard_action(self, conversation_id: UUID, event: ActionEvent) -> str | None:
        """Why an agent of the org may not carry out an action, if it may not.

        A file tool is refused a write outside the boundary of the agent's
        team. Called on the agent's own thread, before the action runs.
        """
        store = self._store
        if store is None or event.action is None:
            return None
        found = store.find_by_conversation(str(conversation_id))
        if found is None:
            return None
        paths = written_paths(event.tool_name, event.action.model_dump())
        if not paths:
            return None
        node, role = found
        sub_teams = store.children(node.id)
        top = _checkout_root(node.worktree_path)
        for named in paths:
            target = Path(os.path.realpath(Path(node.worktree_path) / named))
            if _is_in(target, top):
                relative = target.relative_to(top).as_posix()
                refusal = write_refusal(
                    node, role, sub_teams, "" if relative == "." else relative
                )
                if refusal is not None:
                    return refusal
            elif any(_is_in(target, checkout) for checkout in self._checkouts()):
                return (
                    f"{named!r} is in a checkout of the org that is not your "
                    "team's, so it was not changed."
                )
        return None

    def answer_escalation(
        self, node_id: str, escalation_id: str, answer: str
    ) -> OrgNode:
        """Settle an escalation that waits on the human with their answer.

        The answer goes to the IC of the team that raised it.

        Raises:
            NotFound: The node or the escalation does not exist.
            Conflict: The escalation is still held by a team.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = self.store
        node = store.get(node_id)
        escalation = next(
            (e for e in node.escalations if e.id == escalation_id)
            if node
            else iter(()),
            None,
        )
        if node is None or escalation is None:
            raise NotFound(f"Node {node_id} has no escalation {escalation_id}")
        if escalation.held_by is not None:
            raise Conflict(
                f"Escalation {escalation_id} is still with team "
                f"{escalation.held_by}; it does not wait on the human"
            )
        store.settle_escalation(node_id, escalation_id)
        self.notify(
            agent_conversation(node, "ic"),
            f"The human answered your escalation {escalation.title!r}:\n{answer}",
        )
        return store.get(node_id) or node

    def notify(self, conversation_id: str | None, text: str) -> None:
        """Send an agent a message from the director. Safe from any thread."""
        if conversation_id is None:
            return
        self._enqueue(
            lambda: self._send(conversation_id, from_director(text)),
            self._lane_of(conversation_id),
        )

    def carry_out_split(self, node_id: str) -> None:
        """Start the sub-teams of a node's approved split. Safe from any thread."""
        self._enqueue(lambda: self._carry_out_split(node_id), node_id)

    def prune(
        self,
        node_id: str,
        *,
        approach: str,
        why_failed: str,
        replacement_task: str | None,
    ) -> None:
        """Remove a team and what is below it. Safe from any thread.

        Args:
            replacement_task: What a fresh team does in the same boundary;
                None to leave the boundary to the team above.
        """
        self.store.update_node(node_id, status="blocked")
        self._enqueue(
            lambda: self._prune(node_id, approach, why_failed, replacement_task),
            node_id,
        )

    def request_review(self, node_id: str) -> None:
        """Have the work a team reported done reviewed. Safe from any thread."""
        self._enqueue(lambda: self._request_review(node_id), f"review:{node_id}")

    def request_changes(self, node_id: str, findings: list[ReviewFinding]) -> bool:
        """Send a team's work back to it with what must change.

        Safe from any thread.

        Returns:
            Whether this was once too often: the team is then stopped and
            left blocked instead.

        Raises:
            NotFound: The node does not exist.
            Conflict: None of its work waits for a review.
        """
        store = self.store
        node = store.get(node_id)
        if node is None:
            raise NotFound(f"Node {node_id} does not exist")
        if node.review is None or node.review.state != "pending":
            raise Conflict(f"Team {node_id} has no work waiting for a review")
        review = node.review.model_copy(
            update={"state": "changes_requested", "findings": findings, "at": _now()}
        )
        above = store.get(node.parent_id) if node.parent_id else None
        if review.round >= MAX_REVIEW_ROUNDS:
            store.update_node(node_id, status="blocked", review=review)
            self._enqueue(lambda: self._halt(node_id), node_id)
            self._reviewed(above)
            return True
        store.update_node(node_id, status="working", review=review)
        self._reviewed(above)
        self.notify(
            agent_conversation(node, "ic"), changes_requested(above or node, findings)
        )
        return False

    def declared_status(self, node_id: str) -> None:
        """A team declared its own status, which now stands."""
        self._done_reviewers.discard(node_id)

    def _reviewed(self, reviewer: OrgNode | None) -> None:
        """Put a team that was done before it reviewed back to done, once
        none of its teams waits for it."""
        if reviewer is None or reviewer.id not in self._done_reviewers:
            return
        store = self.store
        waiting = any(
            team is not None
            and team.review is not None
            and team.review.state == "pending"
            for team in map(store.get, reviewer.child_ids)
        )
        if waiting:
            return
        self._done_reviewers.discard(reviewer.id)
        now = store.get(reviewer.id)
        if now is not None and now.status in ("working", "idle"):
            store.update_node(reviewer.id, status="done")

    def accept_review(self, node_id: str, needed_above: str) -> None:
        """Take a team's reviewed work into the team above. Safe from any
        thread.

        Args:
            needed_above: Why a team further up waits on the work; '' when
                none does.
        """
        self._enqueue(lambda: self._accept(node_id, needed_above), f"review:{node_id}")

    def _checkouts(self) -> list[Path]:
        """Where the org's checkouts are: the repository and the worktrees."""
        roots = [
            self._repo_dir,
            *([self._worktree_root] if self._worktree_root else []),
        ]
        return [Path(os.path.realpath(root)) for root in roots]

    def _home(self) -> str:
        """The directory the server was started for, relative to the org's
        repository; '' when it is not below its root."""
        home = Path(os.path.realpath(self._directory))
        top = _checkout_root(str(self._repo_dir))
        return home.relative_to(top).as_posix() if top in home.parents else ""

    def _repo_relative(self, path: str) -> str:
        """``path`` as a clean path relative to the repository.

        Raises:
            Unprocessable: It is not inside the director's directory.
        """
        if not Path(path).is_absolute():
            try:
                return clean_path(path)
            except DomainError as error:
                raise Unprocessable(str(error)) from error
        target = Path(os.path.realpath(path))
        top = _checkout_root(str(self._repo_dir))
        if not _is_in(target, top):
            raise Unprocessable(f"{path} is outside the org's repository {top}")
        relative = target.relative_to(top).as_posix()
        return "" if relative == "." else relative

    def _confine(self, node_id: str, role: AgentRole) -> ToolParams | None:
        """Write the shell an agent's terminal starts.

        Where the server can, the shell holds the agent to its team's
        boundary. An IC's shell also finds a ``git`` that keeps the agent's
        index apart from those of the agents it shares a checkout with.

        Returns:
            What to start the agent's terminal with, or None when it needs
            no shell of its own.
        """
        node = self.store.get(node_id)
        conversation_id = agent_conversation(node, role) if node else None
        if node is None or conversation_id is None:
            return None
        bash, real_git = shutil.which("bash"), shutil.which("git")
        own_git = role == "ic" and os.name != "nt" and bash and real_git
        if self._bwrap is None and not own_git:
            return None
        shells = self._shells()
        git_dir = str(shells.parent)
        sub_teams = [child.boundary.path for child in self.store.children(node_id)]
        rc: Path | None = None
        bin_dir: Path | None = None
        if own_git and real_git:
            directory = node.boundary.path
            rc = shells / f"{conversation_id}.rc"
            bin_dir = shells / f"{conversation_id}.bin"
            excludes = shells / f"{conversation_id}.exclude"
            write_shell(
                bin_dir / "git",
                git_script(
                    real_git,
                    common_dir=git_dir,
                    conversation_id=conversation_id,
                    directory=directory,
                    sub_team_directories=sub_teams,
                    excludes=excludes,
                    flock=shutil.which("flock"),
                ),
            )
            atomic_write_text(
                excludes, excludes_file(self._repo_dir, directory, sub_teams)
            )
            atomic_write_text(rc, rcfile(bin_dir))
        script = shells / f"{conversation_id}.sh"
        write_shell(
            script,
            shell_script(
                self._bwrap,
                bash or "/bin/sh",
                checkouts=self._checkouts(),
                private=shells,
                directory=node.boundary.path if role == "ic" else None,
                sub_team_directories=sub_teams,
                rcfile=rc,
            ),
        )
        # An agent that runs its own tools has no terminal of the server's;
        # its whole process is started by this instead.
        write_shell(
            shells / f"{conversation_id}{_ACP_LAUNCHER}",
            command_script(
                self._bwrap,
                checkouts=self._checkouts(),
                private=shells,
                directory=node.boundary.path if role == "ic" else None,
                sub_team_directories=sub_teams,
                first_on_path=bin_dir,
            ),
        )
        # Only the subprocess terminal starts the shell it is given.
        return {"terminal": {"terminal_type": "subprocess", "shell_path": str(script)}}

    def _confine_all(self) -> None:
        for node in self.store.all():
            for agent in node.agents:
                try:
                    self._confine(node.id, agent.role)
                except (GitError, OSError):
                    logger.exception("Could not write the shell of node %s", node.id)

    def _require_launcher(self) -> PodLauncher:
        if self._launcher is None:
            raise Unavailable("The director is not running agents on this server")
        return self._launcher

    def _load_store(self) -> OrgStore:
        store = OrgStore()
        if self._fixture:
            store.replace_all(build_fixture_nodes())
            logger.info("Director serving the fixture tree (%d nodes)", len(store))
            return store
        remembered = self._remembered_repo()
        if remembered is None or not self._load_from(remembered, store):
            self._load_from(self._directory, store)
        store.on_change(self._schedule_save)
        if self._loop is not None:
            self._enqueue(self._resync)
        return store

    def _load_from(self, directory: Path, store: OrgStore) -> bool:
        """Load the org saved in the repository of ``directory``, or failing
        that in the nearest repository above that has it as a submodule.

        Returns:
            Whether one was found. The director then works in its repository.
        """
        repository: Path | None = directory
        while repository is not None and is_git_repository(repository):
            if load_org(create_org_board(repository), store):
                self._repo_dir = _checkout_root(str(repository))
                logger.info("Director loaded %d nodes from %s", len(store), repository)
                return True
            repository = enclosing_repository(repository)
        return False

    def _join(self, workspace: Path) -> bool:
        """Take up the org saved at or above ``workspace``, if there is one."""
        if not self._load_from(workspace, self.store):
            return False
        self._remember(repo_path=str(self._repo_dir))
        return True

    def _read_remembered(self) -> dict[str, object]:
        """What the director keeps between starts of the server."""
        if self._remembered is None:
            return {}
        try:
            kept = json.loads(self._remembered.read_text())
        except (OSError, ValueError):
            return {}
        return kept if isinstance(kept, dict) else {}

    def _remembered_repo(self) -> Path | None:
        repo_path = self._read_remembered().get("repo_path")
        return Path(repo_path) if isinstance(repo_path, str) else None

    def _remember(self, **changes: object) -> None:
        """Keep ``changes`` for the next start of the server."""
        if self._remembered is None:
            return
        try:
            self._remembered.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                self._remembered,
                json.dumps({**self._read_remembered(), **changes}) + "\n",
            )
        except OSError:
            logger.exception("Could not write %s", self._remembered)

    def _enqueue(self, effect: Effect, lane: str = ORG_LANE) -> None:
        loop = self._loop
        if loop is None:
            raise Unavailable("The director is not running agents on this server")
        loop.call_soon_threadsafe(self._put, effect, lane)

    def _put(self, effect: Effect, lane: str) -> None:
        self._unsettled += 1
        self._quiet.clear()
        waiting = self._lanes.get(lane)
        if waiting is not None:
            waiting.append(effect)
            return
        self._lanes[lane] = deque([effect])
        worker = asyncio.create_task(self._work(lane), name=f"director-{lane}")
        self._workers.add(worker)
        worker.add_done_callback(self._workers.discard)

    async def _work(self, lane: str) -> None:
        waiting = self._lanes[lane]
        try:
            while waiting:
                effect = waiting.popleft()
                try:
                    await effect()
                except Exception:
                    logger.exception("A director action failed")
                finally:
                    self._unsettled -= 1
                    if not self._unsettled:
                        self._quiet.set()
        finally:
            del self._lanes[lane]

    def _lane_of(self, conversation_id: str) -> str:
        """The lane of the team an agent's conversation belongs to."""
        found = self.store.find_by_conversation(conversation_id)
        return found[0].id if found else ORG_LANE

    async def _when_there_is_room[T](
        self, node_id: str, start: Callable[[], Awaitable[T]]
    ) -> T:
        """Run ``start`` for a team; while the server is full, wait in line
        for a conversation to stop running and try again."""
        try:
            return await start()
        except ConversationRunLimitExceeded:
            logger.info("Server is at its run limit; node %s waits", node_id)
        store = self.store
        with suppress(NotFound):
            store.update_node(node_id, waiting_to_start=True)
        try:
            async with self._start_gate:
                while True:
                    self._room.clear()
                    try:
                        return await start()
                    except ConversationRunLimitExceeded:
                        pass
                    # Also after a pause: not every conversation is the org's.
                    with suppress(TimeoutError):
                        await asyncio.wait_for(
                            self._room.wait(), timeout=self._run_limit_retry
                        )
        finally:
            with suppress(NotFound):
                store.update_node(node_id, waiting_to_start=False)

    def _schedule_save(self) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._arm_save)

    def _arm_save(self) -> None:
        loop = self._loop
        if loop is None or self._save_timer is not None:
            return
        self._save_timer = loop.call_later(
            self._save_delay, lambda: self._enqueue(self._save_due)
        )

    async def _save_due(self) -> None:
        self._save_timer = None
        await self._save()

    async def _save(self) -> None:
        store = self._store
        if store is None or self._fixture:
            return
        try:
            await asyncio.to_thread(save_org, create_org_board(self._repo_dir), store)
        except GitError:
            logger.exception("Could not save the org tree to %s", self._repo_dir)

    def _on_status(
        self, conversation_id: UUID, status: ConversationExecutionStatus
    ) -> None:
        if status != ConversationExecutionStatus.RUNNING and self._loop is not None:
            self._loop.call_soon_threadsafe(self._room.set)
        store = self._store
        if store is None:
            return
        found = store.find_by_conversation(str(conversation_id))
        if found is not None:
            node, role = found
            store.set_agent(node.id, role, status=_AGENT_STATUS.get(status, "idle"))

    async def _resync(self) -> None:
        """Read every agent's status back from its conversation."""
        store = self.store
        agents: list[tuple[str, AgentRole, UUID]] = [
            (node.id, agent.role, UUID(agent.conversation_id))
            for node in store.all()
            for agent in node.agents
            if agent.conversation_id is not None
        ]
        if not agents:
            return
        # Shells that were lost with the git directory come back for the
        # agents that still start theirs.
        await asyncio.to_thread(self._confine_all)
        statuses = await self._require_launcher().statuses(
            [conversation_id for _, _, conversation_id in agents]
        )
        for node_id, role, conversation_id in agents:
            status = statuses.get(conversation_id)
            with suppress(NotFound):
                store.set_agent(
                    node_id,
                    role,
                    status=AGENT_LOST
                    if status is None
                    else _AGENT_STATUS.get(status, "idle"),
                )

    async def _send(self, conversation_id: str, text: str) -> None:
        try:
            await self._require_launcher().send(UUID(conversation_id), text)
        except NotFound:
            store = self.store
            found = store.find_by_conversation(conversation_id)
            if found is not None:
                store.set_agent(found[0].id, found[1], status=AGENT_LOST)
            logger.warning("Agent conversation %s is gone", conversation_id)

    def _briefing(self, node: OrgNode, role: AgentRole, task: str) -> str:
        blackboard = GitBlackboard(self._repo_dir)
        path = node.boundary.path
        above = self.store.get(node.parent_id) if node.parent_id else None
        specs, contracts = blackboard.applies_to(
            path, above.boundary.path if above else None
        )
        return brief(
            node,
            role,
            task,
            specs,
            contracts,
            blackboard.inherited_rules(path),
            blackboard.autopsies(path),
        )

    async def _brief(self, node_id: str, role: AgentRole, task: str) -> None:
        node = self.store.get(node_id)
        if node is None:
            return
        conversation_id = agent_conversation(node, role)
        if conversation_id is not None:
            text = await asyncio.to_thread(self._briefing, node, role, task)
            await self._send(conversation_id, text)

    async def _start_agent(
        self,
        node_id: str,
        role: AgentRole,
        parent_conversation: str | None,
        task: str,
    ) -> bool:
        """Start one agent of a node from ``parent_conversation``.

        Returns:
            Whether it runs. A failure is recorded on the node.
        """
        store = self.store
        node = store.get(node_id)
        conversation_id = agent_conversation(node, role) if node else None
        if node is None or conversation_id is None:
            return False
        launcher = self._require_launcher()
        text = await asyncio.to_thread(self._briefing, node, role, task)
        try:
            tool_params = await asyncio.to_thread(self._confine, node_id, role)
            if parent_conversation is None:
                raise NotFound(f"Node {node_id} has nobody to derive its {role} from")
            parent_id = UUID(parent_conversation)
            # A team given a profile of its own starts its IC from it; its
            # Manager, like every Manager, is derived from the IC.
            profile = node.agent_profile if role == "ic" else None
            kind = await (
                launcher.profile_kind(profile)
                if profile
                else launcher.agent_kind(parent_id)
            )
            isolation = node.isolation if role == "ic" else "shared"
            if kind == "acp" and role == "ic" and self._bwrap is None:
                # Nothing holds its process to the team's directory, so its
                # work is kept apart and checked when it is merged.
                isolation = "worktree"
            adapt = await asyncio.to_thread(self._adapter, conversation_id)
            title = _agent_title(node.title, role)

            async def start() -> StartedAgent:
                if profile:
                    return await launcher.start_from_profile(
                        profile,
                        parent_id,
                        UUID(conversation_id),
                        [Tool(name=name) for name in DIRECTOR_TOOL_NAMES],
                        title=title,
                        task=text,
                        isolation=isolation,
                        suffixes=[POD_PROTOCOL],
                        tool_params=tool_params,
                        adapt_agent=adapt,
                    )
                return await launcher.start_derived(
                    parent_id,
                    UUID(conversation_id),
                    title=title,
                    task=text,
                    isolation=isolation,
                    suffixes=[] if role == "ic" else [MANAGER_ROLE],
                    tool_params=tool_params,
                    adapt_agent=adapt,
                )

            started = await self._when_there_is_room(node_id, start)
        except Exception:
            logger.exception("Could not start the %s of node %s", role, node_id)
            store.set_agent(node_id, role, status=AGENT_LOST)
            store.update_node(node_id, status="blocked")
            return False
        store.set_agent(node_id, role, kind=started.kind, acp_server=started.acp_server)
        if role == "ic":
            store.update_node(
                node_id,
                worktree_path=started.workspace,
                isolation=started.isolation,
                branch=await asyncio.to_thread(_branch_of, started.workspace),
                base_commit=await asyncio.to_thread(_base_of, started.workspace),
                boundary_enforcement=self._enforcement(started.kind),
            )
        return True

    async def _launch_manager(self, node_id: str, task: str) -> None:
        node = self.store.get(node_id)
        if node is not None:
            await self._start_agent(
                node_id, "manager", agent_conversation(node, "ic"), task
            )

    async def _carry_out_split(self, node_id: str) -> None:
        store = self.store
        node = store.get(node_id)
        proposal = node.pending_split if node else None
        if node is None or proposal is None or proposal.status != "approved":
            return
        parent_ic = agent_conversation(node, "ic")
        try:
            # The org may have grown since the split was proposed.
            check_limits(store, node, len(proposal.children), self.settings)
        except OrgStoreError as error:
            store.update_node(
                node_id,
                pending_split=proposal.model_copy(
                    update={"status": "refine_requested", "feedback": str(error)}
                ),
            )
            for conversation_id in (parent_ic, agent_conversation(node, "manager")):
                self.notify(
                    conversation_id,
                    f"The approved split cannot be carried out: {error}",
                )
            return
        if proposal.contracts.strip():
            await asyncio.to_thread(self._record_contracts, node, proposal.contracts)

        started: list[str] = []
        teams: list[tuple[str, str]] = []
        for child in proposal.children:
            child_node = store.upsert(self._child_node(store, node, child))
            started.append(f"{child.title} ({child.path})")
            teams.append((child_node.id, child.task))
        # The sub-teams start side by side, each in its own lane; the split
        # is over when the last of them has.
        left = len(teams)

        async def start(child_id: str, task: str) -> None:
            nonlocal left
            try:
                if await self._start_agent(child_id, "ic", parent_ic, task):
                    await self._launch_manager(child_id, task)
            finally:
                left -= 1
                if not left:
                    self._enqueue(lambda: self._finish_split(node_id, started), node_id)
            await self._summarize(child_id, task, first=True)

        for child_id, task in teams:
            self._enqueue(
                lambda child_id=child_id, task=task: start(child_id, task), child_id
            )

    async def _finish_split(self, node_id: str, started: list[str]) -> None:
        store = self.store
        node = store.get(node_id)
        if node is None:
            return
        store.update_node(node_id, pending_split=None)
        # The shell the IC starts next can no longer write what it gave away.
        with suppress(GitError, OSError):
            await asyncio.to_thread(self._confine, node_id, "ic")
        self.notify(
            agent_conversation(node, "ic"),
            "Your split was approved and these sub-teams now own their "
            f"directories: {', '.join(started)}. You can no longer write "
            "there; carry on with what remains in your boundary.",
        )

    async def _prune(
        self,
        node_id: str,
        approach: str,
        why_failed: str,
        replacement_task: str | None,
    ) -> None:
        store = self.store
        node = store.get(node_id)
        parent = store.get(node.parent_id) if node and node.parent_id else None
        if node is None or parent is None:
            return
        launcher = self._require_launcher()
        # The team first, then everything below it.
        teams = [node]
        for team in teams:
            teams.extend(store.children(team.id))
        conversations = [
            UUID(agent.conversation_id)
            for team in teams
            for agent in team.agents
            if agent.conversation_id is not None
        ]
        for conversation_id in conversations:
            await launcher.stop(conversation_id)
        now = datetime.now(UTC)
        saved = await asyncio.to_thread(
            self._discard_work, teams, now.strftime("%Y%m%dT%H%M%SZ")
        )
        for conversation_id in conversations:
            await launcher.delete(conversation_id)
        autopsy = Autopsy(
            node_id=node_id,
            at=now.isoformat(timespec="seconds"),
            approach=approach,
            why_failed=why_failed,
            saved_ref=saved,
        )
        try:
            await asyncio.to_thread(
                GitBlackboard(self._repo_dir).append_autopsy,
                node.boundary.path,
                autopsy,
            )
        except (GitError, DomainError):
            logger.exception("Could not record the autopsy of node %s", node_id)

        parent_ic = agent_conversation(parent, "ic")
        where = node.boundary.path
        if replacement_task is None:
            store.remove(node_id)
            # The shell the IC above starts next can write there again.
            with suppress(GitError, OSError):
                await asyncio.to_thread(self._confine, parent.id, "ic")
            outcome = f"{where} is yours again"
        else:
            for child in store.children(node_id):
                store.remove(child.id)
            store.update_node(
                node_id,
                agents=_new_pod(uuid4()),
                status="idle",
                worktree_path=parent.worktree_path,
                branch=parent.branch,
                escalations=[],
                pending_split=None,
                review=None,
                headline=provisional_summary(replacement_task).headline,
                spec_summary=_summary_of(replacement_task),
            )
            if await self._start_agent(node_id, "ic", parent_ic, replacement_task):
                await self._launch_manager(node_id, replacement_task)
            self._enqueue(
                lambda: self._summarize(node_id, replacement_task, first=True),
                node_id,
            )
            outcome = f"a fresh team now works in {where}"
        self.notify(
            parent_ic,
            f"Your Manager pruned the team {node.title!r} ({where}): "
            f"{why_failed} Its uncommitted changes were discarded"
            + (f" (kept in git as {saved})" if saved else "")
            + f"; {outcome}.",
        )

    async def _own_account(
        self, node: OrgNode
    ) -> tuple[str, list[ReviewFinding] | None]:
        """What a team's IC says of the work its team reported done.

        Returns:
            Its description of the changes, '' when it gave none, and what
            it found reviewing them, None when it gave no usable answer or
            was not asked.
        """
        ic = agent_conversation(node, "ic")
        if ic is None or not node.base_commit:
            return "", None
        files = await asyncio.to_thread(
            bugbot.changed_files,
            Path(node.worktree_path),
            node.base_commit,
            node.boundary.path,
        )
        if files is None:
            return "", None
        if not files:
            return "", []
        description = await self._ask_agent(ic, bugbot.write_up_question(files))
        instructions = await asyncio.to_thread(self._find_review_instructions)
        if instructions is None:
            return description or "", None
        above = self.store.get(node.parent_id) if node.parent_id else None
        specs, contracts = await asyncio.to_thread(
            GitBlackboard(self._repo_dir).applies_to,
            node.boundary.path,
            above.boundary.path if above else None,
        )
        answer = await self._ask_agent(
            ic,
            bugbot.review_question(
                instructions,
                specs[0].spec if specs else None,
                [agreed.contracts for agreed in contracts],
                files,
            ),
        )
        return description or "", bugbot.parse_findings(answer) if answer else None

    async def _ask_agent(self, conversation_id: str, question: str) -> str | None:
        try:
            return await asyncio.wait_for(
                self._require_launcher().ask_agent(UUID(conversation_id), question),
                timeout=self._model_call_timeout,
            )
        except (NotFound, TimeoutError):
            return None

    async def _ask(
        self, conversation_id: str, system: str, user: str, usage_id: str
    ) -> str | None:
        try:
            return await asyncio.wait_for(
                self._require_launcher().ask(
                    UUID(conversation_id), system, user, usage_id=usage_id
                ),
                timeout=self._model_call_timeout,
            )
        except (NotFound, TimeoutError):
            return None

    async def _request_review(self, node_id: str) -> None:
        store = self.store
        node = store.get(node_id)
        asked = node.review if node else None
        if node is None or asked is None:
            return
        if asked.state not in ("pending", "unreviewed"):
            return
        description, found = await self._own_account(node)
        node = store.get(node_id)
        if node is None or node.review != asked:
            return
        review = asked.model_copy(
            update={
                "description": description,
                "automated_ran": found is not None,
                "automated": found or [],
            }
        )
        node = store.update_node(node_id, review=review)
        above = store.get(node.parent_id) if node.parent_id else None
        if above is not None:
            if above.status == "done":
                # Reviewing is work; it is done again once it has decided.
                self._done_reviewers.add(above.id)
                store.update_node(above.id, status="working")
            self.notify(agent_conversation(above, "ic"), review_request(node, review))
        elif found:
            self.notify(
                agent_conversation(node, "ic"),
                "Asked to review your team's work by the code review "
                "instructions, you found, for you to verify:\n"
                + "\n".join(f"- {finding.issue}" for finding in found),
            )

    async def _accept(self, node_id: str, needed_above: str) -> None:
        store = self.store
        node = store.get(node_id)
        above = store.get(node.parent_id) if node and node.parent_id else None
        if node is None or above is None:
            return
        if node.review is None or node.review.state != "pending":
            return
        reviewer = agent_conversation(above, "ic")
        where = node.boundary.path or "the repository"
        team, checkout = Path(node.worktree_path), Path(above.worktree_path)
        merged = False
        if _checkout_root(str(team)) != _checkout_root(str(checkout)):
            try:
                result = await asyncio.to_thread(
                    merge_back, checkout, team, node.boundary.path, node.title
                )
            except (GitError, OSError) as error:
                self.notify(
                    reviewer,
                    f"The work of {node.title!r} could not be merged into your "
                    f"checkout, and still waits for your review: {error}",
                )
                return
            if result.outcome == "refused":
                self.notify(
                    reviewer,
                    f"git would not merge the work of {node.title!r} into your "
                    f"checkout as it is:\n{result.detail}\nCommit or put aside "
                    "what is in the way, then accept it again with `review_team`.",
                )
                return
            if result.outcome in ("conflicts", "outside"):
                issue = (
                    "These files do not merge with the checkout of the team "
                    "above. Merge its branch into yours, resolve them and "
                    "commit."
                    if result.outcome == "conflicts"
                    else f"These files are outside {where}, which is all your "
                    "team owns. Take them out of your commits."
                )
                halted = self.request_changes(
                    node_id, [ReviewFinding(path=", ".join(result.files), issue=issue)]
                )
                self.notify(
                    reviewer,
                    f"The work of {node.title!r} was not taken into your "
                    f"checkout: {issue} ({', '.join(result.files)}) "
                    + (
                        "It was sent back for the last time; the team is "
                        "stopped and blocked."
                        if halted
                        else "It was sent back to the team."
                    ),
                )
                return
            merged = result.outcome == "merged"
        tip = await asyncio.to_thread(head_of, team)
        store.update_node(
            node_id,
            review=node.review.model_copy(update={"state": "accepted", "at": _now()}),
            base_commit=tip or node.base_commit,
        )
        self._reviewed(above)
        self.notify(
            agent_conversation(node, "ic"),
            f"The IC of {above.title!r} accepted your team's work"
            + (" and it is merged into that team's checkout." if merged else "."),
        )
        if merged:
            self.notify(
                reviewer, f"The work of {node.title!r} is merged into your checkout."
            )
        further = store.get(above.parent_id) if above.parent_id else None
        if needed_above and further is not None:
            self.notify(
                agent_conversation(further, "ic"),
                f"Team {above.title!r} (node {above.id}) below you accepted the "
                f"work of its sub-team {node.title!r} in {where} and says a team "
                f"above it waits on that work: {needed_above}\nIt is in "
                f"{above.worktree_path}"
                + (f" on branch {above.branch}." if above.branch else "."),
            )

    async def _halt(self, node_id: str) -> None:
        """Stop the agents of a team where they are."""
        node = self.store.get(node_id)
        if node is None:
            return
        launcher = self._require_launcher()
        for agent in node.agents:
            if agent.conversation_id is not None:
                await launcher.stop(UUID(agent.conversation_id))

    async def _consolidate_regularly(self) -> None:
        while True:
            await asyncio.sleep(self._consolidate_every)
            # One pass at a time.
            done = asyncio.Event()

            async def consolidate(done: asyncio.Event = done) -> None:
                try:
                    await self._consolidate()
                finally:
                    done.set()

            self._enqueue(consolidate, MODELS_LANE)
            await done.wait()

    async def _consolidate(self) -> None:
        """Have a model rewrite the rules of every boundary that has gained
        enough of them."""
        store = self._store
        root = store.get(store.root_id) if store and store.root_id else None
        asker = agent_conversation(root, "ic") if root else None
        if asker is None or not is_git_repository(self._repo_dir):
            return
        board = GitBlackboard(self._repo_dir)
        for item in await asyncio.to_thread(memory.due, board):
            answer = await self._ask(asker, *memory.prompt(item), memory.USAGE_ID)
            rules = memory.parse_rules(answer, item) if answer else None
            if rules is None:
                logger.warning(
                    "No usable consolidation of the rules of %r", item.directory
                )
                continue
            await asyncio.to_thread(memory.apply, board, item, rules)

    def _discard_work(self, teams: list[OrgNode], stamp: str) -> str | None:
        """Discard the uncommitted work of ``teams``, the pruned team first.

        Returns:
            The ref that keeps what the pruned team itself had uncommitted.
        """
        store = self.store
        kept: str | None = None
        for index, team in enumerate(teams):
            self._forget_shells(team)
            above = store.get(team.parent_id) if team.parent_id else None
            # A team in the checkout of the one above is covered by it.
            if index and above and above.worktree_path == team.worktree_path:
                continue
            ref = f"{PRUNED_REFS}/{team.id}/{stamp}"
            try:
                saved = save_and_discard(
                    Path(team.worktree_path), team.boundary.path, ref
                )
            except (GitError, OSError):
                logger.exception("Could not discard the work of node %s", team.id)
                continue
            if index == 0 and saved is not None:
                kept = ref
        return kept

    def _forget_shells(self, team: OrgNode) -> None:
        """Remove what the shells of a team's agents were made of."""
        try:
            common, own = (
                Path(
                    git(
                        directory, ["rev-parse", "--path-format=absolute", kind]
                    ).strip()
                )
                for directory, kind in (
                    (self._repo_dir, "--git-common-dir"),
                    (Path(team.worktree_path), "--git-dir"),
                )
            )
        except (GitError, OSError):
            return
        for agent in team.agents:
            name = agent.conversation_id
            if name is None:
                continue
            for left in (
                *(common / _SHELLS_DIR).glob(f"{name}.*"),
                *own.glob(f"agent-index-{name}*"),
            ):
                if left.is_dir():
                    shutil.rmtree(left, ignore_errors=True)
                else:
                    left.unlink(missing_ok=True)

    async def _summarize(
        self, node_id: str, task: str, *, rename: bool = False, first: bool = False
    ) -> None:
        """Have a model write a node's headline and, with ``rename``, its title.

        Args:
            first: This is the summary of the task the team was made for. It
                is dropped once the team has reported, which may happen
                before the model was asked.
        """
        store = self.store

        def reported() -> bool:
            current = store.get(node_id)
            return (
                first
                and current is not None
                and current.headline != provisional_summary(task).headline
            )

        node = store.get(node_id)
        conversation_id = agent_conversation(node, "ic") if node else None
        if node is None or conversation_id is None or reported():
            return
        launcher = self._require_launcher()
        try:
            async with self._model_calls:
                summary = await asyncio.wait_for(
                    launcher.summarize(UUID(conversation_id), task),
                    timeout=self._summary_timeout,
                )
        except (NotFound, TimeoutError):
            return
        if summary is None or reported():
            return
        changes: dict[str, object] = {"headline": summary.headline}
        if rename:
            changes["title"] = summary.title
        try:
            node = store.update_node(node_id, **changes)
        except NotFound:
            return
        if not rename:
            return
        for agent in node.agents:
            if agent.conversation_id is not None:
                with suppress(NotFound):
                    await launcher.rename(
                        UUID(agent.conversation_id),
                        _agent_title(summary.title, agent.role),
                    )

    def _record_contracts(self, node: OrgNode, contracts: str) -> None:
        """Add the contracts of a split to those of the node's boundary."""
        blackboard = GitBlackboard(self._repo_dir)
        path = node.boundary.path
        earlier = blackboard.read(
            f"{path}/{CONTRACTS_FILE}" if path else CONTRACTS_FILE
        )
        text = f"{contracts.strip()}\n"
        blackboard.write(
            path,
            CONTRACTS_FILE,
            f"{earlier.rstrip()}\n\n{text}" if earlier and earlier.strip() else text,
        )

    @staticmethod
    def _child_node(
        store: OrgStore, parent: OrgNode, child: SplitChild
    ) -> OrgNodeInput:
        base = ".".join(
            re.sub(r"[^A-Za-z0-9_-]+", "-", segment)
            for segment in child.path.split("/")
        )
        node_id, attempt = base, 1
        while store.get(node_id) is not None:
            attempt += 1
            node_id = f"{base}-{attempt}"
        return OrgNodeInput(
            id=node_id,
            parent_id=parent.id,
            title=child.title,
            boundary=child_boundary(child),
            isolation=child.isolation,
            agent_profile=child.agent_profile,
            worktree_path=parent.worktree_path,
            branch=parent.branch,
            agents=_new_pod(uuid4()),
            status="idle",
            headline=provisional_summary(child.task).headline,
            spec_summary=_summary_of(child.task),
        )


def _new_pod(ic_id: UUID) -> list[AgentRef]:
    return [
        AgentRef(role="ic", conversation_id=str(ic_id), status="idle"),
        AgentRef(role="manager", conversation_id=str(uuid4()), status="idle"),
    ]


def _base_of(workspace: str) -> str:
    return head_of(Path(workspace)) or ""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _branch_of(workspace: str) -> str:
    try:
        return git(Path(workspace), ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    except (GitError, OSError):
        return ""
