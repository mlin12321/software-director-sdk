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
from collections.abc import Awaitable, Callable, Collection
from contextlib import suppress
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from openhands.agent_server.config import Config
from openhands.agent_server.conversation_service import ToolParams
from openhands.agent_server.director import bugbot, memory, org_sync
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
from openhands.agent_server.director.changes import require_editable, write_by_hand
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
    BlackboardSync,
    Boundary,
    CreateOrgRequest,
    DirectorSettings,
    DirectorSettingsState,
    DirectorSettingsUpdate,
    FileEdit,
    OrgConflict,
    OrgMerge,
    OrgNode,
    OrgNodeInput,
    OrgStateEvent,
    OrgSummary,
    OrgSync,
    PauseReason,
    Recovery,
    RecoveryItem,
    ResolveMergeRequest,
    ReviewFinding,
    RoutedMessage,
    RouteMessageRequest,
    Side,
    SplitChild,
    StartAgainRequest,
    SyncResult,
    TeamEnforcement,
    TeamMatch,
)
from openhands.agent_server.director.org_merge import (
    Nodes,
    by_id,
    merge_orgs,
    rehomed,
)
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    load_org,
    org_file,
    save_org,
    saved_nodes,
    stored_nodes,
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
from openhands.agent_server.director.recovery import (
    STARTING,
    describe,
    fingerprint,
    has_work,
    put_aside_stale,
    restore_worktrees,
)
from openhands.agent_server.director.run_state import (
    REVIEWING,
    RUNNING,
    TEAM,
    WAITING,
    Intent,
    RunState,
    RunStateError,
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
# An agent that was running when its org was paused, and runs again when it
# is resumed.
AGENT_PAUSED = "paused"
# What an agent is told when the server ended in the middle of its step.
RESTART_NOTE = (
    "The server was restarted while you were working. The command or edit "
    "you were in the middle of may not have finished: check what is there "
    "before you carry on with your task."
)
# What a team's IC is told when a pull found other commits for its branch.
OTHER_WORK_NOTE = (
    "A pull brought other commits for your team's branch {branch} from "
    "{remote}. They are at {ref} and are not in your checkout. Merge them "
    "into your branch (`git merge {ref}`), settle what clashes, and carry on."
)
# How often a merge is made again because the org was saved meanwhile.
_MERGE_ATTEMPTS = 4
# Where, in the repository's git directory, the director keeps what it is in
# the middle of.
_RUN_STATE = "director/state.db"
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
# The lane agents are paused in, so that nothing waits for their last step.
PAUSE_LANE = "pause"
# The steps the director takes for a team's review, in its own lane.
_REVIEW_STEPS = ("request_review", "accept")
# How many of the director's own model calls run at once.
MODEL_CALLS_AT_ONCE = 2

_AGENT_STATUS: dict[ConversationExecutionStatus, str] = {
    ConversationExecutionStatus.RUNNING: "working",
    ConversationExecutionStatus.ERROR: "blocked",
    ConversationExecutionStatus.STUCK: "blocked",
}

type Effect = Callable[[], Awaitable[None]]


class _Held(Exception):
    """The org was paused before a step could be finished."""


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
        self._run: RunState | None = None
        self._run_of: Path | None = None
        self._paused: PauseReason | None = None
        self._state_listeners: list[Callable[[OrgStateEvent], None]] = []
        # The intents and messages being carried out right now, so that
        # resuming does not carry one out a second time.
        self._in_flight: set[int] = set()
        self._delivering: set[int] = set()
        # The conversations last recorded as running.
        self._running: set[str] = set()
        # The conversations a resume is about to run again.
        self._resuming: set[str] = set()
        self._save_asked = False
        # Teams that waited for a run when the org was paused, in the order
        # they waited: each tries to start only when it is the first.
        self._line: deque[str] = deque()
        self._line_moved = asyncio.Event()
        self._steps: dict[str, Callable[[Intent], Awaitable[None]]] = {
            "brief": self._do_brief,
            "launch_manager": self._do_launch_manager,
            "split": self._carry_out_split,
            "start_team": self._do_start_team,
            "finish_split": self._do_finish_split,
            "request_review": self._do_request_review,
            "accept": self._do_accept,
            "prune": self._prune,
            "halt": self._do_halt,
        }
        self._consolidate_every = consolidate_every
        self._consolidator: asyncio.Task[None] | None = None
        self._bwrap: str | None = None
        self._stop_guard: Callable[[], None] | None = None
        self._save_delay = save_delay
        self._run_limit_retry = run_limit_retry
        self._lock = threading.Lock()
        self._remember_lock = threading.Lock()
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
        # Held by a pull, a push or a step of a merge, one at a time.
        self._sync_lock = asyncio.Lock()

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

    def state(self) -> OrgStateEvent:
        """Whether the org is paused or in the middle of a merge."""
        run = self._runs()
        return OrgStateEvent(paused=self._paused, merging=run.get("merge") is not None)

    def watch_state(
        self, listener: Callable[[OrgStateEvent], None]
    ) -> Callable[[], None]:
        """Call ``listener`` whenever the org is paused, resumed, or begins
        or ends a merge. It may be called from any thread.

        Returns:
            A function that stops the calls.
        """
        with self._lock:
            self._state_listeners.append(listener)

        def unwatch() -> None:
            with self._lock:
                if listener in self._state_listeners:
                    self._state_listeners.remove(listener)

        return unwatch

    def _announce_state(self) -> None:
        with self._lock:
            listeners = list(self._state_listeners)
        if not listeners:
            return
        state = self.state()
        for listener in listeners:
            listener(state)

    def _set_merge(self, state: dict[str, Any] | None) -> None:
        self._runs().put("merge", json.dumps(state) if state is not None else None)
        self._announce_state()

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
            paused=self._paused if self._runs() else None,
            merging=self._runs().get("merge") is not None,
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
            self._enqueue(self._came_back)
        elif await asyncio.to_thread(self._remembered_repo) is not None:
            # An org that was left in the middle of its work is looked at
            # now, not when somebody first asks for it.
            await asyncio.to_thread(lambda: self.store)

    async def aclose(self) -> None:
        """Stop running agents and save the tree if a save was pending.

        An org that is in the middle of its work is left paused, so that the
        next start finds what to take up again.
        """
        bind_director(None)
        store = self._store
        if store is not None and store.root_id is not None and not self._fixture:
            run = self._runs()
            if self._paused is None and has_work(run):
                self._hold("shutdown")
            run.put("clean", "1")
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
        with self._lock:
            if self._run is not None:
                self._run.close()
                self._run = None

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
            await asyncio.to_thread(self._working)
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
        self._begin("brief", ROOT_ID, role="ic", task=request.task)
        self._begin("launch_manager", ROOT_ID, task=request.task)
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
                await self._came_back()
        path = await asyncio.to_thread(
            lambda: (
                self._home()
                if request.path is None
                else self._repo_relative(request.path)
            )
        )
        node = nearest_team(store.all(), path, lost=AGENT_LOST)
        if node is None and request.conversation is not None:
            # Nobody here can take it: the nearest team whose agents have no
            # conversation on this server is started again to take it.
            gone = nearest_team(store.all(), path, lost="")
            if gone is not None and gone.status != "done":
                node = await self.start_again(
                    gone.id, StartAgainRequest(conversation=request.conversation)
                )
        conversation_id = agent_conversation(node, "ic") if node else None
        if node is None or conversation_id is None:
            raise Conflict(
                "No team of the org can take a message about "
                f"{path or 'the repository'!r}"
            )
        text = request.text
        if path != node.boundary.path:
            text = f"{text}\n\n(The human sent this from {path}.)"
        self._post(conversation_id, text)
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
        if conversation_id is not None:
            self._post(conversation_id, from_director(text))

    def carry_out_split(self, node_id: str) -> None:
        """Start the sub-teams of a node's approved split. Safe from any thread."""
        self._begin("split", node_id)

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
        self._begin(
            "prune",
            node_id,
            approach=approach,
            why_failed=why_failed,
            replacement_task=replacement_task,
        )

    def request_review(self, node_id: str) -> None:
        """Have the work a team reported done reviewed. Safe from any thread."""
        self._begin("request_review", node_id)

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
            self._begin("halt", node_id)
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
        self._runs().unmark(REVIEWING, node_id)

    def _reviewed(self, reviewer: OrgNode | None) -> None:
        """Put a team that was done before it reviewed back to done, once
        none of its teams waits for it."""
        run = self._runs()
        if reviewer is None or reviewer.id not in run.marked(REVIEWING):
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
        run.unmark(REVIEWING, reviewer.id)
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
        self._begin("accept", node_id, needed_above=needed_above)

    async def pause(self) -> Recovery:
        """Hold the org: its agents stop after the step they are in, and
        the director's own steps and messages wait.

        What agents still report is recorded and acted on after
        :meth:`resume`.

        Raises:
            Conflict: There is no org, or it is the fixture.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = await asyncio.to_thread(lambda: self.store)
        if store.root_id is None or self._fixture:
            raise Conflict("There is no org to pause")
        if self._paused is None:
            self._hold("human")
            self._enqueue(self._pause_agents, PAUSE_LANE)
        return await self.recovery()

    async def resume(self) -> Recovery:
        """Take a paused org up again.

        Run state that no longer fits the org is put aside first. Then the
        teams that waited to start are started in the order they waited,
        the steps the director had begun are finished, held messages are
        delivered and the agents that were running run again.

        Returns:
            What was put aside, and nothing else.

        Raises:
            Unavailable: The director is not running.
        """
        launcher = self._require_launcher()
        store = await asyncio.to_thread(lambda: self.store)
        if self._paused is None:
            return await self.recovery()
        run = self._runs()
        await asyncio.to_thread(self._put_aside_stale)
        await asyncio.to_thread(
            restore_worktrees, self._repo_dir, store.all(), self._home()
        )
        dropped = self._dropped()
        was_running = run.marked(RUNNING)
        waited = list(run.marked(WAITING))
        statuses = await launcher.statuses([UUID(agent) for agent in was_running])
        again: dict[str, str | None] = {}
        for conversation_id in was_running:
            status = statuses.get(UUID(conversation_id))
            if status == ConversationExecutionStatus.ERROR:
                again[conversation_id] = RESTART_NOTE
            elif status == ConversationExecutionStatus.PAUSED:
                again[conversation_id] = None
        self._resuming = set(again)
        self._release()
        await self._resync()
        for node_id in waited:
            run.unmark(WAITING, node_id)
        intents = run.open_intents()
        starting = {intent.node_id for intent in intents}
        self._line = deque(node_id for node_id in waited if node_id in starting)
        for intent in intents:
            self._enqueue(
                lambda intent=intent: self._carry_out(intent.id), _lane_for(intent)
            )
        for held in run.held():
            self._enqueue(
                lambda held=held: self._deliver(
                    held.seq, held.conversation_id, held.text
                ),
                held.node_id or ORG_LANE,
            )
        for conversation_id, note in again.items():
            self._enqueue(
                lambda conversation_id=conversation_id, note=note: self._run_again(
                    conversation_id, note
                ),
                was_running[conversation_id],
            )
        return Recovery(paused=None, items=dropped, messages_held=0)

    async def recovery(self) -> Recovery:
        """What resuming the org does, and what was put aside as stale."""
        store = await asyncio.to_thread(lambda: self.store)
        run = self._runs()
        if self._paused is None:
            return Recovery(paused=None, items=[], messages_held=0)
        launcher = self._launcher
        statuses = (
            await launcher.statuses(
                [
                    UUID(agent.conversation_id)
                    for node in store.all()
                    for agent in node.agents
                    if agent.conversation_id is not None
                ]
            )
            if launcher is not None
            else {}
        )
        items = await asyncio.to_thread(describe, run, store, statuses, self._repo_dir)
        return Recovery(
            paused=self._paused,
            items=[*items, *self._dropped()],
            messages_held=len(run.held()),
        )

    async def pull_org(self, remote: str) -> OrgSync:
        """Take in the org a remote of the repository has, as a merge.

        The org is paused while it is merged and, unless the merge stops on
        conflicts, resumed if it was running. What only the remote changed
        is taken; what both changed in the same place is left as a conflict
        and nothing is written until each has a choice (:meth:`resolve_merge`)
        or the merge is abandoned (:meth:`abort_merge`).

        Raises:
            Conflict: A merge waits for choices, or this is the fixture.
            NotFound: The repository has no such remote.
            UpstreamFailed: The remote could not be reached.
            Unavailable: The director is not running, or has no repository.
        """
        self._require_launcher()
        if self._fixture:
            raise Conflict("The director serves a fixture tree; it has no remote")
        store = await asyncio.to_thread(lambda: self.store)
        async with self._sync_lock:
            if self._merging() is not None:
                raise Conflict(
                    "A merge waits for choices. Resolve it or abandon it first."
                )
            repo = await asyncio.to_thread(self._repository)
            await asyncio.to_thread(org_sync.fetch, repo, remote)
            theirs = await asyncio.to_thread(
                org_sync.commit_at, repo, org_sync.remote_org(remote)
            )
            if store.root_id is None:
                return await self._take_org(repo, remote, theirs)
            resume = self._paused is None
            if resume:
                await self.pause()
                await self._after(PAUSE_LANE)
            return await self._merge(
                {"remote": remote, "theirs": theirs, "choices": {}, "resume": resume}
            )

    async def push_org(self, remote: str) -> OrgSync:
        """Give a remote of the repository the org, the blackboard and the
        branches of the teams' work.

        Raises:
            Conflict: There is no org, a merge waits for choices, or the
                remote has changes that were not pulled.
            NotFound: The repository has no such remote.
            UpstreamFailed: The remote could not be reached or refused.
        """
        store = await asyncio.to_thread(lambda: self.store)
        if store.root_id is None or self._fixture:
            raise Conflict("There is no org to push")
        async with self._sync_lock:
            if self._merging() is not None:
                raise Conflict(
                    "A merge waits for choices. Resolve it or abandon it first."
                )
            repo = self._repo_dir

            def push() -> OrgSync:
                self._save_tree(store)
                org_sync.fetch(repo, remote)
                if org_sync.behind(repo, remote):
                    raise Conflict(
                        f"{remote} has changes of the org that are not here. "
                        "Pull first."
                    )
                branches = org_sync.push(repo, remote, store.all())
                return OrgSync(
                    remote=remote,
                    result="pushed",
                    head=create_org_board(repo).head(),
                    branches=branches,
                )

            return await asyncio.to_thread(push)

    async def merge(self) -> OrgMerge | None:
        """The merge that waits for choices, if there is one."""
        state = self._merging()
        if state is None:
            return None
        return (await asyncio.to_thread(self._weigh, state)).merge

    async def resolve_merge(self, request: ResolveMergeRequest) -> OrgSync:
        """Choose a side for conflicts of the merge that waits. When every
        conflict has a choice the merge is finished.

        Raises:
            Conflict: No merge waits for choices.
            Unprocessable: A choice names no conflict of the merge.
        """
        self._require_launcher()
        async with self._sync_lock:
            state = self._merging()
            if state is None:
                raise Conflict("No merge waits for choices")
            open_ = (await asyncio.to_thread(self._weigh, state)).merge
            known = {conflict.id for conflict in open_.conflicts} if open_ else set()
            unknown = sorted(set(request.choices) - known)
            if unknown:
                raise Unprocessable(
                    f"The merge has no conflict named {', '.join(unknown)}"
                )
            choices: dict[str, str] = {**state["choices"], **request.choices}
            if request.rest is not None:
                choices = {each: request.rest for each in known} | choices
            return await self._merge({**state, "choices": choices})

    async def abort_merge(self) -> OrgSync:
        """Abandon the merge that waits; the org stays as it was.

        Raises:
            Conflict: No merge waits for choices.
        """
        async with self._sync_lock:
            state = self._merging()
            if state is None:
                raise Conflict("No merge waits for choices")
            self._set_merge(None)
            if state["resume"]:
                await self.resume()
            return OrgSync(
                remote=state["remote"],
                result="up_to_date",
                head=await asyncio.to_thread(create_org_board(self._repo_dir).head),
            )

    async def edit_file(self, node_id: str, path: str, content: str) -> FileEdit:
        """Write one file of a team by hand, commit it and tell its IC.

        Raises:
            NotFound: There is no such team, or its checkout is not here.
            Conflict: An agent of the team is in the middle of a turn.
            Forbidden: The team's agents may not write the file either.
            Invalid: The path does not name a file that can be committed.
        """
        store = await asyncio.to_thread(lambda: self.store)
        node = store.get(node_id)
        if node is None:
            raise NotFound(f"Unknown node: {node_id}")
        sub_teams = store.children(node_id)
        require_editable(node, sub_teams, path)
        agents = [
            UUID(agent.conversation_id)
            for agent in node.agents
            if agent.conversation_id is not None
        ]
        if agents and self._launcher is not None:
            statuses = await self._launcher.statuses(agents)
            if ConversationExecutionStatus.RUNNING in statuses.values():
                raise Conflict(
                    f"An agent of team {node.title!r} is working. Pause the "
                    "org or wait for it to finish, then save again."
                )
        edit = await asyncio.to_thread(write_by_hand, node, sub_teams, path, content)
        ic = agent_conversation(node, "ic")
        if edit.commit is not None and ic is not None and self._loop is not None:
            self.notify(
                ic,
                f"The human changed {edit.path} in your team's checkout by hand "
                f"and committed it as {edit.commit[:12]}. Read the file again "
                "before you change it, and keep what they wrote unless they "
                "tell you otherwise.",
            )
        return edit

    async def start_again(self, node_id: str, request: StartAgainRequest) -> OrgNode:
        """Start afresh a team whose agents have no conversation on this
        server, on the work its branch has.

        The team's IC starts in the team's own checkout and is briefed from
        the team's spec and last headline; its Manager is derived from it.
        The IC starts as the given conversation, or without one like the IC
        of the team above (from the team's own agent profile if it has one).

        Raises:
            NotFound: There is no such team.
            Conflict: The team is done, its IC has a conversation here, or
                no conversation was given and the team above has no IC here
                to start it like.
            Unavailable: The director is not running.
        """
        launcher = self._require_launcher()
        store = await asyncio.to_thread(lambda: self.store)
        node = store.get(node_id)
        if node is None:
            raise NotFound(f"There is no team {node_id}")
        if node.status == "done":
            raise Conflict(f"Team {node.title} is done")
        ic = agent_conversation(node, "ic")
        if ic is not None and await launcher.started(UUID(ic)) is not None:
            raise Conflict(f"The IC of team {node.title} has a conversation here")
        conversation = request.conversation
        if conversation is None:
            above = store.get(node.parent_id) if node.parent_id else None
            if above is None:
                raise Conflict(
                    f"Team {node.title} is the top of the org: it is started "
                    "again by a conversation started in its folder"
                )
            above_ic = agent_conversation(above, "ic")
            if above_ic is None or await launcher.started(UUID(above_ic)) is None:
                raise Conflict(
                    f"Team {above.title} above it has no IC here. Start that "
                    "team again first."
                )
        await asyncio.to_thread(restore_worktrees, self._repo_dir, [node], self._home())
        ic_id = uuid4()
        task = await asyncio.to_thread(self._task_of, node)
        if conversation is None:
            store.update_node(node_id, agents=_new_pod(ic_id), status="idle")
            self._begin(
                "start_team",
                node_id,
                task=task,
                parent=node.parent_id,
                checkout=node.worktree_path,
            )
            return store.get(node_id) or node
        additions = conversation.agent_launch_additions or AgentLaunchAdditions()
        suffix = "\n\n".join(
            part
            for part in (additions.system_message_suffix_append, POD_PROTOCOL)
            if part
        )
        store.update_node(node_id, agents=_new_pod(ic_id), status="idle")
        try:
            started = await launcher.start_root(
                conversation.model_copy(
                    update={
                        "conversation_id": ic_id,
                        "initial_message": None,
                        "autotitle": False,
                        "worktree": False,
                        "workspace": conversation.workspace.model_copy(
                            update={"working_dir": node.worktree_path}
                        ),
                        "agent_launch_additions": additions.model_copy(
                            update={"system_message_suffix_append": suffix}
                        ),
                    }
                ),
                [Tool(name=name) for name in DIRECTOR_TOOL_NAMES],
                title=_agent_title(node.title, "ic"),
                tool_params=await asyncio.to_thread(self._confine, node_id, "ic"),
                adapt_agent=await asyncio.to_thread(self._adapter, str(ic_id)),
            )
        except BaseException:
            store.update_node(node_id, agents=node.agents, status=node.status)
            raise
        store.set_agent(node_id, "ic", kind=started.kind, acp_server=started.acp_server)
        self._begin("brief", node_id, role="ic", task=task)
        self._begin("launch_manager", node_id, task=task)
        return store.get(node_id) or node

    def _task_of(self, node: OrgNode) -> str:
        """What a team that is started again is to do, from what the org
        kept of it."""
        above = self.store.get(node.parent_id) if node.parent_id else None
        specs, _ = self.blackboard.applies_to(
            node.boundary.path, above.boundary.path if above else None
        )
        task = specs[0].spec if specs else node.spec_summary
        if node.headline and node.headline not in task:
            task = f"{task}\n\nWhere the team last was: {node.headline}"
        return (
            f"{task}\n\nThis team was worked on before, on another machine. "
            "What it committed is in your checkout: look at it first and "
            "carry on from there."
        )

    def _repository(self) -> Path:
        """The repository the org is in, or would be in.

        Raises:
            Unavailable: The director's directory is not in a git repository.
        """
        if self.store.root_id is not None:
            return self._repo_dir
        if not is_git_repository(self._directory):
            raise Unavailable(
                f"An org needs a git repository; {self._directory} is not one"
            )
        return _checkout_root(str(self._directory))

    def _merging(self) -> dict[str, Any] | None:
        kept = self._runs().get("merge")
        return json.loads(kept) if kept else None

    async def _after(self, lane: str) -> None:
        """Wait until what ``lane`` was asked to do so far is done."""
        done = asyncio.Event()

        async def reached() -> None:
            done.set()

        self._enqueue(reached, lane)
        await done.wait()

    async def _take_org(self, repo: Path, remote: str, theirs: str | None) -> OrgSync:
        """Take the remote's org as the org of a repository that has none."""
        if theirs is None:
            return OrgSync(remote=remote, result="up_to_date", head=None)

        def take() -> OrgSync:
            board = create_org_board(repo)
            if not board.move_to(theirs):
                raise Conflict("The org was changed meanwhile. Pull once more.")
            nodes = saved_nodes(board, theirs)
            here = rehomed(nodes, {}, _shared_root(nodes), str(repo))
            if here != nodes:
                board.land([org_file(here)], f"org: at home in {repo.name}")
            blackboard = self._pull_blackboard(repo, remote, ())
            self._join(repo)
            return OrgSync(
                remote=remote,
                result="fast_forward",
                head=board.head(),
                blackboard=blackboard,
                branches=org_sync.take_in_branches(repo, remote, self.store.all()),
            )

        outcome = await asyncio.to_thread(take)
        await asyncio.to_thread(self._working)
        await self._came_back()
        return outcome

    async def _merge(self, state: dict[str, Any]) -> OrgSync:
        """Carry a merge as far as its choices allow."""
        outcome = await asyncio.to_thread(self._weigh, state, True)
        if outcome.result == "conflicts":
            self._set_merge(state)
            return outcome
        self._set_merge(None)
        await self._resync()
        store = self.store
        for branch in outcome.branches:
            node = store.get(branch.node_id)
            ic = agent_conversation(node, "ic") if node else None
            if branch.result == "diverged" and ic is not None and branch.ref:
                self._post(
                    ic,
                    OTHER_WORK_NOTE.format(
                        branch=branch.branch, remote=outcome.remote, ref=branch.ref
                    ),
                )
        if state["resume"]:
            await self.resume()
        return outcome

    def _weigh(self, state: dict[str, Any], land: bool = False) -> OrgSync:
        """Work a merge out and, when ``land`` is set and no conflict is
        open, write it."""
        store = self.store
        repo, remote = self._repo_dir, state["remote"]
        theirs: str | None = state["theirs"]
        choices: dict[str, Side] = state["choices"]
        board = create_org_board(repo)
        blackboard = GitBlackboard(repo)
        their_board = org_sync.commit_at(repo, org_sync.remote_blackboard(remote))
        files = [
            OrgConflict(
                id=f"file:{clash.path}",
                kind="file",
                node_id="",
                title="",
                field=clash.path,
                mine=clash.ours,
                theirs=clash.theirs,
                choice=choices.get(f"file:{clash.path}"),
            )
            for clash in (blackboard.clashes(their_board) if their_board else [])
        ]
        for _ in range(_MERGE_ATTEMPTS):
            save_org(board, store)
            head = board.head()
            mine = by_id(saved_nodes(board, head))
            merged = None
            other: Nodes = {}
            if theirs is not None and not board.contains(theirs):
                other = by_id(saved_nodes(board, theirs))
                base = by_id(saved_nodes(board, board.base_with(theirs)))
                merged = merge_orgs(base, mine, other, choices)
            conflicts = [*(merged.conflicts if merged else []), *files]
            if any(conflict.choice is None for conflict in conflicts):
                return OrgSync(
                    remote=remote,
                    result="conflicts",
                    head=head,
                    merge=OrgMerge(
                        remote=remote, theirs=theirs or "", conflicts=conflicts
                    ),
                )
            if not land:
                return OrgSync(remote=remote, result="up_to_date", head=head)
            result: SyncResult = "up_to_date"
            if merged is not None and theirs is not None:
                nodes = rehomed(
                    merged.nodes, mine, _shared_root(list(other.values())), str(repo)
                )
                if by_id(nodes) == other:
                    landed, result = board.move_to(theirs), "fast_forward"
                else:
                    landed = board.land(
                        [org_file(nodes)], f"org: merge {remote}", merging=theirs
                    )
                    result = "merged"
                if not landed:
                    continue
                before = {node.id: fingerprint(node) for node in store.all()}
                self._apply(store, nodes)
                run = self._runs()
                # What the run state was written for, so that what the merge
                # changed under it is found.
                run.replace_marks(TEAM, before)
                run.put("org_commit", head)
            break
        else:
            raise Conflict("The org kept changing while it was merged. Pull once more.")
        taken = [conflict.field for conflict in files if conflict.choice == "theirs"]
        synced = self._pull_blackboard(repo, remote, taken)
        if result == "up_to_date" and synced is not None and synced.changed:
            result = "merged"
        branches = org_sync.take_in_branches(repo, remote, store.all())
        already = len(self._dropped())
        self._put_aside_stale()
        self._save_tree(store)
        return OrgSync(
            remote=remote,
            result=result,
            head=board.head(),
            blackboard=synced,
            branches=branches,
            put_aside=self._dropped()[already:],
        )

    def _pull_blackboard(
        self, repo: Path, remote: str, take_theirs: Collection[str]
    ) -> BlackboardSync | None:
        ref = org_sync.remote_blackboard(remote)
        if org_sync.commit_at(repo, ref) is None:
            return None
        board = GitBlackboard(repo)
        before = board.head()
        merged = board.merge(ref, take_theirs)
        return BlackboardSync(
            remote=remote,
            head=merged.head,
            changed=merged.head != before,
            kept_local=merged.kept_local,
            pushed=False,
        )

    def _apply(self, store: OrgStore, nodes: list[dict[str, Any]]) -> None:
        """Make ``store`` hold the merged tree, announcing what changed."""
        wanted = by_id(nodes)
        for node in store.all():
            moved = node.id in wanted and wanted[node.id]["parent_id"] != node.parent_id
            if (node.id not in wanted or moved) and store.get(node.id) is not None:
                store.remove(node.id)
        current = by_id(stored_nodes(store))
        for node in nodes:
            if current.get(node["id"]) != node:
                store.upsert(OrgNodeInput.model_validate(node))

    def _runs(self) -> RunState:
        """What the director is in the middle of, in the org's repository."""
        _ = self.store  # settles which repository the org is in
        with self._lock:
            if self._run is None or self._run_of != self._repo_dir:
                if self._run is not None:
                    self._run.close()
                self._run = self._open_run_state()
                self._run_of = self._repo_dir
                kept = self._run.get("paused")
                self._paused = kept if kept in ("human", "shutdown", "crash") else None
                self._running = set(self._run.marked(RUNNING))
            return self._run

    def _open_run_state(self) -> RunState:
        if self._fixture or not is_git_repository(self._repo_dir):
            return RunState(None)
        try:
            return RunState(self._shells().parent / _RUN_STATE)
        except (GitError, OSError, RunStateError):
            logger.exception("Run state of %s is kept in memory only", self._repo_dir)
            return RunState(None)

    def _hold(self, reason: PauseReason) -> None:
        self._runs().put("paused", reason)
        self._paused = reason
        self._announce_state()

    def _release(self) -> None:
        run = self._runs()
        run.put("paused", None)
        run.put("dropped", None)
        self._paused = None
        self._announce_state()

    def _dropped(self) -> list[RecoveryItem]:
        kept = self._runs().get("dropped")
        return [RecoveryItem.model_validate(item) for item in json.loads(kept or "[]")]

    def _instance(self) -> str:
        """Names this server, to tell its run state from another's."""
        if self._remembered is None:
            return ""
        kept = self._read_remembered().get("instance")
        if not isinstance(kept, str):
            kept = uuid4().hex
            self._remember(instance=kept)
        return kept

    def _working(self) -> None:
        """Record that the org is run by this server and has not been
        stopped."""
        run = self._runs()
        run.put("instance", self._instance())
        run.put("clean", "0")

    def _put_aside_stale(self) -> None:
        run = self._runs()
        head = create_org_board(self._repo_dir).head()
        dropped = put_aside_stale(
            run, self.store, instance=self._instance(), org_commit=head
        )
        if run.get("paused") is None:
            # Nothing that was left is this server's to take up.
            self._paused = None
            self._announce_state()
        if dropped:
            run.put(
                "dropped",
                json.dumps(
                    [item.model_dump() for item in [*self._dropped(), *dropped]]
                ),
            )

    async def _came_back(self) -> None:
        """Look at what an org that was just loaded was left in the middle
        of. A server that ended without stopping its org leaves it paused."""
        await asyncio.to_thread(self._look_at_what_was_left)
        await self._resync()
        if self._paused in ("shutdown", "crash") and self.settings.resume_on_start:
            await self.resume()

    def _look_at_what_was_left(self) -> None:
        if self._fixture or self.store.root_id is None:
            return
        run = self._runs()
        self._put_aside_stale()
        if self._paused is None and run.get("clean") == "0" and has_work(run):
            self._hold("crash")
        if self._paused is None:
            for item in self._dropped():
                logger.warning("Run state put aside: %s", item.detail)
            run.put("dropped", None)
        self._working()

    async def _pause_agents(self) -> None:
        store = self.store
        run = self._runs()
        launcher = self._require_launcher()
        agents: list[tuple[str, AgentRole, UUID]] = [
            (node.id, agent.role, UUID(agent.conversation_id))
            for node in store.all()
            for agent in node.agents
            if agent.conversation_id is not None
        ]
        statuses = await launcher.statuses([agent for _, _, agent in agents])
        stopping: list[UUID] = []
        for node_id, role, conversation_id in agents:
            if statuses.get(conversation_id) != ConversationExecutionStatus.RUNNING:
                continue
            run.mark(RUNNING, str(conversation_id), node_id)
            self._running.add(str(conversation_id))
            with suppress(NotFound):
                store.set_agent(node_id, role, status=AGENT_PAUSED)
            stopping.append(conversation_id)
        await asyncio.gather(*(launcher.pause(agent) for agent in stopping))

    async def _run_again(self, conversation_id: str, note: str | None) -> None:
        self._resuming.discard(conversation_id)
        with suppress(NotFound):
            await self._require_launcher().resume(UUID(conversation_id), note)

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
            self._enqueue(self._came_back)
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
        with self._remember_lock:
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

    def _begin(self, kind: str, node_id: str, **payload: object) -> None:
        """Record something of several steps the director does for a team,
        then do it. Safe from any thread.

        It is finished after a restart or a pause from the steps that were
        done, so each step must be safe to take twice.
        """
        if self._loop is None:
            raise Unavailable("The director is not running agents on this server")
        intent = self._runs().begin(kind, node_id, payload)
        self._save_soon()
        self._enqueue(lambda: self._carry_out(intent.id), _lane_for(intent))

    async def _carry_out(self, intent_id: int) -> None:
        run = self._runs()
        if self._paused is not None or intent_id in self._in_flight:
            return
        intent = next(
            (open_ for open_ in run.open_intents() if open_.id == intent_id), None
        )
        if intent is None:
            return
        step = self._steps.get(intent.kind)
        if step is None:
            run.finish(intent.id)
            self._leave_line(intent.node_id)
            return
        self._in_flight.add(intent.id)
        try:
            await step(intent)
        except _Held:
            return
        except Exception:
            logger.exception("A director action failed")
            run.finish(intent.id)
        else:
            if intent.kind in _SAVED_BEFORE_FORGOTTEN:
                # What it changed in the tree is saved before it is
                # forgotten: a server that ends in between does the step
                # again, never loses it.
                await self._save()
            run.finish(intent.id)
        finally:
            self._in_flight.discard(intent.id)
            self._leave_line(intent.node_id)
        if intent.kind == "start_team":
            above = intent.payload["parent"]
            others = [
                other
                for other in run.open_intents()
                if other.kind == "start_team" and other.payload["parent"] == above
            ]
            if not others:
                self._begin("finish_split", above)

    def _post(self, conversation_id: str, text: str) -> None:
        """Record a message to an agent, then deliver it. Safe from any
        thread. While the org is paused the message waits."""
        if self._loop is None:
            raise Unavailable("The director is not running agents on this server")
        found = self.store.find_by_conversation(conversation_id)
        node_id = found[0].id if found else ""
        seq = self._runs().hold(conversation_id, node_id, text)
        self._save_soon()
        self._enqueue(
            lambda: self._deliver(seq, conversation_id, text), node_id or ORG_LANE
        )

    async def _deliver(self, seq: int, conversation_id: str, text: str) -> None:
        run = self._runs()
        if self._paused is not None or seq in self._delivering:
            return
        if not run.is_held(seq):
            return
        self._delivering.add(seq)
        try:
            await self._require_launcher().send(UUID(conversation_id), text)
        except NotFound:
            store = self.store
            found = store.find_by_conversation(conversation_id)
            if found is not None:
                store.set_agent(found[0].id, found[1], status=AGENT_LOST)
            logger.warning("Agent conversation %s is gone", conversation_id)
        finally:
            self._delivering.discard(seq)
        run.delivered(seq)

    async def _when_there_is_room[T](
        self, node_id: str, start: Callable[[], Awaitable[T]]
    ) -> T:
        """Run ``start`` for a team; while the server is full, wait in line
        for a conversation to stop running and try again."""
        while node_id in self._line and self._line[0] != node_id:
            await self._line_moved.wait()
        try:
            return await start()
        except ConversationRunLimitExceeded:
            logger.info("Server is at its run limit; node %s waits", node_id)
        finally:
            # Nothing is awaited from here to the gate, so the team is in
            # line at the gate before the next one tries.
            self._leave_line(node_id)
        store = self.store
        run = self._runs()
        run.mark(WAITING, node_id)
        with suppress(NotFound):
            store.update_node(node_id, waiting_to_start=True)
        keeps_place = False
        try:
            async with self._start_gate:
                while True:
                    if self._paused is not None:
                        raise _Held
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
        except (_Held, asyncio.CancelledError):
            # Stopped while it waited: it is still next when the org resumes.
            keeps_place = True
            raise
        finally:
            if not keeps_place:
                run.unmark(WAITING, node_id)
            with suppress(NotFound):
                store.update_node(node_id, waiting_to_start=False)

    def _leave_line(self, node_id: str) -> None:
        if node_id in self._line:
            self._line.remove(node_id)
            moved, self._line_moved = self._line_moved, asyncio.Event()
            moved.set()

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

    def _save_soon(self) -> None:
        """Save the tree without waiting to gather more changes: what was
        just recorded as begun refers to it."""
        if not self._save_asked:
            self._save_asked = True
            self._enqueue(self._save_asked_for)

    async def _save_asked_for(self) -> None:
        self._save_asked = False
        await self._save()

    async def _save(self) -> None:
        store = self._store
        if store is None or self._fixture:
            return
        try:
            await asyncio.to_thread(self._save_tree, store)
        except GitError:
            logger.exception("Could not save the org tree to %s", self._repo_dir)

    def _save_tree(self, store: OrgStore) -> None:
        board = create_org_board(self._repo_dir)
        save_org(board, store)
        run = self._runs()
        run.replace_marks(TEAM, {node.id: fingerprint(node) for node in store.all()})
        run.put("org_commit", board.head())

    def _on_status(
        self, conversation_id: UUID, status: ConversationExecutionStatus
    ) -> None:
        if status != ConversationExecutionStatus.RUNNING and self._loop is not None:
            self._loop.call_soon_threadsafe(self._room.set)
        store = self._store
        if store is None:
            return
        agent = str(conversation_id)
        found = store.find_by_conversation(agent)
        if found is None:
            return
        node, role = found
        if self._paused is None:
            self._note_running(
                agent, node.id, status == ConversationExecutionStatus.RUNNING
            )
        store.set_agent(node.id, role, status=self._shown(agent, status))

    def _note_running(self, conversation_id: str, node_id: str, running: bool) -> None:
        if running == (conversation_id in self._running):
            return
        run = self._runs()
        if running:
            self._running.add(conversation_id)
            run.mark(RUNNING, conversation_id, node_id)
        else:
            self._running.discard(conversation_id)
            run.unmark(RUNNING, conversation_id)

    def _shown(self, conversation_id: str, status: ConversationExecutionStatus) -> str:
        """An agent's status as the org shows it. One that was running when
        the org was paused is paused, whatever stopping did to it."""
        held = self._paused is not None and conversation_id in self._running
        if held or conversation_id in self._resuming:
            if status != ConversationExecutionStatus.RUNNING:
                return AGENT_PAUSED
        return _AGENT_STATUS.get(status, "idle")

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
        starting = {
            intent.node_id
            for intent in self._runs().open_intents()
            if intent.kind in STARTING
        }
        for node_id, role, conversation_id in agents:
            status = statuses.get(conversation_id)
            agent = str(conversation_id)
            if self._paused is None:
                self._note_running(
                    agent, node_id, status == ConversationExecutionStatus.RUNNING
                )
            if status is not None:
                shown = self._shown(agent, status)
            else:
                shown = "idle" if node_id in starting else AGENT_LOST
            with suppress(NotFound):
                store.set_agent(node_id, role, status=shown)

    async def _send(self, conversation_id: str, text: str) -> None:
        """Deliver a message as part of a step, in its place among them."""
        found = self.store.find_by_conversation(conversation_id)
        seq = self._runs().hold(conversation_id, found[0].id if found else "", text)
        await self._deliver(seq, conversation_id, text)

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
        checkout: str | None = None,
    ) -> bool:
        """Start one agent of a node from ``parent_conversation``.

        Args:
            checkout: The team's checkout when it has one already and the
                agent is to work on in it.

        Returns:
            Whether it runs. A failure is recorded on the node.
        """
        store = self.store
        node = store.get(node_id)
        conversation_id = agent_conversation(node, role) if node else None
        if node is None or conversation_id is None:
            return False
        launcher = self._require_launcher()
        try:
            started = await launcher.started(UUID(conversation_id))
            if started is None:
                started = await self._start_new(
                    node, role, conversation_id, parent_conversation, task, checkout
                )
                if self._paused is not None:
                    self._note_running(conversation_id, node_id, True)
                    await launcher.pause(UUID(conversation_id))
        except _Held:
            raise
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
                base_commit=node.base_commit
                or await asyncio.to_thread(_base_of, started.workspace),
                boundary_enforcement=self._enforcement(started.kind),
            )
        return True

    async def _start_new(
        self,
        node: OrgNode,
        role: AgentRole,
        conversation_id: str,
        parent_conversation: str | None,
        task: str,
        checkout: str | None = None,
    ) -> StartedAgent:
        """Start the conversation of an agent that has none yet.

        Raises:
            NotFound: There is nobody to derive the agent from.
            _Held: The org was paused while the team waited for a run.
        """
        node_id = node.id
        launcher = self._require_launcher()
        text = await asyncio.to_thread(self._briefing, node, role, task)
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
                    checkout=checkout,
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
                checkout=checkout,
            )

        return await self._when_there_is_room(node_id, start)

    async def _launch_manager(self, node_id: str, task: str) -> None:
        node = self.store.get(node_id)
        if node is not None:
            await self._start_agent(
                node_id, "manager", agent_conversation(node, "ic"), task
            )

    async def _do_brief(self, intent: Intent) -> None:
        await self._brief(
            intent.node_id, intent.payload["role"], intent.payload["task"]
        )

    async def _do_launch_manager(self, intent: Intent) -> None:
        await self._launch_manager(intent.node_id, intent.payload["task"])

    async def _do_start_team(self, intent: Intent) -> None:
        node_id, task = intent.node_id, intent.payload["task"]
        above = self.store.get(intent.payload["parent"])
        parent_ic = agent_conversation(above, "ic") if above else None
        checkout = intent.payload.get("checkout")
        if await self._start_agent(node_id, "ic", parent_ic, task, checkout):
            await self._launch_manager(node_id, task)
        self._enqueue(lambda: self._summarize(node_id, task, first=True), node_id)

    async def _do_finish_split(self, intent: Intent) -> None:
        await self._finish_split(intent.node_id)

    async def _do_request_review(self, intent: Intent) -> None:
        await self._request_review(intent.node_id)

    async def _do_accept(self, intent: Intent) -> None:
        await self._accept(intent.node_id, intent.payload["needed_above"])

    async def _do_halt(self, intent: Intent) -> None:
        await self._halt(intent.node_id)

    async def _carry_out_split(self, intent: Intent) -> None:
        store = self.store
        run = self._runs()
        node_id = intent.node_id
        node = store.get(node_id)
        proposal = node.pending_split if node else None
        if node is None or proposal is None or proposal.status != "approved":
            return
        if "planned" not in intent.steps:
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
                for conversation_id in (
                    agent_conversation(node, "ic"),
                    agent_conversation(node, "manager"),
                ):
                    self.notify(
                        conversation_id,
                        f"The approved split cannot be carried out: {error}",
                    )
                return
            taken = {team.id for team in store.all()}
            # Fixed before any of them exists, so that carrying the split
            # out again makes the same teams and no others.
            intent = run.step_done(
                intent,
                "planned",
                teams=[
                    _child_node(taken, node, child).model_dump(mode="json")
                    for child in proposal.children
                ],
            )
        if proposal.contracts.strip():
            await asyncio.to_thread(self._record_contracts, node, proposal.contracts)
        teams = [OrgNodeInput.model_validate(team) for team in intent.payload["teams"]]
        for team in teams:
            if store.get(team.id) is None:
                store.upsert(team)
        await self._save()
        # The sub-teams start side by side, each in its own lane; the split
        # is over when the last of them has.
        for team, child in zip(teams, proposal.children, strict=True):
            step = f"start:{team.id}"
            if step not in intent.steps:
                self._begin("start_team", team.id, task=child.task, parent=node_id)
                intent = run.step_done(intent, step)

    async def _finish_split(self, node_id: str) -> None:
        store = self.store
        node = store.get(node_id)
        proposal = node.pending_split if node else None
        if node is None or proposal is None or proposal.status != "approved":
            return
        store.update_node(node_id, pending_split=None)
        # The shell the IC starts next can no longer write what it gave away.
        with suppress(GitError, OSError):
            await asyncio.to_thread(self._confine, node_id, "ic")
        started = [f"{child.title} ({child.path})" for child in proposal.children]
        self.notify(
            agent_conversation(node, "ic"),
            "Your split was approved and these sub-teams now own their "
            f"directories: {', '.join(started)}. You can no longer write "
            "there; carry on with what remains in your boundary.",
        )

    async def _prune(self, intent: Intent) -> None:
        store = self.store
        run = self._runs()
        node_id = intent.node_id
        launcher = self._require_launcher()
        if "planned" not in intent.steps:
            node = store.get(node_id)
            above = store.get(node.parent_id) if node and node.parent_id else None
            if node is None or above is None:
                return
            # The team first, then everything below it.
            teams = [node]
            for team in teams:
                teams.extend(store.children(team.id))
            now = datetime.now(UTC)
            # Fixed before anything is removed, so that pruning again after
            # a restart removes the same things and records them once.
            intent = run.step_done(
                intent,
                "planned",
                title=node.title,
                path=node.boundary.path,
                parent=above.id,
                teams=[team.id for team in teams],
                conversations=[
                    agent.conversation_id
                    for team in teams
                    for agent in team.agents
                    if agent.conversation_id is not None
                ],
                stamp=now.strftime("%Y%m%dT%H%M%SZ"),
                at=now.isoformat(timespec="seconds"),
                fresh=str(uuid4()),
            )
        plan = intent.payload
        where, replacement_task = plan["path"], plan["replacement_task"]
        conversations = [UUID(agent) for agent in plan["conversations"]]
        if "stopped" not in intent.steps:
            for conversation_id in conversations:
                await launcher.stop(conversation_id)
            intent = run.step_done(intent, "stopped")
        if "discarded" not in intent.steps:
            teams = [team for team in map(store.get, plan["teams"]) if team is not None]
            saved = await asyncio.to_thread(self._discard_work, teams, plan["stamp"])
            intent = run.step_done(intent, "discarded", saved=saved)
        saved = intent.payload["saved"]
        if "deleted" not in intent.steps:
            for conversation_id in conversations:
                await launcher.delete(conversation_id)
            intent = run.step_done(intent, "deleted")
        autopsy = Autopsy(
            node_id=node_id,
            at=plan["at"],
            approach=plan["approach"],
            why_failed=plan["why_failed"],
            saved_ref=saved,
        )
        try:
            await asyncio.to_thread(self._record_autopsy, where, autopsy)
        except (GitError, DomainError):
            logger.exception("Could not record the autopsy of node %s", node_id)

        above = store.get(plan["parent"])
        parent_ic = agent_conversation(above, "ic") if above else None
        if "removed" not in intent.steps:
            if replacement_task is None:
                store.remove(node_id)
                # The shell the IC above starts next can write there again.
                with suppress(GitError, OSError):
                    await asyncio.to_thread(self._confine, plan["parent"], "ic")
            elif above is not None:
                for child in store.children(node_id):
                    store.remove(child.id)
                store.update_node(
                    node_id,
                    agents=_new_pod(UUID(plan["fresh"])),
                    status="idle",
                    worktree_path=above.worktree_path,
                    branch=above.branch,
                    base_commit="",
                    escalations=[],
                    pending_split=None,
                    review=None,
                    headline=provisional_summary(replacement_task).headline,
                    spec_summary=_summary_of(replacement_task),
                )
            await self._save()
            intent = run.step_done(intent, "removed")
        if replacement_task is None:
            outcome = f"{where} is yours again"
        else:
            if await self._start_agent(node_id, "ic", parent_ic, replacement_task):
                await self._launch_manager(node_id, replacement_task)
            self._enqueue(
                lambda: self._summarize(node_id, replacement_task, first=True),
                node_id,
            )
            outcome = f"a fresh team now works in {where}"
        self.notify(
            parent_ic,
            f"Your Manager pruned the team {plan['title']!r} ({where}): "
            f"{plan['why_failed']} Its uncommitted changes were discarded"
            + (f" (kept in git as {saved})" if saved else "")
            + f"; {outcome}.",
        )

    def _record_autopsy(self, where: str, autopsy: Autopsy) -> None:
        board = GitBlackboard(self._repo_dir)
        if autopsy not in board.autopsies(where):
            board.append_autopsy(where, autopsy)

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
                self._runs().mark(REVIEWING, above.id)
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
        if self._paused is not None:
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
        if self._paused is not None:
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
        if earlier and text.strip() in earlier:
            return
        blackboard.write(
            path,
            CONTRACTS_FILE,
            f"{earlier.rstrip()}\n\n{text}" if earlier and earlier.strip() else text,
        )


def _child_node(taken: set[str], parent: OrgNode, child: SplitChild) -> OrgNodeInput:
    """The team a split makes of ``child``, under an id that is not in
    ``taken`` and is added to it."""
    base = ".".join(
        re.sub(r"[^A-Za-z0-9_-]+", "-", segment) for segment in child.path.split("/")
    )
    node_id, attempt = base, 1
    while node_id in taken:
        attempt += 1
        node_id = f"{base}-{attempt}"
    taken.add(node_id)
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


def _shared_root(nodes: list[dict[str, Any]]) -> str | None:
    """Where the repository is on the machine that saved ``nodes``, when
    their root team works in the repository's own checkout."""
    root = next((node for node in nodes if node["parent_id"] is None), None)
    if root is None or root["isolation"] != "shared":
        return None
    return root["worktree_path"]


# Steps whose outcome is in the tree alone. The steps that start agents are
# not among them: their outcome is the conversation, and nothing may come
# between an IC's start and its Manager's.
_SAVED_BEFORE_FORGOTTEN = frozenset({"accept", "request_review", "halt"})


def _lane_for(intent: Intent) -> str:
    if intent.kind in _REVIEW_STEPS:
        return f"review:{intent.node_id}"
    return intent.node_id


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
