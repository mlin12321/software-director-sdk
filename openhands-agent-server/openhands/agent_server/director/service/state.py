"""What every part of the director's service works on: the org tree, the
team memory, the settings, the queue of steps and the agents' shells."""

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Self
from uuid import UUID, uuid4

from openhands.agent_server.config import Config
from openhands.agent_server.conversation_service import ToolParams
from openhands.agent_server.director import bugbot
from openhands.agent_server.director.acp import (
    agent_token,
    as_org_agent,
    conversation_of,
    tools_server,
)
from openhands.agent_server.director.git import (
    GitError,
    common_dir,
    enclosing_repository,
    git,
    head_of,
    is_git_repository,
    top_of,
)
from openhands.agent_server.director.git_index import (
    excludes_file,
    git_script,
    rcfile,
)
from openhands.agent_server.director.git_store import (
    CONTRACTS_FILE,
    SPEC_FILE,
    GitStore,
    Landing,
    normalize_store_path,
    swap_together,
    taking_in,
)
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.merge_checkout import MergeCheckout, discard
from openhands.agent_server.director.models import (
    ROLE_NAMES,
    AgentKind,
    AgentRef,
    AgentRole,
    DirectorSettings,
    DirectorSettingsState,
    DirectorSettingsUpdate,
    FormerTeam,
    HandOverMode,
    OrgStateEvent,
    OrgSummary,
    PauseReason,
    RecoveryItem,
    RoutedRequest,
    SessionTeam,
    SessionTeamInput,
    SplitChild,
    TeamEnforcement,
    TeamMatch,
)
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    load_org,
    save_org,
    saving,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    is_division_lead,
    session_team_id,
)
from openhands.agent_server.director.run_state import (
    MERGING,
    PROGRESS,
    RUNNING,
    SESSION,
    Intent,
    RunState,
    RunStateError,
)
from openhands.agent_server.director.sandbox import (
    command_script,
    find_bwrap,
    shell_script,
    write_shell,
)
from openhands.agent_server.director.session_state import (
    kept_sessions,
    restore_sessions,
)
from openhands.agent_server.director.split import (
    child_boundary,
    clean_path,
)
from openhands.sdk.agent import ACPAgent
from openhands.sdk.agent.base import AgentBase
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import (
    DomainError,
    NotFound,
    OperationFailed,
    Unavailable,
    Unprocessable,
)
from openhands.sdk.tool.spec import Tool
from openhands.sdk.utils.files import atomic_write_text


logger = logging.getLogger(__name__)

ROOT_ID = "root"
_TOOLS_OF_BOTH = (
    "read_team_context",
    "raise_escalation",
    "resolve_escalation",
    "report",
)
# The director's tools an agent is offered, by its role; they register in
# ``director.tools``. `agent_actions` refuses the rest whatever is offered.
TOOLS_OF: dict[AgentRole, tuple[str, ...]] = {
    "ic": (
        *_TOOLS_OF_BOTH,
        "write_spec",
        "msg_manager",
        "propose_teams",
        "msg_team",
        "hand_over",
        "stop_team",
        "submit_review",
        "review_team",
    ),
    "manager": (
        *_TOOLS_OF_BOTH,
        "msg_ic",
        "approve_teams",
        "decline_teams",
        "prune_team",
    ),
}
HAND_OVER_TOOLS = ("read_team_context", "hand_over", "write_spec")
# What an IC has of the director's tools where it only reviews the work of
# a session.
DIVISION_LEAD_IC_TOOLS = ("read_team_context", "report", "review_team")
# What a Manager has there: it settles the sub-teams a session and the org
# each made something else of.
DIVISION_LEAD_MANAGER_TOOLS = ("read_team_context", "report", "settle_teams")


def tools_of(place: SessionTeam, role: AgentRole) -> tuple[str, ...]:
    """The director's tools an agent of ``node`` is offered."""
    if is_division_lead(place):
        return DIVISION_LEAD_IC_TOOLS if role == "ic" else DIVISION_LEAD_MANAGER_TOOLS
    return TOOLS_OF[role]


DIRECTOR_TOOL_NAMES = tuple(
    dict.fromkeys((*TOOLS_OF["ic"], *TOOLS_OF["manager"], *DIVISION_LEAD_MANAGER_TOOLS))
)
# How a team that was set up and given no task reads to the IC above.
NOT_STARTED = "not_started"
# What follows a team's purpose in its task once something started it.
_STARTED_WITH = "\n\nWhat your team is started with:\n"
# How many of the latest events of its sub-teams an agent reads at once.
# How many of the human's direct requests to the teams below an IC is told.
ASKED_TOLD = 10
# How much of such a request is kept in the record of the team above.
ASKED_KEPT = 1000
HISTORY_TOLD = 30
# How long tree changes are gathered before they are saved as one commit.
SAVE_DELAY_SECONDS = 2.0
# How long to wait before starting an agent again when the server is full.
RUN_LIMIT_RETRY_SECONDS = 5.0
# How often the director looks for teams that show no progress.
STALL_POLL_SECONDS = 60.0
# How many commits must have changed two proposed directories together
# before a Manager is told: a first commit of everything says nothing.
TOGETHER_WORTH_SAYING = 3
# How many earlier declines for the same directories a proposal is shown with.
DECLINED_TOLD = 3
# How long a model may take to name a team before the team keeps the name
# it has; a session's first IC is not briefed until then.
SUMMARY_TIMEOUT_SECONDS = 20.0
# How long a model may take over a team's diff, or over an agent's reports.
MODEL_CALL_TIMEOUT_SECONDS = 120.0
# A team whose work is sent back this many times is stopped.
# Where, in the org's repository, the uncommitted work a prune discarded is
# kept.
PRUNED_REFS = "refs/agents/pruned"
# Where what an IC left uncommitted in the directory of a sub-team with a
# checkout of its own is kept: under `left`, what went to the team as it
# started; under `aside`, what was in the way of the team's accepted work.
LEFT_REFS = "refs/agents/left"
ASIDE_REFS = "refs/agents/aside"
# An agent whose conversation is gone, or could not be started.
# An agent that was running when its org was paused, and runs again when it
# is resumed.
AGENT_PAUSED = "paused"
# What an agent is told when the server ended in the middle of its step.
RESTART_NOTE = (
    "The server was restarted while you were working. The command or edit "
    "you were in the middle of may not have finished: check what is there "
    "before you carry on with your task."
)
# How often a merge is made again because the org was saved meanwhile.
_MERGE_ATTEMPTS = 4
# Where, in the repository's git directory, the director keeps what it is in
# the middle of.
_RUN_STATE = "director/state.db"
# Where, in the repository's git directory, the director keeps the shells
# its agents' terminals start, and what those shells need.
_SHELLS_DIR = "director-shells"
# In the git directory, and not for an agent's shell to write.
_PRIVATE = ("director", "hooks", "config")
# In the repository's git directory: the merges reviewers finish.
_MERGING_DIR = "director-merging"
# After a conversation id: the script an ACP agent's process is started by.
_ACP_LAUNCHER = ".acp.sh"
_REMEMBERED_FILE = "director.json"
# The lane of what concerns the org as a whole, such as saving its tree.
ORG_LANE = "org"
# The lane of everything that changes what the org shares: the repository's
# checkout, the tree and team memory. One thing at a time, in order.
BASE_LANE = "base"
# The lane reports are rewritten shorter in.
MODELS_LANE = "models"
# The lane agents are paused in, so that nothing waits for their last step.
PAUSE_LANE = "pause"
STALL_LANE = "stalls"
# The steps the director takes for a team's review, in its own lane.
_REVIEW_STEPS = ("request_review", "accept")
# The escalation that asks the human to clear the way for a merge.
MERGE_WAITS = "merge-waits"
# The id of the escalation raised when a Manager does not settle the teams
# and contracts of accepted work.
ORG_WAITS = "org-waits"
# The escalation that asks the human whether a session that reached its
# token limit goes on.
LIMIT_WAITS = "limit-reached"
# How much of an error is kept as the reason an agent failed.
_REASON_KEPT = 600
# The escalation that tells the human a team shows no progress where no IC
# works above it.
STALL_WAITS = "stalled"
# How many of the director's own model calls run at once.
MODEL_CALLS_AT_ONCE = 2


def failure_waits(role: AgentRole) -> str:
    """The id of the escalation that tells the human an agent failed."""
    return f"{role}-failed"


_AGENT_STATUS: dict[ConversationExecutionStatus, str] = {
    ConversationExecutionStatus.RUNNING: "working",
    ConversationExecutionStatus.ERROR: "blocked",
    ConversationExecutionStatus.STUCK: "blocked",
}

# How long a message waits for an IC that submitted to end its turn.
_REST_CHECKS = 600
_REST_CHECK_SECONDS = 0.05

type Effect = Callable[[], Awaitable[None]]
# The lane the running step is carried out in; '' outside the lanes.
_LANE: ContextVar[str] = ContextVar("director_lane", default="")
# How often a landing is tried when another writer moves a ref first.
_LANDING_ATTEMPTS = 8


class _Held(Exception):
    """The org was paused before a step could be finished."""


def agent_conversation(place: SessionTeam, role: AgentRole) -> str | None:
    """The conversation of the team's agent in ``role``, if it has one."""
    for agent in place.agents:
        if agent.role == role:
            return agent.conversation_id
    return None


def hand_over_of(place: SessionTeam) -> HandOverMode | None:
    """The hand-over mode of the team's IC, if it is or was in it."""
    for agent in place.agents:
        if agent.role == "ic":
            return agent.hand_over
    return None


def _agent_title(
    team: str, role: AgentRole, session: str = "", *, division_lead: bool = False
) -> str:
    """What an agent's conversation is called. The session is named unless
    it is called what its team is, and the two agents a session's work is
    reviewed by are called its Division Leads."""
    where = f"{team} · {session}" if session and session != team else team
    name = ROLE_NAMES[role]
    return f"{where} ({'Division Lead ' + name if division_lead else name})"


def _summary_of(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line[:200]


@lru_cache(maxsize=4096)
def _checkout_root(workspace: str) -> Path:
    """The top of the git checkout ``workspace`` is in, symlinks resolved."""
    try:
        top = top_of(Path(workspace))
    except (GitError, OSError):
        top = workspace
    return Path(os.path.realpath(top))


def apart(team: SessionTeam, above: SessionTeam) -> bool:
    """Whether a team works in a checkout of its own, not in the one of
    the team above."""
    return _checkout_root(team.worktree_path) != _checkout_root(above.worktree_path)


def _is_in(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


class ServiceState:
    """The state of the director for one server, and what guards and saves it."""

    def __init__(
        self,
        repo_dir: Path,
        *,
        save_delay: float = SAVE_DELAY_SECONDS,
        run_limit_retry: float = RUN_LIMIT_RETRY_SECONDS,
        worktree_root: Path | None = None,
        find_sandbox: Callable[[], str | None] = lambda: None,
        summary_timeout: float = SUMMARY_TIMEOUT_SECONDS,
        remembered: Path | None = None,
        find_review_instructions: Callable[[], str | None] = lambda: None,
        model_call_timeout: float = MODEL_CALL_TIMEOUT_SECONDS,
        max_concurrent_runs: int | None = None,
        server_url: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        stall_poll: float = STALL_POLL_SECONDS,
    ) -> None:
        """
        Args:
            clock: The time, for telling how long a team has shown no
                progress.
            stall_poll: How many seconds lie between two looks for teams
                that show no progress.
            server_url: Where this server answers, for agents that reach the
                director's tools from a process of their own; None to take
                it from the requests the director gets.
            max_concurrent_runs: How many conversations the server runs at
                once, to tell the human; None when not known.
            find_review_instructions: Returns what a model is told when it
                reads a team's diff before the IC above reviews it, or None
                to leave that pass out.
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
        # Messages that do not wait for their agent's turn to end: what the
        # human said, and what waited for a turn that has now ended.
        self._at_once: set[int] = set()
        # The conversations last recorded as running.
        self._running: set[str] = set()
        # The sessions that changed something of the org's after their
        # work was accepted, and whether their IC was told to submit it.
        self._changed_since_accept: dict[str, bool] = {}
        # The teams each step waits for the agents of to stop, by the id of
        # its intent, and the lane the step is carried out in.
        self._settling: dict[int, tuple[set[str], str]] = {}
        # The conversations a resume is about to run again.
        self._resuming: set[str] = set()
        self._save_asked = False
        self._sessions_asked = False
        # Teams that waited for a run when the org was paused, in the order
        # they waited: each tries to start only when it is the first.
        self._line: deque[str] = deque()
        self._line_moved = asyncio.Event()
        # The records of removed teams by the team they were below, as read
        # at one state of team memory.
        self._former_read: tuple[str | None, dict[str, list[FormerTeam]]] | None = None
        self._routed_read: tuple[str | None, list[RoutedRequest]] | None = None
        self._bwrap: str | None = None
        self._stop_guard: Callable[[], None] | None = None
        self._save_delay = save_delay
        self._run_limit_retry = run_limit_retry
        self._lock = threading.Lock()
        # Held from looking a need up in the record to keeping it, so that
        # an agent that tries the same file again is counted once.
        self._keeping_needs = threading.Lock()
        # Held while one of a session's own records is read and added to.
        self._keeping = threading.Lock()
        # Held while the tree is saved, and all through a landing, which
        # changes the tree and saves it with team memory in one step.
        self._saving = threading.RLock()
        # Held while what waits of a session's work is read and changed.
        self._weighing = threading.RLock()
        self._adding_manager = threading.Lock()
        self._counting = threading.Lock()
        self._clock = clock
        self._stall_poll = stall_poll
        self._stall_watch: asyncio.Task[None] | None = None
        self._remember_lock = threading.Lock()
        self._store: OrgStore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._launcher: TeamLauncher | None = None
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
    def from_config(cls, config: Config) -> Self:
        return cls(
            config.director_repo_path or config.workspace_path,
            worktree_root=config.conversation_worktree_root,
            find_sandbox=find_bwrap,
            remembered=config.conversations_path.parent / _REMEMBERED_FILE,
            find_review_instructions=bugbot.review_instructions,
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
    def team_memory(self) -> GitStore:
        """The team memory of the director's repository.

        Raises:
            Unavailable: The director's directory is not a git repository.
        """
        _ = self.store  # settles which repository the org is in
        if not is_git_repository(self._repo_dir):
            raise Unavailable(
                f"The director has no team memory: {self._repo_dir} is not a git "
                "repository. Point OH_DIRECTOR_REPO_PATH at one."
            )
        return self._memory()

    @property
    def repo_dir(self) -> Path:
        """The repository the org is in."""
        _ = self.store  # settles which repository the org is in
        return self._repo_dir

    def _memory(self) -> GitStore:
        return GitStore(self._repo_dir)

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
        for node in self.store.teams():
            levels[node.id] = levels.get(node.parent_id or "", 0) + 1
        return DirectorSettingsState(
            **self.settings.model_dump(),
            teams=len(levels),
            depth=max(levels.values(), default=0),
            max_concurrent_runs=self._max_concurrent_runs,
        )

    def update_settings(self, update: DirectorSettingsUpdate) -> DirectorSettingsState:
        """Change the limits on new teams. Existing teams are left alone."""
        values = update.model_dump(exclude_none=True)
        changed = self.settings.model_copy(update=values)
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

    def _adapter(
        self, conversation_id: str, offered: Sequence[str]
    ) -> Callable[[AgentBase], AgentBase]:
        """What is changed on an agent as its conversation starts.

        An agent that runs its own tools gets the director's through an MCP
        server and is started by the script that holds its process to its
        team's directory. Any other agent keeps the tools it has and, of
        the director's, gets the ones in ``offered``: an agent derived from
        another carries that agent's, which are another role's for a
        Manager.
        """
        launchers = self._shells()
        launcher = launchers / f"{conversation_id}{_ACP_LAUNCHER}"

        def adapt(agent: AgentBase) -> AgentBase:
            if not isinstance(agent, ACPAgent):
                kept = [
                    tool
                    for tool in agent.tools
                    if tool.name not in DIRECTOR_TOOL_NAMES or tool.name in offered
                ]
                has = {tool.name for tool in kept}
                added = [Tool(name=name) for name in offered if name not in has]
                return agent.model_copy(update={"tools": [*kept, *added]})
            url = self._url()
            if url is None:
                raise Unavailable(
                    "The director does not know the address of its own "
                    "server, which an ACP agent needs to reach its tools"
                )
            if self._bwrap is not None and not launcher.exists():
                logger.warning(
                    "The ACP agent of conversation %s starts with nothing that "
                    "holds its process to its team's folder: %s is missing",
                    conversation_id,
                    launcher,
                )
            return as_org_agent(
                agent,
                tools=tools_server(url, agent_token(self._key(), conversation_id)),
                launcher=launcher if launcher.exists() else None,
                launchers=launchers,
            )

        return adapt

    def _shells(self) -> Path:
        return common_dir(self._repo_dir) / _SHELLS_DIR

    def _merging_root(self) -> Path:
        return self._shells().parent / _MERGING_DIR

    def merge_checkout(self, place_id: str) -> MergeCheckout | None:
        """The merge of a team's work that its reviewer finishes."""
        kept = self._runs().marked(MERGING).get(place_id)
        return MergeCheckout.model_validate_json(kept) if kept else None

    def _drop_merge_checkout(self, place_id: str) -> None:
        merge = self.merge_checkout(place_id)
        if merge is not None:
            discard(self._repo_dir, Path(merge.where))
            self._runs().unmark(MERGING, place_id)

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
        for node in self.store.teams():
            text = f"{node.title}\n{node.boundary.path}".lower()
            if all(word in text for word in words):
                found.append(
                    TeamMatch(id=node.id, title=node.title, path=node.boundary.path)
                )
                if len(found) == limit:
                    break
        return found

    async def settled(self) -> None:
        """Wait until everything asked for so far has been carried out."""
        # Let what other threads asked for reach the queue first.
        await asyncio.sleep(0)
        # What is carried out may ask for more.
        while self._unsettled:
            await self._quiet.wait()
            await asyncio.sleep(0)

    def _conversations_below(self, team: SessionTeam) -> list[str]:
        """The conversations of a team and of every team below it."""
        store = self.store
        teams = [team]
        for found in teams:
            teams.extend(store.below(found.id))
        return [
            conversation_id
            for found in teams
            for agent in found.agents
            for conversation_id in (
                agent.conversation_id,
                agent.hand_over.conversation_id if agent.hand_over else None,
            )
            if conversation_id is not None
        ]

    def _is_stopped(self, conversation_id: str) -> bool:
        """Whether the agent's team, or a team above it, is stopped."""
        store = self.store
        found = store.find_by_conversation(conversation_id)
        if found is None:
            return False
        node = found[0]
        return node.stopped is not None or any(
            above.stopped is not None for above in store.above(node.id)
        )

    def _restart_clocks(self, run: RunState) -> None:
        """Count the time teams show no progress from now: the time an org
        was paused, or its server was not running, is nobody's."""
        now = self._clock().isoformat()
        for place_id, kept in run.marked(PROGRESS).items():
            seen = _progress_of(kept)
            # A team without a task was told of once, and stays told.
            told = {"told": now} if "told" in seen else {}
            again = {"at": now, "head": seen.get("head")}
            run.mark(PROGRESS, place_id, json.dumps(again | told))

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
                self._restart_clocks(self._run)
            return self._run

    def _open_run_state(self) -> RunState:
        if not is_git_repository(self._repo_dir):
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
        self._restart_clocks(run)
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

    def _confine(
        self, place_id: str, role: AgentRole, hand_over: str | None = None
    ) -> ToolParams | None:
        """Write the shell an agent's terminal starts.

        Where the server can, the shell holds the agent to its team's
        boundary. An IC's shell also finds a ``git`` that keeps the agent's
        index apart from those of the agents it shares a checkout with.

        Returns:
            What to start the agent's terminal with, or None when it needs
            no shell of its own.
        """
        place = self.store.place(place_id)
        conversation_id = hand_over or (
            agent_conversation(place, role) if place else None
        )
        if place is None or conversation_id is None:
            return None
        # An IC writes in its boundary from its main conversation alone.
        writes = role == "ic" and hand_over is None
        bash, real_git = shutil.which("bash"), shutil.which("git")
        own_git = writes and os.name != "nt" and bash and real_git
        if self._bwrap is None and not own_git:
            return None
        shells = self._shells()
        git_dir = str(shells.parent)
        children = self.store.below(place_id)
        sub_teams = [child.boundary.path for child in children]
        # Its git commits nothing in a sub-team's directory; it may still
        # write in that of a team that has not started.
        closed = [child.boundary.path for child in children if child.agents]
        rc: Path | None = None
        bin_dir: Path | None = None
        if own_git and real_git:
            directory = place.boundary.path
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
        # An agent's commits need the git directory, but not what the director
        # keeps there, nor the hooks and settings git obeys for everyone.
        private = [shells, *(shells.parent / name for name in _PRIVATE)]
        script = shells / f"{conversation_id}.sh"
        write_shell(
            script,
            shell_script(
                self._bwrap,
                bash or "/bin/sh",
                checkouts=self._checkouts(),
                private=private,
                secret=self._remembered,
                directory=place.boundary.path if writes else None,
                sub_team_directories=closed,
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
                private=private,
                secret=self._remembered,
                directory=place.boundary.path if writes else None,
                sub_team_directories=closed,
                first_on_path=bin_dir,
            ),
        )
        # Only the subprocess terminal starts the shell it is given.
        return {"terminal": {"terminal_type": "subprocess", "shell_path": str(script)}}

    def _confine_all(self) -> None:
        for node in self.store.places():
            for agent in node.agents:
                try:
                    self._confine(node.id, agent.role)
                except (GitError, OSError):
                    logger.exception("Could not write the shell of team %s", node.id)

    def _require_launcher(self) -> TeamLauncher:
        if self._launcher is None:
            raise Unavailable("The director is not running agents on this server")
        return self._launcher

    def _load_store(self) -> OrgStore:
        store = OrgStore()
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
                logger.info("Director loaded %d teams from %s", len(store), repository)
                kept = self._open_run_state()
                try:
                    restore_sessions(store, kept.marked(SESSION))
                finally:
                    kept.close()
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
        _LANE.set(lane)
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

    def next_submission(self) -> int:
        """The place in the order of the work that is submitted now."""
        return self._runs().count("submitted")

    async def _in_base(self, effect: Effect) -> None:
        """Carry ``effect`` out in the lane of what the org shares and wait
        for it: after everything that was asked for there before."""
        if _LANE.get() == BASE_LANE:
            await effect()
            return
        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()

        async def carried_out() -> None:
            try:
                await effect()
            except Exception as error:
                done.set_exception(error)
            else:
                done.set_result(None)

        self._put(carried_out, BASE_LANE)
        await done

    def _begin(self, kind: str, place_id: str, **payload: object) -> None:
        """Record something of several steps the director does for a team,
        then do it. Safe from any thread.

        It is finished after a restart or a pause from the steps that were
        done, so each step must be safe to take twice.
        """
        if self._loop is None:
            raise Unavailable("The director is not running agents on this server")
        intent = self._runs().begin(kind, place_id, payload)
        self._save_soon()
        self._enqueue(lambda: self._carry_out(intent.id), _lane_for(intent))

    def _keep_failure(self, intent: Intent, error: Exception) -> None:
        """Keep a step of the director that failed. The service does."""

    async def _carry_out(self, intent_id: int) -> None:
        run = self._runs()
        if self._paused is not None or intent_id in self._in_flight:
            return
        intent = next(
            (open_ for open_ in run.open_intents() if open_.id == intent_id), None
        )
        if intent is None:
            return
        step = self._steps().get(intent.kind)
        if step is None:
            run.finish(intent.id)
            self._leave_line(intent.node_id)
            return
        self._in_flight.add(intent.id)
        try:
            await step(intent)
        except _Held:
            return
        except Exception as error:
            logger.exception("A director action failed")
            run.finish(intent.id)
            self._keep_failure(intent, error)
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

    def _leave_line(self, place_id: str) -> None:
        if place_id in self._line:
            self._line.remove(place_id)
            moved, self._line_moved = self._line_moved, asyncio.Event()
            moved.set()

    def _schedule_save(self) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._arm_save)

    def _arm_save(self) -> None:
        loop = self._loop
        if loop is None:
            return
        if not self._sessions_asked:
            # What a session has is only here: it is kept at once, not
            # when the tree is next saved.
            self._sessions_asked = True
            self._enqueue(self._keep_sessions)
        if self._save_timer is not None:
            return
        self._save_timer = loop.call_later(
            self._save_delay, lambda: self._enqueue(self._save_due)
        )

    async def _keep_sessions(self) -> None:
        self._sessions_asked = False
        store = self._store
        if store is None:
            return

        def keep() -> None:
            with self._saving:
                self._runs().replace_marks(SESSION, kept_sessions(store))

        try:
            await asyncio.to_thread(keep)
        except (GitError, RunStateError):
            logger.exception("Could not keep the sessions of %s", self._repo_dir)

    async def _base[T](self, work: Callable[[], Awaitable[T]]) -> T:
        """What ``work`` returns, carried out in the lane of what the org
        shares."""
        results: list[T] = []

        async def effect() -> None:
            results.append(await work())

        await self._in_base(effect)
        return results[0]

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
        if store is None:
            return
        try:
            await asyncio.to_thread(self._save_tree, store)
        except GitError:
            logger.exception("Could not save the org tree to %s", self._repo_dir)

    def _commit_landing(
        self, landing: Landing, message: str, tree: bool
    ) -> dict[str, str | None]:
        """Take a landing into team memory, and with ``tree`` save the org
        tree as it is now with it: both refs move together or neither.

        Returns:
            The files that were not written, with the text each has now.

        Raises:
            OperationFailed: Another writer kept moving a ref first.
        """
        memory = self._memory()
        if not tree:
            return memory.take_in(landing, message)
        store = self.store
        with self._saving:
            for _ in range(_LANDING_ATTEMPTS):
                mutate, left = taking_in(landing)
                board = create_org_board(self._repo_dir)
                staged = [
                    made
                    for made in (
                        memory.stage(mutate, message),
                        board.stage(saving(store), f"org: {message}"),
                    )
                    if made is not None
                ]
                if not staged or swap_together(self._repo_dir, staged):
                    self._runs().replace_marks(SESSION, kept_sessions(store))
                    return dict(left)
        raise OperationFailed(
            f"Gave up landing in {self._repo_dir} after {_LANDING_ATTEMPTS} "
            "contended attempts"
        )

    def _save_tree(self, store: OrgStore) -> None:
        with self._saving:
            board = create_org_board(self._repo_dir)
            save_org(board, store)
            run = self._runs()
            # Sessions are this machine's: kept here, not with the org.
            run.replace_marks(SESSION, kept_sessions(store))

    def _wait_for_save(self) -> None:
        with self._saving:
            pass

    def _shown(self, conversation_id: str, status: ConversationExecutionStatus) -> str:
        """An agent's status as the org shows it. One that was running when
        the org was paused is paused, whatever stopping did to it."""
        held = self._paused is not None and conversation_id in self._running
        if held or conversation_id in self._resuming:
            if status != ConversationExecutionStatus.RUNNING:
                return AGENT_PAUSED
        return _AGENT_STATUS.get(status, "idle")

    def _forget_shells(self, team: SessionTeam) -> None:
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

    def _steps(self) -> dict[str, Callable[[Intent], Awaitable[None]]]:
        """What carries out each kind of intent. The service names its own."""
        return {}

    async def _came_back(self) -> None:
        """Take up what the org had begun before the server stopped."""


def _spec_file(directory: str) -> str:
    return f"{directory}/{SPEC_FILE}" if directory else SPEC_FILE


def _contracts_file(directory: str) -> str:
    """The contracts between the sub-teams of ``directory``, in team memory."""
    path = normalize_store_path(directory)
    return f"{path}/{CONTRACTS_FILE}" if path else CONTRACTS_FILE


def _child_node(
    taken: set[str], parent: SessionTeam, child: SplitChild
) -> SessionTeamInput:
    """The team a split makes of ``child`` in the session of ``parent``,
    for a team whose id is not in ``taken`` and is added to it."""
    base = ".".join(
        re.sub(r"[^A-Za-z0-9_-]+", "-", segment) for segment in child.path.split("/")
    )
    team_id, attempt = base, 1
    while team_id in taken:
        attempt += 1
        team_id = f"{base}-{attempt}"
    taken.add(team_id)
    return SessionTeamInput(
        id=session_team_id(team_id, parent.session),
        session=parent.session,
        team_id=team_id,
        parent_id=parent.id,
        title=child.title,
        boundary=child_boundary(child),
        isolation=child.isolation,
        agent_profile=child.agent_profile,
        worktree_path=parent.worktree_path,
        branch=parent.branch,
        agents=[],
        status="idle",
        headline=provisional_summary(child.task).headline,
        spec_summary=_summary_of(child.task),
        since=_instant(),
    )


# Steps whose outcome is in the tree alone. The steps that start agents are
# not among them: their outcome is the conversation.
_SAVED_BEFORE_FORGOTTEN = frozenset(
    {"accept", "finish_split", "request_review", "land_org"}
)


def _lane_for(intent: Intent) -> str:
    if intent.kind == "land_org" or intent.payload.get("base"):
        return BASE_LANE
    if intent.kind in _REVIEW_STEPS:
        return f"review:{intent.node_id}"
    return intent.node_id


def _new_agents(ic_id: UUID) -> list[AgentRef]:
    """The agents a team starts with. Its Manager joins when something is
    first sent to it (see :meth:`DirectorService.tell_manager`)."""
    return [AgentRef(role="ic", conversation_id=str(ic_id), status="idle")]


def _progress_of(kept: str | None) -> dict[str, Any]:
    """What is known of a team's progress: when it last got anywhere
    (``at``), the last commit in its directory then (``head``), and when
    the IC above was last told that it shows none (``told``)."""
    try:
        seen = json.loads(kept) if kept else {}
    except ValueError:
        return {}
    return seen if isinstance(seen, dict) else {}


def _base_of(workspace: str) -> str:
    return head_of(Path(workspace)) or ""


def _instant() -> str:
    """Now, fine enough to put what two agents record in order."""
    return datetime.now(UTC).isoformat()


def _branch_of(workspace: str) -> str:
    try:
        return git(Path(workspace), ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    except (GitError, OSError):
        return ""
