"""What every part of the director's service works on: the org tree, the
team memory, the settings, the queue of steps and the agents' shells."""

import asyncio
import json
import os
import re
import secrets
import sqlite3
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self
from uuid import UUID, uuid4

from openhands.agent_server.config import Config
from openhands.agent_server.conversation_service import ToolParams
from openhands.agent_server.director import bugbot
from openhands.agent_server.director.acp import (
    conversation_of,
)
from openhands.agent_server.director.agent_types import (
    DEFAULT_TYPE,
    TeamTemplate,
)
from openhands.agent_server.director.boundary import scope_within
from openhands.agent_server.director.confinement import (
    Confinement,
    checkout_root,
    is_in,
)
from openhands.agent_server.director.documents.store import (
    DocumentStore,
    GitDocumentStore,
    TeamRef,
)
from openhands.agent_server.director.documents.team_documents import (
    TeamDocuments,
)
from openhands.agent_server.director.documents.tree import TeamTree
from openhands.agent_server.director.git import (
    GitError,
    enclosing_repository,
    git,
    head_of,
    is_git_repository,
)
from openhands.agent_server.director.git_store import (
    GitStore,
)
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.intents import Accept, LandOrg
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.merge_checkout import MergeCheckout, discard
from openhands.agent_server.director.models import (
    AgentKind,
    AgentRef,
    FormerTeam,
    HandOverMode,
    OrgStateEvent,
    OrgSummary,
    PauseReason,
    RecoveryItem,
    Report,
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
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    is_division_lead,
    session_team_id,
    team_ref,
)
from openhands.agent_server.director.remembered import RememberedFile
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
    find_bwrap,
)
from openhands.agent_server.director.scope import MergeArea, Scope
from openhands.agent_server.director.service.model_call_service import (
    ModelCallService,
)
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.stall_watch_service import progress_of
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.agent_server.director.service.work_queue_service import (
    WorkQueueService,
)
from openhands.agent_server.director.session_state import (
    restore_sessions,
)
from openhands.agent_server.director.split import (
    child_boundary,
    clean_path,
)
from openhands.sdk.agent.base import AgentBase
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import (
    DomainError,
    NotFound,
    Unavailable,
    Unprocessable,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

ROOT_ID = "root"
# Tools that start agents nothing holds to the team's folder. An org's
# agent hands work on by proposing sub-teams.
_SUB_AGENT_TOOLS = (
    "task_tool_set",
    "task",
    "delegate",
    "workflow_tool_set",
    "workflow",
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
# In the git directory, and not for an agent's shell to write.
# In the repository's git directory: the merges reviewers finish.
_MERGING_DIR = "director-merging"
# After a conversation id: the script an ACP agent's process is started by.
_REMEMBERED_FILE = "director.json"
# Set by the server's entry point to the address it is bound to.
_INTERNAL_SERVER_URL_ENV = "OH_INTERNAL_SERVER_URL"
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
# of accepted work, and what is settled with them.
ORG_WAITS = "org-waits"
# The escalation that asks the human whether a session that reached its
# token limit goes on.
LIMIT_WAITS = "limit-reached"
# How much of an error is kept as the reason an agent failed.
_REASON_KEPT = 600


def failure_waits(role: str) -> str:
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


def conversations_of(teams: Iterable[SessionTeam]) -> list[str]:
    """Every conversation of ``teams``' agents, hand-over ones included."""
    return [
        conversation_id
        for team in teams
        for agent in team.agents
        for conversation_id in (
            agent.conversation_id,
            agent.hand_over.conversation_id if agent.hand_over else None,
        )
        if conversation_id is not None
    ]


def hand_over_of(place: SessionTeam) -> HandOverMode | None:
    """The hand-over mode of the team's IC, if it is or was in it."""
    for agent in place.agents:
        if agent.hand_over is not None:
            return agent.hand_over
    return None


def _agent_title(
    team: str, name: str, session: str = "", *, division_lead: bool = False
) -> str:
    """What an agent's conversation is called. The session is named unless
    it is called what its team is, and the agents a session's work is
    reviewed by are called its Division Leads.

    Args:
        name: What the agent's member is called.
    """
    where = f"{team} · {session}" if session and session != team else team
    return f"{where} ({'Division Lead ' + name if division_lead else name})"


def _summary_of(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line[:200]


def apart(team: SessionTeam, above: SessionTeam) -> bool:
    """Whether a team works in a checkout of its own, not in the one of
    the team above."""
    return checkout_root(team.worktree_path) != checkout_root(above.worktree_path)


class _TeamsOfTheOrg:
    """Where a team stands among the others: the org's tree for a team of
    the org, its session's for the work of one in a session."""

    def __init__(self, org: Callable[[], OrgStore]) -> None:
        self._org = org

    def above(self, team: TeamRef) -> TeamRef | None:
        store = self._org()
        node = store.get(team.id)
        above = store.get(node.parent_id) if node and node.parent_id else None
        return team_ref(above) if above else None

    def below(self, team: TeamRef) -> list[TeamRef]:
        store = self._org()
        if store.place(team.id) is not None:
            return [team_ref(place) for place in store.below(team.id)]
        return [team_ref(node) for node in store.sub_teams(team.id)]


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
        config_file: Path | None = None,
    ) -> None:
        """
        Args:
            config_file: The org's ``director.yaml``: its settings as a
                person writes them. The org's own agent types are in the
                folder ``types`` beside it. None: the director's defaults
                and the types it ships.
            clock: The time, for telling how long a team has shown no
                progress.

        Raises:
            InvalidOrgConfig: ``config_file`` does not hold settings.
            stall_poll: How many seconds lie between two looks for teams
                that show no progress.
            server_url: Where this server answers, for agents that reach the
                director's tools from a process of their own; None for the
                address the server is bound to.
            max_concurrent_runs: How many conversations the server runs at
                once, to tell the human; None when not known.
            find_review_instructions: The code review skill given to the
                IC above with a team's write-up, or None to leave it out.
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
        self.remembered = RememberedFile(remembered)
        self._find_review_instructions = find_review_instructions
        # The public skills that were looked for, by name; '' for one that
        # could not be loaded.
        self._review_instructions: dict[str, str] = {}
        self._review_instructions_ready: asyncio.Task[None] | None = None
        self._loading_review_instructions = threading.Lock()
        self._run: RunState | None = None
        self._run_of: Path | None = None
        self._paused: PauseReason | None = None
        self._state_listeners: list[Callable[[OrgStateEvent], None]] = []
        # The messages being delivered right now, so that resuming does
        # not deliver one a second time.
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
        self._stop_resume: Callable[[], None] | None = None
        self._run_limit_retry = run_limit_retry
        self._lock = threading.Lock()
        # Held from looking a need up in the record to keeping it, so that
        # an agent that tries the same file again is counted once.
        self._keeping_needs = threading.Lock()
        self.session_drafts = SessionDraftService(run_state=self._runs)
        self.documents: DocumentStore = GitDocumentStore(
            self._memory, self._runs, self._document_changed, self
        )
        self.tree: TeamTree = _TeamsOfTheOrg(lambda: self.org)
        self.team_types = TeamTypeService(
            config_file=config_file,
            documents=self.documents,
            tree=self.tree,
            org=lambda: self.org,
        )
        self.org_settings = OrgSettingsService(
            config_file=config_file,
            remembered=self.remembered,
            team_types=self.team_types,
            tree=self.tree,
            org=lambda: self.org,
            repo_dir=lambda: self.repo_dir,
            max_concurrent_runs=max_concurrent_runs,
        )
        # Held while what waits of a session's work is read and changed.
        self._weighing = threading.RLock()
        self._adding_manager = threading.Lock()
        self._counting = threading.Lock()
        self._clock = clock
        self._stall_poll = stall_poll
        self._stall_timer: asyncio.Task[None] | None = None
        self._store: OrgStore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._launcher: TeamLauncher | None = None
        self._server_url = server_url
        self._tool_key: bytes | None = None
        self.work_queue = WorkQueueService(default_lane=ORG_LANE)
        self.org_save = OrgSaveService(
            org=lambda: self.org,
            repo_dir=lambda: self._repo_dir,
            team_memory=self._memory,
            run_state=self._runs,
            work_queue=self.work_queue,
            delay=save_delay,
        )
        self.model_calls = ModelCallService(
            launcher=self._require_launcher,
            sessions=lambda: self._store.sessions() if self._store else [],
            call_timeout=model_call_timeout,
            summary_timeout=summary_timeout,
        )
        # Set when a conversation stopped running, so a start may fit now.
        self._room = asyncio.Event()
        # Held by the start that waits for room, so that starts are served
        # in the order they began to wait.
        self._start_gate = asyncio.Lock()
        self._stop_status: Callable[[], None] | None = None
        self._create_lock = asyncio.Lock()
        self._first_message_lock = asyncio.Lock()
        # Held by a pull, a push or a step of a merge, one at a time.
        self._sync_lock = asyncio.Lock()
        self._resume_lock = asyncio.Lock()

    def review_instructions(self, kind: TeamTemplate | None = None) -> str:
        """What an IC reviewing a team below checks its work by: the skill
        the team's type names, then the type's own checklist.

        A skill is loaded once. One that cannot be loaded is left out, and
        the review request stands without it.

        Args:
            kind: The type of the team whose work is reviewed; the default
                type when None.
        """
        kind = kind or self.team_types.agent_types.get(DEFAULT_TYPE)
        if kind is None:
            return ""
        parts: list[str] = []
        skill = kind.review.checklist_skill
        if skill:
            # One fetch at a time: two would clone the same repository.
            with self._loading_review_instructions:
                if skill not in self._review_instructions:
                    found = (
                        self._find_review_instructions()
                        if skill in bugbot.CODE_REVIEW_SKILLS
                        else bugbot.public_skill((skill,))
                    )
                    self._review_instructions[skill] = found.strip() if found else ""
            parts.append(self._review_instructions[skill])
        parts.append(kind.review.checklist.strip())
        return "\n\n".join(part for part in parts if part)

    def _load_review_instructions(self) -> None:
        """Fetch what reviewers of every type check work by, so that it is
        there when a review is requested. Safe from any thread."""
        try:
            for kind in self.team_types.agent_types.values():
                self.review_instructions(kind)
        except Exception:
            logger.exception("Could not load the review instructions")

    @classmethod
    def from_config(cls, config: Config) -> Self:
        return cls(
            config.director_repo_path or config.workspace_path,
            worktree_root=config.conversation_worktree_root,
            find_sandbox=find_bwrap,
            remembered=config.conversations_path.parent / _REMEMBERED_FILE,
            find_review_instructions=bugbot.review_instructions,
            config_file=config.director_config_path,
            max_concurrent_runs=config.max_concurrent_runs,
            server_url=config.director_server_url,
        )

    @property
    def org(self) -> OrgStore:
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
        _ = self.org  # settles which repository the org is in
        if not is_git_repository(self._repo_dir):
            raise Unavailable(
                f"The director has no team memory: {self._repo_dir} is not a git "
                "repository. Point OH_DIRECTOR_REPO_PATH at one."
            )
        return self._memory()

    @property
    def repo_dir(self) -> Path:
        """The repository the org is in."""
        _ = self.org  # settles which repository the org is in
        return self._repo_dir

    def _memory(self) -> GitStore:
        return GitStore(self._repo_dir)

    def entries(self, team: TeamRef, name: str) -> list[Report]:
        """What was added to a log of a team; the records keep it."""
        raise NotImplementedError

    def append(self, team: TeamRef, name: str, entry: Report) -> None:
        """Add to a log of a team; the records keep it."""
        raise NotImplementedError

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
        store = self.org
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

    def _document_changed(self, session: str) -> None:
        """A session wrote or removed a document."""

    @property
    def team_documents(self) -> TeamDocuments:
        """The documents of the org as each team reads, writes and leaves
        them."""
        return TeamDocuments(
            self.documents, self.team_types.protocols, self.org_settings.org_kind
        )

    def profile_exists(self, name: str) -> bool:
        """Whether the server has an agent profile of that name. Called
        from an agent's tool, off the loop; where it cannot be asked, the
        answer is yes and the team's start says otherwise."""
        launcher, loop = self._launcher, self._loop
        if launcher is None or loop is None:
            return True
        try:
            if asyncio.get_running_loop() is loop:
                return True
        except RuntimeError:
            pass
        asked = asyncio.run_coroutine_threadsafe(launcher.profile_kind(name), loop)
        return asked.result(timeout=30) is not None

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
        """Where an agent that runs its own tools reaches this server: the
        address it was configured with, or the one it is bound to."""
        return self._server_url or os.environ.get(_INTERNAL_SERVER_URL_ENV)

    def _key(self) -> bytes:
        """What the tokens of the org's agents are signed with. It is kept
        beside the server's conversations, never in the repository."""
        with self._lock:
            if self._tool_key is None:
                kept = self.remembered.read().get("tool_key")
                if not isinstance(kept, str):
                    kept = secrets.token_hex(32)
                    self.remembered.keep(tool_key=kept)
                self._tool_key = kept.encode()
            return self._tool_key

    def _confinement(self) -> Confinement:
        """What holds the org's agents to their scopes, as the server is
        now."""
        return Confinement(
            repo_dir=self._repo_dir,
            checkouts=tuple(self._checkouts()),
            bwrap=self._bwrap,
            secret=self.remembered.path,
            server_url=self._url(),
            key=self._key,
        )

    def scope_of(
        self, place: SessionTeam, role: str, hand_over: str | None = None
    ) -> Scope:
        """What an agent of ``place`` may write: the one place the org
        decides what its agents are held to.

        Args:
            hand_over: The IC's hand-over conversation, when the scope is
                that conversation's and not the IC's own.
        """
        children = self.org.below(place.id)
        kind = self.team_types.type_of(place)
        reviewer = self.team_types.team(place).reviewer
        # A team that has not started holds no one out of its directory:
        # the IC above may finish what it has there.
        started = [child for child in children if child.agents]
        merges = []
        for team_id, kept in self._runs().marked(MERGING).items():
            merge = MergeCheckout.model_validate_json(kept)
            team = self.org.place(team_id)
            merges.append(
                MergeArea(
                    where=merge.where,
                    files=tuple(merge.files),
                    own=role == reviewer
                    and team is not None
                    and team.parent_id == place.id,
                )
            )
        within = scope_within(
            place,
            started,
            # A hand-over conversation is no agent of the place.
            checkout="" if hand_over else place.worktree_path,
            edits=role == kind.writer,
        )
        return replace(
            within,
            # It writes in its boundary from its main conversation alone;
            # at a Division Lead place it only reviews, in the human's own
            # checkout.
            writes=role == kind.writer
            and hand_over is None
            and not is_division_lead(place),
            aside=hand_over is not None,
            # The repository's own checkout is the human's, on whatever
            # branch they have it.
            branch=place.branch or ""
            if place.worktree_path
            and checkout_root(place.worktree_path) != checkout_root(str(self._repo_dir))
            else "",
            # Its git commits nothing in a sub-team's directory; it may still
            # write in that of a team that has not started.
            hidden=tuple(child.boundary.path for child in children),
            merges=tuple(merges),
            merge_root=str(self._merges_for(place.id))
            if role == reviewer and hand_over is None
            else "",
        )

    def with_tools_again(self, conversation_id: UUID, agent: AgentBase) -> AgentBase:
        """The saved agent of a conversation that is loaded again, able to
        reach the director's tools. Safe from any thread."""
        return self._confinement().with_tools_again(str(conversation_id), agent)

    def _adapter(
        self, conversation_id: str, offered: Sequence[str]
    ) -> Callable[[AgentBase], AgentBase]:
        """What is changed on an agent as its conversation starts: of the
        director's tools it gets the ones in ``offered``, and the tools
        that start agents nothing holds are taken from it."""
        return self._confinement().adapter(
            conversation_id,
            offered,
            known=self.team_types.tool_names,
            never=_SUB_AGENT_TOOLS,
        )

    def _merging_root(self) -> Path:
        return self._confinement().shells.parent / _MERGING_DIR

    def _merges_for(self, place_id: str) -> Path:
        """Where the merges the reviewer of a team finishes are laid out:
        one folder for each reviewer, which its shell alone may write."""
        return self._merging_root() / place_id.replace("@", "-")

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
        return self._confinement().enforcement(kind == "acp")

    def search(self, query: str, limit: int) -> list[TeamMatch]:
        """Teams whose title or directory has every word of ``query``."""
        words = query.lower().split()
        found: list[TeamMatch] = []
        if not words:
            return found
        for node in self.org.teams():
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
        await self.work_queue.settled()

    def _conversations_below(self, team: SessionTeam) -> list[str]:
        """The conversations of a team and of every team below it."""
        store = self.org
        teams = [team]
        for found in teams:
            teams.extend(store.below(found.id))
        return conversations_of(teams)

    def _is_stopped(self, conversation_id: str) -> bool:
        """Whether the agent's team, or a team above it, is stopped."""
        store = self.org
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
            seen = progress_of(kept)
            # A team without a task was told of once, and stays told.
            told = {"told": now} if "told" in seen else {}
            again = {"at": now, "head": seen.get("head"), "checked": now}
            run.mark(PROGRESS, place_id, json.dumps(again | told))

    def _repository(self) -> Path:
        """The repository the org is in, or would be in.

        Raises:
            Unavailable: The director's directory is not in a git repository.
        """
        if self.org.root_id is not None:
            return self._repo_dir
        if not is_git_repository(self._directory):
            raise Unavailable(
                f"An org needs a git repository; {self._directory} is not one"
            )
        return checkout_root(str(self._directory))

    def _merging(self) -> dict[str, Any] | None:
        kept = self._runs().get("merge")
        return json.loads(kept) if kept else None

    def _runs(self) -> RunState:
        """What the director is in the middle of, in the org's repository."""
        _ = self.org  # settles which repository the org is in
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
            return RunState(self._confinement().shells.parent / _RUN_STATE)
        except (GitError, OSError, RunStateError, sqlite3.Error):
            logger.exception("Run state of %s is kept in memory only", self._repo_dir)
            return RunState(None)

    def _hold(self, reason: PauseReason) -> None:
        self._runs().put("paused", reason)
        self._paused = reason
        # Those that wait for room to start see it at once.
        self._room.set()
        self._line_moved.set()
        self._line_moved = asyncio.Event()
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
        if self.remembered.path is None:
            return ""
        kept = self.remembered.read().get("instance")
        if not isinstance(kept, str):
            kept = uuid4().hex
            self.remembered.keep(instance=kept)
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
        top = checkout_root(str(self._repo_dir))
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
        top = checkout_root(str(self._repo_dir))
        if not is_in(target, top):
            raise Unprocessable(f"{path} is outside the org's repository {top}")
        relative = target.relative_to(top).as_posix()
        return "" if relative == "." else relative

    def _confine(
        self, place_id: str, role: str | None = None, hand_over: str | None = None
    ) -> ToolParams | None:
        """Write the shell an agent's terminal starts.

        Args:
            role: The agent's member; None for the one that works in the
                team's folder.

        Returns:
            What to start the agent's terminal with, or None when it needs
            no shell of its own.
        """
        place = self.org.place(place_id)
        if place is None:
            return None
        team = self.team_types.team(place)
        role = role or team.writer
        conversation_id = hand_over or team.conversation(role)
        if conversation_id is None:
            return None
        return self._confinement().confine(
            conversation_id, self.scope_of(place, role, hand_over)
        )

    def _confine_all(self) -> None:
        for node in self.org.places():
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
        store = OrgStore(self.team_types.type_of)
        remembered = self._remembered_repo()
        if remembered is None or not self._load_from(remembered, store):
            self._load_from(self._directory, store)
        store.on_change(self.org_save.tree_changed)
        if self._loop is not None:
            self.work_queue.enqueue(self._came_back)
        return store

    def _load_from(
        self, directory: Path, store: OrgStore, *, climb: bool = True
    ) -> bool:
        """Load the org saved in the repository of ``directory``.

        With ``climb``, a repository that has none is followed by the nearest
        one above that has it as a submodule. Without it, only a ``.git`` in
        ``directory`` itself counts.

        Returns:
            Whether one was found. The director then works in its repository.
        """
        repository: Path | None = directory
        while repository is not None and is_git_repository(repository):
            if not climb and not (repository / ".git").exists():
                return False
            if load_org(create_org_board(repository), store):
                self._repo_dir = checkout_root(str(repository))
                logger.info("Director loaded %d teams from %s", len(store), repository)
                kept = self._open_run_state()
                try:
                    # None of another server's run state is used, so its
                    # sessions are not taken up without it either.
                    if kept.get("instance") in (None, self._instance()):
                        restore_sessions(store, kept.marked(SESSION))
                finally:
                    kept.close()
                return True
            if not climb:
                return False
            repository = enclosing_repository(repository)
        return False

    def _join(self, workspace: Path, climb: bool = True) -> bool:
        """Take up the org saved at ``workspace``, or above it when ``climb``."""
        if not self._load_from(workspace, self.org, climb=climb):
            return False
        self.remembered.keep(repo_path=str(self._repo_dir))
        return True

    def _remembered_repo(self) -> Path | None:
        repo_path = self.remembered.read().get("repo_path")
        return Path(repo_path) if isinstance(repo_path, str) else None

    def next_submission(self) -> int:
        """The place in the order of the work that is submitted now."""
        return self._runs().count("submitted")

    def _leave_line(self, place_id: str) -> None:
        if place_id in self._line:
            self._line.remove(place_id)
            moved, self._line_moved = self._line_moved, asyncio.Event()
            moved.set()

    async def _base[T](self, work: Callable[[], Awaitable[T]]) -> T:
        """What ``work`` returns, carried out in the lane of what the org
        shares."""
        results: list[T] = []

        async def effect() -> None:
            results.append(await work())

        await self.work_queue.run_in(BASE_LANE, effect)
        return results[0]

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
        self._confinement().forget(
            [agent.conversation_id for agent in team.agents if agent.conversation_id],
            Path(team.worktree_path),
        )

    async def _came_back(self) -> None:
        """Take up what the org had begun before the server stopped."""


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
        agent_type=child.agent_type,
        worktree_path=parent.worktree_path,
        branch=parent.branch,
        agents=[],
        status="idle",
        headline=provisional_summary(child.task).headline,
        summary=_summary_of(child.task),
        since=_instant(),
    )


# Steps whose outcome is in the tree alone. The steps that start agents are
# not among them: their outcome is the conversation.
SAVED_BEFORE_FORGOTTEN = frozenset(
    {"accept", "finish_split", "request_review", "land_org"}
)


def lane_for(intent: Intent) -> str:
    payload = intent.payload
    if isinstance(payload, LandOrg) or (isinstance(payload, Accept) and payload.base):
        return BASE_LANE
    if intent.kind in _REVIEW_STEPS:
        return f"review:{intent.node_id}"
    return intent.node_id


def _new_agents(ic_id: UUID, member: str) -> list[AgentRef]:
    """The agents a team starts with: the member that works in its folder.
    Any other joins when something is first sent to it (see
    :meth:`DirectorService.tell`)."""
    return [AgentRef(role=member, conversation_id=str(ic_id), status="idle")]


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
