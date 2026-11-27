"""What every part of the director's service works on: the org tree, the
team memory, the settings, the queue of steps and the agents' shells."""

import asyncio
import os
import re
import threading
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Self
from uuid import UUID, uuid4

from openhands.agent_server.config import Config
from openhands.agent_server.director import bugbot
from openhands.agent_server.director.agent_types import (
    DEFAULT_TYPE,
    TeamTemplate,
)
from openhands.agent_server.director.confinement import (
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
    enclosing_repository,
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
    AgentRef,
    HandOverMode,
    OrgSummary,
    Report,
    SessionTeam,
    SessionTeamInput,
    SplitChild,
    TeamMatch,
)
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    load_org,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    session_team_id,
    team_ref,
)
from openhands.agent_server.director.remembered import RememberedFile
from openhands.agent_server.director.run_state import (
    MERGING,
    SESSION,
    Intent,
)
from openhands.agent_server.director.sandbox import (
    find_bwrap,
)
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.model_call_service import (
    ModelCallService,
)
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.start_line_service import StartLineService
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
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import (
    DomainError,
    Unavailable,
    Unprocessable,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

ROOT_ID = "root"
# How a team that was set up and given no task reads to the IC above.
NOT_STARTED = "not_started"
# What follows a team's purpose in its task once something started it.
_STARTED_WITH = "\n\nWhat your team is started with:\n"
# How many of the latest events of its sub-teams an agent reads at once.
# How many of the human's direct requests to the teams below an IC is told.
ASKED_TOLD = 10
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
# After a conversation id: the script an ACP agent's process is started by.
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
# of accepted work, and what is settled with them.
ORG_WAITS = "org-waits"
# The escalation that asks the human whether a session that reached its
# token limit goes on.
LIMIT_WAITS = "limit-reached"


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
        self.remembered = RememberedFile(remembered)
        self._find_review_instructions = find_review_instructions
        # The public skills that were looked for, by name; '' for one that
        # could not be loaded.
        self._review_instructions: dict[str, str] = {}
        self._review_instructions_ready: asyncio.Task[None] | None = None
        self._loading_review_instructions = threading.Lock()
        # The sessions that changed something of the org's after their
        # work was accepted, and whether their IC was told to submit it.
        self._changed_since_accept: dict[str, bool] = {}
        # The conversations a resume is about to run again.
        self._resuming: set[str] = set()
        self._stop_guard: Callable[[], None] | None = None
        self._stop_resume: Callable[[], None] | None = None
        self._lock = threading.Lock()
        self._clock = clock
        self._run_limit_retry = run_limit_retry
        self.org_state = OrgStateService(repo_dir=lambda: self.repo_dir, clock=clock)
        self.start_line = StartLineService(
            org=lambda: self.org, org_state=self.org_state, retry=run_limit_retry
        )
        self.session_drafts = SessionDraftService(run_state=self.org_state.run_state)
        self.documents: DocumentStore = GitDocumentStore(
            self._memory, self.org_state.run_state, self._document_changed, self
        )
        self.tree: TeamTree = _TeamsOfTheOrg(lambda: self.org)
        self.team_types = TeamTypeService(
            config_file=config_file,
            documents=self.documents,
            tree=self.tree,
            org=lambda: self.org,
        )
        self.confinement = AgentConfinementService(
            repo_dir=lambda: self._repo_dir,
            worktree_root=worktree_root,
            find_sandbox=find_sandbox,
            remembered=self.remembered,
            server_url=server_url,
            org=lambda: self.org,
            team_types=self.team_types,
            run_state=self.org_state.run_state,
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
        self._stall_poll = stall_poll
        self._stall_timer: asyncio.Task[None] | None = None
        self._store: OrgStore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._launcher: TeamLauncher | None = None
        self.work_queue = WorkQueueService(default_lane=ORG_LANE)
        self.org_save = OrgSaveService(
            org=lambda: self.org,
            repo_dir=lambda: self._repo_dir,
            team_memory=self._memory,
            run_state=self.org_state.run_state,
            work_queue=self.work_queue,
            delay=save_delay,
        )
        self.model_calls = ModelCallService(
            launcher=self._require_launcher,
            sessions=lambda: self._store.sessions() if self._store else [],
            call_timeout=model_call_timeout,
            summary_timeout=summary_timeout,
        )
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

    def summary(self) -> OrgSummary:
        store = self.org
        return OrgSummary(
            root_id=store.root_id,
            node_count=len(store),
            repo_path=str(self._repo_dir),
            directory=self._home(),
            is_git_repository=is_git_repository(self._repo_dir),
            boundary_enforcement=(
                "sandbox" if self.confinement.is_sandboxed else "checks"
            ),
            paused=self.org_state.paused if self.org_state.run_state() else None,
            merging=self.org_state.run_state().get("merge") is not None,
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

    def merge_checkout(self, place_id: str) -> MergeCheckout | None:
        """The merge of a team's work that its reviewer finishes."""
        kept = self.org_state.run_state().marked(MERGING).get(place_id)
        return MergeCheckout.model_validate_json(kept) if kept else None

    def _drop_merge_checkout(self, place_id: str) -> None:
        merge = self.merge_checkout(place_id)
        if merge is not None:
            discard(self._repo_dir, Path(merge.where))
            self.org_state.run_state().unmark(MERGING, place_id)

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
        run = self.org_state.run_state()
        run.put("instance", self._instance())
        run.put("clean", "0")

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
                kept = self.org_state.open(self._repo_dir)
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
        return self.org_state.run_state().count("submitted")

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
        held = (
            self.org_state.paused is not None
            and conversation_id in self.org_state.running
        )
        if held or conversation_id in self._resuming:
            if status != ConversationExecutionStatus.RUNNING:
                return AGENT_PAUSED
        return _AGENT_STATUS.get(status, "idle")

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


def _instant() -> str:
    """Now, fine enough to put what two agents record in order."""
    return datetime.now(UTC).isoformat()
