"""What every part of the director's service works on: the org tree, the
team memory, the settings, the queue of steps and the agents' shells."""

import asyncio
import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Self
from uuid import uuid4

from openhands.agent_server.config import Config
from openhands.agent_server.director import bugbot
from openhands.agent_server.director.agent_types import (
    DEFAULT_TYPE,
    TeamTemplate,
)
from openhands.agent_server.director.documents.store import (
    DocumentStore,
    RecordDocumentStore,
    TeamRef,
)
from openhands.agent_server.director.documents.team_documents import (
    TeamDocuments,
)
from openhands.agent_server.director.documents.tree import TeamTree
from openhands.agent_server.director.environment import Checkouts, CodingEnvironment
from openhands.agent_server.director.environment.checkouts import MergeCheckout
from openhands.agent_server.director.environment.git import GitCodingEnvironment
from openhands.agent_server.director.environment.records import (
    RecordStore,
)
from openhands.agent_server.director.intents import Accept, LandOrg
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import (
    OrgSummary,
    TeamMatch,
)
from openhands.agent_server.director.org_persistence import (
    load_org,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
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
from openhands.agent_server.director.service.team_report_service import (
    TeamReportService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.agent_server.director.service.work_queue_service import (
    BASE_LANE,
    ORG_LANE,
    WorkQueueService,
)
from openhands.agent_server.director.session_state import (
    restore_sessions,
)
from openhands.agent_server.director.split import (
    clean_path,
)
from openhands.sdk.errors import (
    DomainError,
    Unavailable,
    Unprocessable,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

ROOT_ID = "root"
# How long tree changes are gathered before they are saved as one commit.
SAVE_DELAY_SECONDS = 2.0
# How long to wait before starting an agent again when the server is full.
RUN_LIMIT_RETRY_SECONDS = 5.0
# How often the director looks for teams that show no progress.
STALL_POLL_SECONDS = 60.0
# How long a model may take to name a team before the team keeps the name
# it has; a session's first IC is not briefed until then.
SUMMARY_TIMEOUT_SECONDS = 20.0
# How long a model may take over a team's diff, or over an agent's reports.
MODEL_CALL_TIMEOUT_SECONDS = 120.0
# A team whose work is sent back this many times is stopped.
# Where what an IC left uncommitted in the directory of a sub-team with a
# checkout of its own is kept: under `left`, what went to the team as it
# started; under `aside`, what was in the way of the team's accepted work.
LEFT_REFS = "refs/agents/left"
ASIDE_REFS = "refs/agents/aside"
# After a conversation id: the script an ACP agent's process is started by.
_REMEMBERED_FILE = "director.json"
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


# How long a message waits for an IC that submitted to end its turn.
_REST_CHECKS = 600
_REST_CHECK_SECONDS = 0.05


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
        self._stop_guard: Callable[[], None] | None = None
        self._stop_resume: Callable[[], None] | None = None
        self._lock = threading.Lock()
        self._clock = clock
        self._run_limit_retry = run_limit_retry
        self.environment: CodingEnvironment = GitCodingEnvironment()
        self.checkouts: Checkouts = self.environment.checkouts
        self.org_state = OrgStateService(
            repo_dir=lambda: self.repo_dir, clock=clock, environment=self.environment
        )
        self.start_line = StartLineService(
            org=lambda: self.org, org_state=self.org_state, retry=run_limit_retry
        )
        self.session_drafts = SessionDraftService(run_state=self.org_state.run_state)
        self.team_reports = TeamReportService(
            org=lambda: self.org,
            team_memory=self._memory,
            session_drafts=self.session_drafts,
        )
        self.documents: DocumentStore = RecordDocumentStore(
            self._memory,
            self.org_state.run_state,
            self._note_change,
            self.team_reports,
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
            environment=self.environment,
        )
        self.org_settings = OrgSettingsService(
            environment=self.environment,
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
        self._stall_poll = stall_poll
        self._stall_timer: asyncio.Task[None] | None = None
        self._store: OrgStore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._launcher: TeamLauncher | None = None
        self.work_queue = WorkQueueService(default_lane=ORG_LANE)
        self.org_save = OrgSaveService(
            environment=self.environment,
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
    def team_memory(self) -> RecordStore:
        """The team memory of the director's repository.

        Raises:
            Unavailable: The director's directory is not a git repository.
        """
        _ = self.org  # settles which repository the org is in
        if not self.checkouts.is_repository(self._repo_dir):
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

    def _memory(self) -> RecordStore:
        return self.environment.team_memory(self._repo_dir)

    def summary(self) -> OrgSummary:
        store = self.org
        return OrgSummary(
            root_id=store.root_id,
            node_count=len(store),
            repo_path=str(self._repo_dir),
            directory=self._home(),
            is_git_repository=self.checkouts.is_repository(self._repo_dir),
            boundary_enforcement=(
                "sandbox" if self.confinement.is_sandboxed else "checks"
            ),
            paused=self.org_state.paused if self.org_state.run_state() else None,
            merging=self.org_state.run_state().get("merge") is not None,
        )

    def _note_change(self, session: str) -> None:
        """Remember that a session whose work is accepted has changed
        something since: the org takes it only when it is submitted again."""
        top = next(iter(self.org.session(session)), None)
        if top is not None and top.review is not None:
            if top.review.state == "accepted" and self.org.opens_session(top):
                self._changed_since_accept.setdefault(session, False)

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
            self.checkouts.drop_merge(self._repo_dir, Path(merge.where))
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

    def _repository(self) -> Path:
        """The repository the org is in, or would be in.

        Raises:
            Unavailable: The director's directory is not in a git repository.
        """
        if self.org.root_id is not None:
            return self._repo_dir
        if not self.checkouts.is_repository(self._directory):
            raise Unavailable(
                f"An org needs a git repository; {self._directory} is not one"
            )
        return self.checkouts.root_of(str(self._directory))

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
        top = self.checkouts.root_of(str(self._repo_dir))
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
        top = self.checkouts.root_of(str(self._repo_dir))
        if not target.is_relative_to(top):
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
        while repository is not None and self.checkouts.is_repository(repository):
            if not climb and not (repository / ".git").exists():
                return False
            if load_org(self.environment.org_tree(repository), store):
                self._repo_dir = self.checkouts.root_of(str(repository))
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
            repository = self.checkouts.enclosing(repository)
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

    async def _came_back(self) -> None:
        """Take up what the org had begun before the server stopped."""


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
