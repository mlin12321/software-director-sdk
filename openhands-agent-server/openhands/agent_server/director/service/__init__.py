"""The director's state for one server: an org tree and its team_memory."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Self
from uuid import UUID

from openhands.agent_server.config import Config
from openhands.agent_server.director import bugbot
from openhands.agent_server.director.documents.store import (
    DocumentStore,
    RecordDocumentStore,
)
from openhands.agent_server.director.documents.tree import TeamTree
from openhands.agent_server.director.environment import Checkouts, CodingEnvironment
from openhands.agent_server.director.environment.git import GitCodingEnvironment
from openhands.agent_server.director.intents import Accept, LandOrg
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.org_store import (
    OrgStore,
    hand_over_of,
)
from openhands.agent_server.director.recovery import (
    has_work,
)
from openhands.agent_server.director.remembered import RememberedFile
from openhands.agent_server.director.run_state import (
    Intent,
)
from openhands.agent_server.director.runtime import bind_director
from openhands.agent_server.director.sandbox import (
    find_bwrap,
)
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.agent_launch_service import (
    AgentLaunchService,
)
from openhands.agent_server.director.service.check_count_service import (
    CheckCountService,
)
from openhands.agent_server.director.service.message_delivery_service import (
    MessageDeliveryService,
)
from openhands.agent_server.director.service.model_call_service import (
    ModelCallService,
)
from openhands.agent_server.director.service.org_load_service import (
    OrgLoadService,
    TeamsOfTheOrg,
)
from openhands.agent_server.director.service.org_run_service import (
    AGENT_PAUSED,
    RESTART_NOTE,
    OrgRunService,
)
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.org_sync_service import OrgSyncService
from openhands.agent_server.director.service.org_watch_service import (
    OrgWatchService,
)
from openhands.agent_server.director.service.request_routing_service import (
    RequestRoutingService,
)
from openhands.agent_server.director.service.resumable_work_service import (
    ResumableWorkService,
)
from openhands.agent_server.director.service.review_service import (
    ReviewService,
)
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.session_lifecycle_service import (
    SessionLifecycleService,
)
from openhands.agent_server.director.service.session_merge_service import (
    SessionMergeService,
)
from openhands.agent_server.director.service.session_record_service import (
    SessionRecordService,
)
from openhands.agent_server.director.service.start_line_service import StartLineService
from openhands.agent_server.director.service.team_context_service import (
    TeamContextService,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from openhands.agent_server.director.service.team_member_start_service import (
    TeamMemberStartService,
)
from openhands.agent_server.director.service.team_messaging_service import (
    TeamMessagingService,
)
from openhands.agent_server.director.service.team_split_service import TeamSplitService
from openhands.agent_server.director.service.team_summary_service import (
    TeamSummaryService,
)
from openhands.agent_server.director.service.team_type_service import (
    DIRECTOR_TOOL_NAMES,
    TeamTypeService,
)
from openhands.agent_server.director.service.work_queue_service import (
    BASE_LANE,
    ORG_LANE,
    WorkQueueService,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

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
# After a conversation id: the script an ACP agent's process is started by.
_REMEMBERED_FILE = "director.json"
# The steps the director takes for a team's review, in its own lane.
_REVIEW_STEPS = ("request_review", "accept")


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


__all__ = [
    "AGENT_PAUSED",
    "DIRECTOR_TOOL_NAMES",
    "RESTART_NOTE",
    "DirectorService",
    "hand_over_of",
]


class DirectorService:
    """Owns the org tree and the team memory of one repository.

    An org lives in the repository its first conversation was opened in,
    which need not be the directory the server was started for; the director
    remembers it across restarts. Until there is an org, the director looks
    for a saved one in the server's directory. A repository that has no org
    and is a submodule of one that has stands for the nearest such
    repository above it, in both cases.

    Nothing is read from disk until the tree or the team memory is first used,
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
        self.remembered = RememberedFile(remembered)
        self._review_instructions_ready: asyncio.Task[None] | None = None
        self._stop_guard: Callable[[], None] | None = None
        self._stop_resume: Callable[[], None] | None = None
        self._clock = clock
        self._run_limit_retry = run_limit_retry
        self.environment: CodingEnvironment = GitCodingEnvironment()
        self.checkouts: Checkouts = self.environment.checkouts
        self.org_state = OrgStateService(
            repo_dir=lambda: self.org_load.repo_dir,
            clock=clock,
            environment=self.environment,
        )
        self.org_load = OrgLoadService(
            directory=repo_dir,
            environment=self.environment,
            remembered=self.remembered,
            org_state=self.org_state,
            type_of=lambda node: self.team_types.type_of(node),
            tree_changed=lambda: self.org_save.tree_changed(),
            loaded=self._org_loaded,
        )
        self.start_line = StartLineService(
            org=self._org, org_state=self.org_state, retry=run_limit_retry
        )
        self.session_drafts = SessionDraftService(run_state=self.org_state.run_state)
        self.team_history = TeamHistoryService(
            org=self._org,
            team_memory=self.org_load.memory,
            require_team_memory=lambda: self.org_load.team_memory,
            session_drafts=self.session_drafts,
        )
        self.documents: DocumentStore = RecordDocumentStore(
            self.org_load.memory,
            self.org_state.run_state,
            self.team_history.note_change,
            self.team_history,
        )
        self.tree: TeamTree = TeamsOfTheOrg(self._org)
        self.team_types = TeamTypeService(
            config_file=config_file,
            documents=self.documents,
            tree=self.tree,
            org=self._org,
            find_review_instructions=find_review_instructions,
        )
        self.confinement = AgentConfinementService(
            repo_dir=self.org_load.repo_dir_now,
            worktree_root=worktree_root,
            find_sandbox=find_sandbox,
            remembered=self.remembered,
            server_url=server_url,
            org=self._org,
            team_types=self.team_types,
            run_state=self.org_state.run_state,
            environment=self.environment,
            find_org=self.org_load.find_org,
            record_need=self.team_history.record_need,
        )
        self.org_settings = OrgSettingsService(
            environment=self.environment,
            config_file=config_file,
            remembered=self.remembered,
            team_types=self.team_types,
            tree=self.tree,
            documents=self.documents,
            org=self._org,
            repo_dir=lambda: self.org_load.repo_dir,
            max_concurrent_runs=max_concurrent_runs,
        )
        self._stall_poll = stall_poll
        self._stall_timer: asyncio.Task[None] | None = None
        self.agent_launch = AgentLaunchService()
        self.work_queue = WorkQueueService(default_lane=ORG_LANE)
        self.org_save = OrgSaveService(
            environment=self.environment,
            org=self._org,
            repo_dir=self.org_load.repo_dir_now,
            team_memory=self.org_load.memory,
            run_state=self.org_state.run_state,
            work_queue=self.work_queue,
            delay=save_delay,
        )
        self.model_calls = ModelCallService(
            launcher=self.agent_launch.require,
            sessions=lambda: (
                store.sessions() if (store := self.org_load.find_org()) else []
            ),
            call_timeout=model_call_timeout,
            summary_timeout=summary_timeout,
        )
        self._stop_status: Callable[[], None] | None = None
        self._create_lock = asyncio.Lock()
        self._first_message_lock = asyncio.Lock()
        self._create_lock = asyncio.Lock()
        self._first_message_lock = asyncio.Lock()
        self._build_services()
        for kind, step in {
            "brief": self.member_start.do_brief,
            "launch_member": self.member_start.do_launch_member,
            "split": self.team_splits.do_split,
            "start_team": self.member_start.do_start_team,
            "finish_split": self.team_splits.do_finish_split,
            "hand_over": self.team_splits.do_hand_over,
            "request_review": self.reviews.do_request_review,
            "accept": self.reviews.do_accept,
            "land_org": self.session_merges.do_land_org,
            "prune": self.team_splits.do_prune,
            "close_session": self.session_lifecycle.do_close_session,
        }.items():
            self.resumable_work.register(kind, step)

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

    def _org(self) -> OrgStore:
        """The org as it is loaded now, for the services that ask for it
        each time."""
        return self.org_load.org

    def _org_loaded(self) -> None:
        if self.agent_launch.is_running:
            self.work_queue.enqueue(self._came_back)

    def is_agent(self, conversation_id: UUID) -> bool:
        """Whether a conversation is one of an agent of the org. Asked by
        the server's conversation routes."""
        store = self.org_load.find_org()
        return store is not None and (
            store.find_by_conversation(str(conversation_id)) is not None
            or store.find_hand_over(str(conversation_id)) is not None
        )

    def _build_services(self) -> None:
        """Build what the director holds, each service after the ones it
        is given. ``__init__`` has built the ones that need nothing of these.

        A service is given another, never the director. Two are given a
        function that asks for one built after them, because they call up
        to what is built on them: ``message_delivery`` (a step is taken up
        when a team comes to rest) and ``team_messaging`` (reviews are
        released when a team is stopped or its IC is woken).
        """
        self.team_context = TeamContextService(
            checkouts=self.checkouts,
            org=self._org,
            org_state=self.org_state,
            org_settings=self.org_settings,
            team_types=self.team_types,
            team_history=self.team_history,
            session_drafts=self.session_drafts,
            documents=self.documents,
        )
        self.check_counts = CheckCountService(
            org=self._org, run_state=self.org_state.run_state
        )
        self.session_records = SessionRecordService(
            org=self._org,
            team_memory=self.org_load.memory,
            require_team_memory=lambda: self.org_load.team_memory,
            run_state=self.org_state.run_state,
            documents=self.documents,
            team_documents=self.org_settings.team_documents,
            team_types=self.team_types,
            session_drafts=self.session_drafts,
            org_save=self.org_save,
            event=self.team_history.event,
            confinement=self.confinement,
        )
        self.message_delivery = MessageDeliveryService(
            org=self._org,
            org_state=self.org_state,
            org_save=self.org_save,
            work_queue=self.work_queue,
            team_types=self.team_types,
            agent_launch=self.agent_launch,
            context_of=self.team_context.context_of,
            lane_of=lane_for,
            take_up=lambda intent_id, lane: self.resumable_work.take_up(
                intent_id, lane
            ),
            retry=self._run_limit_retry,
        )
        self.resumable_work = ResumableWorkService(
            run_state=self.org_state.run_state,
            work_queue=self.work_queue,
            lane_of=lane_for,
            is_paused=lambda: self.org_state.paused is not None,
            save=self.org_save.save,
            save_soon=self.org_save.save_soon,
            let_go=self.message_delivery.let_go,
            keep_failure=self.team_history.keep_failure,
            left=self.start_line.leave,
            saved_before_forgotten=SAVED_BEFORE_FORGOTTEN,
        )
        self.team_summaries = TeamSummaryService(
            org=self._org,
            team_memory=self.org_load.memory,
            org_state=self.org_state,
            org_settings=self.org_settings,
            team_types=self.team_types,
            model_calls=self.model_calls,
            work_queue=self.work_queue,
            launcher=self.agent_launch.require,
        )
        self.team_messaging = TeamMessagingService(
            org=self._org,
            org_state=self.org_state,
            org_settings=self.org_settings,
            org_save=self.org_save,
            team_types=self.team_types,
            team_history=self.team_history,
            session_drafts=self.session_drafts,
            documents=self.documents,
            check_counts=self.check_counts,
            message_delivery=self.message_delivery,
            in_place=self.org_load.in_place,
            resumable_work=self.resumable_work,
            work_queue=self.work_queue,
            launcher=self.agent_launch.require,
            release_reviews=lambda place_id: self.reviews.release_reviews(place_id),
        )
        self.member_start = TeamMemberStartService(
            checkouts=self.checkouts,
            org=self._org,
            repo_dir=self.org_load.repo_dir_now,
            org_state=self.org_state,
            start_line=self.start_line,
            message_delivery=self.message_delivery,
            confinement=self.confinement,
            team_types=self.team_types,
            team_documents=self.org_settings.team_documents,
            launcher=self.agent_launch.require,
            context_of=self.team_context.context_of,
            tell_of_failure=self.team_messaging.tell_of_failure,
            check_counts=self.check_counts,
            team_summaries=self.team_summaries,
            work_queue=self.work_queue,
        )
        self.session_merges = SessionMergeService(
            confinement=self.confinement,
            org=self._org,
            team_memory=self.org_load.memory,
            org_state=self.org_state,
            org_save=self.org_save,
            org_settings=self.org_settings,
            team_types=self.team_types,
            team_history=self.team_history,
            team_summaries=self.team_summaries,
            team_messaging=self.team_messaging,
            message_delivery=self.message_delivery,
            resumable_work=self.resumable_work,
            session_records=self.session_records,
        )
        self.reviews = ReviewService(
            org=self._org,
            repo_dir=self.org_load.repo_dir_now,
            checkouts=self.checkouts,
            launcher=self.agent_launch.require,
            org_state=self.org_state,
            org_save=self.org_save,
            org_settings=self.org_settings,
            team_types=self.team_types,
            team_history=self.team_history,
            team_context=self.team_context,
            team_messaging=self.team_messaging,
            message_delivery=self.message_delivery,
            resumable_work=self.resumable_work,
            work_queue=self.work_queue,
            check_counts=self.check_counts,
            confinement=self.confinement,
            model_calls=self.model_calls,
            session_records=self.session_records,
            session_merges=self.session_merges,
        )
        self.org_watch = OrgWatchService(
            org=self._org,
            has_org=lambda: self.org_load.has_org,
            checkouts=self.checkouts,
            launcher=self.agent_launch.require,
            poll_seconds=self._stall_poll,
            org_state=self.org_state,
            org_settings=self.org_settings,
            org_save=self.org_save,
            team_types=self.team_types,
            team_history=self.team_history,
            team_messaging=self.team_messaging,
            reviews=self.reviews,
            session_merges=self.session_merges,
            model_calls=self.model_calls,
            work_queue=self.work_queue,
        )
        self.team_splits = TeamSplitService(
            checkouts=self.checkouts,
            org=self._org,
            org_state=self.org_state,
            org_save=self.org_save,
            org_settings=self.org_settings,
            team_types=self.team_types,
            team_history=self.team_history,
            team_context=self.team_context,
            team_summaries=self.team_summaries,
            team_messaging=self.team_messaging,
            message_delivery=self.message_delivery,
            member_start=self.member_start,
            check_counts=self.check_counts,
            confinement=self.confinement,
            session_records=self.session_records,
            session_drafts=self.session_drafts,
            resumable_work=self.resumable_work,
            work_queue=self.work_queue,
            agent_launch=self.agent_launch,
            retry=self._run_limit_retry,
        )
        self.org_run = OrgRunService(
            org_watch=self.org_watch,
            checkouts=self.checkouts,
            org_load=self.org_load,
            team_history=self.team_history,
            org_state=self.org_state,
            org_save=self.org_save,
            org_settings=self.org_settings,
            start_line=self.start_line,
            message_delivery=self.message_delivery,
            team_messaging=self.team_messaging,
            team_types=self.team_types,
            member_start=self.member_start,
            confinement=self.confinement,
            resumable_work=self.resumable_work,
            work_queue=self.work_queue,
            team_splits=self.team_splits,
            agent_launch=self.agent_launch,
            lane_of=lane_for,
            reviews=self.reviews,
        )
        self.request_routing = RequestRoutingService(
            checkouts=self.checkouts,
            org=self._org,
            model_calls=self.model_calls,
            checkout=self.org_load.repo_dir_now,
            purpose=self.team_messaging.purpose,
            routed_requests=self.team_history.routed_requests,
            keep_routed=self.team_history.keep_routed,
            paused=lambda: self.org_state.paused is not None,
        )
        self.session_lifecycle = SessionLifecycleService(
            checkouts=self.checkouts,
            org_state=self.org_state,
            org_save=self.org_save,
            org_settings=self.org_settings,
            team_types=self.team_types,
            team_history=self.team_history,
            team_summaries=self.team_summaries,
            team_messaging=self.team_messaging,
            message_delivery=self.message_delivery,
            member_start=self.member_start,
            check_counts=self.check_counts,
            confinement=self.confinement,
            session_records=self.session_records,
            session_drafts=self.session_drafts,
            team_splits=self.team_splits,
            resumable_work=self.resumable_work,
            work_queue=self.work_queue,
            request_routing=self.request_routing,
            launcher=self.agent_launch.require,
            reviews=self.reviews,
            session_merges=self.session_merges,
            org_load=self.org_load,
            org_run=self.org_run,
        )
        self.org_sync = OrgSyncService(
            environment=self.environment,
            org_load=self.org_load,
            org_state=self.org_state,
            org_save=self.org_save,
            work_queue=self.work_queue,
            org_run=self.org_run,
            launcher=self.agent_launch.require,
        )

    async def _came_back(self) -> None:
        await self.org_run.came_back()

    async def start(self, launcher: TeamLauncher) -> None:
        """Begin running agents through ``launcher``, on the running loop."""
        self.agent_launch.begin(launcher)
        await self.work_queue.__aenter__()
        await self.org_save.__aenter__()
        self._stop_status = launcher.on_status(self.org_run.on_status)
        self._stop_guard = launcher.on_action(self.confinement.guard_action)
        self._stop_resume = launcher.on_resume(self.confinement.with_tools_again)
        await asyncio.to_thread(self.confinement.look_for_sandbox)
        bind_director(self)
        # What reviewers check work by is fetched now, off the loop, so
        # that no review request is the first to wait for it.
        self._review_instructions_ready = asyncio.create_task(
            asyncio.to_thread(self.team_types.load_review_instructions),
            name="director-review-instructions",
        )
        self._stall_timer = asyncio.create_task(
            self.org_watch.watch(), name="director-stall-watch"
        )
        if self.org_load.find_org() is not None:
            self.work_queue.enqueue(self._came_back)
        elif await asyncio.to_thread(self.org_load.remembered_repo) is not None:
            # An org that was left in the middle of its work is looked at
            # now, not when somebody first asks for it.
            await asyncio.to_thread(lambda: self.org_load.org)

    async def aclose(self) -> None:
        """Stop running agents and save the tree if a save was pending.

        An org that is in the middle of its work is left paused, so that the
        next start finds what to take up again.
        """
        bind_director(None)
        store = self.org_load.find_org()
        if store is not None and store.root_id is not None:
            run = self.org_state.run_state()
            if self.org_state.paused is None and has_work(run):
                self.org_state.hold("shutdown")
                self.start_line.org_held()
            run.put("clean", "1")
        if self._stop_status is not None:
            self._stop_status()
            self._stop_status = None
        if self._stop_guard is not None:
            self._stop_guard()
            self._stop_guard = None
        if self._stop_resume is not None:
            self._stop_resume()
            self._stop_resume = None
        await self.work_queue.__aexit__(None, None, None)
        timers = [
            task
            for task in (self._stall_timer, self._review_instructions_ready)
            if task is not None
        ]
        self._stall_timer = self._review_instructions_ready = None
        for task in timers:
            task.cancel()
        await asyncio.gather(*timers, return_exceptions=True)
        await self.org_save.__aexit__(None, None, None)
        self.agent_launch.end()
        self.org_state.close()
