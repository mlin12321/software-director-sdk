"""The director's state for one server: an org tree and its blackboard."""

import asyncio
import logging
import re
import threading
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from uuid import UUID, uuid4

from openhands.agent_server.config import Config
from openhands.agent_server.director.fixture import build_fixture_nodes
from openhands.agent_server.director.git import GitError, git, is_git_repository
from openhands.agent_server.director.git_blackboard import SPEC_FILE, GitBlackboard
from openhands.agent_server.director.models import (
    AgentRef,
    AgentRole,
    Boundary,
    CreateOrgRequest,
    OrgNode,
    OrgNodeInput,
    OrgSummary,
    SplitChild,
)
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    load_org,
    save_org,
)
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.pods import PodLauncher
from openhands.agent_server.director.prompts import (
    MANAGER_ROLE,
    POD_PROTOCOL,
    brief,
    from_director,
)
from openhands.agent_server.director.runtime import bind_director
from openhands.agent_server.director.split import child_boundary
from openhands.agent_server.event_service import ConversationRunLimitExceeded
from openhands.sdk.conversation.request import AgentLaunchAdditions
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import Conflict, NotFound, Unavailable, Unprocessable
from openhands.sdk.tool.spec import Tool


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
)
# How long tree changes are gathered before they are saved as one commit.
SAVE_DELAY_SECONDS = 2.0
# How long to wait before starting an agent again when the server is full.
RUN_LIMIT_RETRY_SECONDS = 5.0
# An agent whose conversation is gone, or could not be started.
AGENT_LOST = "lost"

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


def _summary_of(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line[:200]


class DirectorService:
    """Owns the org tree and the blackboard of one repository.

    Nothing is read from disk until the tree or the blackboard is first used,
    so creating the service is free for servers that never use the director.

    The tree can be read and taught without :meth:`start`. Running agents
    needs it: starting them, sending them messages and saving the tree all
    happen on one worker task, in the order they were asked for, so nothing
    that asks (a route, an agent's tool on its own thread) waits on a
    conversation.
    """

    def __init__(
        self,
        repo_dir: Path,
        *,
        fixture: bool = False,
        save_delay: float = SAVE_DELAY_SECONDS,
        run_limit_retry: float = RUN_LIMIT_RETRY_SECONDS,
    ) -> None:
        self._repo_dir = repo_dir
        self._fixture = fixture
        self._save_delay = save_delay
        self._run_limit_retry = run_limit_retry
        self._lock = threading.Lock()
        self._store: OrgStore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._launcher: PodLauncher | None = None
        self._effects: asyncio.Queue[Effect] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._save_timer: asyncio.TimerHandle | None = None
        self._stop_status: Callable[[], None] | None = None
        self._create_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: Config) -> "DirectorService":
        return cls(
            config.director_repo_path or config.workspace_path,
            fixture=config.director_fixture,
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
            is_git_repository=is_git_repository(self._repo_dir),
            fixture=self._fixture,
        )

    async def start(self, launcher: PodLauncher) -> None:
        """Begin running agents through ``launcher``, on the running loop."""
        self._loop = asyncio.get_running_loop()
        self._launcher = launcher
        self._effects = asyncio.Queue()
        self._worker = asyncio.create_task(self._work(), name="director-effects")
        self._stop_status = launcher.on_status(self._on_status)
        bind_director(self)
        if self._store is not None:
            self._enqueue(self._resync)

    async def aclose(self) -> None:
        """Stop running agents and save the tree if a save was pending."""
        bind_director(None)
        if self._stop_status is not None:
            self._stop_status()
            self._stop_status = None
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
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
        await self._effects.join()

    async def create_org(self, request: CreateOrgRequest) -> OrgNode:
        """Start the root team of a new org and brief its IC with the task.

        Raises:
            Conflict: The director already has an org, or serves the fixture.
            Unavailable: The director is not running, or its directory is not
                a git repository.
            Unprocessable: The conversation cannot lead an org here.
        """
        launcher = self._require_launcher()
        if self._fixture:
            raise Conflict("The director serves a fixture tree; it cannot run an org")
        store = await asyncio.to_thread(lambda: self.store)
        if not await asyncio.to_thread(is_git_repository, self._repo_dir):
            raise Unavailable(
                f"An org needs a git repository; {self._repo_dir} is not one"
            )
        conversation = request.conversation
        workspace = Path(conversation.workspace.working_dir).resolve()
        if not workspace.is_relative_to(self._repo_dir.resolve()):
            raise Unprocessable(
                f"The org works in {self._repo_dir}; the conversation's "
                f"workspace {workspace} is outside it"
            )
        if conversation.conversation_runtime == "docker":
            raise Unprocessable(
                "An org's agents run in the server's own process, not in Docker"
            )

        ic_id = conversation.conversation_id or uuid4()
        additions = conversation.agent_launch_additions or AgentLaunchAdditions()
        suffix = "\n\n".join(
            part
            for part in (additions.system_message_suffix_append, POD_PROTOCOL)
            if part
        )
        async with self._create_lock:
            if store.root_id is not None:
                raise Conflict("The director already has an org")
            store.upsert(
                OrgNodeInput(
                    id=ROOT_ID,
                    parent_id=None,
                    title=request.title,
                    boundary=Boundary(path="", writable=["**"]),
                    isolation="worktree" if conversation.worktree else "shared",
                    worktree_path=str(workspace),
                    branch="",
                    agents=_new_pod(ic_id),
                    status="idle",
                    spec_summary=_summary_of(request.task),
                )
            )
            try:
                started = await launcher.start_root(
                    conversation.model_copy(
                        update={
                            "conversation_id": ic_id,
                            "initial_message": None,
                            "agent_launch_additions": additions.model_copy(
                                update={"system_message_suffix_append": suffix}
                            ),
                        }
                    ),
                    [Tool(name=name) for name in DIRECTOR_TOOL_NAMES],
                )
            except BaseException:
                store.remove(ROOT_ID)
                raise
            root = store.update_node(
                ROOT_ID,
                worktree_path=started.workspace,
                isolation=started.isolation,
                branch=await asyncio.to_thread(_branch_of, started.workspace),
            )
        self._enqueue(lambda: self._brief(ROOT_ID, "ic", request.task))
        self._enqueue(lambda: self._launch_manager(ROOT_ID, request.task))
        return root

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
        self._enqueue(lambda: self._send(conversation_id, from_director(text)))

    def carry_out_split(self, node_id: str) -> None:
        """Start the sub-teams of a node's approved split. Safe from any thread."""
        self._enqueue(lambda: self._carry_out_split(node_id))

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
        if is_git_repository(self._repo_dir) and load_org(
            create_org_board(self._repo_dir), store
        ):
            logger.info("Director loaded %d nodes from %s", len(store), self._repo_dir)
        store.on_change(self._schedule_save)
        if self._loop is not None:
            self._enqueue(self._resync)
        return store

    def _enqueue(self, effect: Effect) -> None:
        loop = self._loop
        if loop is None:
            raise Unavailable("The director is not running agents on this server")
        loop.call_soon_threadsafe(self._effects.put_nowait, effect)

    async def _work(self) -> None:
        while True:
            effect = await self._effects.get()
            try:
                await effect()
            except Exception:
                logger.exception("A director action failed")
            finally:
                self._effects.task_done()

    async def _when_there_is_room[T](self, start: Callable[[], Awaitable[T]]) -> T:
        """Run ``start``, again after a pause for as long as the server is full."""
        while True:
            try:
                return await start()
            except ConversationRunLimitExceeded:
                logger.info("Server is at its run limit; the director waits")
                await asyncio.sleep(self._run_limit_retry)

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
        return brief(
            node,
            role,
            task,
            blackboard.inherited(path),
            blackboard.inherited_rules(path),
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
            if parent_conversation is None:
                raise NotFound(f"Node {node_id} has nobody to derive its {role} from")
            parent_id = UUID(parent_conversation)
            started = await self._when_there_is_room(
                lambda: launcher.start_derived(
                    parent_id,
                    UUID(conversation_id),
                    title=f"{node.title} ({'IC' if role == 'ic' else 'Manager'})",
                    task=text,
                    isolation=node.isolation if role == "ic" else "shared",
                    suffixes=[] if role == "ic" else [MANAGER_ROLE],
                )
            )
        except Exception:
            logger.exception("Could not start the %s of node %s", role, node_id)
            store.set_agent(node_id, role, status=AGENT_LOST)
            store.update_node(node_id, status="blocked")
            return False
        if role == "ic":
            store.update_node(
                node_id,
                worktree_path=started.workspace,
                isolation=started.isolation,
                branch=await asyncio.to_thread(_branch_of, started.workspace),
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
        if proposal.contracts.strip():
            await asyncio.to_thread(self._record_contracts, node, proposal.contracts)

        parent_ic = agent_conversation(node, "ic")
        started: list[str] = []
        for child in proposal.children:
            child_node = store.upsert(self._child_node(store, node, child))
            if await self._start_agent(child_node.id, "ic", parent_ic, child.task):
                await self._launch_manager(child_node.id, child.task)
            started.append(f"{child.title} ({child.path})")
        store.update_node(node_id, pending_split=None)
        self.notify(
            parent_ic,
            "Your split was approved and these sub-teams now own their "
            f"directories: {', '.join(started)}. Leave those directories to "
            "them and carry on with what remains in your boundary.",
        )

    def _record_contracts(self, node: OrgNode, contracts: str) -> None:
        blackboard = GitBlackboard(self._repo_dir)
        path = node.boundary.path
        own = next(
            (spec.spec for spec in blackboard.inherited(path) if not spec.readonly), ""
        )
        section = f"## Contracts between sub-teams\n\n{contracts.strip()}\n"
        blackboard.write(
            path, SPEC_FILE, f"{own.rstrip()}\n\n{section}" if own.strip() else section
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
            worktree_path=parent.worktree_path,
            branch=parent.branch,
            agents=_new_pod(uuid4()),
            status="idle",
            spec_summary=_summary_of(child.task),
        )


def _new_pod(ic_id: UUID) -> list[AgentRef]:
    return [
        AgentRef(role="ic", conversation_id=str(ic_id), status="idle"),
        AgentRef(role="manager", conversation_id=str(uuid4()), status="idle"),
    ]


def _branch_of(workspace: str) -> str:
    try:
        return git(Path(workspace), ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    except (GitError, OSError):
        return ""
