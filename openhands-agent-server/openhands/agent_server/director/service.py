"""The director's state for one server: an org tree and its blackboard."""

import asyncio
import json
import logging
import os
import re
import shutil
import threading
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from uuid import UUID, uuid4

from openhands.agent_server.config import Config
from openhands.agent_server.conversation_service import ToolParams
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
    is_git_repository,
    save_and_discard,
)
from openhands.agent_server.director.git_blackboard import SPEC_FILE, GitBlackboard
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.models import (
    AgentRef,
    AgentRole,
    Autopsy,
    Boundary,
    CreateOrgRequest,
    OrgNode,
    OrgNodeInput,
    OrgSummary,
    RoutedMessage,
    RouteMessageRequest,
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
from openhands.agent_server.director.sandbox import (
    find_bwrap,
    shell_script,
    write_shell,
)
from openhands.agent_server.director.split import child_boundary, clean_path
from openhands.agent_server.event_service import ConversationRunLimitExceeded
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
    "prune_team",
)
# How long tree changes are gathered before they are saved as one commit.
SAVE_DELAY_SECONDS = 2.0
# How long to wait before starting an agent again when the server is full.
RUN_LIMIT_RETRY_SECONDS = 5.0
# How long a model may take to name a team before the team keeps the name
# it has; the root's agents are not briefed until then.
SUMMARY_TIMEOUT_SECONDS = 20.0
# Where, in the org's repository, the uncommitted work a prune discarded is
# kept.
PRUNED_REFS = "refs/agents/pruned"
# An agent whose conversation is gone, or could not be started.
AGENT_LOST = "lost"
# Where, in the repository's git directory, the director keeps the shells
# it confines its agents' terminals with.
_SHELLS_DIR = "director-shells"
_REMEMBERED_FILE = "director.json"

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
        worktree_root: Path | None = None,
        find_sandbox: Callable[[], str | None] = lambda: None,
        summary_timeout: float = SUMMARY_TIMEOUT_SECONDS,
        remembered: Path | None = None,
    ) -> None:
        """
        Args:
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
        self._bwrap: str | None = None
        self._stop_guard: Callable[[], None] | None = None
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
        self._first_message_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: Config) -> "DirectorService":
        return cls(
            config.director_repo_path or config.workspace_path,
            fixture=config.director_fixture,
            worktree_root=config.conversation_worktree_root,
            find_sandbox=find_bwrap,
            remembered=config.conversations_path.parent / _REMEMBERED_FILE,
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

    async def start(self, launcher: PodLauncher) -> None:
        """Begin running agents through ``launcher``, on the running loop."""
        self._loop = asyncio.get_running_loop()
        self._launcher = launcher
        self._effects = asyncio.Queue()
        self._worker = asyncio.create_task(self._work(), name="director-effects")
        self._stop_status = launcher.on_status(self._on_status)
        self._stop_guard = launcher.on_action(self.guard_action)
        self._bwrap = await asyncio.to_thread(self._find_sandbox)
        if self._bwrap is None:
            logger.info("Director cannot confine shells here; it checks file tools")
        bind_director(self)
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
                )
            except BaseException:
                store.remove(ROOT_ID)
                self._repo_dir = configured
                raise
            await asyncio.to_thread(self._remember)
            root = store.update_node(
                ROOT_ID,
                worktree_path=started.workspace,
                isolation=started.isolation,
                branch=await asyncio.to_thread(_branch_of, started.workspace),
            )
        # First, so that both agents are told the name the team keeps.
        self._enqueue(
            lambda: self._summarize(
                ROOT_ID, request.task, rename=request.title is None, first=True
            )
        )
        self._enqueue(lambda: self._brief(ROOT_ID, "ic", request.task))
        self._enqueue(lambda: self._launch_manager(ROOT_ID, request.task))
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
        self._enqueue(lambda: self._send(conversation_id, text))
        return RoutedMessage(
            node_id=node.id, conversation_id=conversation_id, started_org=False
        )

    def refresh_headline(self, node_id: str, report: str) -> None:
        """Have a node's headline rewritten after it reported. Any thread."""
        node = self.store.get(node_id)
        if node is not None:
            task = f"{node.spec_summary}\n\nLatest report of the team: {report}"
            self._enqueue(lambda: self._summarize(node_id, task))

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
        self._enqueue(lambda: self._send(conversation_id, from_director(text)))

    def carry_out_split(self, node_id: str) -> None:
        """Start the sub-teams of a node's approved split. Safe from any thread."""
        self._enqueue(lambda: self._carry_out_split(node_id))

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
            lambda: self._prune(node_id, approach, why_failed, replacement_task)
        )

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
        """Write the shell that holds an agent's terminal to its boundary.

        Returns:
            What to start the agent's terminal with, or None when shells are
            not confined on this server.
        """
        node = self.store.get(node_id)
        conversation_id = agent_conversation(node, role) if node else None
        if self._bwrap is None or node is None or conversation_id is None:
            return None
        git_dir = git(
            self._repo_dir, ["rev-parse", "--path-format=absolute", "--git-common-dir"]
        ).strip()
        shells = Path(git_dir) / _SHELLS_DIR
        script = shells / f"{conversation_id}.sh"
        write_shell(
            script,
            shell_script(
                self._bwrap,
                shutil.which("bash") or "/bin/sh",
                checkouts=self._checkouts(),
                private=shells,
                directory=node.boundary.path if role == "ic" else None,
                sub_team_directories=[
                    child.boundary.path for child in self.store.children(node_id)
                ],
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
        remembered = self._read_remembered()
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
        self._remember()
        return True

    def _read_remembered(self) -> Path | None:
        if self._remembered is None:
            return None
        try:
            return Path(json.loads(self._remembered.read_text())["repo_path"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _remember(self) -> None:
        """Keep the org's repository for the next start of the server."""
        if self._remembered is None:
            return
        try:
            self._remembered.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                self._remembered, json.dumps({"repo_path": str(self._repo_dir)}) + "\n"
            )
        except OSError:
            logger.exception("Could not remember the org's repository")

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
        return brief(
            node,
            role,
            task,
            blackboard.inherited(path),
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
            started = await self._when_there_is_room(
                lambda: launcher.start_derived(
                    parent_id,
                    UUID(conversation_id),
                    title=_agent_title(node.title, role),
                    task=text,
                    isolation=node.isolation if role == "ic" else "shared",
                    suffixes=[] if role == "ic" else [MANAGER_ROLE],
                    tool_params=tool_params,
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
        summarized: list[tuple[str, str]] = []
        for child in proposal.children:
            child_node = store.upsert(self._child_node(store, node, child))
            if await self._start_agent(child_node.id, "ic", parent_ic, child.task):
                await self._launch_manager(child_node.id, child.task)
            started.append(f"{child.title} ({child.path})")
            summarized.append((child_node.id, child.task))
        store.update_node(node_id, pending_split=None)
        # The shell the IC starts next can no longer write what it gave away.
        with suppress(GitError, OSError):
            await asyncio.to_thread(self._confine, node_id, "ic")
        for child_id, task in summarized:
            self._enqueue(
                lambda child_id=child_id, task=task: self._summarize(
                    child_id, task, first=True
                )
            )
        self.notify(
            parent_ic,
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
                headline=provisional_summary(replacement_task).headline,
                spec_summary=_summary_of(replacement_task),
            )
            if await self._start_agent(node_id, "ic", parent_ic, replacement_task):
                await self._launch_manager(node_id, replacement_task)
            self._enqueue(
                lambda: self._summarize(node_id, replacement_task, first=True)
            )
            outcome = f"a fresh team now works in {where}"
        self.notify(
            parent_ic,
            f"Your Manager pruned the team {node.title!r} ({where}): "
            f"{why_failed} Its uncommitted changes were discarded"
            + (f" (kept in git as {saved})" if saved else "")
            + f"; {outcome}.",
        )

    def _discard_work(self, teams: list[OrgNode], stamp: str) -> str | None:
        """Discard the uncommitted work of ``teams``, the pruned team first.

        Returns:
            The ref that keeps what the pruned team itself had uncommitted.
        """
        store = self.store
        kept: str | None = None
        for index, team in enumerate(teams):
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
            headline=provisional_summary(child.task).headline,
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
