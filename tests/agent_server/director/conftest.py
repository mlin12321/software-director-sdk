import asyncio
import subprocess
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.conversation_service import (
    AgentAdapter,
    ExecutionStatusListener,
    ToolParams,
)
from openhands.agent_server.director.headline import TeamSummary
from openhands.agent_server.director.models import (
    AgentKind,
    AgentRef,
    Boundary,
    Escalation,
    EscalationHandoff,
    EscalationKind,
    IsolationMode,
    NodeStatus,
    OrgNodeInput,
)
from openhands.agent_server.director.pods import StartedAgent
from openhands.agent_server.event_service import (
    ActionGuard,
    ConversationRunLimitExceeded,
)
from openhands.agent_server.models import StartConversationRequest
from openhands.sdk import LLM, Agent
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import NotFound
from openhands.sdk.tool.spec import Tool
from openhands.sdk.workspace import LocalWorkspace


def run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repository with one commit on ``main`` and a tracked file."""
    path = tmp_path / "repo"
    path.mkdir()
    run_git(path, "init", "--quiet", "--initial-branch=main")
    run_git(path, "config", "user.name", "Test")
    run_git(path, "config", "user.email", "test@example.invalid")
    (path / "README.md").write_text("hello\n")
    run_git(path, "add", "README.md")
    run_git(path, "commit", "--quiet", "-m", "init")
    return path


@pytest.fixture
def submodule(repo: Path, tmp_path: Path) -> Path:
    """A repository checked out as the submodule ``libs/payments`` of ``repo``."""
    source = tmp_path / "payments"
    source.mkdir()
    run_git(source, "init", "--quiet", "--initial-branch=main")
    run_git(source, "config", "user.name", "Test")
    run_git(source, "config", "user.email", "test@example.invalid")
    (source / "charge.py").write_text("amount = 1\n")
    run_git(source, "add", "charge.py")
    run_git(source, "commit", "--quiet", "-m", "init")
    run_git(
        repo,
        *("-c", "protocol.file.allow=always"),
        *("submodule", "--quiet", "add", str(source), "libs/payments"),
    )
    run_git(repo, "commit", "--quiet", "-m", "add payments")
    return repo / "libs" / "payments"


def make_node(
    node_id: str,
    parent_id: str | None,
    *,
    status: NodeStatus = "working",
    path: str | None = None,
    escalations: list[Escalation] | None = None,
) -> OrgNodeInput:
    return OrgNodeInput(
        id=node_id,
        parent_id=parent_id,
        title=node_id,
        boundary=Boundary(path=node_id if path is None else path),
        isolation="worktree",
        worktree_path=".",
        branch="main",
        agents=[AgentRef(role="ic", status="idle")],
        status=status,
        escalations=escalations or [],
    )


def make_escalation(
    escalation_id: str,
    *,
    kind: EscalationKind = "design_decision",
    raised_at: str = "2026-11-10T00:00:00Z",
    passed_up_by: tuple[str, ...] = (),
    held_by: str | None = None,
) -> Escalation:
    """An escalation that ``passed_up_by`` (nearest node first) each passed up.

    Leave ``held_by`` unset when that is every node above: it then waits on
    the human.
    """
    return Escalation(
        id=escalation_id,
        kind=kind,
        title=escalation_id,
        raised_at=raised_at,
        held_by=held_by,
        handoffs=[
            EscalationHandoff(
                node_id=node_id, reason=f"{node_id} could not", at=raised_at
            )
            for node_id in passed_up_by
        ],
    )


def small_tree() -> list[OrgNodeInput]:
    """root -> (a -> (a1, a2), b)."""
    return [
        make_node("root", None, path=""),
        make_node("a", "root", status="blocked", path="svc/a"),
        make_node("b", "root", status="done", path="svc/b"),
        make_node("a1", "a", status="done", path="svc/a/one"),
        make_node("a2", "a", status="idle", path="svc/a/two"),
    ]


@dataclass
class DerivedStart:
    parent_id: UUID
    conversation_id: UUID
    title: str
    task: str
    isolation: IsolationMode
    suffixes: list[str]
    tool_params: ToolParams | None = None
    adapt_agent: AgentAdapter | None = None
    # The agent profile it was started from, when it was not derived.
    profile: str | None = None
    # The checkout it was told to work on in, when it got no new one.
    checkout: str | None = None


@dataclass
class FakeLauncher:
    """Stands in for the server's conversations: records, runs nothing."""

    root: StartConversationRequest | None = None
    root_tools: list[str] = field(default_factory=list)
    root_title: str | None = None
    root_tool_params: ToolParams | None = None
    derived: list[DerivedStart] = field(default_factory=list)
    sent: list[tuple[UUID, str]] = field(default_factory=list)
    known: dict[UUID, ConversationExecutionStatus] = field(default_factory=dict)
    listeners: list[ExecutionStatusListener] = field(default_factory=list)
    # The message a ``send`` is about to deliver, for ``before`` to look at.
    sending: str = ""
    # How many of the next derived starts the server is too full for.
    full_for: int = 0
    root_error: Exception | None = None
    guards: list[ActionGuard] = field(default_factory=list)
    # What the model answers when asked to summarize a task; None: nothing.
    summary: TeamSummary | None = None
    # Called with the id of each conversation derived, once it runs.
    on_derived: Callable[[UUID], None] = lambda _: None
    summary_delay: float = 0
    summarized: list[tuple[UUID, str]] = field(default_factory=list)
    titles: dict[UUID, str] = field(default_factory=dict)
    stopped: list[UUID] = field(default_factory=list)
    paused: list[UUID] = field(default_factory=list)
    resumed: list[tuple[UUID, str | None]] = field(default_factory=list)
    # Called with the name of a call before it is carried out; a test ends
    # the server in the middle of a step by raising from it.
    before: Callable[[str], None] = lambda _: None
    deleted: list[UUID] = field(default_factory=list)
    # What the model answers to a question of the director, by usage id.
    answers: dict[str, str | None] = field(default_factory=dict)
    asked: list[tuple[UUID, str, str, str]] = field(default_factory=list)
    # What an agent answers to a question on top of its conversation, by a
    # text the question contains.
    agent_answers: dict[str, str | None] = field(default_factory=dict)
    agents_asked: list[tuple[UUID, str]] = field(default_factory=list)
    # While set, an agent that is asked does not answer.
    agents_silent: asyncio.Event | None = None
    # What the root's request resolves to.
    root_kind: AgentKind = "openhands"
    root_adapt: AgentAdapter | None = None
    kinds: dict[UUID, AgentKind] = field(default_factory=dict)
    # The stored agent profiles, by name.
    profiles: dict[str, AgentKind] = field(default_factory=dict)

    async def start_root(
        self,
        request: StartConversationRequest,
        tools: Sequence[Tool],
        *,
        title: str,
        tool_params: ToolParams | None = None,
        adapt_agent: AgentAdapter | None = None,
    ) -> StartedAgent:
        if self.root_error is not None:
            raise self.root_error
        assert request.conversation_id is not None
        self.root = request
        self.root_adapt = adapt_agent
        self.kinds[request.conversation_id] = self.root_kind
        self.root_tools = [tool.name for tool in tools]
        self.root_title = title
        self.root_tool_params = tool_params
        self.titles[request.conversation_id] = title
        self.known[request.conversation_id] = ConversationExecutionStatus.IDLE
        return StartedAgent(
            workspace=str(request.workspace.working_dir),
            isolation="shared",
            kind=self.root_kind,
            acp_server="claude-code" if self.root_kind == "acp" else None,
        )

    async def start_derived(
        self,
        parent_id: UUID,
        conversation_id: UUID,
        *,
        title: str,
        task: str,
        isolation: IsolationMode,
        suffixes: Sequence[str] = (),
        tool_params: ToolParams | None = None,
        adapt_agent: AgentAdapter | None = None,
        profile: str | None = None,
        checkout: str | None = None,
    ) -> StartedAgent:
        if self.full_for > 0:
            self.full_for -= 1
            raise ConversationRunLimitExceeded("full")
        if parent_id not in self.known:
            raise NotFound(f"Conversation {parent_id} does not exist")
        self.derived.append(
            DerivedStart(
                parent_id,
                conversation_id,
                title,
                task,
                isolation,
                list(suffixes),
                tool_params,
                adapt_agent,
                profile,
                checkout,
            )
        )
        kind = self.profiles[profile] if profile else self.kinds[parent_id]
        self.kinds[conversation_id] = kind
        self.titles[conversation_id] = title
        self.known[conversation_id] = ConversationExecutionStatus.IDLE
        self.on_derived(conversation_id)
        assert self.root is not None
        return StartedAgent(
            workspace=checkout or str(self.root.workspace.working_dir),
            isolation=isolation,
            kind=kind,
            acp_server="claude-code" if kind == "acp" else None,
        )

    async def start_from_profile(
        self,
        profile: str,
        parent_id: UUID,
        conversation_id: UUID,
        tools: Sequence[Tool],
        *,
        title: str,
        task: str,
        isolation: IsolationMode,
        suffixes: Sequence[str] = (),
        tool_params: ToolParams | None = None,
        adapt_agent: AgentAdapter | None = None,
        checkout: str | None = None,
    ) -> StartedAgent:
        if profile not in self.profiles:
            raise NotFound(f"There is no agent profile named {profile!r}")
        return await self.start_derived(
            parent_id,
            conversation_id,
            title=title,
            task=task,
            isolation=isolation,
            suffixes=suffixes,
            tool_params=tool_params,
            adapt_agent=adapt_agent,
            profile=profile,
            checkout=checkout,
        )

    async def agent_kind(self, conversation_id: UUID) -> AgentKind | None:
        return self.kinds.get(conversation_id)

    async def profile_kind(self, profile: str) -> AgentKind | None:
        return self.profiles.get(profile)

    async def send(self, conversation_id: UUID, text: str) -> None:
        if conversation_id not in self.known:
            raise NotFound(f"Conversation {conversation_id} does not exist")
        self.sending = text
        self.before("send")
        self.sent.append((conversation_id, text))

    async def started(self, conversation_id: UUID) -> StartedAgent | None:
        if conversation_id not in self.known:
            return None
        assert self.root is not None
        return StartedAgent(
            workspace=str(self.root.workspace.working_dir),
            isolation="shared",
            kind=self.kinds.get(conversation_id, "openhands"),
        )

    async def pause(self, conversation_id: UUID) -> None:
        self.paused.append(conversation_id)
        if conversation_id in self.known:
            self.report(str(conversation_id), ConversationExecutionStatus.PAUSED)

    async def resume(self, conversation_id: UUID, note: str | None) -> None:
        if conversation_id not in self.known:
            raise NotFound(f"Conversation {conversation_id} does not exist")
        self.resumed.append((conversation_id, note))
        self.report(str(conversation_id), ConversationExecutionStatus.RUNNING)

    async def stop(self, conversation_id: UUID) -> None:
        self.before("stop")
        self.stopped.append(conversation_id)

    async def delete(self, conversation_id: UUID) -> None:
        self.before("delete")
        self.deleted.append(conversation_id)
        self.known.pop(conversation_id, None)

    async def statuses(
        self, conversation_ids: Collection[UUID]
    ) -> dict[UUID, ConversationExecutionStatus]:
        return {
            conversation_id: self.known[conversation_id]
            for conversation_id in conversation_ids
            if conversation_id in self.known
        }

    def on_status(self, listener: ExecutionStatusListener) -> Callable[[], None]:
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def on_action(self, guard: ActionGuard) -> Callable[[], None]:
        self.guards.append(guard)
        return lambda: self.guards.remove(guard)

    async def summarize(self, conversation_id: UUID, task: str) -> TeamSummary | None:
        if conversation_id not in self.known:
            raise NotFound(f"Conversation {conversation_id} does not exist")
        self.summarized.append((conversation_id, task))
        await asyncio.sleep(self.summary_delay)
        return self.summary

    async def ask(
        self, conversation_id: UUID, system: str, user: str, *, usage_id: str
    ) -> str | None:
        if conversation_id not in self.known:
            raise NotFound(f"Conversation {conversation_id} does not exist")
        self.asked.append((conversation_id, usage_id, system, user))
        return self.answers.get(usage_id)

    async def ask_agent(self, conversation_id: UUID, question: str) -> str | None:
        if conversation_id not in self.known:
            raise NotFound(f"Conversation {conversation_id} does not exist")
        self.before("ask_agent")
        self.agents_asked.append((conversation_id, question))
        if self.agents_silent is not None:
            await self.agents_silent.wait()
        for part, answer in self.agent_answers.items():
            if part in question:
                return answer
        return None

    async def rename(self, conversation_id: UUID, title: str) -> None:
        if conversation_id not in self.known:
            raise NotFound(f"Conversation {conversation_id} does not exist")
        self.titles[conversation_id] = title

    def report(self, conversation_id: str, status: ConversationExecutionStatus) -> None:
        """A conversation changed status."""
        self.known[UUID(conversation_id)] = status
        for listener in list(self.listeners):
            listener(UUID(conversation_id), status)

    def messages_to(self, conversation_id: str | None) -> list[str]:
        return [text for target, text in self.sent if str(target) == conversation_id]


def root_conversation(workspace: Path) -> StartConversationRequest:
    return StartConversationRequest(
        agent=Agent(llm=LLM(model="gpt-4o", usage_id="test-llm"), tools=[]),
        workspace=LocalWorkspace(working_dir=str(workspace)),
    )
