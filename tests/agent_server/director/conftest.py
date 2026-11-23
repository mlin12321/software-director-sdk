import asyncio
import subprocess
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import pytest
import yaml

from openhands.agent_server.conversation_service import (
    AgentAdapter,
    ExecutionStatusListener,
    ToolParams,
)
from openhands.agent_server.director.agent_types import (
    DEFAULT_TYPE,
    SHIPPED_TYPES,
    TeamTemplate,
    load_types,
)
from openhands.agent_server.director.git_store import GitStore, Landing, team_key
from openhands.agent_server.director.headline import TeamSummary
from openhands.agent_server.director.launch import StartedAgent
from openhands.agent_server.director.models import (
    AgentKind,
    AgentRef,
    Boundary,
    Escalation,
    EscalationHandoff,
    EscalationKind,
    FormerTeam,
    IsolationMode,
    OrgNodeInput,
    Report,
    SessionInput,
    SessionTeam,
    SessionTeamInput,
    SplitChild,
    TeamEvent,
    Usage,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    division_lead_id,
    session_team_id,
)
from openhands.agent_server.director.service import (
    DIRECTOR_TOOL_NAMES,
    DirectorService,
)
from openhands.agent_server.director.split import ProposedTeam
from openhands.agent_server.event_service import (
    ActionGuard,
    ConversationRunLimitExceeded,
    ResumeAdapter,
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


def make_team(
    node_id: str, parent_id: str | None, *, path: str | None = None
) -> OrgNodeInput:
    """A team of the org tree, which may write all of its folder."""
    path = node_id if path is None else path
    return OrgNodeInput(
        id=node_id,
        parent_id=parent_id,
        title=node_id,
        boundary=Boundary(path=path, writable=[f"{path}/**" if path else "**"]),
        isolation="worktree",
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
        raised_by="ic",
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


def as_proposed(children: list[SplitChild]) -> list[ProposedTeam]:
    """``children`` as an IC names them in ``propose_teams``."""
    return [
        ProposedTeam(
            name=child.title,
            path=child.path,
            instruction=child.task,
            reason=child.reason,
            model=child.agent_profile,
            writable=child.writable,
            readonly=child.readonly,
            isolation=child.isolation,
            # Only a type the test names: one left out is the org's default.
            type=(child.agent_type if "agent_type" in child.model_fields_set else None),
        )
        for child in children
    ]


def small_tree() -> list[OrgNodeInput]:
    """root -> (a -> (a1, a2), b)."""
    return [
        make_team("root", None, path=""),
        make_team("a", "root", path="svc/a"),
        make_team("b", "root", path="svc/b"),
        make_team("a1", "a", path="svc/a/one"),
        make_team("a2", "a", path="svc/a/two"),
    ]


def make_place(
    team: OrgNodeInput,
    session_id: str,
    parent_id: str | None,
    *,
    checkout: str = ".",
    division_lead: bool = False,
    agents: Sequence[AgentRef] | None = None,
    escalations: Sequence[Escalation] = (),
) -> SessionTeamInput:
    """What the session ``session_id`` has at ``team``, below the place
    ``parent_id``: an idle IC unless ``agents`` says otherwise, and nobody
    at the session's Division Lead place."""
    standing = [] if division_lead else [AgentRef(role="ic", status="idle")]
    return SessionTeamInput(
        id=(division_lead_id if division_lead else session_team_id)(
            team.id, session_id
        ),
        parent_id=parent_id,
        title=team.title,
        boundary=team.boundary,
        isolation=team.isolation,
        session=session_id,
        team_id=team.id,
        division_lead=division_lead,
        worktree_path=checkout,
        branch="main",
        agents=list(standing if agents is None else agents),
        status="idle",
        escalations=list(escalations),
    )


def session_places(
    teams: Sequence[OrgNodeInput],
    team_id: str,
    session_id: str,
    *,
    checkout: str = ".",
) -> tuple[SessionInput, list[SessionTeamInput]]:
    """A session opened at ``team_id`` of the tree ``teams`` (parents before
    children): its Division Lead place, the place it is opened at and
    a place at every team below, in that order."""
    by_id = {team.id: team for team in teams}
    top = by_id[team_id]
    above_id = top.parent_id or top.id
    session = SessionInput(
        id=session_id, title=session_id, team_id=team_id, above_id=above_id
    )
    review = make_place(
        by_id[above_id], session_id, None, checkout=checkout, division_lead=True
    )
    places = [review, make_place(top, session_id, review.id, checkout=checkout)]
    inside = {team_id}
    for team in teams:
        if team.parent_id is not None and team.parent_id in inside:
            inside.add(team.id)
            above = session_team_id(team.parent_id, session_id)
            places.append(make_place(team, session_id, above, checkout=checkout))
    return session, places


def open_session_in(
    store: OrgStore,
    team_id: str = "root",
    session_id: str = "s",
    *,
    checkout: str = ".",
) -> SessionTeam:
    """Open a session at a team of the store's tree, with a place at that
    team and at every team below it.

    Returns:
        The place the session was opened at.
    """
    session, places = session_places(
        store.teams(), team_id, session_id, checkout=checkout
    )
    return store.open_session(session, places)


@dataclass
class DerivedStart:
    parent_id: UUID
    conversation_id: UUID
    title: str
    task: str
    isolation: IsolationMode
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
    resume_adapters: list[ResumeAdapter] = field(default_factory=list)
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
    # What each conversation's agent has used of its model.
    used: dict[UUID, Usage] = field(default_factory=dict)
    # What a conversation that failed said of it.
    failures: dict[UUID, str] = field(default_factory=dict)
    condensations: dict[UUID, str] = field(default_factory=dict)
    # What starting the next derived agent fails with.
    derive_error: Exception | None = None
    joined: list[UUID] = field(default_factory=list)
    # What the model answers to a question of the director, by usage id.
    answers: dict[str, str | list[str | None] | None] = field(default_factory=dict)
    asked: list[tuple[UUID, str, str, str]] = field(default_factory=list)
    # What an agent answers to a question on top of its conversation, by a
    # text the question contains.
    agent_answers: dict[str, str | None] = field(default_factory=dict)
    agents_asked: list[tuple[UUID, str]] = field(default_factory=list)
    # While set, an agent that is asked does not answer.
    agents_silent: asyncio.Event | None = None
    # What the root's request resolves to.
    root_kind: AgentKind = "openhands"
    # The model every agent it starts is said to run on.
    model: str | None = None
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
            model=self.model,
        )

    async def start_derived(
        self,
        parent_id: UUID,
        conversation_id: UUID,
        *,
        title: str,
        task: str,
        isolation: IsolationMode,
        tool_params: ToolParams | None = None,
        adapt_agent: AgentAdapter | None = None,
        profile: str | None = None,
        checkout: str | None = None,
        base: str | None = None,
    ) -> StartedAgent:
        if self.full_for > 0:
            self.full_for -= 1
            raise ConversationRunLimitExceeded("full")
        if self.derive_error is not None:
            error, self.derive_error = self.derive_error, None
            raise error
        if parent_id not in self.known:
            raise NotFound(f"Conversation {parent_id} does not exist")
        self.derived.append(
            DerivedStart(
                parent_id,
                conversation_id,
                title,
                task,
                isolation,
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
            model=self.model,
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
        tool_params: ToolParams | None = None,
        adapt_agent: AgentAdapter | None = None,
        checkout: str | None = None,
        base: str | None = None,
    ) -> StartedAgent:
        if profile not in self.profiles:
            raise NotFound(f"There is no agent profile named {profile!r}")
        return await self.start_derived(
            parent_id,
            conversation_id,
            title=title,
            task=task,
            isolation=isolation,
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

    async def usage(self, conversation_ids: Collection[UUID]) -> dict[UUID, Usage]:
        return {
            conversation_id: self.used.get(conversation_id, Usage())
            for conversation_id in conversation_ids
            if conversation_id in self.known
        }

    async def failure(self, conversation_id: UUID) -> str | None:
        return self.failures.get(conversation_id)

    async def condensed(self, conversation_id: UUID) -> str | None:
        return self.condensations.get(conversation_id)

    def on_status(self, listener: ExecutionStatusListener) -> Callable[[], None]:
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def on_action(self, guard: ActionGuard) -> Callable[[], None]:
        self.guards.append(guard)
        return lambda: self.guards.remove(guard)

    def on_resume(self, adapter: ResumeAdapter) -> Callable[[], None]:
        self.resume_adapters.append(adapter)
        return lambda: self.resume_adapters.remove(adapter)

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
        answer = self.answers.get(usage_id)
        if isinstance(answer, list):
            # One answer for each time it is asked, in turn.
            return answer.pop(0) if answer else None
        return answer

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

    async def mark_joined(self, conversation_id: UUID) -> None:
        if conversation_id not in self.known:
            raise NotFound(f"Conversation {conversation_id} does not exist")
        self.joined.append(conversation_id)

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


async def tasked(service: DirectorService) -> None:
    """Give every team that was only set up its first message, which starts it."""
    for team in service.org.places():
        if not team.agents:
            service.tell_writer(team.id, "Begin.")
    await service.settled()


def former_of(service: DirectorService, team_id: str) -> list[FormerTeam]:
    """The records of the teams removed from an id: the org's, then the ones
    sessions keep to themselves until their work is accepted."""
    kept = [
        record
        for session in service.org.sessions()
        for record in service.former_drafts(session.id)
        if record.id == team_key(team_id)
    ]
    return [*service.team_memory.former(team_id), *kept]


def record_of(service: DirectorService, team_id: str) -> list[TeamEvent]:
    """What became of the teams below a team: the org's record, and what
    the open sessions kept since their work was last accepted."""
    own = [
        event
        for session in service.org.sessions()
        for event in service.own_events(session.id, team_id)
    ]
    return [*service.team_memory.history(team_id), *own]


def take_in(
    store: GitStore,
    *,
    reports: dict[str, list[Report]] | None = None,
    events: dict[str, list[TeamEvent]] | None = None,
) -> None:
    """Add to team memory as an acceptance does."""
    store.take_in(
        Landing(
            files={},
            seen={},
            reports=reports or {},
            former=[],
            forget=[],
            events=events or {},
            routed=[],
            members=("ic", "manager"),
        ),
        "memory: for a test",
    )


def memory_of(service: DirectorService, file: str) -> str | None:
    """A file of team memory as the first session has it: what it wrote
    there and nobody accepted yet, else the org's."""
    [session, *_] = service.org.sessions()
    return service.read_file(service.org.session(session.id)[0], file)


def commit(checkout: Path, path: str, content: str) -> None:
    """Write a file in a checkout and commit it."""
    file = checkout / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)
    run_git(checkout, "add", path)
    run_git(checkout, "commit", "--quiet", "-m", f"Write {path}")


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


SHIPPED = load_types([SHIPPED_TYPES], DIRECTOR_TOOL_NAMES)


def shipped_type(node: OrgNodeInput) -> TeamTemplate:
    """The type of a team, among the ones the director ships."""
    return SHIPPED.get(node.agent_type) or SHIPPED[DEFAULT_TYPE]


def shipped_tools(member: str) -> tuple[str, ...]:
    """The tools a member of a team of the default type has."""
    return tuple(SHIPPED[DEFAULT_TYPE].agents[member].tools)


def members_yaml(ic: str = "", manager: str = "") -> str:
    """The `agents:` part of a type file for a team of an IC and a Manager,
    each told what is given here."""
    agents = {
        name: member.model_dump(exclude_defaults=True) | {"text": text}
        for (name, member), text in zip(
            SHIPPED[DEFAULT_TYPE].agents.items(), (ic, manager), strict=True
        )
    }
    return yaml.safe_dump({"agents": agents}, sort_keys=False)


def agent_conversation(place: SessionTeam, role: str) -> str | None:
    """The conversation of the team's agent in ``role``, if it has one."""
    return next(
        (agent.conversation_id for agent in place.agents if agent.role == role), None
    )
