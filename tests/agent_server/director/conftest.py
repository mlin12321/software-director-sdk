import asyncio
import subprocess
from collections.abc import AsyncIterator, Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest
import yaml

from openhands.agent_server.config import Config
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
    CreateOrgRequest,
    Escalation,
    EscalationHandoff,
    EscalationKind,
    FormerTeam,
    IsolationMode,
    OpenSessionRequest,
    OrgNodeInput,
    Report,
    ReviewFinding,
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
    session_id_of,
    session_team_id,
    team_id_of,
)
from openhands.agent_server.director.service import (
    DIRECTOR_TOOL_NAMES,
    DirectorService,
)
from openhands.agent_server.director.split import ProposedTeam
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    ProposeTeamsTool,
    RaiseEscalationTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    ApproveTeamsAction,
    ProposeTeamsAction,
    PruneTeamAction,
    RaiseEscalationAction,
    ReviewTeamAction,
    SubmitReviewAction,
)
from openhands.agent_server.director.tools.impl import DirectorObservation
from openhands.agent_server.event_service import (
    ActionGuard,
    ConversationRunLimitExceeded,
    ResumeAdapter,
)
from openhands.agent_server.models import StartConversationRequest
from openhands.sdk import LLM, Agent
from openhands.sdk.conversation.state import (
    ConversationExecutionStatus,
    ConversationState,
)
from openhands.sdk.errors import NotFound
from openhands.sdk.event import ActionEvent
from openhands.sdk.tool.spec import Tool
from openhands.sdk.tool.tool import Action, ToolDefinition
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.file_editor.definition import FileEditorAction


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


async def started_with_server(
    repo: Path, launcher: FakeLauncher, **options
) -> DirectorService:
    service = DirectorService(
        repo,
        save_delay=0,
        run_limit_retry=0,
        remembered=repo.parent / "state" / "director.json",
        server_url="http://127.0.0.1:8000",
        **options,
    )
    await service.start(launcher)
    await service.create_org(
        CreateOrgRequest(
            title="Checkout", task="Build", conversation=root_conversation(repo)
        )
    )
    await service.settled()
    return service


async def use(
    service: DirectorService,
    conversation_id: str | None,
    tool: type[ToolDefinition[Any, Any]],
    action: Action,
) -> DirectorObservation:
    """Call ``tool`` as the agent of ``conversation_id`` does: off the loop."""
    assert conversation_id is not None
    state = cast(ConversationState, SimpleNamespace(id=UUID(conversation_id)))
    executor = tool.create(state)[0].executor
    assert executor is not None
    observation = await asyncio.to_thread(executor, action)
    await service.settled()
    assert isinstance(observation, DirectorObservation)
    return observation


def node(service: DirectorService, node_id: str) -> SessionTeam:
    """A place by its id, or the place of the session opened first at the
    team of that id."""
    found = service.org.place(node_id)
    if "@" not in node_id:
        found = next(
            (
                place
                for place in service.org.places()
                if not place.division_lead and place.team_id == node_id
            ),
            None,
        )
    assert found is not None
    return found


def division_lead_ic(service: DirectorService, node_id: str = "root") -> str | None:
    """The IC that reviews the session of the place ``node_id``."""
    place = service.org.place(division_lead_of(service, node_id))
    assert place is not None
    return agent_conversation(place, "ic")


def division_lead_of(service: DirectorService, node_id: str = "root") -> str:
    session = service.org.session_of(node(service, node_id))
    assert session is not None
    return division_lead_id(session.above_id, session.id)


def ic(service: DirectorService, node_id: str = "root") -> str | None:
    return agent_conversation(node(service, node_id), "ic")


def manager(service: DirectorService, node_id: str = "root") -> str | None:
    return agent_conversation(node(service, node_id), "manager")


async def woken(service: DirectorService, node_id: str = "root") -> str:
    """The Manager of a team, started as the first message to it does."""
    if manager(service, node_id) is None:
        service.tell_overseer(node(service, node_id).id, "Look in on your team.")
        await service.settled()
    found = manager(service, node_id)
    assert found is not None
    return found


def proposing(
    children: list[SplitChild],
    contracts: str = "",
    keep: list[str] | None = None,
    text: str = "",
) -> ProposeTeamsAction:
    return ProposeTeamsAction(
        teams=as_proposed(children), contracts=contracts, keep=keep or [], text=text
    )


SPLIT = proposing(
    [
        SplitChild(title="API", path="services/api", task="Serve the checkout API"),
        SplitChild(
            title="Web",
            path="apps/web",
            task="Build the checkout pages",
            isolation="worktree",
        ),
    ],
    "POST /checkout returns {order_id}",
)


APPROVE = ApproveTeamsAction()


async def split_root(service: DirectorService) -> None:
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)


def sub_teams(service: DirectorService, node_id: str = "root") -> list[SessionTeam]:
    """What the session opened first has directly below a team."""
    return service.org.below(node(service, node_id).id)


def has_place(service: DirectorService, team_id: str) -> bool:
    """Whether a session has the team ``team_id``."""
    return any(
        session_id_of(place) and team_id_of(place) == team_id
        for place in service.org.places()
    )


CENTS = RaiseEscalationAction(
    category="design_decision", title="Cents or decimals?", detail="For totals."
)


PRUNE_API = PruneTeamAction(
    team="services.api",
    reason="Every schema change broke all of them.",
)


CODE_REVIEW = "Review the diff for bugs."


ANALYST = (
    members_yaml(
        ic="Your team analyses results. It runs no experiments.\n",
        manager="Approve a proposal if each sub-team reads one set of results.\n",
    )
    + """\
description: Reads results and says what they mean.
actions:
  submit_review: Submit one table and what it shows.
review:
  write_up: |
    Write the account of your tables for the reviewer.
  checklist: |
    Every number in a table is in a committed result file.
omit_tools: [propose_teams, read_team_context]
"""
)


TEAMS = proposing(
    [
        SplitChild(
            title="Tables", path="tables", task="Tabulate", agent_type="analyst"
        ),
        SplitChild(title="Harness", path="harness", task="Build the harness"),
    ],
    contracts="Tables reads what Harness writes to `out/`.",
)


def org_config(folder: Path, settings: str = "") -> Path:
    """An org's ``director.yaml`` with the type ``analyst`` beside it."""
    config = folder / "config" / "director.yaml"
    (config.parent / "types").mkdir(parents=True, exist_ok=True)
    config.write_text(settings)
    (config.parent / "types" / "analyst.yaml").write_text(ANALYST)
    return config


async def an_org(
    repo: Path,
    launcher: FakeLauncher,
    config_file: Path | None = None,
    agent_type: str | None = None,
) -> DirectorService:
    service = DirectorService(
        repo,
        save_delay=0,
        run_limit_retry=0,
        find_review_instructions=lambda: CODE_REVIEW,
        config_file=config_file,
        remembered=repo.parent / "director.json",
    )
    await service.start(launcher)
    await service.create_org(
        CreateOrgRequest(
            title="Lab",
            task="Find out what helps training",
            agent_type=agent_type,
            conversation=root_conversation(repo),
        )
    )
    await service.settled()
    return service


async def split_into_typed_teams(service: DirectorService) -> None:
    await use(service, ic(service), ProposeTeamsTool, TEAMS)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)


WRITE_UP = "pull request description"


FINDING = ReviewFinding(path="services/api/app.py", line=1, issue="Returns a string.")


DONE = SubmitReviewAction(summary="Endpoint shipped.")


async def api_submits(service: DirectorService, repo: Path) -> DirectorObservation:
    commit(repo, "services/api/app.py", "order_id = '1'\n")
    return await use(service, ic(service, "services.api"), SubmitReviewTool, DONE)


async def reviewer_wakes(
    service: DirectorService, launcher: FakeLauncher, node_id: str = "root"
) -> None:
    """The IC of a team starts a turn, as when anything is sent to it."""
    reviewer = ic(service, node_id)
    assert reviewer is not None
    launcher.report(reviewer, ConversationExecutionStatus.RUNNING)
    await service.settled()


def requests(service: DirectorService, launcher: FakeLauncher) -> list[str]:
    """The review requests the top team's IC got."""
    return [
        text for text in launcher.messages_to(ic(service)) if "Review request" in text
    ]


def reject(node_id: str = "services.api") -> ReviewTeamAction:
    return ReviewTeamAction(team=node_id, decision="reject", findings=[FINDING])


def accept(node_id: str = "services.api", needed_above: str = "") -> ReviewTeamAction:
    return ReviewTeamAction(team=node_id, decision="accept", needed_above=needed_above)


async def split_api(service: DirectorService) -> None:
    await use(
        service,
        ic(service, "services.api"),
        ProposeTeamsTool,
        proposing(
            [
                SplitChild(
                    title="Schema", path="services/api/schema", task="Define orders"
                )
            ],
            "Order has an id.",
        ),
    )
    await use(service, manager(service, "services.api"), ApproveTeamsTool, APPROVE)
    await tasked(service)


def in_own_worktree(
    service: DirectorService, tmp_path: Path, repo: Path, *teams: str
) -> Path:
    """Give teams (the Web team, when none is named) the one worktree a
    conversation of theirs would have made."""
    teams = teams or ("apps.web",)
    name = teams[0].rsplit(".", 1)[-1]
    worktree = tmp_path / name
    run_git(repo, "worktree", "add", "--quiet", "-b", name, str(worktree))
    for team in teams:
        service.org.update_place(
            node(service, team).id, worktree_path=str(worktree), branch=name
        )
    return worktree


SCHEMA = "services.api.schema"


async def landed(service: DirectorService) -> None:
    """Have the work of the session opened first reviewed and accepted."""
    await use(service, ic(service), SubmitReviewTool, DONE)
    await use(
        service,
        division_lead_ic(service),
        ReviewTeamTool,
        accept(node(service, "root").id),
    )


async def at_api(service: DirectorService) -> tuple[str, str]:
    """A session opened at API once the org has it.

    Returns:
        The place it was opened at and its IC's conversation.
    """
    await split_root(service)
    await landed(service)
    opened = await service.open_session(
        OpenSessionRequest(text="Add refunds.", node_id="services.api")
    )
    await service.settled()
    return opened.session.teams[0].id, opened.conversation_id


JOBS = "services.api.jobs"


def sets_up_jobs(
    service: DirectorService,
    top: str,
    path: str = "services/api/jobs",
    team_id: str = JOBS,
):
    """Have a session set up a team for a directory under the team it was
    opened at."""
    place = service.org.place(top)
    assert isinstance(place, SessionTeam)
    service.org.upsert(
        SessionTeamInput(
            **place.model_dump(include={"session", "title", "isolation", "status"}),
            worktree_path=place.worktree_path,
            branch=place.branch,
            id=f"{team_id}@{place.session}",
            team_id=team_id,
            parent_id=top,
            agents=[],
            boundary=place.boundary.model_copy(update={"path": path}),
            since="set up by the session",
        )
    )


def org_gains(service: DirectorService, team_id: str, path: str) -> None:
    """Have other accepted work give the org a team under API."""
    api = service.org.team("services.api")
    assert api is not None
    service.org.upsert(
        OrgNodeInput(
            **api.model_dump(include={"title", "isolation"}),
            id=team_id,
            parent_id=api.id,
            boundary=api.boundary.model_copy(update={"path": path}),
            since="set up by other work",
        )
    )


def waiting(service: DirectorService, top: str) -> list[str]:
    """The teams of a session that wait to be settled."""
    place = service.org.place(division_lead_of(service, top))
    assert isinstance(place, SessionTeam)
    return [
        item.key
        for item in (place.merge.items if place.merge else [])
        if item.kind == "team"
    ]


def division_lead_manager(service: DirectorService, top: str) -> str | None:
    place = service.org.place(division_lead_of(service, top))
    assert place is not None
    return agent_conversation(place, "manager")


def path_of(service: DirectorService, team_id: str) -> str | None:
    team = service.org.team(team_id)
    return team.boundary.path if team else None


async def session_accepted(
    service: DirectorService, top: str, conversation: str
) -> str:
    """Have a session's work submitted and accepted.

    Returns:
        What its reviewer was answered.
    """
    await use(service, conversation, SubmitReviewTool, DONE)
    answer = await use(
        service, division_lead_ic(service, top), ReviewTeamTool, accept(top)
    )
    await service.settled()
    return answer.text


async def started(repo: Path, launcher: FakeLauncher, **options) -> DirectorService:
    service = DirectorService(
        repo,
        save_delay=0,
        remembered=repo.parent / "state" / "director.json",
        **{"run_limit_retry": 0, **options},
    )
    await service.start(launcher)
    await service.create_org(
        CreateOrgRequest(
            title="Checkout", task="Build", conversation=root_conversation(repo)
        )
    )
    await service.settled()
    return service


def split(below: str, count: int) -> ProposeTeamsAction:
    return proposing(
        [
            SplitChild(title=f"{below} {n}", path=f"{below}/t{n}", task="Work")
            for n in range(count)
        ]
    )


async def eventually(condition) -> None:
    for _ in range(400):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("It never happened")


def looks(service: DirectorService, launcher: FakeLauncher, team: str = "root"):
    """The looks the director asked the Manager of a team to take."""
    return [
        text
        for text in launcher.messages_to(manager(service, team))
        if "A look at your teams" in text
    ]


async def team_accepted(
    service: DirectorService, repo: Path, team: str, times: int = 1
):
    """Have a team submit work and the top team's IC accept it."""
    directory = node(service, team).boundary.path
    for _ in range(times):
        count = len(list((repo / directory).glob("*.txt"))) if directory else 0
        commit(repo, f"{directory}/part{count}.txt", "done\n")
        await use(service, ic(service, team), SubmitReviewTool, DONE)
        await use(service, ic(service), ReviewTeamTool, accept(team))
    await service.settled()


async def escalates(service: DirectorService, team: str, times: int = 1) -> None:
    for number in range(times):
        await use(
            service,
            ic(service, team),
            RaiseEscalationTool,
            RaiseEscalationAction(category="design_decision", title=f"Q{number}?"),
        )
    await service.settled()


async def sent_back(service: DirectorService, repo: Path, team: str, times: int = 1):
    """Have a team submit work and the top team's IC send it back."""
    directory = node(service, team).boundary.path
    for _ in range(times):
        count = len(list((repo / directory).glob("*.txt")))
        commit(repo, f"{directory}/try{count}.txt", "again\n")
        await use(service, ic(service, team), SubmitReviewTool, DONE)
        await use(service, ic(service), ReviewTeamTool, reject(team))
    await service.settled()


class ServerEnded(BaseException):
    """Ends whatever the director was doing, as a killed server does."""


async def again(repo: Path, launcher: FakeLauncher, **options) -> DirectorService:
    """The director of the server's next start, on the same org."""
    launcher.listeners.clear()
    launcher.before = lambda _: None
    launcher.on_derived = lambda _: None
    service = DirectorService(
        repo,
        save_delay=0,
        run_limit_retry=0,
        remembered=repo.parent / "state" / "director.json",
        **options,
    )
    await service.start(launcher)
    await service.settled()
    return service


def end_server_at(launcher: FakeLauncher, call: str, nth: int = 1) -> asyncio.Event:
    """Have the server end at the ``nth`` time the director makes ``call``.

    Returns:
        Set once it has ended.
    """
    ended = asyncio.Event()
    seen = 0

    def before(name: str) -> None:
        nonlocal seen
        seen += name == call
        if ended.is_set() or seen >= nth:
            ended.set()
            raise ServerEnded

    launcher.before = before
    launcher.on_derived = lambda _: before("derived")
    return ended


async def cut_off(launcher: FakeLauncher, ended: asyncio.Event, doing) -> None:
    """Let the server end in the middle of ``doing``."""
    task = asyncio.create_task(doing)
    await ended.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def server_config(repo_dir: Path, **config) -> Config:
    """A server whose director is started for ``repo_dir`` and keeps what it
    remembers beside it."""
    return Config(
        **{
            "session_api_keys": [],
            "director_repo_path": repo_dir,
            "conversations_path": repo_dir.parent / "state" / "conversations",
            **config,
        }
    )


def attempt(
    service: DirectorService, conversation_id: str | None, tool: str, action: Action
) -> str | None:
    """What the director says to an agent that is about to use ``tool``."""
    event = cast(ActionEvent, SimpleNamespace(tool_name=tool, action=action))
    return service.guard_action(UUID(conversation_id or ""), event)


def create(path: Path | str) -> FileEditorAction:
    return FileEditorAction(command="create", path=str(path), file_text="x")


API = "services.api"


DB = "services.api.db"


async def propose_and_approve(
    service: DirectorService,
    node_id: str,
    *children: SplitChild,
    keep: list[str] | None = None,
) -> None:
    proposal = proposing(list(children), keep=keep)
    await use(service, ic(service, node_id), ProposeTeamsTool, proposal)
    await use(service, manager(service, node_id), ApproveTeamsTool, APPROVE)
    await tasked(service)


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher) -> AsyncIterator[DirectorService]:
    """An org that was just created. A module that needs another overrides it."""
    service = await started(repo, launcher)
    yield service
    await service.aclose()


# A type whose members have names of their own, and whose reviewer is not
# the one that writes.
BENCH = [
    "read_team_context",
    "raise_escalation",
    "report",
    "write_spec",
    "submit_review",
    "propose_teams",
    "msg_team",
]
LAB = f"""\
description: A lab.
agents:
  bench:
    writes: true
    text: Runs the experiments.
    tools: [{", ".join(BENCH)}]
    lead_tools: [read_team_context]
  referee:
    text: Checks what the benches below found.
    tools: [read_team_context, review_team]
    lead_tools: [read_team_context, review_team]
  chair:
    tools: [read_team_context, approve_teams, decline_teams]
    lead_tools: [read_team_context, settle_teams]
"""
