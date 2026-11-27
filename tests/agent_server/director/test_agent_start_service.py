"""Starting agents by itself: a repository, a store of teams, a run state
in memory and a launcher that records what it is asked to start."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from openhands.agent_server.director.documents.team_documents import TeamDocuments
from openhands.agent_server.director.environment.git import (
    GitCheckouts,
    GitCodingEnvironment,
)
from openhands.agent_server.director.git_store import GitStore
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.remembered import RememberedFile
from openhands.agent_server.director.run_state import TOLD
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.agent_start_service import (
    AgentStartService,
    agent_title,
)
from openhands.agent_server.director.service.message_delivery_service import (
    MessageDeliveryService,
)
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.start_line_service import (
    StartLineService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import (
    FakeLauncher,
    open_session_in,
    small_tree,
)
from tests.agent_server.director.test_team_documents import (
    AnyOrg,
    RootAboveApi,
    TextsInADict,
)


SESSION = "s1"
ROOT = f"root@{SESSION}"
A = f"a@{SESSION}"


class Starts:
    """A start service over a small tree whose top team's IC runs."""

    def __init__(self, repo: Path) -> None:
        self.store = OrgStore(lambda node: self.team_types.type_of(node))
        self.repo = repo
        self.documents = documents = TextsInADict()
        self.team_types = TeamTypeService(
            config_file=None,
            documents=documents,
            tree=RootAboveApi(),
            org=lambda: self.store,
        )
        for node in small_tree():
            self.store.upsert(node)
        open_session_in(self.store, "root", SESSION, checkout=str(repo))
        self.top, self.agent = uuid4(), uuid4()
        self.store.set_agent(ROOT, "ic", conversation_id=str(self.top))
        self.store.set_agent(A, "ic", conversation_id=str(self.agent))
        self.launcher = FakeLauncher()
        self.launcher.root = SimpleNamespace(  # type: ignore[assignment]
            workspace=SimpleNamespace(working_dir=repo)
        )
        self.launcher.known[self.top] = ConversationExecutionStatus.IDLE
        self.launcher.kinds[self.top] = "openhands"
        self.failures: list[tuple[str, str, str]] = []
        self.queue = WorkQueueService(default_lane="org")
        self.org_state = OrgStateService(
            repo_dir=lambda: repo,
            clock=lambda: datetime.now(UTC),
            environment=GitCodingEnvironment(),
        )
        self.start_line = start_line = StartLineService(
            org=lambda: self.store, org_state=self.org_state, retry=0
        )
        self.org_save = OrgSaveService(
            org=lambda: self.store,
            repo_dir=lambda: repo,
            team_memory=lambda: GitStore(repo),
            run_state=self.org_state.run_state,
            work_queue=self.queue,
            delay=60,
        )
        context = {"status": "idle", "task": "Build the parser"}
        self.message_delivery = message_delivery = MessageDeliveryService(
            org=lambda: self.store,
            org_state=self.org_state,
            org_save=self.org_save,
            work_queue=self.queue,
            team_types=self.team_types,
            launcher=lambda: self.launcher,
            is_running=lambda: True,
            context_of=lambda place, role: dict(context),
            is_stopped=lambda conversation_id: False,
            lane_of=lambda intent: intent.node_id,
            take_up=lambda intent_id, lane: None,
            default_lane="org",
            retry=0,
        )
        self.service = AgentStartService(
            checkouts=GitCheckouts(),
            org=lambda: self.store,
            repo_dir=lambda: repo,
            org_state=self.org_state,
            start_line=start_line,
            message_delivery=message_delivery,
            confinement=AgentConfinementService(
                checkouts=GitCheckouts(),
                repo_dir=lambda: repo,
                worktree_root=None,
                find_sandbox=lambda: None,
                remembered=RememberedFile(repo.parent / "director.json"),
                server_url=None,
                org=lambda: self.store,
                team_types=self.team_types,
                run_state=self.org_state.run_state,
            ),
            team_types=self.team_types,
            team_documents=lambda: TeamDocuments(
                documents, self.team_types.protocols, AnyOrg(RootAboveApi())
            ),
            launcher=lambda: self.launcher,
            context_of=lambda place, role: dict(context),
            tell_of_failure=lambda *said: self.failures.append(said),
        )


@pytest.fixture
async def starts(repo: Path) -> AsyncIterator[Starts]:
    made = Starts(repo)
    async with made.queue:
        yield made


async def test_an_agent_is_derived_from_the_team_above_and_briefed(starts: Starts):
    assert await starts.service.started_ic(starts.store.place(A)) == str(starts.top)

    ran = await starts.service.start_agent(A, "ic", str(starts.top), "Build it")

    assert ran
    [derived] = starts.launcher.derived
    assert (derived.parent_id, derived.conversation_id) == (starts.top, starts.agent)
    assert "Build it" in derived.task and "Build the parser" in derived.task
    assert str(starts.agent) in starts.org_state.run_state().marked(TOLD)
    place = starts.store.place(A)
    assert place is not None and place.agents[0].kind == "openhands"


async def test_an_agent_that_runs_already_is_not_started_again(starts: Starts):
    await starts.service.start_agent(A, "ic", str(starts.top), "Build it")

    assert await starts.service.start_agent(A, "ic", str(starts.top), "Build it")

    assert len(starts.launcher.derived) == 1


async def test_a_start_that_fails_marks_the_agent_lost_and_tells_of_it(
    starts: Starts,
):
    starts.launcher.derive_error = OSError("disk full")

    ran = await starts.service.start_agent(A, "ic", str(starts.top), "Build it")

    assert not ran
    place = starts.store.place(A)
    assert place is not None and place.agents[0].status == "lost"
    [(place_id, role, reason)] = starts.failures
    assert (place_id, role) == (A, "ic") and "disk full" in reason


async def test_an_agent_with_nobody_to_derive_it_from_is_not_started(
    starts: Starts,
):
    assert not await starts.service.start_agent(A, "ic", None, "Build it")

    assert starts.launcher.derived == []
    assert len(starts.failures) == 1


async def test_an_agent_started_while_the_org_is_paused_is_paused_at_once(
    starts: Starts,
):
    starts.org_state.hold("human")

    assert await starts.service.start_agent(A, "ic", str(starts.top), "Build it")

    assert starts.launcher.paused == [starts.agent]
    assert str(starts.agent) in starts.org_state.running


async def test_a_briefing_is_sent_to_an_agent_that_runs_already(starts: Starts):
    starts.launcher.known[starts.agent] = ConversationExecutionStatus.IDLE

    await starts.service.brief(A, "ic", "Now build the lexer")

    [text] = starts.launcher.messages_to(str(starts.agent))
    assert "Now build the lexer" in text


def test_the_team_a_session_is_opened_at_works_in_the_checkout_it_has(
    starts: Starts,
):
    top = starts.store.place(ROOT)
    below = starts.store.place(A)
    assert top is not None and below is not None

    assert not starts.service.in_place(below)
    assert starts.service.derived_from(below) == ROOT


def test_a_conversation_is_called_by_its_team_its_session_and_its_member():
    assert agent_title("API", "IC") == "API (IC)"
    assert agent_title("API", "IC", "Faster parser") == "API · Faster parser (IC)"
    assert (
        agent_title("API", "Manager", "API", division_lead=True)
        == "API (Division Lead Manager)"
    )
