"""Messages to agents by themselves: a store of teams, a run state in
memory and a launcher that records what it is sent."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from openhands.agent_server.director.environment.git import (
    GitCodingEnvironment,
)
from openhands.agent_server.director.environment.git.git_store import GitStore
from openhands.agent_server.director.intents import LaunchMember
from openhands.agent_server.director.models import Stopped
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.run_state import RUNNING, TOLD, Intent
from openhands.agent_server.director.service.agent_launch_service import (
    AgentLaunchService,
)
from openhands.agent_server.director.service.message_delivery_service import (
    MessageDeliveryService,
)
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.resumable_work_service import Held
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import (
    FakeLauncher,
    open_session_in,
    shipped_type,
    small_tree,
)
from tests.agent_server.director.test_team_documents import RootAboveApi


SESSION = "s1"
A = f"a@{SESSION}"


class Delivery:
    """A delivery service over a small tree, with one agent at team a."""

    def __init__(self, directory: Path) -> None:
        self.store = OrgStore(shipped_type)
        for node in small_tree():
            self.store.upsert(node)
        open_session_in(self.store, "root", SESSION)
        self.agent = uuid4()
        self.store.set_agent(A, "ic", conversation_id=str(self.agent))
        self.launcher = FakeLauncher()
        self.agent_launch = AgentLaunchService()
        self.agent_launch.launcher = self.launcher
        self.launcher.known[self.agent] = ConversationExecutionStatus.IDLE
        self.context: dict[str, str] = {
            "status": "idle",
            "task": "Build the parser",
        }
        self.taken_up: list[tuple[int, str]] = []
        self.queue = WorkQueueService(default_lane="org")
        self.org_state = OrgStateService(
            repo_dir=lambda: directory,
            clock=lambda: datetime.now(UTC),
            environment=GitCodingEnvironment(),
        )
        self.service = MessageDeliveryService(
            org=lambda: self.store,
            org_state=self.org_state,
            org_save=OrgSaveService(
                environment=GitCodingEnvironment(),
                org=lambda: self.store,
                repo_dir=lambda: directory,
                team_memory=lambda: GitStore(directory),
                run_state=self.org_state.run_state,
                work_queue=self.queue,
                delay=60,
            ),
            work_queue=self.queue,
            team_types=TeamTypeService(
                config_file=None,
                documents=None,  # type: ignore[arg-type]
                tree=RootAboveApi(),
                org=lambda: self.store,
            ),
            agent_launch=self.agent_launch,
            context_of=lambda place, role: dict(self.context),
            lane_of=lambda intent: intent.node_id,
            take_up=lambda intent_id, lane: self.taken_up.append((intent_id, lane)),
            retry=0,
        )

    def sent(self) -> list[str]:
        return self.launcher.messages_to(str(self.agent))

    def held(self) -> list[str]:
        return [held.text for held in self.org_state.run_state().held()]


@pytest.fixture
async def delivery(repo: Path) -> AsyncIterator[Delivery]:
    made = Delivery(repo)
    async with made.queue:
        yield made


def intent_for(delivery: Delivery, node_id: str) -> Intent:
    return delivery.org_state.run_state().begin(
        node_id, LaunchMember(task="Build it", role="ic")
    )


async def test_a_message_is_sent_with_the_context_the_agent_was_not_told(
    delivery: Delivery,
):
    delivery.service.notify(str(delivery.agent), "Go on.")
    await delivery.queue.settled()

    [first] = delivery.sent()
    assert "Go on." in first and "Build the parser" in first
    assert delivery.held() == []

    delivery.service.notify(str(delivery.agent), "And again.")
    await delivery.queue.settled()

    assert "Build the parser" not in delivery.sent()[1]


async def test_what_an_agent_wrote_itself_is_not_sent_back_to_it(delivery: Delivery):
    delivery.service.notify(str(delivery.agent), "Go on.")
    await delivery.queue.settled()
    delivery.context["task"] = "Build the lexer"

    delivery.service.knows(A, "ic", "task")
    delivery.service.notify(str(delivery.agent), "And again.")
    await delivery.queue.settled()

    assert "Build the lexer" not in delivery.sent()[1]
    assert "lexer" in delivery.org_state.run_state().marked(TOLD)[str(delivery.agent)]


async def test_a_message_waits_while_the_org_is_paused(delivery: Delivery):
    delivery.org_state.hold("human")

    delivery.service.notify(str(delivery.agent), "Go on.")
    await delivery.queue.settled()

    assert delivery.sent() == []
    assert len(delivery.held()) == 1


async def test_a_message_to_a_stopped_team_waits(delivery: Delivery):
    delivery.store.set_stopped(
        A, Stopped(at="2026-01-02T00:00:00+00:00", reason="Wait.")
    )

    delivery.service.notify(str(delivery.agent), "Go on.")
    await delivery.queue.settled()

    assert delivery.sent() == []
    assert len(delivery.held()) == 1


async def test_notices_wait_for_the_turn_of_an_agent_that_runs_its_own_tools(
    delivery: Delivery,
):
    delivery.store.set_agent(A, "ic", kind="acp")
    delivery.service.note_running(str(delivery.agent), A, True)

    delivery.service.notify(str(delivery.agent), "First notice.")
    delivery.service.notify(str(delivery.agent), "Second notice.")
    await delivery.queue.settled()
    assert delivery.sent() == []

    delivery.service.note_running(str(delivery.agent), A, False)
    await delivery.queue.settled()

    [together] = delivery.sent()
    assert "First notice." in together and "Second notice." in together
    assert delivery.held() == []


async def test_what_the_human_says_does_not_wait_for_the_turn_to_end(
    delivery: Delivery,
):
    delivery.store.set_agent(A, "ic", kind="acp")
    delivery.service.note_running(str(delivery.agent), A, True)

    delivery.service.notify(str(delivery.agent), "Stop that.", at_once=True)
    await delivery.queue.settled()

    assert len(delivery.sent()) == 1


async def test_a_step_waits_for_a_team_to_be_quiet_and_holds_its_messages(
    delivery: Delivery,
):
    intent = intent_for(delivery, A)
    delivery.service.note_running(str(delivery.agent), A, True)

    with pytest.raises(Held):
        delivery.service.wait_until_quiet(intent, [A])
    assert delivery.service.is_settling(A)
    delivery.service.notify(str(delivery.agent), "Go on.")
    await delivery.queue.settled()
    assert delivery.sent() == []

    delivery.service.note_running(str(delivery.agent), A, False)
    assert delivery.taken_up == [(intent.id, A)]
    assert delivery.org_state.run_state().marked(RUNNING) == {}

    delivery.service.let_go(intent)
    await delivery.queue.settled()

    assert len(delivery.sent()) == 1
    assert not delivery.service.is_settling(A)


async def test_an_agent_whose_conversation_is_gone_is_marked_lost(
    delivery: Delivery,
):
    del delivery.launcher.known[delivery.agent]

    delivery.service.notify(str(delivery.agent), "Go on.")
    await delivery.queue.settled()

    place = delivery.store.place(A)
    assert place is not None
    assert [agent.status for agent in place.agents] == ["lost"]
    assert delivery.held() == []


def test_a_message_to_no_conversation_is_not_sent(delivery: Delivery):
    delivery.service.notify(None, "Nobody reads this.")

    assert delivery.held() == []
