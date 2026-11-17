"""What a session's agents use of their models, and the limits on it."""

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.director import launch
from openhands.agent_server.director.launch import ConversationTeamLauncher
from openhands.agent_server.director.models import (
    DirectorSettingsUpdate,
    OpenSessionRequest,
    Session,
    Usage,
)
from openhands.agent_server.director.run_state import USE
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import LIMIT_WAITS
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import Conflict
from openhands.sdk.workspace import LocalWorkspace
from tests.agent_server.director.conftest import FakeLauncher
from tests.agent_server.director.test_director_limits import started
from tests.agent_server.director.test_director_tools import (
    ic,
    node,
    split_root,
)


API = "services.api"


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher) -> AsyncIterator[DirectorService]:
    """An org of API and Web whose timer never fires by itself."""
    service = await started(repo, launcher, stall_poll=3600)
    await split_root(service)
    yield service
    await service.aclose()


def uses(launcher: FakeLauncher, conversation: str | None, *used: float) -> None:
    assert conversation is not None
    tokens_in, tokens_out, cost = used
    launcher.used[UUID(conversation)] = Usage(
        input_tokens=int(tokens_in), output_tokens=int(tokens_out), cost=cost
    )


async def counted(service: DirectorService) -> Session:
    """The director adds up what was used, as its timer does."""
    await service._count_use()
    await service.settled()
    [session] = service.store.sessions()
    return session


async def test_what_the_agents_of_a_session_used_is_added_up(
    service: DirectorService, launcher: FakeLauncher
):
    uses(launcher, ic(service), 1000, 200, 0.5)
    uses(launcher, ic(service, API), 300, 50, 0.25)

    session = await counted(service)

    assert session.usage == Usage(input_tokens=1300, output_tokens=250, cost=0.75)
    assert session.usage.tokens == 1550

    # Only what was used since is added.
    uses(launcher, ic(service, API), 400, 80, 0.3)
    session = await counted(service)

    assert session.usage == Usage(input_tokens=1400, output_tokens=280, cost=0.8)


async def test_a_count_that_begins_again_takes_nothing_away(
    service: DirectorService, launcher: FakeLauncher
):
    uses(launcher, ic(service), 1000, 200, 0.5)
    await counted(service)

    # A server that starts again counts its own calls from nothing.
    uses(launcher, ic(service), 100, 10, 0.25)
    session = await counted(service)
    assert session.usage == Usage(input_tokens=1000, output_tokens=200, cost=0.5)

    uses(launcher, ic(service), 150, 30, 0.5)
    session = await counted(service)
    assert session.usage == Usage(input_tokens=1050, output_tokens=220, cost=0.75)


async def test_a_session_at_its_limit_rests_until_the_human_lets_it_go_on(
    service: DirectorService, launcher: FakeLauncher
):
    service.update_settings(DirectorSettingsUpdate(session_token_limit=1000))
    api = ic(service, API)
    assert api is not None
    launcher.report(api, ConversationExecutionStatus.RUNNING)
    uses(launcher, ic(service), 700, 100, 0)
    session = await counted(service)
    assert session.token_allowance == 1000
    assert node(service, "root").stopped is None

    uses(launcher, api, 250, 50, 0)
    session = await counted(service)

    top = node(service, "root")
    assert top.stopped is not None and top.stopped.running == [api]
    assert UUID(api) in launcher.paused
    [asked] = top.escalations
    assert (asked.id, asked.held_by) == (LIMIT_WAITS, None)
    assert "1,100 tokens (950 in, 150 out)" in asked.detail

    # Counting again asks nothing more.
    await counted(service)
    assert len(node(service, "root").escalations) == 1

    service.answer_escalation(top.id, LIMIT_WAITS, "Go on, but finish the API.")
    await service.settled()

    top = node(service, "root")
    assert top.stopped is None and top.escalations == []
    [session] = service.store.sessions()
    assert session.token_allowance == 2100
    assert (UUID(api), None) in launcher.resumed
    assert (
        "The human lets it go on, and says:\nGo on, but finish the API."
        in (launcher.messages_to(ic(service))[-1])
    )


async def test_nothing_rests_while_the_org_has_no_limit(
    service: DirectorService, launcher: FakeLauncher
):
    uses(launcher, ic(service), 5_000_000, 100_000, 90.0)

    session = await counted(service)

    assert session.token_allowance == 0
    assert node(service, "root").stopped is None
    assert node(service, "root").escalations == []


async def test_a_closed_sessions_count_goes_with_it(
    service: DirectorService, launcher: FakeLauncher
):
    uses(launcher, ic(service), 1000, 200, 0.5)
    session = await counted(service)
    assert session.id in service._runs().marked(USE)

    service.close_session(session.id)
    await service.settled()

    assert service._runs().marked(USE) == {}


async def test_no_more_sessions_are_opened_than_the_settings_allow(
    service: DirectorService,
):
    service.update_settings(DirectorSettingsUpdate(max_open_sessions=1))

    with pytest.raises(Conflict, match="1 sessions are open"):
        await service.open_session(
            OpenSessionRequest(text="Add refunds", node_id="root")
        )

    service.update_settings(DirectorSettingsUpdate(max_open_sessions=2))
    opened = await service.open_session(
        OpenSessionRequest(text="Add refunds", node_id="root")
    )
    await service.settled()

    assert len(service.store.sessions()) == 2
    assert opened.session.token_allowance == 0


async def test_a_team_started_from_a_profile_spends_under_its_parents_caps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    parent_id, child_id = uuid4(), uuid4()
    started_with = []
    parent = SimpleNamespace(
        stored=SimpleNamespace(
            workspace=LocalWorkspace(working_dir=str(tmp_path)), max_iterations=40
        ),
        get_conversation=lambda: SimpleNamespace(max_budget_per_run=2.5),
    )

    async def get_event_service(conversation_id: UUID):
        return parent if conversation_id == parent_id else None

    async def start_conversation(request, **_):
        started_with.append(request)
        info = SimpleNamespace(
            id=child_id,
            workspace=SimpleNamespace(working_dir=str(tmp_path)),
            agent=object(),
        )
        return info, True

    async def update_conversation(*_):
        return True

    conversations = SimpleNamespace(
        get_event_service=get_event_service,
        start_conversation=start_conversation,
        update_conversation=update_conversation,
    )
    monkeypatch.setattr(
        launch,
        "get_agent_profile_store",
        lambda: SimpleNamespace(load=lambda _: SimpleNamespace(id=uuid4())),
    )

    launcher = ConversationTeamLauncher(cast(ConversationService, conversations))
    await launcher.start_from_profile(
        "claude-code",
        parent_id,
        child_id,
        [],
        title="API (IC)",
        task="Serve the orders API.",
        isolation="shared",
    )

    [request] = started_with
    assert (request.max_iterations, request.max_budget_per_run) == (40, 2.5)
