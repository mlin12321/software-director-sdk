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
from openhands.agent_server.director.pricing import priced
from openhands.agent_server.director.run_state import USE
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import LIMIT_WAITS
from openhands.agent_server.models import StartConversationRequest
from openhands.sdk import LLM, Agent
from openhands.sdk.agent import ACPAgent
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import Conflict, NotFound
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm.utils.metrics import Metrics, TokenUsage
from openhands.sdk.profiles.resolver import ProfileNotFound
from openhands.sdk.security.confirmation_policy import NeverConfirm
from openhands.sdk.workspace import LocalWorkspace
from tests.agent_server.director.conftest import (
    FakeLauncher,
    ic,
    node,
    split_root,
    started,
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
    [session] = service.org.sessions()
    return session


async def test_what_the_agents_of_a_session_used_is_added_up(
    service: DirectorService, launcher: FakeLauncher
):
    uses(launcher, ic(service), 1000, 200, 0.5)
    uses(launcher, ic(service, API), 300, 50, 0.25)

    session = await counted(service)

    assert session.usage == Usage(input_tokens=1300, output_tokens=250, cost=0.75)

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
    service.org_settings.update_settings(DirectorSettingsUpdate(session_cost_limit=2.0))
    api = ic(service, API)
    assert api is not None
    launcher.report(api, ConversationExecutionStatus.RUNNING)
    uses(launcher, ic(service), 700, 100, 1.5)
    session = await counted(service)
    assert session.cost_allowance == 2.0
    assert node(service, "root").stopped is None

    uses(launcher, api, 250, 50, 0.75)
    session = await counted(service)

    top = node(service, "root")
    assert top.stopped is not None and top.stopped.running == [api]
    assert UUID(api) in launcher.paused
    [asked] = top.escalations
    assert (asked.id, asked.held_by) == (LIMIT_WAITS, None)
    assert asked.title == "'Checkout' reached its limit of $2.00"
    assert "used $2.25 (950 tokens in, 150 out)" in asked.detail

    # Counting again asks nothing more.
    await counted(service)
    assert len(node(service, "root").escalations) == 1

    service.answer_escalation(top.id, LIMIT_WAITS, "Go on, but finish the API.")
    await service.settled()

    top = node(service, "root")
    assert top.stopped is None and top.escalations == []
    [session] = service.org.sessions()
    assert session.cost_allowance == 4.25
    assert (UUID(api), None) in launcher.resumed
    assert (
        "The human lets it go on, and says:\nGo on, but finish the API."
        in (launcher.messages_to(ic(service))[-1])
    )


async def test_an_open_session_follows_a_limit_that_is_changed(
    service: DirectorService, launcher: FakeLauncher
):
    service.org_settings.update_settings(DirectorSettingsUpdate(session_cost_limit=2.0))
    uses(launcher, ic(service), 700, 100, 1.5)
    assert (await counted(service)).cost_allowance == 2.0

    service.org_settings.update_settings(DirectorSettingsUpdate(session_cost_limit=5.0))
    uses(launcher, ic(service), 900, 150, 2.25)
    session = await counted(service)

    assert session.cost_allowance == 5.0
    assert node(service, "root").stopped is None


async def test_nothing_rests_while_the_org_has_no_limit(
    service: DirectorService, launcher: FakeLauncher
):
    uses(launcher, ic(service), 5_000_000, 100_000, 90.0)

    session = await counted(service)

    assert session.cost_allowance == 0
    assert node(service, "root").stopped is None
    assert node(service, "root").escalations == []


async def test_a_closed_sessions_count_goes_with_it(
    service: DirectorService, launcher: FakeLauncher
):
    uses(launcher, ic(service), 1000, 200, 0.5)
    session = await counted(service)
    assert session.id in service.org_state.run_state().marked(USE)

    service.close_session(session.id)
    await service.settled()

    assert service.org_state.run_state().marked(USE) == {}


async def test_no_more_sessions_are_opened_than_the_settings_allow(
    service: DirectorService,
):
    service.org_settings.update_settings(DirectorSettingsUpdate(max_open_sessions=1))

    with pytest.raises(Conflict, match="allow 1 open session at once"):
        await service.open_session(
            OpenSessionRequest(text="Add refunds", node_id="root")
        )

    service.org_settings.update_settings(DirectorSettingsUpdate(max_open_sessions=2))
    opened = await service.open_session(
        OpenSessionRequest(text="Add refunds", node_id="root")
    )
    await service.settled()

    assert len(service.org.sessions()) == 2
    assert opened.session.cost_allowance == 0


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
            agent=Agent(llm=LLM(model="gpt-4o", usage_id="test-llm"), tools=[]),
        )
        return info, True

    async def update_conversation(*_):
        return True

    conversations = SimpleNamespace(
        get_event_service=get_event_service,
        start_conversation=start_conversation,
        update_conversation=update_conversation,
        stores=SimpleNamespace(
            agent_profiles=SimpleNamespace(load=lambda _: SimpleNamespace(id=uuid4()))
        ),
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
    assert request.parent_conversation_id == parent_id

    # Started again in the checkout it had, it names no parent: one in
    # another workspace is refused.
    own = tmp_path / "own"
    await launcher.start_from_profile(
        "claude-code",
        parent_id,
        child_id,
        [],
        title="API (IC)",
        task="Serve the orders API.",
        isolation="worktree",
        checkout=str(own),
    )
    again = started_with[1]
    assert again.parent_conversation_id is None and not again.worktree
    assert again.workspace.working_dir == str(own)


def test_an_agent_is_known_by_the_model_it_is_configured_with():
    own = Agent(llm=LLM(model="openai/gpt-5.2", usage_id="test-llm"), tools=[])
    claude = ACPAgent(acp_command=["claude-agent-acp"], acp_model="claude-opus-5")
    default = ACPAgent(acp_command=["claude-agent-acp"])

    assert launch._kind_of(own)["model"] == "openai/gpt-5.2"
    assert launch._kind_of(claude)["model"] == "claude-opus-5"
    # Left on its own default, which the server cannot tell.
    assert launch._kind_of(default)["model"] is None


async def test_the_org_says_which_model_each_started_agent_runs_on(
    repo: Path, launcher: FakeLauncher
):
    launcher.model = "openai/gpt-5.2"
    service = await started(repo, launcher)

    top = node(service, "root")
    assert [agent.model for agent in top.agents if agent.role == "ic"] == [
        "openai/gpt-5.2"
    ]
    await service.aclose()


def counted_as(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def metered(model: str, cost: float, **tokens: int) -> Metrics:
    return Metrics(
        model_name=model,
        accumulated_cost=cost,
        accumulated_token_usage=counted_as(**tokens),
    )


async def used_by(agent: object, *metrics: Metrics) -> Usage:
    """What the launcher reads of a conversation whose agent ran on these."""
    conversation = uuid4()

    async def get_event_service(_: UUID):
        async def get_state():
            return SimpleNamespace(
                agent=agent,
                stats=SimpleNamespace(
                    usage_to_metrics={str(at): m for at, m in enumerate(metrics)}
                ),
            )

        return SimpleNamespace(get_state=get_state)

    conversations = SimpleNamespace(
        get_event_service=get_event_service, loaded_event_service=lambda _: None
    )
    launcher = ConversationTeamLauncher(cast(ConversationService, conversations))
    return (await launcher.usage([conversation]))[conversation]


async def test_an_agent_derived_from_another_is_its_own_in_what_the_human_did_there():
    asked: dict = {}

    async def start_child_conversation(*_, **options):
        asked.update(options)
        return SimpleNamespace(workspace="/repo", isolation="shared")

    async def get_event_service(_: UUID):
        return SimpleNamespace(
            get_conversation=lambda: SimpleNamespace(agent=ACPAgent(acp_command=["x"]))
        )

    conversations = SimpleNamespace(
        start_child_conversation=start_child_conversation,
        get_event_service=get_event_service,
    )
    launcher = ConversationTeamLauncher(cast(ConversationService, conversations))

    await launcher.start_derived(
        uuid4(), uuid4(), title="API (IC)", task="Serve orders.", isolation="shared"
    )

    # The human wrote in the conversation it comes from, not in it, and is
    # not there to confirm what it does.
    assert asked["extra_tags"] == {"startedby": "director", "humanjoined": None}
    assert asked["confirmation_policy"] == NeverConfirm()


async def test_a_conversation_that_cannot_be_started_is_refused_as_any_other_is():
    async def start_conversation(*_, **__):
        raise ProfileNotFound("No profile 'gone'")

    conversations = SimpleNamespace(start_conversation=start_conversation)
    launcher = ConversationTeamLauncher(cast(ConversationService, conversations))
    request = StartConversationRequest(
        agent=ACPAgent(acp_command=["x"]), workspace=LocalWorkspace(working_dir="/r")
    )

    # Not a failure of the server: the request names what is not there.
    with pytest.raises(NotFound, match="gone"):
        await launcher.start_root(request, [], title="Shop (IC)")


async def test_counting_use_does_not_keep_an_idle_conversation_loaded():
    conversation, loads = uuid4(), 0

    async def get_event_service(_: UUID):
        nonlocal loads
        loads += 1

        async def get_state():
            stats = SimpleNamespace(
                usage_to_metrics={"0": metered("claude-sonnet-4-5", 0.5)}
            )
            return SimpleNamespace(agent=object(), stats=stats)

        return SimpleNamespace(get_state=get_state)

    # It was evicted for being idle: nothing of it is loaded.
    conversations = SimpleNamespace(
        get_event_service=get_event_service, loaded_event_service=lambda _: None
    )
    launcher = ConversationTeamLauncher(cast(ConversationService, conversations))

    first = await launcher.usage([conversation])
    again = await launcher.usage([conversation])

    assert first == again and first[conversation].cost == 0.5
    assert loads == 1


RUN = {
    "prompt_tokens": 298,
    "completion_tokens": 10_744,
    "cache_read_tokens": 463_280,
    "cache_write_tokens": 32_563,
}


async def test_an_agent_with_its_own_model_says_its_cost_and_its_cache_counts():
    claude = ACPAgent(acp_command=["claude-agent-acp"])

    used = await used_by(claude, metered("acp-managed", 0.97, **RUN))

    assert used == Usage(input_tokens=496_141, output_tokens=10_744, cost=0.97)


async def test_a_cost_the_model_reported_is_taken_as_it_is():
    used = await used_by(object(), metered("claude-sonnet-4-5", 0.97, **RUN))

    assert used == Usage(input_tokens=298, output_tokens=10_744, cost=0.97)


async def test_a_model_that_reported_no_cost_is_priced_as_the_model_its_name_ends_in():
    # Behind a gateway the input it reports has the cached tokens in it.
    tokens = {**RUN, "prompt_tokens": 500_000}
    gateway = "litellm_proxy/anthropic/claude-sonnet-4-5"

    used = await used_by(object(), metered(gateway, 0, **tokens))
    uncached = await used_by(
        object(),
        metered(gateway, 0, prompt_tokens=500_000, completion_tokens=10_744),
    )

    assert used.unpriced_tokens == 0
    assert used.cost == priced("claude-sonnet-4-5", counted_as(**tokens))
    # What was read from the cache is cheaper than what was not.
    assert 0 < used.cost < uncached.cost


async def test_tokens_of_a_model_with_no_price_are_counted_apart():
    claude = ACPAgent(acp_command=["claude-agent-acp"])
    local = metered("openai/a-model-of-our-own", 0, **RUN)

    used = await used_by(object(), local, metered("gpt-4o", 0.5, prompt_tokens=10))
    silent = await used_by(claude, metered("acp-managed", 0, **RUN))

    assert used == Usage(
        input_tokens=308, output_tokens=10_744, cost=0.5, unpriced_tokens=11_042
    )
    assert (silent.cost, silent.unpriced_tokens) == (0, 506_885)


def test_a_model_is_looked_up_under_each_of_its_names():
    tokens = counted_as(prompt_tokens=1000, completion_tokens=100)

    assert priced("openai/claude-sonnet-4-5", tokens) == priced(
        "claude-sonnet-4-5", tokens
    )
    assert priced("a-model-of-our-own", tokens) is None


async def test_a_session_set_going_while_the_org_is_paused_goes_on_when_it_resumes(
    service: DirectorService, launcher: FakeLauncher
):
    service.org_settings.update_settings(DirectorSettingsUpdate(session_cost_limit=2.0))
    api = ic(service, API)
    assert api is not None
    launcher.report(api, ConversationExecutionStatus.RUNNING)
    uses(launcher, api, 250, 50, 2.75)
    await counted(service)
    top = node(service, "root")
    assert top.stopped is not None and top.stopped.running == [api]
    await service.pause()
    await service.settled()

    service.answer_escalation(top.id, LIMIT_WAITS, "Go on.")
    await service.settled()
    await service.resume()
    await service.settled()

    # The agent that was cut off in its turn runs again.
    assert (UUID(api), None) in launcher.resumed


async def test_the_last_shortening_of_a_conversations_history_is_found():
    asked: dict = {}
    event = Condensation(
        forgotten_event_ids={"e1", "e2"}, summary="So far.", llm_response_id="r1"
    )

    async def search_events(**query):
        asked.update(query)
        return SimpleNamespace(items=[event])

    async def get_event_service(_: UUID):
        return SimpleNamespace(search_events=search_events)

    conversations = SimpleNamespace(get_event_service=get_event_service)
    launcher = ConversationTeamLauncher(cast(ConversationService, conversations))

    assert await launcher.condensed(uuid4()) == str(event.id)
    assert (asked["kind"], asked["limit"]) == ("Condensation", 1)


async def test_a_conversation_whose_history_was_never_shortened_says_so():
    async def search_events(**_):
        return SimpleNamespace(items=[])

    async def get_event_service(conversation_id: UUID):
        return (
            None
            if conversation_id == GONE
            else SimpleNamespace(search_events=search_events)
        )

    conversations = SimpleNamespace(get_event_service=get_event_service)
    launcher = ConversationTeamLauncher(cast(ConversationService, conversations))

    assert await launcher.condensed(uuid4()) is None
    assert await launcher.condensed(GONE) is None


GONE = uuid4()
