"""Tests for sessions that something outside the org opens: an event an
automation received, with nobody there to read the session or close it."""

from pathlib import Path

import pytest

from openhands.agent_server.director.models import (
    DirectorSettingsUpdate,
    EventSession,
    EventSource,
    OpenSessionRequest,
    OrgEventRequest,
    OrgSettings,
    SplitChild,
)
from openhands.agent_server.director.org_store import team_id_of
from openhands.agent_server.director.routing import USAGE_ID
from openhands.agent_server.director.service import DirectorService
from openhands.sdk.errors import Conflict, Unavailable
from tests.agent_server.director.conftest import (
    API,
    DB,
    FakeLauncher,
    landed,
    node,
    propose_and_approve,
    session_accepted,
    split_root,
    started,
)


TO_DB = f"Teams: {DB}\nReason: It is about the schema."
TICKET = {"issue": {"key": "SHOP-7", "fields": {"summary": "Orders are slow"}}}


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher):
    """An org of API (with DB below it) and Web."""
    service = await started(repo, launcher)
    await split_root(service)
    await propose_and_approve(
        service,
        API,
        SplitChild(title="DB", path="services/api/db", task="Own the schema"),
    )
    await landed(service)
    yield service
    await service.aclose()


async def happened(
    service: DirectorService, text: str = "Add an index on orders.", **rest
) -> EventSession:
    rest.setdefault("source", EventSource(name="Jira", event_key="SHOP-7"))
    opened = await service.session_lifecycle.open_from_event(
        OrgEventRequest(text=text, **rest)
    )
    await service.work_queue.settled()
    return opened


def briefing(launcher: FakeLauncher, conversation_id: str) -> str:
    """What the IC of a session was started with."""
    [start] = [
        start
        for start in launcher.derived
        if str(start.conversation_id) == conversation_id
    ]
    return start.task


def open_sessions(service: DirectorService) -> list[str]:
    return [session.id for session in service.org_load.org.sessions()]


async def test_an_event_opens_a_session_at_the_team_the_director_chooses(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = TO_DB

    opened = await happened(service, payload=TICKET)

    # Chosen as for a message the human sends with `auto`.
    assert team_id_of(node(service, opened.node_id)) == DB
    assert (opened.team, opened.chosen_by) == ("DB", "model")
    assert opened.reason == "It is about the schema."
    assert not opened.duplicate
    [session] = [
        session
        for session in service.org_load.org.sessions()
        if session.id == opened.session_id
    ]
    assert (session.source.kind, session.source.name) == ("automation", "Jira")
    assert session.source.event_key == "SHOP-7"
    # Its IC is given the request, who sent it and the event itself.
    task = briefing(launcher, opened.conversation_id)
    assert "Add an index on orders." in task
    assert "No human sent this request: Jira did" in task
    assert '"key": "SHOP-7"' in task
    assert "closed once its work is accepted" in task
    [kept] = service.team_history.routed_requests()
    assert kept.source.kind == "automation"


async def test_a_session_the_human_opens_has_the_human_as_its_source(
    service: DirectorService,
):
    opened = await service.session_lifecycle.open_session(
        OpenSessionRequest(text="Add refunds.", node_id=API)
    )

    assert opened.session.source.kind == "human"
    assert not opened.session.source.unattended


async def test_an_event_sent_twice_opens_one_session(
    service: DirectorService, launcher: FakeLauncher
):
    first = await happened(service)
    before = open_sessions(service)

    again = await happened(service)

    assert again.duplicate
    assert (again.session_id, again.node_id) == (first.session_id, first.node_id)
    assert again.conversation_id == first.conversation_id
    assert open_sessions(service) == before


async def test_events_without_a_key_each_open_a_session(service: DirectorService):
    first = await happened(service, source=EventSource(name="nightly"))
    second = await happened(service, source=EventSource(name="nightly"))

    assert first.session_id != second.session_id
    assert not second.duplicate


async def test_a_long_event_is_shown_shortened(
    service: DirectorService, launcher: FakeLauncher
):
    opened = await happened(service, payload={"body": "x" * 20_000})

    task = briefing(launcher, opened.conversation_id)
    assert "(shortened)" in task
    assert "x" * 5_000 in task and "x" * 7_000 not in task


async def test_an_event_is_refused_while_the_org_is_paused(service: DirectorService):
    await service.org_run.pause()
    before = open_sessions(service)

    with pytest.raises(Unavailable, match="paused"):
        await happened(service)

    assert open_sessions(service) == before
    # Sent again once the org is resumed, it opens its session.
    await service.org_run.resume()
    assert not (await happened(service)).duplicate


async def test_an_event_is_refused_where_there_is_no_org(
    repo: Path, launcher: FakeLauncher
):
    service = DirectorService(
        repo, save_delay=0, remembered=repo.parent / "state" / "director.json"
    )
    await service.start(launcher)

    with pytest.raises(Unavailable, match="no org"):
        await happened(service)

    await service.aclose()


async def test_an_event_opens_no_session_beyond_the_most_the_org_may_have(
    service: DirectorService,
):
    most = len(open_sessions(service))
    service.org_settings.update_settings(DirectorSettingsUpdate(max_open_sessions=most))

    with pytest.raises(Conflict, match="at once"):
        await happened(service)

    # Nothing was kept of it: it opens its session once there is room.
    service.org_settings.update_settings(
        DirectorSettingsUpdate(max_open_sessions=most + 1)
    )
    assert not (await happened(service)).duplicate


async def test_a_session_an_event_opened_is_closed_once_its_work_is_accepted(
    service: DirectorService,
):
    opened = await happened(service, path="services/api")
    assert service.org_settings.org_settings_state().close_unattended_sessions

    await session_accepted(service, opened.node_id, opened.conversation_id)

    assert opened.session_id not in open_sessions(service)
    # The event is still known: sent again, it opens no second session.
    again = await happened(service, path="services/api")
    assert (again.duplicate, again.session_id) == (True, opened.session_id)
    assert (again.node_id, again.conversation_id) == ("", "")


async def test_the_org_can_keep_such_a_session_open(service: DirectorService):
    state = service.org_settings.update_org_settings(
        OrgSettings(close_unattended_sessions=False)
    )
    assert not state.close_unattended_sessions
    opened = await happened(service, path="services/api")

    await session_accepted(service, opened.node_id, opened.conversation_id)

    assert opened.session_id in open_sessions(service)
    review = node(service, opened.node_id).review
    assert review is not None and review.state == "accepted"


async def test_a_session_the_human_opened_stays_open_when_its_work_is_accepted(
    service: DirectorService,
):
    opened = await service.session_lifecycle.open_session(
        OpenSessionRequest(text="Add refunds.", node_id=API)
    )
    await service.work_queue.settled()
    top = opened.session.teams[0].id

    await session_accepted(service, top, opened.conversation_id)

    assert opened.session.id in open_sessions(service)
