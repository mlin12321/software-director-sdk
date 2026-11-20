"""The director HTTP API, served by the real app."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from openhands.agent_server import director_router
from openhands.agent_server.api import create_app
from openhands.agent_server.config import Config
from openhands.agent_server.director.git_store import GitStore
from openhands.agent_server.director.models import (
    OrgNodeInput,
    Report,
    TeamEvent,
    ThreadMessage,
)
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    save_org,
)
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.service import (
    DirectorService,
    agent_conversation,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import NotFound
from tests.agent_server.director.conftest import (
    FakeLauncher,
    make_team,
    open_session_in,
    root_conversation,
    run_git,
    small_tree,
    take_in,
)


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


def make_client(repo_dir: Path, **config) -> TestClient:
    app = create_app(server_config(repo_dir, **config))
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def saved_tree(repo: Path) -> Path:
    """A repository whose hidden org ref holds ``small_tree()``."""
    store = OrgStore()
    store.replace_all(small_tree())
    save_org(create_org_board(repo), store)
    return repo


@pytest.fixture
def client(saved_tree: Path) -> TestClient:
    return make_client(saved_tree)


def test_returns_a_team_with_its_ancestors_and_children(client: TestClient):
    response = client.get("/api/director/org/nodes/a")

    assert response.status_code == 200
    body = response.json()
    assert body["node"]["id"] == "a"
    assert [node["id"] for node in body["ancestors"]] == ["root"]
    assert [node["id"] for node in body["children"]] == ["a1", "a2"]


def test_an_unknown_team_is_not_found_on_every_team_route(client: TestClient):
    responses = [
        client.get("/api/director/org/nodes/missing"),
        client.get("/api/director/org/nodes/missing/memory"),
        client.get("/api/director/org/nodes/missing/ownership"),
        client.get("/api/director/org/tree-stream", params={"node": "missing"}),
    ]

    assert [response.status_code for response in responses] == [404] * 4


def test_no_session_is_open_in_a_saved_tree_and_an_unknown_one_is_not_found(
    client: TestClient,
):
    answer = {"answer": "x"}

    assert client.get("/api/director/org/sessions").json() == []
    assert client.get("/api/director/org/sessions/s000000").status_code == 404
    assert client.delete("/api/director/org/sessions/s000000").status_code == 404
    assert (
        client.post(
            "/api/director/org/sessions/s000000/teams/a/escalations/e/answer",
            json=answer,
        ).status_code
        == 404
    )


def test_routes_require_the_session_key_when_one_is_configured(saved_tree: Path):
    app = create_app(server_config(saved_tree, session_api_keys=["secret"]))
    client = TestClient(app, raise_server_exceptions=False)

    denied = client.get("/api/director/org/nodes/root")
    allowed = client.get(
        "/api/director/org/nodes/root", headers={"X-Session-API-Key": "secret"}
    )

    assert (denied.status_code, allowed.status_code) == (401, 200)


def test_the_tree_is_empty_without_a_saved_tree_or_a_repository(
    tmp_path: Path, repo: Path
):
    for directory in (tmp_path / "missing", tmp_path, repo):
        client = make_client(directory)

        assert client.get("/api/director/org/nodes/root").status_code == 404
        assert client.get("/api/director/org/tree-stream").status_code == 404


def test_ownership_lists_a_teams_files_by_the_report_that_owns_them(repo: Path):
    (repo / "svc" / "a").mkdir(parents=True)
    (repo / "svc" / "a" / "app.py").write_text("a\n")
    store = OrgStore()
    store.replace_all(
        [make_team("root", None, path=""), make_team("a", "root", path="svc/a")]
    )
    save_org(create_org_board(repo), store)

    response = make_client(repo).get("/api/director/org/nodes/root/ownership")

    assert response.status_code == 200
    assert response.json() == {
        "node_id": "root",
        "owners": [
            {
                "node_id": "a",
                "title": "a",
                "path": "svc/a",
                "isolation": "worktree",
                "branch": "",
            }
        ],
        "files": [
            {"path": "README.md", "owner": None},
            {"path": "svc/a/app.py", "owner": "a"},
        ],
        "truncated": False,
        # Nobody works at a team of the org, so no work can collide there.
        "collisions": [],
        "collisions_checked": False,
    }


def renamed(team: OrgNodeInput, title: str) -> OrgNodeInput:
    return team.model_copy(update={"title": title})


class TestTeamMemory:
    def test_returns_what_applies_to_the_nodes_boundary_and_what_its_agents_reported(
        self, repo: Path
    ):
        store = OrgStore()
        store.replace_all(small_tree())
        save_org(create_org_board(repo), store)
        board = GitStore(repo)
        board.write("", "spec.md", "root spec")
        board.write("svc/a", "spec.md", "a spec")
        board.write("svc/b", "spec.md", "b spec")
        board.write("", "contracts.md", "a serves b")
        later = Report(at="2026-11-02T00:00:00+00:00", role="ic", text="Chose SQLite.")
        earlier = Report(
            at="2026-11-01T00:00:00+00:00", role="manager", text="Two sub-teams."
        )
        take_in(board, reports={"a": [later, earlier], "b": [later]})

        response = make_client(repo).get("/api/director/org/nodes/a/memory")

        assert response.status_code == 200
        assert response.json() == {
            "specs": [
                {
                    "path": "svc/a",
                    "spec": "a spec",
                    "readonly": False,
                    "old": False,
                    "left_by": None,
                }
            ],
            "contracts": [{"path": "", "contracts": "a serves b", "own": False}],
            "reports": [earlier.model_dump(), later.model_dump()],
            "former_teams": [],
            "history": [],
        }

    async def test_returns_the_threads_a_places_agents_take_part_in(
        self, running, repo: Path
    ):
        client, _, service = running
        opened = (await client.post("/api/director/org", json=org_body(repo))).json()
        await service.settled()
        top = opened["session"]["teams"][0]["id"]
        asked = ThreadMessage(
            at="2026-11-01T00:00:00+00:00", sender="ic", text="Split?"
        )
        told = ThreadMessage(
            at="2026-11-02T00:00:00+00:00", sender="manager", text="Status?"
        )
        service._keep_message(top, top, asked)
        service._keep_message(top, "below@other", told)
        service._keep_message("above@other", "below@other", told)

        response = await client.get(f"/api/director/org/nodes/{top}/messages")
        at_the_team = await client.get("/api/director/org/nodes/root/messages")
        missing = await client.get("/api/director/org/nodes/missing/messages")

        assert response.status_code == 200
        assert response.json() == [
            {
                "manager_node_id": top,
                "ic_node_id": top,
                "ic_to_ic": False,
                "messages": [asked.model_dump()],
            },
        ]
        # Nobody works at a team of the org itself, so nothing was said there.
        assert at_the_team.json() == []
        assert missing.status_code == 404


class TestTreeStream:
    """Drives the endpoint's own response body, without an HTTP client.

    The stream never ends, so each test reads the events it expects and then
    closes the body the way a disconnecting client does.
    """

    @pytest.fixture
    def service(self, tmp_path: Path) -> DirectorService:
        service = DirectorService(tmp_path)
        service.store.replace_all(small_tree())
        return service

    @staticmethod
    async def open_stream(
        service: DirectorService,
        node: str | None = None,
        expanded: list[str] | None = None,
        full: bool = False,
    ) -> AsyncIterator[str]:
        response = await director_router.stream_org_tree(service, node, expanded, full)
        assert response.media_type == "text/event-stream"
        body = response.body_iterator
        assert isinstance(body, AsyncIterator)

        async def frames() -> AsyncIterator[str]:
            try:
                async for frame in body:
                    assert isinstance(frame, str)
                    yield frame
            finally:
                await body.aclose()  # type: ignore[attr-defined]

        return frames()

    @staticmethod
    async def next_event(body: AsyncIterator[str]) -> tuple[str, dict]:
        frame = await asyncio.wait_for(anext(body), timeout=5)
        event_line, data_line = frame.strip().split("\n")
        return event_line.removeprefix("event: "), json.loads(
            data_line.removeprefix("data: ")
        )

    async def test_starts_with_a_snapshot_of_the_root_then_follows_changes(
        self, service: DirectorService
    ):
        body = await self.open_stream(service)

        name, snapshot = await self.next_event(body)
        await asyncio.to_thread(
            service.store.upsert, renamed(make_team("b", "root"), "Billing")
        )
        _, upsert = await self.next_event(body)
        await asyncio.to_thread(service.store.remove, "b")
        _, removed = await self.next_event(body)
        _, above = await self.next_event(body)

        assert name == "snapshot"
        assert snapshot["focus_id"] == "root"
        assert [node["id"] for node in snapshot["nodes"]] == ["root", "a", "b"]
        assert (upsert["type"], upsert["node"]["title"]) == ("node_upsert", "Billing")
        assert removed == {"type": "node_removed", "id": "b"}
        assert (above["node"]["id"], above["node"]["child_ids"]) == ("root", ["a"])

    async def test_follows_only_the_requested_node(self, service: DirectorService):
        body = await self.open_stream(service, "b")

        _, snapshot = await self.next_event(body)
        service.store.upsert(renamed(make_team("a1", "a"), "One"))
        service.store.upsert(renamed(make_team("b", "root"), "Billing"))
        _, first_change = await self.next_event(body)

        assert [node["id"] for node in snapshot["nodes"]] == ["root", "b"]
        # The change to a1 does not reach this viewer.
        assert first_change["node"]["id"] == "b"

    async def test_follows_the_reports_of_expanded_nodes(
        self, service: DirectorService
    ):
        body = await self.open_stream(service, expanded=["a", "missing"])

        _, snapshot = await self.next_event(body)
        service.store.upsert(renamed(make_team("a1", "a"), "One"))
        _, change = await self.next_event(body)

        assert [node["id"] for node in snapshot["nodes"]] == [
            "root",
            "a",
            "b",
            "a1",
            "a2",
        ]
        assert change["node"]["id"] == "a1"

    async def test_a_full_stream_follows_every_node_below_the_focus(
        self, service: DirectorService
    ):
        body = await self.open_stream(service, "a", full=True)

        _, snapshot = await self.next_event(body)
        service.store.upsert(make_team("deep", "a1"))
        _, change = await self.next_event(body)

        assert [node["id"] for node in snapshot["nodes"]] == ["root", "a", "a1", "a2"]
        assert change["node"]["id"] == "deep"

    async def test_sends_what_waits_on_the_human_and_each_change_to_it(
        self, service: DirectorService
    ):
        store = service.store
        top = open_session_in(store, "a", "s1")
        store.raise_escalation(top.id, "jwt", kind="design_decision", title="jwt")
        body = await self.open_stream(service)

        _, snapshot = await self.next_event(body)
        store.settle_escalation(top.id, "jwt")
        events = [await self.next_event(body) for _ in range(2)]

        [waiting] = snapshot["escalations"]
        assert (waiting["node_id"], waiting["session_id"]) == ("a", "s1")
        assert waiting["escalation"]["id"] == "jwt"
        assert [step["title"] for step in waiting["path"]] == ["a"]
        # Nothing of the session is a node of the tree.
        assert [node["id"] for node in snapshot["nodes"]] == ["root", "a", "b"]
        assert ("escalations", {"type": "escalations", "escalations": []}) in events

    async def test_says_when_the_org_is_paused_and_when_it_runs_again(
        self, service: DirectorService
    ):
        service._hold("human")
        body = await self.open_stream(service)

        _, snapshot = await self.next_event(body)
        await asyncio.to_thread(service._release)
        name, running = await self.next_event(body)
        await body.aclose()  # type: ignore[attr-defined]

        assert (snapshot["paused"], snapshot["merging"]) == ("human", False)
        assert name == "org_state"
        assert running == {"type": "org_state", "paused": None, "merging": False}
        assert service._state_listeners == []

    async def test_an_idle_stream_sends_heartbeats(
        self, service: DirectorService, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(director_router, "STREAM_HEARTBEAT_SECONDS", 0.01)
        body = await self.open_stream(service)
        await self.next_event(body)

        frame = await asyncio.wait_for(anext(body), timeout=5)

        assert frame == ": keep-alive\n\n"

    async def test_a_closed_stream_stops_listening(self, service: DirectorService):
        delivered: list[object] = []
        body = await self.open_stream(service)
        await self.next_event(body)
        other = await self.open_stream(service)
        await self.next_event(other)

        await body.aclose()  # type: ignore[attr-defined]
        service.store.subscribe("root", delivered.append)
        service.store.upsert(renamed(make_team("b", "root"), "Billing"))

        # The store has exactly the open stream and this test's listener left.
        assert len(service.store._listeners) == 2
        assert (await self.next_event(other))[0] == "node_upsert"
        assert len(delivered) == 1

    async def test_an_empty_tree_or_unknown_node_is_refused_before_streaming(
        self, tmp_path: Path, service: DirectorService
    ):
        with pytest.raises(NotFound):
            await self.open_stream(DirectorService(tmp_path))
        with pytest.raises(NotFound):
            await self.open_stream(service, "missing")

        assert service.store._listeners == {}


def org_body(workspace: Path) -> dict:
    return {
        "title": "Checkout",
        "task": "Build the checkout flow",
        "conversation": root_conversation(workspace).model_dump(mode="json"),
    }


@pytest.fixture
async def running(
    repo: Path,
) -> AsyncIterator[tuple[httpx.AsyncClient, FakeLauncher, DirectorService]]:
    """The app with its director running agents through a fake launcher."""
    app = create_app(server_config(repo))
    launcher = FakeLauncher()
    service: DirectorService = app.state.director_service
    await service.start(launcher)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, launcher, service
    await service.aclose()


def test_the_summary_says_there_is_no_org_yet_and_where_one_would_work(repo: Path):
    response = make_client(repo).get("/api/director/org")

    assert response.status_code == 200
    assert response.json() == {
        "root_id": None,
        "node_count": 0,
        "repo_path": str(repo),
        "directory": "",
        "is_git_repository": True,
        "boundary_enforcement": "checks",
        "paused": None,
        "merging": False,
    }


def test_the_summary_reports_an_existing_org(client: TestClient):
    body = client.get("/api/director/org").json()

    assert (body["root_id"], body["node_count"]) == ("root", 5)


def test_an_org_cannot_be_started_while_the_director_runs_no_agents(repo: Path):
    response = make_client(repo).post("/api/director/org", json=org_body(repo))

    assert response.status_code == 503


async def test_starting_an_org_returns_its_first_session_and_only_works_once(
    running, repo
):
    client, launcher, _ = running

    created = await client.post("/api/director/org", json=org_body(repo))
    again = await client.post("/api/director/org", json=org_body(repo))

    assert created.status_code == 201
    session = created.json()["session"]
    top = session["teams"][0]
    assert (top["id"], top["title"]) == (f"root@{session['id']}", "Checkout")
    assert [(agent["role"], agent["conversation_id"]) for agent in top["agents"]] == [
        ("ic", created.json()["conversation_id"])
    ]
    assert launcher.root is not None
    assert again.status_code == 409
    summary = (await client.get("/api/director/org")).json()
    assert summary["root_id"] == "root"


async def test_an_org_cannot_be_started_outside_a_git_repository(
    running, tmp_path: Path
):
    client, _, _ = running
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    response = await client.post("/api/director/org", json=org_body(elsewhere))

    assert response.status_code == 503
    assert (await client.get("/api/director/org")).json()["root_id"] is None


async def test_an_org_is_paused_asked_what_it_holds_and_resumed(running, repo):
    client, launcher, service = running
    no_org = await client.post("/api/director/org/pause")
    created = await client.post("/api/director/org", json=org_body(repo))
    await service.settled()
    root_ic = created.json()["conversation_id"]
    top = created.json()["session"]["teams"][0]
    delivered = len(launcher.messages_to(root_ic))

    paused = await client.post("/api/director/org/pause")
    asked = await client.post("/api/director/org/messages", json={"text": "Refunds"})
    service.tell_ic(top["id"], "Add a cart", human=True)
    await service.settled()
    summary = await client.get("/api/director/org")
    recovery = await client.get("/api/director/org/recovery")

    assert (no_org.status_code, asked.status_code) == (409, 409)
    assert paused.status_code == 200 and paused.json()["paused"] == "human"
    assert summary.json()["paused"] == "human"
    assert recovery.json() == {
        "paused": "human",
        "items": [
            {
                "node_id": top["id"],
                "title": top["title"],
                "kind": "message",
                "detail": "1 message is delivered.",
            }
        ],
        "messages_held": 1,
    }
    assert len(launcher.messages_to(root_ic)) == delivered

    resumed = await client.post("/api/director/org/resume")
    await service.settled()

    assert resumed.json() == {"paused": None, "items": [], "messages_held": 0}
    assert launcher.messages_to(root_ic)[-1] == "Add a cart"
    assert (await client.get("/api/director/org/recovery")).json()["items"] == []


async def test_an_org_is_pushed_pulled_and_its_merge_asked_for(running, repo, tmp_path):
    client, _, service = running
    nothing = await client.post("/api/director/org/push")
    created = await client.post("/api/director/org", json=org_body(repo))
    await service.settled()
    session = created.json()["session"]["id"]
    start_again = f"/api/director/org/sessions/{session}/teams/root/start-again"
    no_remote = await client.post("/api/director/org/pull", json={"remote": "nope"})
    bare = tmp_path / "remote.git"
    run_git(tmp_path, "init", "--quiet", "--bare", str(bare))
    run_git(repo, "remote", "add", "origin", str(bare))

    pushed = await client.post("/api/director/org/push")
    pulled = await client.post("/api/director/org/pull", json={"remote": "origin"})
    merge = await client.get("/api/director/org/merge")
    resolve = await client.post("/api/director/org/merge/resolve", json={})
    abort = await client.post("/api/director/org/merge/abort")
    again = await client.post(
        start_again, json={"conversation": org_body(repo)["conversation"]}
    )

    assert nothing.status_code == 409 and no_remote.status_code == 404
    assert pushed.status_code == 200 and pushed.json()["result"] == "pushed"
    assert pulled.json()["result"] == "up_to_date" and pulled.json()["merge"] is None
    assert merge.status_code == 200 and merge.json() is None
    assert resolve.status_code == 409 and abort.status_code == 409
    assert again.status_code == 409
    unasked = await client.post(start_again, json={})
    assert unasked.status_code == 409
    assert (await client.get("/api/director/org")).json()["merging"] is False


async def test_answering_an_escalation_settles_it_and_tells_the_team(running, repo):
    client, launcher, service = running
    opened = (await client.post("/api/director/org", json=org_body(repo))).json()
    session = opened["session"]
    service.store.raise_escalation(
        session["teams"][0]["id"],
        "jwt",
        kind="design_decision",
        title="JWT or opaque tokens?",
    )
    url = (
        f"/api/director/org/sessions/{session['id']}/teams/root/escalations/{{}}/answer"
    )

    blank = await client.post(url.format("jwt"), json={"answer": ""})
    unknown = await client.post(url.format("nothing"), json={"answer": "Opaque."})
    answered = await client.post(url.format("jwt"), json={"answer": "Opaque."})
    await service.settled()

    assert (blank.status_code, unknown.status_code) == (422, 404)
    assert answered.status_code == 200
    assert answered.json()["escalations"] == []
    assert "Opaque." in launcher.messages_to(opened["conversation_id"])[-1]


async def test_the_first_message_starts_an_org_and_later_ones_open_sessions_in_it(
    running, repo
):
    client, launcher, service = running
    conversation = root_conversation(repo).model_dump(mode="json")

    first = await client.post(
        "/api/director/org/messages",
        json={"text": "Build the checkout flow", "conversation": conversation},
    )
    await service.settled()

    assert first.status_code == 200
    [session] = service.sessions()
    top = session.teams[0]
    assert top.title == "Build the checkout flow"
    assert first.json() == {
        "node_id": top.id,
        "team": "Build the checkout flow",
        "conversation_id": agent_conversation(top, "ic"),
        "started_org": True,
        "chosen_by": "path",
        "reason": "",
    }

    later = await client.post(
        "/api/director/org/messages",
        json={"text": "Add refunds", "path": "services/api"},
    )
    await service.settled()

    _, opened = service.sessions()
    place = opened.teams[0]
    assert (opened.team_id, place.id) == ("root", f"root@{opened.id}")
    assert later.json() == {
        **first.json(),
        "node_id": place.id,
        "conversation_id": agent_conversation(place, "ic"),
        "started_org": False,
    }
    start = launcher.derived[-1]
    assert str(start.conversation_id) == later.json()["conversation_id"]
    assert "Add refunds\n\n(The human sent this from services/api.)" in start.task

    said = await client.post(
        "/api/director/org/messages",
        json={"text": "And partial ones.", "session_id": opened.id},
    )
    unknown = await client.post(
        "/api/director/org/messages", json={"text": "Hi", "session_id": "s000000"}
    )
    await service.settled()

    # What is said in a session goes to its IC and opens no other session.
    assert said.json() == {**later.json(), "chosen_by": "path", "reason": ""}
    assert unknown.status_code == 404
    assert len(service.sessions()) == 2
    assert launcher.messages_to(said.json()["conversation_id"])[-1] == (
        "And partial ones."
    )


async def test_a_message_that_cannot_be_routed_is_refused(running, repo):
    client, _, service = running

    no_org = await client.post("/api/director/org/messages", json={"text": "Hi"})
    assert no_org.status_code == 422

    await client.post("/api/director/org", json=org_body(repo))
    await service.settled()
    outside = await client.post(
        "/api/director/org/messages", json={"text": "Hi", "path": "/etc"}
    )
    assert outside.status_code == 422
    blank = await client.post("/api/director/org/messages", json={"text": ""})
    assert blank.status_code == 422


def test_the_history_of_the_org_is_one_list_newest_first(saved_tree: Path):
    memory = GitStore(saved_tree)

    def happened(above: str, at: str, kind: str, team: str) -> None:
        event = {"at": at, "kind": kind, "team_id": team, "team": team.upper()}
        take_in(memory, events={above: [TeamEvent.model_validate(event)]})

    happened("root", "2026-11-01T10:00:00+00:00", "set_up", "a")
    happened("a", "2026-11-01T11:00:00+00:00", "accepted", "a1")
    happened("root", "2026-11-01T12:00:00+00:00", "sent_back", "b")
    happened("root", "2026-11-01T12:00:00+00:00", "accepted", "b")
    # Counted for an IC, and no event of the org.
    happened("root", "2026-11-01T13:00:00+00:00", "needed", "a")
    client = make_client(saved_tree)

    latest = client.get("/api/director/org/history", params={"limit": 2}).json()
    earlier = client.get(
        "/api/director/org/history",
        params={"before": latest["events"][-1]["at"]},
    ).json()
    refused = client.get("/api/director/org/history", params={"limit": 0})

    assert [(e["kind"], e["team"], e["above_id"]) for e in latest["events"]] == [
        # Of two in one second, the one recorded later comes first.
        ("accepted", "B", "root"),
        ("sent_back", "B", "root"),
    ]
    assert latest["earlier"] is True
    assert [e["kind"] for e in earlier["events"]] == ["accepted", "set_up"]
    # A page does not end inside a second: reading on would miss the rest.
    one = client.get("/api/director/org/history", params={"limit": 1}).json()
    assert [e["kind"] for e in one["events"]] == ["accepted", "sent_back"]
    assert earlier["earlier"] is False
    assert refused.status_code == 422


def test_the_limits_on_new_teams_are_read_changed_and_kept(saved_tree: Path):
    with make_client(saved_tree, max_concurrent_runs=7) as client:
        before = client.get("/api/director/settings").json()
        changed = client.patch("/api/director/settings", json={"max_teams": 100})
        refused = client.patch("/api/director/settings", json={"max_depth": 0})
    with make_client(saved_tree) as client:
        after = client.get("/api/director/settings").json()

    assert before == {
        "max_teams": 32,
        "max_depth": 8,
        "allowed_profiles": [],
        "default_agent_type": "software-engineer",
        "allowed_agent_types": [],
        "resume_on_start": False,
        "report_compaction_chars": 24_000,
        "stall_check_minutes": 120,
        "manager_check_accepted_reviews": 5,
        "manager_check_ic_setbacks": 3,
        "route_requests": True,
        "session_cost_limit": 0.0,
        "max_open_sessions": 0,
        "teams": 5,
        "depth": 3,
        "max_concurrent_runs": 7,
    }
    assert changed.status_code == 200 and changed.json()["max_teams"] == 100
    assert refused.status_code == 422
    assert (after["max_teams"], after["max_depth"]) == (100, 8)


def test_teams_are_searched_by_title_and_directory(client: TestClient):
    found = client.get("/api/director/org/search", params={"q": "svc/a", "limit": 2})

    assert found.status_code == 200
    assert [team["id"] for team in found.json()] == ["a", "a1"]
    assert found.json()[0] == {"id": "a", "title": "a", "path": "svc/a"}
    assert client.get("/api/director/org/search").status_code == 422


async def test_a_teams_changes_are_listed_shown_and_edited_by_hand(running, repo):
    client, launcher, service = running
    opened = (await client.post("/api/director/org", json=org_body(repo))).json()
    await service.settled()
    ic = opened["conversation_id"]
    (repo / "README.md").write_text("hello\nfrom the agent\n")
    url = f"/api/director/org/nodes/{opened['session']['teams'][0]['id']}"

    listed = await client.get(f"{url}/changes")
    sides = await client.get(f"{url}/changes/file", params={"path": "README.md"})
    launcher.report(ic, ConversationExecutionStatus.RUNNING)
    busy = await client.put(
        f"{url}/files", params={"path": "README.md"}, json={"content": "mine\n"}
    )
    outside_while_busy = await client.put(
        f"{url}/files", params={"path": ".git/config"}, json={"content": "no\n"}
    )
    launcher.report(ic, ConversationExecutionStatus.FINISHED)
    saved = await client.put(
        f"{url}/files", params={"path": "README.md"}, json={"content": "mine\n"}
    )
    await service.settled()
    outside = await client.put(
        f"{url}/files", params={"path": "../x"}, json={"content": "no\n"}
    )
    unknown = await client.get("/api/director/org/nodes/nobody/changes")
    between = {"base": saved.json()["commit"] + "^", "head": saved.json()["commit"]}
    not_ids = await client.get("/api/director/org/accepted/changes", params=between)
    between["base"] = run_git(repo, "rev-parse", between["base"]).strip()
    taken = await client.get("/api/director/org/accepted/changes", params=between)
    taken_file = await client.get(
        "/api/director/org/accepted/changes/file",
        params={**between, "path": "README.md"},
    )

    assert [(f["path"], f["committed"]) for f in listed.json()["files"]] == [
        ("README.md", False)
    ]
    assert sides.json()["before"] == "hello\n"
    assert sides.json()["after"] == "hello\nfrom the agent\n"
    assert busy.status_code == 409 and "working" in busy.json()["detail"]
    assert saved.status_code == 200
    assert saved.json()["commit"] == run_git(repo, "rev-parse", "HEAD").strip()
    assert (repo / "README.md").read_text() == "mine\n"
    told = launcher.messages_to(ic)[-1]
    assert "README.md" in told and saved.json()["commit"][:12] in told
    assert (outside.status_code, unknown.status_code) == (400, 404)
    assert outside_while_busy.status_code == 403
    assert not_ids.status_code == 400
    assert [f["path"] for f in taken.json()["files"]] == ["README.md"]
    assert taken_file.json()["after"] == "mine\n"


async def test_an_escalation_at_a_division_lead_place_is_answered_there(running, repo):
    client, _, service = running
    opened = (await client.post("/api/director/org", json=org_body(repo))).json()
    session = opened["session"]
    # A session opened at the top team has two places there.
    [lead] = [
        place for place in service.store.session(session["id"]) if place.division_lead
    ]
    service.store.raise_escalation(
        lead.id, "agent-failed", kind="critical_issue", title="Its IC failed"
    )

    answered = await client.post(
        f"/api/director/org/sessions/{session['id']}/teams/root"
        "/escalations/agent-failed/answer",
        json={"answer": "Start it again."},
    )
    await service.settled()

    assert answered.status_code == 200
    now = service.store.place(lead.id)
    assert now is not None and now.escalations == []


async def test_the_conversation_of_an_agent_of_the_org_cannot_be_forked(running, repo):
    client, _, service = running
    opened = (await client.post("/api/director/org", json=org_body(repo))).json()

    forked = await client.post(
        f"/api/conversations/{opened['conversation_id']}/fork", json={}
    )

    assert forked.status_code == 409
    assert "agent of the org" in forked.json()["detail"]
    # Nor is it deleted from under its session.
    deleted = await client.delete(f"/api/conversations/{opened['conversation_id']}")
    assert deleted.status_code == 409
    # Nor does it start a conversation of its own, which no team would hold.
    child = await client.post(
        f"/api/conversations/{opened['conversation_id']}/children",
        json={"task": "Do it for me"},
    )
    assert child.status_code == 409
