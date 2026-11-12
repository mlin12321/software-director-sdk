"""The director HTTP API, served by the real app."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openhands.agent_server import director_router
from openhands.agent_server.api import create_app
from openhands.agent_server.config import Config
from openhands.agent_server.director.git_blackboard import GitBlackboard
from openhands.agent_server.director.models import Autopsy
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    save_org,
)
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.service import DirectorService
from openhands.sdk.errors import NotFound
from tests.agent_server.director.conftest import (
    make_escalation,
    make_node,
    small_tree,
)


def make_client(repo_dir: Path, **config) -> TestClient:
    app = create_app(Config(session_api_keys=[], director_repo_path=repo_dir, **config))
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


def test_returns_a_node_with_its_ancestors_and_children(client: TestClient):
    response = client.get("/api/director/org/nodes/a")

    assert response.status_code == 200
    body = response.json()
    assert body["node"]["id"] == "a"
    assert body["node"]["descendant_counts"]["idle"] == 1
    assert [node["id"] for node in body["ancestors"]] == ["root"]
    assert [node["id"] for node in body["children"]] == ["a1", "a2"]


def test_an_unknown_node_is_not_found_on_every_node_route(client: TestClient):
    responses = [
        client.get("/api/director/org/nodes/missing"),
        client.get("/api/director/org/nodes/missing/blackboard"),
        client.post("/api/director/org/nodes/missing/teach", json={"rule": "x"}),
        client.get("/api/director/org/tree-stream", params={"node": "missing"}),
    ]

    assert [response.status_code for response in responses] == [404] * 4


def test_routes_require_the_session_key_when_one_is_configured(saved_tree: Path):
    app = create_app(Config(session_api_keys=["secret"], director_repo_path=saved_tree))
    client = TestClient(app, raise_server_exceptions=False)

    denied = client.get("/api/director/org/nodes/root")
    allowed = client.get(
        "/api/director/org/nodes/root", headers={"X-Session-API-Key": "secret"}
    )

    assert (denied.status_code, allowed.status_code) == (401, 200)


def test_the_fixture_tree_is_served_when_enabled(tmp_path: Path):
    client = make_client(tmp_path, director_fixture=True)

    response = client.get("/api/director/org/nodes/payments.api")

    assert response.status_code == 200
    assert len(response.json()["children"]) == 15
    assert [node["id"] for node in response.json()["ancestors"]] == [
        "root",
        "payments",
    ]


def test_the_tree_is_empty_without_a_saved_tree_or_a_repository(
    tmp_path: Path, repo: Path
):
    for directory in (tmp_path / "missing", tmp_path, repo):
        client = make_client(directory)

        assert client.get("/api/director/org/nodes/root").status_code == 404
        assert client.get("/api/director/org/tree-stream").status_code == 404


class TestBlackboard:
    def test_returns_what_applies_to_the_nodes_boundary(
        self, client: TestClient, saved_tree: Path
    ):
        board = GitBlackboard(saved_tree)
        board.write("", "spec.md", "root spec")
        board.write("svc/a", "spec.md", "a spec")
        board.write("svc/b", "spec.md", "b spec")
        board.teach("svc/a", "Rule for a.")
        autopsy = Autopsy(
            node_id="a", at="2026-11-01T00:00:00Z", approach="x", why_failed="y"
        )
        board.append_autopsy("svc/a", autopsy)

        response = client.get("/api/director/org/nodes/a/blackboard")

        assert response.status_code == 200
        assert response.json() == {
            "specs": [
                {"path": "", "spec": "root spec", "readonly": True},
                {"path": "svc/a", "spec": "a spec", "readonly": False},
            ],
            "memory": {"rules": ["Rule for a."]},
            "autopsies": [autopsy.model_dump()],
        }

    def test_teaching_stores_the_rule_in_the_nodes_own_boundary(
        self, client: TestClient, saved_tree: Path
    ):
        first = client.post(
            "/api/director/org/nodes/a/teach", json={"rule": "  Be kind.  "}
        )
        again = client.post(
            "/api/director/org/nodes/a/teach", json={"rule": "Be kind."}
        )

        assert first.status_code == 200
        assert first.json() == {"rules": ["Be kind."]}
        assert again.json() == {"rules": ["Be kind."]}
        board = GitBlackboard(saved_tree)
        assert json.loads(board.read("svc/a/memory.json") or "") == {
            "rules": ["Be kind."]
        }
        assert board.memory("svc/b").rules == []

    @pytest.mark.parametrize(
        ("body", "status"),
        [({"rule": "   "}, 400), ({"rule": ""}, 422), ({}, 422)],
        ids=["blank", "empty", "missing"],
    )
    def test_a_rule_without_text_is_rejected(
        self, client: TestClient, saved_tree: Path, body: dict, status: int
    ):
        response = client.post("/api/director/org/nodes/a/teach", json=body)

        assert response.status_code == status
        assert GitBlackboard(saved_tree).head() is None

    def test_is_unavailable_when_the_directory_is_not_a_repository(
        self, tmp_path: Path
    ):
        client = make_client(tmp_path, director_fixture=True)

        read = client.get("/api/director/org/nodes/root/blackboard")
        write = client.post("/api/director/org/nodes/root/teach", json={"rule": "x"})

        assert (read.status_code, write.status_code) == (503, 503)


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
            service.store.upsert, make_node("b", "root", status="blocked")
        )
        _, upsert = await self.next_event(body)
        _, rollup = await self.next_event(body)
        await asyncio.to_thread(service.store.remove, "b")
        _, removed = await self.next_event(body)

        assert name == "snapshot"
        assert snapshot["focus_id"] == "root"
        assert [node["id"] for node in snapshot["nodes"]] == ["root", "a", "b"]
        assert (upsert["type"], upsert["node"]["status"]) == (
            "node_upsert",
            "blocked",
        )
        assert rollup["node"]["id"] == "root"
        assert removed == {"type": "node_removed", "id": "b"}

    async def test_follows_only_the_requested_node(self, service: DirectorService):
        body = await self.open_stream(service, "b")

        _, snapshot = await self.next_event(body)
        service.store.upsert(make_node("a1", "a", status="working"))
        service.store.upsert(make_node("b", "root", status="working"))
        _, first_change = await self.next_event(body)

        assert [node["id"] for node in snapshot["nodes"]] == ["root", "b"]
        # The change to a1 only reaches this viewer as root's new rollup.
        assert first_change["node"]["id"] == "root"

    async def test_follows_the_reports_of_expanded_nodes(
        self, service: DirectorService
    ):
        body = await self.open_stream(service, expanded=["a", "missing"])

        _, snapshot = await self.next_event(body)
        service.store.upsert(make_node("a1", "a", status="working"))
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
        service.store.upsert(make_node("deep", "a1"))
        _, change = await self.next_event(body)

        assert [node["id"] for node in snapshot["nodes"]] == ["root", "a", "a1", "a2"]
        assert change["node"]["id"] == "deep"

    async def test_sends_what_waits_on_the_human_and_each_change_to_it(
        self, service: DirectorService
    ):
        service.store.upsert(
            make_node(
                "a1",
                "a",
                escalations=[make_escalation("jwt", passed_up_by=("a", "root"))],
            )
        )
        body = await self.open_stream(service)

        _, snapshot = await self.next_event(body)
        # a1 is outside this viewer's nodes; only the escalation reaches it.
        service.store.upsert(make_node("a1", "a"))
        events = [await self.next_event(body) for _ in range(3)]

        assert snapshot["escalations"] == [
            {
                "node_id": "a1",
                "escalation": make_escalation(
                    "jwt", passed_up_by=("a", "root")
                ).model_dump(),
                "path": [
                    {"id": "root", "title": "root"},
                    {"id": "a", "title": "a"},
                    {"id": "a1", "title": "a1"},
                ],
            }
        ]
        assert [node["escalation_count"] for node in snapshot["nodes"]] == [1, 1, 0]
        # a's and root's rollups, then the now-empty list.
        assert [name for name, _ in events] == [
            "node_upsert",
            "node_upsert",
            "escalations",
        ]
        assert events[-1][1] == {"type": "escalations", "escalations": []}

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
        service.store.upsert(make_node("b", "root", status="blocked"))

        # The store has exactly the open stream and this test's listener left.
        assert len(service.store._listeners) == 2
        assert (await self.next_event(other))[0] == "node_upsert"
        # b itself, then root's changed rollup.
        assert len(delivered) == 2

    async def test_an_empty_tree_or_unknown_node_is_refused_before_streaming(
        self, tmp_path: Path, service: DirectorService
    ):
        with pytest.raises(NotFound):
            await self.open_stream(DirectorService(tmp_path))
        with pytest.raises(NotFound):
            await self.open_stream(service, "missing")

        assert service.store._listeners == {}
