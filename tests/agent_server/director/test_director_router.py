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
from openhands.agent_server.director.git import git_version
from openhands.agent_server.director.git_blackboard import GitBlackboard
from openhands.agent_server.director.models import Autopsy
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    save_org,
)
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.service import (
    DirectorService,
    agent_conversation,
)
from openhands.sdk.errors import NotFound
from tests.agent_server.director.conftest import (
    FakeLauncher,
    make_escalation,
    make_node,
    root_conversation,
    run_git,
    small_tree,
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
        client.get("/api/director/org/nodes/missing/ownership"),
        client.post("/api/director/org/nodes/missing/teach", json={"rule": "x"}),
        client.get("/api/director/org/tree-stream", params={"node": "missing"}),
    ]

    assert [response.status_code for response in responses] == [404] * 5


def test_routes_require_the_session_key_when_one_is_configured(saved_tree: Path):
    app = create_app(server_config(saved_tree, session_api_keys=["secret"]))
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


def test_ownership_lists_a_nodes_files_by_the_report_that_owns_them(repo: Path):
    (repo / "svc" / "a").mkdir(parents=True)
    (repo / "svc" / "a" / "app.py").write_text("a\n")
    store = OrgStore()
    store.replace_all(
        [
            node.model_copy(update={"worktree_path": str(repo)})
            for node in (
                make_node("root", None, path=""),
                make_node("a", "root", path="svc/a"),
            )
        ]
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
                "branch": "main",
            }
        ],
        "files": [
            {"path": "README.md", "owner": None},
            {"path": "svc/a/app.py", "owner": "a"},
        ],
        "truncated": False,
        "collisions": [],
        "collisions_checked": True,
    }


class TestBlackboard:
    def test_returns_what_applies_to_the_nodes_boundary(
        self, client: TestClient, saved_tree: Path
    ):
        board = GitBlackboard(saved_tree)
        board.write("", "spec.md", "root spec")
        board.write("svc/a", "spec.md", "a spec")
        board.write("svc/b", "spec.md", "b spec")
        board.write("", "contracts.md", "a serves b")
        board.teach("svc/a", "Rule for a.")
        autopsy = Autopsy(
            node_id="a", at="2026-11-01T00:00:00Z", approach="x", why_failed="y"
        )
        board.append_autopsy("svc/a", autopsy)

        response = client.get("/api/director/org/nodes/a/blackboard")

        assert response.status_code == 200
        assert response.json() == {
            "specs": [{"path": "svc/a", "spec": "a spec", "readonly": False}],
            "contracts": [{"path": "", "contracts": "a serves b", "own": False}],
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
        "fixture": False,
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


async def test_starting_an_org_returns_its_root_and_only_works_once(running, repo):
    client, launcher, _ = running

    created = await client.post("/api/director/org", json=org_body(repo))
    again = await client.post("/api/director/org", json=org_body(repo))

    assert created.status_code == 201
    root = created.json()
    assert (root["id"], root["title"]) == ("root", "Checkout")
    assert [agent["role"] for agent in root["agents"]] == ["ic", "manager"]
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
    root_ic = created.json()["agents"][0]["conversation_id"]
    delivered = len(launcher.messages_to(root_ic))

    paused = await client.post("/api/director/org/pause")
    await client.post("/api/director/org/messages", json={"text": "Add a cart"})
    await service.settled()
    summary = await client.get("/api/director/org")
    recovery = await client.get("/api/director/org/recovery")

    assert no_org.status_code == 409
    assert paused.status_code == 200 and paused.json()["paused"] == "human"
    assert summary.json()["paused"] == "human"
    assert recovery.json() == {
        "paused": "human",
        "items": [
            {
                "node_id": "root",
                "title": created.json()["title"],
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
    await client.post("/api/director/org", json=org_body(repo))
    await service.settled()
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
        "/api/director/org/nodes/root/start_again",
        json={"conversation": org_body(repo)["conversation"]},
    )

    assert nothing.status_code == 409 and no_remote.status_code == 404
    assert pushed.status_code == 200 and pushed.json()["result"] == "pushed"
    assert pulled.json()["result"] == "up_to_date" and pulled.json()["merge"] is None
    assert merge.status_code == 200 and merge.json() is None
    assert resolve.status_code == 409 and abort.status_code == 409
    assert again.status_code == 409
    assert (await client.get("/api/director/org")).json()["merging"] is False


async def test_answering_an_escalation_settles_it_and_tells_the_team(running, repo):
    client, launcher, service = running
    root = (await client.post("/api/director/org", json=org_body(repo))).json()
    service.store.raise_escalation(
        "root", "jwt", kind="design_decision", title="JWT or opaque tokens?"
    )
    url = "/api/director/org/nodes/root/escalations/{}/answer"

    blank = await client.post(url.format("jwt"), json={"answer": ""})
    unknown = await client.post(url.format("nothing"), json={"answer": "Opaque."})
    answered = await client.post(url.format("jwt"), json={"answer": "Opaque."})
    await service.settled()

    assert (blank.status_code, unknown.status_code) == (422, 404)
    assert answered.status_code == 200
    assert answered.json()["escalations"] == []
    assert "Opaque." in launcher.messages_to(root["agents"][0]["conversation_id"])[-1]


async def test_the_first_message_starts_an_org_and_later_ones_find_its_teams(
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
    root = service.store.get("root")
    assert root is not None and root.title == "Build the checkout flow"
    assert first.json() == {
        "node_id": "root",
        "conversation_id": agent_conversation(root, "ic"),
        "started_org": True,
    }

    later = await client.post(
        "/api/director/org/messages",
        json={"text": "Add refunds", "path": "services/api"},
    )
    await service.settled()

    assert later.json() == {**first.json(), "started_org": False}
    assert launcher.messages_to(later.json()["conversation_id"])[-1] == (
        "Add refunds\n\n(The human sent this from services/api.)"
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


def test_the_blackboard_is_pushed_to_and_pulled_from_a_remote(
    saved_tree: Path, tmp_path: Path
):
    if git_version(saved_tree) < (2, 38):
        pytest.skip("needs git 2.38")
    bare, clone = tmp_path / "remote.git", tmp_path / "clone"
    run_git(tmp_path, "clone", "--quiet", "--bare", str(saved_tree), str(bare))
    run_git(saved_tree, "remote", "add", "origin", str(bare))
    run_git(tmp_path, "clone", "--quiet", str(bare), str(clone))
    GitBlackboard(saved_tree).teach("svc/a", "Be kind.")
    client = make_client(saved_tree)

    pushed = client.post("/api/director/org/blackboard/push")
    pulled = make_client(clone).post(
        "/api/director/org/blackboard/pull", json={"remote": "origin"}
    )

    assert pushed.status_code == 200 and pushed.json()["pushed"] is True
    assert pulled.status_code == 200
    assert pulled.json() == {
        "remote": "origin",
        "head": GitBlackboard(saved_tree).head(),
        "changed": True,
        "kept_local": [],
        "pushed": False,
    }
    assert GitBlackboard(clone).memory("svc/a").rules == ["Be kind."]


def test_sharing_the_blackboard_reports_what_is_missing(saved_tree: Path):
    client = make_client(saved_tree)

    no_remote = client.post("/api/director/org/blackboard/pull")
    run_git(saved_tree, "remote", "add", "origin", str(saved_tree / "nowhere.git"))
    nothing_to_push = client.post("/api/director/org/blackboard/push")
    GitBlackboard(saved_tree).teach("", "Rule.")
    unreachable = client.post("/api/director/org/blackboard/push")

    assert [
        response.status_code for response in (no_remote, nothing_to_push, unreachable)
    ] == [404, 409, 502]


def test_the_limits_on_new_teams_are_read_changed_and_kept(saved_tree: Path):
    with make_client(saved_tree, max_concurrent_runs=7) as client:
        before = client.get("/api/director/settings").json()
        changed = client.patch("/api/director/settings", json={"max_agents": 200})
        refused = client.patch("/api/director/settings", json={"max_depth": 0})
    with make_client(saved_tree) as client:
        after = client.get("/api/director/settings").json()

    assert before == {
        "max_agents": 64,
        "max_depth": 8,
        "allowed_profiles": [],
        "resume_on_start": False,
        "agents": 10,
        "depth": 3,
        "max_concurrent_runs": 7,
    }
    assert changed.status_code == 200 and changed.json()["max_agents"] == 200
    assert refused.status_code == 422
    assert (after["max_agents"], after["max_depth"]) == (200, 8)


def test_teams_are_searched_by_title_and_directory(client: TestClient):
    found = client.get("/api/director/org/search", params={"q": "svc/a", "limit": 2})

    assert found.status_code == 200
    assert [team["id"] for team in found.json()] == ["a", "a1"]
    assert found.json()[0] == {"id": "a", "title": "a", "path": "svc/a"}
    assert client.get("/api/director/org/search").status_code == 422
