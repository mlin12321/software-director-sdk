"""OrgStore, the fixture tree and persistence in a hidden git ref."""

from pathlib import Path

import pytest

from openhands.agent_server.director.fixture import build_fixture_nodes
from openhands.agent_server.director.models import HealthCounts
from openhands.agent_server.director.org_persistence import (
    ORG_REF,
    create_org_board,
    load_org,
    save_org,
)
from openhands.agent_server.director.org_store import (
    OrgChange,
    OrgStore,
    OrgStoreError,
)
from tests.agent_server.director.conftest import make_node, run_git, small_tree


@pytest.fixture
def store() -> OrgStore:
    store = OrgStore()
    for node in small_tree():
        store.upsert(node)
    return store


def record(store: OrgStore, focus_id: str) -> list[OrgChange]:
    events: list[OrgChange] = []
    store.subscribe(focus_id, events.append)
    return events


def describe(events: list[OrgChange]) -> list[tuple[str, str]]:
    return [
        (event.type, event.node.id if event.type == "node_upsert" else event.id)
        for event in events
    ]


def test_derives_children_ancestors_and_subtree_rollups(store: OrgStore):
    root = store.get("root")
    assert root is not None

    assert root.child_ids == ["a", "b"]
    assert [node.id for node in store.ancestors("a2")] == ["root", "a"]
    assert [node.id for node in store.all()] == ["root", "a", "b", "a1", "a2"]
    assert root.descendant_counts == HealthCounts(
        collision=1, done=2, needs_attention=1
    )
    assert store.children("a")[0].descendant_counts == HealthCounts()


def test_snapshot_holds_the_focus_its_ancestors_and_direct_reports(store: OrgStore):
    snapshot = store.snapshot("a")

    assert snapshot is not None
    assert snapshot.focus_id == "a"
    assert [node.id for node in snapshot.nodes] == ["root", "a", "a1", "a2"]
    assert store.snapshot("missing") is None


def test_a_health_change_announces_each_ancestor_whose_rollup_changed(
    store: OrgStore,
):
    events = record(store, "a2")

    store.upsert(make_node("a2", "a", health="done", path="svc/a/two"))

    assert describe(events) == [
        ("node_upsert", "a2"),
        ("node_upsert", "a"),
        ("node_upsert", "root"),
    ]
    root = store.get("root")
    assert root is not None
    assert root.descendant_counts == HealthCounts(collision=1, done=3)


def test_an_update_that_keeps_the_rollups_announces_only_the_node(store: OrgStore):
    events = record(store, "a2")
    unchanged_health = make_node("a2", "a", health="needs_attention", path="svc/a/two")

    store.upsert(unchanged_health.model_copy(update={"title": "renamed"}))

    assert describe(events) == [("node_upsert", "a2")]


def test_updating_a_node_keeps_its_children_and_rollup(store: OrgStore):
    updated = store.upsert(make_node("a", "root", health="done", path="svc/a"))

    assert updated.child_ids == ["a1", "a2"]
    assert updated.descendant_counts == HealthCounts(done=1, needs_attention=1)


def test_subscribers_only_hear_about_nodes_in_their_snapshot(store: OrgStore):
    at_b = record(store, "b")
    at_root = record(store, "root")

    # a1 is a grandchild of root and unrelated to b; its change alters a's
    # rollup (a child of root) and root's own.
    store.upsert(make_node("a1", "a", health="pruned", path="svc/a/one"))

    assert describe(at_root) == [("node_upsert", "a"), ("node_upsert", "root")]
    assert describe(at_b) == [("node_upsert", "root")]


def test_a_new_child_announces_itself_and_its_parents_child_list(store: OrgStore):
    events = record(store, "b")

    store.upsert(make_node("b1", "b", path="svc/b/one"))

    assert describe(events)[:2] == [("node_upsert", "b1"), ("node_upsert", "b")]
    parent = events[1]
    assert parent.type == "node_upsert"
    assert parent.node.child_ids == ["b1"]


def test_removing_a_subtree_announces_leaves_first_and_fixes_rollups(
    store: OrgStore,
):
    inside = record(store, "a1")
    above = record(store, "root")

    assert store.remove("a") is True

    # A viewer inside the removed subtree hears its focus and the subtree's
    # root go, and nothing after that.
    assert describe(inside) == [("node_removed", "a1"), ("node_removed", "a")]
    assert describe(above) == [("node_removed", "a"), ("node_upsert", "root")]
    root = store.get("root")
    assert root is not None
    assert root.child_ids == ["b"]
    assert root.descendant_counts == HealthCounts(done=1)
    assert store.get("a2") is None
    assert store.remove("a") is False


def test_the_root_can_be_removed_and_replaced(store: OrgStore):
    store.remove("root")

    assert len(store) == 0
    assert store.root_id is None
    assert store.upsert(make_node("new-root", None)).id == "new-root"


def test_unsubscribing_stops_delivery(store: OrgStore):
    events: list[OrgChange] = []
    snapshot, unsubscribe = store.subscribe("root", events.append)
    assert snapshot is not None

    unsubscribe()
    store.upsert(make_node("b", "root", health="collision", path="svc/b"))

    assert events == []


def test_subscribing_to_an_unknown_node_registers_nothing(store: OrgStore):
    events: list[OrgChange] = []

    snapshot, _ = store.subscribe("missing", events.append)
    store.upsert(make_node("missing", "root"))

    assert snapshot is None
    assert events == []


@pytest.mark.parametrize(
    "node",
    [
        make_node("second-root", None),
        make_node("orphan", "nowhere"),
        make_node("a1", "b"),
    ],
    ids=["second root", "missing parent", "moved to another parent"],
)
def test_rejects_nodes_that_do_not_fit_the_tree(store: OrgStore, node):
    before = store.all()

    with pytest.raises(OrgStoreError):
        store.upsert(node)

    assert store.all() == before


class TestReplaceAll:
    def test_loads_nodes_in_any_order_and_computes_rollups(self):
        store = OrgStore()

        store.replace_all(reversed(small_tree()))

        root = store.get("root")
        assert root is not None
        assert root.child_ids == ["b", "a"]
        assert root.descendant_counts == HealthCounts(
            collision=1, done=2, needs_attention=1
        )

    def test_accepts_an_empty_tree(self, store: OrgStore):
        store.replace_all([])

        assert len(store) == 0
        assert store.root_id is None

    @pytest.mark.parametrize(
        "nodes",
        [
            [make_node("root", None), make_node("root", None)],
            [make_node("root", None), make_node("x", "nowhere")],
            [make_node("r1", None), make_node("r2", None)],
            [make_node("x", "y"), make_node("y", "x")],
            [make_node("root", None), make_node("x", "y"), make_node("y", "x")],
        ],
        ids=["duplicate id", "orphan", "two roots", "no root", "detached cycle"],
    )
    def test_a_malformed_tree_leaves_the_store_untouched(self, store: OrgStore, nodes):
        before = store.all()

        with pytest.raises(OrgStoreError):
            store.replace_all(nodes)

        assert store.all() == before


def test_the_fixture_is_one_consistent_deterministic_tree():
    store = OrgStore()

    store.replace_all(build_fixture_nodes())

    root = store.get("root")
    assert root is not None
    assert len(store) == 487
    assert sum(root.descendant_counts.model_dump().values()) == 486
    assert len(root.child_ids) == 6
    assert build_fixture_nodes() == build_fixture_nodes()


class TestPersistence:
    def test_the_tree_round_trips_through_a_hidden_ref(
        self, repo: Path, store: OrgStore
    ):
        board = create_org_board(repo)

        save_org(board, store)
        head = board.head()
        save_org(board, store)
        loaded = OrgStore()

        assert load_org(board, loaded) is True
        assert loaded.all() == store.all()
        # Saving an unchanged tree adds no commit, and nothing leaks into the
        # working tree.
        assert board.head() == head
        assert run_git(repo, "rev-parse", ORG_REF).strip() == head
        assert run_git(repo, "status", "--porcelain") == ""

    def test_loading_finds_nothing_before_the_first_save(self, repo: Path):
        store = OrgStore()

        assert load_org(create_org_board(repo), store) is False
        assert len(store) == 0

    def test_a_tree_saved_in_an_unknown_version_is_refused(self, repo: Path):
        board = create_org_board(repo)
        board.write("", "org.json", '{"version": 99, "nodes": []}')

        with pytest.raises(ValueError, match="99"):
            load_org(board, OrgStore())
