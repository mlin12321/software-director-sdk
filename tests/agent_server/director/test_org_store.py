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
from openhands.sdk.errors import NotFound
from tests.agent_server.director.conftest import (
    make_escalation,
    make_node,
    run_git,
    small_tree,
)


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
    described: list[tuple[str, str]] = []
    for event in events:
        if event.type == "node_upsert":
            described.append((event.type, event.node.id))
        elif event.type == "node_removed":
            described.append((event.type, event.id))
        else:
            ids = ",".join(item.escalation.id for item in event.escalations)
            described.append((event.type, ids))
    return described


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


def test_snapshot_adds_the_reports_of_expanded_nodes(store: OrgStore):
    snapshot = store.snapshot("root", ["a", "b", "missing"])

    assert snapshot is not None
    assert [node.id for node in snapshot.nodes] == ["root", "a", "b", "a1", "a2"]


def test_subscribers_hear_about_the_reports_of_nodes_they_expanded(store: OrgStore):
    collapsed: list[OrgChange] = []
    expanded: list[OrgChange] = []
    store.subscribe("root", collapsed.append)
    store.subscribe("root", expanded.append, ["a"])

    store.upsert(make_node("a1", "a", health="done", path="renamed"))
    store.remove("a2")

    assert describe(collapsed) == [("node_upsert", "a"), ("node_upsert", "root")]
    assert describe(expanded) == [
        ("node_upsert", "a1"),
        ("node_removed", "a2"),
        ("node_upsert", "a"),
        ("node_upsert", "root"),
    ]


class TestEscalations:
    @pytest.fixture
    def store(self, store: OrgStore) -> OrgStore:
        store.upsert(
            make_node(
                "a2",
                "a",
                escalations=[
                    make_escalation(
                        "late",
                        raised_at="2026-11-12T00:00:00Z",
                        passed_up_by=("a", "root"),
                    )
                ],
            )
        )
        store.upsert(
            make_node(
                "b",
                "root",
                escalations=[
                    make_escalation("early", passed_up_by=("root",)),
                    make_escalation(
                        "broken",
                        kind="critical_issue",
                        raised_at="2026-11-13T00:00:00Z",
                        passed_up_by=("root",),
                    ),
                ],
            )
        )
        return store

    def test_every_node_counts_the_escalations_at_or_below_it(self, store: OrgStore):
        counts = {node.id: node.escalation_count for node in store.all()}

        assert counts == {"root": 3, "a": 1, "b": 2, "a1": 0, "a2": 1}

    def test_lists_critical_issues_first_then_the_longest_waiting(
        self, store: OrgStore
    ):
        listed = store.escalations("root")

        assert [item.escalation.id for item in listed] == ["broken", "early", "late"]
        assert [step.id for step in listed[-1].path] == ["root", "a", "a2"]
        assert listed[-1].node_id == "a2"

    def test_lists_only_what_is_at_or_below_the_focused_node(self, store: OrgStore):
        assert [item.escalation.id for item in store.escalations("a")] == ["late"]
        assert store.escalations("a1") == []
        assert store.escalations("missing") == []

    def test_a_snapshot_carries_what_waits_on_the_human_anywhere_in_the_tree(
        self, store: OrgStore
    ):
        snapshot = store.snapshot("a")

        assert snapshot is not None
        assert [item.escalation.id for item in snapshot.escalations] == [
            "broken",
            "early",
            "late",
        ]

    def test_raising_and_settling_announce_the_new_list(self, store: OrgStore):
        events = record(store, "a")

        store.upsert(
            make_node(
                "a1",
                "a",
                escalations=[
                    make_escalation(
                        "new", kind="critical_issue", passed_up_by=("a", "root")
                    )
                ],
            )
        )
        store.upsert(make_node("a2", "a"))

        assert describe(events) == [
            ("node_upsert", "a1"),
            ("node_upsert", "a"),
            ("node_upsert", "root"),
            ("escalations", "new,broken,early,late"),
            ("node_upsert", "a2"),
            ("node_upsert", "a"),
            ("node_upsert", "root"),
            ("escalations", "new,broken,early"),
        ]

    def test_changes_that_leave_the_list_alone_announce_nothing(self, store: OrgStore):
        events = record(store, "a")

        store.upsert(make_node("a1", "a", health="pruned"))
        # Held by the node above: not the human's business yet.
        store.upsert(
            make_node("a1", "a", escalations=[make_escalation("held", held_by="a")])
        )

        assert "escalations" not in [event.type for event in events]

    def test_removing_a_subtree_drops_its_escalations(self, store: OrgStore):
        events = record(store, "root")

        store.remove("a")

        root = store.get("root")
        assert root is not None
        assert root.escalation_count == 2
        assert describe(events)[-1] == ("escalations", "broken,early")

    def test_renaming_a_node_on_the_path_announces_the_new_path(self, store: OrgStore):
        events = record(store, "root")

        store.upsert(
            make_node("a", "root", health="collision").model_copy(
                update={"title": "renamed"}
            )
        )

        last = events[-1]
        assert last.type == "escalations"
        assert [step.title for step in last.escalations[-1].path] == [
            "root",
            "renamed",
            "a2",
        ]

    def test_only_what_waits_on_the_human_is_listed_and_counted(self, store: OrgStore):
        store.upsert(
            make_node("a1", "a", escalations=[make_escalation("held", held_by="a")])
        )

        root = store.get("root")
        assert root is not None
        assert root.escalation_count == 3
        assert "held" not in [e.escalation.id for e in store.escalations("root")]

    def test_escalations_survive_a_save_and_load(self, store: OrgStore, repo: Path):
        board = create_org_board(repo)
        save_org(board, store)
        loaded = OrgStore()

        load_org(board, loaded)

        assert loaded.all() == store.all()
        assert loaded.escalations("root") == store.escalations("root")


class TestEscalationChain:
    """An escalation climbs one node at a time and reaches the human last."""

    def test_a_raised_escalation_goes_to_the_node_above_not_to_the_human(
        self, store: OrgStore
    ):
        events = record(store, "root")

        raised = store.raise_escalation(
            "a2", "jwt", kind="design_decision", title="JWT or opaque?"
        )

        root = store.get("root")
        assert root is not None
        assert (raised.held_by, raised.handoffs) == ("a", [])
        assert root.escalation_count == 0
        assert store.escalations("root") == []
        assert "escalations" not in [event.type for event in events]

    def test_reaches_the_human_only_after_every_node_above_passed_it_up(
        self, store: OrgStore
    ):
        store.raise_escalation("a2", "jwt", kind="design_decision", title="t")
        events = record(store, "root")

        first = store.pass_up_escalation("a2", "jwt", reason="  needs b too ")
        waiting_after_first = store.escalations("root")
        second = store.pass_up_escalation("a2", "jwt", reason="needs a human")

        assert first.held_by == "root"
        assert waiting_after_first == []
        assert second.held_by is None
        assert [(h.node_id, h.reason) for h in second.handoffs] == [
            ("a", "needs b too"),
            ("root", "needs a human"),
        ]
        assert [e.escalation.id for e in store.escalations("root")] == ["jwt"]
        assert {node.id: node.escalation_count for node in store.all()} == {
            "root": 1,
            "a": 1,
            "b": 0,
            "a1": 0,
            "a2": 1,
        }
        assert describe(events)[-1] == ("escalations", "jwt")

    def test_an_escalation_raised_by_the_root_has_only_the_human_above_it(
        self, store: OrgStore
    ):
        raised = store.raise_escalation(
            "root", "budget", kind="critical_issue", title="t"
        )

        assert raised.held_by is None
        assert [e.escalation.id for e in store.escalations("root")] == ["budget"]

    def test_settling_closes_it_wherever_it_is_held(self, store: OrgStore):
        store.raise_escalation("a2", "jwt", kind="design_decision", title="t")
        store.raise_escalation("b", "leak", kind="critical_issue", title="t")
        store.pass_up_escalation("b", "leak", reason="cannot")

        store.settle_escalation("a2", "jwt")
        store.settle_escalation("b", "leak")

        assert [node.escalations for node in store.all()] == [[]] * 5
        assert store.escalations("root") == []

    def test_cannot_be_passed_up_past_the_human_or_without_a_reason(
        self, store: OrgStore
    ):
        store.raise_escalation("b", "leak", kind="critical_issue", title="t")

        with pytest.raises(OrgStoreError, match="needs a reason"):
            store.pass_up_escalation("b", "leak", reason="   ")
        store.pass_up_escalation("b", "leak", reason="cannot")
        with pytest.raises(OrgStoreError, match="already waits on the human"):
            store.pass_up_escalation("b", "leak", reason="again")

    def test_refuses_a_duplicate_id_and_unknown_nodes_or_escalations(
        self, store: OrgStore
    ):
        store.raise_escalation("b", "leak", kind="critical_issue", title="t")

        with pytest.raises(OrgStoreError, match="already has an escalation"):
            store.raise_escalation("b", "leak", kind="critical_issue", title="t")
        with pytest.raises(NotFound):
            store.raise_escalation("nope", "x", kind="critical_issue", title="t")
        with pytest.raises(NotFound):
            store.pass_up_escalation("b", "nope", reason="r")
        with pytest.raises(NotFound):
            store.settle_escalation("b", "nope")

    @pytest.mark.parametrize(
        "escalation",
        [
            # Straight to the human.
            make_escalation("x"),
            # Skipped its parent.
            make_escalation("x", passed_up_by=("root",)),
            # Passed up by its parent but still held by it.
            make_escalation("x", passed_up_by=("a",), held_by="a"),
            # Held by a node that is not above it.
            make_escalation("x", held_by="b"),
        ],
        ids=["no-handoffs", "skipped-parent", "wrong-holder", "holder-not-above"],
    )
    def test_refuses_an_escalation_that_skipped_a_node_on_its_way_up(
        self, store: OrgStore, escalation
    ):
        before = store.all()

        with pytest.raises(OrgStoreError, match="must be passed up through"):
            store.upsert(make_node("a2", "a", escalations=[escalation]))
        with pytest.raises(OrgStoreError, match="must be passed up through"):
            OrgStore().replace_all(
                [
                    *small_tree()[:4],
                    make_node("a2", "a", escalations=[escalation]),
                ]
            )

        assert store.all() == before


class TestFullSubscription:
    def test_a_full_snapshot_holds_every_node_below_the_focus(self, store: OrgStore):
        whole = store.snapshot("root", full=True)
        below_a = store.snapshot("a", full=True)

        assert whole is not None and below_a is not None
        assert [node.id for node in whole.nodes] == ["root", "a", "b", "a1", "a2"]
        assert [node.id for node in below_a.nodes] == ["root", "a", "a1", "a2"]

    def test_a_full_subscriber_hears_about_every_node_below_its_focus(
        self, store: OrgStore
    ):
        events: list[OrgChange] = []
        store.subscribe("a", events.append, full=True)

        store.upsert(make_node("deep", "a1"))
        store.upsert(make_node("deeper", "deep"))
        store.upsert(make_node("elsewhere", "b"))
        store.remove("deep")

        assert describe(events) == [
            ("node_upsert", "deep"),
            ("node_upsert", "a1"),
            ("node_upsert", "a"),
            ("node_upsert", "root"),
            ("node_upsert", "deeper"),
            ("node_upsert", "deep"),
            ("node_upsert", "a1"),
            ("node_upsert", "a"),
            ("node_upsert", "root"),
            ("node_upsert", "root"),
            ("node_removed", "deeper"),
            ("node_removed", "deep"),
            ("node_upsert", "a1"),
            ("node_upsert", "a"),
            ("node_upsert", "root"),
        ]


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
    # Four are open; three are still held by a node above the one that
    # raised them.
    assert sum(len(node.escalations) for node in store.all()) == 4
    assert root.escalation_count == 1
    assert [item.node_id for item in store.escalations("root")] == [
        "data.worker.module-03"
    ]
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
