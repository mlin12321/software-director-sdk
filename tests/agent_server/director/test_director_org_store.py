"""OrgStore and persistence in a hidden git ref."""

from pathlib import Path

import pytest

from openhands.agent_server.director.models import (
    AgentRef,
    AgentRole,
    Escalation,
    Review,
    ReviewState,
    SnapshotEvent,
)
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
    division_lead_id,
    session_team_id,
    status_of,
)
from openhands.sdk.errors import NotFound
from tests.agent_server.director.conftest import (
    make_escalation,
    make_team,
    open_session_in,
    run_git,
    session_places,
    small_tree,
)


AT = "2026-01-01T00:00:00+00:00"
SESSION = "s"


def at(team_id: str) -> str:
    """The place the session has at a team."""
    return session_team_id(team_id, SESSION)


@pytest.fixture
def store() -> OrgStore:
    store = OrgStore()
    for node in small_tree():
        store.upsert(node)
    return store


@pytest.fixture
def working(store: OrgStore) -> OrgStore:
    """The small tree with a session opened at its top team."""
    open_session_in(store, "root", SESSION)
    return store


def record(store: OrgStore, focus_id: str, *, full: bool = False) -> list[OrgChange]:
    events: list[OrgChange] = []
    store.subscribe(focus_id, events.append, full=full)
    return events


def describe(events: list[OrgChange]) -> list[tuple[str, str]]:
    """The changes to the tree and to what waits on the human."""
    described: list[tuple[str, str]] = []
    for event in events:
        if event.type == "node_upsert":
            described.append((event.type, event.node.id))
        elif event.type == "node_removed":
            described.append((event.type, event.id))
        elif event.type == "escalations":
            ids = ",".join(item.escalation.id for item in event.escalations)
            described.append((event.type, ids))
    return described


def test_derives_the_teams_below_and_above_each_team(store: OrgStore):
    root = store.team("root")
    assert root is not None

    assert root.child_ids == ["a", "b"]
    assert [node.id for node in store.sub_teams("a")] == ["a1", "a2"]
    assert [node.id for node in store.teams_above("a2")] == ["root", "a"]
    assert [node.id for node in store.teams()] == ["root", "a", "b", "a1", "a2"]


def test_snapshot_holds_the_focus_its_ancestors_and_direct_reports(store: OrgStore):
    snapshot = store.snapshot("a")

    assert snapshot is not None
    assert snapshot.focus_id == "a"
    assert [node.id for node in snapshot.nodes] == ["root", "a", "a1", "a2"]
    assert store.snapshot("missing") is None


def test_snapshot_adds_the_reports_of_expanded_teams(store: OrgStore):
    snapshot = store.snapshot("root", ["a", "b", "missing"])

    assert snapshot is not None
    assert [node.id for node in snapshot.nodes] == ["root", "a", "b", "a1", "a2"]


def test_subscribers_hear_about_the_reports_of_teams_they_expanded(store: OrgStore):
    collapsed: list[OrgChange] = []
    expanded: list[OrgChange] = []
    store.subscribe("root", collapsed.append)
    store.subscribe("root", expanded.append, ["a"])

    store.upsert(make_team("a1", "a", path="renamed"))
    store.remove("a2")

    assert describe(collapsed) == [("node_upsert", "a")]
    assert describe(expanded) == [
        ("node_upsert", "a1"),
        ("node_removed", "a2"),
        ("node_upsert", "a"),
    ]


def test_a_session_has_a_place_at_its_team_and_at_every_team_below(store: OrgStore):
    opened = open_session_in(store, "a", SESSION)

    assert opened.id == at("a") and store.opens_session(opened)
    assert opened.parent_id == division_lead_id("root", SESSION)
    assert [place.id for place in store.below(at("a"))] == [at("a1"), at("a2")]
    assert [place.id for place in store.above(at("a2"))] == [at("a")]
    assert store.above(at("a")) == []
    assert store.teams_below(at("a")) == 2
    assert store.instance("a2", SESSION) == store.place(at("a2"))
    assert store.place(at("b")) is None


def test_a_sessions_status_is_of_all_its_teams(working: OrgStore):
    def status() -> str:
        [session] = working.sessions()
        return session.status

    assert status() == "idle"

    # Only a team below works, or only the reviewer.
    working.set_agent(at("a1"), "ic", status="working")
    assert status() == "working"
    working.set_agent(at("a1"), "ic", status="idle")
    lead = division_lead_id("root", SESSION)
    working.update_place(
        lead, agents=[AgentRef(role="ic", conversation_id="lead", status="idle")]
    )
    working.set_agent(lead, "ic", status="working")
    assert status() == "working"

    # A team that waits on an answer shows once nothing else is at work,
    working.raise_escalation(at("b"), "ask", kind="design_decision", title="t")
    assert status() == "working"
    working.set_agent(lead, "ic", status="idle")
    assert status() == "blocked"

    # and the team it was opened at shows whatever else is at work.
    working.set_agent(lead, "ic", status="working")
    working.raise_escalation(at("root"), "top", kind="design_decision", title="t")
    assert status() == "blocked"


def test_a_place_is_no_team_of_the_tree(working: OrgStore):
    snapshot = working.snapshot("root", full=True)
    root = working.team("root")

    assert snapshot is not None and root is not None
    assert [node.id for node in snapshot.nodes] == ["root", "a", "b", "a1", "a2"]
    assert root.child_ids == ["a", "b"]
    assert working.team(at("a")) is None and working.place("a") is None
    assert working.get(at("a")) == working.place(at("a"))
    assert working.get("a") == working.team("a")


class TestEscalations:
    @pytest.fixture
    def store(self, working: OrgStore) -> OrgStore:
        working.update_place(
            at("a2"),
            escalations=[
                make_escalation(
                    "late",
                    raised_at="2026-11-12T00:00:00Z",
                    passed_up_by=(at("a"), at("root")),
                )
            ],
        )
        working.update_place(
            at("b"),
            escalations=[
                make_escalation("early", passed_up_by=(at("root"),)),
                make_escalation(
                    "broken",
                    kind="critical_issue",
                    raised_at="2026-11-13T00:00:00Z",
                    passed_up_by=(at("root"),),
                ),
            ],
        )
        return working

    def test_lists_critical_issues_first_then_the_longest_waiting(
        self, store: OrgStore
    ):
        listed = store.escalations()

        assert [item.escalation.id for item in listed] == ["broken", "early", "late"]
        assert [step.id for step in listed[-1].path] == ["root", "a", "a2"]
        assert (listed[-1].node_id, listed[-1].session_id) == ("a2", SESSION)

    def test_a_snapshot_carries_what_waits_on_the_human_anywhere_in_the_tree(
        self, store: OrgStore
    ):
        snapshot = store.snapshot("a1")

        assert snapshot is not None
        assert [item.escalation.id for item in snapshot.escalations] == [
            "broken",
            "early",
            "late",
        ]

    def test_raising_and_settling_announce_the_new_list(self, store: OrgStore):
        events = record(store, "a")

        store.update_place(
            at("a1"),
            escalations=[
                make_escalation(
                    "new", kind="critical_issue", passed_up_by=(at("a"), at("root"))
                )
            ],
        )
        store.update_place(at("a2"), escalations=[])

        assert describe(events) == [
            ("escalations", "new,broken,early,late"),
            ("escalations", "new,broken,early"),
        ]

    def test_changes_that_leave_the_list_alone_announce_nothing(self, store: OrgStore):
        events = record(store, "a")

        store.set_agent(at("a1"), "ic", status="working")
        # Held by the place above: not the human's business yet.
        store.update_place(
            at("a1"), escalations=[make_escalation("held", held_by=at("a"))]
        )

        assert "escalations" not in [event.type for event in events]

    def test_removing_a_place_drops_its_escalations_and_those_below(
        self, store: OrgStore
    ):
        events = record(store, "root")

        store.remove(at("a"))

        assert [item.escalation.id for item in store.escalations()] == [
            "broken",
            "early",
        ]
        assert describe(events) == [("escalations", "broken,early")]

    def test_closing_a_session_drops_its_escalations(self, store: OrgStore):
        events = record(store, "root")

        assert store.close_session(SESSION) is True

        assert store.escalations() == []
        assert describe(events) == [("escalations", "")]

    def test_renaming_a_place_on_the_path_announces_the_new_path(self, store: OrgStore):
        events = record(store, "root")

        store.update_place(at("a"), title="renamed")

        told = [event for event in events if event.type == "escalations"]
        assert len(told) == 1
        assert [step.title for step in told[0].escalations[-1].path] == [
            "root",
            "renamed",
            "a2",
        ]

    def test_only_what_waits_on_the_human_is_listed(self, store: OrgStore):
        store.update_place(
            at("a1"), escalations=[make_escalation("held", held_by=at("a"))]
        )

        assert "held" not in [item.escalation.id for item in store.escalations()]
        assert len(store.escalations()) == 3

    def test_escalations_are_not_saved_with_the_tree(self, store: OrgStore, repo: Path):
        # What a team waits on is of the work, which is done in sessions
        # and kept with them; the saved tree is the org's reference only.
        board = create_org_board(repo)
        save_org(board, store)
        loaded = OrgStore()

        load_org(board, loaded)

        assert store.escalations()
        assert len(loaded) == len(store)
        assert (loaded.places(), loaded.escalations()) == ([], [])


class TestEscalationChain:
    """An escalation climbs one place at a time and reaches the human last."""

    @pytest.fixture
    def store(self, working: OrgStore) -> OrgStore:
        return working

    def test_a_raised_escalation_goes_to_the_place_above_not_to_the_human(
        self, store: OrgStore
    ):
        events = record(store, "root")

        raised = store.raise_escalation(
            at("a2"), "jwt", kind="design_decision", title="JWT or opaque?"
        )

        assert (raised.held_by, raised.handoffs) == (at("a"), [])
        assert store.escalations() == []
        assert "escalations" not in [event.type for event in events]

    def test_reaches_the_human_only_after_every_place_above_passed_it_up(
        self, store: OrgStore
    ):
        store.raise_escalation(at("a2"), "jwt", kind="design_decision", title="t")
        events = record(store, "root")

        first = store.pass_up_escalation(at("a2"), "jwt", reason="  needs b too ")
        waiting_after_first = store.escalations()
        second = store.pass_up_escalation(at("a2"), "jwt", reason="needs a human")

        assert first.held_by == at("root")
        assert waiting_after_first == []
        assert second.held_by is None
        assert [(h.node_id, h.reason) for h in second.handoffs] == [
            (at("a"), "needs b too"),
            (at("root"), "needs a human"),
        ]
        assert [e.escalation.id for e in store.escalations()] == ["jwt"]
        assert describe(events) == [("escalations", "jwt")]

    def test_an_escalation_of_the_place_a_session_was_opened_at_waits_on_the_human(
        self, store: OrgStore
    ):
        # The session's Division Lead place is above, and nobody
        # stands there.
        raised = store.raise_escalation(
            at("root"), "budget", kind="critical_issue", title="t"
        )

        assert raised.held_by is None
        assert [e.escalation.id for e in store.escalations()] == ["budget"]

    def test_a_team_of_the_tree_raises_nothing(self, store: OrgStore):
        with pytest.raises(NotFound):
            store.raise_escalation("a2", "jwt", kind="design_decision", title="t")

        assert store.escalations() == []

    def test_settling_closes_it_wherever_it_is_held(self, store: OrgStore):
        store.raise_escalation(at("a2"), "jwt", kind="design_decision", title="t")
        store.raise_escalation(at("b"), "leak", kind="critical_issue", title="t")
        store.pass_up_escalation(at("b"), "leak", reason="cannot")

        store.settle_escalation(at("a2"), "jwt")
        store.settle_escalation(at("b"), "leak")

        assert [place.escalations for place in store.places()] == [[]] * 6
        assert store.escalations() == []

    def test_cannot_be_passed_up_past_the_human_or_without_a_reason(
        self, store: OrgStore
    ):
        store.raise_escalation(at("b"), "leak", kind="critical_issue", title="t")

        with pytest.raises(OrgStoreError, match="needs a reason"):
            store.pass_up_escalation(at("b"), "leak", reason="   ")
        store.pass_up_escalation(at("b"), "leak", reason="cannot")
        with pytest.raises(OrgStoreError, match="already waits on the human"):
            store.pass_up_escalation(at("b"), "leak", reason="again")

    def test_refuses_a_duplicate_id_and_unknown_places_or_escalations(
        self, store: OrgStore
    ):
        store.raise_escalation(at("b"), "leak", kind="critical_issue", title="t")

        with pytest.raises(OrgStoreError, match="already has an escalation"):
            store.raise_escalation(at("b"), "leak", kind="critical_issue", title="t")
        with pytest.raises(NotFound):
            store.raise_escalation("nope", "x", kind="critical_issue", title="t")
        with pytest.raises(NotFound):
            store.pass_up_escalation(at("b"), "nope", reason="r")
        with pytest.raises(NotFound):
            store.settle_escalation(at("b"), "nope")

    @pytest.mark.parametrize(
        "escalation",
        [
            # Straight to the human.
            make_escalation("x"),
            # Skipped the place above.
            make_escalation("x", passed_up_by=(at("root"),)),
            # Passed up by the place above but still held by it.
            make_escalation("x", passed_up_by=(at("a"),), held_by=at("a")),
            # Held by a place that is not above it.
            make_escalation("x", held_by=at("b")),
            # Held by the session's Division Lead place.
            make_escalation(
                "x",
                passed_up_by=(at("a"), at("root")),
                held_by=division_lead_id("root", SESSION),
            ),
        ],
        ids=[
            "no-handoffs",
            "skipped-parent",
            "wrong-holder",
            "holder-not-above",
            "held-by-the-review-place",
        ],
    )
    def test_refuses_an_escalation_that_skipped_a_place_on_its_way_up(
        self, store: OrgStore, escalation: Escalation
    ):
        before = store.places()
        fresh = OrgStore()
        fresh.replace_all(small_tree())
        session, places = session_places(small_tree(), "root", SESSION)
        places[-1] = places[-1].model_copy(update={"escalations": [escalation]})

        with pytest.raises(OrgStoreError, match="must be passed up through"):
            store.update_place(at("a2"), escalations=[escalation])
        with pytest.raises(OrgStoreError, match="must be passed up through"):
            fresh.open_session(session, places)

        assert store.places() == before
        assert (fresh.places(), fresh.sessions()) == ([], [])


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
        events = record(store, "a", full=True)

        store.upsert(make_team("deep", "a1"))
        store.upsert(make_team("deeper", "deep"))
        store.upsert(make_team("elsewhere", "b"))
        store.remove("deep")

        assert describe(events) == [
            ("node_upsert", "deep"),
            ("node_upsert", "a1"),
            ("node_upsert", "deeper"),
            ("node_upsert", "deep"),
            ("node_removed", "deeper"),
            ("node_removed", "deep"),
            ("node_upsert", "a1"),
        ]


def test_an_update_announces_only_the_team(store: OrgStore):
    events = record(store, "a2")

    store.update_team("a2", title="renamed")

    assert describe(events) == [("node_upsert", "a2")]


def test_updating_a_team_keeps_its_children(store: OrgStore):
    updated = store.upsert(make_team("a", "root", path="svc/a"))

    assert updated.child_ids == ["a1", "a2"]


def test_subscribers_only_hear_about_teams_in_their_snapshot(store: OrgStore):
    at_b = record(store, "b")
    at_root = record(store, "root")

    # a1 is a grandchild of root and unrelated to b; a is a child of root.
    store.update_team("a1", headline="one")
    store.update_team("a", headline="two")
    store.update_team("root", headline="three")

    assert describe(at_root) == [("node_upsert", "a"), ("node_upsert", "root")]
    assert describe(at_b) == [("node_upsert", "root")]


def test_a_new_child_announces_itself_and_its_parents_child_list(store: OrgStore):
    events = record(store, "b")

    store.upsert(make_team("b1", "b", path="svc/b/one"))

    assert describe(events) == [("node_upsert", "b1"), ("node_upsert", "b")]
    parent = events[1]
    assert parent.type == "node_upsert"
    assert parent.node.child_ids == ["b1"]


def test_removing_a_subtree_announces_leaves_first(store: OrgStore):
    inside = record(store, "a1")
    above = record(store, "root")
    whole = record(store, "root", full=True)

    assert store.remove("a") is True

    # A viewer inside the removed subtree hears its focus and the subtree's
    # root go, and nothing after that.
    assert describe(inside) == [("node_removed", "a1"), ("node_removed", "a")]
    assert describe(above) == [("node_removed", "a"), ("node_upsert", "root")]
    assert describe(whole)[-2:] == [("node_removed", "a"), ("node_upsert", "root")]
    assert sorted(describe(whole)[:2]) == [
        ("node_removed", "a1"),
        ("node_removed", "a2"),
    ]
    root = store.team("root")
    assert root is not None
    assert root.child_ids == ["b"]
    assert store.team("a2") is None
    assert store.remove("a") is False


def test_the_root_can_be_removed_and_replaced(store: OrgStore):
    store.remove("root")

    assert len(store) == 0
    assert store.root_id is None
    assert store.upsert(make_team("new-root", None)).id == "new-root"


def test_unsubscribing_stops_delivery(store: OrgStore):
    events: list[OrgChange] = []
    snapshot, unsubscribe = store.subscribe("root", events.append)
    assert snapshot is not None

    unsubscribe()
    store.update_team("b", headline="changed")

    assert events == []


def test_subscribing_to_an_unknown_team_registers_nothing(store: OrgStore):
    events: list[OrgChange] = []

    snapshot, _ = store.subscribe("missing", events.append)
    store.upsert(make_team("missing", "root"))

    assert snapshot is None
    assert events == []


@pytest.mark.parametrize(
    "node",
    [
        make_team("second-root", None),
        make_team("orphan", "nowhere"),
        make_team("a1", "b"),
    ],
    ids=["second root", "missing parent", "moved to another parent"],
)
def test_rejects_teams_that_do_not_fit_the_tree(store: OrgStore, node):
    before = store.teams()

    with pytest.raises(OrgStoreError):
        store.upsert(node)

    assert store.teams() == before


class TestReplaceAll:
    def test_loads_nodes_in_any_order(self):
        store = OrgStore()

        store.replace_all(reversed(small_tree()))

        root = store.team("root")
        assert root is not None
        assert root.child_ids == ["b", "a"]
        assert [node.id for node in store.teams_above("a1")] == ["root", "a"]

    def test_accepts_an_empty_tree(self, store: OrgStore):
        store.replace_all([])

        assert len(store) == 0
        assert store.root_id is None

    @pytest.mark.parametrize(
        "nodes",
        [
            [make_team("root", None), make_team("root", None)],
            [make_team("root", None), make_team("x", "nowhere")],
            [make_team("r1", None), make_team("r2", None)],
            [make_team("x", "y"), make_team("y", "x")],
            [make_team("root", None), make_team("x", "y"), make_team("y", "x")],
        ],
        ids=["duplicate id", "orphan", "two roots", "no root", "detached cycle"],
    )
    def test_a_malformed_tree_leaves_the_store_untouched(self, store: OrgStore, nodes):
        before = store.teams()

        with pytest.raises(OrgStoreError):
            store.replace_all(nodes)

        assert store.teams() == before


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
        assert loaded.teams() == store.teams()
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


def give_agents(store: OrgStore, place_id: str, ic: str, manager: str) -> None:
    store.update_place(
        place_id,
        agents=[
            AgentRef(role="ic", conversation_id=ic, status="idle"),
            AgentRef(role="manager", conversation_id=manager, status="idle"),
        ],
    )


def test_an_agents_conversation_leads_back_to_its_place_and_role(working: OrgStore):
    give_agents(working, at("root"), "conv-ic", "conv-mgr")

    found = working.find_by_conversation("conv-mgr")

    assert found is not None
    place, role = found
    assert (place.id, role) == (at("root"), "manager")
    assert working.find_by_conversation("conv-unknown") is None


def test_a_removed_or_replaced_conversation_no_longer_leads_to_a_place(
    working: OrgStore,
):
    give_agents(working, at("root"), "conv-ic", "conv-mgr")
    give_agents(working, at("a"), "conv-a-ic", "conv-a-mgr")

    working.set_agent(at("root"), "ic", conversation_id="conv-ic-2")
    working.remove(at("a"))

    assert working.find_by_conversation("conv-ic") is None
    assert working.find_by_conversation("conv-a-ic") is None
    found = working.find_by_conversation("conv-ic-2")
    assert found is not None and found[1] == "ic"


def test_a_closed_sessions_conversations_lead_nowhere(working: OrgStore):
    give_agents(working, at("a"), "conv-a-ic", "conv-a-mgr")

    working.close_session(SESSION)

    assert working.find_by_conversation("conv-a-ic") is None
    assert working.places() == []


def test_a_replaced_tree_keeps_its_sessions_and_their_agents(working: OrgStore):
    give_agents(working, at("a"), "conv-a-ic", "conv-a-mgr")

    working.replace_all(small_tree())

    found = working.find_by_conversation("conv-a-ic")
    assert found is not None and found[0].id == at("a")
    assert [session.id for session in working.sessions()] == [SESSION]


def test_setting_an_agent_status_announces_the_session_once(working: OrgStore):
    events = record(working, "a2")

    working.set_agent(at("a2"), "ic", status="working")
    working.set_agent(at("a2"), "ic", status="working")

    assert [event.type for event in events] == ["sessions"]
    place = working.place(at("a2"))
    assert place is not None and place.agents[0].status == "working"


@pytest.mark.parametrize(
    ("agent", "review", "escalated_by", "expected"),
    [
        pytest.param("idle", None, None, "idle", id="nothing under way"),
        pytest.param("working", None, None, "working", id="an agent runs"),
        pytest.param("blocked", None, None, "blocked", id="an agent failed"),
        pytest.param("lost", None, None, "blocked", id="an agent is gone"),
        pytest.param("idle", "pending", None, "done", id="its work waits for review"),
        pytest.param("idle", "changes_requested", None, "idle", id="work sent back"),
        pytest.param("idle", "accepted", None, "done", id="its work is accepted"),
        pytest.param("working", "accepted", None, "working", id="it works on"),
        pytest.param("lost", "accepted", None, "done", id="done and then gone"),
        pytest.param("working", None, "ic", "blocked", id="its IC waits on an answer"),
        pytest.param("idle", "accepted", "ic", "blocked", id="an answer comes first"),
        pytest.param("working", None, "manager", "working", id="its Manager waits"),
    ],
)
def test_a_teams_status_is_worked_out_and_never_declared(
    working: OrgStore,
    agent: str,
    review: ReviewState | None,
    escalated_by: AgentRole | None,
    expected: str,
):
    if review is not None:
        working.set_review(at("a2"), Review(state=review, at=AT))
    if escalated_by is not None:
        working.raise_escalation(
            at("a2"),
            "jwt",
            kind="design_decision",
            title="JWT?",
            raised_by=escalated_by,
        )

    place = working.set_agent(at("a2"), "ic", status=agent)

    assert place.status == expected
    assert status_of(place) == expected


def test_a_team_stops_being_blocked_when_its_ics_escalation_is_settled(
    working: OrgStore,
):
    working.set_agent(at("a2"), "ic", status="working")
    blocked = working.raise_escalation(
        at("a2"), "jwt", kind="design_decision", title="JWT?", raised_by="ic"
    )
    waiting = working.place(at("a2"))

    working.settle_escalation(at("a2"), blocked.id)
    after = working.place(at("a2"))

    assert waiting is not None and waiting.status == "blocked"
    assert after is not None and after.status == "working"


def test_a_review_set_on_a_team_shows_in_its_status_and_keeps_other_changes(
    working: OrgStore,
):
    place = working.set_review(
        at("a2"), Review(state="accepted", at=AT), base_commit="abc"
    )

    assert (place.status, place.base_commit) == ("done", "abc")
    with pytest.raises(NotFound):
        working.set_review("a2", Review(state="pending", at=AT))


def test_a_place_blocked_by_a_failed_agent_resumes_when_that_agent_does(
    working: OrgStore,
):
    working.set_agent(at("a2"), "ic", status="blocked")

    assert working.set_agent(at("a2"), "ic", status="working").status == "working"


def test_setting_an_agent_that_is_not_there_is_refused(working: OrgStore):
    with pytest.raises(NotFound):
        working.set_agent(at("a"), "manager", status="working")
    with pytest.raises(NotFound):
        working.set_agent("nowhere", "ic", status="working")
    # A team of the tree has no agents: only its places do.
    with pytest.raises(NotFound):
        working.set_agent("a", "ic", status="working")


def test_updating_a_team_keeps_its_other_fields(store: OrgStore):
    store.update_team("a1", spec_summary="stuck on the schema")

    node = store.team("a1")
    assert node is not None
    assert node.spec_summary == "stuck on the schema"
    assert (node.boundary.path, node.parent_id) == ("svc/a/one", "a")
    with pytest.raises(NotFound):
        store.update_team("missing", spec_summary="x")


def test_updating_a_place_keeps_its_other_fields(working: OrgStore):
    working.update_place(at("a1"), base_commit="abc")

    place = working.place(at("a1"))
    assert place is not None
    assert (place.base_commit, place.team_id) == ("abc", "a1")
    assert [agent.role for agent in place.agents] == ["ic"]
    with pytest.raises(NotFound):
        working.update_place("a1", base_commit="abc")


def test_updating_a_team_cannot_move_it(working: OrgStore):
    with pytest.raises(OrgStoreError):
        working.update_team("a1", parent_id="b")
    with pytest.raises(OrgStoreError):
        working.update_place(at("a1"), parent_id=at("b"))


def test_a_moved_team_takes_what_is_below_it(store: OrgStore):
    events = record(store, "root", full=True)
    under = record(store, "a1")

    moved = store.move("a", "b")

    assert moved.parent_id == "b"
    assert [node.id for node in store.sub_teams("root")] == ["b"]
    assert [node.id for node in store.sub_teams("b")] == ["a"]
    assert [node.id for node in store.teams_above("a1")] == ["root", "b", "a"]
    # Every subscriber is shown its part anew, also one below the team,
    # which no single team's change would reach.
    [shown] = events
    assert isinstance(shown, SnapshotEvent)
    assert {node.id: node.parent_id for node in shown.nodes}["a"] == "b"
    [below] = under
    assert isinstance(below, SnapshotEvent)
    assert [node.id for node in below.nodes] == ["root", "b", "a", "a1"]
    # It now hears of the team that came to be above it.
    store.update_team("b", title="B again")
    assert describe(under) == [("node_upsert", "b")]


def test_moving_a_team_of_the_tree_leaves_the_places_of_a_session(working: OrgStore):
    working.move("a1", "b")

    assert [place.id for place in working.below(at("a"))] == [at("a1"), at("a2")]
    assert working.below(at("b")) == []


def test_a_moved_place_takes_what_is_below_it(working: OrgStore):
    moved = working.move_place(at("a"), at("b"))

    assert moved.parent_id == at("b")
    assert [place.id for place in working.below(at("root"))] == [at("b")]
    assert [place.id for place in working.above(at("a1"))] == [
        at("root"),
        at("b"),
        at("a"),
    ]
    assert [node.id for node in working.sub_teams("root")] == ["a", "b"]


def test_an_escalation_of_a_moved_place_is_held_by_the_place_now_above_it(
    working: OrgStore,
):
    working.raise_escalation(at("a1"), "jwt", kind="design_decision", title="JWT?")
    working.pass_up_escalation(at("a1"), "jwt", reason="needs b too")
    working.raise_escalation(at("a2"), "fresh", kind="design_decision", title="Cache?")

    working.move_place(at("a1"), at("b"))
    working.move_place(at("a2"), at("root"))

    a1, a2 = working.place(at("a1")), working.place(at("a2"))
    assert a1 is not None and a2 is not None
    # a is no longer above a1, so what it said goes and b gets to answer.
    assert (a1.escalations[0].held_by, a1.escalations[0].handoffs) == (at("b"), [])
    assert a2.escalations[0].held_by == at("root")


@pytest.mark.parametrize(
    ("node_id", "parent_id", "error"),
    [
        pytest.param("root", "a", OrgStoreError, id="the root"),
        pytest.param("a", "a", OrgStoreError, id="under itself"),
        pytest.param("a", "a1", OrgStoreError, id="under a team below it"),
        pytest.param("a", "missing", NotFound, id="under a team there is not"),
        pytest.param("missing", "a", NotFound, id="a team there is not"),
    ],
)
def test_a_team_moves_only_to_where_the_tree_stays_a_tree(
    store: OrgStore, node_id: str, parent_id: str, error: type[Exception]
):
    with pytest.raises(error):
        store.move(node_id, parent_id)

    assert [node.id for node in store.sub_teams("root")] == ["a", "b"]


@pytest.mark.parametrize(
    ("place_id", "parent_id", "error"),
    [
        pytest.param(at("root"), at("a"), OrgStoreError, id="where it was opened"),
        pytest.param(at("a"), at("a"), OrgStoreError, id="under itself"),
        pytest.param(at("a"), at("a1"), OrgStoreError, id="under a place below it"),
        pytest.param(
            at("a1"),
            division_lead_id("root", SESSION),
            OrgStoreError,
            id="under its Division Lead place",
        ),
        pytest.param(
            at("a1"),
            session_team_id("b", "other"),
            OrgStoreError,
            id="under a place of another session",
        ),
        pytest.param(at("a"), "b", NotFound, id="under a team of the tree"),
        pytest.param("a", at("b"), NotFound, id="a team of the tree"),
    ],
)
def test_a_place_moves_only_inside_its_own_session(
    working: OrgStore, place_id: str, parent_id: str, error: type[Exception]
):
    open_session_in(working, "b", "other")

    with pytest.raises(error):
        working.move_place(place_id, parent_id)

    assert [place.id for place in working.below(at("root"))] == [at("a"), at("b")]
    assert [place.id for place in working.below(at("a"))] == [at("a1"), at("a2")]


def test_change_listeners_hear_every_change_until_stopped(store: OrgStore):
    calls: list[int] = []
    stop = store.on_change(lambda: calls.append(1))

    store.update_team("b", headline="one")
    store.remove("a2")
    stop()
    store.update_team("b", headline="two")

    assert len(calls) == 2


def test_saving_does_not_record_the_sessions_or_their_agents(
    repo: Path, working: OrgStore
):
    board = create_org_board(repo)
    save_org(board, working)
    saved = run_git(repo, "rev-parse", ORG_REF)

    give_agents(working, at("root"), "conv-ic", "conv-mgr")
    working.set_agent(at("root"), "ic", status="working")
    save_org(board, working)

    assert run_git(repo, "rev-parse", ORG_REF) == saved
    loaded = OrgStore()
    assert load_org(board, loaded)
    assert loaded.find_by_conversation("conv-ic") is None
    assert loaded.sessions() == []


def test_sessions_come_back_whatever_order_their_places_were_kept_in():
    session, places = session_places(small_tree(), "root", "s")
    store = OrgStore()
    store.replace_all(small_tree())

    # A place that was made before the one it now hangs under.
    store.restore_sessions([(session, list(reversed(places)))])

    assert [kept.id for kept in store.sessions()] == ["s"]
    assert {place.id for place in store.session("s")} == {p.id for p in places}


def test_a_tree_read_again_is_shown_to_those_who_follow_it(store: OrgStore):
    events = record(store, "root", full=True)

    store.replace_all(node for node in small_tree() if node.id in ("root", "b"))

    [shown] = events
    assert isinstance(shown, SnapshotEvent)
    assert [node.id for node in shown.nodes] == ["root", "b"]
