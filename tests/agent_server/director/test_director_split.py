"""Reading and checking the sub-teams an IC proposes."""

import pytest

from openhands.agent_server.director.models import DirectorSettings, SplitChild
from openhands.agent_server.director.org_store import (
    OrgStore,
    OrgStoreError,
    session_team_id,
)
from openhands.agent_server.director.split import (
    MAX_SPLIT_CHILDREN,
    ProposedTeam,
    Split,
    as_written,
    check_limits,
    home_of,
    normalize_split,
    proposed_teams,
)
from tests.agent_server.director.conftest import (
    make_team,
    open_session_in,
    shipped_type,
    small_tree,
)


SESSION = "s"


def at(team_id: str) -> str:
    """The place the session has at a team."""
    return session_team_id(team_id, SESSION)


@pytest.fixture
def store() -> OrgStore:
    """A session at the top of the small tree, each of its places in a
    checkout of its own."""
    store = OrgStore(shipped_type)
    store.replace_all(small_tree())
    open_session_in(store, "root", SESSION)
    for team in small_tree():
        store.update_place(at(team.id), worktree_path=f"/wt/{team.id}")
    return store


def child(title: str, path: str, **fields) -> SplitChild:
    return SplitChild(title=title, path=path, task=f"build {title}", **fields)


def check(store: OrgStore, team_id: str, children: list[SplitChild]) -> Split:
    place = store.place(at(team_id))
    assert place is not None
    return normalize_split(store, place, children)


def replaced(split: Split) -> list[str]:
    return [team.team_id for team in split.replaces]


def test_a_proposal_is_read_as_the_teams_it_names():
    proposed = proposed_teams(
        [
            ProposedTeam(
                name="API",
                path="services/api",
                instruction="Build the orders endpoint",
                model="fast",
                writable=["services/api/src/**"],
                readonly=["shared/**"],
                isolation="worktree",
            ),
            ProposedTeam(name="Web", path="apps/web", instruction="Build the pages"),
        ],
        "API serves /orders.\n",
        ["Billing"],
        " The service has two concerns. ",
    )

    assert proposed.contracts == "API serves /orders.\n"
    assert proposed.keep == ["Billing"]
    assert proposed.reason == "The service has two concerns."
    assert proposed.children == [
        SplitChild(
            title="API",
            path="services/api",
            task="Build the orders endpoint",
            agent_profile="fast",
            writable=["services/api/src/**"],
            readonly=["shared/**"],
            isolation="worktree",
        ),
        SplitChild(title="Web", path="apps/web", task="Build the pages"),
    ]
    assert as_written(proposed).startswith("contracts: |\n  API serves /orders.\n")


def test_sub_teams_get_clean_paths_and_write_their_whole_directory(store: OrgStore):
    result = check(
        store,
        "b",
        [
            child("api", "./svc/b/api/", readonly=["./shared/**"]),
            child("db", "svc/b/db"),
        ],
    )

    assert [(c.path, c.writable, c.readonly) for c in result.children] == [
        ("svc/b/api", ["svc/b/api/**"], ["shared/**"]),
        ("svc/b/db", ["svc/b/db/**"], []),
    ]
    assert result.replaces == []


def test_the_root_can_hand_out_any_directory(store: OrgStore):
    result = check(store, "root", [child("docs", "docs")])

    assert result.children[0].path == "docs"


@pytest.mark.parametrize(
    "children",
    [
        pytest.param([child("x", "svc/b")], id="the parent's own directory"),
        pytest.param([child("x", "svc/c/x")], id="outside the parent"),
        pytest.param([child("x", "svc/bee")], id="a sibling name sharing a prefix"),
        pytest.param([child("x", "svc/b/../a/x")], id="climbing out"),
        pytest.param([child("x", "/svc/b/x")], id="absolute"),
        pytest.param([child("x", "")], id="empty"),
        pytest.param(
            [child("x", "svc/b/x"), child("y", "svc/b/x/inner")],
            id="nested in each other",
        ),
        pytest.param(
            [child("x", "svc/b/x"), child("y", "svc/b/x")], id="the same directory"
        ),
        pytest.param(
            [child("x", "svc/b/x"), child("x", "svc/b/y")], id="the same title"
        ),
        pytest.param(
            [child("x", "svc/b/x", writable=["svc/b/shared/**"])],
            id="writing outside its directory",
        ),
        pytest.param(
            [child("x", "svc/b/x", readonly=["../elsewhere/**"])],
            id="reading outside the repository",
        ),
        pytest.param(
            [child(f"t{i}", f"svc/b/t{i}") for i in range(MAX_SPLIT_CHILDREN + 1)],
            id="too many",
        ),
        pytest.param([], id="none"),
    ],
)
def test_refuses_a_split_that_cannot_be_told_apart(store: OrgStore, children):
    with pytest.raises(OrgStoreError):
        check(store, "b", children)


def test_a_team_proposes_only_inside_what_it_may_write(store: OrgStore):
    b = make_team("b", "root", path="svc/b")
    narrow = b.boundary.model_copy(update={"writable": ["svc/b/src/**"]})
    store.update_place(at("b"), boundary=narrow)

    assert check(store, "b", [child("core", "svc/b/src/core")]).children
    with pytest.raises(OrgStoreError, match="inside what your team may write"):
        check(store, "b", [child("docs", "svc/b/docs")])


def test_a_new_team_beside_the_ones_a_team_has_replaces_none(store: OrgStore):
    result = check(store, "a", [child("three", "svc/a/three")])

    assert result.children[0].path == "svc/a/three" and result.replaces == []
    with pytest.raises(OrgStoreError, match="titled 'a1'"):
        check(store, "a", [child("a1", "svc/a/three")])


def test_a_team_over_the_directories_of_existing_ones_replaces_them(store: OrgStore):
    merged = check(store, "root", [child("services", "svc")])

    assert replaced(merged) == ["a", "b"]


def test_a_team_with_the_directory_of_an_existing_one_replaces_it(store: OrgStore):
    again = check(store, "root", [child("a", "svc/a"), child("docs", "docs")])

    assert replaced(again) == ["a"]


def test_teams_inside_the_directory_of_an_existing_one_replace_it(store: OrgStore):
    split_down = check(
        store, "root", [child("http", "svc/b/http"), child("db", "svc/b/db")]
    )

    assert replaced(split_down) == ["b"]


def test_a_new_team_may_not_sit_inside_a_team_that_stays(store: OrgStore):
    # a is replaced, but its sub-team a1 (svc/a/one) stays and is moved.
    with pytest.raises(OrgStoreError, match="would sit inside 'a1'"):
        check(store, "root", [child("deep", "svc/a/one/deep")])
    with pytest.raises(OrgStoreError, match="would sit inside 'a1'"):
        check(store, "root", [child("one", "svc/a/one")])


def test_a_team_others_work_in_the_checkout_of_cannot_be_replaced(store: OrgStore):
    a = store.place(at("a"))
    assert a is not None
    store.update_place(at("a2"), worktree_path=a.worktree_path)

    with pytest.raises(OrgStoreError, match="work in its checkout"):
        check(store, "root", [child("services", "svc")])


def test_the_sub_teams_of_a_replaced_team_report_to_the_team_that_holds_them(
    store: OrgStore,
):
    a1, a2 = store.below(at("a"))

    assert home_of(a1, "root", {"svc/a/one": "new.one", "docs": "docs"}) == "new.one"
    assert home_of(a1, "root", {"svc": "services"}) == "services"
    assert home_of(a2, "root", {"svc/a/one": "new.one"}) == "root"


def test_what_a_proposal_keeps_is_asked_for_only_when_it_is_first_checked(
    store: OrgStore,
):
    root = store.place(at("root"))
    assert root is not None
    new = [child("C", "svc/c")]

    kept = normalize_split(store, root, new, ["a", "b"])
    again = normalize_split(store, root, new)

    assert [team.team_id for team in kept.keeps] == ["a", "b"]
    assert (replaced(kept), again.keeps) == ([], [])
    with pytest.raises(OrgStoreError, match="neither replaced by a proposed team"):
        normalize_split(store, root, new, ["a"])
    with pytest.raises(OrgStoreError, match="neither replaced by a proposed team"):
        normalize_split(store, root, new, [])


def limits_under_a(max_depth: int = 4, max_teams: int = 32) -> tuple[OrgStore, str]:
    """A session opened at ``a``, one team below the top, and the id of its
    place at ``a1``."""
    store = OrgStore(shipped_type)
    store.replace_all(small_tree())
    open_session_in(store, "a", "low")
    return store, session_team_id("a1", "low")


def test_levels_are_counted_from_the_top_of_the_org_not_of_the_session():
    store, a1 = limits_under_a()
    place = store.place(a1)
    assert place is not None

    # root, a and a1 are three levels, so teams under a1 are the fourth.
    check_limits(store, place, 1, DirectorSettings(max_depth=4))
    with pytest.raises(OrgStoreError, match="would be level 4"):
        check_limits(store, place, 1, DirectorSettings(max_depth=3))


def test_teams_are_counted_for_the_teams_of_the_org_above_the_session_too():
    store, a1 = limits_under_a()
    place = store.place(a1)
    assert place is not None

    # The org is five teams and a is three of them.
    check_limits(store, place, 1, DirectorSettings(max_teams=6))
    with pytest.raises(OrgStoreError, match="'root' is 5"):
        check_limits(store, place, 1, DirectorSettings(max_teams=5))
    with pytest.raises(OrgStoreError, match="'a' is 3"):
        check_limits(store, place, 1, DirectorSettings(max_teams=3))


def test_a_sub_team_may_only_read_what_its_parent_may_only_read_in_its_folder(
    store: OrgStore,
):
    place = store.place(at("b"))
    assert place is not None
    store.update_place(
        place.id,
        boundary=place.boundary.model_copy(
            update={
                "readonly": ["svc/b/api/gen/**", "svc/b/lock.json", "**/generated/**"]
            }
        ),
    )

    [api] = check(store, "b", [child("api", "svc/b/api")]).children

    assert api.readonly == ["svc/b/api/gen/**", "**/generated/**"]


def test_a_sub_team_cannot_own_what_its_parent_may_only_read(store: OrgStore):
    place = store.place(at("b"))
    assert place is not None
    store.update_place(
        place.id,
        boundary=place.boundary.model_copy(update={"readonly": ["svc/b/gen/**"]}),
    )

    with pytest.raises(OrgStoreError, match="inside what your team may write"):
        check(store, "b", [child("gen", "svc/b/gen")])
    assert check(store, "b", [child("api", "svc/b/api")]).children


def test_a_sub_team_cannot_write_more_of_its_folder_than_its_parent_may(
    store: OrgStore,
):
    place = store.place(at("b"))
    assert place is not None
    store.update_place(
        place.id,
        boundary=place.boundary.model_copy(update={"writable": ["svc/b/*.py"]}),
    )

    with pytest.raises(OrgStoreError, match="inside what your team may write"):
        check(store, "b", [child("api", "svc/b/api")])


def test_a_team_that_may_write_nothing_proposes_no_sub_team(store: OrgStore):
    place = store.place(at("b"))
    assert place is not None
    store.update_place(
        place.id, boundary=place.boundary.model_copy(update={"writable": []})
    )

    with pytest.raises(OrgStoreError, match="inside what your team may write"):
        check(store, "b", [child("api", "svc/b/api")])
