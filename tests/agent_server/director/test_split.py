"""Checks on an IC's proposal to split its boundary."""

import pytest

from openhands.agent_server.director.models import SplitChild
from openhands.agent_server.director.org_store import OrgStore, OrgStoreError
from openhands.agent_server.director.split import MAX_SPLIT_CHILDREN, normalize_split
from tests.agent_server.director.conftest import small_tree


@pytest.fixture
def store() -> OrgStore:
    store = OrgStore()
    for node in small_tree():
        store.upsert(node)
    return store


def child(title: str, path: str, **fields) -> SplitChild:
    return SplitChild(title=title, path=path, task=f"build {title}", **fields)


def check(store: OrgStore, node_id: str, children: list[SplitChild]):
    node = store.get(node_id)
    assert node is not None
    return normalize_split(node, store.children(node_id), children)


def test_sub_teams_get_clean_paths_and_write_their_whole_directory(store: OrgStore):
    result = check(store, "b", [child("api", "./svc/b/api/"), child("db", "svc/b/db")])

    assert [(c.path, c.writable) for c in result] == [
        ("svc/b/api", ["svc/b/api/**"]),
        ("svc/b/db", ["svc/b/db/**"]),
    ]


def test_the_root_can_hand_out_any_directory(store: OrgStore):
    result = check(store, "root", [child("docs", "docs")])

    assert result[0].path == "docs"


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
            [child(f"t{i}", f"svc/b/t{i}") for i in range(MAX_SPLIT_CHILDREN + 1)],
            id="too many",
        ),
    ],
)
def test_refuses_a_split_that_cannot_be_told_apart(store: OrgStore, children):
    with pytest.raises(OrgStoreError):
        check(store, "b", children)


def test_a_new_sub_team_may_not_overlap_one_the_node_already_has(store: OrgStore):
    with pytest.raises(OrgStoreError):
        check(store, "a", [child("deeper", "svc/a/one/deeper")])
    with pytest.raises(OrgStoreError):
        check(store, "a", [child("a1", "svc/a/three")])

    assert check(store, "a", [child("three", "svc/a/three")])[0].path == "svc/a/three"
