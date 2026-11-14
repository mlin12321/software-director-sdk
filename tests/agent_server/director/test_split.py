"""Reading and checking the sub-teams an IC proposes."""

import pytest

from openhands.agent_server.director.models import SplitChild
from openhands.agent_server.director.org_store import OrgStore, OrgStoreError
from openhands.agent_server.director.split import (
    MAX_SPLIT_CHILDREN,
    Split,
    home_of,
    normalize_split,
    proposed_teams,
)
from tests.agent_server.director.conftest import make_node, small_tree


@pytest.fixture
def store() -> OrgStore:
    """The small tree, each team in a checkout of its own."""
    store = OrgStore()
    for node in small_tree():
        store.upsert(node.model_copy(update={"worktree_path": f"/wt/{node.id}"}))
    return store


def child(title: str, path: str, **fields) -> SplitChild:
    return SplitChild(title=title, path=path, task=f"build {title}", **fields)


def check(store: OrgStore, node_id: str, children: list[SplitChild]) -> Split:
    node = store.get(node_id)
    assert node is not None
    return normalize_split(store, node, children)


def replaced(split: Split) -> list[str]:
    return [team.id for team in split.replaces]


WRITTEN = """\
I would split the service in two.

```teams
contracts: |
  API serves /orders.
teams:
  - name: API
    path: services/api
    instruction: Build the orders endpoint
    model: fast
    writable: [services/api/src/**]
    readonly: [shared/**]
    isolation: worktree
  - name: Web
    path: apps/web
    instruction: Build the pages
```

Tell me what you think.
"""


def test_a_message_proposes_the_teams_in_its_block():
    proposed = proposed_teams(WRITTEN)

    assert proposed is not None
    assert proposed.contracts == "API serves /orders.\n"
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


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("The API team is behind.", id="no block"),
        pytest.param("```yaml\nteams: []\n```", id="a block of another kind"),
        pytest.param("Use ```teams when you propose.", id="the fence in a sentence"),
    ],
)
def test_a_message_without_a_teams_block_proposes_nothing(text: str):
    assert proposed_teams(text) is None


@pytest.mark.parametrize(
    ("block", "problem"),
    [
        pytest.param("teams: [", "is not YAML", id="broken YAML"),
        pytest.param("- API\n- Web", "does not describe sub-teams", id="a list"),
        pytest.param("teams: []", "teams", id="no teams"),
        pytest.param(
            "teams:\n  - name: API\n    path: a", "instruction", id="no instruction"
        ),
        pytest.param(
            "teams:\n  - name: API\n    path: a\n    instruction: x\n    manager: y",
            "manager",
            id="a field teams do not have",
        ),
        pytest.param(
            "teams:\n  - name: API\n    path: a\n    instruction: x\n"
            "    isolation: container",
            "isolation",
            id="an isolation there is not",
        ),
    ],
)
def test_a_block_that_is_not_a_team_proposal_is_refused(block: str, problem: str):
    with pytest.raises(OrgStoreError, match=problem):
        proposed_teams(f"```teams\n{block}\n```")


def test_a_message_proposes_teams_only_once():
    block = "```teams\nteams:\n  - {name: A, path: a, instruction: x}\n```"

    with pytest.raises(OrgStoreError, match="2 ```teams blocks"):
        proposed_teams(f"{block}\nor\n{block}")


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
    b = make_node("b", "root", path="svc/b")
    narrow = b.boundary.model_copy(update={"writable": ["svc/b/src/**"]})
    store.upsert(b.model_copy(update={"boundary": narrow}))

    assert check(store, "b", [child("core", "svc/b/src/core")]).children
    with pytest.raises(OrgStoreError, match="inside what your team may write"):
        check(store, "b", [child("docs", "svc/b/docs")])


def test_a_new_team_beside_the_ones_a_node_has_replaces_none(store: OrgStore):
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
    a, a2 = store.get("a"), make_node("a2", "a", path="svc/a/two")
    assert a is not None
    store.upsert(a2.model_copy(update={"worktree_path": a.worktree_path}))

    with pytest.raises(OrgStoreError, match="work in its checkout"):
        check(store, "root", [child("services", "svc")])


def test_the_sub_teams_of_a_replaced_team_report_to_the_team_that_holds_them(
    store: OrgStore,
):
    a1, a2 = store.children("a")

    assert home_of(a1, "root", {"svc/a/one": "new.one", "docs": "docs"}) == "new.one"
    assert home_of(a1, "root", {"svc": "services"}) == "services"
    assert home_of(a2, "root", {"svc/a/one": "new.one"}) == "root"
