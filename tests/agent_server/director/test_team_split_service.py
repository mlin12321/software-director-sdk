"""What a split makes of a proposed sub-team. The service itself is
exercised through the director in `test_director_split.py` and
`test_director_former_teams.py`: it is given most of the director's
services."""

from openhands.agent_server.director.models import SplitChild
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.service.team_split_service import child_node
from tests.agent_server.director.conftest import (
    open_session_in,
    shipped_type,
    small_tree,
)


def parent_place():
    store = OrgStore(shipped_type)
    for node in small_tree():
        store.upsert(node)
    open_session_in(store, "root", "s1")
    place = store.place("a@s1")
    assert place is not None
    return place


def child(path: str) -> SplitChild:
    return SplitChild(title="Parser", path=path, task="Build the parser\nIn full.")


def test_a_sub_team_is_named_by_its_folder_and_placed_below_its_parent():
    parent = parent_place()

    made = child_node(set(), parent, child("svc/a/parser"))

    assert (made.id, made.team_id) == ("svc.a.parser@s1", "svc.a.parser")
    assert (made.parent_id, made.session) == (parent.id, "s1")
    assert made.summary == "Build the parser"
    assert made.agents == []


def test_a_sub_team_whose_name_is_taken_is_numbered():
    parent = parent_place()
    taken = {"svc.a.parser"}

    made = child_node(taken, parent, child("svc/a/parser"))

    assert made.team_id == "svc.a.parser-2"
    assert taken == {"svc.a.parser", "svc.a.parser-2"}
