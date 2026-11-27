"""What the director keeps of a team, as it is read back."""

from openhands.agent_server.director.models import (
    FormerTeam,
    MergeItem,
    OrgNodeInput,
)


def test_a_team_saved_with_a_spec_summary_is_read_with_that_as_its_purpose():
    saved = {
        "id": "root",
        "parent_id": None,
        "title": "Checkout",
        "boundary": {"path": ""},
        "isolation": "shared",
        "spec_summary": "Take payments.",
    }

    team = OrgNodeInput.model_validate(
        {**saved, "old_specs": [{"path": "services/api", "team": "API"}]}
    )

    assert team.purpose == "Take payments."
    assert [(left.name, left.path, left.team) for left in team.left_documents] == [
        ("spec", "services/api", "API")
    ]


def test_an_item_kept_with_a_documents_name_as_its_kind_names_the_document():
    kept = {"kind": "contracts", "key": "services/api", "gate": "manager"}

    item = MergeItem.model_validate(kept)

    assert (item.kind, item.document) == ("document", "contracts")
    assert item.waits_with_teams


def test_a_record_kept_with_a_spec_and_contracts_has_them_as_its_documents():
    kept = {
        "id": "services.api",
        "title": "API",
        "path": "services/api",
        "purpose": "Serve orders.",
        "ended": "pruned",
        "at": "2026-01-01T00:00:00+00:00",
        "contracts": "Order has an id.",
        "spec": "",
    }

    assert FormerTeam.model_validate(kept).documents == {
        "contracts": "Order has an id."
    }
