"""What the director keeps of a team, as it is read back."""

from openhands.agent_server.director.models import MergeItem, OrgNodeInput


def test_a_team_saved_with_a_spec_summary_is_read_with_that_as_its_purpose():
    saved = {
        "id": "root",
        "parent_id": None,
        "title": "Checkout",
        "boundary": {"path": ""},
        "isolation": "shared",
        "spec_summary": "Take payments.",
    }

    assert OrgNodeInput.model_validate(saved).purpose == "Take payments."


def test_an_item_kept_with_a_documents_name_as_its_kind_names_the_document():
    kept = {"kind": "contracts", "key": "services/api", "gate": "manager"}

    item = MergeItem.model_validate(kept)

    assert (item.kind, item.document) == ("document", "contracts")
    assert item.waits_with_teams
