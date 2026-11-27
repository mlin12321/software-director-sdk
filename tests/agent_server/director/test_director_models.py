"""What the director keeps of a team, as it is read back."""

from openhands.agent_server.director.models import OrgNodeInput


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
