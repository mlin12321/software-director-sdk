"""A type names the members of its teams and the tools each has."""

from pathlib import Path

import pytest

from openhands.agent_server.director.agent_types import (
    InvalidAgentType,
    TeamTemplate,
    load_types,
    read_type,
)
from openhands.agent_server.director.documents.declared import load_documents
from openhands.agent_server.director.orgs import SHIPPED_DOCUMENTS, SHIPPED_TYPES
from openhands.agent_server.director.service.state import (
    DIRECTOR_TOOL_NAMES,
)


DOCUMENTS = load_documents([SHIPPED_DOCUMENTS])


LAB = """\
description: A lab.
agents:
  bench:
    writes: true
    text: Runs the experiments.
    tools: [read_team_context, raise_escalation, report, write_spec,
            submit_review, propose_teams, msg_team]
    lead_tools: [read_team_context]
  referee:
    tools: [read_team_context, review_team]
    lead_tools: [read_team_context, review_team]
  chair:
    tools: [read_team_context, approve_teams, decline_teams]
    lead_tools: [read_team_context, settle_teams]
"""


def lab(tmp_path: Path, text: str = LAB) -> TeamTemplate:
    path = tmp_path / "lab.yaml"
    path.write_text(text)
    return read_type(path, DIRECTOR_TOOL_NAMES, DOCUMENTS)


def test_the_shipped_types_have_an_ic_that_works_and_a_manager():
    types = load_types(SHIPPED_TYPES, DIRECTOR_TOOL_NAMES, DOCUMENTS)

    assert set(types) >= {"software-engineer", "researcher", "research-engineer"}
    for kind in types.values():
        assert list(kind.agents) == ["ic", "manager"]
        assert kind.titles == {"ic": "IC", "manager": "Manager"}
        assert kind.agents["ic"].text and kind.agents["manager"].text
        assert kind.writer == "ic"
        assert kind.member_with("submit_review") == "ic"
        assert kind.member_with("review_team") == "ic"
        assert kind.member_with("approve_teams") == "manager"
        assert kind.member_with("settle_teams", lead=True) == "manager"
        assert not kind.has("manager", "write_spec")
        assert not kind.has("ic", "prune_team")


def test_a_type_names_its_own_members(tmp_path: Path):
    kind = lab(tmp_path)

    assert list(kind.agents) == ["bench", "referee", "chair"]
    assert kind.writer == "bench"
    assert kind.member_with("review_team") == "referee"
    assert kind.member_with("approve_teams") == "chair"
    assert kind.has("chair", "settle_teams") and not kind.has("bench", "review_team")
    assert "Runs the experiments." in kind.text_for("bench")
    assert kind.text_for("referee") == "" and kind.text_for("nobody") == ""
    # A member is called by its name unless its type gives it a title.
    assert kind.title_of("referee") == "referee"


def test_a_type_names_its_members_itself(tmp_path: Path):
    with pytest.raises(InvalidAgentType, match="agents"):
        lab(tmp_path, "description: A lab.\n")


REFEREE = "  referee:\n    tools: [read_team_context, review"
REFEREE_LEAD = "review_team]\n    lead_tools: [read_team_context, review_team]"
CHAIR = "[read_team_context, approve_teams,"
ONE_WRITER = "works in the team's folder"
ONE_REVIEWER = "`review_team` is held by one member"


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (("    writes: true\n", ""), ONE_WRITER),
        (
            (REFEREE, REFEREE.replace("referee:", "referee:\n    writes: true")),
            ONE_WRITER,
        ),
        ((REFEREE_LEAD, "msg_team]"), ONE_REVIEWER),
        ((CHAIR, CHAIR.replace("approve", "review_team, approve")), ONE_REVIEWER),
        (("settle_teams", "report"), "`settle_teams` is held by one member"),
        (("write_spec,", "write_spec, no_such_tool,"), "no tool named no_such_tool"),
        (("A lab.", "A lab.\nic: Works."), "ic"),
        (("  referee:\n", "  referee:\n    like: bench\n"), "like"),
    ],
)
def test_a_type_the_director_could_not_run_is_refused(
    tmp_path: Path, change: tuple[str, str], said: str
):
    assert change[0] in LAB

    with pytest.raises(InvalidAgentType, match=said):
        lab(tmp_path, LAB.replace(*change, 1))
