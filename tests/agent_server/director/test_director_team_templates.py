"""A type names the members of its teams and the tools each has."""

from pathlib import Path

import pytest

from openhands.agent_server.director.agent_types import (
    SHIPPED_TYPES,
    AgentType,
    InvalidAgentType,
    TeamTemplate,
    _read,
    default_agents,
    load_types,
)
from openhands.agent_server.director.service.state import (
    DIRECTOR_TOOL_NAMES,
)


BOTH = ("read_team_context", "raise_escalation", "resolve_escalation", "report")

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
    like: bench
    tools: [read_team_context, review_team]
    lead_tools: [read_team_context, review_team]
  chair:
    like: bench
    tools: [read_team_context, approve_teams, decline_teams]
    lead_tools: [read_team_context, settle_teams]
"""


def lab(tmp_path: Path, text: str = LAB) -> AgentType:
    path = tmp_path / "lab.yaml"
    path.write_text(text)
    return _read(path, DIRECTOR_TOOL_NAMES)


def test_a_type_that_names_no_members_has_an_ic_that_works_and_a_manager():
    kind = AgentType(name="plain", description="A team.")

    assert list(kind.agents) == ["ic", "manager"]
    assert kind.writer == "ic" and kind.agents["manager"].like == "ic"
    # Whom the director turns to is found by tool, here as in any type.
    assert kind.member_with("submit_review") == "ic"
    assert kind.member_with("review_team") == "ic"
    assert kind.member_with("approve_teams") == "manager"
    assert kind.member_with("settle_teams", lead=True) == "manager"
    assert not kind.has("manager", "write_spec") and not kind.has("ic", "prune_team")


def test_the_shipped_types_keep_the_two_members_and_their_texts():
    types = load_types([SHIPPED_TYPES], DIRECTOR_TOOL_NAMES)

    assert set(types) >= {"software-engineer", "researcher", "research-engineer"}
    for kind in types.values():
        assert kind.agents.keys() == default_agents().keys()
        assert kind.agents["ic"].text == kind.ic
        assert kind.agents["manager"].text == kind.manager
        assert kind.writer == "ic"
        assert kind.member_with("review_team") == "ic"
        assert kind.member_with("approve_teams") == "manager"
        assert kind.member_with("settle_teams", lead=True) == "manager"


def test_a_type_names_its_own_members(tmp_path: Path):
    kind = lab(tmp_path)

    assert TeamTemplate is AgentType
    assert list(kind.agents) == ["bench", "referee", "chair"]
    assert kind.writer == "bench"
    assert kind.member_with("review_team") == "referee"
    assert kind.member_with("approve_teams") == "chair"
    assert kind.has("chair", "settle_teams") and not kind.has("bench", "review_team")
    assert "Runs the experiments." in kind.text_for("bench")
    assert kind.text_for("referee") == "" and kind.text_for("nobody") == ""


REFEREE = "    like: bench\n    tools: [read_team_context, review"
REFEREE_LEAD = "review_team]\n    lead_tools: [read_team_context, review_team]"
CHAIR = "[read_team_context, approve_teams,"
ONE_WRITER = "works in the team's folder"
ONE_REVIEWER = "`review_team` is held by one member"


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (("    writes: true\n", ""), ONE_WRITER),
        ((REFEREE, REFEREE.replace("like: bench", "writes: true")), ONE_WRITER),
        ((REFEREE_LEAD, "msg_team]"), ONE_REVIEWER),
        ((CHAIR, CHAIR.replace("approve", "review_team, approve")), ONE_REVIEWER),
        (("settle_teams", "report"), "`settle_teams` is held by one member"),
        (("referee:\n    like: bench", "referee:\n    like: dean"), "no other member"),
        (("write_spec,", "write_spec, no_such_tool,"), "no tool named no_such_tool"),
        (("A lab.", "A lab.\nic: Works."), "no member of that name"),
    ],
)
def test_a_type_the_director_could_not_run_is_refused(
    tmp_path: Path, change: tuple[str, str], said: str
):
    assert change[0] in LAB

    with pytest.raises(InvalidAgentType, match=said):
        lab(tmp_path, LAB.replace(*change, 1))
