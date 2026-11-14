"""What an agent is told of its team's context, and of how it changed.

An agent gets the whole context in its briefing. What changes afterwards is
not sent by itself, since every message starts a turn: it is attached to the
next message the agent gets anyway.
"""

import json
from typing import Any

from openhands.agent_server.director.models import AgentRole


UPDATE_HEADING = "[director] context update"

# What the briefing says in its own words; the rest is listed below them.
IN_THE_BRIEFING = (
    "boundary",
    "spec",
    "contracts_with_the_teams_beside_you",
    "contracts_between_your_sub_teams",
    "reports_of_your_team",
)
# Lists an agent adds to itself; what it added is not told back to it.
_OWN = {"escalations_you_raised": None, "reports_of_your_team": "by"}
_PASSING = ("working", "idle")


def _settled(status: str) -> str:
    return "at work" if status in _PASSING else status


def steady(context: dict[str, Any]) -> dict[str, Any]:
    """The context without what comes and goes with every turn: whether a
    team works or idles just now, and the one-line headlines."""
    return {
        **context,
        "status": _settled(context["status"]),
        "sub_teams": [
            {
                **{key: value for key, value in team.items() if key != "headline"},
                "status": _settled(team["status"]),
            }
            for team in context["sub_teams"]
        ],
    }


def changed(
    before: dict[str, Any], now: dict[str, Any], role: AgentRole
) -> dict[str, Any]:
    """The parts of a steady context that differ from what an agent was
    last told. A list that only grew gives what was added to it."""
    found: dict[str, Any] = {}
    for key, value in now.items():
        was = before.get(key)
        if value == was:
            continue
        if not (isinstance(value, list) and isinstance(was, list)):
            found[key] = value
            continue
        if value[: len(was)] != was:
            found[key] = value
            continue
        added = value[len(was) :]
        if key in _OWN:
            by = _OWN[key]
            added = [item for item in added if by is not None and item[by] != role]
        if added:
            found[key] = {"added": added}
    return found


def briefing_block(context: dict[str, Any]) -> str:
    """The rest of the context, for the end of a briefing."""
    rest = {key: value for key, value in context.items() if key not in IN_THE_BRIEFING}
    return (
        "The rest of your team's context. When any of the context changes, "
        f"a `{UPDATE_HEADING}` is added to the next message you get, so "
        "`read_node_context` is not needed in normal work; it returns all of "
        "it again if you have lost track.\n" + json.dumps(rest, indent=2)
    )


def update_block(found: dict[str, Any]) -> str:
    """What changed, for the end of a message the agent gets anyway."""
    return (
        f"{UPDATE_HEADING} (what changed for your team since you were last "
        "told; it asks nothing of you by itself):\n" + json.dumps(found, indent=2)
    )
