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
    "specs_left_by_teams_before_you",
    "contracts_with_the_teams_beside_you",
    "contracts_between_your_sub_teams",
    "reports_of_your_team",
)
# Lists an agent adds to itself; what it added is not told back to it.
_OWN = {"escalations_you_raised": None, "reports_of_your_team": "by"}
# Lists with one entry for each team, told team by team.
_BY_TEAM = (
    "sub_teams",
    "other_sessions_of_your_team",
    "sessions_whose_work_you_review",
)
# Lists of which only the latest are kept: what is new in one is told, and
# what dropped out of it is not.
_LATEST = ("asked_directly_of_teams_below",)
_PASSING = ("working", "idle")


def _settled(status: str) -> str:
    return "at work" if status in _PASSING else status


def steady(context: dict[str, Any]) -> dict[str, Any]:
    """The context without what comes and goes with every turn: whether a
    team works or idles just now, and the one-line headlines."""
    return {
        **context,
        "status": _settled(context["status"]),
        **{
            key: [
                {
                    **{
                        name: value
                        for name, value in team.items()
                        if name != "headline"
                    },
                    "status": _settled(team["status"]),
                }
                for team in context[key]
            ]
            for key in _BY_TEAM
            if key in context
        },
    }


def changed(
    before: dict[str, Any], now: dict[str, Any], role: AgentRole
) -> dict[str, Any]:
    """The parts of a steady context that differ from what an agent was
    last told. A list that only grew gives what was added to it, and the
    list of sub-teams gives the teams that changed or went."""
    found: dict[str, Any] = {}
    for key, value in now.items():
        was = before.get(key)
        if value == was:
            continue
        if not (isinstance(value, list) and isinstance(was, list)):
            found[key] = value
            continue
        if key in _BY_TEAM and was:
            found[key] = _teams_changed(was, value)
            continue
        if key in _LATEST:
            new = [item for item in value if item not in was]
            if new:
                found[key] = {"added": new}
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


def _teams_changed(
    was: list[dict[str, Any]], now: list[dict[str, Any]]
) -> dict[str, Any]:
    known = {team["node_id"]: team for team in was}
    left = {team["node_id"] for team in now}
    found: dict[str, Any] = {
        "changed": [team for team in now if known.get(team["node_id"]) != team]
    }
    removed = [node_id for node_id in known if node_id not in left]
    if removed:
        found["removed"] = removed
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
