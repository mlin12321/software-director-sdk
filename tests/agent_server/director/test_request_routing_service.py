"""Handing back a request, with an org and collaborators made by hand and
no service. Choosing a team is tested through the service, with a real
tree (``test_director_routing.py``)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from openhands.agent_server.director.environment.git import (
    GitCheckouts,
)
from openhands.agent_server.director.models import RoutedRequest, SessionTeam
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.service.model_call_service import (
    ModelCallService,
)
from openhands.agent_server.director.service.request_routing_service import (
    RequestRoutingService,
)
from openhands.sdk.errors import Invalid


ABOVE = SimpleNamespace(id="root", parent_id=None, title="Checkout")
TEAM = cast(SessionTeam, SimpleNamespace(id="api", parent_id="root", title="API"))


def request(kind: str, team_id: str = "api") -> RoutedRequest:
    return RoutedRequest.model_validate(
        {
            "at": "2026-01-01T00:00:00Z",
            "kind": kind,
            "team_id": team_id,
            "team": "API",
            "text": "Add a currency field",
            "chosen_by": "model",
        }
    )


def routing(
    requests: list[RoutedRequest], kept: list[tuple[Any, ...]], handed: list[Any]
) -> RequestRoutingService:
    places = {"root": ABOVE}

    def hand_on(place: Any, above: Any, routed: RoutedRequest, why: str) -> str:
        handed.append((place.id, above.id, routed.text, why))
        return "Handed on."

    return RequestRoutingService(
        checkouts=GitCheckouts(),
        org=lambda: cast(OrgStore, SimpleNamespace(place=places.get)),
        model_calls=cast(ModelCallService, None),
        checkout=lambda: Path("."),
        purpose=lambda node: "",
        routed_requests=lambda: requests,
        keep_routed=lambda *said: kept.append(said),
        hand_on=hand_on,
        paused=lambda: False,
    )


def test_a_request_a_team_says_is_not_its_own_is_kept_and_handed_on():
    kept: list[tuple[Any, ...]] = []
    handed: list[Any] = []

    told = routing([request("asked")], kept, handed).not_ours(TEAM, "It is billing's")

    assert told == "Handed on."
    assert kept == [
        ("misrouted", TEAM, "Add a currency field", "model", "It is billing's")
    ]
    assert handed == [("api", "root", "Add a currency field", "It is billing's")]


@pytest.mark.parametrize(
    "requests",
    [[], [request("asked", "web")], [request("asked"), request("misrouted")]],
    ids=["nothing sent", "sent to another team", "handed on already"],
)
def test_nothing_is_handed_on_that_is_not_with_the_team(
    requests: list[RoutedRequest],
):
    kept: list[tuple[Any, ...]] = []
    handed: list[Any] = []

    with pytest.raises(Invalid):
        routing(requests, kept, handed).not_ours(TEAM, "Not ours")

    assert not kept and not handed
