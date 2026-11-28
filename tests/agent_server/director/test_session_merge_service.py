"""What a session's work makes of the org, by itself: the stages of
settling it, over the services of ``test_review_service.py``."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openhands.agent_server.director.models import (
    DocumentChoice,
    MergeItem,
    TeamChoice,
)
from openhands.sdk.errors import Conflict, Invalid
from tests.agent_server.director.test_review_service import Reviews
from tests.agent_server.director.test_team_member_start_service import (
    ROOT,
)


@pytest.fixture
async def reviews(repo: Path) -> AsyncIterator[Reviews]:
    made = Reviews(repo)
    async with made.queue:
        yield made


TEAM = MergeItem(kind="team", key="billing", gate="manager", title="Billing")
NOTES = MergeItem(
    kind="document",
    key="svc/billing",
    document="contracts",
    gate="manager",
    session="The session's.",
    org="The org's.",
)


async def test_each_submission_comes_after_the_one_before(reviews: Reviews):
    first = reviews.merges.next_submission()

    assert reviews.merges.next_submission() == first + 1


async def test_nothing_is_settled_where_nothing_waits(reviews: Reviews):
    with pytest.raises(Conflict):
        reviews.merges.settle_teams(ROOT, [], "")


async def test_a_team_that_does_not_wait_cannot_be_settled(reviews: Reviews):
    reviews.merges._check_teams_wait([TEAM], [TeamChoice(team="billing", use="org")])

    with pytest.raises(Invalid, match="The teams that wait: billing"):
        reviews.merges._check_teams_wait([TEAM], [TeamChoice(team="orders", use="org")])


async def test_a_document_is_chosen_only_where_one_waits(reviews: Reviews):
    choice = DocumentChoice(document="contracts", folder="svc/billing/", use="org")

    chosen = reviews.merges._document_choices([TEAM, NOTES], [choice])

    assert chosen == {("contracts", "svc/billing"): choice}
    with pytest.raises(Invalid, match="svc/orders"):
        reviews.merges._document_choices(
            [NOTES],
            [DocumentChoice(document="contracts", folder="svc/orders", use="org")],
        )


async def test_the_sessions_holds_unless_the_manager_said_otherwise(
    reviews: Reviews,
):
    [team, notes] = reviews.merges._settled([TEAM, NOTES], {}, {})
    assert (team.use, notes.use) == ("session", "session")

    placed = {"billing": TeamChoice(team="billing", use="session", under="root")}
    replaced = DocumentChoice(
        document="contracts",
        folder="svc/billing",
        use="replacement",
        replacement="Both.",
    )
    [team, notes] = reviews.merges._settled(
        [TEAM, NOTES], placed, {("contracts", "svc/billing"): replaced}
    )

    assert (team.use, team.under) == ("session", "root")
    assert (notes.use, notes.replacement) == ("replacement", "Both.")
