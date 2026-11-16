"""Tests for how the sub-teams of a session and of the org are merged."""

import pytest

from openhands.agent_server.director.split import (
    Shape,
    ShapeConflict,
    Side,
    merge_shapes,
)


TOP = ("api", "", "services/api")
JOBS = ("jobs", "api", "services/api/jobs")
CACHE = ("cache", "api", "services/api/cache")


@pytest.mark.parametrize(
    ("base", "theirs", "now", "take", "conflicts"),
    [
        pytest.param([TOP], [TOP], [TOP, CACHE], {}, [], id="org-added"),
        pytest.param([TOP, JOBS], [TOP, JOBS], [TOP], {}, [], id="org-removed"),
        pytest.param(
            [TOP], [TOP, JOBS], [TOP], {"jobs": JOBS[1:]}, [], id="session-added"
        ),
        pytest.param(
            [TOP, JOBS], [TOP], [TOP, JOBS], {"jobs": None}, [], id="session-removed"
        ),
        pytest.param(
            [TOP],
            [TOP, JOBS],
            [TOP, CACHE],
            {"jobs": JOBS[1:]},
            [],
            id="each-added-one",
        ),
        pytest.param([TOP], [TOP, JOBS], [TOP, JOBS], {}, [], id="both-the-same"),
        pytest.param(
            [TOP, JOBS],
            [TOP, ("jobs", "api", "services/api/queue")],
            [TOP, ("jobs", "api", "services/api/workers")],
            {},
            [
                ShapeConflict(
                    "jobs",
                    ("api", "services/api/queue"),
                    ("api", "services/api/workers"),
                )
            ],
            id="both-changed-one",
        ),
        pytest.param(
            [TOP, JOBS],
            [TOP],
            [TOP, ("jobs", "api", "services/api/workers")],
            {},
            [ShapeConflict("jobs", None, ("api", "services/api/workers"))],
            id="removed-and-changed",
        ),
        pytest.param(
            [TOP],
            [TOP, JOBS],
            [],
            {},
            [ShapeConflict("jobs", JOBS[1:], None)],
            id="under-a-team-the-org-lost",
        ),
        pytest.param([TOP], [TOP], [], {}, [], id="org-lost-the-team"),
        pytest.param(
            [TOP, JOBS],
            [TOP],
            [TOP, JOBS, ("nightly", "jobs", "services/api/jobs/nightly")],
            {},
            [ShapeConflict("jobs", None, JOBS[1:])],
            id="removed-above-a-team-the-org-added",
        ),
        pytest.param(
            [TOP],
            [TOP, JOBS],
            [TOP, ("workers", "api", "services/api/jobs")],
            {},
            [ShapeConflict("jobs", JOBS[1:], None)],
            id="one-directory-two-teams",
        ),
    ],
)
def test_sub_teams_are_merged_team_by_team(
    base: Shape,
    theirs: Shape,
    now: Shape,
    take: dict[str, Side],
    conflicts: list[ShapeConflict],
):
    assert merge_shapes(base, theirs, now) == (take, conflicts)


def test_what_hangs_on_a_conflict_waits_with_it():
    below = ("nightly", "jobs", "services/api/jobs/nightly")

    take, conflicts = merge_shapes([TOP], [TOP, JOBS, below], [])

    assert take == {}
    assert [conflict.team for conflict in conflicts] == ["jobs", "nightly"]
