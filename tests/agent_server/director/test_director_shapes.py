"""Tests for how the sub-teams of a session and of the org are merged."""

import pytest

from openhands.agent_server.director.split import (
    Shape,
    ShapeConflict,
    Side,
    home_of_lost,
    merge_shapes,
    settle,
    shape_of,
)
from tests.agent_server.director.conftest import make_team


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
        pytest.param(
            [TOP],
            [TOP, JOBS],
            [TOP, ("nightly", "api", "services/api/jobs/nightly")],
            {},
            [ShapeConflict("jobs", JOBS[1:], None)],
            id="side-by-side-one-inside-the-other",
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


@pytest.mark.parametrize(
    ("base", "theirs", "now", "take", "conflicts"),
    [
        pytest.param({"a": 1}, {"a": 1}, {"a": 2}, {}, [], id="only-the-org-changed"),
        pytest.param({"a": 1}, {"a": 2}, {"a": 1}, {"a": 2}, [], id="only-the-session"),
        pytest.param({"a": 1}, {"a": 2}, {"a": 2}, {}, [], id="both-the-same"),
        pytest.param({"a": 1}, {"a": 2}, {"a": 3}, {}, ["a"], id="both-differently"),
        pytest.param({"a": 1}, {}, {"a": 1}, {"a": None}, [], id="session-removed"),
        pytest.param({"a": 1}, {}, {"a": 2}, {}, ["a"], id="removed-and-changed"),
        pytest.param({}, {"a": 1}, {}, {"a": 1}, [], id="session-added"),
        pytest.param({}, {"a": 1}, {"a": 2}, {}, ["a"], id="both-added-differently"),
    ],
)
def test_what_a_session_made_is_settled_against_what_the_org_has_now(
    base: dict[str, int],
    theirs: dict[str, int],
    now: dict[str, int],
    take: dict[str, int | None],
    conflicts: list[str],
):
    assert settle(base, theirs, now) == (take, conflicts)


def test_a_team_is_also_compared_by_its_name_its_globs_and_when_it_was_set_up():
    api = make_team("api", None, path="services/api")
    renamed = api.model_copy(update={"title": "Payments"})
    widened = api.model_copy(
        update={"boundary": api.boundary.model_copy(update={"readonly": ["docs/**"]})}
    )
    another = api.model_copy(update={"since": "2026-11-16T10:00:00+00:00"})
    base = shape_of([api])

    for changed, by_the_org in (
        (renamed, another),
        (widened, renamed),
        (another, renamed),
    ):
        theirs = shape_of([changed])
        # Changed by the session alone it is taken; by both, it waits.
        assert list(merge_shapes(base, theirs, base)[0]) == ["api"]
        _, conflicts = merge_shapes(base, theirs, shape_of([by_the_org]))
        assert [conflict.team for conflict in conflicts] == ["api"]
    # A headline or a summary a model wrote does not make another team of it.
    assert shape_of([api.model_copy(update={"headline": "Busy"})]) == base


def test_what_a_lost_team_left_goes_to_the_nearest_team_holding_its_directory():
    teams = [
        make_team("root", None, path=""),
        make_team("services", "root", path="services"),
        make_team("web", "root", path="apps/web"),
    ]

    assert home_of_lost("services/api/jobs", teams, "root") == "services"
    assert home_of_lost("apps/web", teams, "root") == "web"
    assert home_of_lost("tools", teams, "root") == "root"
    assert home_of_lost("tools", teams[1:], "services") == "services"
