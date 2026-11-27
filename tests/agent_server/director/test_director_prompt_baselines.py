"""What the director tells each member of a shipped team of how the org
works, held against the texts kept in ``prompt_baselines``.

A change to these texts changes how every agent of an org behaves. To change
one on purpose, run with ``UPDATE_DIRECTOR_PROMPT_BASELINES=1`` and review
the difference in the kept files.
"""

import os
from pathlib import Path

import pytest

from tests.agent_server.director.conftest import (
    LAB,
    FakeLauncher,
    an_org,
    division_lead_of,
    node,
    started,
)


BASELINES = Path(__file__).parent / "prompt_baselines"
UPDATE = "UPDATE_DIRECTOR_PROMPT_BASELINES"


def held(name: str, text: str) -> None:
    """Hold ``text`` against the baseline ``name``."""
    kept = BASELINES / f"{name}.txt"
    if os.environ.get(UPDATE):
        kept.write_text(text)
    assert kept.exists(), f"No baseline {kept.name}: run with {UPDATE}=1"
    assert text == kept.read_text(), (
        f"{kept.name} differs from what the director says now; if that is "
        f"meant, run with {UPDATE}=1 and review the change"
    )


@pytest.mark.parametrize("member", ["ic", "manager"])
@pytest.mark.parametrize("lead", [False, True], ids=["team", "division-lead"])
async def test_what_a_member_of_a_shipped_team_is_told_of_the_org(
    repo: Path, launcher: FakeLauncher, member: str, lead: bool
):
    service = await started(repo, launcher)
    place = node(service, "root")
    if lead:
        above = service.org_load.org.place(division_lead_of(service))
        assert above is not None
        place = above

    try:
        held(
            f"software-engineer-{'division-lead-' if lead else ''}{member}",
            service.team_types.role_text(place, member),
        )
    finally:
        await service.aclose()


@pytest.mark.parametrize("member", ["bench", "referee", "chair"])
async def test_what_a_member_with_a_name_of_its_own_is_told_of_the_org(
    repo: Path, launcher: FakeLauncher, tmp_path: Path, member: str
):
    config = tmp_path / "config" / "director.yaml"
    (config.parent / "types").mkdir(parents=True)
    config.write_text("")
    (config.parent / "types" / "lab.yaml").write_text(LAB)
    service = await an_org(repo, launcher, config, agent_type="lab")

    try:
        held(
            f"lab-{member}", service.team_types.role_text(node(service, "root"), member)
        )
    finally:
        await service.aclose()
