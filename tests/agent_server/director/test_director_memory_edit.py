"""Changing a spec or the contracts in one place, and adding to the
contracts what accepted work establishes."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ReviewTeamTool,
    SubmitReviewTool,
    WriteSpecTool,
)
from openhands.agent_server.director.tools.definition import (
    EditContractsAction,
    EditContractsTool,
    ReviewFinding,
    ReviewTeamAction,
    SubmitReviewAction,
    WriteSpecAction,
)
from tests.agent_server.director.conftest import FakeLauncher, commit
from tests.agent_server.director.test_director_agent_types import (
    an_org,
    org_config,
    split,
)
from tests.agent_server.director.test_director_tools import (
    division_lead_ic,
    ic,
    node,
    use,
)


@pytest.fixture
async def service(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
) -> AsyncIterator[DirectorService]:
    service = await an_org(repo, launcher, org_config(tmp_path))
    yield service
    await service.aclose()


async def test_accepting_work_can_add_to_the_contracts_every_sub_team_reads(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split(service)
    commit(repo, "tables/t1.md", "| warmup | 0.97 |\n")
    await use(
        service,
        ic(service, "tables"),
        SubmitReviewTool,
        SubmitReviewAction(summary="One table."),
    )
    finding = "Checked: 2k warmup gives 0.97 (Tables)."

    refused = await use(
        service,
        ic(service),
        ReviewTeamTool,
        ReviewTeamAction(
            team="tables",
            decision="reject",
            findings=[ReviewFinding(issue="No command given.")],
            contracts=finding,
        ),
    )
    assert refused.is_error and "leave it empty when you reject" in refused.text
    assert finding not in service.own_contracts(node(service, "root"))

    accepted = await use(
        service,
        ic(service),
        ReviewTeamTool,
        ReviewTeamAction(team="tables", decision="accept", contracts=finding),
    )

    assert not accepted.is_error
    contracts = service.own_contracts(node(service, "root"))
    assert "Tables reads what Harness writes" in contracts
    assert contracts.rstrip().endswith(finding)
    # The team beside it reads it as a contract it works to.
    beside = service.context_of(node(service, "harness"), "ic")
    assert finding in "\n".join(beside["contracts_with_the_teams_beside_you"])


async def test_the_spec_is_changed_in_one_place_as_a_file_editor_changes_a_file(
    service: DirectorService,
):
    root = node(service, "root")
    await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(text="# Lab\n\nBaseline: 0.98\nSeeds: 3\n"),
    )

    changed = await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(
            command="str_replace", old_str="Baseline: 0.98", new_str="Baseline: 0.97"
        ),
    )
    assert not changed.is_error
    [spec] = service.specs_of(root)
    assert spec.spec == "# Lab\n\nBaseline: 0.97\nSeeds: 3\n"

    twice = await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(command="str_replace", old_str=": ", new_str="="),
    )
    assert twice.is_error and "Multiple occurrences" in twice.text
    missing = await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(
            command="str_replace", old_str="Epochs: 9", new_str="Epochs: 10"
        ),
    )
    assert missing.is_error and "did not appear verbatim in the spec" in missing.text
    both = await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(
            command="str_replace", text="# New", old_str="Seeds: 3", new_str="Seeds: 5"
        ),
    )
    assert both.is_error
    whole = await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(command="create", old_str="Seeds: 3", new_str="Seeds: 5"),
    )
    assert whole.is_error
    assert service.specs_of(root)[0].spec == spec.spec

    # As the editor inserts: after a line, and at the end with none named.
    await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(command="insert", insert_line=1, new_str="Owner: Lab"),
    )
    await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(command="insert", new_str="Epochs: 9"),
    )
    assert service.specs_of(root)[0].spec == (
        "# Lab\nOwner: Lab\n\nBaseline: 0.97\nSeeds: 3\n\nEpochs: 9\n"
    )


async def test_a_team_without_a_spec_has_nothing_to_change_in_one_place(
    service: DirectorService,
):
    answer = await use(
        service,
        ic(service),
        WriteSpecTool,
        WriteSpecAction(command="str_replace", old_str="Baseline", new_str="Base"),
    )

    assert answer.is_error and "no spec to change yet" in answer.text


async def test_the_contracts_are_changed_by_the_ic_whose_sub_teams_keep_them(
    service: DirectorService, launcher: FakeLauncher
):
    await split(service)
    root = node(service, "root")

    await use(
        service,
        ic(service),
        EditContractsTool,
        EditContractsAction(
            command="str_replace", old_str="to `out/`", new_str="to `results/`"
        ),
    )
    await use(
        service,
        ic(service),
        EditContractsTool,
        EditContractsAction(
            command="insert", new_str="Unchecked: 4k warmup diverges (Tables)."
        ),
    )

    assert service.own_contracts(root) == (
        "Tables reads what Harness writes to `results/`.\n\n"
        "Unchecked: 4k warmup diverges (Tables).\n"
    )
    # A team without sub-teams sets no contracts, and the IC that only
    # reviews a session's work has none of its own.
    leaf = await use(
        service,
        ic(service, "harness"),
        EditContractsTool,
        EditContractsAction(command="insert", new_str="Mine now."),
    )
    assert leaf.is_error and "sets no contracts" in leaf.text
    assert division_lead_ic(service) is None
