"""An org whose types keep no documents works from a split to landed work,
and what a team reads follows where it stands in the tree."""

from pathlib import Path

from openhands.agent_server.director.git import git
from openhands.agent_server.director.models import OrgSettings, SplitChild
from openhands.agent_server.director.orgs.software import SoftwareOrg
from openhands.agent_server.director.service.state import team_ref
from tests.agent_server.director.conftest import (
    API,
    APPROVE,
    ApproveTeamsTool,
    FakeLauncher,
    ProposeTeamsTool,
    an_org,
    ic,
    landed,
    manager,
    node,
    proposing,
    split_root,
    started,
    tasked,
    use,
)


def _a_type_that_keeps_nothing() -> str:
    assert SoftwareOrg.types_folder is not None
    shipped = (SoftwareOrg.types_folder / "software-engineer.yaml").read_text()
    kept, _, rest = shipped.partition("documents:\n")
    actions = rest[rest.index("actions:\n") :]
    bare = kept.replace("name: software-engineer", "name: bare", 1) + actions
    for tool in ("report", "write_spec", "edit_contracts"):
        bare = bare.replace(f"            - {tool}\n", "")
    return bare


async def test_an_org_whose_teams_keep_no_documents_splits_and_lands_its_work(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    config = tmp_path / "config" / "director.yaml"
    (config.parent / "types").mkdir(parents=True)
    config.write_text("")
    (config.parent / "types" / "bare.yaml").write_text(_a_type_that_keeps_nothing())
    service = await an_org(repo, launcher, config, agent_type="bare")
    root = node(service, "root")
    assert service.documents_of(root) == []
    context = service.context_of(root, "ic")
    assert "spec" not in context and "reports_of_your_team" not in context

    service.update_org_settings(OrgSettings(default_agent_type="bare"))
    await split_root(service)
    api = node(service, API)
    assert service.type_of(api).name == "bare"
    assert service.documents_of(api) == []
    checkout = Path(root.worktree_path)
    (checkout / "README.md").write_text("# Checkout\n")
    git(checkout, ["add", "-A"])
    git(checkout, ["commit", "-m", "readme"])
    await landed(service)

    assert (repo / "README.md").read_text() == "# Checkout\n"
    review = node(service, "root").review
    assert review is not None and review.state == "accepted"
    assert service.drafts(root.session) == {}
    await service.aclose()


async def test_a_team_reads_the_contracts_of_whichever_team_is_above_it_now(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    await use(
        service,
        ic(service, API),
        ProposeTeamsTool,
        proposing(
            [SplitChild(title="Jobs", path="services/api/jobs", task="Run the jobs")],
            "Jobs take an order id.",
        ),
    )
    await use(service, manager(service, API), ApproveTeamsTool, APPROVE)
    await tasked(service)
    jobs = node(service, "services.api.jobs")

    def read_from_above() -> list[str]:
        place = node(service, jobs.id)
        return [
            shown.text.strip()
            for shown in service.documents_shown(place)
            if shown.name == "contracts" and shown.folder != team_ref(place).folder
        ]

    assert read_from_above() == ["Jobs take an order id."]

    # Under another team it reads that team's, with nothing sent or stored.
    service.org.move_place(jobs.id, node(service, "root").id)

    assert read_from_above() == ["POST /checkout returns {order_id}"]
    await service.aclose()
