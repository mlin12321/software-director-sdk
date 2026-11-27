"""An org whose types keep no documents works from a split to landed work,
what a team reads follows where it stands in the tree, and a document of an
org's own is left behind or settled as its protocol says, by no name the
director knows."""

from pathlib import Path
from typing import ClassVar

from openhands.agent_server.director.documents.protocol import DocumentProtocolBase
from openhands.agent_server.director.documents.registry import register_protocol
from openhands.agent_server.director.environment.git.cli import git
from openhands.agent_server.director.models import (
    DocumentChoice,
    LeftoverDocument,
    OpenSessionRequest,
    OrgSettings,
    SplitChild,
)
from openhands.agent_server.director.org_store import team_ref
from openhands.agent_server.director.orgs.software import SoftwareOrg
from openhands.agent_server.director.prompts import KeptDocuments, session_brief
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import PruneTeamTool, SettleTeamsTool
from tests.agent_server.director.conftest import (
    API,
    APPROVE,
    PRUNE_API,
    SPLIT,
    SPLIT_CONTRACTS,
    ApproveTeamsTool,
    FakeLauncher,
    ProposeTeamsTool,
    an_org,
    division_lead_manager,
    division_lead_of,
    ic,
    landed,
    manager,
    node,
    proposing,
    session_accepted,
    split_root,
    started,
    tasked,
    use,
    write_contracts,
)
from tests.agent_server.director.document_tools import SettleTeamsAction


def _a_type_that_keeps_nothing() -> str:
    assert SoftwareOrg.types_folder is not None
    shipped = (SoftwareOrg.types_folder / "software-engineer.yaml").read_text()
    kept, _, rest = shipped.partition("documents:\n")
    actions = rest[rest.index("actions:\n") :]
    bare = kept.replace("name: software-engineer", "name: bare", 1) + actions
    for tool in ("report", "write_spec", "write_contracts"):
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
    assert service.team_documents.listed(service.team_types.reader_of(root)) == []
    context = service.team_context.context_of(root, "ic")
    assert "spec" not in context and "reports_of_your_team" not in context

    service.org_settings.update_org_settings(OrgSettings(default_agent_type="bare"))
    # A team that keeps no contracts has no tool to write any, and the
    # tools that set up its sub-teams ask for none.
    refused = await write_contracts(service, SPLIT_CONTRACTS)
    assert refused.is_error
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)
    api = node(service, API)
    assert service.team_types.type_of(api).name == "bare"
    assert service.team_documents.listed(service.team_types.reader_of(api)) == []
    checkout = Path(root.worktree_path)
    (checkout / "README.md").write_text("# Checkout\n")
    git(checkout, ["add", "-A"])
    git(checkout, ["commit", "-m", "readme"])
    await landed(service)

    assert (repo / "README.md").read_text() == "# Checkout\n"
    review = node(service, "root").review
    assert review is not None and review.state == "accepted"
    assert service.session_documents.drafts(root.session) == {}
    await service.aclose()


async def test_a_team_reads_the_contracts_of_whichever_team_is_above_it_now(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    await write_contracts(service, "Jobs take an order id.", API)
    await use(
        service,
        ic(service, API),
        ProposeTeamsTool,
        proposing(
            [SplitChild(title="Jobs", path="services/api/jobs", task="Run the jobs")],
        ),
    )
    await use(service, manager(service, API), ApproveTeamsTool, APPROVE)
    await tasked(service)
    jobs = node(service, "services.api.jobs")

    def read_from_above() -> list[str]:
        place = node(service, jobs.id)
        return [
            shown.text.strip()
            for shown in service.team_documents.shown(
                service.team_types.reader_of(place)
            )
            if shown.name == "contracts" and shown.folder != team_ref(place).folder
        ]

    assert read_from_above() == ["Jobs take an order id."]

    # Under another team it reads that team's, with nothing sent or stored.
    service.org.move_place(jobs.id, node(service, "root").id)

    assert read_from_above() == ["POST /checkout returns {order_id}"]
    await service.aclose()


class _Runbook(DocumentProtocolBase):
    """A document of an org's own that a removed team leaves behind."""

    stays_when_team_is_removed: ClassVar[bool] = True


class _Ledger(DocumentProtocolBase):
    """A document of an org's own that is settled with the teams."""

    waits_for_team_changes: ClassVar[bool] = True


register_protocol("left-behind", _Runbook)
register_protocol("settled-with-teams", _Ledger)


async def _an_org_with_documents_of_its_own(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
) -> DirectorService:
    assert SoftwareOrg.types_folder is not None
    shipped = (SoftwareOrg.types_folder / "software-engineer.yaml").read_text()
    ops = shipped.replace("name: software-engineer", "name: ops", 1).replace(
        "documents:\n", "documents:\n    runbook:\n    ledger:\n", 1
    )
    assert ops.count("    runbook:\n") == 1
    config = tmp_path / "config" / "director.yaml"
    (config.parent / "types").mkdir(parents=True)
    (config.parent / "documents").mkdir()
    config.write_text("")
    (config.parent / "types" / "ops.yaml").write_text(ops)
    (config.parent / "documents" / "runbook.yaml").write_text(
        "description: How the team runs what it built.\nprotocol: left-behind\n"
    )
    (config.parent / "documents" / "ledger.yaml").write_text(
        "description: What the sub-teams owe each other.\n"
        "protocol: settled-with-teams\n"
    )
    service = await an_org(repo, launcher, config, agent_type="ops")
    service.org_settings.update_org_settings(OrgSettings(default_agent_type="ops"))
    return service


async def test_a_document_of_an_orgs_own_stays_when_its_team_is_removed(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    service = await _an_org_with_documents_of_its_own(repo, launcher, tmp_path)
    await split_root(service)
    api = node(service, API)
    service.documents.write(
        api.session, team_ref(api).keeps("runbook"), "Restart the queue first.\n"
    )

    await use(service, manager(service), PruneTeamTool, PRUNE_API)
    await service.settled()

    # The director knows no document by name: the runbook stays because
    # its protocol says so, as the spec does.
    root = node(service, "root")
    assert root.leftover_documents == [
        LeftoverDocument(name="runbook", path="services/api", team="API")
    ]
    [left] = service.team_documents.left_to(service.team_types.reader_of(root))
    assert (left.name, left.text) == ("runbook", "Restart the queue first.\n")
    context = service.team_context.context_of(root, "ic")
    assert context["documents_left_by_teams_before_you"] == [
        {
            "document": "runbook",
            "directory": "services/api",
            "team": "API",
            "text": "Restart the queue first.\n",
        }
    ]
    assert service.team_types.kept_documents(root).stay_when_removed == (
        "runbook",
        "spec",
    )
    await service.aclose()


async def test_a_document_of_an_orgs_own_is_settled_with_the_teams(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    service = await _an_org_with_documents_of_its_own(repo, launcher, tmp_path)
    await split_root(service)
    await landed(service)
    opened = await service.session_lifecycle.open_session(
        OpenSessionRequest(text="Add refunds.", node_id="services.api")
    )
    await service.settled()
    top, conversation = opened.session.teams[0].id, opened.conversation_id
    place = node(service, top)
    service.documents.write(
        place.session, team_ref(place).keeps("ledger"), "Refunds owe Jobs a retry."
    )
    # Other accepted work wrote the ledger of the same folder meanwhile.
    service.team_memory.write("services/api", "ledger.md", "Orders owe Jobs.\n")

    await session_accepted(service, top, conversation)

    # The code is merged, and the ledger waits for the Manager above as
    # contracts do.
    assert service.team_memory.read("services/api/ledger.md") == "Orders owe Jobs.\n"
    merge = node(service, division_lead_of(service, top)).merge
    assert merge is not None
    assert [(item.document, item.key, item.gate) for item in merge.items] == [
        ("ledger", "services/api", "manager")
    ]
    asked = division_lead_manager(service, top)
    assert asked is not None
    assert "the ledger of services/api" in launcher.messages_to(asked)[-1]

    settled = await use(
        service,
        asked,
        SettleTeamsTool,
        SettleTeamsAction(
            merge_choices=[
                DocumentChoice(
                    document="ledger",
                    folder="services/api",
                    use="replacement",
                    replacement="Orders and refunds owe Jobs.",
                )
            ]
        ),
    )

    assert not settled.is_error
    assert service.team_memory.read("services/api/ledger.md") == (
        "Orders and refunds owe Jobs.\n"
    )
    told = launcher.messages_to(conversation)[-1]
    assert "What holds as the ledger of services/api is not what you wrote" in told
    await service.aclose()


async def test_a_team_that_keeps_no_record_is_not_told_to_report(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    config = tmp_path / "config" / "director.yaml"
    (config.parent / "types").mkdir(parents=True)
    config.write_text("")
    (config.parent / "types" / "bare.yaml").write_text(_a_type_that_keeps_nothing())
    bare = await an_org(repo, launcher, config, agent_type="bare")
    root = node(bare, "root")
    names = bare.team_types.team(root).names

    assert bare.team_types.kept_documents(root) == KeptDocuments()
    brief = session_brief(root, "Refunds", [], names=names, kept=KeptDocuments())
    assert "`report`" not in brief and "spec" not in brief
    assert "You start from the team's records." in brief
    await bare.aclose()

    # A team that keeps one is told to.
    kept = KeptDocuments(names=["spec", "contracts"], reports=True)
    assert "`report` what they must know" in session_brief(
        root, "Refunds", [], names=names, kept=kept
    )
