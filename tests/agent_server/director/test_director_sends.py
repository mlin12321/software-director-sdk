"""What agents send each other, and what the kind of an org lets them."""

from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.director.documents.store import DocumentRef, TeamRef
from openhands.agent_server.director.documents.tree import TeamTree
from openhands.agent_server.director.models import DirectorSettingsUpdate
from openhands.agent_server.director.org import OrgBase, TeamMember
from openhands.agent_server.director.org_store import team_ref
from openhands.agent_server.director.orgs import ORG_KINDS
from openhands.agent_server.director.orgs.top_down import TopDownOrg
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools.remote import call_tool
from openhands.sdk.errors import Invalid, NotFound
from tests.agent_server.director.conftest import (
    API,
    FakeLauncher,
    an_org,
    ic,
    node,
    split_root,
    started,
)


WEB = "apps.web"


class _Line(TeamTree):
    """Teams one above the other, the first at the top."""

    def __init__(self, *teams: TeamRef) -> None:
        self.teams = list(teams)

    def above(self, team: TeamRef) -> TeamRef | None:
        at = self.teams.index(team)
        return self.teams[at - 1] if at else None

    def below(self, team: TeamRef) -> list[TeamRef]:
        at = self.teams.index(team)
        return self.teams[at + 1 : at + 2]


TOP = TeamRef("top@s", "s", "")
MID = TeamRef("mid@s", "s", "services")
LOW = TeamRef("low@s", "s", "services/api")
SPEC = DocumentRef("services", "spec")


def test_an_org_by_itself_refuses_nothing():
    org = OrgBase(_Line(TOP, MID, LOW))
    low, top = TeamMember(LOW, "ic"), TeamMember(TOP, "manager")

    assert org.allows_msg(low, top)
    assert org.allows_ref(low, top, SPEC)
    assert org.allows_read(TOP, LOW, "spec")


def test_a_top_down_org_passes_documents_down_only():
    org = TopDownOrg(_Line(TOP, MID, LOW))
    top, mid, low = (TeamMember(team, "ic") for team in (TOP, MID, LOW))

    assert org.allows_ref(mid, low, SPEC)
    assert org.allows_ref(top, low, SPEC)
    assert not org.allows_ref(mid, top, SPEC)
    assert not org.allows_ref(mid, mid, SPEC)
    assert org.allows_msg(low, top)


def _write_spec(service: DirectorService, team: str, text: str) -> DocumentRef:
    place = node(service, team)
    ref = team_ref(place).keeps("spec")
    service.documents.write(place.session, ref, text)
    return ref


async def test_a_document_passed_to_a_team_is_in_its_context_as_it_is_now(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    root, api = node(service, "root"), node(service, API)
    ref = _write_spec(service, "root", "# Checkout\nCards only.")

    service.team_messaging.send_ref(root.id, "ic", api.id, "ic", ref)
    service.team_messaging.send_ref(root.id, "ic", api.id, "ic", ref)

    [passed] = service.team_context.context_of(api, "ic")["documents_passed_to_you"]
    assert passed == {
        "document": "spec",
        "of": "the repository",
        "text": "# Checkout\nCards only.",
    }
    _write_spec(service, "root", "# Checkout\nCards and wallets.")
    [passed] = service.team_context.context_of(api, "ic")["documents_passed_to_you"]
    assert passed["text"].endswith("Cards and wallets.")
    assert service.team_context.context_of(root, "ic")["documents_passed_to_you"] == []
    await service.aclose()


async def test_the_shipped_orgs_pass_no_document_up_or_sideways(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    root, api, web = (node(service, name) for name in ("root", API, WEB))
    ref = _write_spec(service, API, "# API")

    with pytest.raises(Invalid, match="does not let"):
        service.team_messaging.send_ref(api.id, "ic", root.id, "ic", ref)
    with pytest.raises(Invalid, match="does not let"):
        service.team_messaging.send_ref(api.id, "ic", web.id, "ic", ref)
    with pytest.raises(NotFound, match="There is no contracts"):
        service.team_messaging.send_ref(
            root.id, "ic", api.id, "ic", team_ref(web).keeps("contracts")
        )

    assert service.team_context.context_of(root, "ic")["documents_passed_to_you"] == []
    await service.aclose()


async def test_an_open_org_passes_a_document_anywhere(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    root, api, web = (node(service, name) for name in ("root", API, WEB))
    ref = _write_spec(service, API, "# API")

    service.org_settings.update_settings(DirectorSettingsUpdate(org_kind="open"))
    service.team_messaging.send_ref(api.id, "ic", root.id, "ic", ref)
    service.team_messaging.send_ref(api.id, "ic", web.id, "ic", ref)

    assert [
        passed["of"]
        for passed in service.team_context.context_of(web, "ic")[
            "documents_passed_to_you"
        ]
    ] == ["services/api"]
    with pytest.raises(Invalid, match="no kind of org named 'flat'"):
        service.org_settings.update_settings(DirectorSettingsUpdate(org_kind="flat"))
    await service.aclose()


class _Silent(OrgBase):
    name = "silent"

    def allows_msg(self, sender: TeamMember, receiver: TeamMember) -> bool:
        return sender.team.id == receiver.team.id


async def test_the_kind_of_an_org_may_refuse_a_message(
    repo: Path, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    service = await started(repo, launcher)
    await split_root(service)
    root, api = node(service, "root"), node(service, API)
    monkeypatch.setitem(ORG_KINDS, "silent", _Silent)
    service.org_settings.update_settings(DirectorSettingsUpdate(org_kind="silent"))

    service.team_messaging.send_msg(root.id, "manager", root.id, "ic", "Go on.")
    with pytest.raises(Invalid, match="does not let"):
        service.team_messaging.send_msg(root.id, "ic", api.id, "ic", "Use cents.")
    await service.aclose()


SENDS = "            - msg_team\n            - send_document\n"


async def test_a_type_gives_the_tool_a_document_is_passed_with(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    from openhands.agent_server.director.orgs.software import SoftwareOrg

    assert SoftwareOrg.types_folder is not None
    shipped = (SoftwareOrg.types_folder / "software-engineer.yaml").read_text()
    config = tmp_path / "config" / "director.yaml"
    (config.parent / "types").mkdir(parents=True)
    config.write_text("")
    (config.parent / "types" / "sender.yaml").write_text(
        shipped.replace("name: software-engineer", "name: sender", 1).replace(
            "            - msg_team\n", SENDS, 1
        )
    )
    service = await an_org(repo, launcher, config, agent_type="sender")
    await split_root(service)
    _write_spec(service, "root", "# Lab")
    caller = ic(service)
    assert caller is not None

    sent = call_tool(
        service, UUID(caller), "send_document", {"team": API, "document": "spec"}
    )
    unknown = call_tool(
        service, UUID(caller), "send_document", {"team": API, "document": "minutes"}
    )

    assert not sent.is_error and "reads your team's spec" in sent.text
    assert unknown.is_error and "no document named 'minutes'" in unknown.text
    [passed] = service.team_context.context_of(node(service, API), "ic")[
        "documents_passed_to_you"
    ]
    assert passed["text"] == "# Lab"
    await service.aclose()
