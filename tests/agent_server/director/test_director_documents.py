"""The documents teams keep: where they are kept and what one is said to be."""

from pathlib import Path

import pytest

from openhands.agent_server.director.agent_types import InvalidAgentType, read_type
from openhands.agent_server.director.documents.declared import (
    InvalidDocument,
    load_documents,
    read_document,
)
from openhands.agent_server.director.documents.events import (
    SubTeamsProposed,
    TeamRef,
    TeamRemoved,
)
from openhands.agent_server.director.documents.protocol import DocumentProtocolBase
from openhands.agent_server.director.documents.registry import (
    UnknownProtocol,
    protocol_of,
)
from openhands.agent_server.director.documents.schema import (
    DocumentsBriefed,
    ShownDocument,
)
from openhands.agent_server.director.documents.store import (
    DocumentDraft,
    DocumentRef,
    GitDocumentStore,
)
from openhands.agent_server.director.documents.writer import (
    DocumentWriterBase,
    EditorWriter,
)
from openhands.agent_server.director.git_store import GitStore
from openhands.agent_server.director.org import OrgBase
from openhands.agent_server.director.orgs import (
    SHIPPED_DOCUMENTS,
    SHIPPED_TYPES,
    ResearchOrg,
    SoftwareOrg,
)
from openhands.agent_server.director.orgs.documents import ContractsProtocol
from openhands.agent_server.director.prompts import org_protocol
from openhands.agent_server.director.run_state import RunState
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import DIRECTOR_TOOL_NAMES
from openhands.sdk.errors import Invalid
from tests.agent_server.director.conftest import DEFAULT_TYPE, SHIPPED


NOTES = DocumentRef("services/api", "notes")


class FakeTree:
    """Teams by the id of the team above each."""

    def __init__(self, above: dict[str, str | None]) -> None:
        self._above = above

    def ref(self, team: str) -> TeamRef:
        return TeamRef(id=team, session="one", folder=team.replace(".", "/"))

    def above(self, team: TeamRef) -> TeamRef | None:
        above = self._above.get(team.id)
        return self.ref(above) if above else None

    def below(self, team: TeamRef) -> list[TeamRef]:
        return [
            self.ref(name) for name, above in self._above.items() if above == team.id
        ]


TREE = FakeTree(
    {
        "services": None,
        "services.api": "services",
        "services.web": "services",
        "services.api.auth": "services.api",
    }
)


@pytest.fixture
def store(repo: Path) -> GitDocumentStore:
    runs = RunState(None)
    return GitDocumentStore(lambda: GitStore(repo), lambda: runs)


def test_a_document_is_kept_in_its_teams_folder_under_its_name():
    assert NOTES.path == "services/api/notes.md"
    assert DocumentRef("", "notes").path == "notes.md"
    assert DocumentRef.at("services/api/notes.md") == NOTES
    assert DocumentRef.at("notes.md") == DocumentRef("", "notes")


def test_what_a_session_writes_is_its_own_until_it_is_taken(
    store: GitDocumentStore, repo: Path
):
    GitStore(repo).write("services/api", "notes.md", "The org's.")

    store.write("one", NOTES, "Session one's.")

    assert store.read("one", NOTES) == "Session one's."
    assert store.read("two", NOTES) == "The org's."
    assert store.read("", NOTES) == "The org's."
    assert store.drafts("one") == {NOTES: DocumentDraft("Session one's.", "The org's.")}
    assert store.drafts("two") == {}


def test_a_session_keeps_what_it_first_wrote_over(store: GitDocumentStore, repo: Path):
    GitStore(repo).write("services/api", "notes.md", "First.")
    store.write("one", NOTES, "Mine.")
    GitStore(repo).write("services/api", "notes.md", "The org moved on.")

    store.write("one", NOTES, "Mine, again.")

    assert store.drafts("one")[NOTES] == DocumentDraft("Mine, again.", "First.")


def test_a_session_may_remove_a_document_or_forget_what_it_made_of_it(
    store: GitDocumentStore, repo: Path
):
    GitStore(repo).write("services/api", "notes.md", "The org's.")

    store.remove("one", NOTES)

    assert store.read("one", NOTES) is None
    assert store.drafts("one")[NOTES] == DocumentDraft(None, "The org's.")

    store.discard("one", NOTES)

    assert store.read("one", NOTES) == "The org's."
    assert store.drafts("one") == {}


def test_the_shipped_documents_are_the_ones_the_shipped_types_keep():
    documents = load_documents([SHIPPED_DOCUMENTS])

    assert set(documents) == {"spec", "contracts", "report"}
    assert documents["spec"].written_with == "write_spec"
    assert documents["contracts"].written_with == "write_contracts"
    for folder in SHIPPED_TYPES:
        for path in folder.glob("*.yaml"):
            kind = read_type(path, DIRECTOR_TOOL_NAMES, documents)
            assert set(kind.documents) == set(documents)
            assert all(kind.documents.values())


def test_a_document_needs_no_more_than_a_name_and_what_it_holds(tmp_path: Path):
    path = tmp_path / "decision-log.yaml"
    path.write_text("description: What was decided and why.\n")

    document = read_document(path)

    assert document.name == "decision-log"
    assert document.written_with == "write_decision_log"
    assert document.protocol is None


def test_a_file_that_is_no_document_is_left_out(tmp_path: Path):
    (tmp_path / "notes.yaml").write_text("description: Notes.\n")
    (tmp_path / "empty.yaml").write_text("name: empty\n")
    (tmp_path / "other.yaml").write_text("name: another\ndescription: Something.\n")
    (tmp_path / "flagged.yaml").write_text("description: Something.\nappend: true\n")

    with pytest.raises(InvalidDocument, match="name the file after it"):
        read_document(tmp_path / "other.yaml")

    assert set(load_documents([tmp_path])) == {"notes"}


def test_a_type_says_what_a_document_holds_for_it_or_takes_the_documents_own(
    tmp_path: Path,
):
    (tmp_path / "notes.yaml").write_text("description: Notes of the team.\n")
    (tmp_path / "plan.yaml").write_text("description: The plan.\n")
    documents = load_documents([tmp_path])
    path = tmp_path / "planner.yaml"
    path.write_text(
        "description: Plans.\n"
        "agents:\n"
        "    ic:\n"
        "        writes: true\n"
        "        tools: [submit_review, review_team, propose_teams, approve_teams]\n"
        "        lead_tools: [settle_teams]\n"
        "documents:\n"
        "    notes:\n"
        "    plan: The steps, in order.\n"
    )

    kind = read_type(path, DIRECTOR_TOOL_NAMES, documents)

    assert kind.documents == {
        "notes": "Notes of the team.",
        "plan": "The steps, in order.",
    }
    assert "- notes: Notes of the team." in kind.text_for("ic")


def test_a_type_cannot_keep_a_document_there_is_none_of(tmp_path: Path):
    path = tmp_path / "planner.yaml"
    path.write_text(
        "description: Plans.\n"
        "agents:\n"
        "    ic:\n"
        "        writes: true\n"
        "        tools: [submit_review, review_team, propose_teams, approve_teams]\n"
        "        lead_tools: [settle_teams]\n"
        "documents:\n"
        "    roadmap:\n"
    )

    with pytest.raises(InvalidAgentType, match="no document named roadmap"):
        read_type(path, DIRECTOR_TOOL_NAMES, {})


def test_a_document_without_a_protocol_is_a_plain_text(
    store: GitDocumentStore, tmp_path: Path
):
    (tmp_path / "notes.yaml").write_text("description: Notes.\n")

    protocol = protocol_of(read_document(tmp_path / "notes.yaml"), store, TREE)

    assert type(protocol) is DocumentProtocolBase
    assert not protocol.waits_for_team_changes


def test_a_document_does_what_the_protocol_it_names_does(
    store: GitDocumentStore, tmp_path: Path
):
    # Any document may name the protocol: it is not the name `contracts`
    # that makes a document wait for the teams.
    (tmp_path / "glossary.yaml").write_text(
        "description: Terms.\nprotocol: contracts\n"
    )
    (tmp_path / "plan.yaml").write_text("description: The plan.\nprotocol: nobodys\n")

    protocol = protocol_of(read_document(tmp_path / "glossary.yaml"), store, TREE)

    assert isinstance(protocol, ContractsProtocol)
    assert protocol.waits_for_team_changes
    with pytest.raises(UnknownProtocol, match="nobodys"):
        protocol_of(read_document(tmp_path / "plan.yaml"), store, TREE)


def test_the_director_asks_a_files_document_what_it_does(repo: Path):
    service = DirectorService(repo)

    assert set(service.protocols) == {"spec", "contracts", "report"}
    contracts = service.protocol_at("services/api/contracts.md")
    assert contracts is not None and contracts.waits_for_team_changes
    spec = service.protocol_at("spec.md")
    assert spec is not None and not spec.waits_for_team_changes
    assert service.protocol_at("services/api/README.md") is None


TEAM = TeamRef(id="services.api@one", session="one", folder="services/api")
CONTRACTS = DocumentRef("services/api", "contracts")


@pytest.fixture
def contracts(store: GitDocumentStore) -> ContractsProtocol:
    document = load_documents([SHIPPED_DOCUMENTS])["contracts"]
    protocol = protocol_of(document, store, TREE)
    assert isinstance(protocol, ContractsProtocol)
    return protocol


def test_contracts_must_be_written_before_sub_teams_are_proposed(
    contracts: ContractsProtocol, store: GitDocumentStore
):
    with pytest.raises(Invalid, match="write_contracts"):
        contracts.before_sub_teams_are_proposed(SubTeamsProposed(TEAM))

    store.write("one", CONTRACTS, "Cart owns the cart page.\n")

    contracts.before_sub_teams_are_proposed(SubTeamsProposed(TEAM))
    # A plain document refuses nothing.
    plain = DocumentProtocolBase(contracts.document, store, TREE)
    plain.before_sub_teams_are_proposed(SubTeamsProposed(TEAM))


def test_a_removed_teams_contracts_go_into_its_record_and_out_of_its_folder(
    contracts: ContractsProtocol, store: GitDocumentStore, repo: Path
):
    GitStore(repo).write("services/api", "contracts.md", "Cart owns the cart page.\n")

    kept = contracts.when_team_is_removed(TeamRemoved(TEAM))

    assert kept == "Cart owns the cart page.\n"
    assert store.read("one", CONTRACTS) is None
    assert store.read("", CONTRACTS) == "Cart owns the cart page.\n"
    # A plain document keeps nothing of itself and stays where it is.
    plain = DocumentProtocolBase(contracts.document, store, TREE)
    assert plain.when_team_is_removed(TeamRemoved(TEAM)) is None


def test_the_store_tells_of_every_change_a_session_makes(repo: Path):
    told: list[str] = []
    runs = RunState(None)
    store = GitDocumentStore(lambda: GitStore(repo), lambda: runs, told.append)

    store.write("one", NOTES, "Mine.")
    store.remove("one", NOTES)
    store.discard("one", NOTES)

    assert told == ["one", "one"]


def test_a_plain_document_is_read_by_the_team_that_keeps_it_and_no_other(
    store: GitDocumentStore, tmp_path: Path
):
    (tmp_path / "notes.yaml").write_text("description: Notes.\n")
    plain = protocol_of(read_document(tmp_path / "notes.yaml"), store, TREE)

    assert plain.read_from(TREE.ref("services.api")) == [TREE.ref("services.api")]


def test_a_team_reads_the_contracts_of_the_team_above_and_its_own(
    contracts: ContractsProtocol,
):
    assert contracts.read_from(TREE.ref("services.api")) == [
        TREE.ref("services"),
        TREE.ref("services.api"),
    ]
    # Not those two teams up, nor those of the team beside it.
    assert contracts.read_from(TREE.ref("services.api.auth")) == [
        TREE.ref("services.api"),
        TREE.ref("services.api.auth"),
    ]
    assert contracts.read_from(TREE.ref("services")) == [TREE.ref("services")]


def test_an_org_refuses_nothing_unless_its_kind_says_so():
    class AnyOrg(OrgBase):
        pass

    org = AnyOrg(TREE)

    assert org.allows_read(TREE.ref("services"), TREE.ref("services.api"), "notes")
    assert org.allows_read(TREE.ref("services.api"), TREE.ref("services.web"), "notes")


@pytest.mark.parametrize("kind", [SoftwareOrg, ResearchOrg])
def test_in_the_shipped_orgs_what_is_written_is_read_down_the_tree_only(
    kind: type[OrgBase],
):
    org = kind(TREE)
    api, auth = TREE.ref("services.api"), TREE.ref("services.api.auth")

    assert org.allows_read(api, api, "spec")
    assert org.allows_read(api, TREE.ref("services"), "contracts")
    assert org.allows_read(auth, TREE.ref("services"), "contracts")
    assert not org.allows_read(api, auth, "contracts")
    assert not org.allows_read(api, TREE.ref("services.web"), "contracts")


def test_a_document_says_what_the_members_of_its_team_are_told_of_it(tmp_path: Path):
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / "glossary.yaml").write_text(
        "description: What the words mean.\n"
        "rules:\n"
        "    everyone: Use the words of the glossary; your {count} agents share it.\n"
        "    writer: Add a word to the glossary before you use it.\n"
    )
    [glossary] = load_documents([folder]).values()
    names = SHIPPED[DEFAULT_TYPE].names

    told = org_protocol(names, [glossary.rules_by_member])
    without = org_protocol(names)

    assert "- Use the words of the glossary; your two agents share it." in told
    assert "- Add a word to the glossary before you use it." in told
    assert "glossary" not in without
    assert "spec" not in without.lower() and "contracts" not in without
    assert "`report`" not in without


def test_a_document_says_its_own_part_of_a_briefing(
    store: GitDocumentStore, tmp_path: Path
):
    (tmp_path / "notes.yaml").write_text("description: Notes.\n")
    plain = protocol_of(read_document(tmp_path / "notes.yaml"), store, TREE)
    contracts = protocol_of(
        load_documents([SHIPPED_DOCUMENTS])["contracts"], store, TREE
    )

    api = TREE.ref("services.api")

    def briefed(*shown: ShownDocument) -> DocumentsBriefed:
        return DocumentsBriefed(api, shown, {"ic": "IC"})

    own = ShownDocument(api.folder, "notes", "# Notes", own=True)
    above = ShownDocument("services", "contracts", "# Agreed", own=False)

    assert list(plain.told_in_briefing(briefed(own))) == [
        ("Your team's notes:", "# Notes")
    ]
    assert list(plain.told_in_briefing(briefed())) == []
    [(heading, text)] = contracts.told_in_briefing(briefed(above))
    assert heading.startswith("Contracts your team must keep")
    assert text == "# Agreed"


def test_a_protocol_names_what_writes_its_document_and_does_no_writing_itself():
    # What a document is and what carries out a call of its tool are two
    # classes, as a tool's definition and its executor are.
    for held in ("write", "edited", "arguments_for"):
        assert not hasattr(DocumentProtocolBase, held)
    assert DocumentProtocolBase.writer_class is EditorWriter
    assert issubclass(EditorWriter, DocumentWriterBase)
