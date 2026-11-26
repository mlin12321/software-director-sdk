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
    HandedOver,
    SubTeamsCreated,
    TeamRef,
    TeamRemoved,
    WorkAccepted,
)
from openhands.agent_server.director.documents.protocol import (
    DocumentProtocolBase,
    UnknownProtocol,
    protocol_of,
)
from openhands.agent_server.director.documents.store import (
    DocRef,
    Draft,
    GitDocumentStore,
)
from openhands.agent_server.director.git_store import GitStore
from openhands.agent_server.director.orgs import SHIPPED_DOCUMENTS, SHIPPED_TYPES
from openhands.agent_server.director.orgs.documents import ContractsProtocol
from openhands.agent_server.director.run_state import RunState
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import DIRECTOR_TOOL_NAMES


NOTES = DocRef("services/api", "notes")


@pytest.fixture
def store(repo: Path) -> GitDocumentStore:
    runs = RunState(None)
    return GitDocumentStore(lambda: GitStore(repo), lambda: runs)


def test_a_document_is_kept_in_its_teams_folder_under_its_name():
    assert NOTES.path == "services/api/notes.md"
    assert DocRef("", "notes").path == "notes.md"
    assert DocRef.at("services/api/notes.md") == NOTES
    assert DocRef.at("notes.md") == DocRef("", "notes")


def test_what_a_session_writes_is_its_own_until_it_is_taken(
    store: GitDocumentStore, repo: Path
):
    GitStore(repo).write("services/api", "notes.md", "The org's.")

    store.write("one", NOTES, "Session one's.")

    assert store.read("one", NOTES) == "Session one's."
    assert store.read("two", NOTES) == "The org's."
    assert store.read("", NOTES) == "The org's."
    assert store.drafts("one") == {NOTES: Draft("Session one's.", "The org's.")}
    assert store.drafts("two") == {}


def test_a_session_keeps_what_it_first_wrote_over(store: GitDocumentStore, repo: Path):
    GitStore(repo).write("services/api", "notes.md", "First.")
    store.write("one", NOTES, "Mine.")
    GitStore(repo).write("services/api", "notes.md", "The org moved on.")

    store.write("one", NOTES, "Mine, again.")

    assert store.drafts("one")[NOTES] == Draft("Mine, again.", "First.")


def test_a_session_may_remove_a_document_or_forget_what_it_made_of_it(
    store: GitDocumentStore, repo: Path
):
    GitStore(repo).write("services/api", "notes.md", "The org's.")

    store.remove("one", NOTES)

    assert store.read("one", NOTES) is None
    assert store.drafts("one")[NOTES] == Draft(None, "The org's.")

    store.discard("one", NOTES)

    assert store.read("one", NOTES) == "The org's."
    assert store.drafts("one") == {}


def test_the_shipped_documents_are_the_ones_the_shipped_types_keep():
    documents = load_documents([SHIPPED_DOCUMENTS])

    assert set(documents) == {"spec", "contracts", "report"}
    assert documents["spec"].written_with == "write_spec"
    assert documents["contracts"].written_with == "edit_contracts"
    for folder in SHIPPED_TYPES:
        for path in folder.glob("*.yaml"):
            kind = read_type(path, DIRECTOR_TOOL_NAMES, documents)
            assert set(kind.documents) == set(documents)
            assert all(kind.documents.values())


def test_a_document_needs_no_more_than_a_name_and_what_it_holds(tmp_path: Path):
    path = tmp_path / "decision-log.yaml"
    path.write_text("holds: What was decided and why.\n")

    document = read_document(path)

    assert document.name == "decision-log"
    assert document.written_with == "write_decision_log"
    assert document.protocol is None


def test_a_file_that_is_no_document_is_left_out(tmp_path: Path):
    (tmp_path / "notes.yaml").write_text("holds: Notes.\n")
    (tmp_path / "empty.yaml").write_text("name: empty\n")
    (tmp_path / "other.yaml").write_text("name: another\nholds: Something.\n")
    (tmp_path / "flagged.yaml").write_text("holds: Something.\nappend: true\n")

    with pytest.raises(InvalidDocument, match="name the file after it"):
        read_document(tmp_path / "other.yaml")

    assert set(load_documents([tmp_path])) == {"notes"}


def test_a_type_says_what_a_document_holds_for_it_or_takes_the_documents_own(
    tmp_path: Path,
):
    (tmp_path / "notes.yaml").write_text("holds: Notes of the team.\n")
    (tmp_path / "plan.yaml").write_text("holds: The plan.\n")
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
    (tmp_path / "notes.yaml").write_text("holds: Notes.\n")

    protocol = protocol_of(read_document(tmp_path / "notes.yaml"), store)

    assert type(protocol) is DocumentProtocolBase
    assert not protocol.waits_for_team_changes()


def test_a_document_does_what_the_protocol_it_names_does(
    store: GitDocumentStore, tmp_path: Path
):
    # Any document may name the protocol: it is not the name `contracts`
    # that makes a document wait for the teams.
    (tmp_path / "glossary.yaml").write_text("holds: Terms.\nprotocol: contracts\n")
    (tmp_path / "plan.yaml").write_text("holds: The plan.\nprotocol: nobodys\n")

    protocol = protocol_of(read_document(tmp_path / "glossary.yaml"), store)

    assert isinstance(protocol, ContractsProtocol)
    assert protocol.waits_for_team_changes()
    with pytest.raises(UnknownProtocol, match="nobodys"):
        protocol_of(read_document(tmp_path / "plan.yaml"), store)


def test_the_director_asks_a_files_document_what_it_does(repo: Path):
    service = DirectorService(repo)

    assert set(service.protocols) == {"spec", "contracts", "report"}
    contracts = service.protocol_at("services/api/contracts.md")
    assert contracts is not None and contracts.waits_for_team_changes()
    spec = service.protocol_at("spec.md")
    assert spec is not None and not spec.waits_for_team_changes()
    assert service.protocol_at("services/api/README.md") is None


TEAM = TeamRef(id="services.api@one", session="one", folder="services/api")
CONTRACTS = DocRef("services/api", "contracts")


@pytest.fixture
def contracts(store: GitDocumentStore) -> ContractsProtocol:
    document = load_documents([SHIPPED_DOCUMENTS])["contracts"]
    protocol = protocol_of(document, store)
    assert isinstance(protocol, ContractsProtocol)
    return protocol


def test_contracts_are_seeded_by_the_proposal_that_made_the_sub_teams(
    contracts: ContractsProtocol, store: GitDocumentStore
):
    contracts.when_sub_teams_are_created(
        SubTeamsCreated(TEAM, {"contracts": "Cart owns the cart page."})
    )
    # A later proposal that replaces no team adds to them, once.
    for _ in range(2):
        contracts.when_sub_teams_are_created(
            SubTeamsCreated(TEAM, {"contracts": "Checkout owns the order."})
        )

    assert store.read("one", CONTRACTS) == (
        "Cart owns the cart page.\n\nCheckout owns the order.\n"
    )
    assert store.read("two", CONTRACTS) is None


def test_contracts_of_a_proposal_that_replaced_teams_take_the_place_of_the_old(
    contracts: ContractsProtocol, store: GitDocumentStore
):
    store.write("one", CONTRACTS, "Cart owns the cart page.\n")

    contracts.when_sub_teams_are_created(
        SubTeamsCreated(TEAM, {"contracts": "Shop owns it all."}, replaced=True)
    )

    assert store.read("one", CONTRACTS) == "Shop owns it all.\n"


def test_contracts_grow_by_what_a_reviewer_establishes_and_a_hand_over_replaces(
    contracts: ContractsProtocol, store: GitDocumentStore
):
    store.write("one", CONTRACTS, "Cart owns the cart page.\n")

    contracts.when_sub_team_work_is_accepted(
        WorkAccepted(TEAM, {"contracts": "Prices are in cents."})
    )

    assert store.read("one", CONTRACTS) == (
        "Cart owns the cart page.\n\nPrices are in cents.\n"
    )

    contracts.when_removed_team_is_handed_over(
        HandedOver(TEAM, {"contracts": "Only the cart is left."})
    )

    assert store.read("one", CONTRACTS) == "Only the cart is left.\n"


def test_contracts_are_told_only_of_what_is_given_for_them(
    contracts: ContractsProtocol, store: GitDocumentStore
):
    contracts.when_sub_teams_are_created(SubTeamsCreated(TEAM, {"notes": "Other."}))
    contracts.when_sub_team_work_is_accepted(WorkAccepted(TEAM, {"contracts": "  "}))

    assert store.drafts("one") == {}


def test_a_removed_teams_contracts_go_into_its_record_and_out_of_its_folder(
    contracts: ContractsProtocol, store: GitDocumentStore, repo: Path
):
    GitStore(repo).write("services/api", "contracts.md", "Cart owns the cart page.\n")

    kept = contracts.when_team_is_removed(TeamRemoved(TEAM))

    assert kept == "Cart owns the cart page.\n"
    assert store.read("one", CONTRACTS) is None
    assert store.read("", CONTRACTS) == "Cart owns the cart page.\n"
    # A plain document keeps nothing of itself and stays where it is.
    plain = DocumentProtocolBase(contracts.document, store)
    assert plain.when_team_is_removed(TeamRemoved(TEAM)) is None


def test_the_store_tells_of_every_change_a_session_makes(repo: Path):
    told: list[str] = []
    runs = RunState(None)
    store = GitDocumentStore(lambda: GitStore(repo), lambda: runs, told.append)

    store.write("one", NOTES, "Mine.")
    store.remove("one", NOTES)
    store.discard("one", NOTES)

    assert told == ["one", "one"]
