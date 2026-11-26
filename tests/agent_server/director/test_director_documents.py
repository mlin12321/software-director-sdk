"""The documents teams keep: where they are kept and what one is said to be."""

from pathlib import Path

import pytest

from openhands.agent_server.director.agent_types import InvalidAgentType, read_type
from openhands.agent_server.director.documents.declared import (
    InvalidDocument,
    load_documents,
    read_document,
)
from openhands.agent_server.director.documents.store import (
    DocRef,
    Draft,
    GitDocumentStore,
)
from openhands.agent_server.director.git_store import GitStore
from openhands.agent_server.director.orgs import SHIPPED_DOCUMENTS, SHIPPED_TYPES
from openhands.agent_server.director.run_state import RunState
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
