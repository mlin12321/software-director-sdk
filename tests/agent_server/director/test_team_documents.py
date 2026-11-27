"""The documents of an org as a team reads them, with no service: a store
that keeps texts in a dict and the protocols an org declared."""

import pytest

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.protocol import DocumentProtocolBase
from openhands.agent_server.director.documents.store import (
    DocumentDraft,
    DocumentRef,
    DocumentStore,
    LogEntry,
    TeamRef,
)
from openhands.agent_server.director.documents.team_documents import (
    Changed,
    Reader,
    RemovedTeam,
    TeamDocuments,
)
from openhands.agent_server.director.models import LeftoverDocument
from openhands.agent_server.director.org import OrgBase
from openhands.sdk.errors import Invalid


SESSION = "s1"
ROOT = TeamRef(id="root", session=SESSION, folder="")
API = TeamRef(id="api", session=SESSION, folder="services/api")


class TextsInADict(DocumentStore):
    """The org's texts, and beside them what each session wrote."""

    def __init__(self) -> None:
        self.org: dict[DocumentRef, str] = {}
        self.written: dict[tuple[str, DocumentRef], DocumentDraft] = {}

    def read(self, session: str, ref: DocumentRef) -> str | None:
        draft = self.written.get((session, ref))
        return draft.text if draft else self.org.get(ref)

    def write(self, session: str, ref: DocumentRef, text: str) -> None:
        self.written[(session, ref)] = DocumentDraft(text, self.org.get(ref))

    def remove(self, session: str, ref: DocumentRef) -> None:
        self.written[(session, ref)] = DocumentDraft(None, self.org.get(ref))

    def discard(self, session: str, ref: DocumentRef) -> None:
        self.written.pop((session, ref), None)

    def drafts(self, session: str) -> dict[DocumentRef, DocumentDraft]:
        return {
            ref: draft for (of, ref), draft in self.written.items() if of == session
        }

    def entries(self, team: TeamRef, name: str) -> list[LogEntry]:  # noqa: ARG002
        return []

    def append(self, team: TeamRef, name: str, entry: LogEntry) -> None:
        raise NotImplementedError


class RootAboveApi:
    def above(self, team: TeamRef) -> TeamRef | None:
        return ROOT if team == API else None

    def below(self, team: TeamRef) -> list[TeamRef]:
        return [API] if team == ROOT else []


class Notes(DocumentProtocolBase):
    """A document of an org's own that stays when its team is removed."""

    stays_when_team_is_removed = True


class Rules(DocumentProtocolBase):
    """One a team's sub-teams read, which waits with the teams."""

    read_by_sub_teams = True
    waits_for_team_changes = True


class AnyOrg(OrgBase):
    name = "any"


class NobodyReadsAbove(OrgBase):
    name = "closed"

    def allows_read(self, reader: TeamRef, owner: TeamRef, document: str) -> bool:  # noqa: ARG002
        return reader == owner


def documents(org: OrgBase | None = None) -> tuple[TeamDocuments, TextsInADict]:
    store, tree = TextsInADict(), RootAboveApi()
    protocols = {
        "notes": Notes(
            Document(name="notes", description="How it works."), store, tree
        ),
        "rules": Rules(Document(name="rules", description="What holds."), store, tree),
    }
    return TeamDocuments(store, protocols, org or AnyOrg(tree)), store


def reader(team: TeamRef, *left: LeftoverDocument) -> Reader:
    return Reader(team=team, keeps=("notes", "rules"), leftover=left)


def test_a_team_is_shown_its_own_and_what_the_team_above_keeps_for_it():
    of_teams, store = documents()
    store.org[API.keeps("notes")] = "Orders first."
    store.org[ROOT.keeps("rules")] = "Ids are integers."
    store.org[ROOT.keeps("notes")] = "The root's own."

    shown = of_teams.shown(reader(API))

    assert [(one.name, one.folder, one.own) for one in shown] == [
        ("notes", "services/api", True),
        ("rules", "", False),
    ]
    assert of_teams.told(reader(API)) == {"notes": "Orders first.", "rules": None}
    assert of_teams.read_by_sub_teams(reader(ROOT)) == [("rules", "Ids are integers.")]


def test_an_org_may_refuse_what_a_document_would_show():
    of_teams, store = documents(NobodyReadsAbove(RootAboveApi()))
    store.org[ROOT.keeps("rules")] = "Ids are integers."

    assert of_teams.shown(reader(API)) == []


def test_what_a_team_before_left_is_old_and_not_the_teams_own():
    of_teams, store = documents()
    store.org[API.keeps("notes")] = "How the old team worked."
    old = LeftoverDocument(name="notes", path="services/api", team="Orders")

    assert of_teams.shown(reader(API, old)) == []
    [left] = of_teams.left_to(reader(API, old))
    assert (left.text, left.left, left.left_by, left.own) == (
        "How the old team worked.",
        True,
        "Orders",
        False,
    )
    [(heading, text)] = of_teams.briefed(reader(API, old))
    assert "left by a team before you (Orders)" in heading and "write_notes" in heading
    assert text == "How the old team worked."


def test_a_removed_team_leaves_the_documents_that_stay_to_the_next_in_its_folder():
    of_teams, store = documents()
    store.org[API.keeps("notes")] = "How it worked."
    store.org[API.keeps("rules")] = "Between its teams."
    removed = [RemovedTeam("services/api", "Orders", [])]

    left = of_teams.left_in(SESSION, "services", removed)

    # Only the kind that stays, and only where there is one.
    assert left == [LeftoverDocument(name="notes", path="services/api", team="Orders")]
    assert of_teams.left_in(SESSION, "apps/web", removed) == []


def test_what_a_session_wrote_over_a_text_the_org_changed_since_is_named():
    of_teams, store = documents()
    store.org[API.keeps("notes")] = "Orders."
    store.org[API.keeps("rules")] = "Ids."
    store.write(SESSION, API.keeps("notes"), "Refunds.")
    store.write(SESSION, API.keeps("rules"), "Uuids.")

    assert of_teams.draft(API, "notes") == ("Refunds.", "Orders.")
    assert of_teams.changed_elsewhere(SESSION) == []

    store.org[API.keeps("notes")] = "Orders and carts."
    store.org[API.keeps("rules")] = "Strings."

    # The one that waits with the teams is not the reviewer's to reconcile.
    assert of_teams.changed_elsewhere(SESSION) == [
        Changed("notes", "services/api", "Orders and carts.", "Refunds.")
    ]


def test_a_removed_teams_document_that_does_not_stay_goes_into_its_record():
    of_teams, store = documents()
    store.org[API.keeps("notes")] = "How it worked."
    store.org[API.keeps("rules")] = "Between its teams."

    kept = of_teams.when_team_is_removed(reader(API))

    # The one that stays is where it was, for the next team in the folder.
    assert kept == {"rules": "Between its teams."}
    assert store.read(SESSION, API.keeps("rules")) is None
    assert store.read(SESSION, API.keeps("notes")) == "How it worked."


def test_a_document_needed_before_sub_teams_refuses_a_proposal_until_written():
    class Needed(Rules):
        needed_before_sub_teams = True

    of_teams, store = documents()
    of_teams.protocols = {
        "rules": Needed(
            Document(name="rules", description="What."), store, RootAboveApi()
        )
    }

    of_teams.before_sub_teams_are_proposed(Reader(team=API, keeps=("notes",)))
    with pytest.raises(Invalid, match="no rules yet.*`write_rules`"):
        of_teams.before_sub_teams_are_proposed(reader(API))

    store.write(SESSION, API.keeps("rules"), "Ids are integers.")
    of_teams.before_sub_teams_are_proposed(reader(API))
