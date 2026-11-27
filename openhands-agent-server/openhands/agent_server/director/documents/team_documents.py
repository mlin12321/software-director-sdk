"""The documents of an org as one team reads, writes and leaves them.

It works on a store of documents and the protocols an org declared, and on
plain values that say who asks (:class:`Reader`). It knows nothing of the
service, of sessions' places or of what the director has under way.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from openhands.agent_server.director.documents.events import (
    SubTeamsProposed,
    TeamRemoved,
)
from openhands.agent_server.director.documents.protocol import DocumentProtocolBase
from openhands.agent_server.director.documents.schema import (
    DocumentsBriefed,
    ShownDocument,
)
from openhands.agent_server.director.documents.store import (
    DocumentRef,
    DocumentStore,
    TeamRef,
)
from openhands.agent_server.director.git_store import normalize_store_path
from openhands.agent_server.director.models import LeftoverDocument, TeamDocument
from openhands.agent_server.director.org import OrgBase
from openhands.agent_server.director.prompts import left_document
from openhands.agent_server.director.scope import is_inside


@dataclass(frozen=True)
class Reader:
    """A team as its documents see it.

    Attributes:
        team: The team, or the work of one in a session.
        keeps: The names of the documents its type keeps, in the order the
            type lists them.
        leftover: What teams before it left in its folder.
        titles: What each member of the team is called.
        descriptions: What its type says each document holds, by name.
    """

    team: TeamRef
    keeps: tuple[str, ...] = ()
    leftover: tuple[LeftoverDocument, ...] = ()
    titles: Mapping[str, str] = field(default_factory=dict)
    descriptions: Mapping[str, str] = field(default_factory=dict)


class RemovedTeam(NamedTuple):
    """A team that is replaced or pruned, as what it leaves is worked out.

    Attributes:
        folder: Its folder.
        title: What it was called.
        leftover: What it was itself left and never rewrote.
    """

    folder: str
    title: str
    leftover: Sequence[LeftoverDocument]


class Changed(NamedTuple):
    """A document a session wrote while the org's in the same folder was
    changed by someone else.

    Attributes:
        name: The name of the document.
        folder: The folder it is kept in.
        now: The org's text now.
        written: What the session wrote.
    """

    name: str
    folder: str
    now: str
    written: str


class TeamDocuments:
    """What the documents of an org are to each of its teams."""

    def __init__(
        self,
        store: DocumentStore,
        protocols: Mapping[str, DocumentProtocolBase],
        org: OrgBase,
    ) -> None:
        """
        Args:
            store: Where documents are kept.
            protocols: What each document the org declares is, by its name.
            org: The kind of org, which may refuse a read.
        """
        self.store = store
        self.protocols = protocols
        self.org = org

    def kept_by(self, reader: Reader) -> list[DocumentProtocolBase]:
        """What each document a team keeps is, as its type lists them."""
        return [self.protocols[name] for name in reader.keeps if name in self.protocols]

    def at(self, file: str) -> DocumentProtocolBase | None:
        """What the document kept in ``file`` of team memory is; None for
        a file that is no document anyone declared."""
        return self.protocols.get(DocumentRef.at(file).name)

    def stays_when_team_is_removed(self, name: str) -> bool:
        """Whether the document named ``name`` is one a removed team
        leaves in its folder: its protocol says."""
        protocol = self.protocols.get(name)
        return protocol is not None and protocol.stays_when_team_is_removed

    def _in_order_for(self, reader: Reader) -> list[DocumentProtocolBase]:
        """Every document of the org: first the ones the team's type
        keeps, in the order the type lists them."""
        kept = self.kept_by(reader)
        return [
            *kept,
            *(found for found in self.protocols.values() if found not in kept),
        ]

    def shown(self, reader: Reader) -> list[ShownDocument]:
        """Every document a team reads, as its session has it: of each
        kind of document, the ones of the teams its protocol names, where
        the org allows it."""
        team = reader.team
        left = {(old.name, normalize_store_path(old.path)) for old in reader.leftover}
        found: dict[DocumentRef, ShownDocument] = {}
        for protocol in self._in_order_for(reader):
            name = protocol.document.name
            for owner in protocol.read_from(team):
                ref = owner.keeps(name)
                # A session opened at the top team has that team's own
                # place above it: one folder, shown once.
                if ref in found or not self.org.allows_read(team, owner, name):
                    continue
                # What a team before this one left in the folder is not
                # this team's own until it writes one itself.
                if (name, ref.folder) in left:
                    continue
                text = self.store.read(team.session, ref)
                if text:
                    found[ref] = ShownDocument(
                        folder=ref.folder,
                        name=name,
                        text=text,
                        own=ref.folder == team.folder,
                    )
        return list(found.values())

    def told(self, reader: Reader) -> dict[str, Any]:
        """What the context of a team's agents says of the documents it
        reads, as each document has it said: the ones its type keeps, in
        the order the type lists them, and any other it is shown one of."""
        shown = self.shown(reader)
        kept = self.kept_by(reader)
        told: dict[str, Any] = {}
        for protocol in self._in_order_for(reader):
            name = protocol.document.name
            of_it = [found for found in shown if found.name == name]
            if of_it or protocol in kept:
                told |= protocol.told_to_reader(reader.team, of_it)
        return told

    def briefed(self, reader: Reader) -> list[tuple[str, str]]:
        """What the first message of a team's agents says of the documents
        it reads, as each document has it said, and after each what teams
        before it left of that document in its folder."""
        shown = self.shown(reader)
        kept = self.kept_by(reader)
        left = self.left_to(reader)
        parts: list[tuple[str, str]] = []
        for protocol in self._in_order_for(reader):
            document = protocol.document
            of_it = [found for found in shown if found.name == document.name]
            if of_it or protocol in kept:
                parts += [
                    (part.heading, part.text)
                    for part in protocol.told_in_briefing(
                        DocumentsBriefed(
                            reader=reader.team, shown=of_it, titles=reader.titles
                        )
                    )
                ]
            parts += [
                (
                    left_document(
                        document.name, old.path, old.left_by, document.written_with
                    ),
                    old.text,
                )
                for old in left
                if old.name == document.name
            ]
        return parts

    def left_to(self, reader: Reader) -> list[TeamDocument]:
        """What teams before this one left in its folder: each is old, and
        none is the team's own until it writes one itself."""
        found = []
        for old in reader.leftover:
            protocol = self.protocols.get(old.name)
            path = normalize_store_path(old.path)
            text = self.store.read(reader.team.session, DocumentRef(path, old.name))
            if protocol is None or not text:
                continue
            found.append(
                TeamDocument(
                    name=old.name,
                    description=reader.descriptions.get(old.name)
                    or protocol.document.description,
                    path=path,
                    text=text,
                    own=False,
                    left=True,
                    left_by=old.team,
                )
            )
        return found

    def listed(self, reader: Reader) -> list[TeamDocument]:
        """The documents a team reads, for whoever looks at the team: the
        ones it is shown, and with them what teams before it left in its
        folder."""
        shown = self.shown(reader)
        left = self.left_to(reader)
        found: list[TeamDocument] = []
        for protocol in self._in_order_for(reader):
            document = protocol.document
            said = reader.descriptions.get(document.name) or document.description
            found += [
                TeamDocument(
                    name=one.name,
                    description=said,
                    path=one.folder,
                    text=one.text,
                    own=one.own,
                )
                for one in shown
                if one.name == document.name
            ]
            found += [old for old in left if old.name == document.name]
        return found

    def read_by_sub_teams(self, reader: Reader) -> list[tuple[str, str]]:
        """The documents of a team that every team directly below it
        reads, as its session has them: the name and the text of each
        that says something."""
        team = reader.team
        found = []
        for protocol in self.kept_by(reader):
            if not protocol.read_by_sub_teams:
                continue
            name = protocol.document.name
            text = self.store.read(team.session, team.keeps(name))
            if text and text.strip():
                found.append((name, text.strip()))
        return found

    def before_sub_teams_are_proposed(self, reader: Reader) -> None:
        """Ask the documents of a team whether its writer may propose
        sub-teams now.

        Raises:
            Invalid: One of them is not ready, with what to do first.
        """
        proposed = SubTeamsProposed(reader.team)
        for protocol in self.kept_by(reader):
            protocol.before_sub_teams_are_proposed(proposed)

    def when_team_is_removed(self, reader: Reader) -> dict[str, str]:
        """Tell the documents of a team that it is gone.

        Returns:
            What is kept of each in the team's record, by its name.
        """
        removed = TeamRemoved(reader.team)
        kept = {
            protocol.document.name: protocol.when_team_is_removed(removed)
            for protocol in self.kept_by(reader)
        }
        return {name: text for name, text in kept.items() if text}

    def draft(self, team: TeamRef, name: str) -> tuple[str, str | None] | None:
        """The document named ``name`` a session wrote for one of its
        teams, which is its own until its work is accepted.

        Returns:
            The text and the team's text it was written over, None when
            the team had none; None when the session wrote none there.
        """
        draft = self.store.drafts(team.session).get(team.keeps(name))
        if draft is None or draft.text is None:
            return None
        return draft.text, draft.over

    def changed_elsewhere(self, session: str) -> list[Changed]:
        """The documents a session wrote while the org's document in the
        same folder was changed by someone else: the ones that land with
        its work, and not those that wait with its teams."""
        found = []
        drafts = self.store.drafts(session)
        for ref in sorted(drafts, key=lambda ref: ref.path):
            draft = drafts[ref]
            protocol = self.protocols.get(ref.name)
            if (
                protocol is None
                or protocol.waits_for_team_changes
                or draft.text is None
            ):
                continue
            # No session: the org's own.
            now = self.store.read("", ref)
            if now != draft.over and now != draft.text:
                found.append(Changed(ref.name, ref.folder, now or "", draft.text))
        return found

    def left_in(
        self, session: str, path: str, removed: Sequence[RemovedTeam]
    ) -> list[LeftoverDocument]:
        """The documents a team that is given ``path`` in ``session``
        finds there, of the kind that stays when a team is removed: the
        ones of the folder itself and those of the ``removed`` teams whose
        folders hold it or lie in it. None of them is its own."""
        own = normalize_store_path(path)
        staying = [
            name
            for name, protocol in self.protocols.items()
            if protocol.stays_when_team_is_removed
        ]
        left: dict[tuple[str, str], str | None] = {}

        def concerns(directory: str) -> bool:
            return (
                directory == own
                or is_inside(own, directory)
                or is_inside(directory, own)
            )

        for team in removed:
            # What a removed team was left and never rewrote is passed on.
            for old in team.leftover:
                if concerns(old.path):
                    left.setdefault((old.name, old.path), old.team)
            theirs = normalize_store_path(team.folder)
            if concerns(theirs):
                for name in staying:
                    left[(name, theirs)] = team.title
        # One in the folder itself that no team here is known to have
        # left: a team pruned long ago, or one written by hand.
        for name in staying:
            left.setdefault((name, own), None)
        return [
            LeftoverDocument(name=name, path=directory, team=team)
            for (name, directory), team in left.items()
            if self.store.read(session, DocumentRef(directory, name))
        ]
