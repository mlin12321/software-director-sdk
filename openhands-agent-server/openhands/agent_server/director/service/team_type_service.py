"""Team types: what a team may be and the documents teams keep, as the
director ships them and as an org declares its own, and what follows from
a team's type for its members."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from openhands.agent_server.director.agent_types import (
    DEFAULT_TYPE,
    TYPES_FOLDER,
    Names,
    TeamTemplate,
    load_types,
)
from openhands.agent_server.director.documents.declared import (
    DOCUMENTS_FOLDER,
    load_documents,
)
from openhands.agent_server.director.documents.protocol import DocumentProtocolBase
from openhands.agent_server.director.documents.registry import (
    UnknownProtocol,
    declare,
    protocol_of,
)
from openhands.agent_server.director.documents.store import REPORT, DocumentStore
from openhands.agent_server.director.documents.team_documents import Reader
from openhands.agent_server.director.documents.tree import TeamTree
from openhands.agent_server.director.models import OrgNodeInput, SessionTeam
from openhands.agent_server.director.org_store import (
    OrgStore,
    is_division_lead,
    team_ref,
)
from openhands.agent_server.director.orgs import SHIPPED_DOCUMENTS, SHIPPED_TYPES
from openhands.agent_server.director.prompts import KeptDocuments, role_text
from openhands.agent_server.director.team import Team
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# Every tool of the director's own; they register in ``director.tools``.
# The documents an org keeps bring theirs beside these. Which of them an
# agent is offered is its team's type's to say, and `agent_actions`
# refuses the rest whatever is offered.
DIRECTOR_TOOL_NAMES = (
    "read_team_context",
    "raise_escalation",
    "resolve_escalation",
    "msg_manager",
    "propose_teams",
    "msg_team",
    "send_document",
    "hand_over",
    "stop_team",
    "submit_review",
    "review_team",
    "msg_ic",
    "approve_teams",
    "decline_teams",
    "prune_team",
    "settle_teams",
)
# What no type takes from an agent: without them it could not read its
# place, hand its work on, or say that it cannot go on.
_TOOLS_NO_TYPE_REMOVES = ("read_team_context", "raise_escalation", "submit_review")
# What the member that takes over what a removed team left is offered,
# beside the tools of the documents it may write while it does.
_HAND_OVER_TOOLS = ("read_team_context", "hand_over")


@dataclass
class TeamTypeService:
    """Reads the types and documents that are declared, and answers for a
    team what its type says: who its members are, what they are called,
    told and offered, and which documents it keeps.

    The files are read again when one among them is added, removed or
    changed, so a new type can be given to the next team without a
    restart.
    """

    config_file: Path | None
    """The org's ``director.yaml``; its own types and documents are in
    the folders beside it. None: only what the director ships."""
    documents: DocumentStore
    """Where documents are kept, for what each document does."""
    tree: TeamTree
    """Where a team stands among the others, for the same."""
    org: Callable[[], OrgStore]
    """The org whose teams are asked about."""
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _agent_types: dict[str, TeamTemplate] | None = field(default=None, init=False)
    _protocols: dict[str, DocumentProtocolBase] = field(
        default_factory=dict, init=False
    )
    # The files the two were read from, with when each was last changed.
    _seen: tuple[tuple[str, int], ...] = field(default=(), init=False)

    @property
    def agent_types(self) -> dict[str, TeamTemplate]:
        """The types a team may be, by name: the ones the director ships
        and the ones beside ``director.yaml``."""
        return self._declared()[0]

    @property
    def protocols(self) -> dict[str, DocumentProtocolBase]:
        """What each document teams may keep does, by the document's name.
        Read again with the types."""
        return self._declared()[1]

    def _declared(
        self,
    ) -> tuple[dict[str, TeamTemplate], dict[str, DocumentProtocolBase]]:
        folders = [*SHIPPED_TYPES]
        documents = [SHIPPED_DOCUMENTS]
        if self.config_file:
            folders.append(self.config_file.parent / TYPES_FOLDER)
            documents.append(self.config_file.parent / DOCUMENTS_FOLDER)
        files = sorted(
            path
            for folder in (*folders, *documents)
            if folder.is_dir()
            for path in folder.iterdir()
            if path.suffix in (".yaml", ".yml")
        )
        seen = tuple((str(path), path.stat().st_mtime_ns) for path in files)
        with self._lock:
            if self._agent_types is None or seen != self._seen:
                protocols: dict[str, DocumentProtocolBase] = {}
                tools: dict[str, str] = {}
                for document in load_documents(documents).values():
                    tool = document.written_with
                    try:
                        if tool in tools:
                            raise ValueError(
                                f"The document {tools[tool]!r} is written with "
                                f"{tool} already"
                            )
                        declare(document)
                        protocols[document.name] = protocol_of(
                            document, self.documents, self.tree
                        )
                    except (UnknownProtocol, ValueError):
                        logger.exception("Ignoring the document %s", document.name)
                        continue
                    tools[tool] = document.name
                self._agent_types = load_types(
                    folders,
                    DIRECTOR_TOOL_NAMES,
                    {name: found.document for name, found in protocols.items()},
                )
                self._protocols = protocols
                self._seen = seen
            return self._agent_types, self._protocols

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Every tool an agent of the org may be offered: the director's
        own and the one of each document the org keeps."""
        written_with = (
            protocol.document.written_with for protocol in self.protocols.values()
        )
        return tuple(dict.fromkeys((*DIRECTOR_TOOL_NAMES, *written_with)))

    @property
    def hand_over_tools(self) -> tuple[str, ...]:
        """What the member that takes over what a removed team left is
        offered in the conversation it does so in."""
        return _HAND_OVER_TOOLS + tuple(
            protocol.document.written_with
            for protocol in self.protocols.values()
            if protocol.usable_in_hand_over
        )

    def type_of(self, node: OrgNodeInput) -> TeamTemplate:
        """The type of a team. A team whose type the director no longer
        finds is told as the default type is."""
        types = self.agent_types
        return types.get(node.agent_type) or types[DEFAULT_TYPE]

    def team(self, place: SessionTeam) -> Team:
        """A team of the org as the director reads it."""
        return Team(self.org(), place)

    def names_at(self, place_id: str) -> Names:
        """What the members the director turns to are called in a team; in
        one that is gone, what they are called in a team of the default
        type."""
        place = self.org().place(place_id)
        if place is None:
            return self.agent_types[DEFAULT_TYPE].names
        return self.team(place).names

    def writer_conversation(self, place: SessionTeam) -> str | None:
        """The conversation of the member that works in a team's folder."""
        team = self.team(place)
        return team.conversation(team.writer)

    def reviewer_conversation(self, place: SessionTeam) -> str | None:
        """The conversation of the member the teams below turn to."""
        team = self.team(place)
        return team.conversation(team.reviewer)

    def tools_for(self, place: SessionTeam, member: str) -> tuple[str, ...]:
        """The director's tools the member of a team is offered: the ones
        its type gives it, less the ones the type leaves its teams
        without. What an agent needs to read its place and hand its work
        on stays."""
        kind = self.type_of(place)
        found = kind.agents.get(member)
        if found is None:
            return ()
        if is_division_lead(place):
            return tuple(found.lead_tools)
        return tuple(
            name
            for name in found.tools
            if name not in kind.omit_tools or name in _TOOLS_NO_TYPE_REMOVES
        )

    def reader_of(self, node: OrgNodeInput) -> Reader:
        """A team as its documents see it."""
        kind = self.type_of(node)
        return Reader(
            team=team_ref(node),
            keeps=tuple(kind.documents),
            leftover=tuple(node.leftover_documents),
            titles=kind.titles,
            descriptions={name: said for name, said in kind.documents.items() if said},
        )

    def protocols_kept_by(self, node: OrgNodeInput) -> list[DocumentProtocolBase]:
        """What each document a team keeps does, in the order its type
        lists them. A document nobody declared is left out."""
        protocols = self.protocols
        return [
            protocols[name]
            for name in self.type_of(node).documents
            if name in protocols
        ]

    def kept_documents(self, place: OrgNodeInput) -> KeptDocuments:
        """The documents a team keeps, as the director's own messages name
        them: what each is called and what its protocol says of it."""
        kept = [
            protocol
            for protocol in self.protocols_kept_by(place)
            if protocol.document.name != REPORT
        ]

        def named(fact: str) -> tuple[str, ...]:
            return tuple(found.document.name for found in kept if getattr(found, fact))

        return KeptDocuments(
            names=tuple(found.document.name for found in kept),
            wait_with_teams=named("waits_for_team_changes"),
            read_by_sub_teams=named("read_by_sub_teams"),
            stay_when_removed=named("stays_when_team_is_removed"),
            reports=REPORT in self.type_of(place).documents,
        )

    def document_rules(self, place: SessionTeam) -> list[dict[str, str]]:
        """What each document a team keeps says of itself to its members,
        with how the org works."""
        return [
            protocol.document.rules_by_member
            for protocol in self.protocols_kept_by(place)
        ]

    def role_text(self, place: SessionTeam, member: str) -> str:
        """How the org works and what the member's part in it is, as its
        briefing says it."""
        team = self.team(place)
        found = team.kind.agents.get(member)
        return role_text(
            team.names,
            works=member == team.writer,
            tools=self.tools_for(place, member),
            title=team.title(member),
            lead=team.is_lead,
            own=found.protocol if found else "",
            rules=self.document_rules(place),
        )
