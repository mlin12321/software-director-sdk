"""The history of the org's teams: what became of the teams below each
team, what the human asked of a team directly, and what removed teams
left."""

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from uuid import uuid4

from openhands.agent_server.director.boundary import owns
from openhands.agent_server.director.environment.records import (
    Landing,
    RecordStore,
    team_key,
)
from openhands.agent_server.director.models import (
    FormerTeam,
    HistoryEvent,
    OrgHistory,
    OrgNode,
    RoutedRequest,
    SessionTeam,
    TeamEvent,
    TeamEventKind,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    session_in,
    session_of,
    team_id_of,
    utc_now,
)
from openhands.agent_server.director.recovery import STEP_NAMES
from openhands.agent_server.director.run_state import Intent
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)


# How much of what the human asked is kept in the record of the team above.
ASKED_KEPT = 1000
# The events of a session that change what the org has of its teams.
CHANGES_THE_ORG: frozenset[TeamEventKind] = frozenset(
    {"set_up", "replaced", "pruned", "moved"}
)


# How much of the reason a step failed is kept.
REASON_KEPT = 600


@dataclass
class TeamHistoryService:
    """Keeps and reads the history of the org's teams.

    What a session records is its own until its work is accepted (the
    session drafts hold it); what the org has on record is in team memory.
    Reading puts the two together. All of it is safe from any thread.
    """

    org: Callable[[], OrgStore]
    """The org whose teams the history is of."""
    team_memory: Callable[[], RecordStore]
    """The org's team memory, for what needs none where there is none."""
    require_team_memory: Callable[[], RecordStore]
    """The same, raising ``Unavailable`` where the org has no team memory."""
    session_drafts: SessionDraftService
    """Where what a session recorded waits for its work to be accepted."""
    org_changed: Callable[[str], None]
    """Told the session that recorded something that changes the org."""
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    # Held from looking a need up in the record to keeping it, so that an
    # agent that tries the same file again is counted once.
    _keeping_needs: threading.Lock = field(default_factory=threading.Lock, init=False)
    # The records of removed teams by the team they were below, and the
    # routed requests, each as read at one state of team memory.
    _former_read: tuple[str | None, dict[str, list[FormerTeam]]] | None = field(
        default=None, init=False
    )
    _routed_read: tuple[str | None, list[RoutedRequest]] | None = field(
        default=None, init=False
    )

    def record_event(
        self,
        above_id: str | None,
        kind: TeamEventKind,
        team: SessionTeam | None = None,
        detail: str = "",
        paths: Sequence[str] = (),
        other: SessionTeam | None = None,
        *,
        other_gone: tuple[str, str] | None = None,
        between: tuple[str, str] = ("", ""),
    ) -> None:
        """Keep, for the IC and the Manager of a team, what became of a team
        directly below it. It is its session's alone until the session's
        work is accepted. Safe from any thread.

        Args:
            above_id: The team whose record it is; nothing is kept for None.
            team: The team it is about, or None when it is about none.
            paths: The directories a proposal was for.
            other: The team beside it whose directory it needed, or the
                team it asked directly.
            other_gone: The id and the title of that team when it is no
                longer in the org.
            between: The two commits an accept took the team's work between.
        """
        session = session_in(above_id or "") or (team.session if team else "")
        if above_id is None or not session:
            return
        event = self.event(kind, team, detail, paths, other, other_gone)
        event = event.model_copy(update={"base": between[0], "head": between[1]})
        self.session_drafts.keep_event(session, team_key(above_id), event)
        if kind in CHANGES_THE_ORG:
            self.org_changed(session)

    def event(
        self,
        kind: TeamEventKind,
        team: SessionTeam | None,
        detail: str = "",
        paths: Sequence[str] = (),
        other: SessionTeam | None = None,
        other_gone: tuple[str, str] | None = None,
    ) -> TeamEvent:
        """An event of now about a team, kept nowhere yet."""
        return TeamEvent(
            at=utc_now(),
            kind=kind,
            # The team of the org, whatever session it was the work of.
            team_id=team.team_id if team else "",
            team=team.title if team else "",
            session=self.org().session_title(team) if team else "",
            detail=detail,
            paths=list(paths),
            other_id=other.team_id if other else (other_gone or ("", ""))[0],
            other=other.title if other else (other_gone or ("", ""))[1],
            id=uuid4().hex,
            unaccepted=True,
        )

    def land_event(self, team_id: str, event: TeamEvent) -> None:
        """Add to the org's record of a team what is no session's to keep."""
        self.team_memory().take_in(
            Landing(
                files={},
                seen={},
                reports={},
                former=[],
                forget=[],
                forget_reports=[],
                events={team_id: [event.model_copy(update={"unaccepted": False})]},
                routed=[],
            ),
            f"memory: history of {team_id}",
        )

    def history_of(self, node_id: str) -> list[TeamEvent]:
        """What became of the teams directly below a team, oldest first:
        the org's record, and for a place what its session kept since."""
        node = self.org().get(node_id)
        org = (
            self.team_memory().history(node_id)
            if node is None or self.org().is_orgs(node)
            else []
        )
        session = session_in(node_id)
        own = self.session_drafts.events(session, team_key(node_id)) if session else []
        return [*org, *own]

    def record_need(self, team: SessionTeam, path: str, how: str) -> None:
        """Keep that a team needed a file of a team beside it; nothing is
        kept when ``path`` is no such team's. The same file needed the same
        way is kept once. Safe from any thread.

        Args:
            path: Relative to the repository.
            how: `NEED_WRITE` or `NEED_FINDING`.
        """
        if team.parent_id is None:
            return
        beside = next(
            (
                other
                for other in self.org().below(team.parent_id)
                if other.id != team.id and owns(other, path)
            ),
            None,
        )
        if beside is None:
            return
        with self._keeping_needs:
            kept = False
            for event in self.history_of(team.parent_id):
                if event.kind == "set_up" and event.team_id in (
                    team.team_id,
                    beside.team_id,
                ):
                    # A new team in that place is counted from its start.
                    kept = False
                elif event.kind == "needed":
                    kept = kept or (
                        event.team_id,
                        event.other_id,
                        event.paths,
                        event.detail,
                    ) == (team.team_id, beside.team_id, [path], how)
            if not kept:
                self.record_event(team.parent_id, "needed", team, how, [path], beside)

    def routed_requests(self) -> list[RoutedRequest]:
        """What the human said that went straight to a team, oldest first:
        what the org has on record, and what the sessions open here kept
        since their work was last accepted. Safe from any thread."""
        team_memory = self.team_memory()
        head = team_memory.head()
        with self._lock:
            read = self._routed_read
        if read is None or read[0] != head:
            read = (head, team_memory.routed())
            with self._lock:
                self._routed_read = read
        own = self.session_drafts.routed()
        return sorted([*read[1], *own], key=lambda request: request.at)

    def keep_routed(
        self, kind: str, team: SessionTeam, text: str, chosen_by: str, reason: str
    ) -> None:
        """Keep a request that went straight to a team, or that the team
        said was not its own, and as an event in the record of the team
        above. Both are its session's alone until the session's work is
        accepted. Safe from any thread."""
        request = RoutedRequest.model_validate(
            {
                "at": utc_now(),
                "kind": kind,
                "team_id": team.id,
                "team": team.title,
                "text": text,
                "chosen_by": chosen_by,
                "reason": reason,
                "id": uuid4().hex,
            }
        )
        self.session_drafts.keep_routed(team.session, request)
        self.record_event(
            team.parent_id,
            "asked" if kind == "asked" else "misrouted",
            team,
            text[:ASKED_KEPT],
        )

    def _former_below(self, team_memory: RecordStore, node_id: str) -> list[FormerTeam]:
        """The records of the teams removed from below a team. They are
        read once for each state of team memory, since a context is built
        for every message."""
        head = team_memory.head()
        with self._lock:
            if self._former_read is None or self._former_read[0] != head:
                self._former_read = (head, {})
            read = self._former_read[1]
        if node_id not in read:
            read[node_id] = team_memory.former_below(node_id)
        return read[node_id]

    def former_teams_of(self, node: OrgNode) -> list[FormerTeam]:
        """What the teams removed from directly below a team left, with
        the ones its own session removed and the org does not know of yet."""
        own = [
            former
            for former in self.session_drafts.former(session_of(node))
            if former.above == team_key(node.id)
        ]
        return [*self._former_below(self.require_team_memory(), node.id), *own]

    def org_history(self, limit: int, before: str | None = None) -> OrgHistory:
        """What became of every team of the org, newest first.

        Teams needing each other's files are left out: they are counted
        for an IC and are no event of the org.

        Args:
            before: Only events before this time (ISO 8601).
        """
        team_memory = self.require_team_memory()
        # A record is the team's, whichever sessions have a place at it.
        teams: dict[str, str] = {}
        for node in (*self.org().teams(), *self.org().places()):
            teams.setdefault(team_id_of(node), node.title)
        events = [
            HistoryEvent(**event.model_dump(), above_id=team_id, above=title)
            for team_id, title in teams.items()
            for event in team_memory.history(team_id)
            if event.kind != "needed" and (before is None or event.at < before)
        ]
        # Times are kept to the second: of events in one second the later
        # recorded comes first, and a page never ends inside a second, so
        # that reading on from its last time misses none.
        for (_, team_id), kept in self.session_drafts.events_by_team().items():
            events += [
                HistoryEvent(
                    **event.model_dump(),
                    above_id=team_id,
                    above=teams.get(team_id, ""),
                )
                for event in kept
                if event.kind != "needed" and (before is None or event.at < before)
            ]
        events.reverse()
        events.sort(key=lambda event: event.at, reverse=True)
        while limit < len(events) and events[limit].at == events[limit - 1].at:
            limit += 1
        return OrgHistory(events=events[:limit], earlier=len(events) > limit)

    def keep_failure(self, intent: Intent, error: Exception) -> None:
        """Keep a step that failed where the team above and the human read
        what became of a team."""
        team = self.org().place(intent.node_id)
        if team is None:
            return
        what = STEP_NAMES.get(intent.kind, intent.kind)
        self.record_event(
            team.parent_id or team.id,
            "failed",
            team,
            f"{what}: not done. {type(error).__name__}: {error}"[:REASON_KEPT],
        )
