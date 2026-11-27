"""What a session's accepted work makes of the org: the order sessions
land in, what a Manager settles of their teams, and the landing itself."""

import asyncio
import hashlib
import json
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import NamedTuple

from pydantic import TypeAdapter

from openhands.agent_server.director.documents.store import DocumentRef
from openhands.agent_server.director.environment.records import (
    RecordStore,
    normalize_store_path,
)
from openhands.agent_server.director.intents import CloseSession, LandOrg
from openhands.agent_server.director.models import (
    DocumentChoice,
    Merge,
    MergeItem,
    OrgNodeInput,
    SessionInput,
    SessionTeam,
    TeamChoice,
    Use,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    OrgStoreError,
    is_division_lead,
    utc_now,
)
from openhands.agent_server.director.prompts import (
    files_not_written,
    merge_request,
    teams_settled,
)
from openhands.agent_server.director.run_state import (
    DECIDED,
    Intent,
)
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.message_delivery_service import (
    MessageDeliveryService,
)
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.resumable_work_service import (
    ResumableWorkService,
)
from openhands.agent_server.director.service.session_record_service import (
    SessionRecordService,
    Taken,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from openhands.agent_server.director.service.team_messaging_service import (
    TeamMessagingService,
)
from openhands.agent_server.director.service.team_summary_service import (
    TeamSummaryService,
)
from openhands.agent_server.director.service.team_type_service import TeamTypeService
from openhands.agent_server.director.split import (
    Shape,
    ShapeConflict,
    Side,
    is_inside,
    merge_shapes,
    shape_of,
)
from openhands.sdk.errors import (
    Conflict,
    Invalid,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# The id of the escalation raised when a Manager does not settle the teams
# of accepted work, and what is settled with them.
ORG_WAITS = "org-waits"


def _in_words(side: Side) -> str:
    """A team as a session or the org has it, for a Manager to read."""
    if side is None:
        return "no such team"
    above, path, *made = side
    words = f"{path or 'the repository'}, under {above}" if above else path
    if not made:
        return words
    title, writable, readonly, profile, since = json.loads(made[0])
    return (
        f"{title!r} in {words}"
        + (f", writes {', '.join(writable)}" if writable else "")
        + (f", reads {', '.join(readonly)}" if readonly else "")
        + (f", runs on {profile}" if profile else "")
        + (f", set up {since}" if since else "")
    )


class Meets(NamedTuple):
    """What the work of a team meets in the checkout above."""

    text: str
    # Names what was compared, so that an answer is held to what was shown.
    mark: str
    # What each document's file said when it was shown.
    seen: dict[str, str | None]
    items: list[MergeItem]


_ITEMS = TypeAdapter(list[MergeItem])


class Landed(NamedTuple):
    """What became of a session's work when it landed."""

    records: Taken
    # The teams that waited and are the org's as the session has them.
    taken: list[str]
    # The teams that waited and stay as the org has them.
    kept: list[str]
    # The documents that waited and are settled: each by name and folder.
    agreed: list[tuple[str, str]]
    # The documents that waited and stay the org's.
    held: list[tuple[str, str]]
    # Where what the session wrote does not hold: each document by name
    # and folder, with the text that does.
    others: list[tuple[str, str, str]]


def _same(item: MergeItem) -> tuple[str | None, ...]:
    """What a decision about something that waits was made of: it holds
    for as long as this is unchanged."""
    return (
        item.kind,
        item.document,
        item.key,
        item.session,
        item.org,
        item.moved_to,
        item.with_team,
    )


@dataclass
class SessionMergeService:
    """Makes a session's accepted work the org's.

    Sessions land in the order their work was submitted. Where a session
    changed the org's teams or wrote a document the org changed too, a
    Manager settles it first (`settle_teams`); the landing is a step of
    resumable work (`do_land_org`). Reviews call this; it calls no review.
    """

    org: Callable[[], OrgStore]
    """The org as it is loaded now."""
    team_memory: Callable[[], RecordStore]
    """The org's team memory, where there is a repository."""
    org_state: OrgStateService
    org_save: OrgSaveService
    org_settings: OrgSettingsService
    team_types: TeamTypeService
    team_history: TeamHistoryService
    team_summaries: TeamSummaryService
    team_messaging: TeamMessagingService
    message_delivery: MessageDeliveryService
    resumable_work: ResumableWorkService
    session_records: SessionRecordService
    confinement: AgentConfinementService
    """Keeps the merges that are laid out for reviewers to finish."""
    _weighing: threading.RLock = field(default_factory=threading.RLock, init=False)
    """Held while what waits of a session is read and decided, so that two
    answers are not given for the same thing."""

    def next_submission(self) -> int:
        """The place in the order of the work that is submitted now."""
        return self.org_state.run_state().count("submitted")

    def document_choices(
        self, place_id: str, choices: Sequence[DocumentChoice]
    ) -> list[dict[str, str]]:
        """What a reviewer decided about the documents of a session's
        work, as it is kept with the acceptance.

        Raises:
            Invalid: The work is not a session's, the org keeps no such
                document or settles it with the teams, a directory is not
                in the work, or a text is missing.
        """
        if not choices:
            return []
        place = self.org().place(place_id)
        if place is None or not self.org().opens_session(place):
            raise Invalid(
                "What holds of a document is decided where the work of a "
                "session is accepted. Leave that empty."
            )
        own = normalize_store_path(place.boundary.path)
        homes = {
            found.home for found in self.session_records.documents_inherited(place)
        }
        chosen = []
        for choice in choices:
            document = choice.document.strip()
            protocol = self.team_types.protocols.get(document)
            if protocol is None:
                raise Invalid(f"The org keeps no document named {document!r}")
            if protocol.waits_for_team_changes:
                raise Invalid(
                    f"What holds of {document} is not decided with the work: it "
                    "waits with the teams, for the member that settles those."
                )
            path = normalize_store_path(choice.folder)
            if not (path == own or is_inside(path, own) or path in homes):
                raise Invalid(
                    f"{choice.folder!r} is not a folder of this work "
                    f"({own or 'the repository'}), so its {document} is not yours "
                    "to decide here."
                )
            if choice.use == "replacement" and not choice.replacement.strip():
                raise Invalid(
                    f"Write the {document} of {choice.folder!r} in `replacement`."
                )
            chosen.append(
                {
                    "document": document,
                    "path": path,
                    "use": choice.use,
                    "replacement": choice.replacement,
                }
            )
        return chosen

    def take_document_choices(
        self, top: SessionTeam, chosen: list[dict[str, str]]
    ) -> None:
        """Make of the documents a session wrote what its reviewer
        decided, before they land."""
        for choice in chosen:
            file = DocumentRef(choice["path"], choice["document"]).path
            if choice["use"] == "org":
                self.session_records.undraft(top.session, file)
            elif choice["use"] == "replacement":
                self.session_records.draft(top.session, file, choice["replacement"])

    def _shapes(
        self, top: SessionTeam
    ) -> tuple[list[SessionTeam], Shape, Shape] | None:
        """The sub-teams a session has under the team it was opened at, in
        the order they hang together, what they make, and what the org has
        there now, which is nothing when the org no longer has the team.
        None when the session is gone."""
        store = self.org()
        session = store.session_of(top)
        if session is None:
            return None
        session_places = [top]
        for place in session_places:
            session_places.extend(store.below(place.id))
        teams = [team for team in [store.team(session.team_id)] if team]
        for team in teams:
            teams.extend(store.sub_teams(team.id))
        return session_places, shape_of(session_places), shape_of(teams)

    def _directory_taken(
        self,
        team_id: str,
        path: str,
        above_id: str,
        *,
        settled: bool,
        leaving: Collection[str],
    ) -> bool:
        """Whether another team of the org has the directory a team would
        take, or is beside it with a directory that holds it or lies in it.

        Args:
            settled: Its Manager put it there, or its place is new: what
                its directory holds goes under it, so only the same
                directory is in the way.
        """
        return any(
            other.id != team_id
            and other.id not in leaving
            and (
                (held := normalize_store_path(other.boundary.path)) == path
                or not settled
                and other.parent_id == above_id
                and (is_inside(path, held) or is_inside(held, path))
            )
            for other in self.org().teams()
        )

    def _write_team(
        self, team_id: str, above_id: str, place: SessionTeam, title: str
    ) -> bool:
        """Add a team to the org as a session has it, or make the org's
        team what the session has.

        Returns:
            False when the org's team could not be moved under ``above_id``.
        """
        store = self.org()
        if store.team(team_id) is None:
            store.upsert(
                OrgNodeInput(
                    id=team_id,
                    parent_id=above_id,
                    title=title,
                    boundary=place.boundary,
                    isolation=place.isolation,
                    agent_profile=place.agent_profile,
                    agent_type=place.agent_type,
                    headline=place.headline,
                    summary=place.summary,
                    leftover_documents=place.leftover_documents,
                    since=place.since,
                )
            )
            return True
        try:
            store.move(team_id, above_id)
        except OrgStoreError:
            logger.exception("Could not move team %s under %s", team_id, above_id)
            return False
        store.update_team(
            team_id,
            title=title,
            boundary=place.boundary,
            agent_profile=place.agent_profile,
            agent_type=place.agent_type,
            summary=place.summary,
            leftover_documents=place.leftover_documents,
            since=place.since,
        )
        return True

    def _put_team(
        self,
        session: SessionInput,
        places: dict[str, SessionTeam],
        team_id: str,
        under: str | None = None,
        title: str | None = None,
        leaving: Collection[str] = (),
    ) -> bool:
        """Make a team of the org what a session has at it. A team the org
        lost is never put back for it: where the team above it is not in
        the org, it goes under the nearest team that holds its directory,
        and that team's sub-teams inside its directory go under it.

        Args:
            under: The team of the org its Manager put it under instead.
            title: What its Manager called it instead.
            leaving: The teams the same landing removes from the org: a
                team set up in place of one may have a directory that
                holds the old one's, or lies in it.

        Returns:
            Whether the org has it now.
        """
        names = self.team_types.names_at(team_id)
        store = self.org()
        place = places.get(team_id)
        above = store.place(place.parent_id or "") if place else None
        if place is None or above is None:
            return False
        above_id = session.above_id if is_division_lead(above) else above.team_id
        if above_id == team_id:
            # The top team of the org.
            return store.team(team_id) is not None
        path = normalize_store_path(place.boundary.path)
        moved = under is not None or store.team(above_id) is None
        if under is not None:
            above_id = under
        elif moved:
            above_id = self.session_records.home_of(places[session.team_id], path)
        if store.team(above_id) is None or self._directory_taken(
            team_id, path, above_id, settled=moved, leaving=leaving
        ):
            return False
        if not self._write_team(team_id, above_id, place, title or place.title):
            return False
        if moved:
            for other in store.sub_teams(above_id):
                if other.id != team_id and is_inside(other.boundary.path, path):
                    store.move(other.id, team_id)
            self.team_history.record_event(
                above_id,
                "moved",
                place,
                f"its {names.manager} put it here"
                if under is not None
                else "the team it was set up under is gone",
            )
        return True

    def _review_of(self, top: SessionTeam) -> SessionTeam | None:
        """A session's Division Lead place."""
        review = self.org().place(top.parent_id or "")
        return review if review is not None and is_division_lead(review) else None

    def merge_of(self, top: SessionTeam) -> Merge | None:
        review = self._review_of(top)
        return review.merge if review else None

    def _keep_merge(self, top: SessionTeam, merge: Merge | None) -> None:
        review = self._review_of(top)
        if review is not None:
            self.org().update_place(review.id, merge=merge)

    def open_merge(self, top: SessionTeam) -> Merge | None:
        """Give the work a session submits its place in the order it goes
        into the org in. Work whose teams still wait from an earlier
        acceptance, with what is settled with them, keeps the place it has."""
        with self._weighing:
            merge = self.merge_of(top)
            if merge is None and self._review_of(top) is not None:
                seq = top.review.seq if top.review and top.review.seq else 0
                run = self.org_state.run_state()
                decided = run.marked(DECIDED).get(top.id)
                run.unmark(DECIDED, top.id)
                merge = Merge(
                    seq=seq or self.next_submission(),
                    items=_ITEMS.validate_json(decided) if decided else [],
                )
                self._keep_merge(top, merge)
            return merge

    def _in_order(self) -> list[SessionTeam]:
        """The places sessions were opened at whose work is on its way
        into the org, in the order it was submitted."""
        store = self.org()
        found: list[tuple[int, SessionTeam]] = []
        for session in store.sessions():
            places = store.session(session.id)
            merge = self.merge_of(places[0]) if places else None
            if merge is not None:
                found.append((merge.seq, places[0]))
        return [top for _, top in sorted(found, key=lambda one: one[0])]

    def _documents_coming_with(
        self,
        lost: SessionTeam,
        home_id: str,
        team_id: str,
        before: Mapping[str, MergeItem],
    ) -> list[MergeItem]:
        """What a lost team wrote of the documents that wait with the
        teams, to wait with the sub-team that would go under ``home_id``:
        each would be added to that team's own.

        Args:
            before: What waited of them when this was last worked out, by
                document. The lost team's own record may have landed
                since, and then this is where its text is kept.
        """
        home = self.org().team(home_id)
        if home is None:
            return []
        path = normalize_store_path(home.boundary.path)
        folder = normalize_store_path(lost.boundary.path)
        found = []
        for name, protocol in self.team_types.protocols.items():
            if not protocol.waits_for_team_changes:
                continue
            theirs = self.session_records.read_file(
                lost, DocumentRef(folder, name).path
            )
            was = before.get(name)
            if was is not None and not (theirs and theirs.strip()):
                theirs = was.session[len(was.org.rstrip()) :]
            if not theirs or not theirs.strip():
                continue
            org = self.team_memory().read(DocumentRef(path, name).path) or ""
            if theirs.strip() in org:
                continue
            both = f"{org.rstrip()}\n\n{theirs.strip()}\n" if org.strip() else theirs
            found.append(
                MergeItem(
                    kind="document",
                    document=name,
                    key=path,
                    gate="manager",
                    session=both,
                    org=org,
                    with_team=team_id,
                )
            )
        return found

    def _manager_items(
        self, top: SessionTeam, before: list[MergeItem]
    ) -> tuple[dict[str, Side], list[MergeItem]]:
        """What a session's work makes of the org's teams, and of the
        documents that wait with them, as the org is now.

        Args:
            before: What waited when this was last worked out. What was
                decided of something holds while it is unchanged.

        Returns:
            The teams to take from the session as they are (None to remove
            one), and what the session and the org each made something
            else of, for the Manager of its Division Lead place.
        """
        store = self.org()
        session = store.session_of(top)
        shapes = self._shapes(top)
        if session is None or shapes is None:
            return {}, []
        session_places, theirs, now = shapes
        take, clashes = merge_shapes(session.shape, theirs, now)
        places = {place.team_id: place for place in session_places}
        items, coming = self._team_items(top, places, clashes, before)
        items += self._document_items(top, coming)
        items += coming.values()
        decided = {_same(item): item for item in before if item.use}
        return take, [
            item.model_copy(
                update={
                    "use": known.use,
                    "replacement": known.replacement,
                    "under": known.under,
                    "named": known.named,
                }
            )
            if (known := decided.get(_same(item))) is not None
            else item
            for item in items
        ]

    def _team_items(
        self,
        top: SessionTeam,
        places: dict[str, SessionTeam],
        clashes: Sequence[ShapeConflict],
        before: list[MergeItem],
    ) -> tuple[list[MergeItem], dict[tuple[str, str], MergeItem]]:
        """The teams a session and the org each made something else of.

        Returns:
            One item for each, and the documents that come with a team
            whose place in the org is gone, by document and directory.
        """
        store = self.org()
        items: list[MergeItem] = []
        coming: dict[tuple[str, str], MergeItem] = {}
        came: dict[str, dict[str, MergeItem]] = {}
        for item in before:
            if item.with_team:
                came.setdefault(item.with_team, {})[item.document] = item
        for clash in clashes:
            place = places.get(clash.team)
            known = place or store.team(clash.team)
            above = store.place(place.parent_id or "") if place else None
            moved_to = None
            if (
                clash.session is not None
                and place is not None
                and above is not None
                and not is_division_lead(above)
                and store.team(above.team_id) is None
            ):
                moved_to = self.session_records.home_of(top, place.boundary.path)
                for agreed in self._documents_coming_with(
                    above, moved_to, clash.team, came.get(clash.team, {})
                ):
                    coming.setdefault((agreed.document, agreed.key), agreed)
            items.append(
                MergeItem(
                    kind="team",
                    key=clash.team,
                    gate="manager",
                    title=known.title if known else "",
                    session=_in_words(clash.session),
                    org=_in_words(clash.org),
                    moved_to=moved_to,
                )
            )
        return items, coming

    def _document_items(
        self, top: SessionTeam, coming: Collection[tuple[str, str]]
    ) -> list[MergeItem]:
        """The documents a session and the org each wrote something else
        in, among those that wait with the teams. ``coming`` are settled
        with their team and left out."""
        items: list[MergeItem] = []
        memory = self.team_memory()
        for file, (text, over) in sorted(
            self.session_records.drafts(top.session).items()
        ):
            path = file.rpartition("/")[0]
            name = DocumentRef.at(file).name
            if not self._waits_for_team_changes(file) or (name, path) in coming:
                continue
            has = memory.read(file)
            if has != over and has != text:
                items.append(
                    MergeItem(
                        kind="document",
                        document=name,
                        key=path,
                        gate="manager",
                        session=text or "",
                        org=has or "",
                    )
                )
        return items

    def org_turn(self) -> None:
        """Move on what the sessions' work makes of the org's teams and of
        what is settled with them, strictly in the order the work was
        submitted: only the first in order is worked out, asked about and
        landed. Safe from any thread, and to call again."""
        with self._weighing:
            for index, top in enumerate(self._in_order()):
                merge = self.merge_of(top)
                if merge is None:
                    continue
                if index:
                    if not merge.org_queued:
                        self._keep_merge(
                            top, merge.model_copy(update={"org_queued": True})
                        )
                    continue
                _, items = self._manager_items(top, merge.items)
                merge = merge.model_copy(
                    update={
                        "items": [
                            *(item for item in merge.items if item.gate == "ic"),
                            *items,
                        ],
                        "org_queued": False,
                    }
                )
                self._keep_merge(top, merge)
                if any(item.use is None for item in items):
                    self._ask_manager(top, merge)
                elif merge.code_landed and all(
                    intent.kind != "land_org" or intent.node_id != top.id
                    for intent in self.org_state.run_state().open_intents()
                ):
                    self.resumable_work.begin(top.id, LandOrg())

    def _ask_manager(self, top: SessionTeam, merge: Merge) -> None:
        """Ask the Manager of a session's Division Lead place to
        settle what waits there, once for each state of it."""
        review = self._review_of(top)
        waiting = [item for item in merge.items if item.gate == "manager"]
        mark = hashlib.sha256(
            json.dumps([_same(item) for item in waiting if item.use is None]).encode()
        ).hexdigest()
        if review is None or merge.asked == mark:
            return
        self._keep_merge(
            top,
            merge.model_copy(
                update={
                    "asked": mark,
                    "asked_at": self.org_state.clock().isoformat(),
                    "reminded": False,
                }
            ),
        )
        self.team_messaging.tell_manager(
            review.id,
            merge_request(
                top,
                self.org().session_title(top),
                waiting,
                merge.code_landed,
                names=self.team_types.team(top).names,
            ),
            division_lead=True,
        )

    def check_managers(self) -> None:
        """Move on what waits for a Manager that does not answer. It
        holds back what every session that submitted later made of
        the org's teams, so it is asked once more after the time a team
        may show no progress, and after as long again the human is asked:
        their answer lets the org's stand wherever nothing was decided."""
        limit = self.org_settings.settings.soft_check_in_minutes
        if not limit:
            return
        with self._weighing:
            order = self._in_order()
            top = order[0] if order else None
            merge = self.merge_of(top) if top else None
            review = self._review_of(top) if top else None
            if top is None or merge is None or review is None or not merge.asked_at:
                return
            waiting = [item for item in merge.items if item.gate == "manager"]
            if all(item.use for item in waiting):
                return
            now = self.org_state.clock()
            if now - datetime.fromisoformat(merge.asked_at) < timedelta(minutes=limit):
                return
            if not merge.reminded:
                self._keep_merge(
                    top,
                    merge.model_copy(
                        update={"reminded": True, "asked_at": now.isoformat()}
                    ),
                )
                self.team_messaging.tell_manager(
                    review.id,
                    "This still waits for you, and with it what every session "
                    "that submitted later made of its teams.\n\n"
                    + merge_request(
                        top,
                        self.org().session_title(top),
                        waiting,
                        merge.code_landed,
                        names=self.team_types.team(top).names,
                    ),
                    division_lead=True,
                )
                return
            if not merge.code_landed or any(e.id == ORG_WAITS for e in top.escalations):
                return
            self.org().raise_escalation(
                top.id,
                ORG_WAITS,
                kind="critical_issue",
                title=f"{self.team_types.team(top).names.A_manager} has not settled "
                "the teams of accepted work",
                detail=(
                    f"The code of {top.title!r} is accepted and merged. What "
                    "it makes of the org's teams waits for the "
                    f"{self.team_types.team(top).names.manager} that reviews it, "
                    "which has not answered, and the "
                    "teams of sessions that submitted later "
                    "wait behind it. Answer this to go on without the "
                    f"{self.team_types.team(top).names.manager}: the org's stay "
                    "wherever it has not decided, and "
                    "what the session wrote for a team that is not taken is "
                    "kept in that team's record."
                ),
                at=utc_now(),
            )
        self.org_save.save_soon()

    def settle_teams(
        self,
        review_id: str,
        teams: list[TeamChoice],
        note: str,
        choices: Sequence[DocumentChoice] = (),
    ) -> str:
        """Settle what a session's work makes of the org's teams, and of
        the documents settled with them, that waits at its Division Lead
        place. A team
        becomes what the session has unless ``teams`` says the org's
        stays; taken from the session, it may be put under another team
        of the org or called otherwise. Of a document that waits, the
        session's holds unless ``choices`` says otherwise. Nothing is the
        org's by this alone: it lands when the session's code is accepted
        and it is its turn. Safe from any thread.

        Raises:
            Conflict: Nothing waits there, or it changed since the Manager
                was shown it.
            Invalid: A team or a directory named does not wait, or a team
                cannot be put where it was asked to.
        """
        names = self.team_types.names_at(review_id)
        store = self.org()
        with self._weighing:
            review = store.place(review_id)
            session = store.session_of(review) if review else None
            merge = review.merge if review else None
            waiting = [
                item
                for item in (merge.items if merge else [])
                if item.gate == "manager"
            ]
            if merge is None or session is None or merge.org_queued or not waiting:
                raise Conflict("Nothing of a session waits for you to settle")
            top = store.session(session.id)[0]
            self._check_teams_wait(waiting, teams)
            placed = {choice.team: choice for choice in teams}
            for choice in teams:
                self._check_placement(top, choice)
            chosen = self._document_choices(waiting, choices)
            others = [item for item in merge.items if item.gate == "ic"]
            _, now = self._manager_items(top, merge.items)
            if {_same(item) for item in now} != {_same(item) for item in waiting}:
                # What it decided on is not what the org has any more.
                shown = merge.model_copy(update={"items": [*others, *now]})
                self._keep_merge(top, shown)
                raise Conflict(
                    "What waits changed since you were shown it. Nothing is "
                    "settled yet.\n\n"
                    + merge_request(
                        top,
                        store.session_title(top),
                        now,
                        merge.code_landed,
                        names=names,
                    )
                )
            settled = self._settled(waiting, placed, chosen)
            self._keep_merge(
                top,
                merge.model_copy(update={"items": [*others, *settled], "note": note}),
            )
            landed = merge.code_landed
        self.org_turn()
        self.org_save.save_soon()
        return "Settled. " + (
            "It is the org's as soon as it is its turn."
            if landed
            else f"It becomes the org's when the {names.reviewer} that reviews the "
            "session's "
            "work has accepted it."
        )

    def _check_teams_wait(
        self, waiting: list[MergeItem], teams: list[TeamChoice]
    ) -> None:
        """Hold the teams a Manager named to those that wait.

        Raises:
            Invalid: A team named does not wait.
        """
        named = [item.key for item in waiting if item.kind == "team"]
        unknown = [choice.team for choice in teams if choice.team not in named]
        if unknown:
            raise Invalid(
                f"Not waiting: {', '.join(unknown)}. The teams that wait: "
                f"{', '.join(named) or 'none'}"
            )

    def _document_choices(
        self, waiting: list[MergeItem], choices: Sequence[DocumentChoice]
    ) -> dict[tuple[str, str], DocumentChoice]:
        """What a Manager chose for each document that waits with the
        teams, by document and folder.

        Raises:
            Invalid: A folder named has no such document waiting.
        """
        agreed = [
            (item.document, item.key) for item in waiting if item.waits_with_teams
        ]
        chosen = {
            (choice.document, normalize_store_path(choice.folder)): choice
            for choice in choices
        }
        for document in {document for document, _ in chosen}:
            unknown = [
                path
                for named, path in chosen
                if named == document and (named, path) not in agreed
            ]
            wait = [path for named, path in agreed if named == document]
            if unknown:
                raise Invalid(
                    f"No {document} wait for: {', '.join(unknown)}. The folders "
                    f"whose {document} wait: {', '.join(wait) or 'none'}"
                )
        return chosen

    def _settled(
        self,
        waiting: list[MergeItem],
        placed: dict[str, TeamChoice],
        chosen: dict[tuple[str, str], DocumentChoice],
    ) -> list[MergeItem]:
        """What waited, with what the Manager settled of each: the
        session's holds where nothing else was said."""
        settled = []
        for item in waiting:
            choice = (
                chosen.get((item.document, item.key)) if item.waits_with_teams else None
            )
            place = placed.get(item.key) if item.kind == "team" else None
            use: Use = "session"
            if place is not None:
                use = place.use
            elif choice is not None:
                use = choice.use
            settled.append(
                item.model_copy(
                    update={
                        "use": use,
                        "replacement": choice.replacement
                        if choice and use == "replacement"
                        else "",
                        "under": place.under if place else None,
                        "named": place.title if place else None,
                    }
                )
            )
        return settled

    def _check_placement(self, top: SessionTeam, choice: TeamChoice) -> None:
        """Hold where a Manager puts a team it takes from a session, and
        what it calls it, to what the org allows.

        Raises:
            Invalid: It cannot be put there or called that.
        """
        store = self.org()
        if choice.under is None and choice.title is None:
            return
        if choice.use == "org":
            raise Invalid(
                f"Team {choice.team} stays as the org has it, so it cannot be "
                "put or called otherwise. Take it from the session for that."
            )
        if choice.title is not None and not choice.title:
            raise Invalid(f"Say what team {choice.team} is to be called.")
        if choice.under is None:
            return
        shapes = self._shapes(top)
        place = next(
            (
                place
                for place in (shapes[0] if shapes else [])
                if place.team_id == choice.team
            ),
            None,
        )
        home = store.team(choice.under)
        if place is None or home is None or choice.under == choice.team:
            raise Invalid(
                f"{choice.under} is not a team of the org that team "
                f"{choice.team} can go under."
            )
        if any(team.id == choice.team for team in store.teams_above(home.id)):
            raise Invalid(
                f"Team {choice.under} is below team {choice.team} in the org, "
                "so that one cannot go under it."
            )
        path = normalize_store_path(place.boundary.path)
        if not is_inside(path, normalize_store_path(home.boundary.path)):
            raise Invalid(
                f"The folder of team {choice.team} ({path}) is not inside "
                f"that of team {choice.under} "
                f"({home.boundary.path or 'the repository'}), so it cannot go "
                "under it."
            )
        # Beside a team that holds its folder, both would own it.
        holder = next(
            (
                team
                for team in store.sub_teams(home.id)
                if team.id != choice.team
                and is_inside(path, normalize_store_path(team.boundary.path))
            ),
            None,
        )
        if holder is not None:
            raise Invalid(
                f"The folder of team {choice.team} ({path}) is inside that of "
                f"team {holder.id}, which is under team {choice.under}, so it "
                f"goes under team {holder.id} or one below it."
            )
        level = len(store.teams_above(home.id)) + 2
        max_depth = self.org_settings.settings.max_depth
        if level > max_depth:
            raise Invalid(
                f"An org may have at most {max_depth} levels of "
                f"teams, and under team {choice.under} it would be level {level}."
            )

    def _land(
        self,
        top: SessionTeam,
        seen: dict[str, str | None] | None,
        org: tuple[dict[str, Side], list[MergeItem]] | None,
    ) -> Landed:
        """Make what a session kept to itself the org's, now that its code
        is in the repository's checkout.

        Args:
            seen: The documents the reviewer was shown, by file.
            org: What its work makes of the org's teams and of the
                documents that wait with them, all of it settled; None
                when that waits. Then only the documents that land with
                its work and the records of the teams the org has land,
                and the rest stays the session's.
        """
        # No save of the tree comes between what this changes of the org
        # and the one commit that takes it and team memory together.
        with self.org_save.lock:
            return self._land_all(top, seen, org)

    def _land_all(
        self,
        top: SessionTeam,
        seen: dict[str, str | None] | None,
        org: tuple[dict[str, Side], list[MergeItem]] | None,
    ) -> Landed:
        """Make what a session kept the org's: its records alone while its
        teams wait (``org`` is None), or its teams, the documents settled
        with them and its records together. A landing that fails leaves
        the tree and the merge as they were."""
        store = self.org()
        session = store.session_of(top)
        shapes = self._shapes(top)
        merge = self.merge_of(top)
        if session is None or shapes is None or merge is None:
            return Landed(Taken([], []), [], [], [], [], [])
        if org is None:
            return Landed(self._land_records(top, merge, seen), [], [], [], [], [])
        # What the tree and the merge were, for a landing that fails: the
        # tree never moves without the team memory that goes with it.
        tree = [
            OrgNodeInput.model_validate(team.model_dump(exclude={"child_ids"}))
            for team in store.teams()
        ]
        shape = list(session.shape)
        try:
            take, items = org
            session_places, theirs, _ = shapes
            places = {place.team_id: place for place in session_places}
            decided = {item.key: item for item in items if item.kind == "team"}
            taken, kept = self._take_teams(session, places, take, decided)
            # What it has now is what it is next compared by: a team the org
            # keeps otherwise is not brought up again unless the session changes
            # it once more.
            store.set_shape(session.id, theirs)
            texts, agreed, held, others = self._take_documents(top, items, taken)
            self._keep_merge(top, None)
            records = self.session_records.land(top, seen, chosen=texts, tree=True)
        except BaseException:
            store.replace_all(tree)
            store.set_shape(session.id, shape)
            self._keep_merge(top, merge)
            raise
        return Landed(records, taken, kept, agreed, held, others)

    def _land_records(
        self, top: SessionTeam, merge: Merge, seen: dict[str, str | None] | None
    ) -> Taken:
        """Land a session's records while what it makes of the org's teams
        still waits for the Manager: those teams' records and the documents
        settled with them are held back."""
        take, items = self._manager_items(top, [])
        waits = {*take, *(item.key for item in items if item.kind == "team")}
        agreements = {
            file
            for file in self.session_records.drafts(top.session)
            if self._waits_for_team_changes(file)
        }
        self._keep_merge(
            top,
            merge.model_copy(
                update={
                    "code_landed": True,
                    "items": [i for i in merge.items if i.gate == "manager"],
                }
            ),
        )
        return self.session_records.land(top, seen, waits=waits, held=agreements)

    def _take_teams(
        self,
        session: SessionInput,
        places: dict[str, SessionTeam],
        take: dict[str, Side],
        decided: dict[str, MergeItem],
    ) -> tuple[list[str], list[str]]:
        """Make the org's teams what the session has, where that was
        settled or needs no settling.

        Returns:
            The teams settled that were taken from the session, and those
            the org keeps as it has them.
        """
        store = self.org()
        taken: list[str] = []
        kept = [team for team, item in decided.items() if item.use != "session"]
        leaving = [
            *(team for team, side in take.items() if side is None),
            *(
                team
                for team, item in decided.items()
                if item.use == "session" and team not in places
            ),
        ]
        # Those above before those below, and what goes after what moved.
        for team in places:
            settled = team in decided and decided[team].use == "session"
            if take.get(team) is not None or settled:
                item = decided.get(team)
                done = self._put_team(
                    session,
                    places,
                    team,
                    item.under if item and settled else None,
                    item.named if item and settled else None,
                    leaving,
                )
                if settled:
                    (taken if done else kept).append(team)
        for team in leaving:
            store.remove(team)
            if team in decided:
                taken.append(team)
        return taken, kept

    def _take_documents(
        self, top: SessionTeam, items: list[MergeItem], taken: list[str]
    ) -> tuple[
        dict[str, str | None],
        list[tuple[str, str]],
        list[tuple[str, str]],
        list[tuple[str, str, str]],
    ]:
        """Work out what becomes of each document that waited with the
        teams: the session's or a replacement lands, the org's stays.

        Returns:
            The text to land for each file (None removes it), the documents
            that land, those held back, and for those whose writer has to
            be told, the text that holds instead of what it wrote.
        """
        texts: dict[str, str | None] = {}
        agreed: list[tuple[str, str]] = []
        held: list[tuple[str, str]] = []
        others: list[tuple[str, str, str]] = []
        for item in items:
            if not item.waits_with_teams:
                continue
            file = DocumentRef(item.key, item.document).path
            if item.use == "org" or (
                item.with_team is not None and item.with_team not in taken
            ):
                self.session_records.undraft(top.session, file)
                held.append((item.document, item.key))
                others.append((item.document, item.key, item.org))
                continue
            text = item.replacement if item.use == "replacement" else item.session
            texts[file] = f"{text.strip()}\n" if text.strip() else None
            agreed.append((item.document, item.key))
            if item.use == "replacement":
                others.append((item.document, item.key, text))
        return texts, agreed, held, others

    def land_accepted(
        self, top: SessionTeam, seen: dict[str, str | None] | None
    ) -> None:
        """Land the work of a session whose code was just merged: with
        what it makes of the org's teams, and what is settled with them,
        when that is settled and it is its turn, and without it otherwise."""
        with self._weighing:
            merge = self.open_merge(top)
            order = self._in_order()
            org = None
            if merge is not None and order and order[0].id == top.id:
                take, items = self._manager_items(top, merge.items)
                if all(item.use for item in items):
                    org = (take, items)
            landed = self._land(top, seen, org)
        self._after_landing(top, landed, merge.note if merge and org else "")

    def _after_landing(self, top: SessionTeam, landed: Landed, note: str) -> None:
        for folder, role in landed.records.grown:
            self.team_summaries.reports_grew(folder, role)
        now = self.org().place(top.id)
        if (
            self.merge_of(top) is None
            and now is not None
            and any(e.id == ORG_WAITS for e in now.escalations)
        ):
            # Its Manager answered after all.
            self.org().settle_escalation(top.id, ORG_WAITS)
        review = self._review_of(top)
        if review is not None and landed.records.left:
            self.message_delivery.notify(
                self.team_types.reviewer_conversation(review),
                files_not_written(top.title, landed.records.left),
            )
        if review is not None and (
            landed.taken or landed.kept or landed.agreed or landed.held
        ):
            self.team_messaging.tell_writer(
                top.id,
                teams_settled(
                    review.title,
                    landed.taken,
                    landed.kept,
                    landed.agreed,
                    landed.held,
                    landed.others,
                    names=self.team_types.team(top).names,
                )
                + (f"\n\n{note}" if note else ""),
            )
        self._record_not_taken(top, landed)
        self.org_turn()
        self._close_unattended(top)

    def _close_unattended(self, top: SessionTeam) -> None:
        """Close a session no human opened once all of its work is the
        org's: nobody is there to close it, and left open it counts against
        the sessions the org may have. The org's settings can keep it."""
        session = self.org().session_of(top)
        if (
            session is None
            or not session.source.unattended
            or not self.org_settings.closes_unattended
            # What it makes of the org's teams still waits to be settled.
            or self.merge_of(top) is not None
        ):
            return
        self.resumable_work.begin(top.id, CloseSession())

    def _record_not_taken(self, top: SessionTeam, landed: Landed) -> None:
        """Keep on record what a session wrote of a document that the org
        did not take, also when the session is closed and nobody is told."""
        for name, directory, _ in landed.others:
            self.team_history.land_event(
                self.session_records.home_of(top, directory),
                self.team_history.event(
                    "not_taken",
                    top,
                    f"the {name} it wrote for "
                    f"{directory or 'the repository'}: its merged code was "
                    "written to it",
                ),
            )

    async def do_land_org(self, intent: Intent) -> None:
        """Land what a session's accepted work makes of the org's teams,
        and what is settled with them, when it is its turn and all of it is settled."""
        top = self.org().place(intent.node_id)
        if top is None:
            return
        landed = await asyncio.to_thread(
            self._land_org, top, intent.of(LandOrg).org_holds
        )
        if landed is not None:
            self._after_landing(top, *landed)

    def _land_org(
        self, top: SessionTeam, org_holds: bool = False
    ) -> tuple[Landed, str] | None:
        """
        Args:
            org_holds: The human went on without the Manager: the org's
                stay wherever nothing was decided.
        """
        with self._weighing:
            merge = self.merge_of(top)
            order = self._in_order()
            if (
                merge is None
                or not merge.code_landed
                or not order
                or order[0].id != top.id
            ):
                return None
            take, items = self._manager_items(top, merge.items)
            if org_holds:
                items = [
                    item if item.use else item.model_copy(update={"use": "org"})
                    for item in items
                ]
            if any(item.use is None for item in items):
                return None
            return self._land(top, None, (take, items)), merge.note

    def settle_on_close(self, top: SessionTeam) -> None:
        """Settle what waits of a session that is being closed: of work
        whose code is in the repository, the org's teams and what is
        settled with them stay wherever nobody decided otherwise, and what
        the session wrote lands where the org has the team and is that
        team's record where it has not."""
        with self._weighing:
            merge = self.merge_of(top)
            if merge is None:
                return
            if not merge.code_landed:
                self._keep_merge(top, None)
                return
            take, items = self._manager_items(top, merge.items)
            landed = self._land(
                top,
                None,
                (
                    take,
                    [
                        item if item.use else item.model_copy(update={"use": "org"})
                        for item in items
                    ],
                ),
            )
        for folder, role in landed.records.grown:
            self.team_summaries.reports_grew(folder, role)
        self._record_not_taken(top, landed)

    def sent_back(self, top: SessionTeam) -> None:
        """The work of a session was sent back: it has no place in the
        order any more, unless teams of work accepted before still wait."""
        with self._weighing:
            merge = self.merge_of(top)
            review = self._review_of(top)
            if merge is None or review is None or merge.code_landed:
                return
            self._keep_merge(top, None)
            decided = [
                item for item in merge.items if item.gate == "manager" and item.use
            ]
            if decided:
                # They hold again for whatever is unchanged when it comes back.
                self.org_state.run_state().mark(
                    DECIDED, top.id, _ITEMS.dump_json(decided).decode()
                )
            if merge.asked and any(
                item.gate == "manager" and item.use is None for item in merge.items
            ):
                self.team_messaging.tell_manager(
                    review.id,
                    f"The work of {top.title!r} was sent back to its team. "
                    "What waited of it for you waits no longer. What you "
                    "decided of it holds again for whatever is unchanged "
                    "when it is submitted again.",
                    division_lead=True,
                )
        self.org_turn()

    def _waits_for_team_changes(self, file: str) -> bool:
        """Whether what a session wrote in a file of team memory becomes
        the org's only with its changes to the teams: its document says."""
        protocol = self.org_settings.team_documents().at(file)
        return protocol is not None and protocol.waits_for_team_changes

    def documents_waiting(self, top: SessionTeam) -> list[tuple[str, str, str]]:
        """What a session wrote of the documents that wait with its teams
        and is not the org's: each by name and folder, with the session's
        text."""
        memory = self.team_memory()
        return [
            (DocumentRef.at(file).name, file.rpartition("/")[0], text or "")
            for file, (text, _) in sorted(
                self.session_records.drafts(top.session).items()
            )
            if self._waits_for_team_changes(file) and memory.read(file) != text
        ]

    def show_reviewer_items(self, top: SessionTeam, meets: Meets | None) -> None:
        """Keep, with what waits of a session's work, what its reviewer
        has to reconcile."""
        with self._weighing:
            merge = self.merge_of(top)
            if merge is None:
                return
            laid_out = self.confinement.merge_checkout(top.id)
            listed = {item.key for item in (meets.items if meets else [])}
            items = [
                *(meets.items if meets else []),
                # The files of a merge that is laid out for the reviewer.
                *(
                    MergeItem(kind="code", key=path, gate="ic")
                    for path in (laid_out.files if laid_out else [])
                    if path not in listed
                ),
                *(item for item in merge.items if item.gate == "manager"),
            ]
            if items != merge.items:
                self._keep_merge(top, merge.model_copy(update={"items": items}))
