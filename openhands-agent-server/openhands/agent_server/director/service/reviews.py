"""A team's work on its way up: submitted, sent back, accepted, merged."""

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple
from uuid import UUID

from pydantic import TypeAdapter

from openhands.agent_server.director import bugbot
from openhands.agent_server.director.boundary import owns
from openhands.agent_server.director.documents.store import DocumentRef
from openhands.agent_server.director.git import (
    GitError,
    branch_of,
    git,
    head_of,
    holds_commit,
    pathspec,
    ref_stamp,
    save_and_discard,
    uncommitted,
)
from openhands.agent_server.director.git_store import (
    normalize_store_path,
)
from openhands.agent_server.director.intents import Accept, LandOrg, RequestReview
from openhands.agent_server.director.merge_back import (
    MergeBack,
    merge_back,
    merge_message,
    take_merge,
)
from openhands.agent_server.director.merge_checkout import (
    MergeCheckout,
    Unfinished,
    finish,
    prepare,
)
from openhands.agent_server.director.models import (
    DocumentChoice,
    Merge,
    MergeItem,
    OrgNodeInput,
    Review,
    ReviewFinding,
    SessionInput,
    SessionTeam,
    TeamChoice,
    Use,
)
from openhands.agent_server.director.org_store import (
    OrgStoreError,
    is_division_lead,
    utc_now,
)
from openhands.agent_server.director.ownership import can_merge_in_memory, conflicts
from openhands.agent_server.director.prompts import (
    NEED_FINDING,
    accept_in_order,
    changes_requested,
    files_not_written,
    merge_prepared,
    merge_request,
    put_aside_notice,
    reconcile_request,
    review_request,
    session_work,
    still_undecided,
    teams_settled,
)
from openhands.agent_server.director.reconcile import changes_of
from openhands.agent_server.director.run_state import (
    CLEARING,
    DECIDED,
    HELD,
    MERGING,
    SHOWN,
    TURN,
    Intent,
)
from openhands.agent_server.director.service.records import (
    Taken,
)
from openhands.agent_server.director.service.requests import Requests
from openhands.agent_server.director.service.state import (
    _LANE,
    _REST_CHECK_SECONDS,
    _REST_CHECKS,
    ASIDE_REFS,
    ASKED_KEPT,
    BASE_LANE,
    MERGE_WAITS,
    ORG_WAITS,
    _branch_of,
    checkout_root,
)
from openhands.agent_server.director.split import (
    Shape,
    Side,
    is_inside,
    merge_shapes,
    shape_of,
)
from openhands.sdk.errors import (
    Conflict,
    Invalid,
    NotFound,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


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


def _submitted(team: SessionTeam) -> tuple[int, str]:
    """When a team's work was submitted, to put work in order by."""
    return (team.review.seq, team.review.at) if team.review else (0, "")


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


class Reviews(Requests):
    """Carries a team's work through its review."""

    def request_review(self, place_id: str) -> None:
        """Have the work a team submitted reviewed. Safe from any thread."""
        place = self.org.place(place_id)
        if place is not None and self.org.opens_session(place):
            self._open_merge(place)
        self._begin(place_id, RequestReview())

    def rest(self, place_id: str) -> None:
        """End the turn of a team's IC after the step it is in: it has
        handed its work over and acts again when something is sent to it.
        Safe from any thread."""
        if self._loop is not None:
            self._enqueue(lambda: self._rest(place_id), place_id)

    async def _rest(self, place_id: str) -> None:
        place = self.org.place(place_id)
        conversation_id = self.writer_conversation(place) if place else None
        if place is None or conversation_id is None or self._paused is not None:
            return
        launcher = self._require_launcher()
        writer = self.team(place).writer
        if any(agent.role == writer and agent.kind == "acp" for agent in place.agents):
            # It runs its own turn, and the tool's answer tells it to end
            # it. Only one that carries on regardless is cut short, which
            # costs it the step it is in.
            await self._until_at_rest(conversation_id)
            if conversation_id in self._running:
                await launcher.stop(UUID(conversation_id))
        else:
            await launcher.pause(UUID(conversation_id))
        # This runs in the lane of the team's messages: nothing is sent to
        # the agent before it has stopped, or it would lie there unread.
        await self._until_at_rest(conversation_id)

    async def _until_at_rest(self, conversation_id: str) -> None:
        for _ in range(_REST_CHECKS):
            if conversation_id not in self._running:
                return
            await asyncio.sleep(_REST_CHECK_SECONDS)

    def request_changes(self, place_id: str, findings: list[ReviewFinding]) -> None:
        """Send a team's work back to it with what must change.

        Safe from any thread.

        Raises:
            NotFound: The team does not exist.
            Conflict: None of its work waits for a review.
        """
        names = self.names_at(place_id)
        store = self.org
        place = store.place(place_id)
        if place is None:
            raise NotFound(f"Team {place_id} does not exist")
        if place.review is None or place.review.state != "pending":
            raise Conflict(f"Team {place_id} has no work waiting for a review")
        review = place.review.model_copy(
            update={"state": "changes_requested", "findings": findings, "at": utc_now()}
        )
        above = store.place(place.parent_id) if place.parent_id else None
        store.set_review(place_id, review)
        self._drop_merge_checkout(place_id)
        self._tell_turns()
        self.progressed(place_id, by_above=True)
        self.record_event(
            place.parent_id,
            "sent_back",
            place,
            f"round {review.round}, {len(findings)} to change",
        )
        for finding in findings:
            # A finding of the runtime names several files in one.
            for path in finding.path.split(", "):
                if path.strip():
                    self.record_need(place, path.strip(), NEED_FINDING)
        limit = self.settings.manager_check_ic_setbacks
        if limit and place.parent_id is not None and review.round % limit == 0:
            with self._counting:
                counted = self._counted(place.parent_id)
            self._wake_manager(
                place.parent_id,
                f"your {names.reviewer} has sent the work of {place.title!r} back "
                f"{review.round} times.",
                counted,
                team=place,
            )
        self.notify(
            self.writer_conversation(place),
            changes_requested(
                above or place,
                findings,
                of_a_session=store.opens_session(place),
                names=names,
            ),
        )
        if store.opens_session(place):
            self._sent_back(place)

    def release_reviews(self, place_id: str | None) -> None:
        """Hand a team's IC the reviews its sub-teams wait for (see
        :meth:`_release_reviews`). Safe from any thread."""
        if place_id is not None and self._loop is not None:
            self._enqueue(lambda: self._released(place_id), f"review:{place_id}")

    async def _released(self, place_id: str) -> None:
        self._release_reviews(place_id)

    def _release_reviews(self, place_id: str | None) -> None:
        """Hand a team's IC the reviews its sub-teams wait for.

        Each goes as soon as its team has submitted: no team waits for
        the teams beside it to be reviewed. What must be whole before work
        moves up is checked where the IC submits its own
        (``agent_actions.submit_review``). The IC may end its turn without
        deciding; the next request names what still waits for it.
        """
        store = self.org
        node = store.place(place_id) if place_id else None
        if node is None or self._paused is not None:
            return
        run = self._runs()
        held = run.marked(HELD)
        teams = store.below(node.id)
        waiting: list[SessionTeam] = []
        for team in teams:
            if team.id not in held:
                continue
            if team.review is not None and team.review.state == "pending":
                waiting.append(team)
            else:
                # Decided on, or taken back, some other way meanwhile.
                run.unmark(HELD, team.id)
        if not waiting:
            return
        requests = []
        # In the order the work was submitted, which is the order it is
        # accepted in.
        for team in sorted(waiting, key=_submitted):
            if team.review is not None:
                requests.append(
                    (
                        session_work(
                            team,
                            store.session_title(team),
                            self._asked_for(team),
                            self._documents_waiting(team),
                            names=self.team(team).names,
                            kept=self.kept_documents(node),
                        )
                        if store.opens_session(team)
                        else ""
                    )
                    + review_request(
                        team,
                        team.review,
                        self._so_far(team),
                        self.asked_of(team),
                        self.review_instructions(self.type_of(team)),
                        self.kept_documents(node).read_by_sub_teams,
                    )
                )
        # What it was asked before and has not decided: it may have waited
        # for this work to compare the two.
        asked_before = [
            team
            for team in teams
            if team not in waiting
            and team.review is not None
            and team.review.state == "pending"
        ]
        if asked_before:
            requests.append(still_undecided(sorted(asked_before, key=_submitted)))
        self.tell(
            node.id,
            self.team(node).reviewer,
            "\n\n---\n\n".join(requests),
            division_lead=True,
        )
        # Only now: a server that ends before the request is kept asks again.
        for team in waiting:
            run.unmark(HELD, team.id)

    def _asked_for(self, team: SessionTeam) -> str:
        """What the session ``team`` belongs to was opened for."""
        session = self.org.session_of(team)
        return session.request if session else ""

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
        place = self.org.place(place_id)
        if place is None or not self.org.opens_session(place):
            raise Invalid(
                "What holds of a document is decided where the work of a "
                "session is accepted. Leave that empty."
            )
        own = normalize_store_path(place.boundary.path)
        homes = {found.home for found in self.documents_inherited(place)}
        chosen = []
        for choice in choices:
            document = choice.document.strip()
            protocol = self.protocols.get(document)
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

    def _take_document_choices(
        self, top: SessionTeam, chosen: list[dict[str, str]]
    ) -> None:
        """Make of the documents a session wrote what its reviewer
        decided, before they land."""
        for choice in chosen:
            file = DocumentRef(choice["path"], choice["document"]).path
            if choice["use"] == "org":
                self._undraft(top.session, file)
            elif choice["use"] == "replacement":
                self._draft(top.session, file, choice["replacement"])

    def accept_review(
        self,
        place_id: str,
        needed_above: str,
        reconciled: str = "",
        choices: list[dict[str, str]] | None = None,
    ) -> None:
        """Take a team's reviewed work into the team above. Safe from any
        thread.

        Args:
            needed_above: Why a team further up waits on the work; '' when
                none does.
            reconciled: What the reviewer decided about what the work meets
                in the checkout above; '' when it was asked nothing.
            choices: What it decided about the documents of a session's work.
        """
        place = self.org.place(place_id)
        self._begin(
            place_id,
            Accept(
                needed_above=needed_above,
                reconciled=reconciled,
                choices=choices or [],
                # The work of a session goes into what the org shares.
                base=place is not None and self.org.opens_session(place),
            ),
        )

    def _shapes(
        self, top: SessionTeam
    ) -> tuple[list[SessionTeam], Shape, Shape] | None:
        """The sub-teams a session has under the team it was opened at, in
        the order they hang together, what they make, and what the org has
        there now, which is nothing when the org no longer has the team.
        None when the session is gone."""
        store = self.org
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

    def _put_team(
        self,
        session: SessionInput,
        places: dict[str, SessionTeam],
        team_id: str,
        under: str | None = None,
        title: str | None = None,
    ) -> bool:
        """Make a team of the org what a session has at it. A team the org
        lost is never put back for it: where the team above it is not in
        the org, it goes under the nearest team that holds its directory,
        and that team's sub-teams inside its directory go under it.

        Args:
            under: The team of the org its Manager put it under instead.
            title: What its Manager called it instead.

        Returns:
            Whether the org has it now.
        """
        names = self.names_at(team_id)
        store = self.org
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
            above_id = self._home_of(places[session.team_id], path)
        home = store.team(above_id)
        if home is None or any(
            other.id != team_id
            and (
                (held := normalize_store_path(other.boundary.path)) == path
                # Beside a team one of whose directories holds the other's.
                # Where its Manager put it, or its place is new, that is
                # settled: what it holds goes under it.
                or not moved
                and other.parent_id == above_id
                and (is_inside(path, held) or is_inside(held, path))
            )
            for other in store.teams()
        ):
            return False
        if store.team(team_id) is None:
            store.upsert(
                OrgNodeInput(
                    id=team_id,
                    parent_id=above_id,
                    title=title or place.title,
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
        else:
            try:
                store.move(team_id, above_id)
            except OrgStoreError:
                logger.exception("Could not move team %s under %s", team_id, above_id)
                return False
            store.update_team(
                team_id,
                title=title or place.title,
                boundary=place.boundary,
                agent_profile=place.agent_profile,
                agent_type=place.agent_type,
                summary=place.summary,
                leftover_documents=place.leftover_documents,
                since=place.since,
            )
        if moved:
            for other in store.sub_teams(above_id):
                if other.id != team_id and is_inside(other.boundary.path, path):
                    store.move(other.id, team_id)
            self.record_event(
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
        review = self.org.place(top.parent_id or "")
        return review if review is not None and is_division_lead(review) else None

    def _merge_of(self, top: SessionTeam) -> Merge | None:
        review = self._review_of(top)
        return review.merge if review else None

    def _keep_merge(self, top: SessionTeam, merge: Merge | None) -> None:
        review = self._review_of(top)
        if review is not None:
            self.org.update_place(review.id, merge=merge)

    def _open_merge(self, top: SessionTeam) -> Merge | None:
        """Give the work a session submits its place in the order it goes
        into the org in. Work whose teams still wait from an earlier
        acceptance, with what is settled with them, keeps the place it has."""
        with self._weighing:
            merge = self._merge_of(top)
            if merge is None and self._review_of(top) is not None:
                seq = top.review.seq if top.review and top.review.seq else 0
                run = self._runs()
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
        store = self.org
        found: list[tuple[int, SessionTeam]] = []
        for session in store.sessions():
            places = store.session(session.id)
            merge = self._merge_of(places[0]) if places else None
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
        home = self.org.team(home_id)
        if home is None:
            return []
        path = normalize_store_path(home.boundary.path)
        folder = normalize_store_path(lost.boundary.path)
        found = []
        for name, protocol in self.protocols.items():
            if not protocol.waits_for_team_changes:
                continue
            theirs = self.read_file(lost, DocumentRef(folder, name).path)
            was = before.get(name)
            if was is not None and not (theirs and theirs.strip()):
                theirs = was.session[len(was.org.rstrip()) :]
            if not theirs or not theirs.strip():
                continue
            org = self._memory().read(DocumentRef(path, name).path) or ""
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
        store = self.org
        session = store.session_of(top)
        shapes = self._shapes(top)
        if session is None or shapes is None:
            return {}, []
        session_places, theirs, now = shapes
        take, clashes = merge_shapes(session.shape, theirs, now)
        places = {place.team_id: place for place in session_places}
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
                moved_to = self._home_of(top, place.boundary.path)
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
        memory = self._memory()
        for file, (text, over) in sorted(self.drafts(top.session).items()):
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

    def _org_turn(self) -> None:
        """Move on what the sessions' work makes of the org's teams and of
        what is settled with them, strictly in the order the work was
        submitted: only the first in order is worked out, asked about and
        landed. Safe from any thread, and to call again."""
        with self._weighing:
            for index, top in enumerate(self._in_order()):
                merge = self._merge_of(top)
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
                    for intent in self._runs().open_intents()
                ):
                    self._begin(top.id, LandOrg())

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
                    "asked_at": self._clock().isoformat(),
                    "reminded": False,
                }
            ),
        )
        self.tell_manager(
            review.id,
            merge_request(
                top,
                self.org.session_title(top),
                waiting,
                merge.code_landed,
                names=self.team(top).names,
            ),
            division_lead=True,
        )

    def _check_managers(self) -> None:
        """Move on what waits for a Manager that does not answer. It
        holds back what every session that submitted later made of
        the org's teams, so it is asked once more after the time a team
        may show no progress, and after as long again the human is asked:
        their answer lets the org's stand wherever nothing was decided."""
        limit = self.settings.soft_check_in_minutes
        if not limit:
            return
        with self._weighing:
            order = self._in_order()
            top = order[0] if order else None
            merge = self._merge_of(top) if top else None
            review = self._review_of(top) if top else None
            if top is None or merge is None or review is None or not merge.asked_at:
                return
            waiting = [item for item in merge.items if item.gate == "manager"]
            if all(item.use for item in waiting):
                return
            now = self._clock()
            if now - datetime.fromisoformat(merge.asked_at) < timedelta(minutes=limit):
                return
            if not merge.reminded:
                self._keep_merge(
                    top,
                    merge.model_copy(
                        update={"reminded": True, "asked_at": now.isoformat()}
                    ),
                )
                self.tell_manager(
                    review.id,
                    "This still waits for you, and with it what every session "
                    "that submitted later made of its teams.\n\n"
                    + merge_request(
                        top,
                        self.org.session_title(top),
                        waiting,
                        merge.code_landed,
                        names=self.team(top).names,
                    ),
                    division_lead=True,
                )
                return
            if not merge.code_landed or any(e.id == ORG_WAITS for e in top.escalations):
                return
            self.org.raise_escalation(
                top.id,
                ORG_WAITS,
                kind="critical_issue",
                title=f"{self.team(top).names.A_manager} has not settled the teams "
                "of accepted work",
                detail=(
                    f"The code of {top.title!r} is accepted and merged. What "
                    "it makes of the org's teams waits for the "
                    f"{self.team(top).names.manager} that reviews it, which has "
                    "not answered, and the "
                    "teams of sessions that submitted later "
                    "wait behind it. Answer this to go on without the "
                    f"{self.team(top).names.manager}: the org's stay wherever it "
                    "has not decided, and "
                    "what the session wrote for a team that is not taken is "
                    "kept in that team's record."
                ),
                at=utc_now(),
            )
        self._save_soon()

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
        names = self.names_at(review_id)
        store = self.org
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
            named = [item.key for item in waiting if item.kind == "team"]
            unknown = [choice.team for choice in teams if choice.team not in named]
            if unknown:
                raise Invalid(
                    f"Not waiting: {', '.join(unknown)}. The teams that wait: "
                    f"{', '.join(named) or 'none'}"
                )
            placed = {choice.team: choice for choice in teams}
            for choice in teams:
                self._check_placement(top, choice)
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
            settled = []
            for item in waiting:
                choice = (
                    chosen.get((item.document, item.key))
                    if item.waits_with_teams
                    else None
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
            self._keep_merge(
                top,
                merge.model_copy(update={"items": [*others, *settled], "note": note}),
            )
            landed = merge.code_landed
        self._org_turn()
        self._save_soon()
        return "Settled. " + (
            "It is the org's as soon as it is its turn."
            if landed
            else f"It becomes the org's when the {names.reviewer} that reviews the "
            "session's "
            "work has accepted it."
        )

    def _check_placement(self, top: SessionTeam, choice: TeamChoice) -> None:
        """Hold where a Manager puts a team it takes from a session, and
        what it calls it, to what the org allows.

        Raises:
            Invalid: It cannot be put there or called that.
        """
        store = self.org
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
        if level > self.settings.max_depth:
            raise Invalid(
                f"An org may have at most {self.settings.max_depth} levels of "
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
        with self._saving:
            return self._land_all(top, seen, org)

    def _land_all(
        self,
        top: SessionTeam,
        seen: dict[str, str | None] | None,
        org: tuple[dict[str, Side], list[MergeItem]] | None,
    ) -> Landed:
        store = self.org
        session = store.session_of(top)
        shapes = self._shapes(top)
        merge = self._merge_of(top)
        if session is None or shapes is None or merge is None:
            return Landed(Taken([], []), [], [], [], [], [])
        if org is None:
            take, items = self._manager_items(top, [])
            waits = {*take, *(item.key for item in items if item.kind == "team")}
            agreements = {
                file
                for file in self.drafts(top.session)
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
            records = self._land_records(top, seen, waits=waits, held=agreements)
            return Landed(records, [], [], [], [], [])
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
            taken: list[str] = []
            kept = [team for team, item in decided.items() if item.use != "session"]
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
                    )
                    if settled:
                        (taken if done else kept).append(team)
            for team in [
                *(team for team, side in take.items() if side is None),
                *(
                    team
                    for team, item in decided.items()
                    if item.use == "session" and team not in places
                ),
            ]:
                store.remove(team)
                if team in decided:
                    taken.append(team)
            # What it has now is what it is next compared by: a team the org
            # keeps otherwise is not brought up again unless the session changes
            # it once more.
            store.set_shape(session.id, theirs)
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
                    self._undraft(top.session, file)
                    held.append((item.document, item.key))
                    others.append((item.document, item.key, item.org))
                    continue
                text = item.replacement if item.use == "replacement" else item.session
                texts[file] = f"{text.strip()}\n" if text.strip() else None
                agreed.append((item.document, item.key))
                if item.use == "replacement":
                    others.append((item.document, item.key, text))
            self._keep_merge(top, None)
            records = self._land_records(top, seen, chosen=texts, tree=True)
        except BaseException:
            store.replace_all(tree)
            store.set_shape(session.id, shape)
            self._keep_merge(top, merge)
            raise
        return Landed(records, taken, kept, agreed, held, others)

    def _land_accepted(
        self, top: SessionTeam, seen: dict[str, str | None] | None
    ) -> None:
        """Land the work of a session whose code was just merged: with
        what it makes of the org's teams, and what is settled with them,
        when that is settled and it is its turn, and without it otherwise."""
        with self._weighing:
            merge = self._open_merge(top)
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
            self.reports_grew(folder, role)
        now = self.org.place(top.id)
        if (
            self._merge_of(top) is None
            and now is not None
            and any(e.id == ORG_WAITS for e in now.escalations)
        ):
            # Its Manager answered after all.
            self.org.settle_escalation(top.id, ORG_WAITS)
        review = self._review_of(top)
        if review is not None and landed.records.left:
            self.notify(
                self.reviewer_conversation(review),
                files_not_written(top.title, landed.records.left),
            )
        if review is not None and (
            landed.taken or landed.kept or landed.agreed or landed.held
        ):
            self.tell_writer(
                top.id,
                teams_settled(
                    review.title,
                    landed.taken,
                    landed.kept,
                    landed.agreed,
                    landed.held,
                    landed.others,
                    names=self.team(top).names,
                )
                + (f"\n\n{note}" if note else ""),
            )
        self._record_not_taken(top, landed)
        self._org_turn()

    def _record_not_taken(self, top: SessionTeam, landed: Landed) -> None:
        """Keep on record what a session wrote of a document that the org
        did not take, also when the session is closed and nobody is told."""
        for name, directory, _ in landed.others:
            self._land_event(
                self._home_of(top, directory),
                self._event(
                    "not_taken",
                    top,
                    f"the {name} it wrote for "
                    f"{directory or 'the repository'}: its merged code was "
                    "written to it",
                ),
            )

    async def _do_land_org(self, intent: Intent) -> None:
        """Land what a session's accepted work makes of the org's teams,
        and what is settled with them, when it is its turn and all of it is settled."""
        top = self.org.place(intent.node_id)
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
            merge = self._merge_of(top)
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

    def _settle_on_close(self, top: SessionTeam) -> None:
        """Settle what waits of a session that is being closed: of work
        whose code is in the repository, the org's teams and what is
        settled with them stay wherever nobody decided otherwise, and what
        the session wrote lands where the org has the team and is that
        team's record where it has not."""
        with self._weighing:
            merge = self._merge_of(top)
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
            self.reports_grew(folder, role)
        self._record_not_taken(top, landed)

    def _sent_back(self, top: SessionTeam) -> None:
        """The work of a session was sent back: it has no place in the
        order any more, unless teams of work accepted before still wait."""
        with self._weighing:
            merge = self._merge_of(top)
            review = self._review_of(top)
            if merge is None or review is None or merge.code_landed:
                return
            self._keep_merge(top, None)
            decided = [
                item for item in merge.items if item.gate == "manager" and item.use
            ]
            if decided:
                # They hold again for whatever is unchanged when it comes back.
                self._runs().mark(DECIDED, top.id, _ITEMS.dump_json(decided).decode())
            if merge.asked and any(
                item.gate == "manager" and item.use is None for item in merge.items
            ):
                self.tell_manager(
                    review.id,
                    f"The work of {top.title!r} was sent back to its team. "
                    "What waited of it for you waits no longer. What you "
                    "decided of it holds again for whatever is unchanged "
                    "when it is submitted again.",
                    division_lead=True,
                )
        self._org_turn()

    def _waits_for_team_changes(self, file: str) -> bool:
        """Whether what a session wrote in a file of team memory becomes
        the org's only with its changes to the teams: its document says."""
        protocol = self.protocol_at(file)
        return protocol is not None and protocol.waits_for_team_changes

    def _documents_waiting(self, top: SessionTeam) -> list[tuple[str, str, str]]:
        """What a session wrote of the documents that wait with its teams
        and is not the org's: each by name and folder, with the session's
        text."""
        memory = self._memory()
        return [
            (DocumentRef.at(file).name, file.rpartition("/")[0], text or "")
            for file, (text, _) in sorted(self.drafts(top.session).items())
            if self._waits_for_team_changes(file) and memory.read(file) != text
        ]

    def _merged_as(self, place: SessionTeam) -> str:
        """What the commit that merges a team's work calls it: the session
        too, for the work of one."""
        session = self.org.session_title(place)
        return f"{place.title} · {session}" if session else place.title

    def _to_reconcile(self, place: SessionTeam, above: SessionTeam) -> Meets | None:
        """What the IC above has to settle before the work of a team is
        merged: the files both it and the checkout above changed since it
        began, and the documents it wrote over ones that changed meanwhile.

        Returns:
            What to tell that IC, with what names the state it was said
            of; None when the work meets nothing.

        Raises:
            GitError: A checkout could not be read.
        """
        team, checkout = Path(place.worktree_path), Path(above.worktree_path)
        changes = (
            changes_of(checkout, team)
            if checkout_root(str(team)) != checkout_root(str(checkout))
            else None
        )
        files = []
        if changes is not None:
            ours = head_of(checkout)
            # Lines that do not merge, and files outside the team's
            # directory, go back to the team when it is merged; there is
            # nothing for the reviewer to weigh in them.
            unmergeable = (
                conflicts(checkout, ours, changes.tip)
                if ours and can_merge_in_memory(checkout)
                else ()
            )
            if not unmergeable:
                files = [path for path in changes.both if owns(place, path)]
        opens = self.org.opens_session(place)
        changed = self.documents_to_reconcile(place) if opens else []
        inherited = self.documents_inherited(place) if opens else []
        if not files and not changed and not inherited:
            return None
        text = reconcile_request(
            place,
            above.worktree_path,
            changes.tip if changes else "",
            files,
            list(changes.landed) if changes else [],
            changed,
            inherited,
            names=self.team(place).names,
            reports=self.kept_documents(above).reports,
        )
        there = git(checkout, ["ls-tree", "HEAD", "--", *files]) if files else ""
        compared = json.dumps(
            [changes.tip if changes else "", there, changed, *inherited]
        )
        return Meets(
            text,
            hashlib.sha256(compared.encode()).hexdigest(),
            {
                DocumentRef(found.folder, found.name).path: found.now or None
                for found in changed
            },
            [
                *(MergeItem(kind="code", key=path, gate="ic") for path in files),
                *(
                    MergeItem(
                        kind="document",
                        document=found.name,
                        key=found.folder,
                        gate="ic",
                        session=found.written,
                        org=found.now,
                    )
                    for found in changed
                ),
                *(
                    MergeItem(
                        kind="document",
                        document=found.name,
                        key=found.home,
                        gate="ic",
                        title=found.lost,
                        session=found.written,
                        org=found.now,
                    )
                    for found in inherited
                ),
            ],
        )

    def _before(self, place: SessionTeam) -> SessionTeam | None:
        """The work submitted first of all that was submitted before that
        of ``node`` and is not decided on yet; None when it is its turn."""
        store = self.org
        above = store.place(place.parent_id) if place.parent_id else None
        if above is None:
            return None
        # Every session's work goes into the one checkout of the repository.
        beside = (
            [
                places[0]
                for session in store.sessions()
                if (places := store.session(session.id))
            ]
            if store.opens_session(place)
            else store.below(above.id)
        )
        # Work whose acceptance is being carried out no longer waits.
        accepting = {
            intent.node_id
            for intent in self._runs().open_intents()
            if intent.kind == "accept"
        }
        waiting = [
            other
            for other in beside
            if other.id != place.id
            and other.id not in accepting
            and other.review is not None
            and other.review.state == "pending"
            and _submitted(other) < _submitted(place)
        ]
        return min(waiting, key=_submitted) if waiting else None

    def _tell_turns(self) -> None:
        """Tell each reviewer that was made to wait for work submitted
        earlier that it is its turn, once that work is decided on. Safe
        from any thread."""
        store = self.org
        run = self._runs()
        for place_id in run.marked(TURN):
            place = store.place(place_id)
            above = store.place(place.parent_id or "") if place else None
            pending = place and place.review and place.review.state == "pending"
            if place is None or above is None or not pending:
                run.unmark(TURN, place_id)
                continue
            if self._before(place) is not None:
                continue
            run.unmark(TURN, place_id)
            self.notify(
                self.reviewer_conversation(above),
                f"It is your turn: the work submitted before that of "
                f"{place.title!r} (id {place.id}) is decided on. Accept it "
                "with `review_team` now, or send it back.",
            )

    def check_accept(self, place_id: str, reconciled: str) -> None:
        """Hold the acceptance of a team's work to what is asked of every
        one: work is accepted in the order it was submitted, and work that
        meets other work is reconciled with it first.

        Raises:
            Conflict: Work submitted before it still waits, or it meets
                work the reviewer has not said it reconciled.
        """
        store = self.org
        place = store.place(place_id)
        above = store.place(place.parent_id) if place and place.parent_id else None
        if place is None or above is None or place.review is None:
            return
        if any(
            intent.kind == "accept" and intent.node_id == place_id
            for intent in self._runs().open_intents()
        ):
            raise Conflict(f"The work of {place.title!r} is being accepted already.")
        first = self._before(place)
        if first is not None:
            theirs = first.parent_id != place.parent_id
            if theirs:
                # Its reviewer is told when the other reviewer has decided.
                self._runs().mark(TURN, place.id)
            raise Conflict(
                accept_in_order(
                    place,
                    first,
                    store.session_title(first) if theirs else "",
                    names=self.names_at(place_id),
                )
            )
        try:
            meets = self._to_reconcile(place, above)
        except (GitError, OSError):
            # Merging says what is wrong with the checkouts.
            return
        refusal = self._not_reconciled(place, meets, reconciled)
        if refusal is not None:
            raise Conflict(refusal)

    def work_astray(self, place: SessionTeam) -> str | None:
        """Why what a team's checkout is on is not all of its work, if it
        is not.

        The work of a team is what the branch of its checkout holds. The
        agents' own git keeps a checkout on that branch, but an agent can
        reach another git, so the director looks for itself before work is
        submitted and before it is accepted: the checkout is on its branch,
        and what was accepted from the teams that share it is in it.

        Returns:
            What to tell whoever asked, in words for an agent; None when
            the work is all there.
        """
        if not place.worktree_path:
            return None
        checkout = Path(place.worktree_path)
        ours = checkout_root(place.worktree_path) != checkout_root(str(self._repo_dir))
        on = branch_of(checkout)
        # The repository's own checkout is the human's, on any branch.
        if ours and place.branch and on != place.branch:
            return (
                f"The checkout of {place.title!r} is "
                + (f"on the branch {on!r}" if on else "on no branch")
                + f", and its work is taken from {place.branch!r}. Run any git "
                "command there: the checkout goes back to its branch. Then "
                "bring over what was committed elsewhere"
                + (f" with `git merge {on}`" if on else "")
                + " and submit again."
            )
        for team in self.org.below(place.id):
            review = team.review
            if (
                review is None
                or review.state != "accepted"
                or not review.commit
                or checkout_root(team.worktree_path) != checkout_root(str(checkout))
            ):
                continue
            if not holds_commit(checkout, review.commit):
                return (
                    f"The accepted work of {team.title!r} (commit "
                    f"{review.commit[:10]}) is not in what the checkout of "
                    f"{place.title!r} is on: it was left on another branch. "
                    f"Bring it in with `git merge {review.commit[:10]}`, then "
                    "submit again."
                )
        return None

    def _not_reconciled(
        self, place: SessionTeam, meets: Meets | None, reconciled: str
    ) -> str | None:
        """Hold what a reviewer says it reconciled to what it was shown.

        Returns:
            What to tell the reviewer when the work meets something it has
            not been shown as it is now; None when it may be accepted.
        """
        if self.org.opens_session(place):
            self._show_reviewer_items(place, meets)
        if meets is None:
            return None
        run = self._runs()
        shown = run.marked(SHOWN).get(place.id)
        if reconciled and shown == meets.mark:
            return None
        run.mark(SHOWN, place.id, meets.mark)
        if reconciled and shown is not None:
            return (
                "What the work meets changed since you were shown it. Nothing "
                "is merged yet.\n\n" + meets.text
            )
        return meets.text

    def _show_reviewer_items(self, top: SessionTeam, meets: Meets | None) -> None:
        """Keep, with what waits of a session's work, what its reviewer
        has to reconcile."""
        with self._weighing:
            merge = self._merge_of(top)
            if merge is None:
                return
            laid_out = self.merge_checkout(top.id)
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

    async def _do_request_review(self, intent: Intent) -> None:
        await self._request_review(intent.node_id)

    def _ask_to_clear(
        self, place_id: str, detail: str, reconciled: str, choices: list[dict[str, str]]
    ) -> None:
        """Ask the human, once, to clear the way for accepted work into
        their own checkout. Their answer merges it, as its reviewer
        decided."""
        place = self.org.place(place_id)
        if place is None or any(e.id == MERGE_WAITS for e in place.escalations):
            return
        self._runs().mark(
            CLEARING,
            place_id,
            json.dumps({"reconciled": reconciled, "choices": choices}),
        )
        self.org.raise_escalation(
            place_id,
            MERGE_WAITS,
            kind="critical_issue",
            title="Accepted work cannot be merged into your checkout",
            detail=detail,
            at=utc_now(),
        )
        self._save_soon()

    async def _do_accept(self, intent: Intent) -> None:
        accept = intent.of(Accept)
        await self._accept(
            intent.node_id, accept.needed_above, accept.reconciled, accept.choices
        )

    async def _own_account(self, place: SessionTeam) -> str:
        """What a team's IC says changed, for the IC above.

        Returns:
            Its description of the changes, '' when it gave none or was
            not asked.
        """
        ic = self.writer_conversation(place)
        if ic is None or not place.base_commit:
            return ""
        files = await asyncio.to_thread(
            bugbot.changed_files,
            Path(place.worktree_path),
            place.base_commit,
            place.boundary.path,
        )
        if not files:
            return ""
        asked = self.type_of(place).review.write_up
        description = await self._ask_agent(ic, bugbot.write_up_question(files, asked))
        return description or ""

    async def _request_review(self, place_id: str) -> None:
        store = self.org
        place = store.place(place_id)
        asked = place.review if place else None
        if place is None or asked is None:
            return
        if asked.state != "pending":
            return
        description = await self._own_account(place)
        place = store.place(place_id)
        if place is None or place.review != asked:
            return
        review = asked.model_copy(update={"description": description})
        place = store.set_review(place_id, review)
        if store.opens_session(place):
            # Its Manager is asked what waits for it beside the review.
            await asyncio.to_thread(self._org_turn)
        if place.parent_id is not None:
            self._runs().mark(HELD, place_id)
            self._release_reviews(place.parent_id)

    async def _accept(
        self,
        place_id: str,
        needed_above: str,
        reconciled: str = "",
        choices: list[dict[str, str]] | None = None,
    ) -> None:
        names = self.names_at(place_id)
        store = self.org
        place = store.place(place_id)
        above = store.place(place.parent_id) if place and place.parent_id else None
        if place is None or above is None or place.review is None:
            return
        if place.review.state == "accepted":
            # Cut off after the acceptance was made: what follows it is
            # still owed.
            merge = self._merge_of(place) if store.opens_session(place) else None
            if merge is not None and not merge.code_landed:
                # What its reviewer made of its documents lands with it, as the
                # first time.
                self._take_document_choices(place, choices or [])
                await asyncio.to_thread(self._land_accepted, place, None)
            await self._pass_up(above.id, acted=True)
            # Whoever waited behind this work is still to hear it is their turn.
            self._tell_turns()
        if place.review.state != "pending":
            return
        reviewer = self.reviewer_conversation(above)
        first = self._before(place)
        if first is not None:
            # The acceptance of work submitted earlier was carried out
            # since this one was checked, and left that work waiting.
            self._runs().mark(TURN, place.id)
            self.notify(
                reviewer,
                accept_in_order(place, first, store.session_title(first), names=names),
            )
            return
        where = place.boundary.path or "the repository"
        team, checkout = Path(place.worktree_path), Path(above.worktree_path)
        merged = False
        resolved: tuple[str, ...] = ()
        opens = store.opens_session(place)
        try:
            meets = await asyncio.to_thread(self._to_reconcile, place, above)
        except (GitError, OSError):
            meets = None
        refusal = self._not_reconciled(place, meets, reconciled)
        if refusal is not None:
            # Something came into the checkout between the review and now.
            self.notify(reviewer, refusal)
            return
        astray = await asyncio.to_thread(self.work_astray, place)
        if astray is not None:
            # Accepted, part of the work would not be in what is merged.
            self.notify(
                reviewer,
                f"The work of {place.title!r} is not accepted yet. {astray} "
                "Send it back with `review_team` and that as what to change.",
            )
            return
        if checkout_root(str(team)) != checkout_root(str(checkout)):
            elsewhere = await asyncio.to_thread(self._elsewhere, place, above)
            if elsewhere is not None:
                started_on, on = elsewhere
                self._ask_to_clear(
                    place_id,
                    f"The session started from the branch {started_on!r}, "
                    "and your checkout is now "
                    + ("on no branch" if on in ("", "HEAD") else f"on {on!r}")
                    + f".\nSwitch it back to {started_on!r}, then answer "
                    "this: the work is merged then.",
                    reconciled,
                    choices or [],
                )
                self.notify(
                    reviewer,
                    f"The work of {place.title!r} is accepted and waits for "
                    "the human, whose checkout is on another branch than the "
                    "one the session started from. There is nothing more "
                    "for you to do.",
                )
                return
            if not opens:
                # What is uncommitted in a session's directory is the work
                # of the team it was opened beside, and stays.
                await asyncio.to_thread(self._put_aside, above, place)
            try:
                result = await asyncio.to_thread(
                    merge_back,
                    checkout,
                    team,
                    place.boundary,
                    self._merged_as(place),
                )
            except (GitError, OSError) as error:
                self.notify(
                    reviewer,
                    f"The work of {place.title!r} could not be merged into your "
                    f"checkout, and still waits for your review: {error}",
                )
                return
            if result.outcome == "conflicts":
                finished = await asyncio.to_thread(
                    self._finish_merge, place, above, result
                )
                if isinstance(finished, str):
                    if opens:
                        self._show_reviewer_items(place, meets)
                    self.notify(reviewer, finished)
                    return
                result = finished
                resolved = result.files if result.outcome == "merged" else ()
            else:
                await asyncio.to_thread(self._drop_merge_checkout, place_id)
            if result.outcome == "refused" and is_division_lead(above):
                # Only the human can clear their own checkout: they are
                # asked, and their answer merges the work.
                self._ask_to_clear(
                    place_id,
                    f"git would not merge it as the checkout is:\n"
                    f"{result.detail}\nCommit or put aside what is in "
                    "the way, then answer this: the work is merged then.",
                    reconciled,
                    choices or [],
                )
                self.notify(
                    reviewer,
                    f"The work of {place.title!r} is accepted and waits for "
                    "the human, whose uncommitted changes are in the way of "
                    "the merge. There is nothing more for you to do.",
                )
                return
            if result.outcome == "refused":
                self.notify(
                    reviewer,
                    f"git would not merge the work of {place.title!r} into your "
                    f"checkout as it is:\n{result.detail}\n"
                    + (
                        "What is in the way is work that is not committed "
                        "yet in the same files. Accept it again with "
                        "`review_team` once that is committed."
                        if opens
                        else "Commit or put aside what is in the way, then "
                        "accept it again with `review_team`."
                    ),
                )
                return
            if result.outcome == "outside":
                issue = (
                    f"These files are outside {where}, which is all your "
                    "team owns. Take them out of your commits."
                )
                self.request_changes(
                    place_id,
                    [ReviewFinding(path=", ".join(result.files), issue=issue)],
                )
                self.notify(
                    reviewer,
                    f"The work of {place.title!r} was not taken into your "
                    f"checkout: {issue} ({', '.join(result.files)}) "
                    "It was sent back to the team.",
                )
                return
            merged = result.outcome == "merged"
            if resolved:
                self.notify(
                    self.writer_conversation(place),
                    f"Your team's work did not merge by itself. The {names.reviewer} "
                    "that "
                    "reviewed it finished the merge in these files, which are "
                    f"now as it left them: {', '.join(resolved)}",
                )
        tip = await asyncio.to_thread(head_of, team)
        store.set_review(
            place_id,
            place.review.model_copy(update={"state": "accepted", "at": utc_now()}),
            base_commit=tip or place.base_commit,
        )
        self.progressed(above.id)
        self._work_accepted(place)
        if meets is not None or resolved:
            self.record_event(
                above.id,
                "reconciled",
                place,
                (reconciled or f"merged {', '.join(resolved)}")[:ASKED_KEPT],
            )
        self.record_event(
            above.id,
            "accepted",
            place,
            "at the first review"
            if place.review.round <= 1
            else f"at review {place.review.round}",
            [place.boundary.path],
            between=(place.base_commit, tip) if place.base_commit and tip else ("", ""),
        )
        if opens:
            self._take_document_choices(place, choices or [])
            await asyncio.to_thread(
                self._land_accepted, place, meets.seen if meets else None
            )
            self._changed_since_accept.pop(place.session, None)
        self._runs().unmark(SHOWN, place_id)
        self._runs().unmark(CLEARING, place_id)
        self._tell_turns()
        if not is_division_lead(above):
            self.count_accepted_review(above.id)
        self.notify(
            self.writer_conversation(place),
            f"The {names.reviewer} of {above.title!r} accepted your team's work"
            + (" and it is merged into that team's checkout." if merged else "."),
        )
        if merged:
            self.notify(
                reviewer, f"The work of {place.title!r} is merged into your checkout."
            )
        await self._pass_up(above.id, acted=True)
        further = store.place(above.parent_id) if above.parent_id else None
        if needed_above and further is not None:
            self.notify(
                self.writer_conversation(further),
                f"Team {above.title!r} (id {above.id}) below you accepted the "
                f"work of its sub-team {place.title!r} in {where} and says a team "
                f"above it waits on that work: {needed_above}\nIt is in "
                f"{above.worktree_path}"
                + (f" on branch {above.branch}." if above.branch else "."),
            )

    def _finish_merge(
        self, place: SessionTeam, above: SessionTeam, found: MergeBack
    ) -> MergeBack | str:
        """Take the merge the reviewer of a team's work finished, or lay
        it out for it to finish.

        Returns:
            How taking it went, or what to tell the reviewer when there is
            nothing to take yet.
        """
        checkout = Path(above.worktree_path)
        ours = head_of(checkout)
        merge = self.merge_checkout(place.id)
        if (
            ours is None
            or merge is None
            or (merge.ours, merge.tip) != (ours, found.tip)
            or not Path(merge.where).is_dir()
        ):
            if ours is None:
                return f"The checkout of {above.title!r} has no commit."
            where = self._merges_for(above.id) / place.id.replace("@", "-")
            merge = MergeCheckout(
                where=str(where), ours=ours, tip=found.tip, files=list(found.files)
            )
            try:
                prepare(self._repo_dir, where, ours, found.tip)
            except (GitError, OSError) as error:
                return (
                    f"The work of {place.title!r} does not merge with your "
                    f"checkout ({', '.join(found.files)}), and the merge could "
                    f"not be laid out for you: {error}"
                )
            self._runs().mark(MERGING, place.id, merge.model_dump_json())
            return merge_prepared(place, merge.where, merge.files)
        try:
            commit = finish(
                merge, merge_message(self._merged_as(place), place.boundary.path)
            )
        except Unfinished as error:
            return (
                f"The merge of the work of {place.title!r} in {merge.where} is "
                f"not taken yet, and the work still waits for you: {error}"
            )
        except (GitError, OSError) as error:
            return (
                f"The merge of the work of {place.title!r} in {merge.where} "
                f"could not be committed: {error}"
            )
        taken = take_merge(checkout, commit)
        if taken.outcome != "merged":
            return taken
        self._drop_merge_checkout(place.id)
        return MergeBack("merged", found.tip, tuple(merge.files))

    def _own_changes(
        self, place: SessionTeam, teams: list[SessionTeam]
    ) -> list[str] | None:
        """The files a team changed itself since its work was last taken
        up, committed or not: those in its directory that are in no
        sub-team's.

        Returns:
            None when git cannot tell.
        """
        if not place.base_commit:
            return None
        checkout = Path(place.worktree_path)
        try:
            names = git(
                checkout,
                ["diff", "--name-only", "-z", place.base_commit, "HEAD", "--"]
                + [pathspec(place.boundary.path)],
            ).split("\0")
            names += uncommitted(checkout, place.boundary.path)
        except (GitError, OSError):
            return None
        given = [team.boundary.path for team in teams]
        return sorted(
            {
                name
                for name in names
                if name
                and not any(
                    name == path or name.startswith(f"{path}/") for path in given
                )
            }
        )

    def _elsewhere(
        self, place: SessionTeam, above: SessionTeam
    ) -> tuple[str, str] | None:
        """The branch a session started from and the one the human's
        checkout is on now, when they differ: the session's work is not
        merged into that."""
        session = self.org.session_of(place)
        started_on = session.branch if session else ""
        if not is_division_lead(above) or not started_on:
            return None
        on = _branch_of(above.worktree_path)
        return None if on == started_on else (started_on, on)

    async def _pass_up(self, place_id: str, *, acted: bool) -> None:
        """Take a team's work into the team above without another review,
        when all of it is work of its sub-teams that was accepted.

        Each change is reviewed once, by the IC directly above the team
        that made it. A team that changed something itself submits its work
        as a whole instead.

        Args:
            acted: Whether the team's IC has just accepted the last of its
                sub-teams itself; otherwise their work came up on its own.
        """
        names = self.names_at(place_id)
        store = self.org
        place = store.place(place_id)
        # A Division Lead place reviews; it has no work to hand on.
        if place is None or is_division_lead(place) or not store.below(place_id):
            return
        # A team that was set up and never given work owes none.
        teams = [
            team
            for team in store.below(place_id)
            if team.review is not None or self.writer_conversation(team) is not None
        ]
        if not teams or any(
            team.review is None or team.review.state != "accepted" for team in teams
        ):
            return
        if place.review is not None and place.review.state == "pending":
            return
        opens = store.opens_session(place)
        if opens and _LANE.get() != BASE_LANE:
            # The work of a session goes into what the org shares.
            await self._in_base(lambda: self._pass_up(place_id, acted=acted))
            return
        ic = self.writer_conversation(place)
        above = store.place(place.parent_id) if place.parent_id else None
        # A file in a sub-team's folder is that team's, also while the team
        # has not been given work.
        own = await asyncio.to_thread(self._own_changes, place, store.below(place_id))
        if own is None:
            return
        # The work of a session meets what other sessions delivered: where
        # it does, or the human's checkout has moved to another branch, the
        # reviewer is asked even if the team changed nothing.
        meets = False
        if not own and above is not None and store.opens_session(place):
            try:
                meets = (
                    await asyncio.to_thread(self._to_reconcile, place, above)
                    is not None
                    or await asyncio.to_thread(self._elsewhere, place, above)
                    is not None
                )
            except (GitError, OSError):
                meets = True
        if own or above is None or meets:
            if not acted:
                self.notify(
                    ic,
                    "The work of all your sub-teams is accepted and in your "
                    "checkout. "
                    + (
                        "Your team also changed files itself that nobody has "
                        f"reviewed ({', '.join(own[:10])}). "
                        if own
                        else ""
                    )
                    + "Call `submit_review` when your team's work is complete.",
                )
            return
        team, checkout = Path(place.worktree_path), Path(above.worktree_path)
        if checkout_root(str(team)) != checkout_root(str(checkout)):
            try:
                result = await asyncio.to_thread(
                    merge_back, checkout, team, place.boundary, place.title
                )
                problem = (
                    f"{result.outcome}: {', '.join(result.files) or result.detail}"
                    if result.outcome in ("refused", "conflicts", "outside")
                    else None
                )
            except (GitError, OSError) as error:
                problem = str(error)
            if problem is not None:
                self.notify(
                    ic,
                    "The work of all your sub-teams is accepted, but your "
                    "team's checkout does not merge into the checkout of "
                    f"{above.title!r} ({problem}). Call `submit_review`: "
                    f"the {names.reviewer} of {above.title!r} is given the merge to "
                    "finish "
                    "when it accepts the work.",
                )
                return
        tip = await asyncio.to_thread(head_of, team)
        store.set_review(
            place_id,
            Review(
                state="accepted",
                summary="The accepted work of its sub-teams, with no changes of "
                "its own.",
                passed_up=True,
                commit=tip or "",
                at=utc_now(),
                # It goes into the org after what was submitted before now.
                seq=self.next_submission() if opens else 0,
            ),
            base_commit=tip or place.base_commit,
        )
        self._work_accepted(place)
        if opens:
            await asyncio.to_thread(
                self._land_accepted, store.place(place_id) or place, None
            )
            # As after any accept: what changes from now is said once more.
            self._changed_since_accept.pop(place.session, None)
        self._tell_turns()
        if acted:
            self.notify(
                ic,
                f"Your team changed nothing itself, so its work went up to "
                f"{above.title!r} without another review.",
            )
        self._release_reviews(above.id)
        await self._pass_up(above.id, acted=False)

    def _put_aside(self, above: SessionTeam, team: SessionTeam) -> None:
        """Take out of the checkout above what is uncommitted in the
        directory of a team with a checkout of its own, where it would stop
        the team's accepted work from coming in, and tell the IC above."""
        checkout, path = Path(above.worktree_path), team.boundary.path
        ref = f"{ASIDE_REFS}/{team.id}/{ref_stamp()}"
        try:
            files = uncommitted(checkout, path)
            if not files or save_and_discard(checkout, path, ref) is None:
                return
        except (GitError, OSError):
            logger.exception("Could not put aside what is in %s", path)
            return
        self.notify(
            self.writer_conversation(above),
            put_aside_notice(team, files, ref),
        )
