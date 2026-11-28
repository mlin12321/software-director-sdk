"""A team's work on its way up: submitted, sent back, accepted, merged."""

import asyncio
import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from openhands.agent_server.director import bugbot
from openhands.agent_server.director.agent_types import DEFAULT_TYPE, TeamTemplate
from openhands.agent_server.director.documents.store import DocumentRef
from openhands.agent_server.director.documents.team_documents import TeamDocuments
from openhands.agent_server.director.environment import EnvironmentFailed
from openhands.agent_server.director.environment.checkouts import (
    Checkouts,
    MergeBack,
    MergeCheckout,
    Unfinished,
    time_stamp,
)
from openhands.agent_server.director.intents import Accept, RequestReview
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import (
    MergeItem,
    Review,
    ReviewFinding,
    SessionTeam,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    is_division_lead,
    utc_now,
)
from openhands.agent_server.director.prompts import (
    NEED_FINDING,
    accept_in_order,
    changes_requested,
    merge_prepared,
    put_aside_notice,
    reconcile_request,
    review_request,
    session_work,
    still_undecided,
)
from openhands.agent_server.director.run_state import (
    CLEARING,
    HELD,
    SHOWN,
    TURN,
    Intent,
)
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.check_count_service import (
    CheckCountService,
)
from openhands.agent_server.director.service.message_delivery_service import (
    MessageDeliveryService,
)
from openhands.agent_server.director.service.model_call_service import ModelCallService
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.resumable_work_service import (
    ResumableWorkService,
)
from openhands.agent_server.director.service.session_merge_service import (
    Meets,
    SessionMergeService,
)
from openhands.agent_server.director.service.session_record_service import (
    SessionRecordService,
)
from openhands.agent_server.director.service.team_context_service import (
    TeamContextService,
)
from openhands.agent_server.director.service.team_history_service import (
    ASKED_KEPT,
    TeamHistoryService,
)
from openhands.agent_server.director.service.team_messaging_service import (
    TeamMessagingService,
)
from openhands.agent_server.director.service.team_type_service import TeamTypeService
from openhands.agent_server.director.service.work_queue_service import (
    BASE_LANE,
    WorkQueueService,
)
from openhands.sdk.errors import (
    Conflict,
    NotFound,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# Where what was in the way of a team's accepted work is kept.
ASIDE_REFS = "refs/agents/aside"
# The escalation that asks the human to clear the way for a merge.
MERGE_WAITS = "merge-waits"
# How long a message waits for an IC that submitted to end its turn.
_REST_CHECKS = 600
_REST_CHECK_SECONDS = 0.05


def _submitted(team: SessionTeam) -> tuple[int, str]:
    """When a team's work was submitted, to put work in order by."""
    return (team.review.seq, team.review.at) if team.review else (0, "")


@dataclass
class ReviewService:
    """Carries a team's work through its review.

    Work is reviewed once, by the reviewer of the team above. Requesting a
    review and accepting one are steps of resumable work
    (`do_request_review`, `do_accept`); accepted work is merged into the
    checkout above and passed further up where the team above added
    nothing of its own. What a session's accepted work makes of the org
    is :class:`SessionMergeService`'s.
    """

    org: Callable[[], OrgStore]
    """The org as it is loaded now."""
    repo_dir: Callable[[], Path]
    """The repository the org is in."""
    checkouts: Checkouts
    """Where each team's work is and how it moves."""
    team_documents: Callable[[], TeamDocuments]
    """The org's documents as each team reads and writes them."""
    launcher: Callable[[], TeamLauncher]
    """What runs the agents; raises where the director runs none."""
    find_review_instructions: Callable[[], str | None]
    """Gives the code review skill reviewers check work by."""
    org_state: OrgStateService
    org_save: OrgSaveService
    org_settings: OrgSettingsService
    team_types: TeamTypeService
    team_history: TeamHistoryService
    team_context: TeamContextService
    team_messaging: TeamMessagingService
    message_delivery: MessageDeliveryService
    resumable_work: ResumableWorkService
    work_queue: WorkQueueService
    check_counts: CheckCountService
    confinement: AgentConfinementService
    model_calls: ModelCallService
    session_records: SessionRecordService
    session_merges: SessionMergeService
    """What a session's accepted work makes of the org."""
    _review_instructions: dict[str, str] = field(default_factory=dict, init=False)
    _loading_review_instructions: threading.Lock = field(
        default_factory=threading.Lock, init=False
    )

    def request_review(self, place_id: str) -> None:
        """Have the work a team submitted reviewed. Safe from any thread."""
        place = self.org().place(place_id)
        if place is not None and self.org().opens_session(place):
            self.session_merges.open_merge(place)
        self.resumable_work.begin(place_id, RequestReview())

    def rest(self, place_id: str) -> None:
        """End the turn of a team's IC after the step it is in: it has
        handed its work over and acts again when something is sent to it.
        Safe from any thread."""
        if self.work_queue.is_open:
            self.work_queue.enqueue(lambda: self._rest(place_id), place_id)

    async def _rest(self, place_id: str) -> None:
        place = self.org().place(place_id)
        conversation_id = self.team_types.writer_conversation(place) if place else None
        if (
            place is None
            or conversation_id is None
            or self.org_state.paused is not None
        ):
            return
        launcher = self.launcher()
        writer = self.team_types.team(place).writer
        if any(agent.role == writer and agent.kind == "acp" for agent in place.agents):
            # It runs its own turn, and the tool's answer tells it to end
            # it. Only one that carries on regardless is cut short, which
            # costs it the step it is in.
            await self._until_at_rest(conversation_id)
            if conversation_id in self.org_state.running:
                await launcher.stop(UUID(conversation_id))
        else:
            await launcher.pause(UUID(conversation_id))
        # This runs in the lane of the team's messages: nothing is sent to
        # the agent before it has stopped, or it would lie there unread.
        await self._until_at_rest(conversation_id)

    async def _until_at_rest(self, conversation_id: str) -> None:
        for _ in range(_REST_CHECKS):
            if conversation_id not in self.org_state.running:
                return
            await asyncio.sleep(_REST_CHECK_SECONDS)

    def request_changes(self, place_id: str, findings: list[ReviewFinding]) -> None:
        """Send a team's work back to it with what must change.

        Safe from any thread.

        Raises:
            NotFound: The team does not exist.
            Conflict: None of its work waits for a review.
        """
        names = self.team_types.names_at(place_id)
        store = self.org()
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
        self.confinement.drop_merge_checkout(place_id)
        self.tell_turns()
        self.team_messaging.progressed(place_id, by_above=True)
        self.team_history.record_event(
            place.parent_id,
            "sent_back",
            place,
            f"round {review.round}, {len(findings)} to change",
        )
        for finding in findings:
            # A finding of the runtime names several files in one.
            for path in finding.path.split(", "):
                if path.strip():
                    self.team_history.record_need(place, path.strip(), NEED_FINDING)
        limit = self.org_settings.settings.manager_check_ic_setbacks
        if limit and place.parent_id is not None and review.round % limit == 0:
            with self.check_counts.lock:
                counted = self.check_counts.counted(place.parent_id)
            self.team_messaging.wake_manager(
                place.parent_id,
                f"your {names.reviewer} has sent the work of {place.title!r} back "
                f"{review.round} times.",
                counted,
                team=place,
            )
        self.message_delivery.notify(
            self.team_types.writer_conversation(place),
            changes_requested(
                above or place,
                findings,
                of_a_session=store.opens_session(place),
                names=names,
            ),
        )
        if store.opens_session(place):
            self.session_merges.sent_back(place)

    def release_reviews(self, place_id: str | None) -> None:
        """Hand a team's IC the reviews its sub-teams wait for (see
        :meth:`_release_reviews`). Safe from any thread."""
        if place_id is not None and self.work_queue.is_open:
            self.work_queue.enqueue(
                lambda: self._released(place_id), f"review:{place_id}"
            )

    async def _released(self, place_id: str) -> None:
        self.release_held(place_id)

    def release_held(self, place_id: str | None) -> None:
        """Hand a team's IC the reviews its sub-teams wait for.

        Each goes as soon as its team has submitted: no team waits for
        the teams beside it to be reviewed. What must be whole before work
        moves up is checked where the IC submits its own
        (``agent_actions.submit_review``). The IC may end its turn without
        deciding; the next request names what still waits for it.
        """
        store = self.org()
        node = store.place(place_id) if place_id else None
        if node is None or self.org_state.paused is not None:
            return
        run = self.org_state.run_state()
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
                            self.session_merges.documents_waiting(team),
                            names=self.team_types.team(team).names,
                            kept=self.team_types.kept_documents(node),
                        )
                        if store.opens_session(team)
                        else ""
                    )
                    + review_request(
                        team,
                        team.review,
                        self.team_context.so_far(team),
                        self.team_context.asked_of(team),
                        self.review_instructions(self.team_types.type_of(team)),
                        self.team_types.kept_documents(node).read_by_sub_teams,
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
        self.team_messaging.tell(
            node.id,
            self.team_types.team(node).reviewer,
            "\n\n---\n\n".join(requests),
            division_lead=True,
        )
        # Only now: a server that ends before the request is kept asks again.
        for team in waiting:
            run.unmark(HELD, team.id)

    def _asked_for(self, team: SessionTeam) -> str:
        """What the session ``team`` belongs to was opened for."""
        session = self.org().session_of(team)
        return session.request if session else ""

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
        place = self.org().place(place_id)
        self.resumable_work.begin(
            place_id,
            Accept(
                needed_above=needed_above,
                reconciled=reconciled,
                choices=choices or [],
                # The work of a session goes into what the org shares.
                base=place is not None and self.org().opens_session(place),
            ),
        )

    def _merged_as(self, place: SessionTeam) -> str:
        """What the commit that merges a team's work calls it: the session
        too, for the work of one."""
        session = self.org().session_title(place)
        return f"{place.title} · {session}" if session else place.title

    def _to_reconcile(self, place: SessionTeam, above: SessionTeam) -> Meets | None:
        """What the IC above has to settle before the work of a team is
        merged: the files both it and the checkout above changed since it
        began, and the documents it wrote over ones that changed meanwhile.

        Returns:
            What to tell that IC, with what names the state it was said
            of; None when the work meets nothing.

        Raises:
            EnvironmentFailed: A checkout could not be read.
        """
        changes = self.checkouts.overlap(place, above)
        files = changes.files if changes else []
        opens = self.org().opens_session(place)
        changed = (
            self.team_documents().changed_elsewhere(place.session) if opens else []
        )
        inherited = self.session_records.documents_inherited(place) if opens else []
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
            names=self.team_types.team(place).names,
            reports=self.team_types.kept_documents(above).reports,
        )
        there = changes.seen if changes else ""
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
        store = self.org()
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
            for intent in self.org_state.run_state().open_intents()
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

    def tell_turns(self) -> None:
        """Tell each reviewer that was made to wait for work submitted
        earlier that it is its turn, once that work is decided on. Safe
        from any thread."""
        store = self.org()
        run = self.org_state.run_state()
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
            self.message_delivery.notify(
                self.team_types.reviewer_conversation(above),
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
        store = self.org()
        place = store.place(place_id)
        above = store.place(place.parent_id) if place and place.parent_id else None
        if place is None or above is None or place.review is None:
            return
        if any(
            intent.kind == "accept" and intent.node_id == place_id
            for intent in self.org_state.run_state().open_intents()
        ):
            raise Conflict(f"The work of {place.title!r} is being accepted already.")
        first = self._before(place)
        if first is not None:
            theirs = first.parent_id != place.parent_id
            if theirs:
                # Its reviewer is told when the other reviewer has decided.
                self.org_state.run_state().mark(TURN, place.id)
            raise Conflict(
                accept_in_order(
                    place,
                    first,
                    store.session_title(first) if theirs else "",
                    names=self.team_types.names_at(place_id),
                )
            )
        try:
            meets = self._to_reconcile(place, above)
        except (EnvironmentFailed, OSError):
            # Merging says what is wrong with the checkouts.
            return
        refusal = self._not_reconciled(place, meets, reconciled)
        if refusal is not None:
            raise Conflict(refusal)

    def _not_reconciled(
        self, place: SessionTeam, meets: Meets | None, reconciled: str
    ) -> str | None:
        """Hold what a reviewer says it reconciled to what it was shown.

        Returns:
            What to tell the reviewer when the work meets something it has
            not been shown as it is now; None when it may be accepted.
        """
        if self.org().opens_session(place):
            self.session_merges.show_reviewer_items(place, meets)
        if meets is None:
            return None
        run = self.org_state.run_state()
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

    async def do_request_review(self, intent: Intent) -> None:
        await self._request_review(intent.node_id)

    def ask_to_clear(
        self, place_id: str, detail: str, reconciled: str, choices: list[dict[str, str]]
    ) -> None:
        """Ask the human, once, to clear the way for accepted work into
        their own checkout. Their answer merges it, as its reviewer
        decided."""
        place = self.org().place(place_id)
        if place is None or any(e.id == MERGE_WAITS for e in place.escalations):
            return
        self.org_state.run_state().mark(
            CLEARING,
            place_id,
            json.dumps({"reconciled": reconciled, "choices": choices}),
        )
        self.org().raise_escalation(
            place_id,
            MERGE_WAITS,
            kind="critical_issue",
            title="Accepted work cannot be merged into your checkout",
            detail=detail,
            at=utc_now(),
        )
        self.org_save.save_soon()

    async def do_accept(self, intent: Intent) -> None:
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
        ic = self.team_types.writer_conversation(place)
        if ic is None or not place.base_commit:
            return ""
        files = await asyncio.to_thread(self.checkouts.changes_listed, place)
        if not files:
            return ""
        asked = self.team_types.type_of(place).review.write_up
        description = await self.model_calls.ask_agent(
            ic, bugbot.write_up_question(files, asked)
        )
        return description or ""

    async def _request_review(self, place_id: str) -> None:
        store = self.org()
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
            await asyncio.to_thread(self.session_merges.org_turn)
        if place.parent_id is not None:
            # A skill is fetched over the network the first time: not on
            # the loop, where every agent's turn would wait for it.
            await asyncio.to_thread(
                self.review_instructions, self.team_types.type_of(place)
            )
            self.org_state.run_state().mark(HELD, place_id)
            self.release_held(place.parent_id)

    async def _accept(
        self,
        place_id: str,
        needed_above: str,
        reconciled: str = "",
        choices: list[dict[str, str]] | None = None,
    ) -> None:
        """Accept the work of a team, as its reviewer asked: a step of
        resumable work, so each stage holds when it is run again."""
        store = self.org()
        place = store.place(place_id)
        above = store.place(place.parent_id) if place and place.parent_id else None
        if place is None or above is None or place.review is None:
            return
        if place.review.state == "accepted":
            await self._finish_accepted(place, above, choices or [])
        if place.review.state != "pending":
            return
        reviewer = self.team_types.reviewer_conversation(above)
        if not self._is_its_turn(place, reviewer):
            return
        allowed, meets = await self._may_merge(place, above, reviewer, reconciled)
        if not allowed:
            return
        merged, resolved = False, ()
        if self.checkouts.is_apart(place, above):
            taken = await self._merge_above(
                place, above, reviewer, meets, reconciled, choices or []
            )
            if taken is None:
                return
            merged, resolved = taken
        tip = await asyncio.to_thread(self.checkouts.version, Path(place.worktree_path))
        self._record_accepted(place, above, tip, meets, resolved, reconciled)
        if store.opens_session(place):
            self.session_merges.take_document_choices(place, choices or [])
            await asyncio.to_thread(
                self.session_merges.land_accepted, place, meets.seen if meets else None
            )
            self.session_merges.changed_since_accept.pop(place.session, None)
        self._tell_accepted(place, above, reviewer, merged)
        await self.pass_up(above.id, acted=True)
        if needed_above:
            self._tell_who_waits(place, above, needed_above)

    async def _finish_accepted(
        self, place: SessionTeam, above: SessionTeam, choices: list[dict[str, str]]
    ) -> None:
        """Carry out what follows an acceptance that was cut off after it
        was made."""
        store = self.org()
        # Cut off after the acceptance was made: what follows it is
        # still owed.
        merge = (
            self.session_merges.merge_of(place) if store.opens_session(place) else None
        )
        if merge is not None and not merge.code_landed:
            # What its reviewer made of its documents lands with it, as the
            # first time.
            self.session_merges.take_document_choices(place, choices)
            await asyncio.to_thread(self.session_merges.land_accepted, place, None)
        await self.pass_up(above.id, acted=True)
        # Whoever waited behind this work is still to hear it is their turn.
        self.tell_turns()

    def _is_its_turn(self, place: SessionTeam, reviewer: str | None) -> bool:
        """Whether no work submitted earlier is still undecided; the
        reviewer is told which when some is."""
        store = self.org()
        names = self.team_types.names_at(place.id)
        first = self._before(place)
        if first is not None:
            # The acceptance of work submitted earlier was carried out
            # since this one was checked, and left that work waiting.
            self.org_state.run_state().mark(TURN, place.id)
            self.message_delivery.notify(
                reviewer,
                accept_in_order(place, first, store.session_title(first), names=names),
            )
            return False
        return True

    async def _may_merge(
        self,
        place: SessionTeam,
        above: SessionTeam,
        reviewer: str | None,
        reconciled: str,
    ) -> tuple[bool, Meets | None]:
        """Whether the work can be merged as it is now, and what its
        reviewer had to reconcile; the reviewer is told why when not."""
        store = self.org()
        try:
            meets = await asyncio.to_thread(self._to_reconcile, place, above)
        except (EnvironmentFailed, OSError):
            meets = None
        refusal = self._not_reconciled(place, meets, reconciled)
        if refusal is not None:
            # Something came into the checkout between the review and now.
            self.message_delivery.notify(reviewer, refusal)
            return False, meets
        astray = await asyncio.to_thread(
            self.checkouts.astray, place, store.below(place.id), self.repo_dir()
        )
        if astray is not None:
            # Accepted, part of the work would not be in what is merged.
            self.message_delivery.notify(
                reviewer,
                f"The work of {place.title!r} is not accepted yet. {astray} "
                "Send it back with `review_team` and that as what to change.",
            )
            return False, meets
        return True, meets

    async def _merge_above(
        self,
        place: SessionTeam,
        above: SessionTeam,
        reviewer: str | None,
        meets: Meets | None,
        reconciled: str,
        choices: list[dict[str, str]],
    ) -> tuple[bool, tuple[str, ...]] | None:
        """Merge the work of a team into the checkout above.

        Returns:
            Whether anything was merged and the files its reviewer finished
            the merge in; None when the work is not accepted now, and whoever
            has to act was told.
        """
        place_id = place.id
        names = self.team_types.names_at(place_id)
        where = place.boundary.path or "the repository"
        opens = self.org().opens_session(place)
        resolved: tuple[str, ...] = ()
        elsewhere = await asyncio.to_thread(self._elsewhere, place, above)
        if elsewhere is not None:
            started_on, on = elsewhere
            self.ask_to_clear(
                place_id,
                f"The session started from the branch {started_on!r}, "
                "and your checkout is now "
                + ("on no branch" if on in ("", "HEAD") else f"on {on!r}")
                + f".\nSwitch it back to {started_on!r}, then answer "
                "this: the work is merged then.",
                reconciled,
                choices,
            )
            self.message_delivery.notify(
                reviewer,
                f"The work of {place.title!r} is accepted and waits for "
                "the human, whose checkout is on another branch than the "
                "one the session started from. There is nothing more "
                "for you to do.",
            )
            return None
        if not opens:
            # What is uncommitted in a session's directory is the work
            # of the team it was opened beside, and stays.
            await asyncio.to_thread(self._put_aside, above, place)
        try:
            result = await asyncio.to_thread(
                self.checkouts.merge, place, above, self._merged_as(place)
            )
        except (EnvironmentFailed, OSError) as error:
            self.message_delivery.notify(
                reviewer,
                f"The work of {place.title!r} could not be merged into your "
                f"checkout, and still waits for your review: {error}",
            )
            return None
        if result.outcome == "conflicts":
            finished = await asyncio.to_thread(self._finish_merge, place, above, result)
            if isinstance(finished, str):
                if opens:
                    self.session_merges.show_reviewer_items(place, meets)
                self.message_delivery.notify(reviewer, finished)
                return None
            result = finished
            resolved = result.files if result.outcome == "merged" else ()
        else:
            await asyncio.to_thread(self.confinement.drop_merge_checkout, place_id)
        if result.outcome == "refused" and is_division_lead(above):
            # Only the human can clear their own checkout: they are
            # asked, and their answer merges the work.
            self.ask_to_clear(
                place_id,
                f"git would not merge it as the checkout is:\n"
                f"{result.detail}\nCommit or put aside what is in "
                "the way, then answer this: the work is merged then.",
                reconciled,
                choices,
            )
            self.message_delivery.notify(
                reviewer,
                f"The work of {place.title!r} is accepted and waits for "
                "the human, whose uncommitted changes are in the way of "
                "the merge. There is nothing more for you to do.",
            )
            return None
        if result.outcome == "refused":
            self.message_delivery.notify(
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
            return None
        if result.outcome == "outside":
            issue = (
                f"These files are outside {where}, which is all your "
                "team owns. Take them out of your commits."
            )
            self.request_changes(
                place_id,
                [ReviewFinding(path=", ".join(result.files), issue=issue)],
            )
            self.message_delivery.notify(
                reviewer,
                f"The work of {place.title!r} was not taken into your "
                f"checkout: {issue} ({', '.join(result.files)}) "
                "It was sent back to the team.",
            )
            return None
        merged = result.outcome == "merged"
        if resolved:
            self.message_delivery.notify(
                self.team_types.writer_conversation(place),
                f"Your team's work did not merge by itself. The {names.reviewer} "
                "that "
                "reviewed it finished the merge in these files, which are "
                f"now as it left them: {', '.join(resolved)}",
            )
        return merged, resolved

    def _record_accepted(
        self,
        place: SessionTeam,
        above: SessionTeam,
        tip: str | None,
        meets: Meets | None,
        resolved: tuple[str, ...],
        reconciled: str,
    ) -> None:
        """Keep that the work is accepted, in the tree and in the record of
        the team above."""
        assert place.review is not None
        store = self.org()
        place_id = place.id
        store.set_review(
            place_id,
            place.review.model_copy(update={"state": "accepted", "at": utc_now()}),
            base_commit=tip or place.base_commit,
        )
        self.team_messaging.progressed(above.id)
        self.check_counts.work_accepted(place)
        if meets is not None or resolved:
            self.team_history.record_event(
                above.id,
                "reconciled",
                place,
                (reconciled or f"merged {', '.join(resolved)}")[:ASKED_KEPT],
            )
        self.team_history.record_event(
            above.id,
            "accepted",
            place,
            "at the first review"
            if place.review.round <= 1
            else f"at review {place.review.round}",
            [place.boundary.path],
            between=(place.base_commit, tip) if place.base_commit and tip else ("", ""),
        )

    def _tell_accepted(
        self,
        place: SessionTeam,
        above: SessionTeam,
        reviewer: str | None,
        merged: bool,
    ) -> None:
        """Tell the team, its reviewer and whoever waited behind the work
        that it is accepted."""
        place_id = place.id
        names = self.team_types.names_at(place_id)
        self.org_state.run_state().unmark(SHOWN, place_id)
        self.org_state.run_state().unmark(CLEARING, place_id)
        self.tell_turns()
        if not is_division_lead(above):
            self.team_messaging.count_accepted_review(above.id)
        self.message_delivery.notify(
            self.team_types.writer_conversation(place),
            f"The {names.reviewer} of {above.title!r} accepted your team's work"
            + (" and it is merged into that team's checkout." if merged else "."),
        )
        if merged:
            self.message_delivery.notify(
                reviewer, f"The work of {place.title!r} is merged into your checkout."
            )

    def _tell_who_waits(
        self, place: SessionTeam, above: SessionTeam, needed_above: str
    ) -> None:
        """Tell the team further up that work it waits on is accepted
        below it."""
        store = self.org()
        where = place.boundary.path or "the repository"
        further = store.place(above.parent_id) if above.parent_id else None
        if further is not None:
            self.message_delivery.notify(
                self.team_types.writer_conversation(further),
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
        ours = self.checkouts.version(checkout)
        merge = self.confinement.merge_checkout(place.id)
        if (
            ours is None
            or merge is None
            or (merge.ours, merge.tip) != (ours, found.tip)
            or not Path(merge.where).is_dir()
        ):
            if ours is None:
                return f"The checkout of {above.title!r} has no commit."
            where = self.confinement.merges_for(above.id) / place.id.replace("@", "-")
            merge = MergeCheckout(
                where=str(where), ours=ours, tip=found.tip, files=list(found.files)
            )
            try:
                self.checkouts.lay_out_merge(self.repo_dir(), where, ours, found.tip)
            except (EnvironmentFailed, OSError) as error:
                return (
                    f"The work of {place.title!r} does not merge with your "
                    f"checkout ({', '.join(found.files)}), and the merge could "
                    f"not be laid out for you: {error}"
                )
            self.confinement.keep_merge_checkout(place.id, merge)
            return merge_prepared(place, merge.where, merge.files)
        try:
            commit = self.checkouts.finish_merge(
                merge, self._merged_as(place), place.boundary.path
            )
        except Unfinished as error:
            return (
                f"The merge of the work of {place.title!r} in {merge.where} is "
                f"not taken yet, and the work still waits for you: {error}"
            )
        except (EnvironmentFailed, OSError) as error:
            return (
                f"The merge of the work of {place.title!r} in {merge.where} "
                f"could not be committed: {error}"
            )
        taken = self.checkouts.take_merge(checkout, commit)
        if taken.outcome != "merged":
            return taken
        self.confinement.drop_merge_checkout(place.id)
        return MergeBack("merged", found.tip, tuple(merge.files))

    def _elsewhere(
        self, place: SessionTeam, above: SessionTeam
    ) -> tuple[str, str] | None:
        """The branch a session started from and the one the human's
        checkout is on now, when they differ: the session's work is not
        merged into that."""
        session = self.org().session_of(place)
        started_on = session.branch if session else ""
        if not is_division_lead(above) or not started_on:
            return None
        on = self.checkouts.branch(above.worktree_path)
        return None if on == started_on else (started_on, on)

    async def pass_up(self, place_id: str, *, acted: bool) -> None:
        """Take a team's work into the team above without another review,
        when all of it is work of its sub-teams that was accepted.

        Each change is reviewed once, by the IC directly above the team
        that made it. A team that changed something itself submits its work
        as a whole instead.

        Args:
            acted: Whether the team's IC has just accepted the last of its
                sub-teams itself; otherwise their work came up on its own.
        """
        names = self.team_types.names_at(place_id)
        store = self.org()
        place = store.place(place_id)
        # A Division Lead place reviews; it has no work to hand on.
        if place is None or is_division_lead(place) or not store.below(place_id):
            return
        # A team that was set up and never given work owes none.
        teams = [
            team
            for team in store.below(place_id)
            if team.review is not None
            or self.team_types.writer_conversation(team) is not None
        ]
        if not teams or any(
            team.review is None or team.review.state != "accepted" for team in teams
        ):
            return
        if place.review is not None and place.review.state == "pending":
            return
        opens = store.opens_session(place)
        if opens and self.work_queue.current != BASE_LANE:
            # The work of a session goes into what the org shares.
            await self.work_queue.run_in(
                BASE_LANE, lambda: self.pass_up(place_id, acted=acted)
            )
            return
        ic = self.team_types.writer_conversation(place)
        above = store.place(place.parent_id) if place.parent_id else None
        # A file in a sub-team's folder is that team's, also while the team
        # has not been given work.
        own = await asyncio.to_thread(
            self.checkouts.own_changes, place, store.below(place_id)
        )
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
            except (EnvironmentFailed, OSError):
                meets = True
        if own or above is None or meets:
            if not acted:
                self.message_delivery.notify(
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
        team = Path(place.worktree_path)
        if self.checkouts.is_apart(place, above):
            try:
                result = await asyncio.to_thread(
                    self.checkouts.merge, place, above, place.title
                )
                problem = (
                    f"{result.outcome}: {', '.join(result.files) or result.detail}"
                    if result.outcome in ("refused", "conflicts", "outside")
                    else None
                )
            except (EnvironmentFailed, OSError) as error:
                problem = str(error)
            if problem is not None:
                self.message_delivery.notify(
                    ic,
                    "The work of all your sub-teams is accepted, but your "
                    "team's checkout does not merge into the checkout of "
                    f"{above.title!r} ({problem}). Call `submit_review`: "
                    f"the {names.reviewer} of {above.title!r} is given the merge to "
                    "finish "
                    "when it accepts the work.",
                )
                return
        tip = await asyncio.to_thread(self.checkouts.version, team)
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
                seq=self.session_merges.next_submission() if opens else 0,
            ),
            base_commit=tip or place.base_commit,
        )
        self.check_counts.work_accepted(place)
        if opens:
            await asyncio.to_thread(
                self.session_merges.land_accepted, store.place(place_id) or place, None
            )
            # As after any accept: what changes from now is said once more.
            self.session_merges.changed_since_accept.pop(place.session, None)
        self.tell_turns()
        if acted:
            self.message_delivery.notify(
                ic,
                f"Your team changed nothing itself, so its work went up to "
                f"{above.title!r} without another review.",
            )
        self.release_held(above.id)
        await self.pass_up(above.id, acted=False)

    def _put_aside(self, above: SessionTeam, team: SessionTeam) -> None:
        """Take out of the checkout above what is uncommitted in the
        directory of a team with a checkout of its own, where it would stop
        the team's accepted work from coming in, and tell the IC above."""
        checkout, path = Path(above.worktree_path), team.boundary.path
        ref = f"{ASIDE_REFS}/{team.id}/{time_stamp()}"
        try:
            files = self.checkouts.uncommitted(checkout, path)
            if not files or self.checkouts.set_aside(checkout, path, ref) is None:
                return
        except (EnvironmentFailed, OSError):
            logger.exception("Could not put aside what is in %s", path)
            return
        self.message_delivery.notify(
            self.team_types.writer_conversation(above),
            put_aside_notice(team, files, ref),
        )

    def review_instructions(self, kind: TeamTemplate | None = None) -> str:
        """What an IC reviewing a team below checks its work by: the skill
        the team's type names, then the type's own checklist.

        A skill is loaded once. One that cannot be loaded is left out, and
        the review request stands without it.

        Args:
            kind: The type of the team whose work is reviewed; the default
                type when None.
        """
        kind = kind or self.team_types.agent_types.get(DEFAULT_TYPE)
        if kind is None:
            return ""
        parts: list[str] = []
        skill = kind.review.checklist_skill
        if skill:
            # One fetch at a time: two would clone the same repository.
            with self._loading_review_instructions:
                if skill not in self._review_instructions:
                    found = (
                        self.find_review_instructions()
                        if skill in bugbot.CODE_REVIEW_SKILLS
                        else bugbot.public_skill((skill,))
                    )
                    self._review_instructions[skill] = found.strip() if found else ""
            parts.append(self._review_instructions[skill])
        parts.append(kind.review.checklist.strip())
        return "\n\n".join(part for part in parts if part)

    def load_review_instructions(self) -> None:
        """Fetch what reviewers of every type check work by, so that it is
        there when a review is requested. Safe from any thread."""
        try:
            for kind in self.team_types.agent_types.values():
                self.review_instructions(kind)
        except Exception:
            logger.exception("Could not load the review instructions")
