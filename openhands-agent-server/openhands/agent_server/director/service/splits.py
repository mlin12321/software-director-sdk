"""Setting up, replacing and pruning teams."""

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from openhands.agent_server.director.documents.store import DocumentRef
from openhands.agent_server.director.git import (
    GitError,
    git,
    head_of,
    ref_stamp,
    save_and_discard,
    uncommitted,
)
from openhands.agent_server.director.git_store import (
    normalize_store_path,
    team_key,
)
from openhands.agent_server.director.intents import FinishSplit, HandOver, Prune, Split
from openhands.agent_server.director.merge_back import merge_back
from openhands.agent_server.director.models import (
    FormerTeam,
    HandOverMode,
    SessionTeam,
)
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    OrgStoreError,
    conversations_of,
)
from openhands.agent_server.director.prompts import (
    HAND_OVER_FAILED,
    approved_notice,
    handed_over_notice,
    pruned_job,
    pruned_notice,
    replaced_job,
    review_request,
)
from openhands.agent_server.director.run_state import (
    CHECKS,
    HANDING,
    PROGRESS,
    TASK,
    TOLD,
    Intent,
)
from openhands.agent_server.director.service.agent_start_service import REASON_KEPT
from openhands.agent_server.director.service.resumable_work_service import Held
from openhands.agent_server.director.service.state import (
    PRUNED_REFS,
    _child_node,
    hand_over_of,
)
from openhands.agent_server.director.service.watch import Watch
from openhands.agent_server.director.split import (
    check_limits,
    home_of,
    normalize_split,
)
from openhands.sdk.errors import (
    NotFound,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class Splits(Watch):
    """Carries out what was approved or decided about teams."""

    def carry_out_split(self, place_id: str) -> None:
        """Start the sub-teams of a team's approved split. Safe from any thread."""
        self.resumable_work.begin(place_id, Split())

    def prune(self, place_id: str, reason: str) -> None:
        """Remove a team and what is below it; its boundary goes back to
        the team above. Safe from any thread."""
        self.resumable_work.begin(place_id, Prune(reason=reason))

    async def _do_finish_split(self, intent: Intent) -> None:
        await self._finish_split(intent.node_id)

    def _decline(
        self, place: SessionTeam, reason: str, left: str = "Nothing was changed."
    ) -> None:
        """Put approved sub-teams aside, tell the IC that proposed them and
        keep it in the record, where its Manager reads it."""
        names = self.team_types.team(place).names
        proposal = place.pending_split
        self.org.update_place(place.id, pending_split=None)
        self.team_history.record_event(
            place.id,
            "declined",
            None,
            f"approved, and the runtime could not set them up: {reason}",
            [child.path for child in proposal.children] if proposal else [],
        )
        self.message_delivery.notify(
            self.team_types.writer_conversation(place),
            f"The sub-teams your {names.manager} approved could not be set up: "
            f"{reason} "
            f"{left} Tell your {names.manager} with msg_manager, with a new "
            "proposal if "
            "you have one.",
        )

    def _take_in(
        self, place: SessionTeam, teams: list[SessionTeam], stamp: str
    ) -> str | None:
        """Bring what the teams committed in checkouts of their own into
        the checkout of ``node``, and put their uncommitted work aside.

        Nothing uncommitted is touched unless every team's work came in.

        Returns:
            Why a team's work could not be brought in; None when all was.
        """
        # A team in the team's own checkout has its files there already.
        apart = [team for team in teams if team.worktree_path != place.worktree_path]
        checkout = Path(place.worktree_path)
        before = head_of(checkout)
        for team in apart:
            refusal = None
            try:
                merged = merge_back(
                    checkout, Path(team.worktree_path), team.boundary, team.title
                )
            except (GitError, OSError) as error:
                refusal = f"could not be read ({error})."
            else:
                if merged.outcome in ("conflicts", "outside", "refused"):
                    detail = ", ".join(merged.files) or merged.detail
                    refusal = (
                        f"does not merge into your checkout ({merged.outcome}: "
                        f"{detail})."
                    )
            if refusal:
                if before and head_of(checkout) != before:
                    # The teams merged before it are taken out again, and
                    # what is uncommitted stays.
                    git(checkout, ["reset", "--merge", "--quiet", before])
                return f"the work {team.title!r} committed {refusal}"
        for team in teams:
            self.confinement.forget_shells(team)
        for team in apart:
            self._discard_work([team], stamp)
        return None

    async def _do_split(self, intent: Intent) -> None:
        try:
            await self._split(intent)
        except Held:
            raise
        except Exception as error:
            # Or the proposal stays approved for good: the IC is refused
            # every other one and never told of this.
            place = self.org.place(intent.node_id)
            if place is not None and place.pending_split is not None:
                self._decline(
                    place,
                    f"{type(error).__name__}: {error}."[:REASON_KEPT],
                    "Some of it may be done: read your team's context.",
                )
            raise

    async def _split(self, intent: Intent) -> None:
        store = self.org
        run = self.org_state.run_state()
        place_id = intent.node_id
        place = store.place(place_id)
        proposal = place.pending_split if place else None
        if place is None or proposal is None or proposal.status != "approved":
            return
        if "planned" not in intent.steps:
            if self.message_delivery.is_settling(place_id):
                # What is closing or pruning it fixed what it removes first.
                self._decline(place, "The team is being closed or removed.")
                return
            try:
                # The org may have changed since the sub-teams were approved.
                split = normalize_split(store, place, proposal.children)
                check_limits(
                    store,
                    place,
                    len(split.children) - len(split.replaces),
                    self.org_settings.settings,
                )
            except OrgStoreError as error:
                self._decline(place, str(error))
                return
            gone = {team.id for team in split.replaces}
            # A new team's id is no team's of the org and none of the
            # session's, so that it can become the org's when the
            # session's work is accepted.
            taken = {team.id for team in store.teams()} | {
                team.team_id
                for team in store.places()
                if team.id not in gone and team.session == place.session
            }
            teams = [_child_node(taken, place, child) for child in split.children]
            teams = [
                team.model_copy(
                    update={
                        "leftover_documents": await asyncio.to_thread(
                            self.session_documents.documents_left_in,
                            place,
                            team.boundary.path,
                            split.replaces,
                        )
                    }
                )
                for team in teams
            ]
            by_path = {team.boundary.path: team.id for team in teams}
            # Fixed before anything changes, so that carrying the split out
            # again makes the same teams, removes the same and no others.
            intent = run.step_done(
                intent,
                "planned",
                teams=teams,
                replaces=sorted(gone),
                moves={
                    below.id: home_of(below, place_id, by_path)
                    for team in split.replaces
                    for below in store.below(team.id)
                },
                conversations=conversations_of(split.replaces),
                # Agents a replaced team that was stopped had cut off: the
                # sub-teams that stay go on.
                cut_off=sorted(
                    {
                        conversation_id
                        for team in split.replaces
                        if team.stopped is not None
                        for conversation_id in team.stopped.running
                    }
                ),
                was_stopped=any(team.stopped is not None for team in split.replaces),
                stamp=ref_stamp(),
            )
        plan = intent.of(Split)
        moves = plan.moves
        # A new team may have the id of one it replaces, and is not that one.
        made = {team.id: team.since for team in plan.teams}
        replaced = [
            team
            for team in map(store.place, plan.replaces)
            if team and team.since != made.get(team.id)
        ]
        if replaced and "replaced" not in intent.steps:
            watched = [team.id for team in replaced]
            for team_id in list(watched):
                watched.extend(below.id for below in store.below(team_id))
            self.message_delivery.wait_until_quiet(intent, watched)
            refusal = await asyncio.to_thread(
                self._take_in, place, replaced, plan.stamp
            )
            if refusal is not None:
                self.message_delivery.let_go(intent)
                self._decline(place, refusal)
                return
            launcher = self._require_launcher()
            for conversation_id in map(UUID, plan.conversations):
                await launcher.stop(conversation_id)
                await launcher.delete(conversation_id)
                run.unmark(TOLD, str(conversation_id))
                run.drop_held(str(conversation_id))
        if "replaced" not in intent.steps:
            intent = run.step_done(intent, "replaced")
        # Done whenever a replaced team is still in the tree: the tree may
        # have been saved before the step that removed it.
        if replaced:
            # Their sub-teams stay: under the team for now, then under the
            # new team that holds them.
            for below_id in moves:
                if store.place(below_id) is not None:
                    store.move_place(below_id, place_id)
            for team in replaced:
                await self._archive(team, "replaced", plan.stamp)
                self.team_history.record_event(place_id, "replaced", team)
                store.remove(team.id)

        teams = plan.teams
        for team in teams:
            if store.place(team.id) is None:
                store.upsert(team)
        for below_id, home in moves.items():
            if store.place(below_id) is not None and store.place(home) is not None:
                store.move_place(below_id, home)
        self.message_delivery.let_go(intent)
        await self.org_save.save()
        if "handed_over" not in intent.steps:
            # A new team starts nothing by being set up. What it is for is
            # kept for the day something is sent to it.
            for team, child in zip(teams, proposal.children, strict=True):
                run.mark(TASK, team.id, child.task)
                made = store.place(team.id)
                if made is not None:
                    self.check_counts.work_accepted(made)
                    self.team_history.record_event(
                        place_id, "set_up", made, child.task[:200]
                    )
            # What a sub-team that stayed had waiting for review is now for
            # the IC of the team it reports to, which this starts if it is
            # a new team.
            for below_id, home in moves.items():
                below, above = store.place(below_id), store.place(home)
                if below and above and below.review and below.review.state == "pending":
                    self.team_messaging.tell(
                        home,
                        self.team_types.team(above).reviewer,
                        review_request(
                            below,
                            below.review,
                            self.team_context.so_far(below),
                            self.team_context.asked_of(below),
                            self.review_instructions(self.team_types.type_of(below)),
                            self.team_types.kept_documents(above).read_by_sub_teams,
                        ),
                    )
                # Under a new team it starts from now: its time without
                # progress, and what was held for it while a team above
                # was stopped.
                self.team_messaging.progressed(below_id, by_above=True)
                if plan.was_stopped:
                    self.work_queue.enqueue(
                        lambda below_id=below_id: self.team_messaging.go_on(
                            below_id, plan.cut_off, own_too=True
                        ),
                        below_id,
                    )
            # Begun first: a server that ends between the two tells the IC
            # twice rather than never.
            self.resumable_work.begin(place_id, FinishSplit())
            intent = run.step_done(intent, "handed_over")

    async def _finish_split(self, place_id: str) -> None:
        store = self.org
        place = store.place(place_id)
        proposal = place.pending_split if place else None
        if place is None or proposal is None or proposal.status != "approved":
            return
        # The shell the IC starts next can no longer write what it gave away.
        with suppress(GitError, OSError):
            await asyncio.to_thread(self.confinement.confine, place_id)
        # The record each replaced team left last: the one just made.
        kept = self.session_drafts.former(place.session)
        former = [
            records[-1]
            for team_id in proposal.replaces
            if (records := [r for r in kept if r.id == team_key(team_id)])
        ]
        unfinished: dict[str, list[str]] = {}
        for child in proposal.children:
            with suppress(GitError, OSError):
                unfinished[child.path] = await asyncio.to_thread(
                    uncommitted, Path(place.worktree_path), child.path
                )
        self.message_delivery.notify(
            self.team_types.writer_conversation(place),
            approved_notice(
                proposal.children,
                former,
                names=self.team_types.team(place).names,
                rules=self.team_types.document_rules(place),
                unfinished=unfinished,
            ),
        )
        named = [(team.id, team.at) for team in former]
        # Begun already by a server that ended before the split was saved.
        begun = any(
            isinstance(intent.payload, HandOver)
            and intent.node_id == place_id
            and intent.payload.former == named
            for intent in self.org_state.run_state().open_intents()
        )
        if former and not begun:
            self.resumable_work.begin(
                place_id,
                HandOver(
                    conversation=str(uuid4()),
                    ended="replaced",
                    former=named,
                    children=proposal.children,
                ),
            )
        # Last: until the IC is told, a server that ends has this to do again.
        store.update_place(place_id, pending_split=None)

    async def _do_prune(self, intent: Intent) -> None:
        store = self.org
        run = self.org_state.run_state()
        place_id = intent.node_id
        launcher = self._require_launcher()
        if "planned" not in intent.steps:
            place = store.place(place_id)
            above = store.place(place.parent_id) if place and place.parent_id else None
            if place is None or above is None:
                return
            # The team first, then everything below it.
            teams = [place]
            for team in teams:
                teams.extend(store.below(team.id))
            # Fixed before anything is removed, so that pruning again after
            # a restart removes the same things and records them once.
            intent = run.step_done(
                intent,
                "planned",
                title=place.title,
                path=place.boundary.path,
                parent=above.id,
                teams=[team.id for team in teams],
                conversations=conversations_of(teams),
                stamp=ref_stamp(),
            )
        plan = intent.of(Prune)
        where = plan.path
        conversations = [UUID(agent) for agent in plan.conversations]
        if "stopped" not in intent.steps:
            # A team is removed only once it and the teams below it are at
            # rest, so nothing is cut off in the middle.
            self.message_delivery.wait_until_quiet(intent, plan.teams)
            for conversation_id in conversations:
                await launcher.stop(conversation_id)
            intent = run.step_done(intent, "stopped")
        if "discarded" not in intent.steps:
            teams = [team for team in map(store.place, plan.teams) if team is not None]
            saved = await asyncio.to_thread(self._discard_work, teams, plan.stamp)
            intent = run.step_done(intent, "discarded", saved=saved)
        saved = intent.of(Prune).saved
        if "deleted" not in intent.steps:
            for conversation_id in conversations:
                await launcher.delete(conversation_id)
                run.unmark(TOLD, str(conversation_id))
                run.drop_held(str(conversation_id))
            intent = run.step_done(intent, "deleted")
        above = store.place(plan.parent)
        parent_ic = self.team_types.writer_conversation(above) if above else None
        if "removed" not in intent.steps:
            pruned = store.place(place_id)
            for team in map(store.place, plan.teams):
                if team is not None:
                    await self._archive(team, "pruned", plan.stamp)
            if "recorded" not in intent.steps:
                self.team_history.record_event(
                    plan.parent, "pruned", pruned, plan.reason
                )
                intent = run.step_done(intent, "recorded")
            if pruned is not None and above is not None:
                # The directory is the team above's again, with what the
                # pruned team left there of how it worked.
                left = await asyncio.to_thread(
                    self.session_documents.documents_left_in,
                    above,
                    pruned.boundary.path,
                    [pruned],
                )
                known = {(old.name, old.path) for old in above.leftover_documents}
                store.update_place(
                    above.id,
                    leftover_documents=[
                        *above.leftover_documents,
                        *(old for old in left if (old.name, old.path) not in known),
                    ],
                )
            store.remove(place_id)
            # The shell the IC above starts next can write there again.
            with suppress(GitError, OSError):
                await asyncio.to_thread(self.confinement.confine, plan.parent)
            await self.org_save.save()
            intent = run.step_done(intent, "removed")
        self.message_delivery.stop_waiting(intent.id)
        if "told" in intent.steps:
            return
        former = [
            record
            for record in self.session_drafts.former(above.session if above else "")
            if record.id == team_key(place_id) and record.at == plan.stamp
        ]
        self.message_delivery.notify(
            parent_ic,
            pruned_notice(
                plan.title,
                where,
                plan.reason,
                saved,
                former,
                names=self.team_types.names_at(intent.node_id),
                rules=self.team_types.document_rules(above) if above else (),
            ),
        )
        if former and parent_ic is not None:
            self.resumable_work.begin(
                plan.parent,
                HandOver(
                    conversation=str(uuid4()),
                    ended="pruned",
                    former=[(team.id, team.at) for team in former],
                    title=plan.title,
                    where=where,
                    reason=plan.reason,
                ),
            )
        run.step_done(intent, "told")

    async def _do_hand_over(self, intent: Intent) -> None:
        """Put a team's IC in hand-over mode: start the side conversation
        in which it passes on what removed teams left."""
        store, run, payload = self.org, self.org_state.run_state(), intent.of(HandOver)
        place = store.place(intent.node_id)
        if place is None:
            return
        conversation = payload.conversation
        wanted = set(payload.former)
        # The records are the session's own until its work is accepted.
        former = [
            record
            for record in self.session_drafts.former(place.session)
            if (record.id, record.at) in wanted
        ]
        stays = self.team_types.kept_documents(place).stay_when_removed
        # What each of them wrote that stays in its folder, by the folder.
        left = {
            team.path: [
                (name, text)
                for name in stays
                if (
                    text := await asyncio.to_thread(
                        self.session_documents.read_file,
                        place,
                        DocumentRef(normalize_store_path(team.path), name).path,
                    )
                )
            ]
            for team in former
        }
        children = payload.children
        if payload.ended == "replaced":
            paths = {child.path for child in children}
            job = replaced_job(
                [team for team in store.below(place.id) if team.boundary.path in paths],
                children,
                former,
                left,
                names=self.team_types.names_at(intent.node_id),
                rules=self.team_types.document_rules(place),
                stays=stays,
            )
        else:
            job = pruned_job(
                payload.title,
                payload.where,
                payload.reason,
                former,
                left,
                names=self.team_types.names_at(intent.node_id),
                rules=self.team_types.document_rules(place),
                stays=stays,
            )
        mode = hand_over_of(place)
        before = mode.conversation_id if mode else None
        if before != conversation:
            store.set_hand_over(
                place.id, HandOverMode(conversation_id=conversation, status="idle")
            )
            run.mark(
                HANDING,
                place.id,
                json.dumps({"conversation": conversation, "given": [], "written": []}),
            )
            if before is not None:
                # One hand-over at a time: the one before it ends here.
                with suppress(NotFound):
                    await self._require_launcher().pause(UUID(before))
        if await self.agent_start.start_hand_over(place, conversation, job):
            return
        # Once more, after what a full server is given to make room.
        await asyncio.sleep(self._run_limit_retry)
        if await self.agent_start.start_hand_over(place, conversation, job):
            return
        run.unmark(HANDING, place.id)
        # The mode stays on the IC as lost, for whoever looks at the team.
        store.set_hand_over(
            place.id, HandOverMode(conversation_id=conversation, status=AGENT_LOST)
        )
        self.message_delivery.notify(
            self.team_types.writer_conversation(place), HAND_OVER_FAILED
        )

    def _hand_over_ended(self, place_id: str) -> None:
        """Tell a team's IC, in its main conversation, what its hand-over
        conversation did. Safe from any thread."""
        run = self.org_state.run_state()
        kept = run.marked(HANDING).get(place_id)
        place = self.org.place(place_id)
        if kept is None or place is None or self._loop is None:
            return
        run.unmark(HANDING, place_id)
        done = json.loads(kept)
        self.message_delivery.notify(
            self.team_types.writer_conversation(place),
            handed_over_notice(
                done["given"],
                done["written"],
                names=self.team_types.names_at(place_id),
                rules=[
                    protocol.document.rules_by_member
                    for protocol in self.team_documents.kept_by(
                        self.team_types.reader_of(place)
                    )
                    if protocol.document.name not in done["written"]
                ],
            ),
        )

    async def _archive(
        self, team: SessionTeam, ended: Literal["replaced", "pruned"], at: str
    ) -> FormerTeam:
        """Keep what a team that is being removed leaves, apart from the
        teams that come after it, and forget what was counted for it. A
        team of the same name or directory then starts from nothing but
        what the old team left in its folder. Doing it again changes nothing."""
        kept = self.session_drafts.former(team.session)
        record = next((f for f in kept if f.id == team.team_id and f.at == at), None)
        if record is None:
            reports = await asyncio.to_thread(self.team_reports.reports_of, team)
            # What each of the team's documents keeps of itself. A document
            # that keeps nothing stays where it is: an old one for whoever
            # has the directory next.
            kept = await asyncio.to_thread(
                self.team_documents.when_team_is_removed,
                self.team_types.reader_of(team),
            )
            record = FormerTeam(
                id=team.team_id,
                title=team.title,
                above=team_key(team.parent_id or ""),
                path=team.boundary.path,
                purpose=self.team_messaging.purpose(team),
                ended=ended,
                at=at,
                review=team.review.state if team.review else None,
                reports=reports,
                compacted_report=await self.team_summaries.as_one_report(reports),
                documents={name: text for name, text in kept.items() if text},
                session=self.org.session_title(team),
                since=team.since,
                threads=self.session_drafts.take_threads(team.id),
            )
            # The record is the session's until its work is accepted.
            self.session_drafts.keep_former(team.session, record)
        run = self.org_state.run_state()
        for kind in (TASK, PROGRESS, CHECKS):
            run.unmark(kind, team.id)
        self.session_drafts.forget_reports(team.id)
        self.check_counts.work_accepted(team)
        return record

    def _discard_work(self, teams: list[SessionTeam], stamp: str) -> str | None:
        """Discard the uncommitted work of ``teams``, the pruned team first.

        Returns:
            The ref that keeps what the pruned team itself had uncommitted.
        """
        store = self.org
        kept: str | None = None
        for index, team in enumerate(teams):
            self.confinement.forget_shells(team)
            above = store.place(team.parent_id) if team.parent_id else None
            # A team in the checkout of the one above is covered by it.
            if index and above and above.worktree_path == team.worktree_path:
                continue
            ref = f"{PRUNED_REFS}/{team.id}/{stamp}"
            try:
                saved = save_and_discard(
                    Path(team.worktree_path), team.boundary.path, ref
                )
            except (GitError, OSError):
                logger.exception("Could not discard the work of team %s", team.id)
                continue
            if index == 0 and saved is not None:
                kept = ref
        return kept
