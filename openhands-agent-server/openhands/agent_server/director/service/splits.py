"""Setting up, replacing and pruning teams."""

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import Literal
from uuid import UUID

from openhands.agent_server.director.git import (
    GitError,
    ref_stamp,
    save_and_discard,
    uncommitted,
)
from openhands.agent_server.director.git_store import (
    normalize_store_path,
    team_key,
)
from openhands.agent_server.director.merge_back import merge_back
from openhands.agent_server.director.models import (
    FormerTeam,
    SessionTeam,
    SessionTeamInput,
)
from openhands.agent_server.director.org_store import (
    OrgStoreError,
)
from openhands.agent_server.director.prompts import (
    approved_notice,
    pruned_notice,
    review_request,
)
from openhands.agent_server.director.run_state import (
    CHECKS,
    PROGRESS,
    REPORTS,
    TASK,
    TOLD,
    Intent,
)
from openhands.agent_server.director.service.state import (
    PRUNED_REFS,
    _child_node,
    _contracts_file,
    _spec_file,
    agent_conversation,
)
from openhands.agent_server.director.service.watch import Watch
from openhands.agent_server.director.split import (
    check_limits,
    home_of,
    normalize_split,
)


logger = logging.getLogger(__name__)


class Splits(Watch):
    """Carries out what was approved or decided about teams."""

    def carry_out_split(self, node_id: str) -> None:
        """Start the sub-teams of a node's approved split. Safe from any thread."""
        self._begin("split", node_id)

    def prune(self, node_id: str, reason: str) -> None:
        """Remove a team and what is below it; its boundary goes back to
        the team above. Safe from any thread."""
        self._begin("prune", node_id, reason=reason)

    async def _do_finish_split(self, intent: Intent) -> None:
        await self._finish_split(intent.node_id)

    def _decline(self, node: SessionTeam, reason: str) -> None:
        """Put approved sub-teams aside, tell the IC that proposed them and
        keep it in the record, where its Manager reads it."""
        proposal = node.pending_split
        self.store.update_place(node.id, pending_split=None)
        self.record_event(
            node.id,
            "declined",
            None,
            f"approved, and the runtime could not set them up: {reason}",
            [child.path for child in proposal.children] if proposal else [],
        )
        self.notify(
            agent_conversation(node, "ic"),
            f"The sub-teams your Manager approved could not be set up: {reason} "
            "Nothing was changed. Tell your Manager with msg_manager, with a "
            "new proposal if you have one.",
        )

    def _take_in(
        self, node: SessionTeam, teams: list[SessionTeam], stamp: str
    ) -> str | None:
        """Bring what the teams committed in checkouts of their own into
        the checkout of ``node``, and put their uncommitted work aside.

        Nothing uncommitted is touched unless every team's work came in.

        Returns:
            Why a team's work could not be brought in; None when all was.
        """
        # A team in the node's own checkout has its files there already.
        apart = [team for team in teams if team.worktree_path != node.worktree_path]
        for team in apart:
            merged = merge_back(
                Path(node.worktree_path),
                Path(team.worktree_path),
                team.boundary.path,
                team.title,
            )
            if merged.outcome in ("conflicts", "outside", "refused"):
                detail = ", ".join(merged.files) or merged.detail
                return (
                    f"the work {team.title!r} committed does not merge into "
                    f"your checkout ({merged.outcome}: {detail})."
                )
        for team in teams:
            self._forget_shells(team)
        for team in apart:
            self._discard_work([team], stamp)
        return None

    async def _carry_out_split(self, intent: Intent) -> None:
        store = self.store
        run = self._runs()
        node_id = intent.node_id
        node = store.place(node_id)
        proposal = node.pending_split if node else None
        if node is None or proposal is None or proposal.status != "approved":
            return
        if "planned" not in intent.steps:
            try:
                # The org may have changed since the sub-teams were approved.
                split = normalize_split(store, node, proposal.children)
                check_limits(
                    store,
                    node,
                    len(split.children) - len(split.replaces),
                    self.settings,
                )
            except OrgStoreError as error:
                self._decline(node, str(error))
                return
            gone = {team.id for team in split.replaces}
            # A new team's id is no team's of the org and none of the
            # session's, so that it can become the org's when the
            # session's work is accepted.
            taken = {team.id for team in store.teams()} | {
                team.team_id
                for team in store.places()
                if team.id not in gone and team.session == node.session
            }
            teams = [_child_node(taken, node, child) for child in split.children]
            teams = [
                team.model_copy(
                    update={
                        "old_specs": await asyncio.to_thread(
                            self._specs_left_in,
                            node,
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
                teams=[team.model_dump(mode="json") for team in teams],
                replaces=sorted(gone),
                moves={
                    below.id: home_of(below, node_id, by_path)
                    for team in split.replaces
                    for below in store.below(team.id)
                },
                conversations=[
                    agent.conversation_id
                    for team in split.replaces
                    for agent in team.agents
                    if agent.conversation_id is not None
                ],
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
        plan = intent.payload
        moves: dict[str, str] = plan["moves"]
        replaced = [team for team in map(store.place, plan["replaces"]) if team]
        if replaced and "replaced" not in intent.steps:
            watched = [team.id for team in replaced]
            for team_id in list(watched):
                watched.extend(below.id for below in store.below(team_id))
            self._wait_until_quiet(intent, watched)
            refusal = await asyncio.to_thread(
                self._take_in, node, replaced, plan["stamp"]
            )
            if refusal is not None:
                self._let_go(intent)
                self._decline(node, refusal)
                return
            launcher = self._require_launcher()
            for conversation_id in map(UUID, plan["conversations"]):
                await launcher.stop(conversation_id)
                await launcher.delete(conversation_id)
                run.unmark(TOLD, str(conversation_id))
        if "replaced" not in intent.steps:
            intent = run.step_done(intent, "replaced")
        # Done whenever a replaced team is still in the tree: the tree may
        # have been saved before the step that removed it.
        if replaced:
            # Their sub-teams stay: under the node for now, then under the
            # new team that holds them.
            for below_id in moves:
                if store.place(below_id) is not None:
                    store.move_place(below_id, node_id)
            for team in replaced:
                await self._archive(team, "replaced", plan["stamp"])
                self.record_event(node_id, "replaced", team)
                store.remove(team.id)
        if proposal.contracts.strip():
            # Contracts that named teams which are gone do not stay beside
            # the new ones.
            await asyncio.to_thread(
                self._record_contracts,
                node,
                proposal.contracts,
                bool(plan["replaces"]),
            )

        teams = [SessionTeamInput.model_validate(team) for team in plan["teams"]]
        for team in teams:
            if store.place(team.id) is None:
                store.upsert(team)
        for below_id, home in moves.items():
            if store.place(below_id) is not None and store.place(home) is not None:
                store.move_place(below_id, home)
        self._let_go(intent)
        await self._save()
        if "handed_over" not in intent.steps:
            # A new team starts nothing by being set up. What it is for is
            # kept for the day something is sent to it.
            for team, child in zip(teams, proposal.children, strict=True):
                run.mark(TASK, team.id, child.task)
                made = store.place(team.id)
                if made is not None:
                    self._work_accepted(made)
                    self.record_event(node_id, "set_up", made, child.task[:200])
            # What a sub-team that stayed had waiting for review is now for
            # the IC of the team it reports to, which this starts if it is
            # a new team.
            for below_id, home in moves.items():
                below = store.place(below_id)
                if below and below.review and below.review.state == "pending":
                    self.tell_ic(
                        home,
                        review_request(
                            below,
                            below.review,
                            self._so_far(below),
                            self.asked_of(below),
                        ),
                    )
                # Under a new team it starts from now: its time without
                # progress, and what was held for it while a team above
                # was stopped.
                self.progressed(below_id)
                if plan.get("was_stopped"):
                    self._enqueue(
                        lambda below_id=below_id: self._go_on(
                            below_id, plan.get("cut_off", []), own_too=True
                        ),
                        below_id,
                    )
            intent = run.step_done(intent, "handed_over")
            self._begin("finish_split", node_id)

    async def _finish_split(self, node_id: str) -> None:
        store = self.store
        node = store.place(node_id)
        proposal = node.pending_split if node else None
        if node is None or proposal is None or proposal.status != "approved":
            return
        store.update_place(node_id, pending_split=None)
        # The shell the IC starts next can no longer write what it gave away.
        with suppress(GitError, OSError):
            await asyncio.to_thread(self._confine, node_id, "ic")
        # The record each replaced team left last: the one just made.
        kept = self.former_drafts(node.session)
        former = [
            records[-1]
            for team_id in proposal.replaces
            if (records := [r for r in kept if r.id == team_key(team_id)])
        ]
        replaced = bool(proposal.contracts.strip())
        unfinished: dict[str, list[str]] = {}
        for child in proposal.children:
            with suppress(GitError, OSError):
                unfinished[child.path] = await asyncio.to_thread(
                    uncommitted, Path(node.worktree_path), child.path
                )
        paths = {child.path for child in proposal.children}
        self.notify(
            agent_conversation(node, "ic"),
            approved_notice(
                proposal.children,
                former,
                contracts_replaced=replaced,
                unfinished=unfinished,
                teams=[
                    team
                    for team in self.store.below(node_id)
                    if team.boundary.path in paths
                ],
                specs=await self._specs_left(node, former),
            ),
        )

    async def _specs_left(
        self, node: SessionTeam, former: list[FormerTeam]
    ) -> dict[str, str]:
        """The spec each removed team wrote, by its directory, as it is in
        the checkout of the team that has the directory again."""
        return {
            team.path: await asyncio.to_thread(
                self.read_file, node, _spec_file(normalize_store_path(team.path))
            )
            or ""
            for team in former
        }

    async def _prune(self, intent: Intent) -> None:
        store = self.store
        run = self._runs()
        node_id = intent.node_id
        launcher = self._require_launcher()
        if "planned" not in intent.steps:
            node = store.place(node_id)
            above = store.place(node.parent_id) if node and node.parent_id else None
            if node is None or above is None:
                return
            # The team first, then everything below it.
            teams = [node]
            for team in teams:
                teams.extend(store.below(team.id))
            # Fixed before anything is removed, so that pruning again after
            # a restart removes the same things and records them once.
            intent = run.step_done(
                intent,
                "planned",
                title=node.title,
                path=node.boundary.path,
                parent=above.id,
                teams=[team.id for team in teams],
                conversations=[
                    agent.conversation_id
                    for team in teams
                    for agent in team.agents
                    if agent.conversation_id is not None
                ],
                stamp=ref_stamp(),
            )
        plan = intent.payload
        where = plan["path"]
        conversations = [UUID(agent) for agent in plan["conversations"]]
        if "stopped" not in intent.steps:
            # A team is removed only once it and the teams below it are at
            # rest, so nothing is cut off in the middle.
            self._wait_until_quiet(intent, plan["teams"])
            for conversation_id in conversations:
                await launcher.stop(conversation_id)
            intent = run.step_done(intent, "stopped")
        if "discarded" not in intent.steps:
            teams = [
                team for team in map(store.place, plan["teams"]) if team is not None
            ]
            saved = await asyncio.to_thread(self._discard_work, teams, plan["stamp"])
            intent = run.step_done(intent, "discarded", saved=saved)
        saved = intent.payload["saved"]
        if "deleted" not in intent.steps:
            for conversation_id in conversations:
                await launcher.delete(conversation_id)
                run.unmark(TOLD, str(conversation_id))
            intent = run.step_done(intent, "deleted")
        above = store.place(plan["parent"])
        parent_ic = agent_conversation(above, "ic") if above else None
        if "removed" not in intent.steps:
            pruned = store.place(node_id)
            for team in map(store.place, plan["teams"]):
                if team is not None:
                    await self._archive(team, "pruned", plan["stamp"])
            self.record_event(
                plan["parent"], "pruned", store.place(node_id), plan["reason"]
            )
            if pruned is not None and above is not None:
                # The directory is the team above's again, with a spec that
                # says how the pruned team worked.
                left = await asyncio.to_thread(
                    self._specs_left_in, above, pruned.boundary.path, [pruned]
                )
                known = {old.path for old in above.old_specs}
                store.update_place(
                    above.id,
                    old_specs=[
                        *above.old_specs,
                        *(old for old in left if old.path not in known),
                    ],
                )
            store.remove(node_id)
            # The shell the IC above starts next can write there again.
            with suppress(GitError, OSError):
                await asyncio.to_thread(self._confine, plan["parent"], "ic")
            await self._save()
            intent = run.step_done(intent, "removed")
        self._settling.pop(intent.id, None)
        former = [
            record
            for record in self.former_drafts(above.session if above else "")
            if record.id == team_key(node_id) and record.at == plan["stamp"]
        ]
        self.notify(
            parent_ic,
            pruned_notice(
                plan["title"],
                where,
                plan["reason"],
                saved,
                former,
                await self._specs_left(above, former) if above else None,
            ),
        )

    async def _archive(
        self, team: SessionTeam, ended: Literal["replaced", "pruned"], at: str
    ) -> FormerTeam:
        """Keep what a team that is being removed leaves, apart from the
        teams that come after it, and forget what was counted for it. A
        team of the same name or directory then starts from nothing but
        the old team's spec. Doing it again changes nothing."""
        kept = self.former_drafts(team.session)
        record = next((f for f in kept if f.id == team.team_id and f.at == at), None)
        if record is None:
            reports = await asyncio.to_thread(self.reports_of, team)
            agreed = _contracts_file(normalize_store_path(team.boundary.path))
            contracts = await asyncio.to_thread(self.read_file, team, agreed)
            record = FormerTeam(
                id=team.team_id,
                title=team.title,
                above=team_key(team.parent_id or ""),
                path=team.boundary.path,
                purpose=self._purpose(team),
                ended=ended,
                at=at,
                review=team.review.state if team.review else None,
                reports=reports,
                compacted_report=await self._as_one_report(reports),
                contracts=contracts or "",
                session=self.store.session_title(team),
                since=team.since,
                threads=self._take_threads(team.id),
            )
            # The record is the session's until its work is accepted, and
            # so is the directory without the contracts of a team that is
            # gone. The spec the team wrote stays the session's too: an
            # old one for whoever has the directory next.
            self._keep_former(team.session, record)
            if contracts:
                await asyncio.to_thread(self._draft, team.session, agreed, None)
        run = self._runs()
        for kind in (TASK, PROGRESS, CHECKS, REPORTS):
            run.unmark(kind, team.id)
        self._work_accepted(team)
        return record

    def _discard_work(self, teams: list[SessionTeam], stamp: str) -> str | None:
        """Discard the uncommitted work of ``teams``, the pruned team first.

        Returns:
            The ref that keeps what the pruned team itself had uncommitted.
        """
        store = self.store
        kept: str | None = None
        for index, team in enumerate(teams):
            self._forget_shells(team)
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
                logger.exception("Could not discard the work of node %s", team.id)
                continue
            if index == 0 and saved is not None:
                kept = ref
        return kept
