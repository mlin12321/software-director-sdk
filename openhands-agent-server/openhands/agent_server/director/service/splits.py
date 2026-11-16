"""Setting up, replacing and pruning teams."""

import asyncio
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from openhands.agent_server.director.git import (
    GitError,
    save_and_discard,
    uncommitted,
)
from openhands.agent_server.director.merge_back import merge_back
from openhands.agent_server.director.models import (
    FormerTeam,
    HandOverMode,
    OrgNode,
    OrgNodeInput,
    SplitChild,
)
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    OrgStoreError,
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
from openhands.agent_server.director.service.state import (
    PRUNED_REFS,
    _child_node,
    _spec_file,
    agent_conversation,
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

    def _decline(self, node: OrgNode, reason: str) -> None:
        """Put approved sub-teams aside, tell the IC that proposed them and
        keep it in the record, where its Manager reads it."""
        proposal = node.pending_split
        self.store.update_node(node.id, pending_split=None)
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

    def _take_in(self, node: OrgNode, teams: list[OrgNode], stamp: str) -> str | None:
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
        node = store.get(node_id)
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
            taken = {team.id for team in store.all()} - gone
            teams = [_child_node(taken, node, child) for child in split.children]
            teams = [
                team.model_copy(
                    update={
                        "old_specs": await asyncio.to_thread(
                            self._specs_left_in, team.boundary.path, split.replaces
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
                    for below in store.children(team.id)
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
                stamp=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            )
        plan = intent.payload
        moves: dict[str, str] = plan["moves"]
        replaced = [team for team in map(store.get, plan["replaces"]) if team]
        if replaced and "replaced" not in intent.steps:
            watched = [team.id for team in replaced]
            for team_id in list(watched):
                watched.extend(below.id for below in store.children(team_id))
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
                if store.get(below_id) is not None:
                    store.move(below_id, node_id)
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

        teams = [OrgNodeInput.model_validate(team) for team in plan["teams"]]
        for team in teams:
            if store.get(team.id) is None:
                store.upsert(team)
        for below_id, home in moves.items():
            if store.get(below_id) is not None and store.get(home) is not None:
                store.move(below_id, home)
        self._let_go(intent)
        await self._save()
        if "handed_over" not in intent.steps:
            # A new team starts nothing by being set up. What it is for is
            # kept for the day something is sent to it.
            for team, child in zip(teams, proposal.children, strict=True):
                run.mark(TASK, team.id, child.task)
                made = store.get(team.id)
                if made is not None:
                    self._work_accepted(made)
                    self.record_event(node_id, "set_up", made, child.task[:200])
            # What a sub-team that stayed had waiting for review is now for
            # the IC of the team it reports to, which this starts if it is
            # a new team.
            for below_id, home in moves.items():
                below = store.get(below_id)
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
        node = store.get(node_id)
        proposal = node.pending_split if node else None
        if node is None or proposal is None or proposal.status != "approved":
            return
        store.update_node(node_id, pending_split=None)
        # The shell the IC starts next can no longer write what it gave away.
        with suppress(GitError, OSError):
            await asyncio.to_thread(self._confine, node_id, "ic")
        team_memory = self._memory()
        # The record each replaced team left last: the one just made.
        former = [
            records[-1]
            for team_id in proposal.replaces
            if (records := await asyncio.to_thread(team_memory.former, team_id))
        ]
        replaced = bool(proposal.contracts.strip())
        unfinished: dict[str, list[str]] = {}
        for child in proposal.children:
            with suppress(GitError, OSError):
                unfinished[child.path] = await asyncio.to_thread(
                    uncommitted, Path(node.worktree_path), child.path
                )
        self.notify(
            agent_conversation(node, "ic"),
            approved_notice(
                proposal.children,
                former,
                contracts_replaced=replaced,
                unfinished=unfinished,
            ),
        )
        if former:
            self._begin(
                "hand_over",
                node_id,
                conversation=str(uuid4()),
                ended="replaced",
                former=[[team.id, team.at] for team in former],
                children=[child.model_dump(mode="json") for child in proposal.children],
                contracts_replaced=replaced,
            )

    async def _prune(self, intent: Intent) -> None:
        store = self.store
        run = self._runs()
        node_id = intent.node_id
        launcher = self._require_launcher()
        if "planned" not in intent.steps:
            node = store.get(node_id)
            above = store.get(node.parent_id) if node and node.parent_id else None
            if node is None or above is None:
                return
            # The team first, then everything below it.
            teams = [node]
            for team in teams:
                teams.extend(store.children(team.id))
            now = datetime.now(UTC)
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
                stamp=now.strftime("%Y%m%dT%H%M%SZ"),
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
            teams = [team for team in map(store.get, plan["teams"]) if team is not None]
            saved = await asyncio.to_thread(self._discard_work, teams, plan["stamp"])
            intent = run.step_done(intent, "discarded", saved=saved)
        saved = intent.payload["saved"]
        if "deleted" not in intent.steps:
            for conversation_id in conversations:
                await launcher.delete(conversation_id)
                run.unmark(TOLD, str(conversation_id))
            intent = run.step_done(intent, "deleted")
        above = store.get(plan["parent"])
        parent_ic = agent_conversation(above, "ic") if above else None
        if "removed" not in intent.steps:
            pruned = store.get(node_id)
            for team in map(store.get, plan["teams"]):
                if team is not None:
                    await self._archive(team, "pruned", plan["stamp"])
            self.record_event(
                plan["parent"], "pruned", store.get(node_id), plan["reason"]
            )
            if pruned is not None and above is not None:
                # The directory is the team above's again, with a spec that
                # says how the pruned team worked.
                left = await asyncio.to_thread(
                    self._specs_left_in, pruned.boundary.path, [pruned]
                )
                known = {old.path for old in above.old_specs}
                store.update_node(
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
        records = await asyncio.to_thread(self._memory().former, node_id)
        former = [record for record in records if record.at == plan["stamp"]]
        self.notify(
            parent_ic,
            pruned_notice(plan["title"], where, plan["reason"], saved, former),
        )
        if former and parent_ic is not None:
            self._begin(
                "hand_over",
                plan["parent"],
                conversation=str(uuid4()),
                ended="pruned",
                former=[[team.id, team.at] for team in former],
                title=plan["title"],
                where=where,
                reason=plan["reason"],
            )

    async def _do_hand_over(self, intent: Intent) -> None:
        """Put a team's IC in hand-over mode: start the side conversation
        in which it passes on what removed teams left."""
        store, run, payload = self.store, self._runs(), intent.payload
        node = store.get(intent.node_id)
        if node is None:
            return
        conversation = payload["conversation"]
        team_memory = self._memory()
        wanted = {(team_id, at) for team_id, at in payload["former"]}
        former = [
            record
            for team_id in dict.fromkeys(team_id for team_id, _ in payload["former"])
            for record in await asyncio.to_thread(team_memory.former, team_id)
            if (record.id, record.at) in wanted
        ]
        specs = {
            team.path: await asyncio.to_thread(team_memory.read, _spec_file(team.path))
            or ""
            for team in former
        }
        children = [SplitChild(**child) for child in payload.get("children", [])]
        if payload["ended"] == "replaced":
            paths = {child.path for child in children}
            job = replaced_job(
                [
                    team
                    for team in store.children(node.id)
                    if team.boundary.path in paths
                ],
                children,
                former,
                specs,
                contracts_replaced=payload["contracts_replaced"],
            )
        else:
            job = pruned_job(
                payload["title"], payload["where"], payload["reason"], former, specs
            )
        mode = hand_over_of(node)
        before = mode.conversation_id if mode else None
        if before != conversation:
            store.set_hand_over(
                node.id, HandOverMode(conversation_id=conversation, status="idle")
            )
            run.mark(
                HANDING,
                node.id,
                json.dumps({"conversation": conversation, "given": [], "spec": False}),
            )
            if before is not None:
                # One hand-over at a time: the one before it ends here.
                with suppress(NotFound):
                    await self._require_launcher().pause(UUID(before))
        if await self._start_hand_over(node, conversation, job):
            return
        # Once more, after what a full server is given to make room.
        await asyncio.sleep(self._run_limit_retry)
        if await self._start_hand_over(node, conversation, job):
            return
        run.unmark(HANDING, node.id)
        # The mode stays on the IC as lost, for whoever looks at the team.
        store.set_hand_over(
            node.id, HandOverMode(conversation_id=conversation, status=AGENT_LOST)
        )
        self.notify(agent_conversation(node, "ic"), HAND_OVER_FAILED)

    def _hand_over_ended(self, node_id: str) -> None:
        """Tell a team's IC, in its main conversation, what its hand-over
        conversation did. Safe from any thread."""
        run = self._runs()
        kept = run.marked(HANDING).get(node_id)
        node = self.store.get(node_id)
        if kept is None or node is None or self._loop is None:
            return
        run.unmark(HANDING, node_id)
        done = json.loads(kept)
        self.notify(
            agent_conversation(node, "ic"),
            handed_over_notice(done["given"], done["spec"]),
        )

    async def _archive(
        self, team: OrgNode, ended: Literal["replaced", "pruned"], at: str
    ) -> FormerTeam:
        """Keep what a team that is being removed leaves, apart from the
        teams that come after it, and forget what was counted for it. A
        team of the same name or directory then starts from nothing but
        the old team's spec. Doing it again changes nothing."""
        team_memory = self._memory()
        kept = await asyncio.to_thread(team_memory.former, team.id)
        record = next((former for former in kept if former.at == at), None)
        if record is None:
            reports = await asyncio.to_thread(self.reports_of, team)
            record = await asyncio.to_thread(
                team_memory.record_former,
                FormerTeam(
                    id=team.id,
                    title=team.title,
                    above=team.parent_id or "",
                    path=team.boundary.path,
                    purpose=self._purpose(team),
                    ended=ended,
                    at=at,
                    review=team.review.state if team.review else None,
                    reports=reports,
                    compacted_report=await self._as_one_report(reports),
                ),
            )
        run = self._runs()
        for kind in (TASK, PROGRESS, CHECKS):
            run.unmark(kind, team.id)
        self._work_accepted(team)
        return record

    def _discard_work(self, teams: list[OrgNode], stamp: str) -> str | None:
        """Discard the uncommitted work of ``teams``, the pruned team first.

        Returns:
            The ref that keeps what the pruned team itself had uncommitted.
        """
        store = self.store
        kept: str | None = None
        for index, team in enumerate(teams):
            self._forget_shells(team)
            above = store.get(team.parent_id) if team.parent_id else None
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
