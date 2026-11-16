"""A team's work on its way up: submitted, sent back, accepted, merged."""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from openhands.agent_server.director import bugbot
from openhands.agent_server.director.boundary import owns
from openhands.agent_server.director.git import (
    GitError,
    git,
    head_of,
    pathspec,
    save_and_discard,
    uncommitted,
)
from openhands.agent_server.director.git_store import (
    GitStore,
)
from openhands.agent_server.director.merge_back import merge_back
from openhands.agent_server.director.models import (
    OrgNode,
    OrgNodeInput,
    Review,
    ReviewFinding,
)
from openhands.agent_server.director.org_store import reviews_only
from openhands.agent_server.director.ownership import can_merge_in_memory, conflicts
from openhands.agent_server.director.prompts import (
    NEED_FINDING,
    accept_in_order,
    changes_requested,
    put_aside_notice,
    reconcile_request,
    review_request,
    session_work,
)
from openhands.agent_server.director.reconcile import changes_of
from openhands.agent_server.director.run_state import (
    HELD,
    Intent,
)
from openhands.agent_server.director.service.requests import Requests
from openhands.agent_server.director.service.state import (
    _REST_CHECK_SECONDS,
    _REST_CHECKS,
    ASIDE_REFS,
    ASKED_KEPT,
    MERGE_WAITS,
    _checkout_root,
    _now,
    agent_conversation,
    team_of,
)
from openhands.agent_server.director.split import Shape, shape_of
from openhands.sdk.errors import (
    Conflict,
    NotFound,
)


logger = logging.getLogger(__name__)


class Reviews(Requests):
    """Carries a team's work through its review."""

    def request_review(self, node_id: str) -> None:
        """Have the work a team submitted reviewed. Safe from any thread."""
        self._begin("request_review", node_id)

    def rest(self, node_id: str) -> None:
        """End the turn of a team's IC after the step it is in: it has
        handed its work over and acts again when something is sent to it.
        Safe from any thread."""
        if self._loop is not None:
            self._enqueue(lambda: self._rest(node_id), node_id)

    async def _rest(self, node_id: str) -> None:
        node = self.store.get(node_id)
        conversation_id = agent_conversation(node, "ic") if node else None
        if node is None or conversation_id is None or self._paused is not None:
            return
        launcher = self._require_launcher()
        if any(agent.role == "ic" and agent.kind == "acp" for agent in node.agents):
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

    def request_changes(self, node_id: str, findings: list[ReviewFinding]) -> None:
        """Send a team's work back to it with what must change.

        Safe from any thread.

        Raises:
            NotFound: The node does not exist.
            Conflict: None of its work waits for a review.
        """
        store = self.store
        node = store.get(node_id)
        if node is None:
            raise NotFound(f"Node {node_id} does not exist")
        if node.review is None or node.review.state != "pending":
            raise Conflict(f"Team {node_id} has no work waiting for a review")
        review = node.review.model_copy(
            update={"state": "changes_requested", "findings": findings, "at": _now()}
        )
        above = store.get(node.parent_id) if node.parent_id else None
        store.set_review(node_id, review)
        self.progressed(node_id)
        self.record_event(
            node.parent_id,
            "sent_back",
            node,
            f"round {review.round}, {len(findings)} to change",
        )
        for finding in findings:
            # A finding of the runtime names several files in one.
            for path in finding.path.split(", "):
                if path.strip():
                    self.record_need(node, path.strip(), NEED_FINDING)
        limit = self.settings.manager_check_ic_setbacks
        if limit and node.parent_id is not None and review.round % limit == 0:
            with self._counting:
                counted = self._counted(node.parent_id)
            self._wake_manager(
                node.parent_id,
                f"your IC has sent the work of {node.title!r} back "
                f"{review.round} times.",
                counted,
                team=node,
            )
        self.notify(
            agent_conversation(node, "ic"), changes_requested(above or node, findings)
        )

    def release_reviews(self, node_id: str | None, *, now: bool = False) -> None:
        """Hand a team's IC the reviews its sub-teams wait for, if it is
        time (see :meth:`_release_reviews`). Safe from any thread."""
        if node_id is not None and self._loop is not None:
            self._enqueue(lambda: self._released(node_id, now), f"review:{node_id}")

    async def _released(self, node_id: str, now: bool) -> None:
        self._release_reviews(node_id, now=now)

    def _release_reviews(self, node_id: str | None, *, now: bool = False) -> None:
        """Hand a team's IC the reviews its sub-teams wait for.

        They are held until every sub-team has submitted its work or is
        blocked, so that the IC is woken once for all of them. They go at
        once when the IC is in a turn anyway, or when ``now`` says that
        something else wakes it.
        """
        store = self.store
        node = store.get(node_id) if node_id else None
        if node is None or self._paused is not None:
            return
        run = self._runs()
        held = run.marked(HELD)
        teams = store.children(node.id)
        opened = store.opened_under(node.id)
        waiting: list[OrgNode] = []
        for team in (*teams, *opened):
            if team.id not in held:
                continue
            if team.review is not None and team.review.state == "pending":
                waiting.append(team)
            else:
                # Decided on, or taken back, some other way meanwhile.
                run.unmark(HELD, team.id)
        if not waiting:
            return
        reviewer = agent_conversation(node, "ic")
        # A team whose own account of its work is still being written has
        # not finished submitting.
        preparing = {
            intent.node_id
            for intent in run.open_intents()
            if intent.kind == "request_review" and intent.node_id not in held
        }
        settled = all(
            team.status == "blocked"
            or agent_conversation(team, "ic") is None
            or (
                team.review is not None
                and team.review.state in ("pending", "accepted")
                and team.id not in preparing
            )
            for team in teams
        )
        # The work of a session waits for no sub-team: no task of this IC
        # brought it.
        of_a_session = any(store.opens_session(team) for team in waiting)
        if not (now or settled or of_a_session or reviewer in self._running):
            return
        requests = []
        # In the order the work was submitted, which is the order it is
        # accepted in.
        for team in sorted(
            waiting, key=lambda team: team.review.at if team.review else ""
        ):
            run.unmark(HELD, team.id)
            if team.review is not None:
                requests.append(
                    (
                        session_work(
                            team, store.session_title(team), self._asked_for(team)
                        )
                        if store.opens_session(team)
                        else ""
                    )
                    + review_request(
                        team, team.review, self._so_far(team), self.asked_of(team)
                    )
                )
        self.tell_ic(node.id, "\n\n---\n\n".join(requests), review=True)

    def _asked_for(self, team: OrgNode) -> str:
        """What the session ``team`` belongs to was opened for."""
        session = self.store.session_of(team)
        return session.request if session else ""

    def accept_review(
        self, node_id: str, needed_above: str, reconciled: str = ""
    ) -> None:
        """Take a team's reviewed work into the team above. Safe from any
        thread.

        Args:
            needed_above: Why a team further up waits on the work; '' when
                none does.
            reconciled: What the reviewer decided about what the work meets
                in the checkout above; '' when it was asked nothing.
        """
        self._begin("accept", node_id, needed_above=needed_above, reconciled=reconciled)

    async def _landed(self, node: OrgNode) -> None:
        """Make what a session kept to itself the org's, now that its work
        is in the repository's checkout: the specs it wrote and the
        sub-teams it set up."""
        if not self.store.opens_session(node):
            return
        # The sub-teams first: what a team the org does not have wrote is
        # dropped with it.
        await self._land_shape(node)
        await asyncio.to_thread(self._land_specs, node)
        for team_id, role in await asyncio.to_thread(self._land_reports, node):
            self.reports_grew(team_id, role)

    def _shapes(self, top: OrgNode) -> tuple[list[OrgNode], Shape, Shape] | None:
        """The sub-teams a session has under the team it was opened at, in
        the order they hang together, what they make, and what the org has
        there now, which is nothing when the org no longer has the team.
        None when the session is gone."""
        store = self.store
        session = store.session_of(top)
        if session is None:
            return None
        seats = [top]
        for seat in seats:
            seats.extend(store.children(seat.id))
        teams = [team for team in [store.team(session.team_id)] if team]
        for team in teams:
            teams.extend(store.children(team.id))
        return seats, shape_of(seats), shape_of(teams)

    def _shape_to_reconcile(self, top: OrgNode) -> tuple[list[str], list[str]] | None:
        """The sub-teams the org has now and the ones a session set up,
        when both changed them since the session was opened."""
        session = self.store.session_of(top)
        shapes = self._shapes(top)
        if session is None or shapes is None:
            return None
        _, theirs, now = shapes
        if theirs == session.shape:
            # It changed none: the org's are kept as they are now.
            return None
        if not now:
            # The team it was opened at was removed from the org meanwhile.
            return (
                [f"none: the org no longer has {top.title!r} ({top.boundary.path})"],
                [f"{team} ({path})" for team, _, path in theirs],
            )
        if now == session.shape or theirs == now:
            return None
        return (
            [f"{team} ({path})" for team, _, path in now[1:]],
            [f"{team} ({path})" for team, _, path in theirs[1:]],
        )

    async def _land_shape(self, top: OrgNode) -> None:
        """Make the sub-teams a session set up the org's. A session that
        changed none leaves the org's as they are, also when the org lost
        teams of it meanwhile: its work is merged and the teams stay gone."""
        store = self.store
        session = store.session_of(top)
        shapes = self._shapes(top)
        if session is None or shapes is None:
            return
        seats, theirs, now = shapes
        if theirs == session.shape or theirs == now:
            if now:
                store.set_shape(session.id, now)
            return
        if not now and store.team(session.above_id) is None:
            # Nowhere in the org to put the team back under.
            return
        was = {team for team, _, _ in now[1:]}
        checkout = str(self._repo_dir)
        # A team the org lost while the session worked comes back with it.
        for seat in seats[1:] if now else seats:
            above = store.get(seat.parent_id or "")
            if above is None:
                continue
            team_id, above_id = team_of(seat), team_of(above)
            if store.team(team_id) is None:
                store.upsert(
                    OrgNodeInput(
                        id=team_id,
                        parent_id=above_id,
                        title=seat.title,
                        boundary=seat.boundary,
                        isolation=seat.isolation,
                        agent_profile=seat.agent_profile,
                        worktree_path=checkout,
                        branch="",
                        agents=[],
                        status="idle",
                        headline=seat.headline,
                        spec_summary=seat.spec_summary,
                        old_specs=seat.old_specs,
                    )
                )
                continue
            store.move(team_id, above_id)
            store.update_node(
                team_id,
                title=seat.title,
                boundary=seat.boundary,
                agent_profile=seat.agent_profile,
                spec_summary=seat.spec_summary,
                old_specs=seat.old_specs,
            )
        # What stays was moved under where it belongs before this.
        for gone in was - {team_of(seat) for seat in seats}:
            store.remove(gone)
        store.set_shape(session.id, theirs)
        await self._save()

    def _merged_as(self, node: OrgNode) -> str:
        """What the commit that merges a team's work calls it: the session
        too, for the work of one."""
        session = self.store.session_title(node)
        return f"{node.title} · {session}" if session else node.title

    def _to_reconcile(self, node: OrgNode, above: OrgNode) -> str | None:
        """What the IC above has to settle before the work of a team is
        merged: the files both it and the checkout above changed since it
        began, and the specs it wrote over ones that changed meanwhile.

        Returns:
            What to tell that IC; None when the work meets nothing.

        Raises:
            GitError: A checkout could not be read.
        """
        team, checkout = Path(node.worktree_path), Path(above.worktree_path)
        changes = (
            changes_of(checkout, team)
            if _checkout_root(str(team)) != _checkout_root(str(checkout))
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
                files = [path for path in changes.both if owns(node, path)]
        specs = self.specs_to_reconcile(node)
        shape = self._shape_to_reconcile(node)
        if not files and not specs and shape is None:
            return None
        return reconcile_request(
            node,
            above.worktree_path,
            changes.tip if changes else "",
            files,
            list(changes.landed) if changes else [],
            specs,
            shape,
        )

    def check_accept(self, node_id: str, reconciled: str) -> None:
        """Hold the acceptance of a team's work to what is asked of every
        one: work is accepted in the order it was submitted, and work that
        meets other work is reconciled with it first.

        Raises:
            Conflict: Work submitted before it still waits, or it meets
                work the reviewer has not said it reconciled.
        """
        store = self.store
        node = store.get(node_id)
        above = store.get(node.parent_id) if node and node.parent_id else None
        if node is None or above is None or node.review is None:
            return
        # Every session's work goes into the one checkout of the repository.
        beside = (
            [store.session(session.id)[0] for session in store.sessions()]
            if store.opens_session(node)
            else store.children(above.id)
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
            if other.id != node.id
            and other.id not in accepting
            and other.review is not None
            and other.review.state == "pending"
            and other.review.at < node.review.at
        ]
        if waiting:
            first = min(
                waiting, key=lambda other: other.review.at if other.review else ""
            )
            raise Conflict(accept_in_order(node, first))
        if reconciled:
            return
        try:
            meets = self._to_reconcile(node, above)
        except (GitError, OSError):
            # Merging says what is wrong with the checkouts.
            return
        if meets is not None:
            raise Conflict(meets)

    async def _do_request_review(self, intent: Intent) -> None:
        await self._request_review(intent.node_id)

    async def _do_accept(self, intent: Intent) -> None:
        await self._accept(
            intent.node_id,
            intent.payload["needed_above"],
            intent.payload.get("reconciled", ""),
        )

    async def _own_account(
        self, node: OrgNode
    ) -> tuple[str, list[ReviewFinding] | None]:
        """What a team's IC says of the work its team reported done.

        Returns:
            Its description of the changes, '' when it gave none, and what
            it found reviewing them, None when it gave no usable answer or
            was not asked.
        """
        ic = agent_conversation(node, "ic")
        if ic is None or not node.base_commit:
            return "", None
        files = await asyncio.to_thread(
            bugbot.changed_files,
            Path(node.worktree_path),
            node.base_commit,
            node.boundary.path,
        )
        if files is None:
            return "", None
        if not files:
            return "", []
        description = await self._ask_agent(ic, bugbot.write_up_question(files))
        instructions = await asyncio.to_thread(self._find_review_instructions)
        if instructions is None:
            return description or "", None
        above = self.store.get(node.parent_id) if node.parent_id else None
        _, contracts = await asyncio.to_thread(
            GitStore(self._repo_dir).applies_to,
            node.boundary.path,
            above.boundary.path if above else None,
        )
        specs = [
            spec
            for spec in await asyncio.to_thread(self.specs_of, node)
            if not spec.old
        ]
        answer = await self._ask_agent(
            ic,
            bugbot.review_question(
                instructions,
                specs[0].spec if specs else None,
                [agreed.contracts for agreed in contracts],
                files,
            ),
        )
        return description or "", bugbot.parse_findings(answer) if answer else None

    async def _request_review(self, node_id: str) -> None:
        store = self.store
        node = store.get(node_id)
        asked = node.review if node else None
        if node is None or asked is None:
            return
        if asked.state not in ("pending", "unreviewed"):
            return
        description, found = await self._own_account(node)
        node = store.get(node_id)
        if node is None or node.review != asked:
            return
        review = asked.model_copy(
            update={
                "description": description,
                "automated_ran": found is not None,
                "automated": found or [],
            }
        )
        node = store.set_review(node_id, review)
        if node.parent_id is not None:
            self._runs().mark(HELD, node_id)
            self._release_reviews(node.parent_id)
        elif found:
            self.notify(
                agent_conversation(node, "ic"),
                "Asked to review your team's work by the code review "
                "instructions, you found, for you to verify:\n"
                + "\n".join(f"- {finding.issue}" for finding in found),
            )

    async def _accept(
        self, node_id: str, needed_above: str, reconciled: str = ""
    ) -> None:
        store = self.store
        node = store.get(node_id)
        above = store.get(node.parent_id) if node and node.parent_id else None
        if node is None or above is None or node.review is None:
            return
        if node.review.state == "accepted":
            # Cut off after the acceptance was made: what follows it is
            # still owed.
            await self._pass_up(above.id, acted=True)
        if node.review.state != "pending":
            return
        reviewer = agent_conversation(above, "ic")
        where = node.boundary.path or "the repository"
        team, checkout = Path(node.worktree_path), Path(above.worktree_path)
        merged = False
        opens = store.opens_session(node)
        try:
            meets = await asyncio.to_thread(self._to_reconcile, node, above)
        except (GitError, OSError):
            meets = None
        if meets is not None and not reconciled:
            # Something came into the checkout between the review and now.
            self.notify(reviewer, meets)
            return
        if _checkout_root(str(team)) != _checkout_root(str(checkout)):
            if not opens:
                # What is uncommitted in a session's directory is the work
                # of the team it was opened beside, and stays.
                await asyncio.to_thread(self._put_aside, above, node)
            try:
                result = await asyncio.to_thread(
                    merge_back,
                    checkout,
                    team,
                    node.boundary.path,
                    self._merged_as(node),
                )
            except (GitError, OSError) as error:
                self.notify(
                    reviewer,
                    f"The work of {node.title!r} could not be merged into your "
                    f"checkout, and still waits for your review: {error}",
                )
                return
            if result.outcome == "refused" and reviews_only(above):
                # Only the human can clear their own checkout: they are
                # asked, and their answer merges the work.
                if all(e.id != MERGE_WAITS for e in node.escalations):
                    store.raise_escalation(
                        node_id,
                        MERGE_WAITS,
                        kind="critical_issue",
                        title="Accepted work cannot be merged into your checkout",
                        detail=(
                            f"git would not merge it as the checkout is:\n"
                            f"{result.detail}\nCommit or put aside what is in "
                            "the way, then answer this: the work is merged then."
                        ),
                        at=_now(),
                    )
                    self._save_soon()
                self.notify(
                    reviewer,
                    f"The work of {node.title!r} is accepted and waits for "
                    "the human, whose uncommitted changes are in the way of "
                    "the merge. There is nothing more for you to do.",
                )
                return
            if result.outcome == "refused":
                self.notify(
                    reviewer,
                    f"git would not merge the work of {node.title!r} into your "
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
            if result.outcome in ("conflicts", "outside"):
                issue = (
                    (
                        "These files do not merge with what was merged into "
                        f"the repository's checkout meanwhile. Merge its "
                        f"branch ({above.branch}) into yours, resolve them "
                        "and commit."
                        if reviews_only(above)
                        else "These files do not merge with the checkout of "
                        "the team above. Merge its branch into yours, "
                        "resolve them and commit."
                    )
                    if result.outcome == "conflicts"
                    else f"These files are outside {where}, which is all your "
                    "team owns. Take them out of your commits."
                )
                self.request_changes(
                    node_id, [ReviewFinding(path=", ".join(result.files), issue=issue)]
                )
                self.notify(
                    reviewer,
                    f"The work of {node.title!r} was not taken into your "
                    f"checkout: {issue} ({', '.join(result.files)}) "
                    "It was sent back to the team.",
                )
                return
            merged = result.outcome == "merged"
        tip = await asyncio.to_thread(head_of, team)
        store.set_review(
            node_id,
            node.review.model_copy(update={"state": "accepted", "at": _now()}),
            base_commit=tip or node.base_commit,
        )
        self.progressed(above.id)
        self._work_accepted(node)
        await self._landed(node)
        if meets is not None:
            self.record_event(above.id, "reconciled", node, reconciled[:ASKED_KEPT])
        self.record_event(
            above.id,
            "accepted",
            node,
            "at the first review"
            if node.review.round <= 1
            else f"at review {node.review.round}",
        )
        if not reviews_only(above):
            self.count_accepted_review(above.id)
        self.notify(
            agent_conversation(node, "ic"),
            f"The IC of {above.title!r} accepted your team's work"
            + (" and it is merged into that team's checkout." if merged else "."),
        )
        if merged:
            self.notify(
                reviewer, f"The work of {node.title!r} is merged into your checkout."
            )
        await self._pass_up(above.id, acted=True)
        further = store.get(above.parent_id) if above.parent_id else None
        if needed_above and further is not None:
            self.notify(
                agent_conversation(further, "ic"),
                f"Team {above.title!r} (node {above.id}) below you accepted the "
                f"work of its sub-team {node.title!r} in {where} and says a team "
                f"above it waits on that work: {needed_above}\nIt is in "
                f"{above.worktree_path}"
                + (f" on branch {above.branch}." if above.branch else "."),
            )

    def _own_changes(self, node: OrgNode, teams: list[OrgNode]) -> list[str] | None:
        """The files a team changed itself since its work was last taken
        up, committed or not: those in its directory that are in no
        sub-team's.

        Returns:
            None when git cannot tell.
        """
        if not node.base_commit:
            return None
        checkout = Path(node.worktree_path)
        try:
            names = git(
                checkout,
                ["diff", "--name-only", node.base_commit, "HEAD", "--"]
                + [pathspec(node.boundary.path)],
            ).splitlines()
            names += uncommitted(checkout, node.boundary.path)
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

    async def _pass_up(self, node_id: str, *, acted: bool) -> None:
        """Take a team's work into the team above without another review,
        when all of it is work of its sub-teams that was accepted.

        Each change is reviewed once, by the IC directly above the team
        that made it. A team that changed something itself submits its work
        as a whole instead.

        Args:
            acted: Whether the team's IC has just accepted the last of its
                sub-teams itself; otherwise their work came up on its own.
        """
        store = self.store
        node = store.get(node_id)
        if node is None or not store.children(node_id):
            return
        # A team that was set up and never given work owes none.
        teams = [
            team
            for team in store.children(node_id)
            if team.review is not None or agent_conversation(team, "ic") is not None
        ]
        if not teams or any(
            team.review is None or team.review.state != "accepted" for team in teams
        ):
            return
        if node.review is not None and node.review.state == "pending":
            return
        ic = agent_conversation(node, "ic")
        above = store.get(node.parent_id) if node.parent_id else None
        own = await asyncio.to_thread(self._own_changes, node, teams)
        if own is None:
            return
        # The work of a session meets what other sessions delivered: where
        # it does, the reviewer is asked even if the team changed nothing.
        meets = False
        if not own and above is not None and store.opens_session(node):
            try:
                meets = (
                    await asyncio.to_thread(self._to_reconcile, node, above)
                ) is not None
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
        team, checkout = Path(node.worktree_path), Path(above.worktree_path)
        if _checkout_root(str(team)) != _checkout_root(str(checkout)):
            try:
                result = await asyncio.to_thread(
                    merge_back, checkout, team, node.boundary.path, node.title
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
                    f"{above.title!r} ({problem}). Merge its branch into "
                    "yours, resolve what is in the way, commit, and call "
                    "`submit_review`.",
                )
                return
        tip = await asyncio.to_thread(head_of, team)
        store.set_review(
            node_id,
            Review(
                state="accepted",
                summary="The accepted work of its sub-teams, with no changes of "
                "its own.",
                passed_up=True,
                at=_now(),
            ),
            base_commit=tip or node.base_commit,
        )
        self._work_accepted(node)
        await self._landed(node)
        if acted:
            self.notify(
                ic,
                f"Your team changed nothing itself, so its work went up to "
                f"{above.title!r} without another review.",
            )
        self._release_reviews(above.id)
        await self._pass_up(above.id, acted=False)

    def _put_aside(self, above: OrgNode, team: OrgNode) -> None:
        """Take out of the checkout above what is uncommitted in the
        directory of a team with a checkout of its own, where it would stop
        the team's accepted work from coming in, and tell the IC above."""
        checkout, path = Path(above.worktree_path), team.boundary.path
        ref = f"{ASIDE_REFS}/{team.id}/{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        try:
            files = uncommitted(checkout, path)
            if not files or save_and_discard(checkout, path, ref) is None:
                return
        except (GitError, OSError):
            logger.exception("Could not put aside what is in %s", path)
            return
        self.notify(
            agent_conversation(above, "ic"),
            put_aside_notice(team, files, ref),
        )
