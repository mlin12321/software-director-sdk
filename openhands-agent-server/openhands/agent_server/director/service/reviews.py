"""A team's work on its way up: submitted, sent back, accepted, merged."""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from openhands.agent_server.director import bugbot
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
    Review,
    ReviewFinding,
)
from openhands.agent_server.director.prompts import (
    NEED_FINDING,
    changes_requested,
    put_aside_notice,
    review_request,
)
from openhands.agent_server.director.run_state import (
    HELD,
    Intent,
)
from openhands.agent_server.director.service.requests import Requests
from openhands.agent_server.director.service.state import (
    _REST_CHECK_SECONDS,
    _REST_CHECKS,
    ASIDE_REFS,
    _checkout_root,
    _now,
    agent_conversation,
)
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
        waiting: list[OrgNode] = []
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
        if not (now or settled or reviewer in self._running):
            return
        requests = []
        for team in waiting:
            run.unmark(HELD, team.id)
            if team.review is not None:
                requests.append(
                    review_request(
                        team, team.review, self._so_far(team), self.asked_of(team)
                    )
                )
        self.tell_ic(node.id, "\n\n---\n\n".join(requests))

    def accept_review(self, node_id: str, needed_above: str) -> None:
        """Take a team's reviewed work into the team above. Safe from any
        thread.

        Args:
            needed_above: Why a team further up waits on the work; '' when
                none does.
        """
        self._begin("accept", node_id, needed_above=needed_above)

    async def _do_request_review(self, intent: Intent) -> None:
        await self._request_review(intent.node_id)

    async def _do_accept(self, intent: Intent) -> None:
        await self._accept(intent.node_id, intent.payload["needed_above"])

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
        specs, contracts = await asyncio.to_thread(
            GitStore(self._repo_dir).applies_to,
            node.boundary.path,
            above.boundary.path if above else None,
        )
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

    async def _accept(self, node_id: str, needed_above: str) -> None:
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
        if _checkout_root(str(team)) != _checkout_root(str(checkout)):
            await asyncio.to_thread(self._put_aside, above, node)
            try:
                result = await asyncio.to_thread(
                    merge_back, checkout, team, node.boundary.path, node.title
                )
            except (GitError, OSError) as error:
                self.notify(
                    reviewer,
                    f"The work of {node.title!r} could not be merged into your "
                    f"checkout, and still waits for your review: {error}",
                )
                return
            if result.outcome == "refused":
                self.notify(
                    reviewer,
                    f"git would not merge the work of {node.title!r} into your "
                    f"checkout as it is:\n{result.detail}\nCommit or put aside "
                    "what is in the way, then accept it again with `review_team`.",
                )
                return
            if result.outcome in ("conflicts", "outside"):
                issue = (
                    "These files do not merge with the checkout of the team "
                    "above. Merge its branch into yours, resolve them and "
                    "commit."
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
        self.record_event(
            above.id,
            "accepted",
            node,
            "at the first review"
            if node.review.round <= 1
            else f"at review {node.review.round}",
        )
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
        if node is None or not node.child_ids:
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
        if own or above is None:
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
