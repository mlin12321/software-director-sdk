"""Sessions: lines of work the human opens at a team beside the org's own."""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from openhands.agent_server.director.git import GitError, git, head_of
from openhands.agent_server.director.headline import provisional_summary
from openhands.agent_server.director.models import (
    ChosenBy,
    OpenedSession,
    OpenSessionRequest,
    OrgNode,
    OrgNodeInput,
    Session,
)
from openhands.agent_server.director.org_store import AGENT_LOST
from openhands.agent_server.director.prompts import session_brief, session_closed
from openhands.agent_server.director.run_state import (
    CHECKS,
    PROGRESS,
    TASK,
    TOLD,
    Intent,
)
from openhands.agent_server.director.service.records import OWN_SESSION
from openhands.agent_server.director.service.splits import Splits
from openhands.agent_server.director.service.state import (
    ASKED_KEPT,
    agent_conversation,
    apart,
    team_of,
)
from openhands.sdk.errors import Conflict, NotFound


logger = logging.getLogger(__name__)

# Where the last commit of a session is kept when the session is closed
# with work that was not merged.
SESSION_REFS = "refs/agents/sessions"
_TITLE_LENGTH = 60


def session_node_id(team_id: str, session_id: str) -> str:
    """The id of the team that is there for ``team_id`` in a session."""
    return f"{team_id}@{session_id}"


class Sessions(Splits):
    """Opens and closes the human's sessions."""

    def sessions(self) -> list[Session]:
        """The sessions that are open, in the order of the org tree."""
        store = self.store
        return [
            self._session_of(node)
            for node in store.all()
            if node.session and store.opens_session(node)
        ]

    def _session_of(self, top: OrgNode) -> Session:
        teams = self.store.session(top.session)
        return Session(
            id=top.session,
            title=top.session_title,
            node_id=top.id,
            team_id=team_of(top),
            team=top.title,
            above_id=top.parent_id or "",
            status=top.status,
            review=top.review.state if top.review else None,
            teams=sum(1 for team in teams if team.agents),
        )

    async def open_session(self, request: OpenSessionRequest) -> OpenedSession:
        """Open a session at a team and start it with the human's request.

        The team and every sub-team below it are put into the org once
        more, for the session alone, under the team whose IC reviews the
        session's work: the team above, or the top team itself for a
        session opened there. Only the team the session is opened at
        starts; a sub-team starts when that IC gives it a task.

        Raises:
            NotFound: The team does not exist.
            Conflict: There is no org, it is paused, the team's sub-teams
                are being changed, or nobody can review the session's work.
            Unavailable: The director is not running.
        """
        self._require_launcher()
        store = await asyncio.to_thread(lambda: self.store)
        root = store.get(store.root_id) if store.root_id else None
        if root is None:
            raise Conflict("There is no org to open a session in")
        if self._paused is not None:
            raise Conflict("The org is paused; resume it before opening a session")
        text = request.text.strip()
        chosen_by: ChosenBy = "path"
        reason = ""
        if request.node_id is not None:
            team = store.get(request.node_id)
            if team is None:
                raise NotFound(f"Unknown node: {request.node_id}")
        elif self.settings.route_requests:
            choice = await self.choose_team(text, root)
            chosen_by, reason = choice.by, choice.reason
            team = store.get(choice.node_id) or root
        else:
            team = root
        above = store.get(team.parent_id) if team.parent_id else team
        reviewer = next(
            (agent for agent in above.agents if agent.role == "ic")
            if above
            else iter(()),
            None,
        )
        if above is None or reviewer is None or reviewer.status == AGENT_LOST:
            raise Conflict(
                f"The IC of {(above or team).title!r} reviews the work of a "
                f"session opened at {team.title!r}, and it is not running "
                "here. Start that team first."
            )
        shape = [team]
        for current in shape:
            shape.extend(store.children(current.id))
        if any(
            current.pending_split is not None
            and current.pending_split.status == "approved"
            for current in shape
        ):
            raise Conflict(
                f"The sub-teams of {team.title!r} are being set up. Open the "
                "session once that is done."
            )
        session_id = f"s{uuid4().hex[:6]}"
        title = request.title.strip() or provisional_summary(text).title
        title = title[:_TITLE_LENGTH]
        others = [
            other.session_title or OWN_SESSION
            for other in store.all()
            if team_of(other) == team_of(team) and other.agents
        ]
        run = self._runs()
        ids = {
            current.id: session_node_id(team_of(current), session_id)
            for current in shape
        }
        for current in shape:
            opened_at = current.id == team.id
            store.upsert(
                OrgNodeInput(
                    id=ids[current.id],
                    parent_id=above.id if opened_at else ids[current.parent_id or ""],
                    title=current.title,
                    boundary=current.boundary,
                    # Its work is kept apart from the team's own until the
                    # IC above has reviewed it.
                    isolation="worktree" if opened_at else current.isolation,
                    agent_profile=current.agent_profile,
                    worktree_path=above.worktree_path,
                    branch=above.branch,
                    agents=[],
                    status="idle",
                    headline=provisional_summary(text).headline
                    if opened_at
                    else current.headline,
                    spec_summary=current.spec_summary,
                    session=session_id,
                    session_title=title,
                    instance_of=team_of(current),
                )
            )
            run.mark(TASK, ids[current.id], self._purpose(current))
        top = store.get(ids[team.id])
        if top is None:
            raise NotFound(f"Unknown node: {ids[team.id]}")
        await asyncio.to_thread(
            self.record_event, above.id, "session_opened", top, text[:ASKED_KEPT]
        )
        conversation_id = self.tell_ic(
            top.id, text + session_brief(top, above, others), human=True
        )
        self._save_soon()
        return OpenedSession(
            session=self._session_of(store.get(top.id) or top),
            conversation_id=conversation_id or "",
            chosen_by=chosen_by,
            reason=reason,
        )

    def close_session(self, session_id: str) -> None:
        """Close a session: its agents are stopped and removed with its
        teams. Work of it that was accepted stays where it was merged;
        the rest is kept under a ref and is in no checkout any more. Safe
        from any thread.

        Raises:
            NotFound: No such session is open.
        """
        store = self.store
        top = next(
            (node for node in store.session(session_id) if store.opens_session(node)),
            None,
        )
        if not session_id or top is None:
            raise NotFound(f"No session {session_id!r} is open")
        self._begin("close_session", top.id)

    async def _close_session(self, intent: Intent) -> None:
        store = self.store
        run = self._runs()
        node_id = intent.node_id
        launcher = self._require_launcher()
        if "planned" not in intent.steps:
            top = store.get(node_id)
            above = store.get(top.parent_id) if top and top.parent_id else None
            if top is None or above is None:
                return
            teams = [top]
            for team in teams:
                teams.extend(store.below(team.id))
            # Fixed before anything is removed, so that closing again after
            # a restart removes the same things and records them once.
            intent = run.step_done(
                intent,
                "planned",
                title=top.session_title,
                team=top.title,
                parent=above.id,
                teams=[team.id for team in teams],
                conversations=[
                    agent.conversation_id
                    for team in teams
                    for agent in team.agents
                    if agent.conversation_id is not None
                ],
                stamp=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            )
        plan = intent.payload
        conversations = [UUID(agent) for agent in plan["conversations"]]
        if "stopped" not in intent.steps:
            self._wait_until_quiet(intent, plan["teams"])
            for conversation_id in conversations:
                await launcher.stop(conversation_id)
            intent = run.step_done(intent, "stopped")
        if "kept" not in intent.steps:
            # Only teams that started have a checkout with work of theirs.
            started = [
                team
                for team in map(store.get, plan["teams"])
                if team is not None and team.agents
            ]
            top, above = store.get(node_id), store.get(plan["parent"])
            kept = None
            # A session whose IC never started has no checkout of its own:
            # what is in the checkout above is not its work.
            if started and top is not None and above is not None and apart(top, above):
                kept = await asyncio.to_thread(
                    self._keep_unmerged, top, above, plan["stamp"]
                )
                await asyncio.to_thread(self._discard_work, started, plan["stamp"])
            intent = run.step_done(intent, "kept", kept=kept)
        if "deleted" not in intent.steps:
            for conversation_id in conversations:
                await launcher.delete(conversation_id)
                run.unmark(TOLD, str(conversation_id))
            intent = run.step_done(intent, "deleted")
        if "removed" not in intent.steps:
            top = store.get(node_id)
            for team in map(store.get, plan["teams"]):
                if team is not None:
                    for kind in (TASK, PROGRESS, CHECKS):
                        run.unmark(kind, team.id)
                    self._work_accepted(team)
            if top is not None:
                self.record_event(
                    plan["parent"],
                    "session_closed",
                    top,
                    "with work that was not merged" if intent.payload["kept"] else "",
                )
            store.remove(node_id)
            await self._save()
            intent = run.step_done(intent, "removed")
        self._settling.pop(intent.id, None)
        above = store.get(plan["parent"])
        self.notify(
            agent_conversation(above, "ic") if above else None,
            session_closed(plan["title"], plan["team"], intent.payload["kept"]),
        )

    def _keep_unmerged(self, top: OrgNode, above: OrgNode, stamp: str) -> str | None:
        """Keep the last commit of a session whose commits are not all in
        the checkout above.

        Returns:
            The ref it is kept under; None when everything was merged.
        """
        own, checkout = Path(top.worktree_path), Path(above.worktree_path)
        ref = f"{SESSION_REFS}/{top.session}/{stamp}"
        try:
            tip, merged_into = head_of(own), head_of(checkout)
            if tip is None or merged_into is None:
                return None
            if tip in git(checkout, ["merge-base", "--all", tip, merged_into]).split():
                return None
            git(checkout, ["update-ref", ref, tip])
        except (GitError, OSError):
            logger.exception("Could not keep the work of session %s", top.session)
            return None
        return ref
