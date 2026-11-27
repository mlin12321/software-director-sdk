"""What the sessions' agents use, and the limit a session is put to rest
at."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import Session, Stopped, Usage
from openhands.agent_server.director.org_store import OrgStore, utc_now
from openhands.agent_server.director.run_state import USE
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.team_messaging_service import (
    TeamMessagingService,
)
from openhands.agent_server.director.service.team_type_service import TeamTypeService
from openhands.agent_server.director.service.work_queue_service import WorkQueueService


# The escalation that asks the human whether a session that reached its
# token limit goes on.
LIMIT_WAITS = "limit-reached"


@dataclass
class SessionLimitService:
    """Counts what each session's agents use and holds a session at its
    limit.

    :meth:`count_use` is run on the director's timer. A session that
    reached its limit has its agents put to rest and the human asked
    (`LIMIT_WAITS`); the answer lets it go on for as much again
    (:meth:`go_on_past_limit`).
    """

    org: Callable[[], OrgStore]
    """The org as it is loaded now."""
    has_org: Callable[[], bool]
    """Whether an org is loaded; asking for one would load it."""
    launcher: Callable[[], TeamLauncher]
    """What runs the agents and knows what each has used."""
    org_state: OrgStateService
    org_settings: OrgSettingsService
    org_save: OrgSaveService
    team_types: TeamTypeService
    team_messaging: TeamMessagingService
    work_queue: WorkQueueService

    async def count_use(self) -> None:
        """Add up what each session's agents have used since it was last
        looked at, and put a session that reached its limit to rest."""
        if not self.has_org():
            return
        store = self.org()
        launcher = self.launcher()
        run = self.org_state.run_state()
        marks = run.marked(USE)
        for session in store.sessions():
            conversations = {
                conversation_id
                for team in session.teams
                for conversation_id in self.org().conversations_below(team)
            }
            now = await launcher.usage([UUID(each) for each in sorted(conversations)])
            last: dict[str, list[float]] = json.loads(marks.get(session.id) or "{}")
            added = Usage()
            for conversation_id, used in now.items():
                before = last.get(str(conversation_id), [0, 0, 0.0, 0])
                # A count that went down began again, with a server.
                added = added.plus(
                    Usage(
                        input_tokens=max(0, used.input_tokens - int(before[0])),
                        output_tokens=max(0, used.output_tokens - int(before[1])),
                        cost=max(0.0, used.cost - before[2]),
                        unpriced_tokens=max(0, used.unpriced_tokens - int(before[3])),
                    )
                )
                last[str(conversation_id)] = [
                    used.input_tokens,
                    used.output_tokens,
                    used.cost,
                    used.unpriced_tokens,
                ]
            run.mark(USE, session.id, json.dumps(last))
            # As it is now: the human may have let it go on while this waited.
            session = next(
                (each for each in store.sessions() if each.id == session.id), None
            )
            if session is None:
                continue
            limit = (
                session.cost_allowance or self.org_settings.settings.session_cost_limit
            )
            kept = store.set_session_use(session.id, session.usage.plus(added), limit)
            if kept is not None:
                self._hold_at_limit(kept)

    def _hold_at_limit(self, session: Session) -> None:
        """Put the agents of a session that reached its limit to rest and
        ask the human whether it goes on."""
        store = self.org()
        top = store.instance(session.team_id, session.id)
        limit = session.cost_allowance
        if (
            not limit
            or session.usage.cost < limit
            or top is None
            or top.stopped is not None
            or self.org_state.paused is not None
        ):
            return
        running = [
            conversation_id
            for conversation_id in self.org().conversations_below(top)
            if conversation_id in self.org_state.running
        ]
        store.set_stopped(
            top.id,
            Stopped(
                at=utc_now(),
                reason=f"The session reached its limit of ${limit:,.2f}.",
                running=sorted(running),
            ),
        )
        self.work_queue.enqueue(lambda: self.team_messaging.stop(top.id), top.id)
        used = session.usage
        if all(e.id != LIMIT_WAITS for e in top.escalations):
            store.raise_escalation(
                top.id,
                LIMIT_WAITS,
                kind="critical_issue",
                title=f"{session.title!r} reached its limit of ${limit:,.2f}",
                detail=(
                    f"Its agents have used ${used.cost:,.2f} "
                    f"({used.input_tokens:,} tokens in, "
                    f"{used.output_tokens:,} out) and are at rest.\nAnswer "
                    "this to let it go on for as much again; your answer is "
                    f"sent to its {self.team_types.team(top).names.writer}. Close the "
                    "session to end it."
                ),
                at=utc_now(),
            )
        self.org_save.save_soon()

    def go_on_past_limit(self, place_id: str) -> None:
        """Let a session that was put to rest at its limit go on for as
        much again, or without a limit when the org has none now."""
        store = self.org()
        top = store.place(place_id)
        session = store.session_of(top) if top else None
        if top is None or session is None:
            return
        limit = self.org_settings.settings.session_cost_limit
        store.set_session_use(
            session.id, session.usage, session.usage.cost + limit if limit else 0.0
        )
        stopped = top.stopped
        if stopped is not None:
            store.set_stopped(top.id, None)
            self.work_queue.enqueue(
                lambda: self.team_messaging.go_on(
                    top.id, stopped.running, own_too=True
                ),
                top.id,
            )
        self.org_save.save_soon()
