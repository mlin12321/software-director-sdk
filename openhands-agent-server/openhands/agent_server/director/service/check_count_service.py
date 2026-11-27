"""What a Manager's checks count: the reviews its IC accepted and the
escalations of each team below on its work."""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openhands.agent_server.director.models import SessionTeam
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.run_state import CHECKS, RunState


@dataclass
class CheckCountService:
    """Reads the counts a Manager is woken by, kept in the run state."""

    org: Callable[[], OrgStore]
    """The org whose teams are counted for."""
    run_state: Callable[[], RunState]
    """Where the counts are kept."""
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    """Held from reading a count to keeping what it becomes."""

    def work_accepted(self, team: SessionTeam) -> None:
        """A team's work was accepted: what its IC escalated on it is
        counted no further, since what it does next is other work."""
        if team.parent_id is None:
            return
        with self.lock:
            counted = self.counted(team.parent_id)
            if counted["escalations"].pop(team.id, None) is not None:
                self.run_state().mark(CHECKS, team.parent_id, json.dumps(counted))

    def counted(self, place_id: str) -> dict[str, Any]:
        """What a Manager's checks have counted: the reviews its IC
        accepted, and the escalations of each team below on its work."""
        try:
            kept = json.loads(self.run_state().marked(CHECKS).get(place_id) or "{}")
        except ValueError:
            kept = {}
        if not isinstance(kept, dict):
            kept = {}
        below = {team.id for team in self.org().below(place_id)}
        return {
            "accepted": int(kept.get("accepted", 0)),
            "escalations": {
                team_id: int(count)
                for team_id, count in dict(kept.get("escalations") or {}).items()
                if team_id in below
            },
        }
