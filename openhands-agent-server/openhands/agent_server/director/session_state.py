"""Keeps the open sessions on the machine that runs them.

A session is not part of the org: the org tree and the team memory are
stored under git refs and exchanged with other machines, and a session's
agents, conversations and checkouts are here only. So the sessions are kept
with what else the director is in the middle of (:mod:`run_state`), and what
a session changes of the org (reports, the record, specs, contracts) is
written to the team memory like anybody's.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from openhands.agent_server.director.models import SessionInput, SessionTeamInput
from openhands.agent_server.director.org_store import OrgStore


logger = logging.getLogger(__name__)


class _KeptSession(BaseModel):
    session: SessionInput
    teams: list[SessionTeamInput]


def _kept(place: SessionTeamInput) -> dict[str, Any]:
    """A place without what its agents' conversations determine.

    Whether an agent is running changes many times a minute and is read back
    from the conversations on load.
    """
    data = place.model_dump(include=set(SessionTeamInput.model_fields))
    for agent in data["agents"]:
        agent["status"] = "idle"
    if data["status"] == "working":
        data["status"] = "idle"
    data["waiting_to_start"] = False
    return data


def kept_sessions(store: OrgStore) -> dict[str, str]:
    """The open sessions of ``store`` as they are kept, by session id."""
    return {
        session.id: json.dumps(
            {
                "session": session.model_dump(include=set(SessionInput.model_fields)),
                "teams": [_kept(team) for team in session.teams],
            }
        )
        for session in store.sessions()
    }


def restore_sessions(store: OrgStore, kept: dict[str, str]) -> None:
    """Take the sessions that were kept back into ``store``. One that
    cannot be read, or whose teams are gone, is left out."""
    found: list[tuple[SessionInput, list[SessionTeamInput]]] = []
    for session_id, raw in kept.items():
        try:
            one = _KeptSession.model_validate_json(raw)
        except ValidationError:
            logger.exception("Could not read the kept session %s", session_id)
            continue
        found.append((one.session, one.teams))
    store.restore_sessions(found)
