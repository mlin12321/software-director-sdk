"""Director: an org tree of agent teams and their team memory.

The director supervises many agents as an organization. It owns two pieces of
state, both stored in hidden git refs so the server needs no database:

- the org tree (``refs/agents/org``), held in memory by :class:`OrgStore`;
- the team memory (``refs/agents/memory``): the documents teams keep and
  the agents' reports, read and written by :class:`RecordStore`.

``director_router`` exposes both under ``/api/director``.
"""

from openhands.agent_server.director.environment.records import RecordStore
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.service import DirectorService


__all__ = ["DirectorService", "RecordStore", "OrgStore"]
