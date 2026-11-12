"""Director: an org tree of agent teams and their git-native blackboard.

The director supervises many agents as an organization. It owns two pieces of
state, both stored in hidden git refs so the server needs no database:

- the org tree (``refs/agents/org``), held in memory by :class:`OrgStore`;
- the blackboard (``refs/agents/blackboard``): specs, boundary memory and
  autopsies, read and written by :class:`GitBlackboard`.

``director_router`` exposes both under ``/api/director``.
"""

from openhands.agent_server.director.git_blackboard import GitBlackboard
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.service import DirectorService


__all__ = ["DirectorService", "GitBlackboard", "OrgStore"]
