"""The organisation that does research."""

from pathlib import Path

from openhands.agent_server.director.org import OrgBase


class ResearchOrg(OrgBase):
    """Teams that each own a research direction, and teams that build what
    their experiments run on."""

    types_folder = Path(__file__).parent / "types"
