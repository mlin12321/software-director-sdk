"""The organisation that does research."""

from pathlib import Path
from typing import ClassVar

from openhands.agent_server.director.orgs.top_down import TopDownOrg


class ResearchOrg(TopDownOrg):
    """Teams that each own a research direction, and teams that build what
    their experiments run on."""

    name: ClassVar[str] = "research"
    types_folder = Path(__file__).parent / "types"
