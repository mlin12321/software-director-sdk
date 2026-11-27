"""The organisation that builds software."""

from pathlib import Path
from typing import ClassVar

from openhands.agent_server.director.orgs.top_down import TopDownOrg


class SoftwareOrg(TopDownOrg):
    """Teams that build and change software, each in its folder."""

    name: ClassVar[str] = "software"
    types_folder = Path(__file__).parent / "types"
