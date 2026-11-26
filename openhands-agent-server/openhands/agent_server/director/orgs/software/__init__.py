"""The organisation that builds software."""

from pathlib import Path

from openhands.agent_server.director.org import OrgBase


class SoftwareOrg(OrgBase):
    """Teams that build and change software, each in its folder."""

    types_folder = Path(__file__).parent / "types"
