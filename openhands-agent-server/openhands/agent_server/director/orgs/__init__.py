"""The kinds of organisation shipped with the director, and the documents
their teams keep.

Everything that is particular to how these organisations work is here and
not in the director itself: which types of team there are, which documents
a team keeps, and what becomes of a document as the team's work goes on.
"""

from pathlib import Path

from openhands.agent_server.director.org import OrgBase
from openhands.agent_server.director.orgs import documents  # noqa: F401
from openhands.agent_server.director.orgs.open import OpenOrg
from openhands.agent_server.director.orgs.research import ResearchOrg
from openhands.agent_server.director.orgs.software import SoftwareOrg


SHIPPED_ORGS: tuple[type[OrgBase], ...] = (SoftwareOrg, ResearchOrg, OpenOrg)
# The kinds an org's settings may name, by what each is called.
ORG_KINDS: dict[str, type[OrgBase]] = {org.name: org for org in SHIPPED_ORGS}
# The types of team of every shipped org: a team of any org may be of any.
SHIPPED_TYPES: tuple[Path, ...] = tuple(
    org.types_folder for org in SHIPPED_ORGS if org.types_folder is not None
)
# The documents the shipped types keep, one YAML file for each.
SHIPPED_DOCUMENTS = Path(__file__).parent / "documents"

__all__ = [
    "ORG_KINDS",
    "SHIPPED_DOCUMENTS",
    "SHIPPED_ORGS",
    "SHIPPED_TYPES",
    "OpenOrg",
    "ResearchOrg",
    "SoftwareOrg",
]
