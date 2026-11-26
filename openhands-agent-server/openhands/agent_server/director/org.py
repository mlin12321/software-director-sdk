"""A kind of organisation: what holds for every team of one tree.

The director itself says nothing of which types of team there are or what
they keep in writing. A kind of organisation does, by subclassing
:class:`OrgBase`. The ones shipped with the director are under
``director/orgs``.
"""

from abc import ABC
from pathlib import Path
from typing import ClassVar


class OrgBase(ABC):
    """A kind of organisation. By itself it is one with no types of team
    of its own."""

    types_folder: ClassVar[Path | None] = None
    """The folder of the types of team it ships, one YAML file for each."""
