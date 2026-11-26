"""A kind of organisation: what holds for every team of one tree.

The director itself says nothing of which types of team there are or what
they keep in writing. A kind of organisation does, by subclassing
:class:`OrgBase`. The ones shipped with the director are under
``director/orgs``.
"""

from abc import ABC
from pathlib import Path
from typing import ClassVar

from openhands.agent_server.director.documents.events import TeamRef
from openhands.agent_server.director.documents.tree import Tree


class OrgBase(ABC):
    """A kind of organisation. By itself it is one with no types of team
    of its own, in which nothing is refused."""

    types_folder: ClassVar[Path | None] = None
    """The folder of the types of team it ships, one YAML file for each."""

    def __init__(self, tree: Tree) -> None:
        """
        Args:
            tree: Where each team of the org stands among the others.
        """
        self.tree = tree

    def allows_read(
        self,
        reader: TeamRef,  # noqa: ARG002
        owner: TeamRef,  # noqa: ARG002
        document: str,  # noqa: ARG002
    ) -> bool:
        """Whether the team ``reader`` may read the document named
        ``document`` that ``owner`` keeps. It is asked only of a document
        whose own protocol would have ``reader`` read it."""
        return True
