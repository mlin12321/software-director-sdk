"""Checks an IC's proposal to split its boundary into sub-teams.

Sub-teams own directories, so that two boundaries can be told apart by
comparing paths: whether two arbitrary sets of globs overlap cannot be
decided reliably.
"""

from collections.abc import Iterable
from pathlib import PurePosixPath

from openhands.agent_server.director.models import (
    Boundary,
    OrgNode,
    SplitChild,
)
from openhands.agent_server.director.org_store import OrgStoreError


MAX_SPLIT_CHILDREN = 16


def normalize_split(
    parent: OrgNode, siblings: Iterable[OrgNode], children: list[SplitChild]
) -> list[SplitChild]:
    """The proposed sub-teams with clean paths and explicit writable globs.

    Args:
        parent: The node whose IC proposes the split.
        siblings: The sub-teams the node already has.

    Raises:
        OrgStoreError: A sub-team is not a directory strictly inside the
            parent's boundary, overlaps another sub-team, repeats a title, or
            may write outside its own directory.
    """
    if not 1 <= len(children) <= MAX_SPLIT_CHILDREN:
        raise OrgStoreError(
            f"A split needs between 1 and {MAX_SPLIT_CHILDREN} sub-teams"
        )
    siblings = list(siblings)
    taken = {sibling.boundary.path: sibling.title for sibling in siblings}
    titles = {sibling.title for sibling in siblings}
    normalized: list[SplitChild] = []
    for child in children:
        path = _clean(child.path)
        if not _inside(path, parent.boundary.path):
            raise OrgStoreError(
                f"Sub-team {child.title!r} must own a directory inside "
                f"{parent.boundary.path or 'the repository'!r}, not {child.path!r}"
            )
        for other_path, other_title in taken.items():
            if (
                path == other_path
                or _inside(path, other_path)
                or _inside(other_path, path)
            ):
                raise OrgStoreError(
                    f"Sub-teams {child.title!r} ({path}) and {other_title!r} "
                    f"({other_path}) overlap"
                )
        if child.title in titles:
            raise OrgStoreError(f"More than one sub-team is titled {child.title!r}")
        writable = [_clean(glob) for glob in child.writable] or [f"{path}/**"]
        for glob in writable:
            if not glob.startswith(f"{path}/"):
                raise OrgStoreError(
                    f"Sub-team {child.title!r} may only write under {path}/, "
                    f"not {glob!r}"
                )
        taken[path] = child.title
        titles.add(child.title)
        normalized.append(child.model_copy(update={"path": path, "writable": writable}))
    return normalized


def child_boundary(child: SplitChild) -> Boundary:
    return Boundary(path=child.path, writable=child.writable, readonly=child.readonly)


def _clean(path: str) -> str:
    """A repo-relative POSIX path without empty, ``.`` or ``..`` segments.

    Raises:
        OrgStoreError: The path is absolute or leaves the repository.
    """
    parts = PurePosixPath(path.strip()).parts
    if "/" in parts[:1] or ".." in parts:
        raise OrgStoreError(f"{path!r} is not a path inside the repository")
    return "/".join(part for part in parts if part != ".")


def _inside(path: str, ancestor: str) -> bool:
    """Whether ``path`` is strictly below ``ancestor`` ('' is the repository)."""
    if not path:
        return False
    return not ancestor or path.startswith(f"{ancestor}/")
