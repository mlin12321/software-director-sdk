"""Which team owns which file below a node, and where their work collides."""

from functools import lru_cache
from itertools import combinations
from pathlib import Path

from openhands.agent_server.director.git import GitError, git, git_version
from openhands.agent_server.director.models import (
    BoundaryOwner,
    Collision,
    NodeOwnership,
    OrgNode,
    OwnedFile,
)


MAX_FILES = 5000
# Above this many checkouts, direct reports are compared with the node only
# and not with each other: the pairs grow with the square of the count.
MAX_CHECKOUTS_COMPARED_PAIRWISE = 12
_MERGE_TREE_MIN_GIT = (2, 38)


def node_ownership(node: OrgNode, reports: list[OrgNode]) -> NodeOwnership:
    """The files of a node's boundary by the direct report that owns them,
    and the files the committed work of two of these teams both changed in
    ways that do not merge."""
    checkout = Path(node.worktree_path)
    files, truncated = _files(checkout, node.boundary.path)
    collisions, checked = _collisions(node, reports)
    return NodeOwnership(
        node_id=node.id,
        owners=[
            BoundaryOwner(
                node_id=report.id,
                title=report.title,
                path=report.boundary.path,
                isolation=report.isolation,
                branch=report.branch,
            )
            for report in reports
        ],
        files=[OwnedFile(path=path, owner=_owner(path, reports)) for path in files],
        truncated=truncated,
        collisions=collisions,
        collisions_checked=checked,
    )


def _owner(path: str, reports: list[OrgNode]) -> str | None:
    owners = [
        report
        for report in reports
        if path == report.boundary.path or path.startswith(f"{report.boundary.path}/")
    ]
    nearest = max(owners, key=lambda report: len(report.boundary.path), default=None)
    return nearest.id if nearest else None


def _files(checkout: Path, directory: str) -> tuple[list[str], bool]:
    """The files under ``directory`` that git does not ignore."""
    spec = f":(top,literal){directory}" if directory else ":(top)"
    try:
        listed = git(
            checkout,
            ["ls-files", "-z", "--full-name", "--cached", "--others"]
            + ["--exclude-standard", "--", spec],
        )
    except (GitError, OSError):
        return [], False
    files = sorted({path for path in listed.split("\0") if path})
    return files[:MAX_FILES], len(files) > MAX_FILES


def _collisions(node: OrgNode, reports: list[OrgNode]) -> tuple[list[Collision], bool]:
    checkout = Path(node.worktree_path)
    try:
        if git_version(checkout) < _MERGE_TREE_MIN_GIT:
            return [], False
    except (GitError, OSError, RuntimeError):
        return [], False
    # Teams that share a checkout have one tip between them; the first stands
    # for it.
    tips: dict[str, str] = {}
    for team in (node, *reports):
        tip = _tip(Path(team.worktree_path))
        if tip is not None:
            tips.setdefault(tip, team.id)
    own = _tip(checkout)
    pairs = [
        (ours, theirs)
        for ours, theirs in combinations(tips, 2)
        if len(tips) <= MAX_CHECKOUTS_COMPARED_PAIRWISE or own in (ours, theirs)
    ]
    teams_by_path: dict[str, set[str]] = {}
    for ours, theirs in pairs:
        for path in _conflicts(checkout, ours, theirs):
            teams_by_path.setdefault(path, set()).update((tips[ours], tips[theirs]))
    order = [node.id, *(report.id for report in reports)]
    return [
        Collision(path=path, node_ids=sorted(teams, key=order.index))
        for path, teams in sorted(teams_by_path.items())
    ], True


def _tip(checkout: Path) -> str | None:
    try:
        return git(checkout, ["rev-parse", "--verify", "--quiet", "HEAD"]).strip()
    except (GitError, OSError):
        return None


@lru_cache(maxsize=4096)
def _conflicts(checkout: Path, ours: str, theirs: str) -> tuple[str, ...]:
    """The files that conflict when two commits are merged, without merging."""
    try:
        git(
            checkout,
            ["merge-tree", "--write-tree", "--name-only", "-z", "--no-messages"]
            + [ours, theirs],
        )
    except GitError as error:
        if error.code != 1:
            return ()
        # The tree that was written comes first, then the files.
        return tuple(path for path in error.stdout.split("\0")[1:] if path)
    return ()
