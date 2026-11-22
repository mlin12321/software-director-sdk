"""Which team may write where, decided from the directories the teams own."""

from collections.abc import Iterable

from openhands.agent_server.director.models import AgentRole, OrgNode
from openhands.agent_server.director.scope import (
    Closed,
    Denied,
    Scope,
    glob_matches,
    is_inside,
    named_paths,
    searched_folder,
    written_paths,
)


__all__ = [
    "glob_matches",
    "hand_over_refusal",
    "named_paths",
    "nearest_team",
    "owns",
    "scope_within",
    "searched_folder",
    "worded",
    "write_refusal",
    "written_paths",
]


def owns(node: OrgNode, path: str) -> bool:
    """Whether ``path`` is the team's directory or lies in it."""
    return path == node.boundary.path or is_inside(path, node.boundary.path)


def hand_over_refusal(path: str) -> str:
    """Why an IC may not write ``path`` from its hand-over conversation."""
    return (
        f"In hand-over mode you do not edit files, so {path!r} was not "
        "changed. Leave changes to your main conversation."
    )


def scope_within(
    node: OrgNode,
    role: AgentRole,
    sub_teams: Iterable[OrgNode],
    *,
    checkout: str = "",
) -> Scope:
    """What an agent of ``node`` may write with a file tool.

    Args:
        sub_teams: The teams below whose folders are closed to it.
    """
    return Scope(
        checkout=checkout,
        directory=node.boundary.path,
        edits=role != "manager",
        writable=tuple(node.boundary.writable),
        readonly=tuple(node.boundary.readonly),
        closed=tuple(
            Closed(team.boundary.path, team.title, team.id) for team in sub_teams
        ),
    )


def worded(denied: Denied, scope: Scope) -> str:
    """Why an agent may not do what ``denied`` names, in words for it."""
    path = denied.named if denied.relative is None else denied.relative
    match denied.kind:
        case "private":
            return f"{path!r} is the director's own, and is not yours to open."
        case "holds_private":
            return (
                f"{path or 'The workspace'!r} holds a file of the "
                "director's own: search a folder inside it instead."
            )
        case "aside":
            return hand_over_refusal(path)
        case "reads_only":
            return (
                f"A Manager does not edit files, so {path!r} was not changed. Have "
                "your IC make the change."
            )
        case "git":
            return f"{path!r} belongs to git, so it was not changed."
        case "outside":
            return (
                f"{path!r} is outside your team's boundary "
                f"({scope.directory or 'the repository'}), so it was "
                "not changed. Files of another team change through that team: "
                "raise an escalation."
            )
        case "closed":
            assert denied.closed is not None
            return (
                f"{path!r} belongs to your sub-team {denied.closed.title!r} "
                f"(id {denied.closed.id}), so it was not changed. Tell that team "
                "what you need instead."
            )
        case "readonly":
            return f"{path!r} is read-only for your team, so it was not changed."
        case "not_writable":
            return (
                f"{path!r} is in your boundary but not among the files your team "
                f"may write ({', '.join(scope.writable) or 'none'}), so it "
                "was not changed."
            )
        case "merge_of_another":
            return (
                f"{path!r} is in a merge another team's reviewer "
                "finishes, so it was not changed."
            )
        case "merged_by_itself":
            return (
                f"{path!r} merged by itself, so it was not changed. "
                f"Yours to edit in that merge: {', '.join(denied.files)}"
            )
        case "other_checkout":
            return (
                f"{path!r} is in a checkout of the org that is not your "
                "team's, so it was not changed."
            )
        case "git_directory":
            return (
                f"{path!r} is in the repository's git directory, so it "
                "was not changed. Use git."
            )


def write_refusal(
    node: OrgNode, role: AgentRole, sub_teams: Iterable[OrgNode], path: str
) -> str | None:
    """Why an agent of ``node`` may not write the repo-relative ``path``.

    Returns:
        The reason, worded for the agent, or None when it may.
    """
    scope = scope_within(node, role, sub_teams)
    denied = scope.refuse(path)
    return None if denied is None else worded(denied, scope)


def _may_take(node: OrgNode, path: str) -> bool:
    """Whether a request about ``path`` is the team's to act on."""
    if path == node.boundary.path:
        return True

    def covers(globs: list[str]) -> bool:
        # A directory is covered when something directly in it would be.
        return any(
            glob_matches(glob, path) or glob_matches(glob, f"{path}/_")
            for glob in globs
        )

    return covers(node.boundary.writable) and not covers(node.boundary.readonly)


def nearest_team(nodes: Iterable[OrgNode], path: str) -> OrgNode | None:
    """The team a request made in the repo-relative directory ``path`` goes to.

    The search starts at ``path`` and moves up one directory at a time to the
    repository root. It stops at the first directory a team owns which may
    write ``path``; a team that can only read it is passed over for the one
    above.
    """
    by_directory: dict[str, list[OrgNode]] = {}
    for node in nodes:
        by_directory.setdefault(node.boundary.path, []).append(node)
    directory = path
    while True:
        for node in by_directory.get(directory, ()):
            if _may_take(node, path):
                return node
        if not directory:
            return None
        directory = directory.rpartition("/")[0]
