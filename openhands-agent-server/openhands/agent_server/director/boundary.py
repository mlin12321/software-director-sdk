"""Which team may write where, decided from the directories the teams own."""

import re
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

from openhands.agent_server.director.models import AgentRole, OrgNode
from openhands.agent_server.director.split import is_inside


# Tools that change the file their action names, and the field that names it.
_PATH_FIELDS: dict[str, str] = {
    "file_editor": "path",
    "planning_file_editor": "path",
    "write_file": "file_path",
    "edit": "file_path",
}
_READ_COMMANDS = frozenset({"view"})
_PATCH_TARGET = re.compile(
    r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+?)\s*$", re.MULTILINE
)


def written_paths(tool_name: str, action: Mapping[str, Any]) -> list[str]:
    """The files an action of a file tool would change, as the agent named them.

    Empty for an action that only reads and for tools that do not name the
    files they change, such as the terminal.
    """
    if tool_name == "apply_patch":
        return _PATCH_TARGET.findall(str(action.get("patch", "")))
    field = _PATH_FIELDS.get(tool_name)
    if field is None or action.get("command") in _READ_COMMANDS:
        return []
    path = action.get(field)
    return [path] if isinstance(path, str) and path else []


@lru_cache(maxsize=1024)
def _glob_pattern(glob: str) -> re.Pattern[str]:
    parts: list[str] = []
    index = 0
    while index < len(glob):
        if glob.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif glob.startswith("**", index):
            parts.append(".*")
            index += 2
        elif glob[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif glob[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(glob[index]))
            index += 1
    return re.compile("".join(parts))


def glob_matches(glob: str, path: str) -> bool:
    """Whether a repo-relative path matches a boundary glob.

    ``*`` and ``?`` stay within one path segment; ``**`` crosses segments.
    """
    return _glob_pattern(glob).fullmatch(path) is not None


def owns(node: OrgNode, path: str) -> bool:
    """Whether ``path`` is the node's directory or lies in it."""
    return path == node.boundary.path or is_inside(path, node.boundary.path)


def write_refusal(
    node: OrgNode, role: AgentRole, sub_teams: Iterable[OrgNode], path: str
) -> str | None:
    """Why an agent of ``node`` may not write the repo-relative ``path``.

    Returns:
        The reason, worded for the agent, or None when it may.
    """
    if role == "manager":
        return (
            f"A Manager does not edit files, so {path!r} was not changed. Have "
            "your IC make the change."
        )
    directory = node.boundary.path or "the repository"
    if not owns(node, path):
        return (
            f"{path!r} is outside your team's boundary ({directory}), so it was "
            "not changed. Files of another team change through that team: "
            "raise an escalation."
        )
    for sub_team in sub_teams:
        if owns(sub_team, path):
            return (
                f"{path!r} belongs to your sub-team {sub_team.title!r} "
                f"(node {sub_team.id}), so it was not changed. Tell that team "
                "what you need instead."
            )
    if any(glob_matches(glob, path) for glob in node.boundary.readonly):
        return f"{path!r} is read-only for your team, so it was not changed."
    if not any(glob_matches(glob, path) for glob in node.boundary.writable):
        return (
            f"{path!r} is in your boundary but not among the files your team "
            f"may write ({', '.join(node.boundary.writable) or 'none'}), so it "
            "was not changed."
        )
    return None


def _may_take(node: OrgNode, path: str) -> bool:
    """Whether a request about ``path`` is the node's to act on."""
    if path == node.boundary.path:
        return True

    def covers(globs: list[str]) -> bool:
        # A directory is covered when something directly in it would be.
        return any(
            glob_matches(glob, path) or glob_matches(glob, f"{path}/_")
            for glob in globs
        )

    return covers(node.boundary.writable) and not covers(node.boundary.readonly)


def nearest_team(nodes: Iterable[OrgNode], path: str, *, lost: str) -> OrgNode | None:
    """The team a request made in the repo-relative directory ``path`` goes to.

    The search starts at ``path`` and moves up one directory at a time to the
    repository root. It stops at the first directory a team owns whose IC is
    running and which may write ``path``; a team that can only read it is
    passed over for the one above.

    Args:
        lost: The status of an agent whose conversation is gone.
    """
    by_directory: dict[str, list[OrgNode]] = {}
    for node in nodes:
        by_directory.setdefault(node.boundary.path, []).append(node)
    directory = path
    while True:
        for node in by_directory.get(directory, ()):
            ic = next((agent for agent in node.agents if agent.role == "ic"), None)
            if (
                ic is not None
                and ic.conversation_id is not None
                and ic.status != lost
                and _may_take(node, path)
            ):
                return node
        if not directory:
            return None
        directory = directory.rpartition("/")[0]
