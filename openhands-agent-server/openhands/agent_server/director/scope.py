"""What one agent may write, as plain paths and globs.

A scope says nothing of who the agent is or why it may write there: the
org works that out and the confinement holds the agent to it.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal


# Tools that change the file their action names, and the field that names it.
_PATH_FIELDS: dict[str, str] = {
    "file_editor": "path",
    "planning_file_editor": "path",
    "write_file": "file_path",
    "edit": "file_path",
}
_READ_COMMANDS = frozenset({"view"})
# Tools that only read the file their action names, and the field that names it.
_READ_FIELDS: dict[str, str] = {"read_file": "file_path"}
# Tools that search a folder, and the field that names it; none is the
# workspace.
_SEARCH_FIELDS: dict[str, str] = {"grep": "path", "glob": "path"}
_PATCH_TARGET = re.compile(
    r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+?)\s*$", re.MULTILINE
)

DeniedKind = Literal[
    # A file of the director's own, or a search of a folder that holds it.
    "private",
    "holds_private",
    # A conversation that edits no file anywhere.
    "aside",
    # In the agent's own checkout.
    "reads_only",
    "git",
    "outside",
    "closed",
    "readonly",
    "not_writable",
    # In a merge that is being finished.
    "merge_of_another",
    "merged_by_itself",
    # Elsewhere in the repository.
    "other_checkout",
    "git_directory",
]


def named_paths(tool_name: str, action: Mapping[str, Any]) -> list[str]:
    """The files an action of a file tool reads or changes, as the agent
    named them. Empty for tools that do not name them, such as the terminal."""
    if tool_name == "apply_patch":
        return _PATCH_TARGET.findall(str(action.get("patch", "")))
    path = action.get((_PATH_FIELDS | _READ_FIELDS).get(tool_name, ""))
    return [path] if isinstance(path, str) and path else []


def searched_folder(tool_name: str, action: Mapping[str, Any]) -> str | None:
    """The folder an action of a search tool looks through, as the agent
    named it; ``""`` for the workspace, None for a tool that searches none."""
    if tool_name not in _SEARCH_FIELDS:
        return None
    folder = action.get(_SEARCH_FIELDS[tool_name])
    return folder if isinstance(folder, str) else ""


def written_paths(tool_name: str, action: Mapping[str, Any]) -> list[str]:
    """The files an action of a file tool would change, as the agent named
    them. Empty for an action that only reads."""
    if action.get("command") in _READ_COMMANDS or tool_name in _READ_FIELDS:
        return []
    return named_paths(tool_name, action)


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


def is_inside(path: str, ancestor: str) -> bool:
    """Whether ``path`` is strictly below ``ancestor`` ('' is the repository)."""
    if not path:
        return False
    return not ancestor or path.startswith(f"{ancestor}/")


def holds(directory: str, path: str) -> bool:
    """Whether ``path`` is ``directory`` or lies in it."""
    return path == directory or is_inside(path, directory)


@dataclass(frozen=True)
class Closed:
    """A folder inside an agent's own that another holds."""

    path: str
    title: str
    id: str


@dataclass(frozen=True)
class MergeArea:
    """A checkout a merge is being finished in.

    Attributes:
        files: The files in it that did not merge by themselves.
        own: Whether it is this agent that finishes the merge.
    """

    where: str
    files: tuple[str, ...]
    own: bool


@dataclass(frozen=True)
class Denied:
    """A write or a read that a scope does not allow.

    Attributes:
        named: The path as the agent named it.
        relative: The path relative to the agent's checkout, where the
            refusal is of a file in it.
        closed: The folder it lies in, for ``closed``.
        files: The files that may be written instead, for
            ``merged_by_itself``.
    """

    kind: DeniedKind
    named: str
    relative: str | None = None
    closed: Closed | None = None
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scope:
    """What one agent may write. All paths and globs are repo-relative.

    Attributes:
        checkout: The agent's workspace; '' when it has none of the org's.
        directory: The folder its writes are held to; '' is the repository.
        writes: Whether it works in that folder from a shell. Without it
            the shell and the agent's git open nothing for writing.
        edits: Whether its file tools may change a file at all.
        aside: A side conversation, which changes no file anywhere.
        closed: Folders inside its own that are closed to it.
        hidden: Folders inside its own that its git leaves alone.
        merges: The merges being finished in the repository.
    """

    checkout: str
    directory: str
    writes: bool = True
    edits: bool = True
    aside: bool = False
    writable: tuple[str, ...] = ()
    readonly: tuple[str, ...] = ()
    closed: tuple[Closed, ...] = ()
    hidden: tuple[str, ...] = ()
    merges: tuple[MergeArea, ...] = ()

    def refuse(self, path: str, named: str | None = None) -> Denied | None:
        """Why the agent may not write ``path`` of its checkout, if it may
        not.

        Args:
            path: Relative to the checkout; '' for its top.
            named: The path as the agent named it, when that differs.
        """
        named = path if named is None else named

        def denied(kind: DeniedKind, closed: Closed | None = None) -> Denied:
            return Denied(kind, named, relative=path, closed=closed)

        if not self.edits:
            return denied("reads_only")
        if path == ".git" or path.startswith(".git/"):
            return denied("git")
        if not holds(self.directory, path):
            return denied("outside")
        for folder in self.closed:
            if holds(folder.path, path):
                return denied("closed", folder)
        if any(glob_matches(glob, path) for glob in self.readonly):
            return denied("readonly")
        if not any(glob_matches(glob, path) for glob in self.writable):
            return denied("not_writable")
        return None
