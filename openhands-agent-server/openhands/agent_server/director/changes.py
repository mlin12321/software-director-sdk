"""What a team changed since its work was last accepted, and a change made
to its files by hand."""

import os
import re
from pathlib import Path

from openhands.agent_server.director.boundary import owns
from openhands.agent_server.director.git import (
    DIRECTOR_IDENTITY,
    GitError,
    git,
    pathspec,
    top_of,
)
from openhands.agent_server.director.merge_back import (
    bring_up_to_date,
    one_at_a_time,
)
from openhands.agent_server.director.models import (
    AcceptedChanges,
    ChangedFile,
    ChangeKind,
    FileEdit,
    FileSides,
    SessionTeam,
    TeamChanges,
)
from openhands.agent_server.director.scope import glob_matches
from openhands.sdk.errors import Forbidden, Invalid, NotFound


MAX_CHANGED_FILES = 2000
MAX_FILE_BYTES = 1_000_000
_KINDS: dict[str, ChangeKind] = {"A": "added", "D": "deleted"}
_COMMIT = re.compile(r"[0-9a-f]{7,64}")


def checkout_top(place: SessionTeam) -> Path:
    """The top of the checkout the team works in, symlinks resolved."""
    workspace = Path(place.worktree_path)
    try:
        top = top_of(workspace)
    except (GitError, OSError) as error:
        raise NotFound(f"Team {place.id} has no checkout on this server") from error
    return Path(os.path.realpath(top))


def team_changes(place: SessionTeam) -> TeamChanges:
    """The files in the team's directory that differ from where its work
    started or was last accepted, uncommitted and new files included.

    Raises:
        NotFound: The team's checkout is not on this server.
    """
    top = checkout_top(place)
    base = place.base_commit
    if not base:
        return TeamChanges(node_id=place.id, base="", files=[], truncated=False)
    spec = pathspec(place.boundary.path)
    try:
        files = _differing(top, [base], spec)
        new = _fields(
            git(top, ["ls-files", "--others", "--exclude-standard", "-z", "--", spec])
        )
    except (GitError, OSError) as error:
        raise NotFound(
            f"Team {place.id}'s changes since {base[:12]} cannot be read"
        ) from error
    uncommitted = set(new)
    try:
        uncommitted.update(
            _fields(git(top, ["diff", "--name-only", "-z", "HEAD", "--", spec]))
        )
    except (GitError, OSError):
        pass
    for path in new:
        files[path] = ChangedFile(
            path=path,
            kind="added",
            added=_line_count(top / path),
            removed=0,
            committed=False,
        )
    for path in uncommitted & files.keys():
        files[path] = files[path].model_copy(update={"committed": False})
    listed = sorted(files.values(), key=lambda file: file.path)
    return TeamChanges(
        node_id=place.id,
        base=base,
        files=listed[:MAX_CHANGED_FILES],
        truncated=len(listed) > MAX_CHANGED_FILES,
    )


def accepted_changes(repo: Path, base: str, head: str, path: str) -> AcceptedChanges:
    """The files under ``path`` that differ between two commits of the
    repository, as an accept in the org's history names them.

    Raises:
        Invalid: ``base`` or ``head`` is not a commit id, or ``path`` leads
            out of the repository.
        NotFound: The repository no longer has one of the commits.
    """
    folder = _clean(path) if path else ""
    try:
        files = _differing(repo, _commits(base, head), pathspec(folder))
    except (GitError, OSError) as error:
        raise NotFound(
            f"The changes from {base[:12]} to {head[:12]} cannot be read"
        ) from error
    listed = sorted(files.values(), key=lambda file: file.path)
    return AcceptedChanges(
        base=base,
        head=head,
        files=listed[:MAX_CHANGED_FILES],
        truncated=len(listed) > MAX_CHANGED_FILES,
    )


def accepted_file(repo: Path, base: str, head: str, path: str) -> FileSides:
    """One file as it was at each of two commits of the repository.

    Raises:
        Invalid: ``base`` or ``head`` is not a commit id, or ``path`` does
            not name a file of the repository.
        NotFound: The repository no longer has one of the commits.
    """
    path = _clean(path)
    for commit in _commits(base, head):
        try:
            git(repo, ["cat-file", "-e", f"{commit}^{{commit}}"])
        except (GitError, OSError) as error:
            raise NotFound(f"The repository has no commit {commit[:12]}") from error
    before, before_unread = _committed(repo, base, path)
    after, after_unread = _committed(repo, head, path)
    return FileSides(
        path=path,
        before=before,
        after=after,
        unreadable=before_unread or after_unread,
    )


def file_sides(place: SessionTeam, path: str) -> FileSides:
    """One file of the team's directory as it was at the team's base and as
    it is in its checkout now.

    Raises:
        Invalid: ``path`` does not name a file of a checkout.
        Forbidden: The file is not in the team's directory.
        NotFound: The team's checkout is not on this server.
    """
    path = _clean(path)
    if not owns(place, path):
        raise Forbidden(f"{path!r} is not in the folder of team {place.title!r}")
    top = checkout_top(place)
    target = _within(top, path)
    before, before_unread = _committed(top, place.base_commit, path)
    after, after_unread = _on_disk(target)
    return FileSides(
        path=path,
        before=before,
        after=after,
        unreadable=before_unread or after_unread,
    )


def edit_refusal(
    place: SessionTeam, sub_teams: list[SessionTeam], path: str
) -> str | None:
    """Why a person may not change ``path`` through this team, if not."""
    directory = place.boundary.path or "the repository"
    if path == ".git" or path.startswith(".git/"):
        return f"{path!r} belongs to git."
    if not owns(place, path):
        return f"{path!r} is outside the folder of team {place.title!r} ({directory})."
    for sub_team in sub_teams:
        if owns(sub_team, path):
            return (
                f"{path!r} belongs to the sub-team {sub_team.title!r}. "
                "Change it in that team."
            )
    if any(glob_matches(glob, path) for glob in place.boundary.readonly):
        return f"{path!r} is read-only for team {place.title!r}."
    if not any(glob_matches(glob, path) for glob in place.boundary.writable):
        return f"{path!r} is not among the files team {place.title!r} may write."
    return None


def require_editable(
    place: SessionTeam, sub_teams: list[SessionTeam], path: str
) -> str:
    """``path`` in its plain form, when a person may change it through
    this team.

    Raises:
        Invalid: ``path`` does not name a file of a checkout.
        Forbidden: The team's agents may not write the file either.
    """
    path = _clean(path)
    refusal = edit_refusal(place, sub_teams, path)
    if refusal is not None:
        raise Forbidden(refusal)
    return path


def write_by_hand(
    place: SessionTeam, sub_teams: list[SessionTeam], path: str, content: str
) -> FileEdit:
    """Write one file in the team's checkout and commit it alone.

    Whatever else is uncommitted in the checkout stays as it is.

    Raises:
        Invalid: ``path`` does not name a file of a checkout, or git
            ignores it.
        Forbidden: The team's agents may not write the file either.
        NotFound: The team's checkout is not on this server.
    """
    path = require_editable(place, sub_teams, path)
    top = checkout_top(place)
    target = _within(top, path)
    if Path(os.path.realpath(target)) != target:
        # What it leads to may be another team's.
        raise Forbidden(f"{path!r} is reached through a link")
    if target.is_dir():
        raise Invalid(f"{path!r} is a folder")
    spec = pathspec(path)
    try:
        git(top, ["check-ignore", "--quiet", "--", path])
    except GitError:
        pass
    else:
        raise Invalid(f"git ignores {path!r}, so a change to it cannot be committed")
    with one_at_a_time(top):
        # Written under the lock too, or an agent's commit in between takes
        # the edit as its own.
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode())
        before = git(top, ["rev-parse", "HEAD"]).strip()
        git(top, ["add", "--", spec])
        if not git(top, ["status", "--porcelain", "--", spec]).strip():
            return FileEdit(path=path, commit=None)
        git(
            top,
            ["commit", "--no-verify", "--only", "-m", f"Edited by hand: {path}"]
            + ["--", spec],
            env=DIRECTOR_IDENTITY,
        )
        # Or the next commit of an agent that works here takes the edit out.
        bring_up_to_date(top, before)
    return FileEdit(path=path, commit=git(top, ["rev-parse", "HEAD"]).strip())


def _commits(base: str, head: str) -> list[str]:
    for commit in (base, head):
        if not _COMMIT.fullmatch(commit):
            raise Invalid(f"{commit!r} is not a commit id")
    return [base, head]


def _differing(top: Path, commits: list[str], spec: str) -> dict[str, ChangedFile]:
    """The committed files that differ between two commits, or between one
    commit and the checkout, by path."""
    ask = ["--no-renames", "-z", *commits, "--", spec]
    kinds = _fields(git(top, ["diff", "--name-status", *ask]))
    lines: dict[str, tuple[int | None, int | None]] = {}
    for entry in _fields(git(top, ["diff", "--numstat", *ask])):
        added, removed, path = entry.split("\t", 2)
        lines[path] = (
            None if added == "-" else int(added),
            None if removed == "-" else int(removed),
        )
    files: dict[str, ChangedFile] = {}
    for letter, path in zip(kinds[::2], kinds[1::2]):
        added, removed = lines.get(path, (None, None))
        files[path] = ChangedFile(
            path=path,
            kind=_KINDS.get(letter[:1], "modified"),
            added=added,
            removed=removed,
            committed=True,
        )
    return files


def _fields(output: str) -> list[str]:
    return [field for field in output.split("\0") if field]


def _clean(path: str) -> str:
    parts = Path(path).parts
    if not path or path.startswith("/") or ".." in parts or "\0" in path:
        raise Invalid(f"{path!r} does not name a file of the checkout")
    return Path(path).as_posix()


def _within(top: Path, path: str) -> Path:
    """Where ``path`` is on disk; refused when a link leads out of ``top``."""
    target = top / path
    resolved = Path(os.path.realpath(target))
    if resolved != top and top not in resolved.parents:
        raise Forbidden(f"{path!r} leads out of the team's checkout")
    return target


def _committed(top: Path, commit: str, path: str) -> tuple[str | None, bool]:
    if not commit:
        return None, False
    name = f"{commit}:{path}"
    try:
        size = int(git(top, ["cat-file", "-s", name]).strip())
    except (GitError, OSError, ValueError):
        return None, False
    if size > MAX_FILE_BYTES:
        return None, True
    text = git(top, ["cat-file", "blob", name])
    return (None, True) if "\0" in text else (text, False)


def _on_disk(target: Path) -> tuple[str | None, bool]:
    if not target.is_file():
        return None, False
    if target.stat().st_size > MAX_FILE_BYTES:
        return None, True
    data = target.read_bytes()
    if b"\0" in data:
        return None, True
    return data.decode(errors="replace"), False


def _line_count(target: Path) -> int | None:
    text, unreadable = _on_disk(target)
    if text is None or unreadable:
        return None
    return len(text.splitlines())
