"""Reads and checks the sub-teams an IC proposes in a message to its Manager.

Sub-teams own directories, so that two boundaries can be told apart by
comparing paths: whether two arbitrary sets of globs overlap cannot be
decided reliably.
"""

import re
from pathlib import PurePosixPath
from typing import NamedTuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from openhands.agent_server.director.models import (
    Boundary,
    DirectorSettings,
    IsolationMode,
    OrgNode,
    SplitChild,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    OrgStoreError,
    session_id_of,
)


MAX_SPLIT_CHILDREN = 16
# The fence a message's team proposal is written in.
TEAMS_FENCE = "teams"
_TEAMS_BLOCK = re.compile(
    rf"^```{TEAMS_FENCE}[ \t]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE
)


def check_limits(
    store: OrgStore, node: OrgNode, added: int, settings: DirectorSettings
) -> None:
    """Refuse ``added`` new sub-teams of ``node`` that the org has no room for.

    Raises:
        OrgStoreError: They would be too deep, or make ``node`` or a team
            above it more teams than a team may be with those below it.
    """
    above = store.ancestors(node.id)
    level = len(above) + 2
    if level > settings.max_depth:
        raise OrgStoreError(
            f"An org may have at most {settings.max_depth} levels of teams, "
            f"and sub-teams of {node.title!r} would be level {level}. Do the "
            "work in your own team."
        )
    for team in (node, *reversed(above)):
        teams = 1 + store.teams_below(team.id)
        room = settings.max_teams - teams
        if added > room:
            raise OrgStoreError(
                f"A team may be at most {settings.max_teams} teams together "
                f"with the teams below it. {team.title!r} is {teams}, and "
                f"you propose {added} more. "
                + (
                    f"At most {room} more sub-teams fit."
                    if room > 0
                    else "No more sub-teams fit."
                )
            )


class _Team(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    path: str
    instruction: str = Field(min_length=1)
    reason: str = ""
    model: str | None = None
    writable: list[str] = Field(default_factory=list)
    readonly: list[str] = Field(default_factory=list)
    isolation: IsolationMode = "shared"


class _Teams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    teams: list[_Team] = Field(min_length=1)
    keep: list[str] = Field(default_factory=list)
    contracts: str = ""
    reason: str = ""


class Proposed(NamedTuple):
    """The sub-teams a message proposes, as written."""

    children: list[SplitChild]
    contracts: str
    keep: list[str]
    """The sub-teams the proposer has now that it says are to stay, by name
    or node id."""
    reason: str = ""
    """Why the proposer wants this make-up of teams, when it says."""


def proposed_teams(text: str) -> Proposed | None:
    """The sub-teams a message proposes in its ``teams`` block, if it has one.

    Raises:
        OrgStoreError: The message has more than one block, or the block is
            not the YAML a team proposal is written in.
    """
    blocks = _TEAMS_BLOCK.findall(text)
    if not blocks:
        return None
    if len(blocks) > 1:
        raise OrgStoreError(
            f"A message can propose sub-teams once: it has {len(blocks)} "
            f"```{TEAMS_FENCE} blocks"
        )
    try:
        written = _Teams.model_validate(yaml.safe_load(blocks[0]))
    except yaml.YAMLError as error:
        raise OrgStoreError(
            f"The ```{TEAMS_FENCE} block is not YAML: {error}"
        ) from error
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc']) or 'the block'}: "
            f"{problem['msg']}"
            for problem in error.errors()
        )
        raise OrgStoreError(
            f"The ```{TEAMS_FENCE} block does not describe sub-teams: {problems}"
        ) from error
    return Proposed(
        [
            SplitChild(
                title=team.name,
                path=team.path,
                task=team.instruction,
                reason=team.reason.strip(),
                agent_profile=team.model,
                writable=team.writable,
                readonly=team.readonly,
                isolation=team.isolation,
            )
            for team in written.teams
        ],
        written.contracts,
        written.keep,
        written.reason.strip(),
    )


class Split(NamedTuple):
    """Proposed sub-teams that passed the checks."""

    children: list[SplitChild]
    replaces: list[OrgNode]
    """The sub-teams the node has now that the proposed ones take the place
    of."""
    keeps: list[OrgNode]
    """The sub-teams the node has now that stay as they are."""


def _overlap(path: str, other: str) -> bool:
    return path == other or is_inside(path, other) or is_inside(other, path)


def _may_write(parent: OrgNode, path: str) -> bool:
    """Whether a directory is part of what the parent's own agents may
    write, as far as that can be told from its globs' directories."""
    if not parent.boundary.writable:
        return True
    for glob in parent.boundary.writable:
        directory = glob.split("*", 1)[0].rstrip("/")
        if not directory or path == directory or is_inside(path, directory):
            return True
    return False


def _kept(
    existing: list[OrgNode], replaced: dict[str, OrgNode], keep: list[str]
) -> list[OrgNode]:
    """The sub-teams a proposal says are to stay.

    Raises:
        OrgStoreError: It names a team the node does not have or one it
            also replaces, or leaves out a team it does not replace.
    """
    named = {name.strip() for name in keep}
    known = {team.id for team in existing} | {team.title for team in existing}
    unknown = sorted(named - known)
    if unknown:
        raise OrgStoreError(
            f"`keep` names {', '.join(map(repr, unknown))}, which "
            + ("is" if len(unknown) == 1 else "are")
            + " not among your sub-teams: "
            + (", ".join(repr(team.title) for team in existing) or "you have none")
        )
    kept = [team for team in existing if team.id in named or team.title in named]
    both = [team.title for team in kept if team.id in replaced]
    if both:
        raise OrgStoreError(
            f"{', '.join(map(repr, both))} cannot be kept and replaced: a "
            "proposed team takes all or part of its directory. Leave it out of "
            "`keep`, or move the proposed team."
        )
    left_out = [
        team.title for team in existing if team.id not in replaced and team not in kept
    ]
    if left_out:
        raise OrgStoreError(
            "Say what becomes of every sub-team you have. "
            f"{', '.join(map(repr, left_out))} "
            + ("is" if len(left_out) == 1 else "are")
            + " neither replaced by a proposed team nor listed under `keep`. "
            "List a team under `keep` to leave it as it is; only your Manager "
            "removes a team without replacing it."
        )
    return kept


def normalize_split(
    store: OrgStore,
    parent: OrgNode,
    children: list[SplitChild],
    keep: list[str] | None = None,
) -> Split:
    """The proposed sub-teams with clean paths and explicit globs, and the
    sub-teams they replace.

    Args:
        keep: The sub-teams the proposer says are to stay, by name or node
            id. None when a proposal that was checked before is checked
            again: what it keeps is then not asked for.

    A proposed team that covers sub-teams the node already has takes their
    place (a merge), and so does one that takes part of an existing
    sub-team's directory (a split of it): what the new teams leave of that
    directory is the node's own again.

    Raises:
        OrgStoreError: A sub-team is not a directory strictly inside the
            parent's boundary, overlaps another proposed one, repeats a
            title, reaches outside its own directory, or would sit inside
            a team that stays, or ``keep`` does not account for every
            sub-team the proposal leaves alone, or a session works in a
            team the proposal would change.
    """
    if session_id_of(parent):
        raise OrgStoreError(
            f"Your team works in the session {store.session_title(parent)!r}, which "
            "has the sub-teams the team had when the session was opened. "
            "Sub-teams are not changed inside a session: do the work with "
            "the sub-teams you have, or in your own team."
        )
    _no_session_in(store, parent, below=False)
    if not 1 <= len(children) <= MAX_SPLIT_CHILDREN:
        raise OrgStoreError(
            f"A split needs between 1 and {MAX_SPLIT_CHILDREN} sub-teams"
        )
    existing = store.children(parent.id)
    proposed: dict[str, str] = {}
    replaced: dict[str, OrgNode] = {}
    normalized: list[SplitChild] = []
    for child in children:
        path = clean_path(child.path)
        if not is_inside(path, parent.boundary.path) or not _may_write(parent, path):
            raise OrgStoreError(
                f"Sub-team {child.title!r} must own a directory inside what "
                f"your team may write ({parent.boundary.path or 'the repository'!r}), "
                f"not {child.path!r}"
            )
        for other_path, other_title in proposed.items():
            if _overlap(path, other_path):
                raise OrgStoreError(
                    f"Sub-teams {child.title!r} ({path}) and {other_title!r} "
                    f"({other_path}) overlap"
                )
        for team in existing:
            if _overlap(path, team.boundary.path):
                replaced[team.id] = team
        writable = [clean_path(glob) for glob in child.writable] or [f"{path}/**"]
        for glob in writable:
            if not glob.startswith(f"{path}/"):
                raise OrgStoreError(
                    f"Sub-team {child.title!r} may only write under {path}/, "
                    f"not {glob!r}"
                )
        readonly = [clean_path(glob) for glob in child.readonly]
        proposed[path] = child.title
        normalized.append(
            child.model_copy(
                update={"path": path, "writable": writable, "readonly": readonly}
            )
        )
    for team in replaced.values():
        _no_session_in(store, team, below=True)
        if team.isolation == "worktree":
            sharing = [
                below.title
                for below in store.children(team.id)
                if below.worktree_path == team.worktree_path
            ]
            if sharing:
                raise OrgStoreError(
                    f"{team.title!r} cannot be replaced while {', '.join(sharing)} "
                    "work in its checkout; have your Manager remove them first"
                )
        for below in store.children(team.id):
            for path, title in proposed.items():
                if path == below.boundary.path or is_inside(path, below.boundary.path):
                    raise OrgStoreError(
                        f"Sub-team {title!r} ({path}) would sit inside "
                        f"{below.title!r} ({below.boundary.path}), which stays. "
                        f"That is for {below.title!r} to propose."
                    )
    titles = [team.title for team in existing if team.id not in replaced]
    titles += [child.title for child in normalized]
    for title in titles:
        if titles.count(title) > 1:
            raise OrgStoreError(f"More than one sub-team is titled {title!r}")
    kept = [] if keep is None else _kept(existing, replaced, keep)
    return Split(normalized, list(replaced.values()), kept)


def _no_session_in(store: OrgStore, team: OrgNode, *, below: bool) -> None:
    """Refuse to change the sub-teams of a team a session works in: the
    session has the teams as they were when it was opened.

    Raises:
        OrgStoreError: A session works in the team or, with ``below``, in
            a team below it.
    """
    sessions = store.sessions_at(team.id, below=below)
    if sessions:
        raise OrgStoreError(
            f"{team.title!r} cannot be changed while "
            + ("the session " if len(sessions) == 1 else "the sessions ")
            + ", ".join(repr(title) for title in sessions)
            + (" works" if len(sessions) == 1 else " work")
            + " in it. Propose it again once the human has closed "
            + ("it." if len(sessions) == 1 else "them.")
        )


def home_of(below: OrgNode, parent_id: str, teams: dict[str, str]) -> str:
    """The team a sub-team of a replaced team reports to afterwards.

    Args:
        parent_id: The node that proposed the split.
        teams: The id of each new team by the directory it owns.

    Returns:
        The new team whose directory holds it, else the proposing node.
    """
    for path, team_id in teams.items():
        if below.boundary.path == path or is_inside(below.boundary.path, path):
            return team_id
    return parent_id


def child_boundary(child: SplitChild) -> Boundary:
    return Boundary(path=child.path, writable=child.writable, readonly=child.readonly)


def clean_path(path: str) -> str:
    """A repo-relative POSIX path without empty, ``.`` or ``..`` segments.

    Raises:
        OrgStoreError: The path is absolute or leaves the repository.
    """
    parts = PurePosixPath(path.strip()).parts
    if "/" in parts[:1] or ".." in parts:
        raise OrgStoreError(f"{path!r} is not a path inside the repository")
    return "/".join(part for part in parts if part != ".")


def is_inside(path: str, ancestor: str) -> bool:
    """Whether ``path`` is strictly below ``ancestor`` ('' is the repository)."""
    if not path:
        return False
    return not ancestor or path.startswith(f"{ancestor}/")
