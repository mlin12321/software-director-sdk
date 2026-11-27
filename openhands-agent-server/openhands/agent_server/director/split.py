"""Checks the sub-teams an IC proposes to its Manager.

Sub-teams own directories, so that two boundaries can be told apart by
comparing paths: whether two arbitrary sets of globs overlap cannot be
decided reliably.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import NamedTuple

import yaml
from pydantic import BaseModel, ConfigDict, Field

from openhands.agent_server.director.git_store import team_key
from openhands.agent_server.director.models import (
    Boundary,
    DirectorSettings,
    IsolationMode,
    OrgNodeInput,
    SessionTeam,
    SplitChild,
)
from openhands.agent_server.director.org_store import (
    OrgStore,
    OrgStoreError,
    team_id_of,
)
from openhands.agent_server.director.scope import is_inside


MAX_SPLIT_CHILDREN = 16


def _size(store: OrgStore, team_id: str) -> int:
    """How many teams of the org the team ``team_id`` is with those below it."""
    return 1 + sum(_size(store, team.id) for team in store.sub_teams(team_id))


def check_limits(
    store: OrgStore, place: SessionTeam, added: int, settings: DirectorSettings
) -> None:
    """Refuse ``added`` new sub-teams of ``node`` that the org has no room for.

    A session is counted where it is in the org: its levels from the top
    team, and its teams for every team of the org above it as well.

    Raises:
        OrgStoreError: They would be too deep, or make ``node`` or a team
            above it more teams than a team may be with those below it.
    """
    above = store.above(place.id)
    top = above[0] if above else place
    higher = store.teams_above(top.team_id)
    level = len(higher) + len(above) + 2
    if level > settings.max_depth:
        raise OrgStoreError(
            f"An org may have at most {settings.max_depth} levels of teams, "
            f"and sub-teams of {place.title!r} would be level {level}. Do the "
            "work in your own team."
        )
    sizes = [
        (team.title, 1 + store.teams_below(team.id))
        for team in (place, *reversed(above))
    ]
    # What the session has at the team it was opened at stands in for what
    # the org has there.
    grown = sizes[-1][1] - (_size(store, top.team_id) if higher else 0)
    sizes += [(team.title, _size(store, team.id) + grown) for team in reversed(higher)]
    for title, teams in sizes:
        room = settings.max_teams - teams
        if added > room:
            raise OrgStoreError(
                f"A team may be at most {settings.max_teams} teams together "
                f"with the teams below it. {title!r} is {teams}, and "
                f"you propose {added} more. "
                + (
                    f"At most {room} more sub-teams fit."
                    if room > 0
                    else "No more sub-teams fit."
                )
            )


class ProposedTeam(BaseModel):
    """One sub-team an IC proposes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    path: str = Field(description="The folder the team owns, strictly inside your own.")
    instruction: str = Field(min_length=1, description="What the team is to deliver.")
    reason: str = Field(
        default="", description="Why it is a team of its own, for your Manager."
    )
    model: str | None = Field(
        default=None,
        description="One of the profiles read_team_context lists. Optional.",
    )
    type: str | None = Field(
        default=None,
        description=(
            "One of the types read_team_context lists: what kind of team it "
            "is. The org's default type when left out."
        ),
    )
    writable: list[str] = Field(
        default_factory=list,
        description="Globs the team may write; all of `path` when left out.",
    )
    readonly: list[str] = Field(
        default_factory=list, description="Globs the team may only read."
    )
    isolation: IsolationMode = "shared"


class Proposed(NamedTuple):
    """The sub-teams an IC proposes, as written."""

    children: list[SplitChild]
    documents: dict[str, str]
    """What the proposal gives for documents, by the name of each."""
    keep: list[str]
    """The sub-teams the proposer has now that it says are to stay, by name
    or team id."""
    reason: str = ""
    """Why the proposer wants this make-up of teams, when it says."""


def proposed_teams(
    teams: Sequence[ProposedTeam],
    documents: Mapping[str, str],
    keep: Sequence[str],
    reason: str,
    default_type: str = "software-engineer",
) -> Proposed:
    """What a call of ``propose_teams`` proposes.

    Args:
        default_type: The type of a sub-team whose proposal names none.
    """
    return Proposed(
        [
            SplitChild(
                title=team.name,
                path=team.path,
                task=team.instruction,
                reason=team.reason.strip(),
                agent_profile=team.model,
                agent_type=team.type or default_type,
                writable=team.writable,
                readonly=team.readonly,
                isolation=team.isolation,
            )
            for team in teams
        ],
        {name: text for name, text in documents.items() if text.strip()},
        list(keep),
        reason.strip(),
    )


class _Written(yaml.SafeDumper):
    """Writes text of several lines as a block, so that it reads as written."""


def _text(dumper: yaml.SafeDumper, text: str) -> yaml.ScalarNode:
    style = "|" if "\n" in text else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", text, style=style)


_Written.add_representer(str, _text)


def as_written(proposed: Proposed) -> str:
    """A proposal as the Manager it is put before reads it."""
    teams = []
    for child in proposed.children:
        team: dict[str, object] = {
            "name": child.title,
            "path": child.path,
            "instruction": child.task,
        }
        if child.reason:
            team["reason"] = child.reason
        if child.agent_profile:
            team["model"] = child.agent_profile
        team["type"] = child.agent_type
        if child.writable:
            team["writable"] = child.writable
        if child.readonly:
            team["readonly"] = child.readonly
        if child.isolation != "shared":
            team["isolation"] = child.isolation
        teams.append(team)
    written: dict[str, object] = {}
    written.update(proposed.documents)
    if proposed.keep:
        written["keep"] = proposed.keep
    written["teams"] = teams
    return yaml.dump(
        written, Dumper=_Written, sort_keys=False, allow_unicode=True
    ).rstrip()


class Split(NamedTuple):
    """Proposed sub-teams that passed the checks."""

    children: list[SplitChild]
    replaces: list[SessionTeam]
    """The sub-teams the team has now that the proposed ones take the place
    of."""
    keeps: list[SessionTeam]
    """The sub-teams the team has now that stay as they are."""


def _overlap(path: str, other: str) -> bool:
    return path == other or is_inside(path, other) or is_inside(other, path)


def _may_write(parent: SessionTeam, path: str) -> bool:
    """Whether a directory is part of what the parent's own agents may
    write, as far as that can be told from its globs' directories."""

    def covers(directory: str) -> bool:
        return path == directory or is_inside(path, directory)

    boundary = parent.boundary
    if any(
        (directory := glob.split("*", 1)[0].rstrip("/")) and covers(directory)
        for glob in boundary.readonly
    ):
        return False
    # Only a glob of all that is below a folder holds all of the new team's.
    return any(
        glob == "**" or (glob.endswith("/**") and covers(glob[: -len("/**")]))
        for glob in boundary.writable
    )


def _kept(
    existing: list[SessionTeam], replaced: dict[str, SessionTeam], keep: list[str]
) -> list[SessionTeam]:
    """The sub-teams a proposal says are to stay.

    Raises:
        OrgStoreError: It names a team the team does not have or one it
            also replaces, or leaves out a team it does not replace.
    """
    named = {name.strip() for name in keep}
    known = (
        {team.id for team in existing}
        | {team_key(team.id) for team in existing}
        | {team.title for team in existing}
    )
    unknown = sorted(named - known)
    if unknown:
        raise OrgStoreError(
            f"`keep` names {', '.join(map(repr, unknown))}, which "
            + ("is" if len(unknown) == 1 else "are")
            + " not among your sub-teams: "
            + (", ".join(repr(team.title) for team in existing) or "you have none")
        )
    kept = [
        team
        for team in existing
        if team.id in named or team_key(team.id) in named or team.title in named
    ]
    both = [team.title for team in kept if team.id in replaced]
    if both:
        raise OrgStoreError(
            f"{', '.join(map(repr, both))} cannot be kept and replaced: a "
            "proposed team takes all or part of its folder. Leave it out of "
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
    parent: SessionTeam,
    children: list[SplitChild],
    keep: list[str] | None = None,
) -> Split:
    """The proposed sub-teams with clean paths and explicit globs, and the
    sub-teams they replace.

    Args:
        keep: The sub-teams the proposer says are to stay, by name or team
            id. None when a proposal that was checked before is checked
            again: what it keeps is then not asked for.

    A proposed team that covers sub-teams the team already has takes their
    place (a merge), and so does one that takes part of an existing
    sub-team's directory (a split of it): what the new teams leave of that
    directory is the team's own again.

    Raises:
        OrgStoreError: A sub-team is not a directory strictly inside the
            parent's boundary, overlaps another proposed one, repeats a
            title, reaches outside its own directory, or would sit inside
            a team that stays, or ``keep`` does not account for every
            sub-team the proposal leaves alone.
    """
    if not 1 <= len(children) <= MAX_SPLIT_CHILDREN:
        raise OrgStoreError(
            f"A split needs between 1 and {MAX_SPLIT_CHILDREN} sub-teams"
        )
    existing = store.below(parent.id)
    proposed: dict[str, str] = {}
    replaced: dict[str, SessionTeam] = {}
    normalized: list[SplitChild] = []
    for child in children:
        path = clean_path(child.path)
        if not is_inside(path, parent.boundary.path) or not _may_write(parent, path):
            raise OrgStoreError(
                f"Sub-team {child.title!r} must own a folder inside what "
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
        # What the team above may only read inside the folder, the new team
        # may only read too: a glob of a folder inside it, or of none.
        readonly += [
            glob
            for glob in parent.boundary.readonly
            if (glob.startswith(f"{path}/") or glob.startswith("*"))
            and glob not in readonly
        ]
        proposed[path] = child.title
        normalized.append(
            child.model_copy(
                update={"path": path, "writable": writable, "readonly": readonly}
            )
        )
    for team in replaced.values():
        if team.isolation == "worktree":
            sharing = [
                below.title
                for below in store.below(team.id)
                if below.worktree_path == team.worktree_path
            ]
            if sharing:
                raise OrgStoreError(
                    f"{team.title!r} cannot be replaced while {', '.join(sharing)} "
                    "work in its checkout; have your Manager remove them first"
                )
        for below in store.below(team.id):
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


def home_of(below: SessionTeam, parent_id: str, teams: dict[str, str]) -> str:
    """The team a sub-team of a replaced team reports to afterwards.

    Args:
        parent_id: The team that proposed the split.
        teams: The id of each new team by the directory it owns.

    Returns:
        The new team whose directory holds it, else the proposing team.
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


# Each team, the team above it, its directory and, where it is known, what
# else makes it that team (:func:`made_of`).
Shape = list[tuple[str, str, str, str]]


def made_of(team: OrgNodeInput) -> str:
    """What makes a team that team besides where it hangs and its
    directory: its name, what it may write and read, the agent it runs on
    and when it was set up."""
    return json.dumps(
        [
            team.title,
            sorted(team.boundary.writable),
            sorted(team.boundary.readonly),
            team.agent_profile or "",
            team.since,
        ]
    )


def shape_of(teams: Sequence[OrgNodeInput]) -> Shape:
    """The sub-teams under a team, as they are compared between the org and
    a session. The first of ``teams`` is the team itself and has nothing
    above it here."""
    top = teams[0].id if teams else ""
    return sorted(
        (
            team_id_of(team),
            "" if team.id == top else team_key(team.parent_id or ""),
            team.boundary.path,
            made_of(team),
        )
        for team in teams
    )


def settle[K, V](
    base: Mapping[K, V], theirs: Mapping[K, V], now: Mapping[K, V]
) -> tuple[dict[K, V | None], list[K]]:
    """Settle, key by key, what a session made of something the org has,
    which may have changed since the session last knew it as ``base``.

    What the session left as it was stays as the org has it now. What only
    the session changed is taken from the session. What both changed, and
    not to the same, is a conflict. A key that is missing is None.

    Returns:
        What to take from the session (None to remove it), and the keys
        both changed.
    """
    take: dict[K, V | None] = {}
    conflicts: list[K] = []
    for key in {*base, *theirs, *now}:
        was, wanted, has = base.get(key), theirs.get(key), now.get(key)
        if wanted == was or wanted == has:
            continue
        if has == was:
            take[key] = wanted
        else:
            conflicts.append(key)
    return take, conflicts


def home_of_lost(path: str, teams: Sequence[OrgNodeInput], top: str) -> str:
    """The team of the org that has what a team it does not have left: the
    one whose directory most closely holds ``path``, else ``top``."""
    holding = [
        team
        for team in teams
        if team.boundary.path == path or is_inside(path, team.boundary.path)
    ]
    if not holding:
        return top
    return max(holding, key=lambda team: len(team.boundary.path)).id


# A team in a shape: the team above it, its directory and, where it is
# known, what else makes it that team. None where the team is not there.
Side = tuple[str, ...] | None


class ShapeConflict(NamedTuple):
    """A team that a session and the org each made something else of."""

    team: str
    session: Side
    org: Side


def merge_shapes(
    base: Sequence[tuple[str, ...]],
    theirs: Sequence[tuple[str, ...]],
    now: Sequence[tuple[str, ...]],
) -> tuple[dict[str, Side], list[ShapeConflict]]:
    """Settle, team by team, what the sub-teams a session has make of the
    org's, which may have changed since (:func:`settle`).

    A team is also a conflict when it would not hang together with the
    rest: under a team that is gone, with a directory another team has or
    one beside it holds, or removed while a team is still below it.

    Args:
        base: The sub-teams as the session last knew the org's.
        theirs: The sub-teams the session has.
        now: The sub-teams the org has.

    Returns:
        What to take from the session, each team as the session has it
        (None to remove it), and the conflicts.
    """
    was, session, org = (
        {team: tuple(side) for team, *side in shape} for shape in (base, theirs, now)
    )
    take, both = settle(was, session, org)
    conflicts = set(both)
    while True:
        after = {
            team: side for team, side in {**org, **take}.items() if side is not None
        }
        apart = {
            team
            for team, side in take.items()
            if (
                any(held[0] == team for held in after.values())
                if side is None
                else (side[0] != "" and side[0] not in after)
                or any(
                    other != team
                    and (
                        held[1] == side[1]
                        # Side by side, one would hold the other's directory.
                        or (held[0] == side[0] and _overlap(held[1], side[1]))
                    )
                    for other, held in after.items()
                )
            )
        }
        if not apart:
            break
        conflicts |= apart
        take = {team: side for team, side in take.items() if team not in apart}
    return take, [
        ShapeConflict(team, session.get(team), org.get(team))
        for team in sorted(conflicts)
    ]
