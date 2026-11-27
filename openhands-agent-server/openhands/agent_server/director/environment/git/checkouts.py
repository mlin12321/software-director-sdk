"""Teams' work in git worktrees."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from openhands.agent_server.director import changes as by_hand
from openhands.agent_server.director.boundary import owns
from openhands.agent_server.director.confinement import apart, checkout_root
from openhands.agent_server.director.environment.checkouts import (
    Checkouts,
    MergeBack,
    MergeCheckout,
    Overlap,
)
from openhands.agent_server.director.git import (
    GitError,
    branch_of,
    bring_over,
    changed_together,
    enclosing_repository,
    git,
    head_of,
    holds_commit,
    is_git_repository,
    last_commit_in,
    pathspec,
    save_and_discard,
    tracked_files,
    uncommitted,
)
from openhands.agent_server.director.merge_back import (
    merge_back,
    merge_message,
    take_merge,
)
from openhands.agent_server.director.merge_checkout import discard, finish, prepare
from openhands.agent_server.director.models import (
    AcceptedChanges,
    FileEdit,
    FileSides,
    NodeOwnership,
    OrgNode,
    SessionTeam,
    TeamChanges,
)
from openhands.agent_server.director.ownership import (
    can_merge_in_memory,
    conflicts,
    node_ownership,
)
from openhands.agent_server.director.reconcile import changes_of
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class GitCheckouts(Checkouts):
    """Each team's work is a branch, checked out in a worktree or in the
    repository's own checkout."""

    def is_repository(self, directory: Path) -> bool:
        return is_git_repository(directory)

    def root_of(self, workspace: str | Path) -> Path:
        return checkout_root(str(workspace))

    def enclosing(self, repository: Path) -> Path | None:
        return enclosing_repository(repository)

    def is_apart(self, team: SessionTeam, above: SessionTeam) -> bool:
        return apart(team, above)

    def missing(
        self, repository: Path, teams: Iterable[SessionTeam]
    ) -> list[tuple[SessionTeam, bool]]:
        return [
            (team, _has_branch(repository, team.branch))
            for team in _missing_worktrees(teams)
        ]

    def restore(
        self, repository: Path, teams: Iterable[SessionTeam], within: str
    ) -> None:
        missing = [
            team for team, restorable in self.missing(repository, teams) if restorable
        ]
        if not missing:
            return
        try:
            git(repository, ["worktree", "prune"])
            for team in missing:
                top = Path(team.worktree_path)
                if within:
                    top = Path(str(top).removesuffix(within).rstrip("/"))
                git(repository, ["worktree", "add", str(top), team.branch])
        except (GitError, OSError):
            logger.exception("Could not make a worktree again in %s", repository)

    def version(self, workspace: str | Path) -> str | None:
        return head_of(Path(workspace))

    def branch(self, workspace: str | Path) -> str:
        """The branch, or ``HEAD`` when the checkout is on none."""
        try:
            return git(Path(workspace), ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        except (GitError, OSError):
            return ""

    def uncommitted(self, workspace: str | Path, path: str) -> list[str]:
        return uncommitted(Path(workspace), path)

    def files(self, workspace: str | Path) -> list[str]:
        try:
            listed = git(
                Path(workspace), ["ls-files", "-z", "--full-name", "--", ":(top)"]
            )
        except (GitError, OSError):
            return []
        return [path for path in listed.split("\0") if path]

    def file_count(self, workspace: str | Path, path: str) -> int:
        return tracked_files(Path(workspace), path)

    def changed_together(self, workspace: str | Path, path: str, other: str) -> int:
        return changed_together(Path(workspace), path, other)

    def last_change(self, workspace: str | Path, path: str) -> str | None:
        return last_commit_in(Path(workspace), path)

    def astray(
        self, team: SessionTeam, below: Sequence[SessionTeam], repository: Path
    ) -> str | None:
        if not team.worktree_path:
            return None
        checkout = Path(team.worktree_path)
        ours = checkout_root(team.worktree_path) != checkout_root(str(repository))
        on = branch_of(checkout)
        # The repository's own checkout is the human's, on any branch.
        if ours and team.branch and on != team.branch:
            return (
                f"The checkout of {team.title!r} is "
                + (f"on the branch {on!r}" if on else "on no branch")
                + f", and its work is taken from {team.branch!r}. Run any git "
                "command there: the checkout goes back to its branch. Then "
                "bring over what was committed elsewhere"
                + (f" with `git merge {on}`" if on else "")
                + " and submit again."
            )
        for sub_team in below:
            review = sub_team.review
            if (
                review is None
                or review.state != "accepted"
                or not review.commit
                or checkout_root(sub_team.worktree_path) != checkout_root(str(checkout))
            ):
                continue
            if not holds_commit(checkout, review.commit):
                return (
                    f"The accepted work of {sub_team.title!r} (commit "
                    f"{review.commit[:10]}) is not in what the checkout of "
                    f"{team.title!r} is on: it was left on another branch. "
                    f"Bring it in with `git merge {review.commit[:10]}`, then "
                    "submit again."
                )
        return None

    def changes_listed(self, team: SessionTeam) -> str | None:
        """git's list of what the team committed since its base, with the
        lines added and removed in each file."""
        checkout, base = Path(team.worktree_path), team.base_commit
        try:
            return git(
                checkout,
                ["diff", "--stat=200", base, "HEAD", "--"]
                + [pathspec(team.boundary.path)],
            ).rstrip()
        except (GitError, OSError):
            logger.warning(
                "Could not diff %s against %s", checkout, base, exc_info=True
            )
            return None

    def own_changes(
        self, team: SessionTeam, below: Sequence[SessionTeam]
    ) -> list[str] | None:
        if not team.base_commit:
            return None
        checkout = Path(team.worktree_path)
        try:
            names = git(
                checkout,
                ["diff", "--name-only", "-z", team.base_commit, "HEAD", "--"]
                + [pathspec(team.boundary.path)],
            ).split("\0")
            names += uncommitted(checkout, team.boundary.path)
        except (GitError, OSError):
            return None
        given = [sub_team.boundary.path for sub_team in below]
        return sorted(
            {
                name
                for name in names
                if name
                and not any(
                    name == path or name.startswith(f"{path}/") for path in given
                )
            }
        )

    def overlap(self, team: SessionTeam, above: SessionTeam) -> Overlap | None:
        own, checkout = Path(team.worktree_path), Path(above.worktree_path)
        if checkout_root(str(own)) == checkout_root(str(checkout)):
            return None
        changes = changes_of(checkout, own)
        if changes is None:
            return None
        files: list[str] = []
        ours = head_of(checkout)
        # Lines that do not merge, and files outside the team's
        # directory, go back to the team when it is merged; there is
        # nothing for the reviewer to weigh in them.
        unmergeable = (
            conflicts(checkout, ours, changes.tip)
            if ours and can_merge_in_memory(checkout)
            else ()
        )
        if not unmergeable:
            files = [path for path in changes.both if owns(team, path)]
        seen = git(checkout, ["ls-tree", "HEAD", "--", *files]) if files else ""
        return Overlap(changes.tip, files, changes.landed, seen)

    def merge(self, team: SessionTeam, above: SessionTeam, title: str) -> MergeBack:
        return merge_back(
            Path(above.worktree_path), Path(team.worktree_path), team.boundary, title
        )

    def undo_merge(self, workspace: str | Path, before: str) -> None:
        git(Path(workspace), ["reset", "--merge", "--quiet", before])

    def lay_out_merge(self, repository: Path, where: Path, ours: str, tip: str) -> None:
        prepare(repository, where, ours, tip)

    def finish_merge(self, merge: MergeCheckout, title: str, path: str) -> str:
        return finish(merge, merge_message(title, path))

    def take_merge(self, workspace: str | Path, version: str) -> MergeBack:
        return take_merge(Path(workspace), version)

    def drop_merge(self, repository: Path, where: Path) -> None:
        discard(repository, where)

    def set_aside(self, workspace: str | Path, path: str, ref: str) -> str | None:
        return save_and_discard(Path(workspace), path, ref)

    def bring_over(
        self, source: str | Path, target: str | Path, path: str, ref: str
    ) -> list[str]:
        return bring_over(Path(source), Path(target), path, ref)

    def keep_unmerged(self, own: str | Path, into: str | Path, ref: str) -> bool:
        checkout = Path(into)
        tip, merged_into = head_of(Path(own)), head_of(checkout)
        if tip is None or merged_into is None:
            return False
        if tip in git(checkout, ["merge-base", "--all", tip, merged_into]).split():
            return False
        git(checkout, ["update-ref", ref, tip])
        return True

    def ownership(
        self, node: OrgNode, reports: Sequence[OrgNode], workspace: str | Path
    ) -> NodeOwnership:
        return node_ownership(node, reports, Path(workspace))

    def team_changes(self, team: SessionTeam) -> TeamChanges:
        return by_hand.team_changes(team)

    def file_sides(self, team: SessionTeam, path: str) -> FileSides:
        return by_hand.file_sides(team, path)

    def accepted_changes(
        self, repository: Path, base: str, head: str, path: str
    ) -> AcceptedChanges:
        return by_hand.accepted_changes(repository, base, head, path)

    def accepted_file(
        self, repository: Path, base: str, head: str, path: str
    ) -> FileSides:
        return by_hand.accepted_file(repository, base, head, path)

    def require_editable(
        self, team: SessionTeam, sub_teams: Sequence[SessionTeam], path: str
    ) -> str:
        return by_hand.require_editable(team, list(sub_teams), path)

    def write_by_hand(
        self,
        team: SessionTeam,
        sub_teams: Sequence[SessionTeam],
        path: str,
        content: str,
    ) -> FileEdit:
        return by_hand.write_by_hand(team, list(sub_teams), path, content)


def _missing_worktrees(teams: Iterable[SessionTeam]) -> list[SessionTeam]:
    """The teams whose own worktree is not on disk, parents first."""
    seen: set[str] = set()
    missing: list[SessionTeam] = []
    for team in teams:
        path = team.worktree_path
        if team.isolation != "worktree" or path in seen:
            continue
        seen.add(path)
        if not Path(path).is_dir():
            missing.append(team)
    return missing


def _has_branch(repository: Path, branch: str) -> bool:
    if not branch:
        return False
    try:
        git(repository, ["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"])
    except (GitError, OSError):
        return False
    return True
