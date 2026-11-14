"""Exchanges an org with a remote of its repository, as git does a branch.

Nothing is exchanged on its own. A push gives the remote the org tree, the
team memory and the branch of every team that has a checkout of its own; a
fetch brings the remote's here, beside what is here, for a merge. The teams'
branches are kept on the remote under ``refs/agents/teams/``, so they do not
appear among its branches.
"""

from collections.abc import Iterable
from pathlib import Path

from openhands.agent_server.director.git import GitError, git
from openhands.agent_server.director.git_store import MEMORY_REF
from openhands.agent_server.director.models import BranchSync, OrgNode
from openhands.agent_server.director.org_persistence import ORG_REF
from openhands.sdk.errors import NotFound, UpstreamFailed


TEAM_REFS = "refs/agents/teams"
_NETWORK_TIMEOUT_SECONDS = 120
# The server has nobody to answer a prompt for a password.
_NO_PROMPT = {"GIT_TERMINAL_PROMPT": "0"}


def remote_org(remote: str) -> str:
    return f"refs/remotes/{remote}/agents/org"


def remote_memory(remote: str) -> str:
    return f"refs/remotes/{remote}/agents/memory"


def _remote_team(remote: str, node_id: str) -> str:
    return f"refs/remotes/{remote}/agents/teams/{node_id}"


def commit_at(repo_dir: Path, ref: str) -> str | None:
    try:
        out = git(repo_dir, ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"])
    except GitError:
        return None
    return out.strip()


def _is_ancestor(repo_dir: Path, older: str, newer: str) -> bool:
    try:
        git(repo_dir, ["merge-base", "--is-ancestor", older, newer])
    except GitError:
        return False
    return True


def fetch(repo_dir: Path, remote: str) -> None:
    """Bring the remote's org, team memory and team branches beside ours.

    Raises:
        NotFound: The repository has no such remote.
        UpstreamFailed: The remote could not be reached.
    """
    if remote not in git(repo_dir, ["remote"]).split():
        raise NotFound(f"The repository has no remote named {remote!r}")
    try:
        git(
            repo_dir,
            [
                "fetch",
                "--no-tags",
                "--quiet",
                "--prune",
                remote,
                f"+refs/agents/*:refs/remotes/{remote}/agents/*",
            ],
            env=_NO_PROMPT,
            timeout=_NETWORK_TIMEOUT_SECONDS,
        )
    except GitError as error:
        raise UpstreamFailed(
            f"Could not fetch the org from {remote}: {error.stderr.strip()}"
        ) from error


def _with_branch(nodes: Iterable[OrgNode]) -> list[OrgNode]:
    """The teams whose work is on a branch of their own, each branch once."""
    seen: set[str] = set()
    teams: list[OrgNode] = []
    for node in nodes:
        if node.isolation == "worktree" and node.branch and node.branch not in seen:
            seen.add(node.branch)
            teams.append(node)
    return teams


def take_in_branches(
    repo_dir: Path, remote: str, nodes: Iterable[OrgNode]
) -> list[BranchSync]:
    """Bring the teams' branches up to what was fetched.

    A branch that is not here is made. One that only the remote added to is
    moved forward. Where both sides added commits, the remote's are kept at
    a ref of their own and nothing here is changed.
    """
    changed: list[BranchSync] = []
    for node in _with_branch(nodes):
        theirs = commit_at(repo_dir, _remote_team(remote, node.id))
        if theirs is None:
            continue
        local = f"refs/heads/{node.branch}"
        ours = commit_at(repo_dir, local)
        result: BranchSync | None = None
        if ours is None:
            git(repo_dir, ["update-ref", local, theirs])
            result = _branch(node, "created")
        elif ours == theirs or _is_ancestor(repo_dir, theirs, ours):
            continue
        elif _is_ancestor(repo_dir, ours, theirs) and _forward(
            repo_dir, node, local, ours, theirs
        ):
            result = _branch(node, "fast_forwarded")
        else:
            kept = f"{TEAM_REFS}/{node.id}"
            git(repo_dir, ["update-ref", kept, theirs])
            result = _branch(node, "diverged", kept)
        changed.append(result)
    return changed


def _forward(repo_dir: Path, node: OrgNode, local: str, ours: str, theirs: str) -> bool:
    """Move a team's branch forward, and its checkout with it.

    Returns:
        False when the checkout has changes the new commits would overwrite.
    """
    checkout = Path(node.worktree_path)
    try:
        if checkout.is_dir():
            git(checkout, ["merge", "--ff-only", "--quiet", theirs])
        else:
            git(repo_dir, ["update-ref", local, theirs, ours])
    except (GitError, OSError):
        return False
    return True


def _branch(node: OrgNode, result: str, ref: str | None = None) -> BranchSync:
    return BranchSync.model_validate(
        {
            "node_id": node.id,
            "title": node.title,
            "branch": node.branch,
            "result": result,
            "ref": ref,
        }
    )


def behind(repo_dir: Path, remote: str) -> bool:
    """Whether the fetched org or team memory has commits ours lacks."""
    for ours, theirs in (
        (ORG_REF, remote_org(remote)),
        (MEMORY_REF, remote_memory(remote)),
    ):
        there = commit_at(repo_dir, theirs)
        if there is None:
            continue
        here = commit_at(repo_dir, ours)
        if here is None or not (here == there or _is_ancestor(repo_dir, there, here)):
            return True
    return False


def push(repo_dir: Path, remote: str, nodes: Iterable[OrgNode]) -> list[BranchSync]:
    """Give the remote the org, the team memory and the teams' branches.

    A team's branch that the remote has other commits on is left out and
    reported as diverged.

    Raises:
        UpstreamFailed: The remote could not be reached or refused the push.
    """
    refs = [
        f"{ref}:{ref}"
        for ref in (ORG_REF, MEMORY_REF)
        if commit_at(repo_dir, ref) is not None
    ]
    branches: list[BranchSync] = []
    for node in _with_branch(nodes):
        ours = commit_at(repo_dir, f"refs/heads/{node.branch}")
        if ours is None:
            continue
        theirs = commit_at(repo_dir, _remote_team(remote, node.id))
        if theirs == ours:
            continue
        if theirs is not None and not _is_ancestor(repo_dir, theirs, ours):
            branches.append(_branch(node, "diverged"))
            continue
        refs.append(f"refs/heads/{node.branch}:{TEAM_REFS}/{node.id}")
        branches.append(_branch(node, "pushed"))
    try:
        git(
            repo_dir,
            ["push", "--quiet", "--atomic", remote, *refs],
            env=_NO_PROMPT,
            timeout=_NETWORK_TIMEOUT_SECONDS,
        )
    except GitError as error:
        raise UpstreamFailed(
            f"Could not push the org to {remote}: {error.stderr.strip()}"
        ) from error
    return branches
