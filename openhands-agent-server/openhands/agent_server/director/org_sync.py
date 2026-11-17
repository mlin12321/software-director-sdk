"""Exchanges an org with a remote of its repository, as git does a branch.

Nothing is exchanged on its own. A push gives the remote the org tree and
the team memory; a fetch brings the remote's here, beside what is here, for
a merge. Sessions, with their checkouts and branches, stay on the machine
that has them.
"""

from pathlib import Path

from openhands.agent_server.director.git import GitError, git
from openhands.agent_server.director.git_store import MEMORY_REF
from openhands.agent_server.director.org_persistence import ORG_REF
from openhands.sdk.errors import NotFound, UpstreamFailed


_NETWORK_TIMEOUT_SECONDS = 120
# The server has nobody to answer a prompt for a password.
_NO_PROMPT = {"GIT_TERMINAL_PROMPT": "0"}


def remote_org(remote: str) -> str:
    return f"refs/remotes/{remote}/agents/org"


def remote_memory(remote: str) -> str:
    return f"refs/remotes/{remote}/agents/memory"


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
    """Bring the remote's org and team memory beside ours, and nothing
    else it keeps under ``refs/agents``: what a server keeps there for
    itself is not another server's.

    Raises:
        NotFound: The repository has no such remote.
        UpstreamFailed: The remote could not be reached.
    """
    if remote not in git(repo_dir, ["remote"]).split():
        raise NotFound(f"The repository has no remote named {remote!r}")
    shared = {ORG_REF: remote_org(remote), MEMORY_REF: remote_memory(remote)}
    try:
        listed = git(
            repo_dir,
            ["ls-remote", "--refs", remote, *shared],
            env=_NO_PROMPT,
            timeout=_NETWORK_TIMEOUT_SECONDS,
        )
        there = {line.split()[-1] for line in listed.splitlines() if line.strip()}
        for ref, beside in shared.items():
            if ref not in there and commit_at(repo_dir, beside) is not None:
                # The remote has it no more.
                git(repo_dir, ["update-ref", "-d", beside])
        if not there & shared.keys():
            return
        git(
            repo_dir,
            ["fetch", "--no-tags", "--quiet", remote]
            + [f"+{ref}:{beside}" for ref, beside in shared.items() if ref in there],
            env=_NO_PROMPT,
            timeout=_NETWORK_TIMEOUT_SECONDS,
        )
    except GitError as error:
        raise UpstreamFailed(
            f"Could not fetch the org from {remote}: {error.stderr.strip()}"
        ) from error


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


def push(repo_dir: Path, remote: str) -> None:
    """Give the remote the org and the team memory.

    Raises:
        UpstreamFailed: The remote could not be reached or refused the push.
    """
    refs = [
        f"{ref}:{ref}"
        for ref in (ORG_REF, MEMORY_REF)
        if commit_at(repo_dir, ref) is not None
    ]
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
