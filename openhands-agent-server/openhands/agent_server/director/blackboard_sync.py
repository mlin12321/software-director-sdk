"""Shares a repository's blackboard with a remote.

Nothing is shared on its own: a pull takes in what the remote has, a push
gives it what is here. Only the blackboard travels; the org tree and what a
prune kept belong to this machine.
"""

from openhands.agent_server.director.git import GitError, git
from openhands.agent_server.director.git_blackboard import (
    BLACKBOARD_REF,
    GitBlackboard,
)
from openhands.agent_server.director.models import BlackboardSync
from openhands.sdk.errors import Conflict, NotFound, UpstreamFailed


_NETWORK_TIMEOUT_SECONDS = 120
# The server has nobody to answer a prompt for a password.
_NO_PROMPT = {"GIT_TERMINAL_PROMPT": "0"}


def _tracking_ref(remote: str) -> str:
    return f"refs/remotes/{remote}/agents/blackboard"


def _require_remote(board: GitBlackboard, remote: str) -> None:
    if remote not in git(board.repo_dir, ["remote"]).split():
        raise NotFound(f"The repository has no remote named {remote!r}")


def _fetch(board: GitBlackboard, remote: str) -> bool:
    """Fetch the remote's blackboard.

    Returns:
        False when the remote has none.
    """
    try:
        git(
            board.repo_dir,
            ["fetch", "--no-tags", "--quiet", remote]
            + [f"+{BLACKBOARD_REF}:{_tracking_ref(remote)}"],
            env=_NO_PROMPT,
            timeout=_NETWORK_TIMEOUT_SECONDS,
        )
    except GitError as error:
        if "couldn't find remote ref" in error.stderr:
            return False
        raise UpstreamFailed(
            f"Could not fetch the blackboard from {remote}: {error.stderr.strip()}"
        ) from error
    return True


def pull(board: GitBlackboard, remote: str) -> BlackboardSync:
    """Take in the remote's blackboard.

    Raises:
        NotFound: The repository has no such remote.
        UpstreamFailed: The remote could not be reached.
        NotSupported: The installed git cannot merge blackboards.
    """
    _require_remote(board, remote)
    before = board.head()
    if not _fetch(board, remote):
        return BlackboardSync(remote=remote, head=before, changed=False, pushed=False)
    merged = board.merge(_tracking_ref(remote))
    return BlackboardSync(
        remote=remote,
        head=merged.head,
        changed=merged.head != before,
        kept_local=merged.kept_local,
        pushed=False,
    )


def _push(board: GitBlackboard, remote: str) -> None:
    git(
        board.repo_dir,
        ["push", "--quiet", remote, f"{BLACKBOARD_REF}:{BLACKBOARD_REF}"],
        env=_NO_PROMPT,
        timeout=_NETWORK_TIMEOUT_SECONDS,
    )


def push(board: GitBlackboard, remote: str) -> BlackboardSync:
    """Give the remote this blackboard.

    When the remote has changes this blackboard lacks, they are pulled in
    first and the push is made once more.

    Raises:
        NotFound: The repository has no such remote.
        Conflict: There is no blackboard to push.
        UpstreamFailed: The remote could not be reached or refused the push.
    """
    _require_remote(board, remote)
    head = board.head()
    if head is None:
        raise Conflict("There is no blackboard to push yet")
    try:
        _push(board, remote)
        return BlackboardSync(remote=remote, head=head, changed=False, pushed=True)
    except GitError as error:
        refused = f"{error.stderr}{error.stdout}"
        if "rejected" not in refused and "fetch first" not in refused:
            raise UpstreamFailed(
                f"Could not push the blackboard to {remote}: {error.stderr.strip()}"
            ) from error
    pulled = pull(board, remote)
    try:
        _push(board, remote)
    except GitError as error:
        raise UpstreamFailed(
            f"Could not push the blackboard to {remote}: {error.stderr.strip()}"
        ) from error
    return pulled.model_copy(update={"pushed": True})
