"""Fits the run state a director finds to the org it belongs to.

Run state is written by the server that ran the agents and is not part of
the org in git, so the two can part ways: the tree is pulled or restored,
or the repository is copied to another machine. What no longer fits is put
aside and named, never applied.
"""

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from uuid import UUID

from openhands.agent_server.director.git import GitError, git
from openhands.agent_server.director.models import (
    OrgNode,
    RecoveryItem,
    RecoveryKind,
)
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.run_state import (
    REVIEWING,
    RUNNING,
    TEAM,
    WAITING,
    Intent,
    RunState,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus


logger = logging.getLogger(__name__)

AGENT_LOST = "lost"
_STARTS = ("start_team", "launch_manager")
# The steps that start agents of their team: until they are finished, an
# agent of it without a conversation has not been lost.
STARTING = ("start_team", "launch_manager", "prune")
_WHAT: dict[str, str] = {
    "brief": "Its IC is given its task",
    "launch_manager": "Its Manager is started",
    "start_team": "The team is started",
    "split": "Its approved split is carried out",
    "finish_split": "Its split is finished",
    "request_review": "Its work is put up for review",
    "accept": "Its accepted work is merged into the team above",
    "prune": "Pruning it is finished",
    "halt": "Its agents are stopped",
}


def fingerprint(node: OrgNode) -> str:
    """What a team is made of, as far as its run state depends on it."""
    return json.dumps(
        [node.parent_id, [agent.conversation_id for agent in node.agents], node.branch]
    )


def has_work(run: RunState) -> bool:
    """Whether anything was left unfinished."""
    return bool(run.marked(RUNNING) or run.open_intents() or run.held())


def put_aside_stale(
    run: RunState, store: OrgStore, *, instance: str, org_commit: str | None
) -> list[RecoveryItem]:
    """Drop the run state that no longer fits the tree.

    Args:
        instance: Names the server this director runs in.
        org_commit: The commit the tree was loaded from.

    Returns:
        What was dropped, and why.
    """
    written_by = run.get("instance")
    if written_by is not None and written_by != instance:
        run.forget()
        return [
            _item(
                None,
                "dropped",
                "The run state beside the repository was written by another "
                "server, so none of it was used.",
            )
        ]
    nodes = {node.id: node for node in store.all()}
    stale: dict[str, str] = {}
    saved_at = run.get("org_commit")
    if saved_at is not None and saved_at != org_commit:
        for node_id, kept in run.marked(TEAM).items():
            node = nodes.get(node_id)
            if node is None:
                stale[node_id] = "the team is no longer in the org"
            elif fingerprint(node) != kept:
                stale[node_id] = "the team was changed since"
    agents = {
        agent.conversation_id
        for node in nodes.values()
        for agent in node.agents
        if agent.conversation_id is not None
    }
    dropped: list[RecoveryItem] = []
    for intent in run.open_intents():
        why = stale.get(intent.node_id) or _unfit(intent, nodes)
        if why is not None:
            run.finish(intent.id)
            dropped.append(
                _item(
                    nodes.get(intent.node_id),
                    "dropped",
                    f"{_WHAT.get(intent.kind, intent.kind)}: not done, {why}.",
                    node_id=intent.node_id,
                )
            )
    for message in run.held():
        why = stale.get(message.node_id)
        if why is None and message.conversation_id not in agents:
            why = "its agent is no longer in the org"
        if why is not None:
            run.delivered(message.seq)
            dropped.append(
                _item(
                    nodes.get(message.node_id),
                    "dropped",
                    f"A message was not delivered, {why}: {message.text.strip()[:120]}",
                    node_id=message.node_id,
                )
            )
    for conversation_id, node_id in run.marked(RUNNING).items():
        if node_id in stale or conversation_id not in agents:
            run.unmark(RUNNING, conversation_id)
    for kind in (WAITING, REVIEWING):
        for node_id in run.marked(kind):
            if node_id in stale or node_id not in nodes:
                run.unmark(kind, node_id)
    return dropped


def describe(
    run: RunState,
    store: OrgStore,
    statuses: Mapping[UUID, ConversationExecutionStatus],
    repo_dir: Path,
) -> list[RecoveryItem]:
    """What resuming the org does, and what it cannot do."""
    nodes = {node.id: node for node in store.all()}
    items: list[RecoveryItem] = []
    for conversation_id, node_id in run.marked(RUNNING).items():
        node = nodes.get(node_id)
        role = next(
            (
                agent.role
                for agent in (node.agents if node else [])
                if agent.conversation_id == conversation_id
            ),
            "ic",
        )
        name = "IC" if role == "ic" else "Manager"
        status = statuses.get(UUID(conversation_id))
        if status is None:
            items.append(
                _item(
                    node,
                    "cannot_resume",
                    f"Its {name} was working, and its conversation is gone.",
                )
            )
        elif status == ConversationExecutionStatus.ERROR:
            items.append(
                _item(
                    node,
                    "run_again",
                    f"Its {name} was cut off in the middle of a step. It is "
                    "told so and runs again.",
                )
            )
        elif status == ConversationExecutionStatus.PAUSED:
            items.append(_item(node, "run_again", f"Its {name} runs again."))
    waited = list(run.marked(WAITING))

    def turn(intent: Intent) -> tuple[int, int]:
        if intent.kind in _STARTS and intent.node_id in waited:
            return 0, waited.index(intent.node_id)
        return 1, intent.id

    starting: set[str] = set()
    for intent in sorted(run.open_intents(), key=turn):
        node = nodes.get(intent.node_id)
        what = _WHAT.get(intent.kind, intent.kind)
        if intent.kind in STARTING:
            starting.add(intent.node_id)
        if intent.kind == "start_team" and node is not None:
            ic = node.agents[0].conversation_id
            if ic is not None and UUID(ic) in statuses:
                what = _WHAT["launch_manager"]
        items.append(
            _item(
                node,
                "start" if intent.kind in _STARTS else "continue",
                f"{what}.",
                node_id=intent.node_id,
            )
        )
    waiting: dict[str, int] = {}
    for message in run.held():
        waiting[message.node_id] = waiting.get(message.node_id, 0) + 1
    for node_id, count in waiting.items():
        items.append(
            _item(
                nodes.get(node_id),
                "message",
                "1 message is delivered."
                if count == 1
                else f"{count} messages are delivered.",
                node_id=node_id,
            )
        )
    for node in missing_worktrees(nodes.values()):
        if _has_branch(repo_dir, node.branch):
            items.append(
                _item(
                    node,
                    "continue",
                    f"Its worktree is gone and is made again from its branch "
                    f"{node.branch}.",
                )
            )
        else:
            items.append(
                _item(
                    node,
                    "cannot_resume",
                    "Its worktree is gone and so is its branch"
                    + (f" {node.branch}." if node.branch else "."),
                )
            )
    for node in nodes.values():
        if node.status == "done" or node.id in starting:
            continue
        for agent in node.agents:
            if agent.status == AGENT_LOST:
                name = "IC" if agent.role == "ic" else "Manager"
                items.append(
                    _item(
                        node,
                        "cannot_resume",
                        f"Its {name} has no conversation on this server. A "
                        "conversation started in the team's folder starts the "
                        "team again on its work.",
                    )
                )
    return items


def missing_worktrees(nodes: Iterable[OrgNode]) -> list[OrgNode]:
    """The teams whose own worktree is not on disk, parents first."""
    seen: set[str] = set()
    missing: list[OrgNode] = []
    for node in nodes:
        path = node.worktree_path
        if node.isolation != "worktree" or path in seen:
            continue
        seen.add(path)
        if not Path(path).is_dir():
            missing.append(node)
    return missing


def restore_worktrees(repo_dir: Path, nodes: Iterable[OrgNode], within: str) -> None:
    """Make the worktrees that are gone again, each from its team's branch.

    Args:
        within: Where in a checkout the teams' workspaces are; '' at its top.
    """
    missing = [
        node for node in missing_worktrees(nodes) if _has_branch(repo_dir, node.branch)
    ]
    if not missing:
        return
    try:
        git(repo_dir, ["worktree", "prune"])
        for node in missing:
            top = Path(node.worktree_path)
            if within:
                top = Path(str(top).removesuffix(within).rstrip("/"))
            git(repo_dir, ["worktree", "add", str(top), node.branch])
    except (GitError, OSError):
        logger.exception("Could not make a worktree again in %s", repo_dir)


def _has_branch(repo_dir: Path, branch: str) -> bool:
    if not branch:
        return False
    try:
        git(repo_dir, ["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"])
    except (GitError, OSError):
        return False
    return True


def _unfit(intent: Intent, nodes: Mapping[str, OrgNode]) -> str | None:
    """Why a step that was begun can no longer be finished, if it cannot."""
    node = nodes.get(intent.node_id)
    if node is None:
        if intent.kind == "prune" and "removed" in intent.steps:
            return None
        return "the team is no longer in the org"
    split, review = node.pending_split, node.review
    if intent.kind in ("split", "finish_split"):
        if split is None or split.status != "approved":
            return "its split is no longer approved"
    elif intent.kind == "request_review":
        if review is None or review.state not in ("pending", "unreviewed"):
            return "its work no longer waits for a review"
    elif intent.kind == "accept":
        if review is None or review.state != "pending":
            return "its work no longer waits for a review"
    return None


def _item(
    node: OrgNode | None, kind: RecoveryKind, detail: str, *, node_id: str = ""
) -> RecoveryItem:
    return RecoveryItem(
        node_id=node.id if node else node_id,
        title=node.title if node else "",
        kind=kind,
        detail=detail,
    )
