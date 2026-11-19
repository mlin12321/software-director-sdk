"""What a team's IC is asked about its own work before the IC above reviews
it: to write up what changed.

The question is put on top of the IC's conversation, which already holds
the work, so no diff is sent to a model. The review instructions are
OpenHands' public code review skill. They go to the IC above, with the
write-up, and that IC applies them when it reviews.
"""

import logging
from pathlib import Path

from openhands.agent_server.director.git import GitError, git, pathspec
from openhands.sdk.skills import KeywordTrigger, Skill, TaskTrigger, load_public_skills


logger = logging.getLogger(__name__)

_SKILL_NAMES = ("codereview", "code-review")
_SKILL_TRIGGER = "/codereview"

_WRITE_UP = """\
Your team reported its work done, and the IC of the team above reviews it \
next. Write the pull request description of everything your team changed \
since it started, for that reviewer. It has not seen your conversation.

Cover, in this order:
- What changed and why, in a few sentences.
- Each file that matters, with what now behaves differently in it. Files \
that only follow from another change can share one line.
- Anything that touches a contract with another team, or that another team \
will notice.
- How the work was tested, and what was not.
- What the reviewer should look at first.

Say only what you know from your work. git lists these changed files; \
account for all of them:
"""


def review_instructions() -> str | None:
    """OpenHands' public code review skill; None when it cannot be loaded."""
    try:
        # Every public skill, whatever the default marketplace lists.
        skills = load_public_skills(marketplace_path=None)
    except Exception:
        logger.warning("Could not load the public skills", exc_info=True)
        return None
    for skill in skills:
        if skill.name in _SKILL_NAMES or _SKILL_TRIGGER in _keywords(skill):
            return skill.content
    return None


def _keywords(skill: Skill) -> list[str]:
    trigger = skill.trigger
    if isinstance(trigger, KeywordTrigger):
        return trigger.keywords
    if isinstance(trigger, TaskTrigger):
        return trigger.triggers
    return []


def changed_files(checkout: Path, base: str, directory: str) -> str | None:
    """git's list of what a team committed under ``directory`` since
    ``base``, with the lines added and removed in each file.

    Returns:
        '' when nothing changed; None when git cannot tell.
    """
    try:
        return git(
            checkout, ["diff", "--stat=200", base, "HEAD", "--"] + [pathspec(directory)]
        ).rstrip()
    except (GitError, OSError):
        logger.warning("Could not diff %s against %s", checkout, base, exc_info=True)
        return None


def write_up_question(files: str) -> str:
    """Asks a team's IC to describe its team's changes for the reviewer."""
    return f"{_WRITE_UP}\n{files}"
