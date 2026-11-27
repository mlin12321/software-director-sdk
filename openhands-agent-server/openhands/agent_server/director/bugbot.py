"""What a team's IC is asked about its own work before the IC above reviews
it: to write up what changed.

The question is put on top of the IC's conversation, which already holds
the work, so no diff is sent to a model. The review instructions are
OpenHands' public code review skill. They go to the IC above, with the
write-up, and that IC applies them when it reviews.
"""

from openhands.sdk.logger import get_logger
from openhands.sdk.skills import KeywordTrigger, Skill, TaskTrigger, load_public_skills


logger = get_logger(__name__)

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


CODE_REVIEW_SKILLS = _SKILL_NAMES


def review_instructions() -> str | None:
    """OpenHands' public code review skill; None when it cannot be loaded."""
    return public_skill(_SKILL_NAMES, _SKILL_TRIGGER)


def public_skill(names: tuple[str, ...], trigger: str | None = None) -> str | None:
    """The public skill of one of ``names``, or the one ``trigger`` calls
    up; None when it cannot be loaded or is not there."""
    try:
        # Every public skill, whatever the default marketplace lists.
        skills = load_public_skills(marketplace_path=None)
    except Exception:
        logger.warning("Could not load the public skills", exc_info=True)
        return None
    for skill in skills:
        if skill.name in names or (trigger and trigger in _keywords(skill)):
            return skill.content
    return None


def _keywords(skill: Skill) -> list[str]:
    trigger = skill.trigger
    if isinstance(trigger, KeywordTrigger):
        return trigger.keywords
    if isinstance(trigger, TaskTrigger):
        return trigger.triggers
    return []


def write_up_question(files: str, asked: str = "") -> str:
    """Asks a team's IC to describe its team's changes for the reviewer.

    Args:
        asked: What a team of its type is asked to write, in place of the
            description of a pull request.
    """
    return f"{asked.strip() or _WRITE_UP.rstrip()}\n\n{files}"
