"""Keeps what one agent has reported from growing without end.

Reports are only ever added. Once an agent's have grown past a size, a model
rewrites them into one shorter report, which then stands in their place.
"""

from openhands.agent_server.director.models import Report


USAGE_ID = "director-memory"
# The share of the size that must be new since the last rewrite, so that
# reports a model could not shorten are not read to it again at once.
_NEW_SHARE = 0.25

_INSTRUCTIONS = """\
These are the reports one agent of a software team recorded while it \
worked: design decisions, setbacks and reflections, oldest first. They \
have grown long. Rewrite them as one shorter report. Keep every decision \
and the reason for it, every approach that failed and why, and anything a \
later reader would need to avoid repeating a mistake. Where a later report \
overrides an earlier one, keep the later. Leave out what merely repeats.

Answer with the report alone, as plain text."""


def size(reports: list[Report]) -> int:
    """How many characters of text ``reports`` hold."""
    return sum(len(report.text) for report in reports)


def due(reports: list[Report], limit: int) -> bool:
    """Whether ``reports`` have outgrown ``limit`` characters and enough of
    that is new."""
    new = size([report for report in reports if not report.compacted])
    return size(reports) > limit and new >= limit * _NEW_SHARE


def prompt(reports: list[Report]) -> tuple[str, str]:
    """The system and the user message that ask for the reports rewritten."""
    return _INSTRUCTIONS, "\n\n".join(
        f"[{report.at}] {report.text}" for report in reports
    )


def compacted(answer: str, reports: list[Report], at: str) -> list[Report] | None:
    """The one report that replaces ``reports``; None unless the answer is
    shorter than they were."""
    text = answer.strip()
    if not reports or not text or len(text) >= size(reports):
        return None
    return [Report(at=at, role=reports[-1].role, text=text, compacted=True)]
