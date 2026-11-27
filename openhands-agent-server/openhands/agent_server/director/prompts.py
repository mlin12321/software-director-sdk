"""What the director tells the agents of an org."""

import textwrap
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from openhands.agent_server.director.agent_types import Names
from openhands.agent_server.director.git_store import team_key
from openhands.agent_server.director.models import (
    FormerTeam,
    MergeItem,
    Review,
    ReviewFinding,
    SessionTeam,
    SplitChild,
    TeamEvent,
    TeamEventKind,
)
from openhands.agent_server.director.split import is_inside


# The director's own texts name a member as `{writer}`, `{reviewer}` or
# `{manager}`, and :func:`in_words` puts in what the reader's team calls it.
_BASICS = (
    "- A team owns one folder. It has {count} agents: {members}.",
    """\
- Your first message says which team and which agent you are, and gives your
  team's context. Later changes arrive under `[director] context update` on
  the next message you get. `read_team_context` returns all of it again.
- Your team has a type. Your first message says under <TEAM_TYPE> what a
  team of that type is for, what the documents it keeps hold, and what the
  tools named here mean for it. Follow it; where it and this
  protocol differ, this protocol holds.
- Messages that start with [director] come from the runtime, not a person.
  A rare event (a review, a stalled team, replaced teams) is explained in
  the message that brings it.""",
    """\
- What you cannot settle, raise with `raise_escalation`. It goes to the same
  role one team up, and from the team your session started at, to the human.
  Settle what you can; pass up the rest with the reason.""",
)

# What the member that works in the folder does, and what the member that
# reviews the teams below does. In most types they are one member, and the
# two read as one part, in this order.
_PARTS: tuple[tuple[Literal["writer", "reviewer"], str], ...] = (
    (
        "writer",
        """\
- Do your team's work yourself.
- Write only inside your folder; the runtime refuses anything else. To get
  another team's files changed, raise an escalation: the {above_writer} above both
  teams decides.
- Do what the human asks your team directly, inside your folder.
- When the work is done: commit, then call `submit_review`. Your turn ends.
  The {above_reviewer} one team up reviews it (for the team your session started at, that
  is the Division Lead {above_reviewer}) and you are woken with the result.
- If you cannot go on, raise an escalation. That shows your team as blocked.
- Propose sub-teams to your {manager} with `propose_teams`. Each gets a
  folder inside yours, a type and an instruction; a `reason` is optional. A
  sub-team is told nothing else of your team. You keep whatever no sub-team
  takes.
- An approved sub-team starts only when you give it a task with `msg_team`.
  It answers through its work or an escalation, not a message.""",  # noqa: E501
    ),
    (
        "reviewer",
        """\
- Review what sub-teams submit with `review_team`.
- You are asked to review a sub-team's work as soon as it submits, while
  the teams beside it may still work. You may end your turn without
  deciding, to compare it with what another sub-team will submit: the next
  request names what still waits for you.""",
    ),
    (
        "writer",
        """\
- Your own `submit_review` is refused while a sub-team of yours works,
  waits for {review_of_yours}, or was sent back and has not submitted again.""",
    ),
    (
        "reviewer",
        """\
- Settle what their {writers} escalate with `resolve_escalation`, or pass it up
  with the reason.""",
    ),
)


def _bullet(line: str) -> str:
    """A point of one line, wrapped as the points written out here are."""
    return textwrap.fill(line, width=76, subsequent_indent="  ")


def rules_said(
    rules: Sequence[Mapping[str, str]], whom: str, words: Mapping[str, str]
) -> list[str]:
    """What the documents a team keeps say to ``whom``: a line for each
    that says something, in the words of the team."""
    return [
        " ".join(said[whom].format_map(words).split())
        for said in rules
        if said.get(whom, "").strip()
    ]


def _rules_for(
    rules: Sequence[Mapping[str, str]], whom: str, words: Mapping[str, str]
) -> list[str]:
    """What the documents a team keeps say to ``whom`` of its members, as
    the points of a list."""
    return [_bullet(f"- {line}") for line in rules_said(rules, whom, words)]


def org_protocol(names: Names, rules: Sequence[Mapping[str, str]] = ()) -> str:
    """How the org works, for every agent of a team whose members are
    called by ``names``: one part for each member that has one.

    Args:
        rules: What each document the team keeps says of itself, by whom
            it says it to (``Document.rules``).
    """
    words = names.words()
    basics = [
        _bullet(text.format_map(words)) if "\n" not in text else text
        for text in _BASICS
    ]
    basics += _rules_for(rules, "everyone", words)
    parts = (
        [(names.writer, ("writer", "reviewer"))]
        if names.reviews_own
        else [(names.writer, ("writer",)), (names.reviewer, ("reviewer",))]
    )
    lines = [
        "<ORG_PROTOCOL>",
        "You are one agent among teams of agents that work in one repository.",
        "",
        "Basics",
        *basics,
    ]
    for called, theirs in parts:
        lines += ["", f"If you are the {called}"]
        lines += [text.format_map(words) for part, text in _PARTS if part in theirs]
        for part in theirs:
            lines += _rules_for(rules, part, words)
    return "\n".join([*lines, "</ORG_PROTOCOL>"])


# What a writer is told once it has sub-teams, with the notice that they are
# set up: none of it concerns a team that never splits.
WITH_SUB_TEAMS = (
    "- Each change is reviewed once, by the {reviewer} directly above the team that "
    "made it. If you change nothing yourself, your sub-teams' accepted work "
    "goes on up without another review. If you do, call `submit_review` "
    "when your own part is complete, and the accepted work goes up with it.",
    "- Tell your {manager} with `msg_manager`, at your own discretion, how "
    "your teams are doing: when one struggles and you cannot fix it, when "
    "two are at odds, when the make-up of the teams no longer fits the work.",
    "- To change the teams, propose again. The block must say what becomes "
    "of each team you have: list under `keep`, by name, every one that is "
    "to stay as it is; every other one must be replaced by a proposed team "
    "that covers its folder or takes part of it. A team is cleared by "
    "proposing a different team for the same folder. When two teams keep "
    "being held up by each other, or a team has nothing left to do, propose "
    "teams that merge or replace them.",
    "- The runtime keeps what became of your teams and of what you "
    "proposed, also why it was declined; a review request says how a "
    "team's earlier work went and `read_team_context` returns the record. "
    "Do not propose again a division that failed or was declined unless "
    "something has changed.",
)

DIVISION_LEAD_IC_ROLE = """\
<ORG_ROLE>
You are the Division Lead {reviewer}: the {reviewer} of the team above the one a session
worked at. You only review that session's work. You give no tasks and
change no teams.
- Judge the work by the human's request and by what your team keeps and its
  records. Read it in the session's checkout; run its checks there.
- Decide with `review_team`: accept, or reject with what must change.
- Work is accepted in the order it was submitted. If earlier work is still
  undecided you are told to wait, and told when it is your turn.
- If the work overlaps what was merged meanwhile, you are shown both. Say in
  `reconciled` what you decided.
- If it does not merge, the runtime lays the merge out and names the files.
  Edit only those files there, then accept again. This is the only time you
  write files. The session's team is not sent back for a merge.
- Teams are not yours: the Division Lead {manager} settles them.
</ORG_ROLE>"""

DIVISION_LEAD_MANAGER_ROLE = """\
<ORG_ROLE>
You are the Division Lead {manager}: the {manager} of the team above the one a
session worked at. Ignore {other_parts} above. You write no files and review
no code; the Division Lead {reviewer} does that, and neither of you waits for the
other.
- You are asked only when the session and other accepted work both changed
  the same team, or what is settled with the teams. The message lists each
  one with the session's version and the org's.
- Settle all of it in one `settle_teams` call. Per team: the org's or the
  session's; a team taken from the session may go under another team or get
  another name.
- Decide by what your team keeps and its records; `read_team_context`
  returns them.
- Decide soon: later sessions' teams, and what is settled with them, wait
  for you. Their code does not.
</ORG_ROLE>"""

MANAGER_ROLE = """\
<ORG_ROLE>
You are your team's {manager}, not its {writer}. Ignore {other_parts} above.
You never edit files or run commands that change the repository; the runtime
refuses it.

Your job: the right sub-teams, working well together. The aim is that the
org as a whole works well, not that every team is busy.
- Your {writer} proposes sub-teams. `approve_teams` if they do not overlap, each
  instruction is clear, and each is a team your team's type says to have;
  they are then set up
  as written and start when your {writer} gives each a task. Otherwise
  `decline_teams` with the reason and what to change. A proposal waits until
  you call one of the two; a message decides nothing.
- `msg_ic` writes to your {writer}, or to the {writer} of a team directly below.
- `read_team_context` shows how every team below has done, also removed
  ones, and what your {writer} proposed before. Weigh a proposal by it.
- After you approve or prune, your {writer} passes on what removed teams left.
  Write to it if the change needs explaining.
- Settle what {managers} directly below escalate with `resolve_escalation`,
  or pass it up with the reason.
- `prune_team` removes a team directly below; your team's type says when.
  Write to a slow team first. Its folder goes back to your {writer}.
- You are woken when your {writer} writes, {a_manager} below escalates, or the
  runtime asks you to look at your teams and says what it counted.
</ORG_ROLE>"""

HAND_OVER_ROLE = """\
<ORG_ROLE>
You are your team's {writer} in hand-over mode: a side conversation started for
one hand-over. It does not carry on the team's work. Do not edit files or
run commands that change the repository. Only `read_team_context` and
`hand_over` work here, and what is named below.
</ORG_ROLE>"""

# The role texts, by whom a document's `rules` say something to.
ROLE_TEXTS: dict[str, str] = {
    "manager": MANAGER_ROLE,
    "lead_reviewer": DIVISION_LEAD_IC_ROLE,
    "lead_manager": DIVISION_LEAD_MANAGER_ROLE,
    "hand_over": HAND_OVER_ROLE,
}


def role_block(whom: str, names: Names, rules: Sequence[Mapping[str, str]] = ()) -> str:
    """One of the role texts, in the words of a team, with what the
    documents the team keeps say to that role."""
    text = ROLE_TEXTS[whom].strip().format_map(names.words())
    said = _rules_for(rules, whom, names.words())
    if not said:
        return text
    body, closing = text.rsplit("\n", 1)
    return "\n".join([body, *said, closing])


DIRECTOR_PREFIX = "[director]"


def _now_with(team: FormerTeam, children: list[SplitChild]) -> str:
    """Who has a replaced team's directory under the new boundaries."""
    where = team.path or "the repository"
    inside = [c for c in children if is_inside(c.path, team.path)]
    over = [c for c in children if c.path == team.path or is_inside(team.path, c.path)]
    if over:
        [child] = over[:1]
        return f"Its folder is now in the folder of {child.title} ({child.path})."
    if inside:
        names = ", ".join(f"{child.title} ({child.path})" for child in inside)
        return (
            f"Its folder is now divided: {names}. What of {where} is in "
            "none of those is yours again."
        )
    return f"No new team has its folder: {where} is yours again."


def left_behind(
    former: list[FormerTeam],
    children: list[SplitChild] | None = None,
    specs: dict[str, str] | None = None,
) -> list[str]:
    """What removed teams left, as the IC that is to pass it on reads it.

    Args:
        children: The teams that replaced them, to say who has each
            directory now.
        specs: The spec each of them wrote, by its directory.
    """
    lines: list[str] = []
    for team in former:
        lines += [
            "",
            f"Team {team.title!r} (id {team.id}, {team.path or 'the repository'}), "
            f"{team.ended}. It was for: {team.purpose or 'nothing that was kept'}",
            "Its reports, as one:",
            team.compacted_report or "It recorded nothing.",
        ]
        if children is not None:
            lines.insert(-2, _now_with(team, children))
        for name, text in team.documents.items():
            if text.strip():
                lines += [f"Its {name}:", text.strip()]
        spec = (specs or {}).get(team.path, "").strip()
        if spec:
            lines += [
                "The spec it wrote (old; it stays in "
                f"{team.path or 'the repository'}):",
                spec,
            ]
    return lines


def _approved(children: list[SplitChild], *, names: Names) -> str:
    teams = ", ".join(f"{child.title} ({child.path})" for child in children)
    return (
        f"Your team was split: your {names.manager} approved your sub-teams and they "
        f"now own their folders: {teams}. None of them has started: a "
        "team starts when you give it its task with msg_team."
    )


def _named(files: list[str], most: int = 10) -> str:
    more = f" and {len(files) - most} more" if len(files) > most else ""
    return ", ".join(files[:most]) + more


def _split_effects(children: list[SplitChild], unfinished: dict[str, list[str]]) -> str:
    """How a split changes what the IC that proposed it may do.

    Args:
        unfinished: By directory of a new team, the files the IC has
            changed there and not committed.
    """
    lines = [
        "How the split changes your own work:",
        "- What remains in your folder outside those folders is yours as before.",
        "- What you committed in those folders is where each team "
        "starts. Your `git` no longer shows or commits anything there.",
    ]
    left = [child for child in children if unfinished.get(child.path)]
    if left:
        lines.append("- You have changes there that are not committed:")
        lines += [
            f"  - {child.title} ({child.path}): {_named(unfinished[child.path])}"
            for child in left
        ]
        lines.append(
            "  They passed to those teams with the folders. You may finish "
            "them: you can write in a team's folder until you give the team "
            "its task. The team then finds them as uncommitted work of its "
            "own, so say in its task what they are for. It checks them, "
            "commits them and submits them, and you review that. You cannot "
            "commit or submit them yourself."
        )
    else:
        lines.append(
            "- You left nothing uncommitted there. Until you give a team its "
            "task you can still write in its folder, but what you write "
            "there is the team's: it commits it and submits it, and you "
            "review that."
        )
    lines.append(
        "- Once a team has its task, a write of yours in its folder is "
        "refused. Tell the team what you need with msg_team."
    )
    return "\n".join(lines)


def put_aside_notice(team: SessionTeam, files: list[str], ref: str) -> str:
    """What an IC is told when changes it made in a started sub-team's
    directory were taken out of its checkout."""
    return (
        f"Your checkout held uncommitted changes in {team.boundary.path}, "
        f"which is the folder of your sub-team {team.title!r}: "
        f"{_named(files)}. They were in the way of the work you accepted "
        f"and were taken out; they are kept at {ref}. If they are still "
        "needed, tell the team with msg_team."
    )


def left_for_team(files: list[str], *, names: Names) -> str:
    """What a team is told, with its task, of the changes the IC above left
    uncommitted in its directory."""
    return (
        f"The {names.above_writer} of the team above changed these files in your "
        "folder "
        f"before your team started and did not commit them: {_named(files)}. "
        "They are yours now. Check them, change what is wrong, and commit "
        "them with your work before you submit it."
    )


_ASIDE = (
    "What they left (reports, contracts, specs) is being passed on in "
    "hand-over mode: a side conversation of yours, which the director has "
    "started with all of it. It has a context of its own and sees nothing "
    "of this conversation. There you give your teams their notes with "
    "`hand_over` and rewrite your team's spec with `write_spec`; you are "
    "told here when it has ended and need do neither here."
)

_HAND_OVER = (
    "- Give each new team what it should know of this with `hand_over`, "
    "in your own words or quoted. Where one team's folder is now divided "
    "among several, give each the part that concerns its folder. The "
    "team is told with its next message and writes what should last into "
    "its own documents. It starts nothing and can be done again later; "
    "`read_team_context` returns what the removed teams left "
    "(`former_teams`)."
)

_SPEC_STAYS = (
    "- A spec a removed team wrote stays in its folder. The team that has "
    "the folder now is told it is old."
)

_REWRITE_SPEC = (
    "- Rewrite your own spec with `write_spec` to say how your folder is divided now."
)


def approved_notice(
    children: list[SplitChild],
    former: list[FormerTeam],
    *,
    unfinished: dict[str, list[str]],
    names: Names,
    rules: Sequence[Mapping[str, str]] = (),
) -> str:
    """What an IC is told once the sub-teams it proposed are set up.

    Args:
        unfinished: By directory of a new team, the files the IC has
            changed there and not committed.
        former: The teams they replaced. What those left is passed on in
            hand-over mode and is not given here.
        rules: What each document the team keeps says of itself; the
            `teams_replaced` of each is told where teams were replaced.
    """
    lines = [_approved(children, names=names), "", _split_effects(children, unfinished)]
    todo = []
    if former:
        gone = ", ".join(
            f"{team.title} ({team.path or 'the repository'})" for team in former
        )
        lines += [
            "",
            f"The teams they replaced are gone: {gone}. What those had "
            f"committed is in your checkout. {_ASIDE}",
        ]
        todo += _rules_for(rules, "teams_replaced", names.words())
    else:
        todo += [_REWRITE_SPEC]
    todo += ["- Give a team its task with `msg_team` when there is one."]
    return "\n".join(
        [
            *lines,
            "",
            "What is now yours to do:",
            *todo,
            "",
            "What holds while you have sub-teams:",
            *(in_words(line, names) for line in WITH_SUB_TEAMS),
        ]
    )


def replaced_job(
    teams: list[SessionTeam],
    children: list[SplitChild],
    former: list[FormerTeam],
    specs: dict[str, str],
    *,
    names: Names,
    rules: Sequence[Mapping[str, str]] = (),
) -> str:
    """What a hand-over conversation is to do after teams were replaced.

    Args:
        teams: The sub-teams that were set up.
        specs: The spec each replaced team wrote, by its directory.
        rules: What each document the team keeps says of itself; the
            `teams_replaced` of each is told.
    """
    lines = [
        f"What changed: your {names.manager} approved the sub-teams you proposed, and "
        "they now own their folders. None of them has started; each "
        "starts when your main conversation gives it its task.",
        *(
            f"- {team.title} (id {team.id}, {team.boundary.path}): "
            f"{team.summary or 'no purpose given'}"
            for team in teams
        ),
        "",
        "The teams they replaced are gone; what those had committed is in the "
        "checkout. What each of them left is below. Nothing of it reaches a "
        "new team unless you hand it over.",
        *left_behind(former, children, specs),
        "",
        "What is yours to do here:",
        _HAND_OVER,
        _SPEC_STAYS,
        *_rules_for(rules, "teams_replaced", names.words()),
        _REWRITE_SPEC,
    ]
    return "\n".join(lines)


def pruned_job(
    title: str,
    where: str,
    reason: str,
    former: list[FormerTeam],
    specs: dict[str, str],
    *,
    names: Names,
) -> str:
    """What a hand-over conversation is to do after a team was pruned."""
    lines = [
        f"What changed: your {names.manager} pruned the team {title!r} ({where}): "
        f"{reason} {where} is your team's again.",
        "",
        "What it left is below. It is yours to use or to pass on.",
        *left_behind(former, None, specs),
        "",
        "What is yours to do here:",
        _REWRITE_SPEC + " The spec the pruned team wrote stays in its "
        "folder and is old.",
        "- If a team you still have should know of what the pruned team "
        "left, give it to that team with `hand_over`.",
    ]
    return "\n".join(lines)


HAND_OVER_FAILED = (
    "Your hand-over mode could not be started: its side conversation did "
    "not start, also when it was tried again. Nothing was handed over."
)


CHANGED_SINCE_ACCEPT = (
    "Your team's work was accepted, and your team has changed since: its "
    "teams, its spec or its contracts. The org does not have these changes "
    "yet. When your team is done, submit with `submit_review`: the org takes "
    "them when that is accepted."
)


def handed_over_notice(given: list[str], spec: bool) -> str:
    """What an IC is told in its main conversation once its hand-over
    conversation has ended."""
    to = (
        f"It handed over to: {', '.join(given)}."
        if given
        else "It gave nothing to any team."
    )
    spec_line = (
        "It rewrote your team's spec."
        if spec
        else "It did not rewrite your team's spec: do that with `write_spec`."
    )
    return (
        f"Your hand-over mode has ended. {to} {spec_line} "
        "`read_team_context` returns what the removed teams left "
        "(`former_teams`) if more of it should be given with `hand_over`."
    )


def pruned_notice(
    title: str,
    where: str,
    reason: str,
    saved: str | None,
    former: list[FormerTeam],
    *,
    names: Names,
) -> str:
    """What the IC above is told once its Manager's prune is carried out.

    Args:
        former: What the pruned team left, if it left a record. It is
            passed on in hand-over mode and is not given here.
    """
    pruned = (
        f"Your {names.manager} pruned the team {title!r} ({where}): {reason} Its "
        "uncommitted changes were discarded"
        + (f" (kept in git as {saved})" if saved else "")
        + f"; {where} is yours again."
    )
    if former:
        return f"{pruned}\n\n{_ASIDE}"
    return "\n".join([pruned, "", "What is now yours to do:", _REWRITE_SPEC])


def left_document(name: str, path: str, left_by: str | None, tool: str) -> str:
    """What is said over a document a team before the reader's left in its
    folder.

    Args:
        tool: The tool the reader's team writes its own with.
    """
    by = f" ({left_by})" if left_by else ""
    return (
        f"A {name} left by a team before you{by}, in "
        f"{path or 'the repository'}. It is old: it says how that "
        "team worked. Take from it what still holds and write your "
        f"team's own with `{tool}`."
    )


def _of_the_team(documents: Sequence[tuple[str, str]], context: str) -> list[str]:
    """The documents and context a briefing gives.

    Args:
        documents: Each part its documents add: a line that says what it
            is, and the text.
    """
    lines: list[str] = []
    for heading, text in documents:
        lines += ["", heading, text]
    if context:
        lines += ["", context]
    return lines


def in_words(text: str, names: Names) -> str:
    """A text of the director's own, with the members named as the reader's
    team names them. Only for text that holds nothing an agent or a person
    wrote: their words are never passed through here."""
    return text.format_map(names.words())


def role_text(
    names: Names,
    *,
    works: bool,
    tools: Sequence[str] = (),
    title: str = "",
    lead: bool = False,
    hand_over: bool = False,
    own: str = "",
    rules: Sequence[Mapping[str, str]] = (),
) -> str:
    """How the org works and what an agent's part in it is: what every
    agent of an org is told in its first message, and again when it reads
    its team's context. It is no part of the agent's system prompt, so an
    agent carries nothing of the org but what its conversation was told.

    Its part is told by what it may do, not by what it is called.

    Args:
        names: What the members of its team are called.
        works: Whether it is the member that works in its team's folder.
        tools: The director's tools it has where it is.
        title: What it is called itself.
        lead: Whether its team only reviews the work of a session.
        hand_over: Whether it is a hand-over conversation.
        own: What its type says of its part, in the type's own words.
        rules: What each document its team keeps says of itself.
    """
    texts = []
    if hand_over:
        texts.append("hand_over")
    elif lead and "review_team" in tools:
        texts.append("lead_reviewer")
    elif lead and "settle_teams" in tools:
        texts.append("lead_manager")
    elif not lead and "approve_teams" in tools:
        texts.append("manager")
    said = [org_protocol(names, rules)]
    said += [role_block(whom, names, rules) for whom in texts]
    if not works and not hand_over and not texts:
        said.append(
            f"You are the {title} of your team. You write no files. What you "
            "do, you do with these tools of the director: "
            f"{', '.join(f'`{tool}`' for tool in tools) or 'none'}."
        )
    if own.strip():
        said.append(own.strip())
    return "\n\n".join(said)


def hand_over_brief(
    place: SessionTeam,
    job: str,
    documents: Sequence[tuple[str, str]],
    context: str = "",
    *,
    role: str = "",
    names: Names,
) -> str:
    """The first message of a hand-over conversation: a side conversation
    of a team's IC that knows nothing but what this gives it.

    Args:
        role: How the org works and what this conversation is for.
        job: What changed, what the removed teams left and what to do.
        documents: What its team's documents say, as each has it said.
        context: The rest of the team's context, as the agent is to read it.
    """
    lines = [
        f"{DIRECTOR_PREFIX} You are the {names.writer} of the team {place.title!r} (id "
        f"{place.id}), in hand-over mode.",
        "Your team was changed, and this side conversation was started for "
        "one job: to pass on what the removed teams left. It has a context of "
        "its own. Your main conversation carries on with the team's work and "
        "sees nothing of this one, and you see nothing of it: what follows "
        "is everything you are given.",
        f"Folder of your team: {place.boundary.path or 'the repository'}",
        *(["", role] if role else []),
        "",
        job,
        "",
        "What your team works to, so that you can judge what to pass on:",
        *_of_the_team(documents, context),
        "",
        "Do the hand-over now. You may read files; change none. When it is "
        "done, end with a line or two on what you gave to whom.",
    ]
    return "\n".join(lines)


def brief(
    place: SessionTeam,
    role: str,
    task: str,
    documents: Sequence[tuple[str, str]],
    context: str = "",
    kind: str = "",
    *,
    works: bool,
    titles: Mapping[str, str],
    role_said: str = "",
    names: Names,
) -> str:
    """The first message of an agent: who it is and what it works to.

    Args:
        role_said: How the org works and what its part in it is.
        documents: What its team's documents say, as each has it said.
        works: Whether it is the member that works in the team's folder;
            None for an IC.
        context: The rest of the team's context, as the agent is to read it.
        kind: What a team of its type is for, as the type says it to an
            agent of this role.
    """
    boundary = place.boundary.path or "the repository"
    lines = [
        f"{DIRECTOR_PREFIX} You are the {'Division Lead ' * place.division_lead}"
        f"{titles.get(role, role)} of the team {place.title!r} "
        f"(id {place.id}).",
        f"Folder: {boundary}",
        f"Writable: {', '.join(place.boundary.writable) or 'nothing'}",
    ]
    if place.boundary.readonly:
        lines.append(f"Read-only: {', '.join(place.boundary.readonly)}")
    if role_said:
        lines += ["", role_said]
    if kind:
        lines += ["", kind]
    lines += ["", "Task of the team:", task]
    lines += _of_the_team(documents, context)
    lines.append("")
    if works:
        lines.append("Start on the task now.")
    elif titles.get(role) == names.manager:
        lines.append(f"Wait until your {names.writer} writes to you.")
    else:
        lines.append("Wait until something is sent to you.")
    return "\n".join(lines)


def _finding_lines(findings: list[ReviewFinding]) -> list[str]:
    lines = []
    for finding in findings:
        where = finding.path + (f":{finding.line}" if finding.line else "")
        lines.append(f"- {where + ': ' if where else ''}{finding.issue}")
        if finding.evidence:
            lines.append(f"  Evidence: {finding.evidence}")
    return lines


def session_work(
    team: SessionTeam,
    title: str,
    request: str,
    contracts: Sequence[tuple[str, str]] = (),
    *,
    names: Names,
) -> str:
    """What comes before the review request for the work of the session
    titled ``title``, which was opened for ``request``.

    Args:
        contracts: The contracts it wrote that are not the org's: each
            directory with the session's text.
    """
    written = "".join(
        f"Contracts it wrote for {directory or 'the repository'}, which its "
        "code is written to. They are not the org's yet: the Division Lead "
        f"{names.manager} settles them if other accepted work changed them too. "
        f"Judge the code by these:\n{text or '(none)'}\n\n"
        for directory, text in contracts
    )
    return (
        f"The session {title!r} at {team.title!r} submits its work. It was "
        f"opened for this request:\n{request}\n\nIt worked in a checkout of "
        f"its own ({team.worktree_path}, branch {team.branch}). Judge its "
        "work by that request and by your team's spec, contracts and "
        "records. Accepted work is merged into the repository's checkout in "
        "the order work was submitted. If it does not merge with what went "
        "in meanwhile, the merge is laid out for you to finish. A spec it "
        "wrote and sub-teams it set up are its own until you accept its "
        "work and the org's from then on.\n\n" + written
    )


def accept_in_order(
    team: SessionTeam, first: SessionTeam, session: str = "", *, names: Names
) -> str:
    """Why work is not accepted while work submitted before it waits.

    Args:
        session: What the session of the earlier work is called, when
            another reviewer than the one that is told decides on it.
    """
    if session:
        return (
            f"The work of {first.title!r} in the session {session!r} was "
            f"submitted before that of {team.title!r}, and its Division Lead "
            f"{names.reviewer} has "
            "not decided on it yet. Work is accepted in the order it was "
            "submitted, so that each is checked against what was merged "
            "before it. There is nothing for you to do until then: you are "
            "told when it is your turn, and accept with `review_team` again "
            "then."
        )
    return (
        f"The work of {first.title!r} (id {first.id}) was submitted before "
        f"that of {team.title!r} and still waits for your review. Work is "
        "accepted in the order it was submitted, so that each is checked "
        "against what was merged before it: accept that one or send it back "
        "first."
    )


def reconcile_request(
    team: SessionTeam,
    checkout: str,
    tip: str,
    files: list[str],
    landed: list[str],
    specs: list[tuple[str, str, str]],
    inherited: Sequence[tuple[str, str, str, str]] = (),
    *,
    names: Names,
) -> str:
    """What a reviewing IC is told when work it accepts meets other work.

    Args:
        checkout: The reviewer's checkout.
        tip: The last commit of the work.
        files: The files changed both by it and in the reviewer's checkout
            since it began.
        landed: The teams whose work was merged there meanwhile.
        specs: For each spec it wrote while the team's spec changed: the
            directory, the team's spec now and the one it wrote.
        inherited: For each spec it wrote for a team the org no longer
            has: the directory of the team it is kept with, that team's
            spec now, what the lost team was called and the spec.
    """
    lines = [
        f"Not merged yet: the work of {team.title!r} (id {team.id}) meets "
        "work that came into your checkout since it began"
        + (f" (merged meanwhile: {', '.join(landed)})." if landed else "."),
    ]
    if files:
        lines += [
            "",
            "Both changed these files. git reports only lines that cannot "
            "be merged; where they can, one may still have undone or "
            "contradicted the other:",
            *(f"- {path}" for path in files[:40]),
            *([f"- and {len(files) - 40} more"] if len(files) > 40 else []),
            "",
            f"Read both, in {checkout}:",
            f"  git diff {tip}...HEAD -- <file>   (what changed in your checkout)",
            f"  git diff HEAD...{tip} -- <file>   (what {team.title} changed)",
        ]
    for directory, theirs, written in specs:
        lines += [
            "",
            f"It wrote a spec for {directory or 'the repository'} while the "
            "team's spec there was changed by someone else. Accepting puts "
            "its spec in place of the team's.",
            "The team's spec now:",
            theirs or "(none)",
            "The spec it wrote:",
            written,
        ]
    for home, theirs, lost, written in inherited:
        lines += [
            "",
            f"It wrote a spec for {lost!r}, a team the org no longer has. "
            f"The team that holds its folder now is the one in "
            f"{home or 'the repository'}, which is left that spec as an old "
            "one. If something in it has to hold on, write the spec of "
            f"{home or 'the repository'} from both in `merge_choices`; otherwise "
            "the next session there is told to.",
            f"The spec of {home or 'the repository'} now:",
            theirs or "(none)",
            f"The spec written for {lost!r}:",
            written,
        ]
    lines += [
        "",
        "Decide what holds. If its work must change, reject it with what "
        "must change. If what is already in your checkout must change, "
        f"change it or give that team the task; a Division Lead {names.reviewer}, "
        "who does "
        "neither, records it with `report`. Then accept again with "
        "`review_team` and say in `reconciled` what you decided; without "
        "that it is not merged."
        + (
            " For each spec above you may say in `merge_choices` whether the org's "
            "stays, the session's holds, or a text you write from both does; "
            "unnamed, the session's holds."
            if specs or inherited
            else ""
        ),
    ]
    return "\n".join(lines)


def merge_prepared(team: SessionTeam, where: str, files: list[str]) -> str:
    """What a reviewer is told when the work it accepts does not merge by
    itself: the merge is laid out for it to finish."""
    return (
        f"The work of {team.title!r} (id {team.id}) does not merge with your "
        f"checkout by itself. These files do not merge: {', '.join(files)}\n\n"
        f"The merge is laid out in {where}, with git's conflict markers in "
        "those files and everything else merged. It is yours to finish, so "
        "that the team does not lose its turn:\n"
        "- Edit those files there until each is what both sides need, with "
        "no conflict markers left. Change no other file there.\n"
        "- Do not run git there: the merge is committed for you.\n"
        "- Then accept the work again with `review_team`, and say in "
        "`reconciled` what you made of them. The merge is taken as you left "
        "it and the team is told what you changed.\n\n"
        "If it is not yours to decide, send the work back with `review_team` "
        "and decision 'reject' instead."
    )


def merge_request(
    team: SessionTeam,
    session: str,
    waiting: list[MergeItem],
    accepted: bool,
    *,
    names: Names,
) -> str:
    """What the Manager of a session's Division Lead place is told
    when teams or contracts of the session wait for it.

    Args:
        waiting: What waits for it, with what it decided before of what is
            unchanged.
        accepted: Whether the session's code is accepted and merged.
    """
    lines = [
        f"The work of {team.title!r} (id {team.id}) in the session "
        f"{session!r} "
        + (
            "is accepted and its code is merged."
            if accepted
            else f"is submitted and its {names.reviewer} reviews the code."
        )
        + " This of its teams and contracts the org made something else of "
        "while the session worked:",
    ]
    for item in waiting:
        if item.kind == "team":
            lines += [
                "",
                f"- {item.title or item.key} (team {item.key})",
                f"  the session: {item.session}",
                f"  the org now: {item.org}",
            ]
            if item.moved_to is not None:
                lines.append(
                    "  the team it was set up under is not in the org any more: "
                    f"taken, it goes under team {item.moved_to}, and that "
                    "team's sub-teams inside its folder go under it. The "
                    "team that is gone is not put back."
                )
        else:
            lines += [
                "",
                f"- the contracts between the sub-teams of "
                f"{item.key or 'the repository'}"
                + (
                    f" (they come with team {item.with_team} and hold only if "
                    "it is taken)"
                    if item.with_team
                    else ""
                ),
                "  the session's:",
                *(f"    {line}" for line in (item.session or "(none)").splitlines()),
                "  the org's now:",
                *(f"    {line}" for line in (item.org or "(none)").splitlines()),
            ]
        if item.use is not None:
            lines.append(
                f"  you decided: {item.use}"
                + (f", under {item.under}" if item.under else "")
                + (f", called {item.named!r}" if item.named else "")
            )
    contracts = any(item.waits_with_teams for item in waiting)
    lines += [
        "",
        "Nothing the session made of its teams and contracts is the org's "
        "until you settle this. Call `settle_teams` once: say in `teams` "
        "for each team above whether the 'org' holds or the 'session'; "
        "unnamed, a team becomes what the session made of it. A team you "
        "take from the session you may put under another team of the org "
        "(`under`) or call otherwise (`title`), so that it fits the org as "
        "it is now."
        + (
            " For each contracts above say in `merge_choices` which holds: "
            "'org', 'session', or 'replacement' with the contracts you write from "
            "both. Unnamed, the session's hold."
            if contracts
            else ""
        ),
        "What you settle becomes the org's together, once the session's "
        "code is accepted and what sessions that submitted earlier made of "
        "the org is in. What the session wrote for a team the org does not "
        "take is kept as that team's record with the nearest team that "
        "holds its folder.",
    ]
    return "\n".join(lines)


def files_not_written(title: str, files: list[str]) -> str:
    """What a Division Lead IC is told when files of accepted work could
    not be written to team memory."""
    return "\n".join(
        [
            f"Accepted work of team {title!r}: these files of team memory "
            "changed while it was merged, so what the session wrote for "
            "them was not written. The session still has its text.",
            *(f"- {file}" for file in files),
            "Nothing is asked of you now: you are shown both texts when "
            "the session next submits work.",
        ]
    )


def teams_settled(
    above: str,
    taken: list[str],
    kept: list[str],
    agreed: list[str],
    held: list[str],
    others: Sequence[tuple[str, str]] = (),
    *,
    names: Names,
) -> str:
    """What a session's IC is told when what waited of its teams is
    settled.

    Args:
        agreed: The directories whose contracts are the session's or what
            the Manager wrote from both.
        held: The directories whose contracts stay the org's.
        others: Where the contracts the session wrote do not hold: each
            directory with the contracts that do.
    """
    follow_up = "".join(
        f"\n\nThe contracts that hold for {directory or 'the repository'} "
        "are not the ones you wrote. Your merged code was written to yours: "
        "check it against these and deliver what must change in this "
        f"session.\n{text or '(none)'}"
        for directory, text in others
    )
    return (
        f"The {names.manager} of {above!r} settled what the org had made something "
        "else of than your session. "
        + (f"The org has your teams now: {', '.join(taken)}. " if taken else "")
        + (
            f"The org's teams stay, and what your session has there is only "
            f"the session's: {', '.join(kept)}. "
            if kept
            else ""
        )
        + (
            f"Contracts settled for: {', '.join(agreed)}; read them with "
            "`read_team_context`. "
            if agreed
            else ""
        )
        + (f"The org's contracts stay for: {', '.join(held)}." if held else "")
    ).strip() + follow_up


def handed_up(sent: SessionTeam, text: str, why: str, *, names: Names) -> str:
    """What the IC of the team above is started with when the team a
    request was sent to said it is not its own."""
    return (
        f"{text}\n\n(The director sent this request of the human's to "
        f"{sent.title!r} (id {sent.id}) first, whose {names.writer} said it is not "
        "that "
        f"team's: {why}\nIt is yours now, in the same session and the same "
        f"checkout. {sent.title!r} is a sub-team of yours here and has done "
        "nothing for it: give it a task with `msg_team` if part of it is "
        "theirs. If it is not your team's either, raise it as `misrouted`.)"
    )


def session_brief(
    team: SessionTeam, title: str, others: list[str], *, names: Names
) -> str:
    """What the IC a session starts with is told after the human's request."""
    beside = (
        f" Other sessions at work on this team's folder: {', '.join(others)}."
        if others
        else " No other session works on the team's folder just now."
    )
    return (
        f"\n\n[director] You are the {names.writer} of {team.title!r} in the session "
        f"{title!r}, which was opened for this request.\n"
        "- Every request is worked on in a session like this one, in a "
        f"checkout and on a branch of its own.{beside} "
        "`other_sessions_of_your_team` in your context names them.\n"
        "- You start from the team's spec, contracts and records. The "
        "team's reports are shared with other sessions: `report` what they "
        "must know.\n"
        f"- A spec you write and sub-teams your {names.manager} approves are this "
        "session's alone until your work is accepted.\n"
        "- The sub-teams you see are not started. One starts when you give "
        "it a task with `msg_team`.\n"
        "- When the request is done, commit and call `submit_review`. The "
        f"Division Lead {names.above_reviewer} (the {names.above_reviewer} one team "
        "up) reviews "
        "it. Accepted work is "
        "merged into the repository's checkout, in the order work was "
        "submitted.\n"
        "- You need not keep up with that checkout while you work. If your "
        "work does not merge with what went in meanwhile, the Division Lead "
        f"{names.reviewer} finishes the merge and you are told which files it changed. "
        "Work comes back to you only if it must change or touches files "
        "outside your folder.\n"
        "- Teams and contracts you changed may become the org's a little "
        f"after your code: the Division Lead {names.manager} settles those that "
        "other accepted work changed too."
    )


def still_undecided(teams: Sequence[SessionTeam]) -> str:
    """Reminds a reviewing IC of the work it was asked to review before and
    has not decided on."""
    names = ", ".join(f"{team.title!r} ({team.id})" for team in teams)
    return (
        f"Still waiting for your decision, as you were asked before: {names}. "
        "Decide on each with review_team; read_team_context has what each "
        "submitted."
    )


def review_request(
    team: SessionTeam,
    review: Review,
    so_far: str = "",
    asked: list[str] | None = None,
    instructions: str = "",
) -> str:
    """What the IC above is told when a team below submits its work.

    Args:
        so_far: What became of the team's work before, in words.
        asked: What the human asked the team directly since its work was
            last accepted.
        instructions: The code review skill, for this IC to apply to the
            diff. Empty when it could not be loaded.
    """
    directory = team.boundary.path or "the repository"
    lines = [
        f"Review request. Team {team.title!r} (id {team.id}) submits its work in "
        f"{directory}"
        + (f" (review round {review.round})." if review.round > 1 else "."),
    ]
    if so_far:
        lines.append(f"Its work so far: {so_far}.")
    if asked:
        lines += [
            "",
            "The human asked this team directly, not through you; work on "
            "that is asked for, though it is not in the task you gave:",
            *(f"- {text}" for text in asked),
        ]
    if review.summary:
        lines += ["", "What it says it delivered:", review.summary]
    if review.description:
        lines += ["", "Its own account of what it changed:", review.description]
    else:
        lines += ["", "It gave no account of what it changed."]
    lines += [
        "",
        "Its changes: run these in " + team.worktree_path,
        f"  git diff --stat {team.base_commit} HEAD -- {directory}",
        f"  git diff {team.base_commit} HEAD -- {directory}",
        "Name a file in place of the folder to read that file alone.",
        "",
    ]
    if instructions.strip():
        lines += [
            "Review the diff by these instructions:",
            instructions.strip(),
            "",
        ]
    lines += [
        "The account is the team's own. Read the diff of "
        "what you rely on, run the repository's checks on it, and compare it "
        "with the contracts you set and with the team's task. Then call "
        f"`review_team` for team {team.id}: `accept`, or `reject` with what "
        "must change. Set `needed_above` only if a team above yours waits on "
        "this work. Accepted work of a team with its own worktree is merged "
        "into your checkout by the runtime.",
    ]
    return "\n".join(lines)


def misrouted_notice(team: SessionTeam, text: str, why: str, *, names: Names) -> str:
    """What the IC above is told when a team says a request the human's
    words brought it directly is not its own."""
    return (
        f"The human asked what follows. The runtime sent it straight to "
        f"team {team.title!r} (id {team.id}), whose {names.writer} says it is not that "
        f"team's: {why or '(no reason given)'}\nIt is yours now: do it, or "
        "give it to the team it belongs to with `msg_team`.\n\n"
        f"The human wrote:\n{text}"
    )


STALL_QUESTION = (
    "The {above_writer} of the team above is about to be told that your team has "
    "shown "
    "no progress for some time: nothing committed in your folder, "
    "nothing submitted, no escalation. In a few sentences, for that {above_writer}: "
    "what are you working on, what has kept you from committing or "
    "submitting, and what do you need to go on? Answer in plain text and "
    "do nothing else."
)


def _span(minutes: int) -> str:
    hours, rest = divmod(minutes, 60)
    if not hours:
        return f"{minutes} min"
    return f"{hours} h {rest} min" if rest else f"{hours} h"


def unstarted_notice(team: SessionTeam, minutes: int, *, names: Names) -> str:
    """What the IC above is told of a team it set up and gave no task."""
    return (
        f"Team {team.title!r} (id {team.id}, "
        f"{team.boundary.path or 'the repository'}) "
        f"was set up {_span(minutes)} ago and has been given no task. Nobody "
        "works in its folder, and nothing there is committed. Give it its "
        f"task with `msg_team`. If it is not needed, write to your {names.manager} to "
        "have it pruned, or propose your teams again without it. If it waits "
        "on purpose, do nothing: you are not told of it again."
    )


def check_in_notice(
    team: SessionTeam,
    minutes: int,
    *,
    in_a_turn: bool,
    head: str | None,
    again: int,
    names: Names,
) -> str:
    """What the IC above is told of a team it has not turned to for the
    time between two check-ins, whether or not the team shows progress.

    Args:
        minutes: How long ago that IC last turned to it or was told of it.
        in_a_turn: Whether the team's IC is in a turn.
        head: The last commit in the team's directory; None when it has none.
        again: After how many minutes it is told of again.
    """
    directory = team.boundary.path or "the repository"
    sent_back = team.review is not None and team.review.state == "changes_requested"
    lines = [
        f"Check-in: team {team.title!r} (id {team.id}) has not submitted, and "
        f"you last turned to it or were told of it {_span(minutes)} ago. This "
        "is sent whether or not it shows progress; nothing has to be wrong."
    ]
    if sent_back and team.review is not None:
        lines.append(
            f"You sent its work back (review round {team.review.round}) and "
            "it has not submitted it again."
        )
    lines += [
        "",
        f"Its {names.writer} is in a turn right now."
        if in_a_turn
        else f"Its {names.writer} is not in a turn right now.",
        f"The last commit in {directory} is {head[:12]}."
        if head
        else f"Nothing is committed in {directory} yet.",
        "",
        "Look for yourself: run these in " + team.worktree_path,
        f"  git log --oneline -5 -- {directory}",
        f"  git status --short -- {directory}",
        "",
        "If it is on course, do nothing. If it is off course or has taken "
        f"too long, write to its {names.writer} with `msg_team`. You are told of it "
        "again "
        f"in {_span(again)}; writing to it, or being told that it shows no "
        "progress, starts that time again.",
    ]
    return "\n".join(lines)


def stall_notice(
    team: SessionTeam,
    minutes: int,
    *,
    in_a_turn: bool,
    account: str | None,
    names: Names,
) -> str:
    """What the IC above is told of a team that shows no progress.

    Args:
        minutes: How long it has shown none.
        in_a_turn: Whether the team's IC is in a turn, and so was not asked.
        account: What the team's IC said when asked; None when it gave no
            answer or was not asked.
    """
    directory = team.boundary.path or "the repository"
    sent_back = team.review is not None and team.review.state == "changes_requested"
    lines = [
        f"Team {team.title!r} (id {team.id}) has shown no progress for "
        f"{_span(minutes)}: no commit in {directory}, nothing submitted, no "
        "escalation."
    ]
    if sent_back and team.review is not None:
        lines.append(
            f"You sent its work back (review round {team.review.round}) and "
            "it has not submitted it again."
        )
    lines.append("")
    if in_a_turn:
        lines.append(
            f"Its {names.writer} is in a turn right now, so it was not asked what "
            "it is doing."
        )
    elif account:
        lines += [f"Asked just now, its {names.writer} says:", account.strip()]
    else:
        lines.append(
            f"Asked just now what it is doing, its {names.writer} gave no answer."
        )
    lines += [
        "",
        "Look for yourself: run these in " + team.worktree_path,
        f"  git log --oneline -5 -- {directory}",
        f"  git status --short -- {directory}",
        "",
        f"Then decide. You can wait; write to its {names.writer} with `msg_team`; or "
        f"write to your {names.manager} with `msg_manager` if the teams should be "
        "made up differently. If it has gone on too long without progress "
        f"and a message did not help, `stop_team` for team {team.id} stops "
        "it. Use that very sparingly: a stopped team does nothing until "
        f"you write to it again or your {names.manager} removes it.",
    ]
    return "\n".join(lines)


def failure_notice(
    team: SessionTeam, name: str, reason: str, *, works: bool, names: Names
) -> str:
    """What the IC above is told of a team whose agent failed.

    Args:
        name: What the agent's member is called.
        works: Whether it is the member that works in the team's folder.
    """
    lines = [
        f"The {name} of team {team.title!r} (id {team.id}) failed and is at "
        f"rest: {reason}",
        "",
    ]
    if works:
        lines.append(
            "Nothing runs it again by itself. A message from you with "
            f"`msg_team` for team {team.id} does. If what failed is not "
            "something a message can mend (its model cannot be reached, it "
            "ran out of budget), raise it with `raise_escalation`."
        )
    else:
        lines.append(
            "It runs again the next time something is sent to it. Nothing is "
            f"asked of you unless that team waits on its {names.manager}; then raise "
            "it with `raise_escalation`."
        )
    return "\n".join(lines)


_EVENT_WORDS: dict[TeamEventKind, str] = {
    "set_up": "set up",
    "accepted": "work accepted",
    "sent_back": "work sent back",
    "reconciled": "work reconciled with what was merged meanwhile",
    "stalled": "showed no progress",
    "failed": "an agent or a step of the director failed",
    "stopped": "stopped by the IC",
    "replaced": "replaced",
    "pruned": "pruned",
    "woken": "Manager woken",
    "proposed": "sub-teams proposed",
    "approved": "sub-teams approved",
    "declined": "sub-teams declined",
    "needed": "needed another team's folder",
    "asked": "asked directly by the human",
    "misrouted": "said a request was not its own",
    "session_opened": "session opened",
    "session_closed": "session closed",
    "not_taken": "not made the org's",
    "moved": "moved here",
}
# What is counted of a team, in the order it is told.
_COUNTED: tuple[TeamEventKind, ...] = ("accepted", "sent_back", "stalled", "stopped")


def _fresh() -> dict[str, Any]:
    """What is counted of a team from the time it is set up."""
    return {kind: 0 for kind in _COUNTED} | {
        "ended": None,
        "tasks": 0,
        "tasks_succeeded": 0,
        "limit_reached": 0,
        "limit_reached_on_the_work_in_hand": False,
    }


def track_records(events: list[TeamEvent]) -> list[dict[str, Any]]:
    """What became of each team the events are of, one entry a team, in the
    order the teams first appear. A team that is gone says how it ended.

    A task is a piece of work of the team that was accepted, or the one it
    has in hand once its IC's escalations on it reached the limit that
    wakes the Manager. `tasks_succeeded` are the accepted ones on which
    that limit was never reached.
    """
    records: dict[str, dict[str, Any]] = {}
    for event in events:
        if not event.team_id:
            continue
        record = records.setdefault(
            event.team_id,
            {"team_id": event.team_id, "team": event.team} | _fresh(),
        )
        if event.kind == "set_up":
            # A new team in the place of one that is gone starts its own.
            record.update(_fresh())
            record["team"] = event.team
        elif event.kind == "woken":
            if not record["limit_reached_on_the_work_in_hand"]:
                record["tasks"] += 1
            record["limit_reached"] += 1
            record["limit_reached_on_the_work_in_hand"] = True
        elif event.kind == "accepted":
            record["accepted"] += 1
            if record["limit_reached_on_the_work_in_hand"]:
                record["limit_reached_on_the_work_in_hand"] = False
            else:
                record["tasks"] += 1
                record["tasks_succeeded"] += 1
        elif event.kind in _COUNTED:
            record[event.kind] += 1
        elif event.kind in ("replaced", "pruned"):
            record["ended"] = event.kind
    return list(records.values())


def record_line(record: dict[str, Any]) -> str:
    """One team's record, in words; empty when nothing is counted of it."""
    parts = [
        f"{record[kind]} {_EVENT_WORDS[kind].removeprefix('work ')}"
        for kind in _COUNTED
        if record.get(kind)
    ]
    return ", ".join(parts)


def success_rate(record: dict[str, Any]) -> str:
    """A team's success rate, in words: how many of its tasks were accepted
    without its escalations reaching the limit. Empty before it has had a
    task."""
    tasks = record.get("tasks", 0)
    if not tasks:
        return ""
    succeeded = record.get("tasks_succeeded", 0)
    return (
        f"success rate {succeeded} of {tasks} task{'s' if tasks != 1 else ''} "
        f"({round(100 * succeeded / tasks)}%)"
    )


def success_rate_meaning(escalation_limit: int, *, names: Names) -> str:
    """What a team's success rate counts, for the Manager that reads it.

    Args:
        escalation_limit: How many escalations of one team on the work it
            has in hand wake the Manager; 0 when that is off.
    """
    if not escalation_limit:
        return "A team's success rate is the share of its tasks that were accepted."
    return (
        "A team's success rate is the share of its tasks that were accepted "
        f"without its {names.writer} escalating {escalation_limit} times on them or "
        f"its work being sent back {escalation_limit} times, the number "
        "that wakes you. A task on which that number was reached counts "
        "against it, also while it is still in hand."
    )


# From how many times two teams needed each other it is worth saying.
NEEDED_WORTH_SAYING = 3
# What is counted as one team needing another, by the event's detail.
NEED_WRITE = "write refused"
NEED_FINDING = "review finding"


def needs_between(events: list[TeamEvent]) -> list[dict[str, Any]]:
    """How often teams that are beside each other needed each other's
    directory, a pair an entry, in the order the pairs first appear.

    A write of one team's agent that was refused because the file is the
    other's counts, and so does a review finding on one team's work that
    names a file of the other. Only teams that are there are counted, and
    each from the time it was set up.
    """
    pairs: dict[frozenset[str], dict[str, Any]] = {}
    for event in events:
        if event.kind in ("set_up", "replaced", "pruned"):
            for key in [key for key in pairs if event.team_id in key]:
                del pairs[key]
        elif event.kind == "needed" and event.team_id and event.other_id:
            pair = pairs.setdefault(
                frozenset((event.team_id, event.other_id)),
                {
                    "team_ids": [event.team_id, event.other_id],
                    "teams": [event.team, event.other],
                    "writes_refused": 0,
                    "review_findings": 0,
                },
            )
            counted = (
                "review_findings" if event.detail == NEED_FINDING else "writes_refused"
            )
            pair[counted] += 1
    return list(pairs.values())


def need_words(pair: dict[str, Any], seen_from: str = "") -> str:
    """One pair of `needs_between`, in words.

    Args:
        seen_from: The team id of the team of the pair the words are said
            about, when they are; the other is then named alone.
    """
    writes, findings = pair["writes_refused"], pair["review_findings"]
    times = writes + findings
    parts = [
        f"{count} {many if count != 1 else single}"
        for count, single, many in (
            (writes, "write refused", "writes refused"),
            (findings, "review finding", "review findings"),
        )
        if count
    ]
    one, other = pair["teams"]
    if seen_from:
        beside = other if pair["team_ids"][0] == seen_from else one
        who = f"it and {beside!r} needed"
    else:
        who = f"{one!r} and {other!r} needed"
    return (
        f"{who} each other's folder {times} time{'s' if times != 1 else ''} "
        f"({', '.join(parts)}): weigh whether the two are one concern"
    )


def event_line(event: TeamEvent) -> str:
    who = f"{event.team!r}: " if event.team else ""
    detail = f" ({event.detail})" if event.detail else ""
    if event.kind == "needed":
        detail = f" ({event.other!r}, {', '.join(event.paths)}: {event.detail})"
    return f"{event.at} {who}{_EVENT_WORDS[event.kind]}{detail}"


def manager_check(
    reason: str,
    teams: list[SessionTeam],
    quiet: dict[str, int],
    blocked: dict[str, int],
    escalations: dict[str, int],
    events: list[TeamEvent] | None = None,
    *,
    after_successes: bool = False,
    escalation_limit: int = 0,
    names: Names,
) -> str:
    """What a Manager is told when the director wakes it to look at its
    teams.

    Args:
        reason: What was counted, as the Manager is to read it.
        quiet: For how many minutes each team has shown no progress, by
            team id, where that is known.
        blocked: For how many minutes each blocked team's IC has waited on
            an escalation it raised, by team id.
        escalations: How many escalations each team's IC has raised on the
            work it has in hand, by team id.
        events: What became of the teams below so far, also of the ones
            that are gone.
        after_successes: The look comes after work that was accepted, not
            after anything that went wrong.
        escalation_limit: How many escalations of one team on the work it
            has in hand wake the Manager, which is what a success rate is
            counted against.
    """
    records = {record["team_id"]: record for record in track_records(events or [])}
    lines = [
        f"A look at your teams that your {names.writer} did not ask for: {reason}",
        "",
        "The teams directly below yours:",
    ]
    for team in teams:
        facts = [team.status if team.agents else "not started: given no task yet"]
        if team.id in blocked:
            facts[0] += (
                f": its {names.writer} has waited {_span(blocked[team.id])} on an "
                "escalation it raised"
            )
        if team.stopped is not None:
            facts.append(f"stopped by your {names.writer} ({team.stopped.reason})")
        if team.review is not None:
            facts.append(
                f"review {team.review.state.replace('_', ' ')}, round "
                f"{team.review.round}"
            )
        else:
            facts.append("nothing submitted yet")
        raised = escalations.get(team.id, 0)
        if raised:
            facts.append(f"{raised} escalated to your {names.writer} on this work")
        waiting = len(team.escalations)
        if waiting:
            facts.append(f"{waiting} unsettled")
        if team.id in quiet and team.id not in blocked:
            facts.append(f"no progress for {_span(quiet[team.id])}")
        record = records.get(team_key(team.id), {})
        so_far = record_line(record)
        if so_far:
            facts.append(f"so far: {so_far}")
        rate = success_rate(record)
        if rate:
            facts.append(rate)
        lines.append(
            f"- {team.title!r} (id {team.id}, "
            f"{team.boundary.path or 'the repository'}): " + "; ".join(facts)
        )
    here = {team.id for team in teams}
    gone = [
        f"- {record['team']!r} (id {record['team_id']}), {record['ended']}"
        + (f": {so_far}" if (so_far := record_line(record)) else "")
        for record in records.values()
        if record["ended"] and record["team_id"] not in here
    ]
    if gone:
        lines += ["", "Teams that were here before and are gone:", *gone]
    if any(record.get("tasks") for record in records.values()):
        lines += ["", success_rate_meaning(escalation_limit, names=names)]
    lines += [
        "",
        (
            "Nothing went wrong: this look comes after work that was accepted. "
            if after_successes
            else ""
        )
        + "Nothing is asked of you, and nothing here is a call to prune. If "
        "the make-up of the teams is what holds the work back, write to an "
        f"{names.writer} with `msg_ic`. `prune_team` is for a team that is grossly "
        "inefficient, not for one that is slow or had one bad task, and a "
        "success rate over one or two tasks says little: wait until a team "
        "has had enough tasks to judge it by. Otherwise end your turn.",
    ]
    return "\n".join(lines)


def proposal_facts(
    sizes: list[tuple[SplitChild, int]],
    together: list[tuple[str, str, int]],
    before: list[str],
    reason: str = "",
    *,
    names: Names,
) -> str:
    """What the director adds, for the Manager, to the sub-teams its IC
    proposes; empty when it knows nothing worth adding.

    Args:
        sizes: Each proposed team with how many files git tracks in its
            directory.
        together: Pairs of proposed directories with how many commits
            changed both, where any did.
        before: What is on record of teams that had these directories.
        reason: Why the IC wants this make-up, when it said.
    """
    lines = []
    if reason:
        lines.append(f"The {names.writer}'s reason for this make-up: {reason}")
    for child, files in sizes:
        why = f" The {names.writer}'s reason: {child.reason}" if child.reason else ""
        lines.append(
            f"- {child.title!r} ({child.path}): {files} file"
            f"{'s' if files != 1 else ''} there now.{why}"
        )
    lines += [
        f"- {path} and {other} were changed in the same commit {count} time"
        f"{'s' if count != 1 else ''}: weigh whether they are one concern."
        for path, other, count in together
    ]
    lines += [f"- {line}" for line in before]
    if not lines:
        return ""
    return "\n".join(["What the runtime knows of them:", *lines])


def changes_requested(
    reviewer: SessionTeam,
    findings: list[ReviewFinding],
    *,
    of_a_session: bool = False,
    names: Names,
) -> str:
    """What a team is told when its work was not accepted.

    Args:
        of_a_session: Whether it is the work a session delivers into the
            repository, which is accepted in the order it was submitted.
    """
    return "\n".join(
        [
            f"The {names.reviewer} of {reviewer.title!r} reviewed your work and "
            "asks for "
            "changes before it is accepted:",
            *_finding_lines(findings),
            "Make them, commit, and submit the work again with `submit_review`.",
            *(
                [
                    "Work goes into the repository in the order it is "
                    "submitted, so yours joins the end of the line again."
                ]
                if of_a_session
                else []
            ),
        ]
    )


def from_director(text: str) -> str:
    return f"{DIRECTOR_PREFIX} {text}"
