"""What the director tells the agents of an org."""

import json
from datetime import datetime
from typing import Any

from openhands.agent_server.director.models import (
    AgentRole,
    BoundaryContracts,
    Escalation,
    FormerTeam,
    InheritedSpec,
    OrgNode,
    Report,
    Review,
    ReviewFinding,
    SplitChild,
    TeamEvent,
    TeamEventKind,
)
from openhands.agent_server.director.split import is_inside


POD_PROTOCOL = """\
<ORG_PROTOCOL>
You are one agent in an org of agent teams that builds software together.
Each team owns a boundary: a directory of the repository. A team is two
agents, an IC and a Manager. Your first message says which team and which of
the two you are, with everything of the org that concerns your team. What
changes of that later is added to the next message you get, under
`[director] context update`. `read_node_context` returns all of it again and
is not needed in normal work.

Messages that begin with [director] come from the org's runtime, not from a
person. What a rarer event asks of you (a review, a team that shows no
progress, teams that were replaced) is said in the message that brings it.

The IC owns the technical work of the team's boundary:
- It writes the team's spec with `write_spec` and does the code work itself.
  The spec is the team's own; no other team reads it.
- It writes only inside the team's boundary, and the org's runtime refuses
  a write anywhere else in the repository. What the human asks of the
  team directly it does inside its boundary, though the task from the team
  above did not ask for it. Files of another team change
  through that team, by raising an escalation: say in its `team_for` what
  is needed of that team, and the runtime takes it straight to that team's
  IC when it is sure which team it is.
- When the team's work is complete it commits it and calls
  `submit_review`. Its turn ends there: the IC of the team above reviews
  the work and the runtime wakes this IC with the outcome, or when anything
  else is sent to it. Nobody declares a status: a team is done once it has
  submitted its work, until that is sent back, and blocked while its IC
  waits on an escalation.
- When the boundary holds concerns that are better kept apart, it proposes
  sub-teams to its Manager with `msg_manager`, as a ```teams block of YAML
  in the message. Each sub-team gets a directory inside the boundary, an
  instruction that says what the team is for, and the contracts it must
  keep with the others. The contracts are all a sub-team is told of the
  team above, so they must say everything it has to respect. The IC keeps
  whatever no sub-team takes. A team, and the block, may carry a `reason`;
  it is optional and is for the Manager.
- It divides by concern first: each team owns one thing that can be
  understood, changed and reviewed by itself, behind a contract short
  enough to state in a few lines. The aim is the work done well and soon,
  with code that stays easy to work in. Teams that work at the same time
  follow from a clean division; they are not a reason for one. It does
  not split what shares files or needs constant agreement, or a part
  smaller than writing its task and reviewing it would cost: every team
  costs a briefing, a review and the messages between them.
- Proposing sub-teams starts no work. An approved team is set up and
  waits: it starts when the IC gives it its task with `msg_team`. That IC
  does not answer in a message; it acts on what was written, or raises an
  escalation.
- It reviews what the teams directly below submit with `review_team`, when
  the runtime asks it to, and settles what their ICs escalate with
  `resolve_escalation`, or passes it up with the reason it could not.

Both agents record what matters with `report`: a design decision and its
reason, a setback, an approach that failed and why, a reflection on how the
work went. Reports are kept for the two agents of the team and nobody else,
and outlast what either remembers of a long conversation.

Anything an agent cannot settle itself is raised with `raise_escalation`.
An IC's goes to the IC of the team above, a Manager's to the Manager
there, never straight to the human; from the top team it waits on the
human. An IC that cannot go on raises one: that is what shows its team as
blocked. Settle what you can; pass up only what you cannot, and say why.
</ORG_PROTOCOL>"""

# What an IC is told once it has sub-teams, with the notice that they are
# set up: none of it concerns a team that never splits.
WITH_SUB_TEAMS = (
    "- Each change is reviewed once, by the IC directly above the team that "
    "made it. If you change nothing yourself, your sub-teams' accepted work "
    "goes on up without another review. If you do, call `submit_review` "
    "when your own part is complete, and the accepted work goes up with it.",
    "- Tell your Manager with `msg_manager`, at your own discretion, how "
    "your teams are doing: when one struggles and you cannot fix it, when "
    "two are at odds, when the make-up of the teams no longer fits the work.",
    "- To change the teams, propose again. The block must say what becomes "
    "of each team you have: list under `keep`, by name, every one that is "
    "to stay as it is; every other one must be replaced by a proposed team "
    "that covers its directory or takes part of it. A team is cleared by "
    "proposing a different team for the same directory. When two teams keep "
    "being held up by each other, or a team has nothing left to do, propose "
    "teams that merge or replace them.",
    "- The runtime keeps what became of your teams and of what you "
    "proposed, also why it was declined; a review request says how a "
    "team's earlier work went and `read_node_context` returns the record. "
    "Do not propose again a division that failed or was declined unless "
    "something has changed.",
)

MANAGER_ROLE = """\
<ORG_ROLE>
You are the Manager of your team, not its IC: what is said above of the IC
is not yours to do. Do not edit files or run commands that change the
repository; the runtime refuses you every write there.

The Manager owns the make-up of the teams below and makes them work well
together. It acts when its IC writes to it or the Manager of a team below
escalates. The runtime also wakes it now and then to look at its teams,
and says then what it counted. Its concern is that the org as a whole
works efficiently, not that every team is busy.
- It reads the sub-teams its IC proposes and approves them with
  `approve_teams` when each is a concern of its own, they are clearly
  separated and each instruction is clear. The runtime adds what it knows
  of the proposed directories and of the teams that had them. They are
  then set up exactly as written, and start when its IC gives each its
  task. Otherwise it declines them with `decline_teams` and says why, and
  what to change if another division would do; the reason is kept in the
  record. What an IC proposed waits on its Manager until one of the two is
  called; a message with `msg_ic` decides nothing.
- It writes to its IC, or to the IC of a team directly below, with `msg_ic`:
  feedback, a decision, what it is about to change. After it approves or
  prunes teams, its IC is told to pass on what removed teams left and to
  rewrite the team's spec; the Manager writes to it when the change needs
  explaining.
- `read_node_context` returns what became of every team below so far, also
  of teams that are gone, and what its IC proposed before. The Manager
  weighs what it approves by it.
- It settles what the Managers of the teams directly below escalate with
  `resolve_escalation`, or passes it up with the reason it could not.
- A team directly below that is grossly inefficient, with no way forward
  or trying again and again what does not work, it may remove with
  `prune_team`, saying why. This is rare and never a precaution: a team
  that is slow, or had one bad task, is first written to with `msg_ic`,
  and a team is judged only once it has had enough tasks to judge it by.
  The team's directory goes back to the IC.
</ORG_ROLE>"""

HAND_OVER_ROLE = """\
<ORG_ROLE>
You are your team's IC in hand-over mode: this is a side conversation
started for one hand-over. It does not carry on the team's work. Do not
edit files or run commands that change the repository. Of the director's
tools only `read_node_context`, `hand_over` and `write_spec` work here.
</ORG_ROLE>"""

DIRECTOR_PREFIX = "[director]"


def report_line(report: Report) -> str:
    """One report, with which agent of the team wrote it."""
    if report.handed_over:
        return (
            "Handed over by the IC of the team above, from the teams that "
            f"were here before yours, {report.at}: {report.text}"
        )
    writer = "IC" if report.role == "ic" else "Manager"
    return f"{writer}, {report.at}: {report.text}"


def _now_with(team: FormerTeam, children: list[SplitChild]) -> str:
    """Who has a replaced team's directory under the new boundaries."""
    where = team.path or "."
    inside = [c for c in children if is_inside(c.path, team.path)]
    over = [c for c in children if c.path == team.path or is_inside(team.path, c.path)]
    if over:
        [child] = over[:1]
        return f"Its directory is now in the boundary of {child.title} ({child.path})."
    if inside:
        names = ", ".join(f"{child.title} ({child.path})" for child in inside)
        return (
            f"Its directory is now divided: {names}. What of {where} is in "
            "none of those is yours again."
        )
    return f"No new team has its directory: {where} is yours again."


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
            f"Team {team.title!r} (node {team.id}, {team.path or '.'}), "
            f"{team.ended}. It was for: {team.purpose or 'nothing that was kept'}",
            "Its reports, as one:",
            team.compacted_report or "It recorded nothing.",
        ]
        if children is not None:
            lines.insert(-2, _now_with(team, children))
        if team.contracts.strip():
            lines += ["The contracts between its sub-teams:", team.contracts.strip()]
        spec = (specs or {}).get(team.path, "").strip()
        if spec:
            lines += [
                f"The spec it wrote (old; it stays in {team.path or '.'}):",
                spec,
            ]
    return lines


def _approved(children: list[SplitChild]) -> str:
    teams = ", ".join(f"{child.title} ({child.path})" for child in children)
    return (
        "Your team was split: your Manager approved your sub-teams and they "
        f"now own their directories: {teams}. None of them has started: a "
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
        "- What remains in your boundary outside those directories is yours as before.",
        "- What you committed in those directories is where each team "
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
            "  They passed to those teams with the directories. You may finish "
            "them: you can write in a team's directory until you give the team "
            "its task. The team then finds them as uncommitted work of its "
            "own, so say in its task what they are for. It checks them, "
            "commits them and submits them, and you review that. You cannot "
            "commit or submit them yourself."
        )
    else:
        lines.append(
            "- You left nothing uncommitted there. Until you give a team its "
            "task you can still write in its directory, but what you write "
            "there is the team's: it commits it and submits it, and you "
            "review that."
        )
    lines.append(
        "- Once a team has its task, a write of yours in its directory is "
        "refused. Tell the team what you need with msg_team."
    )
    return "\n".join(lines)


def put_aside_notice(team: OrgNode, files: list[str], ref: str) -> str:
    """What an IC is told when changes it made in a started sub-team's
    directory were taken out of its checkout."""
    return (
        f"Your checkout held uncommitted changes in {team.boundary.path}, "
        f"which is the directory of your sub-team {team.title!r}: "
        f"{_named(files)}. They were in the way of the work you accepted "
        f"and were taken out; they are kept at {ref}. If they are still "
        "needed, tell the team with msg_team."
    )


def left_for_team(files: list[str]) -> str:
    """What a team is told, with its task, of the changes the IC above left
    uncommitted in its directory."""
    return (
        "The IC of the team above changed these files in your directory "
        f"before your team started and did not commit them: {_named(files)}. "
        "They are yours now. Check them, change what is wrong, and commit "
        "them with your work before you submit it."
    )


def _contracts_now(contracts_replaced: bool) -> str:
    if contracts_replaced:
        return (
            "- The contracts between your sub-teams are now the ones of this "
            "proposal alone."
        )
    return (
        "- The contracts between your sub-teams read as before. If they name "
        "teams that are gone, restate them: the `contracts` of a ```teams "
        "block that replaces teams take the place of the old text."
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
    "- Give each new team what it should know of this with `hand_over`: "
    "notes in your own words or quoted, and, for a team that took over "
    "sub-teams, the contracts those must keep. Where one team's directory "
    "is now divided among several, give each the part that concerns its "
    "directory. It starts nothing and can be done again later; "
    "`read_node_context` returns what the removed teams left "
    "(`former_teams`)."
)

_SPEC_STAYS = (
    "- A spec a removed team wrote stays in its directory. The team that has "
    "the directory now is told it is old."
)

_REWRITE_SPEC = (
    "- Rewrite your own spec with `write_spec` to say how your directory is "
    "divided now."
)


def approved_notice(
    children: list[SplitChild],
    former: list[FormerTeam],
    *,
    contracts_replaced: bool,
    unfinished: dict[str, list[str]],
) -> str:
    """What an IC is told once the sub-teams it proposed are set up.

    Args:
        unfinished: By directory of a new team, the files the IC has
            changed there and not committed.
        former: The teams they replaced. What those left is passed on in
            hand-over mode and is not given here.
        contracts_replaced: Whether the proposal's contracts took the place
            of the ones the node had.
    """
    lines = [_approved(children), "", _split_effects(children, unfinished)]
    todo = []
    if former:
        gone = ", ".join(f"{team.title} ({team.path or '.'})" for team in former)
        lines += [
            "",
            f"The teams they replaced are gone: {gone}. What those had "
            f"committed is in your checkout. {_ASIDE}",
        ]
        todo += [_contracts_now(contracts_replaced)]
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
            *WITH_SUB_TEAMS,
        ]
    )


def replaced_job(
    teams: list[OrgNode],
    children: list[SplitChild],
    former: list[FormerTeam],
    specs: dict[str, str],
    *,
    contracts_replaced: bool,
) -> str:
    """What a hand-over conversation is to do after teams were replaced.

    Args:
        teams: The sub-teams that were set up.
        specs: The spec each replaced team wrote, by its directory.
    """
    lines = [
        "What changed: your Manager approved the sub-teams you proposed, and "
        "they now own their directories. None of them has started; each "
        "starts when your main conversation gives it its task.",
        *(
            f"- {team.title} (node {team.id}, {team.boundary.path}): "
            f"{team.spec_summary or 'no purpose given'}"
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
        _contracts_now(contracts_replaced),
        _REWRITE_SPEC,
    ]
    return "\n".join(lines)


def pruned_job(
    title: str, where: str, reason: str, former: list[FormerTeam], specs: dict[str, str]
) -> str:
    """What a hand-over conversation is to do after a team was pruned."""
    lines = [
        f"What changed: your Manager pruned the team {title!r} ({where}): "
        f"{reason} {where} is your team's again.",
        "",
        "What it left is below. It is yours to use or to pass on.",
        *left_behind(former, None, specs),
        "",
        "What is yours to do here:",
        _REWRITE_SPEC + " The spec the pruned team wrote stays in its "
        "directory and is old.",
        "- If a team you still have should know of what the pruned team "
        "left, give it to that team with `hand_over`.",
    ]
    return "\n".join(lines)


HAND_OVER_FAILED = (
    "Your hand-over mode could not be started: its side conversation did "
    "not start, also when it was tried again. Nothing was handed over."
)


def handed_over_notice(given: list[str], spec: bool) -> str:
    """What an IC is told in its main conversation once its hand-over
    conversation has ended."""
    to = (
        f"It gave notes or contracts to: {', '.join(given)}."
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
        "`read_node_context` returns what the removed teams left "
        "(`former_teams`) if more of it should be given with `hand_over`."
    )


def pruned_notice(
    title: str, where: str, reason: str, saved: str | None, former: list[FormerTeam]
) -> str:
    """What the IC above is told once its Manager's prune is carried out.

    Args:
        former: What the pruned team left, if it left a record. It is
            passed on in hand-over mode and is not given here.
    """
    pruned = (
        f"Your Manager pruned the team {title!r} ({where}): {reason} Its "
        "uncommitted changes were discarded"
        + (f" (kept in git as {saved})" if saved else "")
        + f"; {where} is yours again."
    )
    if former:
        return f"{pruned}\n\n{_ASIDE}"
    return "\n".join([pruned, "", "What is now yours to do:", _REWRITE_SPEC])


def _of_the_team(
    specs: list[InheritedSpec],
    contracts: list[BoundaryContracts],
    reports: list[Report],
    context: str,
) -> list[str]:
    """The specs, contracts, reports and context a briefing gives."""
    lines: list[str] = []
    for spec in specs:
        if spec.old:
            by = f" ({spec.left_by})" if spec.left_by else ""
            lines += [
                "",
                f"A spec left by a team before you{by}, in "
                f"{spec.path or 'the repository'}. It is old: it says how that "
                "team worked. Take from it what still holds and write your "
                "team's own with `write_spec`.",
                spec.spec,
            ]
        else:
            lines += ["", "Your team's spec:", spec.spec]
    for agreed in contracts:
        heading = (
            "Contracts between your sub-teams:"
            if agreed.own
            else "Contracts your team must keep with the teams beside it "
            "(set by the team above; read-only):"
        )
        lines += ["", heading, agreed.contracts]
    if reports:
        lines += ["", "What your team has reported so far:"]
        lines += [f"- {report_line(report)}" for report in reports]
    if context:
        lines += ["", context]
    return lines


def hand_over_brief(
    node: OrgNode,
    job: str,
    specs: list[InheritedSpec],
    contracts: list[BoundaryContracts],
    reports: list[Report],
    context: str = "",
) -> str:
    """The first message of a hand-over conversation: a side conversation
    of a team's IC that knows nothing but what this gives it.

    Args:
        job: What changed, what the removed teams left and what to do.
        context: The rest of the team's context, as the agent is to read it.
    """
    lines = [
        f"{DIRECTOR_PREFIX} You are the IC of the team {node.title!r} (node "
        f"{node.id}), in hand-over mode.",
        "Your team was changed, and this side conversation was started for "
        "one job: to pass on what the removed teams left. It has a context of "
        "its own. Your main conversation carries on with the team's work and "
        "sees nothing of this one, and you see nothing of it: what follows "
        "is everything you are given.",
        f"Boundary of your team: {node.boundary.path or 'the whole repository'}",
        "",
        job,
        "",
        "What your team works to, so that you can judge what to pass on:",
        *_of_the_team(specs, contracts, reports, context),
        "",
        "Do the hand-over now. You may read files; change none. When it is "
        "done, end with a line or two on what you gave to whom.",
    ]
    return "\n".join(lines)


def brief(
    node: OrgNode,
    role: AgentRole,
    task: str,
    specs: list[InheritedSpec],
    contracts: list[BoundaryContracts],
    reports: list[Report],
    context: str = "",
) -> str:
    """The first message of an agent: who it is and what it works to.

    Args:
        context: The rest of the team's context, as the agent is to read it.
    """
    boundary = node.boundary.path or "the whole repository"
    lines = [
        f"{DIRECTOR_PREFIX} You are the {'IC' if role == 'ic' else 'Manager'} of "
        f"the team {node.title!r} (node {node.id}).",
        f"Boundary: {boundary}",
        f"Writable: {', '.join(node.boundary.writable) or 'nothing'}",
    ]
    if node.boundary.readonly:
        lines.append(f"Read-only: {', '.join(node.boundary.readonly)}")
    lines += ["", "Task of the team:", task]
    lines += _of_the_team(specs, contracts, reports, context)
    lines.append("")
    lines.append(
        "Start on the task now."
        if role == "ic"
        else "Wait until your IC writes to you."
    )
    return "\n".join(lines)


def _finding_lines(findings: list[ReviewFinding]) -> list[str]:
    lines = []
    for finding in findings:
        where = finding.path + (f":{finding.line}" if finding.line else "")
        lines.append(f"- {where + ': ' if where else ''}{finding.issue}")
        if finding.evidence:
            lines.append(f"  Evidence: {finding.evidence}")
    return lines


def session_work(team: OrgNode, title: str) -> str:
    """What comes before the review request for the work of the session
    titled ``title``."""
    return (
        f"The human opened the session {title!r} at "
        f"{team.title!r}, beside that team's own work, and asked it for "
        "something directly. It worked in a checkout of its own "
        f"({team.worktree_path}, branch {team.branch}). You gave it no task: "
        "judge its work by what the human asked and by the team's spec. "
        "Accepting it merges it into your checkout, where it meets whatever "
        "was delivered there meanwhile; if the two do not merge, it goes "
        "back to the session to resolve.\n\n"
    )


def session_brief(team: OrgNode, title: str, above: OrgNode, others: list[str]) -> str:
    """What the IC a session starts with is told after the human's request."""
    beside = (
        f" Beside you work: {', '.join(others)}."
        if others
        else " Nobody else works on the team's directory just now."
    )
    return (
        f"\n\n[director] You are the IC of {team.title!r} in the session "
        f"{title!r}, which the human opened for this request "
        "beside the team's other work. You have a checkout and a branch of "
        "your own, started from the checkout of "
        f"{above.title!r}.{beside} Each works on the same directory in its "
        "own checkout; `other_sessions_of_your_team` in your context names "
        "them, and the team's reports are shared, so record with `report` "
        "what the others must know. The team's spec is one for all of you: "
        "rewrite it only where your request changes it. The sub-teams you "
        "see are the team's, "
        "there once more for this session and not started: one starts when "
        "you give it a task with `msg_team`. Sub-teams cannot be proposed, "
        "replaced or pruned inside a session. When the request is done, "
        f"commit and call `submit_review`: the IC of {above.title!r} reviews "
        "it and it is merged into that team's checkout. If it does not "
        "merge with what was delivered there meanwhile, it comes back to "
        "you: merge that team's branch into yours, resolve and submit again."
    )


def session_closed(title: str, team: str, kept: str | None) -> str:
    """What the IC that reviewed a session's work is told when the human
    closes the session."""
    return (
        f"The human closed the session {title!r} at {team!r}. Its agents and "
        "its checkout are gone. What you accepted of its work is in your "
        "checkout. "
        + (
            f"Commits of it that were never merged are kept at {kept}; "
            "nothing is owed for them."
            if kept
            else "It left no commits that were not merged."
        )
    )


def review_request(
    team: OrgNode, review: Review, so_far: str = "", asked: list[str] | None = None
) -> str:
    """What the IC above is told when a team below submits its work.

    Args:
        so_far: What became of the team's work before, in words.
        asked: What the human asked the team directly since its work was
            last accepted.
    """
    directory = team.boundary.path or "."
    lines = [
        f"Review mode. Team {team.title!r} (node {team.id}) submits its work in "
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
        "Name a file in place of the directory to read that file alone.",
        "",
    ]
    if not review.automated_ran:
        lines.append("The team's own review of the changes did not run.")
    elif review.automated:
        lines.append(
            "Asked to review the changes by the code review instructions, "
            "the team's IC found, for you to verify:"
        )
        lines += _finding_lines(review.automated)
    else:
        lines.append(
            "Asked to review the changes by the code review instructions, "
            "the team's IC found nothing."
        )
    lines += [
        "",
        "The account and that review are the team's own. Read the diff of "
        "what you rely on, run the repository's checks on it, and compare it "
        "with the contracts you set and with the team's task. Then call "
        f"`review_team` for node {team.id}: `accept`, or `reject` with what "
        "must change. Set `needed_above` only if a team above yours waits on "
        "this work. Accepted work of a team with its own worktree is merged "
        "into your checkout by the runtime.",
    ]
    return "\n".join(lines)


def asked_by_team(raiser: OrgNode, escalation: Escalation, team_for: str) -> str:
    """What a team's IC is told when the IC of a team that is not above it
    asks something of it directly."""
    return (
        f"The IC of team {raiser.title!r} (node {raiser.id}), which is not "
        f"above yours, asks your team directly {escalation.title!r} "
        f"(escalation {escalation.id}):\n{team_for}"
        + (f"\n{escalation.detail}" if escalation.detail else "")
        + "\nThe runtime brought it straight to you because it concerns what "
        "your team holds. It is a request, not a task: your own task and "
        "your contracts come first, and that team waits on your answer. "
        f"Call `resolve_escalation` for node {raiser.id}: `settle` with what "
        "you did or why you will not, or `pass_up` with the reason if it is "
        "not yours to settle or needs a contract changed. Passed up, it "
        "goes up the chain of teams above the one that asked."
    )


def requests_between(events: list[TeamEvent]) -> list[dict[str, Any]]:
    """What teams below a team asked of each other directly, an entry for
    each asking and asked team, in the order they first appear."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    asked_at: dict[tuple[str, str], list[str]] = {}
    for event in events:
        key = (event.team_id, event.other_id)
        if event.kind == "asked_team":
            row = rows.setdefault(
                key,
                {
                    "from_node": event.team_id,
                    "from": event.team,
                    "to_node": event.other_id,
                    "to": event.other,
                    "asked": 0,
                    "settled": 0,
                    "passed_up": 0,
                    "not_answered": 0,
                    "open": 0,
                    "longest_wait_minutes": 0,
                },
            )
            row["asked"] += 1
            row["open"] += 1
            asked_at.setdefault(key, []).append(event.at)
        elif event.kind == "answered" and key in rows:
            row = rows[key]
            counted = {SETTLED: "settled", PASSED_UP: "passed_up"}.get(
                event.detail, "not_answered"
            )
            row[counted] += 1
            row["open"] = max(0, row["open"] - 1)
            if asked_at.get(key):
                waited = datetime.fromisoformat(event.at) - datetime.fromisoformat(
                    asked_at[key].pop(0)
                )
                row["longest_wait_minutes"] = max(
                    row["longest_wait_minutes"], int(waited.total_seconds() // 60)
                )
    return list(rows.values())


def many_requests_notice(row: dict[str, Any], table: list[dict[str, Any]]) -> str:
    """What the IC above two teams is told when one has asked the other
    directly as often as the org's setbacks number."""
    return (
        f"Team {row['from']!r} has now asked team {row['to']!r} directly "
        f"{row['asked']} times. The runtime takes such requests straight to "
        "the team that holds what is asked for, so they did not pass you. "
        "Weigh whether the contract between the two says enough, or whether "
        "the two are one concern. Nothing else is asked of you.\n"
        "What your teams asked of each other directly so far:\n"
        + json.dumps(table, indent=2)
    )


DIRECT_NOTE = (
    "\n\n(The human sent this to your team directly. The ICs of the teams "
    "above are told of it and have not acted on it. Do what is your team's "
    "own. Raise an escalation for what needs another team's files or "
    "changes a contract. If none of it is your team's, raise an escalation "
    "of the category `misrouted` and it goes to the team above.)"
)


def misrouted_notice(team: OrgNode, text: str, why: str) -> str:
    """What the IC above is told when a team says a request the human's
    words brought it directly is not its own."""
    return (
        f"The human asked what follows. The runtime sent it straight to "
        f"team {team.title!r} (node {team.id}), whose IC says it is not that "
        f"team's: {why or '(no reason given)'}\nIt is yours now: do it, or "
        "give it to the team it belongs to with `msg_team`.\n\n"
        f"The human wrote:\n{text}"
    )


STALL_QUESTION = (
    "The IC of the team above is about to be told that your team has shown "
    "no progress for some time: nothing committed in your directory, "
    "nothing submitted, no escalation. In a few sentences, for that IC: "
    "what are you working on, what has kept you from committing or "
    "submitting, and what do you need to go on? Answer in plain text and "
    "do nothing else."
)


def _span(minutes: int) -> str:
    hours, rest = divmod(minutes, 60)
    if not hours:
        return f"{minutes} min"
    return f"{hours} h {rest} min" if rest else f"{hours} h"


def unstarted_notice(team: OrgNode, minutes: int) -> str:
    """What the IC above is told of a team it set up and gave no task."""
    return (
        f"Team {team.title!r} (node {team.id}, {team.boundary.path or '.'}) "
        f"was set up {_span(minutes)} ago and has been given no task. Nobody "
        "works in its directory, and nothing there is committed. Give it its "
        "task with `msg_team`. If it is not needed, write to your Manager to "
        "have it pruned, or propose your teams again without it. If it waits "
        "on purpose, do nothing: you are not told of it again."
    )


def stall_notice(
    team: OrgNode, minutes: int, *, in_a_turn: bool, account: str | None
) -> str:
    """What the IC above is told of a team that shows no progress.

    Args:
        minutes: How long it has shown none.
        in_a_turn: Whether the team's IC is in a turn, and so was not asked.
        account: What the team's IC said when asked; None when it gave no
            answer or was not asked.
    """
    directory = team.boundary.path or "."
    sent_back = team.review is not None and team.review.state == "changes_requested"
    lines = [
        f"Team {team.title!r} (node {team.id}) has shown no progress for "
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
            "Its IC is in a turn right now, so it was not asked what it is doing."
        )
    elif account:
        lines += ["Asked just now, its IC says:", account.strip()]
    else:
        lines.append("Asked just now what it is doing, its IC gave no answer.")
    lines += [
        "",
        "Look for yourself: run these in " + team.worktree_path,
        f"  git log --oneline -5 -- {directory}",
        f"  git status --short -- {directory}",
        "",
        "Then decide. You can wait; write to its IC with `msg_team`; or "
        "write to your Manager with `msg_manager` if the teams should be "
        "made up differently. If it has gone on too long without progress "
        f"and a message did not help, `stop_team` for node {team.id} stops "
        "it. Use that very sparingly: a stopped team does nothing until "
        "you write to it again or your Manager removes it.",
    ]
    return "\n".join(lines)


_EVENT_WORDS: dict[TeamEventKind, str] = {
    "set_up": "set up",
    "accepted": "work accepted",
    "sent_back": "work sent back",
    "stalled": "showed no progress",
    "stopped": "stopped by the IC",
    "replaced": "replaced",
    "pruned": "pruned",
    "woken": "Manager woken",
    "proposed": "sub-teams proposed",
    "approved": "sub-teams approved",
    "declined": "sub-teams declined",
    "needed": "needed another team's directory",
    "asked": "asked directly by the human",
    "misrouted": "said a request was not its own",
    "asked_team": "asked another team directly",
    "answered": "had its request to another team",
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
            {"node_id": event.team_id, "team": event.team} | _fresh(),
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


def success_rate_meaning(escalation_limit: int) -> str:
    """What a team's success rate counts, for the Manager that reads it.

    Args:
        escalation_limit: How many escalations of one team on the work it
            has in hand wake the Manager; 0 when that is off.
    """
    if not escalation_limit:
        return "A team's success rate is the share of its tasks that were accepted."
    return (
        "A team's success rate is the share of its tasks that were accepted "
        f"without its IC escalating {escalation_limit} times on them or "
        f"its work being sent back {escalation_limit} times, the number "
        "that wakes you. A task on which that number was reached counts "
        "against it, also while it is still in hand."
    )


# From how many times two teams needed each other it is worth saying.
NEEDED_WORTH_SAYING = 3
# What is counted as one team needing another, by the event's detail.
NEED_WRITE = "write refused"
NEED_FINDING = "review finding"
NEED_ASKED = "asked directly"
# How a request one team made of another directly ended.
SETTLED = "settled"
PASSED_UP = "passed up"
UNANSWERED = "not answered"


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
                    "node_ids": [event.team_id, event.other_id],
                    "teams": [event.team, event.other],
                    "writes_refused": 0,
                    "review_findings": 0,
                    "requests": 0,
                },
            )
            counted = {NEED_FINDING: "review_findings", NEED_ASKED: "requests"}.get(
                event.detail, "writes_refused"
            )
            pair[counted] += 1
    return list(pairs.values())


def need_words(pair: dict[str, Any], seen_from: str = "") -> str:
    """One pair of `needs_between`, in words.

    Args:
        seen_from: The node id of the team of the pair the words are said
            about, when they are; the other is then named alone.
    """
    writes, findings = pair["writes_refused"], pair["review_findings"]
    requests = pair["requests"]
    times = writes + findings + requests
    parts = [
        f"{count} {many if count != 1 else single}"
        for count, single, many in (
            (writes, "write refused", "writes refused"),
            (findings, "review finding", "review findings"),
            (requests, "request made directly", "requests made directly"),
        )
        if count
    ]
    one, other = pair["teams"]
    if seen_from:
        beside = other if pair["node_ids"][0] == seen_from else one
        who = f"it and {beside!r} needed"
    else:
        who = f"{one!r} and {other!r} needed"
    return (
        f"{who} each other's directory {times} time{'s' if times != 1 else ''} "
        f"({', '.join(parts)}): weigh whether the two are one concern"
    )


def event_line(event: TeamEvent) -> str:
    who = f"{event.team!r}: " if event.team else ""
    detail = f" ({event.detail})" if event.detail else ""
    if event.kind == "needed":
        detail = f" ({event.other!r}, {', '.join(event.paths)}: {event.detail})"
    if event.kind in ("asked_team", "answered"):
        detail = f" ({event.other!r}: {event.detail})"
    return f"{event.at} {who}{_EVENT_WORDS[event.kind]}{detail}"


def manager_check(
    reason: str,
    teams: list[OrgNode],
    quiet: dict[str, int],
    blocked: dict[str, int],
    escalations: dict[str, int],
    events: list[TeamEvent] | None = None,
    *,
    after_successes: bool = False,
    escalation_limit: int = 0,
) -> str:
    """What a Manager is told when the director wakes it to look at its
    teams.

    Args:
        reason: What was counted, as the Manager is to read it.
        quiet: For how many minutes each team has shown no progress, by
            node id, where that is known.
        blocked: For how many minutes each blocked team's IC has waited on
            an escalation it raised, by node id.
        escalations: How many escalations each team's IC has raised on the
            work it has in hand, by node id.
        events: What became of the teams below so far, also of the ones
            that are gone.
        after_successes: The look comes after work that was accepted, not
            after anything that went wrong.
        escalation_limit: How many escalations of one team on the work it
            has in hand wake the Manager, which is what a success rate is
            counted against.
    """
    records = {record["node_id"]: record for record in track_records(events or [])}
    lines = [
        f"A look at your teams that your IC did not ask for: {reason}",
        "",
        "The teams directly below yours:",
    ]
    for team in teams:
        facts = [team.status if team.agents else "not started: given no task yet"]
        if team.id in blocked:
            facts[0] += (
                f": its IC has waited {_span(blocked[team.id])} on an "
                "escalation it raised"
            )
        if team.stopped is not None:
            facts.append(f"stopped by your IC ({team.stopped.reason})")
        if team.review is not None:
            facts.append(
                f"review {team.review.state.replace('_', ' ')}, round "
                f"{team.review.round}"
            )
        else:
            facts.append("nothing submitted yet")
        raised = escalations.get(team.id, 0)
        if raised:
            facts.append(f"{raised} escalated to your IC on this work")
        waiting = len(team.escalations)
        if waiting:
            facts.append(f"{waiting} unsettled")
        if team.id in quiet and team.id not in blocked:
            facts.append(f"no progress for {_span(quiet[team.id])}")
        record = records.get(team.id, {})
        so_far = record_line(record)
        if so_far:
            facts.append(f"so far: {so_far}")
        rate = success_rate(record)
        if rate:
            facts.append(rate)
        lines.append(
            f"- {team.title!r} (node {team.id}, {team.boundary.path or '.'}): "
            + "; ".join(facts)
        )
    here = {team.id for team in teams}
    gone = [
        f"- {record['team']!r} (node {record['node_id']}), {record['ended']}"
        + (f": {so_far}" if (so_far := record_line(record)) else "")
        for record in records.values()
        if record["ended"] and record["node_id"] not in here
    ]
    if gone:
        lines += ["", "Teams that were here before and are gone:", *gone]
    if any(record.get("tasks") for record in records.values()):
        lines += ["", success_rate_meaning(escalation_limit)]
    lines += [
        "",
        (
            "Nothing went wrong: this look comes after work that was accepted. "
            if after_successes
            else ""
        )
        + "Nothing is asked of you, and nothing here is a call to prune. If "
        "the make-up of the teams is what holds the work back, write to an "
        "IC with `msg_ic`. `prune_team` is for a team that is grossly "
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
        lines.append(f"The IC's reason for this make-up: {reason}")
    for child, files in sizes:
        why = f" The IC's reason: {child.reason}" if child.reason else ""
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


def changes_requested(reviewer: OrgNode, findings: list[ReviewFinding]) -> str:
    """What a team is told when its work was not accepted."""
    return "\n".join(
        [
            f"The IC of {reviewer.title!r} reviewed your work and asks for "
            "changes before it is accepted:",
            *_finding_lines(findings),
            "Make them, commit, and submit the work again with `submit_review`.",
        ]
    )


def from_director(text: str) -> str:
    return f"{DIRECTOR_PREFIX} {text}"
