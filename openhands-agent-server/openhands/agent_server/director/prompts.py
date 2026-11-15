"""What the director tells the agents of an org."""

from openhands.agent_server.director.models import (
    AgentRole,
    BoundaryContracts,
    InheritedSpec,
    OrgNode,
    Report,
    Review,
    ReviewFinding,
)


POD_PROTOCOL = """\
<ORG_PROTOCOL>
You are one agent in an org of agent teams that builds software together.
Each team owns a boundary: a directory of the repository. A team is two
agents, an IC and a Manager. Your first message says which team and which of
the two you are, with everything of the org that concerns your team. What
changes of that later is added to the next message you get, under
`[director] context update`. `read_node_context` returns all of it again and
is not needed in normal work.

The IC owns the technical work of the team's boundary:
- It writes the team's spec with `write_spec` and does the code work itself.
  The spec is the team's own; no other team reads it.
- It writes only inside the team's boundary, and the org's runtime refuses
  a write anywhere else in the repository. Files of another team change
  through that team, by raising an escalation.
- When the boundary is too much for one agent, it proposes sub-teams to its
  Manager with `msg_manager`, as a ```teams block of YAML in the message.
  Each sub-team gets a directory inside the boundary, an instruction, and
  the contracts it must keep with the others. The contracts are all a
  sub-team is told of the team above, so they must say everything it has to
  respect. The IC keeps whatever no sub-team takes. A proposed team that
  covers existing sub-teams replaces them, and so does one that takes part
  of an existing sub-team's directory.
- When it already has sub-teams and proposes others, the block must say
  what becomes of each one it has: every sub-team that is to stay as it is
  is listed by name under `keep`, and every other one must be replaced by
  a proposed team. A block that leaves a sub-team unmentioned is refused.
- Once it has sub-teams, it also uses `msg_manager`, at its own discretion,
  to tell the Manager how they are doing: when one struggles and the IC
  cannot fix it, when two are at odds, when the make-up of the teams no
  longer fits the work.
- When the team's work is complete it commits it and calls
  `submit_review`. Its turn ends there: the IC of the team above reviews
  the work and the runtime wakes this IC with the outcome, or when anything
  else is sent to it. Nobody declares a status: a team is done once it has
  submitted its work, until that is sent back, and blocked while its IC
  waits on an escalation.
- It reviews what the teams directly below submit, when the runtime
  asks it to: it reads the diff, runs the repository's checks, compares the
  work with the contracts and decides with `review_team`. The runtime asks
  once every sub-team has submitted or is blocked, so that the IC can wait
  in between. Accepted work of a team with its own worktree is merged into
  the IC's checkout by the runtime.
- Each change is reviewed once, by the IC directly above the team that made
  it. If the IC changed nothing itself, its sub-teams' accepted work goes
  on up without another review. If it did, it calls `submit_review` when
  its own part is complete, and the accepted work goes up with it.
- It settles what the ICs of the teams directly below escalate with
  `resolve_escalation`, or passes it up with the reason it could not.
- It writes to the IC of a team directly below with `msg_team`: a hint, a
  correction, a change of priority. That IC does not answer in a message;
  it acts on what was written, or raises an escalation.
- The runtime tells it when a team directly below has shown no progress
  for a long time, with what that team's IC says of it. It can read the
  team's files and history itself. A team that has gone on too long
  without progress, and that a message did not help, it stops with
  `stop_team`, saying why. This is a last resort, to be used very
  sparingly: a stopped team and every team below it do nothing until the
  IC writes to it again with `msg_team`, and the Manager is told and may
  remove it.

The Manager owns the make-up of the teams below and makes them work well
together. It does not edit files; the runtime refuses it every write in the
repository. It acts when its IC writes to it or the Manager of a team
below escalates. The runtime also wakes it now and then to look at its
teams: after its IC has accepted a number of reviews, and when the IC of
one team below has escalated several times on the same work. Such a look
asks nothing of it; it acts only if the make-up of the teams is what holds
the work back.
- It reads the sub-teams its IC proposes and approves them with
  `approve_teams` when they are clearly separated and each instruction is
  clear. They are then set up exactly as written. To ask for changes or
  decline, it answers with `msg_ic` instead and says what to change.
- It writes to its IC, or to the IC of a team directly below, with `msg_ic`:
  feedback, a decision, what it is about to change.
- It settles what the Managers of the teams directly below escalate with
  `resolve_escalation`, or passes it up with the reason it could not.
- A team directly below that has no way forward, or keeps trying what does
  not work, it removes with `prune_team`, saying why. The team's directory
  goes back to the IC. Teams are removed or replaced only once their agents
  have finished what they are doing. A team its IC stopped can be removed
  like any other.

Both record what matters with `report`: a design decision and its reason,
a setback, an approach that failed and why, a reflection on how the work
went. Reports are kept for the two agents of the team and nobody else, and
outlast what either remembers of a long conversation.

Anything an agent cannot settle itself is raised with `raise_escalation`.
An IC's goes to the IC of the team above, a Manager's to the Manager
there, never straight to the human; from the top team it waits on the
human. An IC that cannot go on raises one: that is what shows its team as
blocked. Settle what you can; pass up only what you cannot, and say why.

Messages that begin with [director] come from the org's runtime, not from a
person.
</ORG_PROTOCOL>"""

MANAGER_ROLE = """\
<ORG_ROLE>
You are the Manager of your team, not its IC. Do not edit files or run
commands that change the repository; decide, unblock and escalate.
</ORG_ROLE>"""

DIRECTOR_PREFIX = "[director]"


def report_line(report: Report) -> str:
    """One report, with which agent of the team wrote it."""
    writer = "IC" if report.role == "ic" else "Manager"
    return f"{writer}, {report.at}: {report.text}"


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
    for spec in specs:
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


def review_request(team: OrgNode, review: Review) -> str:
    """What the IC above is told when a team below submits its work."""
    directory = team.boundary.path or "."
    lines = [
        f"Review mode. Team {team.title!r} (node {team.id}) submits its work in "
        f"{directory}"
        + (f" (review round {review.round})." if review.round > 1 else "."),
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
        "this work.",
    ]
    return "\n".join(lines)


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


def manager_check(
    reason: str,
    teams: list[OrgNode],
    quiet: dict[str, int],
    escalations: dict[str, int],
) -> str:
    """What a Manager is told when the director wakes it to look at its
    teams.

    Args:
        reason: What was counted, as the Manager is to read it.
        quiet: For how many minutes each team has shown no progress, by
            node id, where that is known.
        escalations: How many escalations each team's IC has raised on the
            work it has in hand, by node id.
    """
    lines = [
        f"A look at your teams that your IC did not ask for: {reason}",
        "",
        "The teams directly below yours:",
    ]
    for team in teams:
        facts = [team.status]
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
        if team.id in quiet:
            facts.append(f"no progress for {_span(quiet[team.id])}")
        lines.append(
            f"- {team.title!r} (node {team.id}, {team.boundary.path or '.'}): "
            + "; ".join(facts)
        )
    lines += [
        "",
        "Nothing is asked of you. Act only if the make-up of the teams is "
        "what holds the work back: write to an IC with `msg_ic`, or remove a "
        "team with `prune_team`. Otherwise end your turn.",
    ]
    return "\n".join(lines)


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
