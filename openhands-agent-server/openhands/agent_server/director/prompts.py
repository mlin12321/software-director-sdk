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
the two you are; `read_node_context` repeats it at any time.

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
- Once it has sub-teams, it also uses `msg_manager`, at its own discretion,
  to tell the Manager how they are doing: when one struggles and the IC
  cannot fix it, when two are at odds, when the make-up of the teams no
  longer fits the work.
- When the team's work is complete it commits it and calls
  `submit_review`, then stops: the IC of the team above reviews it and the
  runtime wakes this IC with the outcome. Nobody declares a status: a team
  is done once its work is accepted, and blocked while its IC waits on an
  escalation.
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

The Manager owns the make-up of the teams below and makes them work well
together. It does not edit files; the runtime refuses it every write in the
repository. It acts when its IC writes to it or the Manager of a team
below escalates, and not otherwise.
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
  have finished what they are doing.

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
) -> str:
    """The first message of an agent: who it is and what it works to."""
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
