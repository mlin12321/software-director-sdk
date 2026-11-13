"""What the director tells the agents of an org."""

from openhands.agent_server.director.models import (
    AgentRole,
    Autopsy,
    BoundaryContracts,
    InheritedSpec,
    OrgNode,
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
- When the boundary is too much for one agent, it proposes to hand parts of
  it to sub-teams with `propose_boundary_split`. Each sub-team gets a
  directory inside the boundary, a task, and the contracts it must keep with
  the others. The contracts are all a sub-team is told of the team above, so
  they must say everything it has to respect. The IC keeps whatever no
  sub-team takes.
- It commits the team's work and reports with `set_status` when it is done
  or blocked. Done work is reviewed by the IC of the team above.
- It reviews what the teams directly below report as done, when the runtime
  asks it to: it reads the diff, runs the repository's checks, compares the
  work with the contracts and decides with `review_team`. Accepted work of a
  team with its own worktree is merged into the IC's checkout by the
  runtime. The IC tells the team above only when a team there waits on that
  work, through `needed_above`; otherwise the work is simply part of its own
  team's, which it reports when all of it is done.
- It records what whoever works in the boundary later must know with
  `record_rule`.

The Manager owns the team's make-up and unblocks it. It does not edit files;
the runtime refuses it every write in the repository.
- It decides the IC's split proposals with `decide_split`: approve when the
  sub-teams are clearly separated and each task is clear, otherwise ask for
  a refinement and say what to change.
- It settles what the teams below escalate with `resolve_escalation`, or
  passes it up with the reason it could not.
- It watches the teams directly below. A team that is blocked with no way
  forward, or keeps trying what does not work, it removes with `prune_team`,
  saying what the team tried and why it failed. With a new task, a fresh
  team takes over the same directory.

Anything a team cannot settle itself is raised with `raise_escalation`. It
goes to the Manager of the team above, never straight to the human. Settle
what you can; pass up only what you cannot, and say why.

Messages that begin with [director] come from the org's runtime, not from a
person.
</ORG_PROTOCOL>"""

MANAGER_ROLE = """\
<ORG_ROLE>
You are the Manager of your team, not its IC. Do not edit files or run
commands that change the repository; decide, unblock and escalate.
</ORG_ROLE>"""

DIRECTOR_PREFIX = "[director]"


def brief(
    node: OrgNode,
    role: AgentRole,
    task: str,
    specs: list[InheritedSpec],
    contracts: list[BoundaryContracts],
    rules: list[str],
    autopsies: list[Autopsy],
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
    if rules:
        lines += ["", "Rules that apply in this boundary:"]
        lines += [f"- {rule}" for rule in rules]
    if autopsies:
        lines += ["", "Approaches that already failed in this boundary:"]
        lines += [
            f"- {autopsy.approach or 'An earlier attempt'}: {autopsy.why_failed}"
            for autopsy in autopsies
        ]
    lines.append("")
    lines.append(
        "Start on the task now."
        if role == "ic"
        else "Wait for your IC's proposals and for escalations from below."
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
    """What the IC above is told when a team below reports its work done."""
    directory = team.boundary.path or "."
    lines = [
        f"Review mode. Team {team.title!r} (node {team.id}) reports its work in "
        f"{directory} done"
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
            "Make them, commit, and report done again with `set_status`.",
        ]
    )


def from_director(text: str) -> str:
    return f"{DIRECTOR_PREFIX} {text}"
