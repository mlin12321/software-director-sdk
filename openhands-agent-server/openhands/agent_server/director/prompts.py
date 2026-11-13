"""What the director tells the agents of an org."""

from openhands.agent_server.director.models import AgentRole, InheritedSpec, OrgNode


POD_PROTOCOL = """\
<ORG_PROTOCOL>
You are one agent in an org of agent teams that builds software together.
Each team owns a boundary: a directory of the repository. A team is two
agents, an IC and a Manager. Your first message says which team and which of
the two you are; `read_node_context` repeats it at any time.

The IC owns the technical work of the team's boundary:
- It writes the team's spec with `write_spec` and does the code work itself.
- It writes only inside the team's boundary. Files of another team change
  through that team, by raising an escalation.
- When the boundary is too much for one agent, it proposes to hand parts of
  it to sub-teams with `propose_boundary_split`. Each sub-team gets a
  directory inside the boundary, a task, and the contracts it must keep with
  the others. The IC keeps whatever no sub-team takes.
- It reports with `set_status` when the team's work is done or blocked.

The Manager owns the team's make-up and unblocks it. It does not edit files.
- It decides the IC's split proposals with `decide_split`: approve when the
  sub-teams are clearly separated and each task is clear, otherwise ask for
  a refinement and say what to change.
- It settles what the teams below escalate with `resolve_escalation`, or
  passes it up with the reason it could not.

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
    rules: list[str],
) -> str:
    """The first message of an agent: who it is and what it inherits."""
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
        owner = "your own spec" if not spec.readonly else "read-only, from above"
        lines += ["", f"Spec of {spec.path or 'the repository'} ({owner}):", spec.spec]
    if rules:
        lines += ["", "Rules that apply in this boundary:"]
        lines += [f"- {rule}" for rule in rules]
    lines.append("")
    lines.append(
        "Start on the task now."
        if role == "ic"
        else "Wait for your IC's proposals and for escalations from below."
    )
    return "\n".join(lines)


def from_director(text: str) -> str:
    return f"{DIRECTOR_PREFIX} {text}"
