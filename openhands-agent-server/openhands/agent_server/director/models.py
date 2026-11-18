"""Wire models for the director API."""

from typing import Literal

from pydantic import BaseModel, Field

from openhands.agent_server.models import StartConversationRequest


# A team has an IC and a Manager. The IC of a team leads it, is a report of
# the team above, and does the code work in its boundary that no sub-team
# has taken on.
AgentRole = Literal["ic", "manager"]
AGENT_ROLES: tuple[AgentRole, ...] = ("ic", "manager")
ROLE_NAMES: dict[AgentRole, str] = {"ic": "IC", "manager": "Manager"}

NodeStatus = Literal["idle", "working", "done", "blocked"]

IsolationMode = Literal["worktree", "shared"]

EscalationKind = Literal["critical_issue", "design_decision"]
# How the team a message went to was found.
ChosenBy = Literal["path", "named", "model", "top"]

SplitStatus = Literal["proposed", "approved"]

# 'pending': the IC of the team above has the team's work to review.
ReviewState = Literal["pending", "changes_requested", "accepted"]

# 'sandbox': an agent's shell can write only its team's directory, and its
# file tools are checked. 'checks': only its file tools are checked.
BoundaryEnforcement = Literal["sandbox", "checks"]
# How one team is held to its boundary. 'merge': its agent runs its own tools
# and no sandbox holds it, so what it commits outside its directory is
# refused only when its work is merged into the team above.
TeamEnforcement = Literal["sandbox", "checks", "merge"]
# 'acp': an external agent, such as Claude Code or Codex, that runs its own
# tools and reaches the director's through an MCP server.
AgentKind = Literal["openhands", "acp"]
# Why an org's agents are held. 'shutdown': the server was stopped while the
# org worked. 'crash': the server ended without stopping the org first.
PauseReason = Literal["human", "shutdown", "crash"]
Side = Literal["mine", "theirs"]
ConflictKind = Literal["field", "team", "file"]
SyncResult = Literal["up_to_date", "fast_forward", "merged", "conflicts", "pushed"]
# 'run_again': an agent that was running runs again. 'start': a team that
# waited to start is started. 'continue': a step the director had begun is
# finished. 'message': messages are delivered. 'dropped': run state that no
# longer fits the org was put aside. 'cannot_resume': a team that cannot be
# taken up as it is.
RecoveryKind = Literal[
    "run_again", "start", "continue", "message", "dropped", "cannot_resume"
]


class Boundary(BaseModel):
    """A codeowner boundary. All paths and globs are repo-relative."""

    path: str = Field(
        description="Directory the boundary is rooted at; '' is the repository root."
    )
    writable: list[str] = Field(
        default_factory=list, description="Globs the team's agents may write."
    )
    readonly: list[str] = Field(
        default_factory=list, description="Globs the team's agents may only read."
    )


class HandOverMode(BaseModel):
    """An IC in hand-over mode: beside its main conversation it has one the
    director started to pass on what removed teams left. That conversation
    has a context of its own and does none of the team's work."""

    conversation_id: str
    status: str


class AgentRef(BaseModel):
    """One agent of a team at its place in a session."""

    role: AgentRole
    hand_over: HandOverMode | None = Field(
        default=None,
        description=(
            "Set on an IC that is, or last was, in hand-over mode: the side "
            "conversation of its latest hand-over and how that stands."
        ),
    )
    conversation_id: str | None = Field(
        default=None,
        description="None until a conversation has been started for this agent.",
    )
    status: str
    reason: str | None = Field(
        default=None,
        description=(
            "Why the agent is `blocked` or `lost`, in words for the human; "
            "None when it is neither or the server could not tell."
        ),
    )
    kind: AgentKind = "openhands"
    acp_server: str | None = Field(
        default=None,
        description=(
            "Which external agent an `acp` agent is, such as `claude-code` or "
            "`codex`; None when the server could not tell."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "The model the agent runs on, as it is configured; None for an "
            "`acp` agent left on its own default, and until the agent started."
        ),
    )


class EscalationHandoff(BaseModel):
    """A team that tried to settle an escalation, could not, and passed it up."""

    node_id: str
    reason: str = Field(min_length=1, description="Why this team could not settle it.")
    at: str = Field(description="ISO 8601 timestamp.")


class Escalation(BaseModel):
    """Something an agent of a team cannot settle itself.

    It goes to the agent of the same role in the team above (an IC's to the
    IC there, a Manager's to the Manager), which settles it or passes it up
    in turn. It waits on the human once every team above the one that raised
    it in its session has passed it up, which for the team the session was
    opened at is at once.
    """

    id: str = Field(description="Unique within the team that raised it.")
    kind: EscalationKind
    raised_by: AgentRole = Field(
        default="ic",
        description=(
            "Which agent of the team raised it. The agents of that role in "
            "the teams above are the ones that settle it or pass it up."
        ),
    )
    title: str
    detail: str = ""
    raised_at: str = Field(description="ISO 8601 timestamp.")
    held_by: str | None = Field(
        description=(
            "The team whose agents are working on it now: the parent of the "
            "last team that passed it up. None when it waits on the human."
        )
    )
    handoffs: list[EscalationHandoff] = Field(
        default_factory=list,
        description=(
            "The teams that passed it up, nearest to the team that raised it "
            "first. Always the unbroken chain from that team's parent."
        ),
    )


class SplitChild(BaseModel):
    """A sub-team an IC proposes to hand part of its boundary to."""

    title: str = Field(min_length=1)
    path: str = Field(
        description=(
            "Directory the sub-team owns, repo-relative and strictly inside "
            "the proposing team's boundary."
        )
    )
    writable: list[str] = Field(
        default_factory=list,
        description="Globs under `path` the sub-team may write. Empty: all of it.",
    )
    readonly: list[str] = Field(default_factory=list)
    isolation: IsolationMode = "shared"
    task: str = Field(min_length=1, description="What the sub-team is to deliver.")
    reason: str = Field(
        default="",
        description=(
            "Why the proposer wants this team: what makes it a concern of "
            "its own, or anything else its Manager should weigh. Optional."
        ),
    )
    agent_profile: str | None = Field(
        default=None,
        description=(
            "Name of the agent profile the sub-team's agents run on, from the "
            "ones `read_team_context` lists. Leave it out for the same agent "
            "as your own."
        ),
    )


class SplitProposal(BaseModel):
    """The sub-teams an IC proposed in a message to its Manager, as they
    were read from the message and checked."""

    children: list[SplitChild] = Field(min_length=1)
    contracts: str = Field(
        default="",
        description="The interfaces the sub-teams must keep between them.",
    )
    reason: str = Field(
        default="",
        description="Why the proposer wants this make-up of teams. Optional.",
    )
    status: SplitStatus = "proposed"
    replaces: list[str] = Field(
        default_factory=list,
        description=(
            "The sub-teams the team already has that the proposed ones take "
            "the place of: ones a proposed team covers, and ones a proposed "
            "team takes part of."
        ),
    )
    keeps: list[str] = Field(
        default_factory=list,
        description=(
            "The sub-teams the team already has that the IC said are to "
            "stay as they are."
        ),
    )
    proposed_at: str = Field(description="ISO 8601 timestamp.")


class OldSpec(BaseModel):
    """A spec a team before left in a directory a team now owns."""

    path: str = Field(description="The directory of the spec; '' at the root.")
    team: str | None = Field(
        default=None, description="The title of the team that left it, when known."
    )


class Stopped(BaseModel):
    """That the IC of the team above stopped a team, and why."""

    at: str = Field(description="ISO 8601 timestamp.")
    reason: str
    running: list[str] = Field(
        default_factory=list,
        description=(
            "The conversations of the team and of the teams below it that "
            "were in a turn when it was stopped."
        ),
    )


class ReviewFinding(BaseModel):
    """One thing a review found wrong in a team's work."""

    path: str = Field(default="", description="The file, relative to the repository.")
    line: int | None = None
    issue: str = Field(min_length=1, description="What is wrong.")
    evidence: str = Field(
        default="", description="What shows it: the code, a failing check's output."
    )


class Review(BaseModel):
    """The review of the work a team submitted.

    The IC of the team above reviews it. Before that, the team's own IC is
    asked to describe the changes and to review them, and both answers are
    handed to that IC.
    """

    state: ReviewState
    seq: int = Field(
        default=0,
        description="Counts up with every piece of work submitted in the org.",
    )
    round: int = Field(
        default=1,
        description="1 for the first time this work is reviewed, one more "
        "each time it comes back after changes were requested.",
    )
    summary: str = Field(default="", description="What the team said it delivered.")
    description: str = Field(
        default="",
        description="The team's own account of what it changed, written as "
        "a pull request description; '' when it gave none.",
    )
    automated_ran: bool = Field(
        default=False,
        description="Whether the team's IC, asked to review the changes by "
        "the code review instructions, gave an answer.",
    )
    automated: list[ReviewFinding] = Field(
        default_factory=list, description="What it found."
    )
    findings: list[ReviewFinding] = Field(
        default_factory=list,
        description="What the reviewing IC wants changed, when it asked.",
    )
    passed_up: bool = Field(
        default=False,
        description="Whether the work went up without a review of its own: "
        "all of it was accepted work of the team's sub-teams.",
    )
    at: str = Field(description="ISO 8601 timestamp of the last change.")


class OrgNodeInput(BaseModel):
    """What a caller supplies for a team of the org tree; the store derives
    the rest.

    It is what an agent that works on the team is briefed from. Nobody works
    at one: the work is at a session's place (:class:`SessionTeamInput`).
    """

    id: str
    parent_id: str | None = Field(description="None for the root team.")
    title: str
    boundary: Boundary
    isolation: IsolationMode = Field(
        description=(
            "'worktree': the team has its own git worktree. "
            "'shared': it works in its parent's directory."
        )
    )
    headline: str = Field(
        default="",
        description=(
            "One line on what the team is delivering, written by a model from "
            "the team's task and its latest report."
        ),
    )
    spec_summary: str = ""
    old_specs: list[OldSpec] = Field(
        default_factory=list,
        description=(
            "Specs that teams before this one left in its directory. They "
            "are old until the team's IC writes its own spec."
        ),
    )
    agent_profile: str | None = Field(
        default=None,
        description=(
            "The agent profile the team's agents were started from; None "
            "when they were derived from the team above."
        ),
    )
    since: str = Field(
        default="",
        description=(
            "When the team was set up. Two teams with one id are the same "
            "team only when this is the same: an id comes from the "
            "directory, so a team that replaces another has its id."
        ),
    )


class OrgNode(OrgNodeInput):
    """A team of the org tree, including the fields the store derives."""

    child_ids: list[str] = Field(default_factory=list)


class OrgNodeDetail(BaseModel):
    """A team of the org tree with the teams around it."""

    node: OrgNode
    ancestors: list[OrgNode] = Field(description="Root first, excluding the team.")
    children: list[OrgNode]


class ThreadMessage(BaseModel):
    """One message between a Manager and an IC."""

    at: str = Field(description="ISO 8601 timestamp.")
    sender: AgentRole = Field(description="Which of the two wrote it.")
    text: str = Field(min_length=1)


class Thread(BaseModel):
    """Everything one Manager and one IC have said to each other, or
    everything the IC of the team above has written to an IC."""

    manager_node_id: str = Field(
        description=(
            "The team the Manager belongs to; when ``ic_to_ic``, the team "
            "of the IC that writes."
        )
    )
    ic_node_id: str = Field(
        description=(
            "The team the IC belongs to: the Manager's own, or one directly below it."
        )
    )
    ic_to_ic: bool = Field(
        default=False,
        description=(
            "Whether these are the messages of the IC of the team above to "
            "this IC, which go one way, and not a Manager's."
        ),
    )
    messages: list[ThreadMessage] = Field(default_factory=list)


# What is kept of a team for the team above it. `woken` is the Manager of
# the team above being woken by a count the org set; it names the team whose
# escalations reached the limit, and no team when the count is not of one
# team. `proposed` and `approved` are of the team's own proposals of
# sub-teams and name no team.
TeamEventKind = Literal[
    "set_up",
    "accepted",
    "sent_back",
    "reconciled",
    "stalled",
    "failed",
    "stopped",
    "replaced",
    "pruned",
    "woken",
    "proposed",
    "approved",
    "declined",
    "needed",
    "asked",
    "misrouted",
    "session_opened",
    "session_closed",
    "not_taken",
    "moved",
]


class TeamEvent(BaseModel):
    """Something that became of a team directly below a team, kept in team
    memory so that its IC and its Manager can weigh what they do next by
    what happened before, also of teams that are gone."""

    at: str
    kind: TeamEventKind
    team_id: str = ""
    team: str = ""
    session: str = Field(
        default="",
        description=(
            "The title of the session that team works in; '' when the event "
            "names no team."
        ),
    )
    detail: str = ""
    paths: list[str] = Field(
        default_factory=list,
        description=(
            "The directories of the sub-teams a proposal was for, on "
            "`proposed`, `approved` and `declined`; on `needed`, the file "
            "in the other team's directory."
        ),
    )
    other_id: str = Field(
        default="",
        description=(
            "On `needed`, the team beside this one whose directory it needed."
        ),
    )
    other: str = Field(default="", description="The title of that team.")
    id: str = Field(
        default="",
        description="Names the event, so that it is added to a record once.",
    )
    unaccepted: bool = Field(
        default=False,
        description=(
            "True while it is only its session's: the session's work is not "
            "accepted yet."
        ),
    )


class Report(BaseModel):
    """Something an agent recorded for itself and the other agent of its
    team: a design decision, a setback, a reflection."""

    at: str = Field(description="ISO 8601 timestamp.")
    role: AgentRole = Field(description="Which agent of the team wrote it.")
    text: str = Field(min_length=1)
    session: str = Field(
        default="",
        description=(
            "The title of the session the agent that wrote it works in; '' "
            "for a report a model rewrote or the IC above handed over."
        ),
    )
    compacted: bool = Field(
        default=False,
        description=(
            "True for the one report a model wrote in place of the earlier "
            "ones, once they had grown long."
        ),
    )
    handed_over: bool = Field(
        default=False,
        description=(
            "True for what the IC of the team above passed on of the teams "
            "that were there before this one. The team's own agents did not "
            "write it."
        ),
    )
    id: str = Field(
        default="",
        description=(
            "Names the report, so that it is added to a team's record once "
            "however often that is tried. '' for one a model rewrote."
        ),
    )


class InheritedSpec(BaseModel):
    """The spec of a boundary."""

    path: str
    spec: str
    readonly: bool = Field(
        description=(
            "False for the spec of the team's own boundary, which its IC writes."
        )
    )
    old: bool = Field(
        default=False,
        description=(
            "True for a spec a team before this one left. It describes how "
            "that team worked and is to be rewritten."
        ),
    )
    left_by: str | None = Field(
        default=None,
        description="The title of the team that left an old spec, when known.",
    )


class FormerThread(BaseModel):
    """Messages a removed team took part in, as they were kept."""

    file: str = Field(description="What names them: who wrote to whom.")
    messages: list[ThreadMessage]


# How a team ended. ``not_taken``: a session set it up or changed it, and
# when the session's work was accepted the org did not make it its own.
FormerEnd = Literal["replaced", "pruned", "not_taken"]


class FormerTeam(BaseModel):
    """What a team that was replaced or pruned left, kept apart from the
    teams that come after it."""

    id: str
    title: str
    above: str = Field(
        default="", description="The team id of the team above it when it went."
    )
    path: str = Field(description="The directory the team owned.")
    purpose: str = Field(description="What the team was for.")
    ended: FormerEnd
    at: str = Field(description="When it was removed; the name of the record.")
    review: ReviewState | None = Field(
        default=None, description="How the review of its work stood."
    )
    reports: list[Report] = Field(
        default_factory=list, description="What its two agents recorded, in full."
    )
    compacted_report: str = Field(
        default="",
        description=(
            "Its reports as one report, rewritten shorter by a model when "
            "there were several. '' when it recorded nothing."
        ),
    )
    contracts: str = Field(
        default="", description="The contracts between its own sub-teams."
    )
    spec: str = Field(
        default="",
        description=(
            "The spec a session wrote for it that the org did not take. '' "
            "for a team whose spec is in its directory."
        ),
    )
    session: str = Field(
        default="",
        description="The title of the session it was a team of; '' when none.",
    )
    since: str = Field(
        default="", description="When the team was set up, which tells it apart."
    )
    threads: list[FormerThread] = Field(default_factory=list)


class BoundaryContracts(BaseModel):
    """What the sub-teams of one boundary must keep between them."""

    path: str = Field(description="The boundary that was split.")
    contracts: str
    own: bool = Field(
        description=(
            "True for the contracts between the team's own sub-teams. False "
            "for the ones the team above set, which the team must keep with "
            "the teams beside it."
        )
    )


class FormerTeamSummary(BaseModel):
    """A team that was removed from below a team, as a client shows it."""

    id: str
    title: str
    path: str
    purpose: str
    ended: FormerEnd
    at: str
    compacted_report: str = ""
    contracts: str = ""
    spec: str = ""
    session: str = ""


class NodeMemory(BaseModel):
    """Everything in the team memory that concerns one team."""

    specs: list[InheritedSpec] = Field(
        description="The spec of the team's own boundary, when it has one."
    )
    contracts: list[BoundaryContracts] = Field(default_factory=list)
    reports: list[Report] = Field(
        default_factory=list,
        description="What the team's own two agents recorded, oldest first.",
    )
    former_teams: list[FormerTeamSummary] = Field(
        default_factory=list,
        description="The teams removed from directly below it, oldest first.",
    )
    history: list[TeamEvent] = Field(
        default_factory=list,
        description="What became of the teams directly below it, oldest first.",
    )


class HistoryEvent(TeamEvent):
    """One event of an org's history, with the team whose record holds it."""

    above_id: str = Field(description="The team directly above the event's team.")
    above: str = Field(description="The title of that team.")


class OrgHistory(BaseModel):
    """What became of the teams of an org, newest first."""

    events: list[HistoryEvent]
    earlier: bool = Field(description="Whether there are events before these.")


class BoundaryOwner(BaseModel):
    """A direct report, as the owner of a directory."""

    node_id: str
    title: str
    path: str = Field(description="The directory it owns, relative to the repository.")
    isolation: IsolationMode
    branch: str


class OwnedFile(BaseModel):
    path: str = Field(description="Relative to the repository.")
    owner: str | None = Field(
        description=(
            "The direct report whose boundary the file is in; None for a "
            "file the team kept for itself."
        )
    )


class Collision(BaseModel):
    """A file that the committed work of two teams changed in ways that do
    not merge."""

    path: str
    node_ids: list[str] = Field(
        description="The teams whose checkouts collide, the team itself included."
    )


ChangeKind = Literal["added", "modified", "deleted"]


class ChangedFile(BaseModel):
    """One file a team changed."""

    path: str = Field(description="Relative to the top of the repository.")
    kind: ChangeKind
    added: int | None = Field(description="Lines added; None when not text.")
    removed: int | None = Field(description="Lines removed; None when not text.")
    committed: bool = Field(
        description="False when the file has changes that are not committed."
    )


class TeamChanges(BaseModel):
    """What a team changed in its directory since its base commit."""

    node_id: str
    base: str = Field(
        description="The commit compared with; '' when the team has none yet."
    )
    files: list[ChangedFile] = Field(description="Sorted by path.")
    truncated: bool = Field(description="There were more files than are listed.")


class FileSides(BaseModel):
    """A file at a team's base commit and in its checkout now."""

    path: str
    before: str | None = Field(description="None when the file was not there.")
    after: str | None = Field(description="None when the file is not there.")
    unreadable: bool = Field(
        description="A side is not text or is too large, and is left out."
    )


class EditFileRequest(BaseModel):
    """The whole new content of one file."""

    content: str


class FileEdit(BaseModel):
    """What came of a change made by hand."""

    path: str
    commit: str | None = Field(
        description="The commit that holds it; None when nothing changed."
    )


class NodeOwnership(BaseModel):
    """How a team's boundary is divided among its direct reports."""

    node_id: str
    owners: list[BoundaryOwner]
    files: list[OwnedFile] = Field(
        description=(
            "The files of the team's boundary in its own checkout that git "
            "does not ignore, sorted by path. For a team of the org, in the "
            "repository's checkout."
        )
    )
    truncated: bool = Field(description="There were more files than are listed.")
    collisions: list[Collision] = Field(
        description=(
            "Found by merging the tips of the checkouts of the team and its "
            "direct reports in memory. Uncommitted changes are not compared. "
            "Always empty for a team of the org, whose work is in sessions."
        )
    )
    collisions_checked: bool = Field(
        description=(
            "False when the installed git is too old to look for them, and "
            "for a team of the org."
        )
    )


class MemorySync(BaseModel):
    """What a pull of the org did to the team memory."""

    remote: str
    head: str | None = Field(
        description="The local team memory's commit afterwards; None when empty."
    )
    changed: bool = Field(description="The local team memory took in remote changes.")
    kept_local: list[str] = Field(
        default_factory=list,
        description=(
            "Specs and contracts both sides had changed. The local text was "
            "kept; the remote text is in the team memory's history."
        ),
    )


class DirectorSettings(BaseModel):
    """Limits on the teams an org may still add, and how it starts again.

    The limits are checked when a team splits. Teams that exist are never
    removed for exceeding a limit that was lowered.
    """

    max_teams: int = Field(
        default=32,
        ge=1,
        description=(
            "The most teams a team may be together with every team below "
            "it. Every team has an IC and a Manager; the Manager is "
            "inactive while the team has no sub-teams."
        ),
    )
    max_depth: int = Field(
        default=8,
        ge=1,
        description="The most levels of teams; the root team is level 1.",
    )

    allowed_profiles: list[str] = Field(
        default_factory=list,
        description=(
            "Names of the agent profiles a team may give a sub-team in place "
            "of its own agent. Empty: every team runs on the agent of the "
            "team above."
        ),
    )
    resume_on_start: bool = Field(
        default=False,
        description=(
            "Whether an org that was stopped in the middle of its work takes "
            "it up again as soon as the server is back, without being asked."
        ),
    )
    report_compaction_chars: int = Field(
        default=24_000,
        ge=1_000,
        description=(
            "How many characters of reports one agent may have before a "
            "model rewrites them into one shorter report."
        ),
    )
    stall_check_minutes: int = Field(
        default=120,
        ge=0,
        description=(
            "How many minutes a team may show no progress, with nothing "
            "submitted or with its work sent back, before the IC of the "
            "team above is told. 0 turns the check off."
        ),
    )
    manager_check_accepted_reviews: int = Field(
        default=5,
        ge=0,
        description=(
            "After how many reviews its IC accepted a Manager is woken to "
            "look at its teams, without a message from its IC. 0 turns the "
            "check off."
        ),
    )
    manager_check_ic_setbacks: int = Field(
        default=3,
        ge=0,
        description=(
            "After how many escalations the IC of one team directly below "
            "raised on the work it has in hand, and after how many times "
            "that work was sent back, a Manager is woken to look at its "
            "teams. 0 turns the check off."
        ),
    )
    route_requests: bool = Field(
        default=True,
        description=(
            "Whether a message sent with `auto` may go straight to the "
            "team the director chooses. Off, it goes to the team its "
            "directory gives."
        ),
    )
    session_cost_limit: float = Field(
        default=0.0,
        ge=0,
        description=(
            "What the agents of one session may cost in US dollars before "
            "they are put to rest and the human is asked whether it goes on "
            "for as much again. 0: no limit."
        ),
    )
    max_open_sessions: int = Field(
        default=0,
        ge=0,
        description="How many sessions may be open at once. 0: no limit.",
    )


class DirectorSettingsUpdate(BaseModel):
    """New values for some of the settings; what is left out stays."""

    max_teams: int | None = Field(default=None, ge=1)
    max_depth: int | None = Field(default=None, ge=1)
    allowed_profiles: list[str] | None = None
    resume_on_start: bool | None = None
    report_compaction_chars: int | None = Field(default=None, ge=1_000)
    stall_check_minutes: int | None = Field(default=None, ge=0)
    manager_check_accepted_reviews: int | None = Field(default=None, ge=0)
    manager_check_ic_setbacks: int | None = Field(default=None, ge=0)
    route_requests: bool | None = None
    session_cost_limit: float | None = Field(default=None, ge=0)
    max_open_sessions: int | None = Field(default=None, ge=0)


class DirectorSettingsState(DirectorSettings):
    """The settings, beside what the org has now."""

    teams: int = Field(description="How many teams the org has.")
    depth: int = Field(description="How many levels of teams the org has.")
    max_concurrent_runs: int | None = Field(
        description=(
            "How many conversations the server runs at once, the org's "
            "agents among them; teams beyond that wait to start. None when "
            "the director was not told."
        )
    )


class TeamMatch(BaseModel):
    """A team found by its title or its directory."""

    id: str
    title: str
    path: str = Field(description="The directory the team owns; '' at the root.")


class RecoveryItem(BaseModel):
    """One thing a paused org takes up again, or cannot."""

    node_id: str = Field(
        description="The place it is about (a team in a session); '' for the whole org."
    )
    title: str
    kind: RecoveryKind
    detail: str = Field(description="What it is, in words for the human.")


class Recovery(BaseModel):
    """What a paused org does when it is resumed."""

    paused: PauseReason | None = Field(
        description="Why the org is paused; None when it is not."
    )
    items: list[RecoveryItem] = Field(
        description=(
            "What resuming does, team by team, and what it cannot do. Empty "
            "when the org is not paused."
        )
    )
    messages_held: int = Field(
        description="How many messages to agents wait for the org to resume."
    )


class OrgConflict(BaseModel):
    """Something both sides of a pull changed, for the human to decide."""

    id: str = Field(description="Names the conflict in a resolve request.")
    kind: ConflictKind = Field(
        description=(
            "`field`: one field of a team was changed on both sides. `team`: "
            "one side removed a team the other changed. `file`: both sides "
            "changed the same lines of a spec or of contracts."
        )
    )
    node_id: str = Field(description="The team; '' for a file of the team_memory.")
    title: str
    field: str = Field(description="The field, or the file's path.")
    mine: str | None = Field(description="What is here; None when removed here.")
    theirs: str | None = Field(
        description="What the remote has; None when removed there."
    )
    choice: Side | None = Field(
        default=None, description="What was chosen; None while it is open."
    )


class OrgMerge(BaseModel):
    """A pull that stopped because both sides changed the same thing.

    Nothing of it has been written to the org. The org stays paused until
    every conflict has a choice, or the merge is abandoned.
    """

    remote: str
    theirs: str = Field(description="The remote's org commit being merged.")
    conflicts: list[OrgConflict]


class OrgSyncRequest(BaseModel):
    """Which remote to exchange the org with."""

    remote: str = Field(default="origin", min_length=1)


class OrgSync(BaseModel):
    """What a pull or a push of the org did."""

    remote: str
    result: SyncResult = Field(
        description=(
            "`up_to_date`: nothing to take in. `fast_forward`: only the "
            "remote had changed. `merged`: both had, and nothing clashed. "
            "`conflicts`: both changed the same thing; see `merge`. "
            "`pushed`: the remote now has what is here."
        )
    )
    head: str | None = Field(description="The local org's commit afterwards.")
    merge: OrgMerge | None = Field(
        default=None, description="The merge waiting for choices, if any."
    )
    memory: MemorySync | None = Field(
        default=None, description="What happened to specs, contracts and reports."
    )


class ResolveMergeRequest(BaseModel):
    """Choices for the conflicts of a merge."""

    choices: dict[str, Side] = Field(
        default_factory=dict, description="A side per conflict id."
    )
    rest: Side | None = Field(
        default=None, description="The side for every conflict not named."
    )


class StartAgainRequest(BaseModel):
    """How to start a team whose agents have no conversation here."""

    conversation: StartConversationRequest | None = Field(
        default=None,
        description=(
            "How to start the team's IC. Its workspace is replaced by the "
            "team's own checkout; the team's Manager is derived from it. "
            "Without one the IC is started like the IC above it in its "
            "session, or from the team's own agent profile, which needs an "
            "IC of the org to run on this server."
        ),
    )


class OrgSummary(BaseModel):
    """Whether the director has an org, and where it would keep one."""

    root_id: str | None = Field(description="None when no org has been started.")
    node_count: int
    repo_path: str = Field(
        description=(
            "The repository the org works in: the one its first conversation "
            "was opened in, or the nearest repository above that one which "
            "has it as a submodule and already had an org. Without an org, "
            "the directory the server was started for."
        )
    )
    directory: str = Field(
        description=(
            "The directory the server was started for, relative to "
            "`repo_path`; '' when they are the same. A message to the org "
            "that names no path is taken to come from here."
        )
    )
    is_git_repository: bool = Field(
        description=(
            "Whether `repo_path` is in a git repository. An org can only be "
            "started in one."
        )
    )
    boundary_enforcement: BoundaryEnforcement = Field(
        description=(
            "How far a team is held to its boundary. 'sandbox': the shell of "
            "an agent can write only the directory its team owns, and its "
            "file tools are refused anything else. 'checks': only the file "
            "tools are refused; the server cannot confine a shell."
        )
    )
    paused: PauseReason | None = Field(
        default=None,
        description=(
            "Why the org's agents are held and its steps and messages wait; "
            "None when the org runs. `GET /org/recovery` says what resuming "
            "does."
        ),
    )
    merging: bool = Field(
        default=False,
        description="A pull stopped on conflicts and waits for choices.",
    )


class CreateOrgRequest(BaseModel):
    """Start an org: its top team, and the session that works on ``task``."""

    title: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The root team's name. Left out, the team is first named after "
            "the task's first line and then by a model that summarizes it."
        ),
    )
    task: str = Field(min_length=1, description="What the org is to deliver.")
    conversation: StartConversationRequest = Field(
        description=(
            "How to start the IC of the org's first session, which works at "
            "the top team. Every other agent of the org is derived from one "
            "that runs already. The org lives in the git repository of its "
            "workspace. Its initial message is ignored: the IC is briefed "
            "with `task`."
        )
    )


class RouteMessageRequest(BaseModel):
    """Something the human says to the org, from somewhere in the repository."""

    text: str = Field(min_length=1)
    path: str | None = Field(
        default=None,
        description=(
            "The directory the message is about: relative to the org's "
            "`repo_path`, or absolute inside it (a submodule's directory is "
            "inside the repository that has the submodule). Defaults to the "
            "org summary's `directory`."
        ),
    )
    auto: bool = Field(
        default=False,
        description=(
            "Let the director choose a team below the one `path` gives, by "
            "the files the text names and by what a model makes of it. It "
            "moves down only when it is sure, and not at all while the "
            "org's `route_requests` is off."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "The session the human says it in, when the message continues "
            "that session's conversation and begins no new one. It goes to "
            "the IC of one of the session's own teams, those it set up "
            "itself included: the team it was opened at unless the director "
            "is sure of one below, and always that team while the org's "
            "`route_requests` is off. `path`, `auto` and `conversation` are "
            "ignored."
        ),
    )
    conversation: StartConversationRequest | None = Field(
        default=None,
        description=(
            "How to start the IC of the session the message opens; without "
            "it that IC is derived from one that runs already. Needed when "
            "there is no org yet: if the git repository of its workspace "
            "already has a saved org, that org is taken up and the message "
            "routed in it; otherwise the message starts an org there and "
            "becomes its task. See `CreateOrgRequest`."
        ),
    )


class RoutedMessage(BaseModel):
    """Where a message to the org went."""

    node_id: str = Field(
        description=(
            "The place that received it: the team's in the session the "
            "message opened, or was said in."
        )
    )
    team: str = Field(default="", description="The title of that team.")
    conversation_id: str = Field(description="The conversation of the IC there.")
    started_org: bool = Field(
        description="There was no org, and the message started one."
    )
    chosen_by: ChosenBy = Field(
        default="path",
        description=(
            "How the team was found. `path`: it owns the directory the "
            "message came from. `named`: the director chose it by the files "
            "the text names. `model`: the director chose it by what a model "
            "made of the text. `top`: the director was asked to choose and "
            "was not sure of any team below the one the directory gives."
        ),
    )
    reason: str = Field(
        default="", description="Why this team, in a sentence; may be empty."
    )


class OpenSessionRequest(BaseModel):
    """A request the human wants worked on, in a session of its own."""

    text: str = Field(min_length=1, description="What the session is to do.")
    title: str = Field(
        default="",
        description="What to call the session; taken from the text when empty.",
    )
    node_id: str | None = Field(
        default=None,
        description=(
            "The team the session is opened at. Without it the director "
            "chooses the team as it does for a message sent with `auto`."
        ),
    )


class ContractsChoice(BaseModel):
    """What a Manager decides about contracts that wait for it."""

    path: str = Field(description="The directory whose contracts wait.")
    take: Literal["org", "session", "text"] = Field(
        description=(
            "'org': the org's stay. 'session': the session's hold. 'text': "
            "what is written in `text` holds."
        )
    )
    text: str = Field(default="", description="The contracts, for 'text'.")


class SpecChoice(BaseModel):
    """What a reviewing IC decides about the spec of a directory."""

    path: str = Field(description="The directory whose spec it is.")
    take: Literal["org", "session", "text"] = Field(
        description=(
            "'org': the org's stays. 'session': the session's holds. 'text': "
            "what is written in `text` holds."
        )
    )
    text: str = Field(default="", description="The spec, for 'text'.")


class TeamChoice(BaseModel):
    """What a Manager decides about a team that waits for it."""

    team: str = Field(description="The id of the waiting team.")
    take: Literal["org", "session"] = Field(
        description=(
            "'org': it stays as the org has it. 'session': it becomes what "
            "the session made of it."
        )
    )
    under: str | None = Field(
        default=None,
        description=(
            "Only with 'session': the id of the team of the org to put it "
            "under, instead of where the session has it. Its directory must "
            "be inside that team's."
        ),
    )
    title: str | None = Field(
        default=None,
        description="Only with 'session': what to call it instead.",
    )


# What a session and the org each made something else of. `code`: a file
# both changed. `spec`, `contracts`: the text of one directory. `team`: a
# sub-team.
MergeKind = Literal["code", "spec", "contracts", "team"]
# Who settles it: the IC that reviews the session's work, before its code
# is merged, or the Manager of its Division Lead place.
Gate = Literal["ic", "manager"]
Take = Literal["org", "session", "text"]


class MergeItem(BaseModel):
    """One thing a session and the org each made something else of since
    the session began, which someone has to settle before it is the org's."""

    kind: MergeKind
    key: str = Field(
        description=(
            "What it is about: a file for `code`, a directory for `spec` and "
            "`contracts`, a team's id for `team`."
        )
    )
    gate: Gate
    title: str = Field(default="", description="What a team is called.")
    session: str = Field(
        default="", description="What the session has: a text, or a team in words."
    )
    org: str = Field(default="", description="What the org has there now.")
    take: Take | None = Field(
        default=None,
        description=(
            "What was decided: the org's stays, the session's holds, or "
            "`text` does. None while nobody has decided."
        ),
    )
    text: str = Field(default="", description="What holds when `take` is 'text'.")
    moved_to: str | None = Field(
        default=None,
        description=(
            "The team of the org a team would be put under, when the team "
            "the session has above it is not in the org any more."
        ),
    )
    with_team: str | None = Field(
        default=None,
        description=(
            "The waiting team contracts come with: they are the session's "
            "only if that team becomes the org's."
        ),
    )
    under: str | None = Field(
        default=None,
        description="The team of the org its Manager decided to put a team under.",
    )
    named: str | None = Field(
        default=None, description="What its Manager decided to call a team."
    )


class Merge(BaseModel):
    """How the work a session submitted becomes the org's: what waits to be
    settled, and how far it has come. Its code and specs go in when the
    reviewing IC accepts; its teams and contracts when it is their turn and
    the Division Lead Manager has settled what waits of them."""

    seq: int = Field(
        description=(
            "Its place in the order: work submitted earlier goes into the org first."
        )
    )
    items: list[MergeItem] = Field(default_factory=list)
    code_landed: bool = Field(
        default=False,
        description="Whether the code of its accepted work is in the repository.",
    )
    org_queued: bool = Field(
        default=False,
        description=(
            "Whether its teams and contracts wait for those of a session "
            "that submitted earlier. What waits of them is not known yet."
        ),
    )
    asked: str = Field(
        default="", description="What names the state its Manager was last asked of."
    )
    asked_at: str = Field(default="", description="When its Manager was last asked.")
    reminded: bool = Field(
        default=False,
        description="Whether its Manager was asked again for not answering.",
    )
    note: str = Field(default="", description="What the Manager said in settling.")


class SessionTeamInput(OrgNodeInput):
    """A team as it is at work in one session.

    It is no team of the org tree. It holds what a session has of its own at
    a team: agents, a checkout, a review, escalations. The team's title and
    boundary are the team's, as they were when the session was opened.
    ``parent_id`` is what is above it in the session: the same session's
    place for the team above, or, for the team the session was opened at,
    the session's Division Lead place, which has none.
    """

    session: str = Field(description="The session it belongs to.")
    worktree_path: str
    branch: str
    agents: list[AgentRef] = Field(default_factory=list)
    status: NodeStatus
    escalations: list[Escalation] = Field(
        default_factory=list,
        description=(
            "Open escalations this team raised, oldest first, wherever in the "
            "chain above they are held."
        ),
    )
    pending_split: SplitProposal | None = Field(
        default=None,
        description="A split of this team's boundary that is not carried out yet.",
    )
    base_commit: str = Field(
        default="",
        description=(
            "The commit the team's work is compared with when it is reviewed: "
            "where its checkout was when the team started, or when its work "
            "was last accepted."
        ),
    )
    review: Review | None = Field(
        default=None,
        description="The review of what the team last submitted.",
    )
    stopped: Stopped | None = Field(
        default=None,
        description=(
            "Set while the IC of the team above has stopped the team: its "
            "agents and those of every team below it are at rest and "
            "nothing is sent to them."
        ),
    )
    boundary_enforcement: TeamEnforcement | None = Field(
        default=None,
        description="How this team is held to its boundary; None until its IC ran.",
    )
    waiting_to_start: bool = Field(
        default=False,
        description=(
            "An agent of the team could not be started yet because the "
            "server runs as many conversations as it may; it starts when "
            "one of them ends."
        ),
    )
    team_id: str = Field(description="The team of the org it is the work of.")
    division_lead: bool = Field(
        default=False,
        description=(
            "Whether this is the session's Division Lead place: "
            "the team above the one the session was opened at, or that team "
            "itself for a session opened at the top. Its IC only reviews."
        ),
    )
    merge: Merge | None = Field(
        default=None,
        description=(
            "On a session's Division Lead place: how the work it "
            "submitted becomes the org's. None when nothing of it waits."
        ),
    )


class SessionTeam(SessionTeamInput, OrgNode):
    """A team of a session, including the fields the store derives.
    ``child_ids`` are the session's records for the team's sub-teams."""


class Usage(BaseModel):
    """What agents have used of their models."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = Field(
        default=0.0,
        description=(
            "In US dollars: what the agents' models reported and, where one "
            "reported none, what its tokens cost by the public price list."
        ),
    )
    unpriced_tokens: int = Field(
        default=0,
        description=(
            "How many of the tokens are of models that reported no cost and "
            "have no price in the list. They are not in `cost`, so a limit "
            "on cost does not hold them."
        ),
    )

    def plus(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost=self.cost + other.cost,
            unpriced_tokens=self.unpriced_tokens + other.unpriced_tokens,
        )


class SessionInput(BaseModel):
    """A line of work the human opened at a team. All work of an org is
    done in sessions."""

    id: str
    title: str
    team_id: str = Field(description="The team it was opened at.")
    above_id: str = Field(description="The team whose IC reviews its work.")
    request: str = Field(default="", description="What the human asked of it.")
    opened_at: str = Field(default="", description="ISO 8601 timestamp.")
    branch: str = Field(
        default="",
        description=(
            "The branch the repository's checkout was on when the session "
            "was opened. Its accepted work is merged into that branch only; "
            "'' when it was on none."
        ),
    )
    usage: Usage = Field(
        default_factory=Usage,
        description=(
            "What its agents have used so far, with the director's own "
            "model calls for them."
        ),
    )
    cost_allowance: float = Field(
        default=0.0,
        description=(
            "What it may cost in US dollars before its agents are put to "
            "rest and the human is asked; 0 while nothing limits it."
        ),
    )
    shape: list[tuple[str, str, str, str]] = Field(
        default_factory=list,
        description=(
            "The sub-teams the org had under the team when the session's "
            "work was last made the org's: each team, the team above it, "
            "its directory and what else makes it that team."
        ),
    )


ConversationRelation = Literal["above", "beside", "below"]


class ConversationLink(BaseModel):
    """An agent's conversation, as it is reached from another one."""

    conversation_id: str
    relation: ConversationRelation = Field(
        description=(
            "Where the agent is from the conversation asked about: `above` "
            "is an agent of the team that reviews its team's work, `beside` "
            "the other agent of its own team, and `below` the IC of a team "
            "directly below."
        )
    )
    node_id: str = Field(description="The team of the org the agent works for.")
    team: str = Field(description="That team's title.")
    role: AgentRole
    session_id: str = Field(default="", description="The session the agent works in.")
    session: str = Field(default="", description="That session's title.")
    status: str = Field(default="", description="The agent's status.")


class ConversationPlace(BaseModel):
    """Where an agent's conversation is in the org, and the conversations
    one level from it."""

    conversation_id: str
    node_id: str = Field(description="The team of the org the agent works for.")
    team: str = Field(description="That team's title.")
    role: AgentRole
    session_id: str = Field(default="", description="The session the agent works in.")
    session: str = Field(default="", description="That session's title.")
    human_joined: bool = Field(
        default=False,
        description="Whether the human has written in this conversation.",
    )
    links: list[ConversationLink] = Field(
        default_factory=list,
        description="Above first, then beside and below.",
    )


class Session(SessionInput):
    """A session as a client reads it.

    A session is apart from the org tree: the tree is the org's structure,
    and a session is work over a part of it. For the team it was opened at
    and each team below that one, the session has agents and a checkout of
    its own. Its work is reviewed by an IC at the team above that only
    reviews and, once accepted, merged into the repository's checkout.
    """

    team: str = Field(description="The title of the team it was opened at.")
    status: NodeStatus = Field(description="Of the team it was opened at.")
    review: ReviewState | None = Field(
        default=None, description="Where the review of its work stands."
    )
    conversation_id: str | None = Field(
        default=None,
        description="The conversation of the IC the session started with.",
    )
    teams: list[SessionTeam] = Field(
        description=(
            "What it has at the team it was opened at, then at each team "
            "below it, and last its Division Lead place. One whose "
            "`agents` is empty was given no task in this session yet."
        )
    )


class OpenedSession(BaseModel):
    """A session that was just opened."""

    session: Session
    conversation_id: str = Field(
        description="The conversation of the IC the session starts with."
    )
    chosen_by: ChosenBy = "path"
    reason: str = ""


class RoutedRequest(BaseModel):
    """Something the human said that the director sent straight to a team
    below the one it was said to, or that such a team said was not its own."""

    at: str = Field(description="ISO 8601 timestamp.")
    kind: Literal["asked", "misrouted"]
    team_id: str
    team: str
    text: str
    chosen_by: ChosenBy = "model"
    reason: str = ""
    id: str = Field(
        default="",
        description="Names the request, so that it is kept for the org once.",
    )


class AnswerEscalationRequest(BaseModel):
    """The human's answer to an escalation that waits on them."""

    answer: str = Field(min_length=1)


class EscalationPathStep(BaseModel):
    """One team on the way to an escalation, from the one its session was
    opened at."""

    id: str
    title: str


class OpenEscalation(BaseModel):
    """An escalation that waits on the human, with the teams it came up through."""

    node_id: str = Field(description="The team that raised it.")
    session_id: str = Field(
        default="",
        description="The session that team raised it in.",
    )
    session: str = Field(default="", description="The title of that session.")
    escalation: Escalation
    path: list[EscalationPathStep] = Field(
        description=(
            "The team its session was opened at first, ending at the team "
            "that raised it."
        )
    )


class SnapshotEvent(BaseModel):
    """The part of the tree one subscriber follows.

    ``nodes`` holds the focused team, its ancestors, its direct reports and
    the direct reports of every expanded team; or, for a full subscription,
    the focused team, its ancestors and every team below it.
    """

    type: Literal["snapshot"] = "snapshot"
    focus_id: str
    nodes: list[OrgNode]
    paused: PauseReason | None = Field(
        default=None, description="Why the org is paused; null while it runs."
    )
    merging: bool = Field(
        default=False,
        description="A pull stopped on conflicts and waits for choices.",
    )
    escalations: list[OpenEscalation] = Field(
        default_factory=list,
        description=(
            "Every escalation of any session that waits on the human, wherever "
            "the subscriber is focused: critical issues first, then longest "
            "waiting first."
        ),
    )
    sessions: list[Session] = Field(
        default_factory=list,
        description="Every open session, wherever the subscriber is focused.",
    )


class NodeUpsertEvent(BaseModel):
    """A team was added or changed."""

    type: Literal["node_upsert"] = "node_upsert"
    node: OrgNode


class NodeRemovedEvent(BaseModel):
    """A team was removed."""

    type: Literal["node_removed"] = "node_removed"
    id: str


class OrgStateEvent(BaseModel):
    """Whether the org is paused, and whether a pull waits for choices.

    Sent whenever either changes; the snapshot has them as they were.
    """

    type: Literal["org_state"] = "org_state"
    paused: PauseReason | None = Field(
        description="Why the org is paused; null while it runs."
    )
    merging: bool = Field(
        description="A pull stopped on conflicts and waits for choices."
    )


class SessionsEvent(BaseModel):
    """A session was opened or closed, or one of its teams changed."""

    type: Literal["sessions"] = "sessions"
    sessions: list[Session] = Field(description="The complete new list.")


class EscalationsEvent(BaseModel):
    """The escalations that wait on the human changed."""

    type: Literal["escalations"] = "escalations"
    escalations: list[OpenEscalation] = Field(
        description="The complete new list, in the snapshot's order."
    )
