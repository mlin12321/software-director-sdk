# Live runs for the stack #101 to #107

Five sessions with real agents on 2026-11-22. Agents: Cursor only, through
ACP (`cursor-agent acp`, version 2026.09.10). The server ran with
bubblewrap, so `boundary_enforcement` was `sandbox`. Each session used a new
repository made for it. The two sessions of a pair ran at the same time, so
that both had the same load.

Cursor reports no token use, so cost is not measured.

"#98" is the branch `refactor/director-team-view`, where the org's protocol
was still in the system-prompt suffix. "Tip" is #106 for runs 1 to 4 and
#107 for run 5.

## The numbers

| Run | Branch | Done after | Agents | Messages to agents | Tool calls | Result |
|---|---|---|---|---|---|---|
| 1. Software | #98 | 300 s | 5 | 12 | 176 | Landed whole; 23 tests pass |
| 2. Software | tip | 230 s | 4 | 11 | 148 | **Landed without one team's work**; 8 tests pass |
| 3. Research | #98 | 360 s | 5 | 13 | 198 | Landed whole |
| 4. Research | tip | 580 s | 5 | 16 | 296 | Landed whole |
| 5. Software, custom type | tip + #107 | 240 s | 5 | 13 | 154 | Landed whole; 15 tests pass |

"Done after" is from the request to the moment the session's work was
accepted and no agent ran, read every 10 seconds.

On the tip every agent had `<ORG_PROTOCOL>` in its first message (4 of 4,
5 of 5, 5 of 5). On #98 none had (0 of 5 twice): it was in the suffix.

## What the comparison shows, and what it does not

One run of each on one model. The differences in time and tool calls are
within what two runs of the same code could differ by: in run 4 the agents
chose to make more commits and write more notes than in run 3, and in run 1
the first agent chose worktrees for its sub-teams where in run 2 it did not.
Nothing here shows the protocol being followed better or worse from a first
message than from the suffix. Agents in all five runs proposed, approved,
gave tasks, submitted and reviewed in the order the protocol asks for.

## Run 2: accepted work was left out of what landed

The sub-teams Calc and Text shared the session's checkout (the proposal
asked for no worktrees). Calc committed and was accepted. The Text agent
then ran

    git checkout -b text/slugify-word-count b82d3af

The director's `git` for agents lets such a command run and then switches
back. The switch back failed: in the sandbox Calc's folder is read-only to
the Text agent, so git could not remove Calc's committed files on the way
to the old commit ("unable to unlink 'calc/ops.py': Read-only file
system") and then refused to come back over them ("untracked working tree
files would be overwritten by checkout"). The checkout stayed on the Text
agent's branch. The session's work is taken from the checkout's branch, so
`main` got Text's commit and not Calc's, which stayed on the old branch.
Both reviews said accepted and the session said done.

`git_index.py` is unchanged by this stack: the same would have happened on
`main`. #107 fixes it. In run 5, on #107, two agents tried to switch
branches and were refused with "the checkout stays on …", and both teams'
commits landed.

## Run 5: a type with its own members

The type `crew` has an `engineer` that writes, a `reviewer` that only
reviews, and a `lead` that approves sub-teams. The request was the one of
runs 1 and 2.

- The engineer proposed two sub-teams and the lead approved them.
- Both sub-teams submitted. The review requests went to the reviewer, not
  to the engineer, and the reviewer accepted both with `review_team`.
- The engineer submitted the whole and the session landed with both
  sub-teams' commits.
- The reviewer's first message says "You are the reviewer of your team".

One thing went wrong. The reviewer's ACP session was restarted by the SDK
when two review requests reached it close together ("Restarting ACP session
after cancelled prompt drain timeout", then "Failed to start ACP server:
Connection closed"). The director showed the reviewer as failed and raised
that to the human. The reviewer then went on by itself with the next
message and did both reviews, but the escalation "The reviewer of 'Tiny
tools' failed" was still open when the session was done. An agent that
recovers should take its failure notice with it; that is not fixed here.

## What these runs do not show

- Cost, for the reason above.
- A conversation long enough to be condensed. Cursor keeps its own history
  and gives the director no sign of it, so #103 cannot be tried with it.
- A restart, pause or resume in the middle of a session.
- A custom type below the top team: in run 5 the sub-teams were
  `software-engineer` teams.
