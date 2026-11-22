# Live runs for #99 (role text in the briefing)

Two sessions with real agents, on this branch at the commit before this
file was added, 2026-11-22. Agents: Cursor only, through ACP
(`cursor-agent acp`, version 2026.09.10). The server ran with bubblewrap,
so `boundary_enforcement` was `sandbox`. Each session used a new
repository made for it.

Cursor reports no token use, so cost is not measured.

## 1. Software session with one split

Request: two small Python modules, `calc/` and `text/`, each with tests;
propose a sub-team for each, review them, submit the whole.

| Time | What happened |
|---|---|
| 0:00 | Org made; the session's IC started |
| 1:00 | IC proposed two sub-teams; its Manager was started by the proposal and approved it |
| 1:30 | Both sub-team ICs working |
| 2:00 | Both submitted; reviews pending |
| 3:00 | Both accepted; top team submitted and was accepted |
| 3:45 | Session done; work merged into the checkout |

Result: three merge commits on `main` (`calc`, `text`, the top team),
`python -m pytest` in the repository gives 6 passed, and the server log
has no error or traceback.

Each of the four conversations has `<ORG_PROTOCOL>` in exactly one event,
its first message. None has it in its stored agent or state: nothing of
the org is in a system prompt.

## 2. Research session (`agent_type: researcher`)

Request: which of hours, sleep and coffee in a 400-row CSV predict a
score; two directions, one experiment each, standard library only.

| Time | What happened |
|---|---|
| 0:00 | Org made; the session's IC started |
| 1:30 | IC proposed two directions by hypothesis (`hours/`, `sleep-coffee/`); Manager approved |
| 2:00 | Both directions working |
| 3:00 | First direction submitted; second a minute later |
| 5:00 | Both accepted by the IC above |
| 5:30 | Top team submitted; the Division Lead IC was started for the review |
| 7:30 | Division Lead accepted; session done; work merged |

Result: each direction committed its script and its result file, and
recorded the experiment before computing it. On accept the IC above added
three `Checked:` lines to `contracts.md` in team memory, each with the
finding, the team, the command and the result file. The data file has one
commit, the first: nobody changed it. The server log has no error or
traceback.

## What these runs do not show

- One run of each, on one model. They show that agents follow the
  protocol from a first message well enough to land a session; they do
  not measure how often.
- No conversation was long enough to be condensed, so the risk named in
  the pull request (the protocol being summarised away) was not tried.
- No comparison run on the branch before this one (#98) was made, so
  there is no measured difference between the text in the system prompt
  and in the briefing.
- No type with its own members was run with real agents.
- No restart, pause or resume in the middle of a session.
