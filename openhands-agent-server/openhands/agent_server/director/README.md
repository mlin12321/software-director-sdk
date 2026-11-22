# Running the director

How to start an org on your own machine. What the director does, part by
part, is in [OVERVIEW.md](OVERVIEW.md); the rules its code keeps to are in
[AGENTS.md](AGENTS.md).

## What it needs

- **A git repository with at least one commit.** The org tree and the team
  memory are kept in it under the hidden refs `refs/agents/*`; accepted work
  is merged into its checkout, on the branch that was checked out when the
  session opened. Canvas offers to make a plain folder one (below).
- **`OH_DIRECTOR_REPO_PATH`:** that repository. Without it the server's
  workspace path is used.
- **`OH_DIRECTOR_SERVER_URL`** (optional): where the server answers, such as
  `http://127.0.0.1:8000`. An agent that runs its own tools (Claude Code)
  calls the director from its own process and needs this address. Left out,
  the director uses the address the server is bound to.
- **`OH_CONVERSATION_WORKTREE_ROOT`:** where each session's worktree is put.
  The default is under `/tmp`, which a reboot clears. Set a folder that
  stays.
- **bubblewrap (`bwrap`)** for an agent that runs its own tools: without it
  the agent is not confined to its team's folder, and the server logs a
  warning.
- **Network** for the code review instructions: they are fetched from
  GitHub and given to the IC that reviews. Without network that checklist
  is left out of the review request.

## Starting it with Canvas

Linux, macOS, or Ubuntu under WSL. Once, to prepare: `make build` in this
repository, `npm ci` in the Canvas checkout (never `npm install`), and a
folder for worktrees.

Then, from the Canvas checkout, with this repository as the server:

```sh
OH_AGENT_SERVER_LOCAL_PATH=/abs/path/to/software-director-sdk \
OH_DIRECTOR_REPO_PATH=/abs/path/to/your/repository \
OH_CONVERSATION_WORKTREE_ROOT=/abs/path/to/worktrees \
npm run dev:minimal
```

Open the address it prints, normally `http://localhost:3001`. Under WSL the
Windows browser reaches it. Ctrl+C stops it.

A stock Agent Server has no director: the org page then says it is
unavailable.

- **Where its state goes.** Profiles, keys and conversations are kept in
  `~/.openhands/agent-canvas`. `OH_CANVAS_SAFE_STATE_DIR=/some/folder` keeps
  a run apart from that.
- **A faster page.** `npm run dev:minimal` serves the page unbundled, for
  editing Canvas: about 1,500 requests and 2 to 3 seconds for every page
  load. `npm run dev:static`, with the same variables, builds the page once
  (about 15 seconds) and serves that: about 300 requests and under a second.
  Use it unless you are changing Canvas.
- **As a desktop window.** `npm run desktop`, with the same variables, opens
  the page in a window of its own and starts the server itself, so there is
  no address to open. Under WSL it needs WSLg (Windows 11). It was started
  this way once on WSL and served the director; an installed app
  (`npm run build:desktop`) was not tried, and without
  `OH_AGENT_SERVER_LOCAL_PATH` it would download the stock server, which has
  no director.
- **One at a time.** A second start fails while the first holds its ports:
  3001 and 18000, and for the static and desktop ones 8000 and 18001 too.

## The first request

- **There is no "create org" step.** Open the org page and write what you
  want built. The first message starts the org, and its first session.
- **The agent it runs on** is the active agent profile. Every agent of the
  org is started from it unless the settings allow other profiles for
  sub-teams.
- **A folder that is not a git repository.** The org page sends you to Home
  to choose a folder. If the one you choose is not in a git repository, a
  command appears under the text box and runs in the folder before your
  first message: `git init`, add what is there, commit. Edit it to leave
  files out, or remove it to start an ordinary conversation with no org.
- **Every later request is a session of its own,** with its own agents and
  its own worktree. Its work is reviewed, then merged into your checkout.
- **Only you close a session.**

## Agents and models

An agent profile is made in the settings of Canvas and made active there.

- **A model the server calls.** Choose the provider and the model and give
  its API key. Models are called through LiteLLM, so any provider it knows
  can be named, OpenRouter included (`openrouter/<vendor>/<model>` with an
  OpenRouter key). No org has been run on OpenRouter yet.
- **Claude Code,** or another agent that runs its own tools, under your own
  login to it on this machine. It needs bubblewrap to be held to its folder.
- **Keys stay with the server's conversations.** They never reach the org's
  git refs, so a pushed org carries none.

## Agent types and `director.yaml`

Every team has a type, and a type is a YAML file: a team template. It says what the team's agents are told (what the team is for, what its spec, contracts and reports hold, what the director's tools mean to it, how its work is reviewed), and it may name the team's members. Three are shipped (`director/types`): `software-engineer` (the default), `researcher` and `research-engineer`. An IC names a type for each sub-team it proposes, so one org can hold all three.

To add one, point the server at a settings file and put the type beside it:

```
OH_DIRECTOR_CONFIG_PATH=/path/to/org/director.yaml
/path/to/org/director.yaml
/path/to/org/types/analyst.yaml
```

```yaml
# director.yaml: any of the director's settings
soft_check_in_minutes: 30
max_depth: 2
```

```yaml
# types/analyst.yaml
description: Reads results and says what they mean.
ic: |
  Your team analyses results. It runs no experiments.
actions:
  submit_review: Submit one table and what it shows.
review:
  checklist: |
    Every number in the table is in a committed result file.
omit_tools: [propose_teams]
```

A type cannot change what the runtime enforces: folders, review and landing are the same for every team. Settings changed on the org page hold over `director.yaml`.

A type that says nothing of members has an IC and a Manager. A type may name its own instead, with `agents`:

```yaml
# types/lab.yaml
description: A lab that checks what its benches find.
agents:
  bench:
    writes: true
    text: Runs the experiments.
    tools: [read_team_context, raise_escalation, report, write_spec,
            submit_review, propose_teams, msg_team]
    lead_tools: [read_team_context]
  referee:
    like: bench
    text: Checks what the benches below found.
    tools: [read_team_context, review_team]
    lead_tools: [read_team_context, review_team]
  chair:
    like: bench
    tools: [read_team_context, approve_teams, decline_teams]
    lead_tools: [read_team_context, settle_teams]
```

A member is whatever the type names it, and what it may do is the tools it is given. The director finds whom to turn to by tool: submitted work goes to the member of the team above that has `review_team`, proposed sub-teams to the one that has `approve_teams`. `lead_tools` are what a member has where its team only reviews a session's work. A type is refused unless exactly one member `writes` (works in the team's folder), and `submit_review`, `review_team`, `propose_teams`, `approve_teams` and (among `lead_tools`) `settle_teams` are each held by exactly one member. The texts the director itself sends still say "IC" and "Manager", so a team with other names is enforced as written but reads those words. No type with its own members has been run with real agents yet.

Every type needs a `description`: the agents and the org page choose a type by it. A type file added while the server runs is found without a restart.

Which type a team gets:

- **The first team of an org** is of the type chosen beside the chat box, or the default one.
- **A sub-team** is of the type its IC names for it in `propose_teams`, or the default one.
- **The default type and the types allowed** are set for every org in `director.yaml` (`default_agent_type`, `allowed_agent_types`; empty allows all), and for one org on its settings page. An org's own are kept in its repository.
- **They hold for new teams only.** A team that exists keeps its type.

A type may also name the agent profile its teams run on, with `sas_profile: <profile name>` (a software-agent-sdk profile, as made in the settings of Canvas). It is used when the IC that proposes the team names no model for it, and it need not be among the profiles the org lets ICs choose. That is the only tie between the two: a type without it runs on whatever the team above does. The first team of an org always runs on the profile picked in the chat box.

For research, an IC of type `researcher` splits by hypothesis, each direction submits after every experiment, and the IC above compares what its sub-teams submit and adds what it checked to the contracts with `review_team`, which every sub-team reads.

### What has been run with the research types

Small sessions on a toy data set (400 rows, four columns), two directions under one `researcher` team, on Cursor and on Claude Code:

- **A whole session lands.** Split by hypothesis, a finding with its command and committed result from each direction, `Checked:` lines added to the contracts on accept, the top team's submit, the Division Lead's accept, and the merge into the checkout.
- **A direction that waits on a long run** (minutes in one command) is told of to the IC above after the stall time and is not disturbed. The IC above is not itself reported as stalled while it waits.
- **A server stopped in the middle of an experiment** comes back paused; after the resume the direction ran its experiment again and submitted, and nothing was reviewed twice. That run had `OH_SECRET_KEY` set; without it the agents used to come back without the director's tools, which they are now given again on load.
- **With the protocol in the briefing.** After the org's protocol moved from the agents' system prompts into their first message, a software session with one split and the research session above each landed again on Cursor (seven minutes for the research one). Neither was long enough for a conversation to be condensed.
- **What it cost.** The same task on Claude Code (five agents, no review sent back) cost $7.57 in nine minutes for 2.1 million tokens. The top team submitted twice, once before and once after it wrote down what was established, so the Division Lead reviewed twice. Cursor reports no use, so a session on Cursor shows no cost and no cost limit holds it.

What a type does not do, since it is text: nothing but the reviewing IC checks that held-out data was left alone or that a claim has a committed result. Directions can read each other's folders. Contracts are never shortened by the director; the IC that owns them trims them with `edit_contracts`. Only trees one or two levels deep have been run.

A setting to start from: `default_agent_type: researcher`, `soft_check_in_minutes` longer than a usual experiment (the notice wakes the IC above, which costs a turn), `max_depth: 2`, and a session cost limit.

## Limits to set before a real run

All are in the settings of the org page. The two on use are off (0) until
you set them.

- **What a session may cost,** in US dollars. A session that reaches it
  rests, and you are asked; your answer lets it go on for as much again.
  The first real run with Claude Code (five agents, two small teams, one
  review sent back) cost $4.84 in nine minutes, for 2.8 million tokens,
  nearly all of them read from its cache.
- **Where the cost comes from.** Claude Code reports its own. For an agent
  on a model of the server's, the cost is LiteLLM's price for the model,
  cached tokens at their own price; a model behind a gateway is priced as
  the model its name ends in. Tokens of a model with no price anywhere are
  shown as "without a price" and no limit holds them.
- **Sessions open at once.**
- **Teams and depth** of the org.
- **How often an IC is told of a team at work** (`hard_check_in_minutes`, 360).
  A team that commits steadily never counts as showing no progress, so the
  IC above is told how it stands this often anyway, counted from when it
  last wrote to the team or was told of it. Each check-in costs that IC a
  turn. It is never more often than `soft_check_in_minutes`; 0 turns it off.
- **`OH_MAX_CONCURRENT_RUNS`** (server): how many agents run at once.

## What to expect

- **After a restart the org is paused.** Resume it on the org page, or turn
  on "resume on start" in the settings.
- **Set `OH_SECRET_KEY` on the server if it may be restarted.** Without it
  the server keeps no secrets between runs, so an agent comes back without
  the keys and tokens it was given. The director's own tools are the
  exception: an agent that runs its own tools (Claude Code, Cursor, any ACP
  agent) is given them again when its conversation is loaded.
- **An agent that fails** shows its reason in the team's details. The IC of
  the team above is told; where there is none, you are asked.
- **What waits on you** is counted on the sidebar's org link and listed at
  the top of the org page.
- **Use is counted about once a minute,** so a session can pass its limit by
  what its agents use in that minute.
- **Claude Code says what a turn cost only when the turn ends.** A session
  held by its limit has passed it by what its running turns cost: $0.21 and
  $0.66 on a $2.00 limit in the second real run. Set the limit below what
  you are willing to pay.
