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

Every team has a type, and a type is a YAML file of what its agents are told: what the team is for, what its spec, contracts and reports hold, what the director's tools mean to it, and how its work is reviewed. One is shipped (`director/types`): `software-engineer`, the default. An IC names a type for each sub-team it proposes, so one org can hold several.

To add one, point the server at a settings file and put the type beside it:

```
OH_DIRECTOR_CONFIG_PATH=/path/to/org/director.yaml
/path/to/org/director.yaml
/path/to/org/types/analyst.yaml
```

```yaml
# director.yaml: any of the director's settings
stall_check_minutes: 30
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

Every type needs a `description`: the agents and the org page choose a type by it. A type file added while the server runs is found without a restart.

Which type a team gets:

- **The first team of an org** is of the type chosen beside the chat box, or the default one.
- **A sub-team** is of the type its IC names for it in `propose_teams`, or the default one.
- **The default type and the types allowed** are set for every org in `director.yaml` (`default_agent_type`, `allowed_agent_types`; empty allows all), and for one org on its settings page. An org's own are kept in its repository.
- **They hold for new teams only.** A team that exists keeps its type.

A type may also name the agent profile its teams run on, with `sas_profile: <profile name>` (a software-agent-sdk profile, as made in the settings of Canvas). It is used when the IC that proposes the team names no model for it, and it need not be among the profiles the org lets ICs choose. That is the only tie between the two: a type without it runs on whatever the team above does. The first team of an org always runs on the profile picked in the chat box.

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
- **`OH_MAX_CONCURRENT_RUNS`** (server): how many agents run at once.

## What to expect

- **After a restart the org is paused.** Resume it on the org page, or turn
  on "resume on start" in the settings.
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
