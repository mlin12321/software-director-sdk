# Running the director

How to start an org on your own machine. How the director works inside is in
[AGENTS.md](AGENTS.md).

## What it needs

- **A git repository with at least one commit.** The org tree and the team
  memory are kept in it under the hidden refs `refs/agents/*`; accepted work
  is merged into its checkout, on the branch that was checked out when the
  session opened.
- **`OH_DIRECTOR_REPO_PATH`:** that repository. Without it the server's
  workspace path is used.
- **`OH_DIRECTOR_SERVER_URL`** (optional): where the server answers, such as
  `http://127.0.0.1:8000`. An agent that runs its own tools (Claude Code)
  calls the director from its own process and needs this address. Left out,
  the director uses the address its own requests arrive at.
- **`OH_CONVERSATION_WORKTREE_ROOT`:** where each session's worktree is put.
  The default is under `/tmp`, which a reboot clears. Set a folder that
  stays.
- **bubblewrap (`bwrap`)** for an agent that runs its own tools: without it
  the agent is not confined to its team's folder, and the server logs a
  warning.
- **Network** for the automated review: its instructions are fetched from
  GitHub. Without network that review is skipped.

## Starting it with Canvas

From the Canvas checkout, with this repository as the server:

```sh
OH_AGENT_SERVER_LOCAL_PATH=/abs/path/to/software-director-sdk \
OH_DIRECTOR_REPO_PATH=/abs/path/to/your/repository \
OH_CONVERSATION_WORKTREE_ROOT=/abs/path/to/worktrees \
npm run dev:minimal
```

A stock Agent Server has no director: the org page then says it is
unavailable.

## The first request

- **There is no "create org" step.** Open the org page and write what you
  want built. The first message starts the org, and its first session.
- **The agent it runs on** is the active agent profile. Every agent of the
  org is started from it unless the settings allow other profiles for
  sub-teams.
- **Every later request is a session of its own,** with its own agents and
  its own worktree. Its work is reviewed, then merged into your checkout.
- **Only you close a session.**

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
- **A team run by Claude Code is named a little late.** It cannot be asked
  for a headline before its first turn, so it is asked when that turn is
  over. Until then the team shows the start of its task.
