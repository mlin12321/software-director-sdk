<ROLE>
You are a collaborative software engineering partner for this repository, with a
strong bias toward simplicity and maintainable code.

# Core Engineering Principles

1. **Simplicity and Clarity** — Prefer solutions that make special cases
   disappear over ones that add conditional checks. Keep functions short and
   single-purpose; avoid deep nesting.
2. **Backward Compatibility** — Stability is a feature, not a constraint.
   Changes should not break existing users or integrations; the enforced
   deprecation policies are in "API compatibility pointers" below.
3. **Pragmatic Problem-Solving** — Solve real problems users face, not
   theoretical edge cases. Prefer proven, straightforward approaches over
   speculative abstractions.

# Working Agreement

- Act on the request rather than restating it back. If a requirement is
  ambiguous in a way that would change the work, ask; otherwise proceed and
  state your assumption.
- When you disagree with an approach, say so once, concisely, with a concrete
  alternative — then proceed.
- Feedback on code should be specific and actionable: what is wrong, why it
  matters, and what to do instead.
</ROLE>

## Cross-Repository Boundaries

This repository owns the Python SDK and Agent Server: agent and tool behavior, conversations, workspaces, events, and the canonical REST/WebSocket API. Related responsibilities live elsewhere:

- [`OpenHands/OpenHands`](https://github.com/OpenHands/OpenHands) owns Agent Canvas UI, frontend state, backend selection, and local-stack orchestration.
- `clients/typescript/` mirrors the Agent Server API for browser-compatible TypeScript clients.
- [`OpenHands/extensions`](https://github.com/OpenHands/extensions) owns reusable skills, plugins, automations, and integrations; [`OpenHands/automation`](https://github.com/OpenHands/automation) owns automation definitions, scheduling, webhooks, run history, dispatch, and sandbox lifecycle orchestration; this repository executes the dispatched conversations.

The usual flow is SDK/Agent Server → OpenAPI contract → `clients/typescript` → Agent Canvas. Implement backend behavior and endpoints in the Python SDK or Agent Server packages first, then update the typed client and downstream applications as needed. If a PR is opened in the wrong repository, explicitly recommend closing and moving it to the repository that owns the change rather than merging it here.

All pull requests must comply with [`.agents/skills/custom-codereview-guide.md`](.agents/skills/custom-codereview-guide.md), in addition to the repository's contribution requirements and CI checks.

For issue triage, read [the review guide](.agents/skills/custom-codereview-guide.md#issue-triage-ownership-and-scope)
for ownership and scope; PR evidence is not an issue-readiness requirement.

## Review-Facing Implementation Checklist

Code should satisfy the repository review checkpoints before the PR is opened:

- Trace cross-layer changes through every affected public entry point, including
  factories, constructors, registries, serialization, REST/WebSocket transport,
  `clients/typescript/`, and create/resume/fork paths. Do not add a field or
  option at one layer while another supported path drops or ignores it.
- Treat public Python and server APIs, defaults, serialized events, persisted
  settings, and stored conversations as compatibility surfaces. Use the
  deprecation, schema-version migration, and golden-fixture mechanisms described
  below instead of one-off shims.
- Give tasks, processes, connections, plugins, event loops, and persistent
  artifacts an owner. Cancellation must stop underlying work; close and rollback
  paths must cover success, failure, and cancellation; shared conversation state
  must use its existing synchronization mechanism.
- Route credentials through `openhands.sdk.utils.pydantic_secrets` and verify the
  full input, serialization, persistence, logging, resume, and delivery path.
  Never introduce a parallel redaction or secret-sentinel implementation.
- Verify imports, executables, dependency installation, paths, and process
  cleanup in every affected production artifact, including the packaged Agent
  Server, Docker images, and relevant host platforms. A mocked unit test alone
  does not validate a packaging or installation change.
- Every LLM request built in this repo must place a `system` message before the
  first `user` message (see "LLM message construction invariant" in
  `DEVELOPMENT.md`). This covers the agent loop and all standalone calls
  (condensers, goal judge, profile pre-flight ping, security analyzers, etc.).
  The first message role must be `system`, or there must be a documented
  exception (ACP agents, subscription/Codex transport).

## Repository Memory
- Async LLM completions propagate through the full call chain: `LLM.acompletion()`/`LLM.aresponses()` → `_atransport_call()` (litellm `acompletion`/`aresponses`) → `RetryMixin.retry_decorator()` (tenacity `retry`, which wraps coroutines natively — there is no separate async retry path) → condenser `acondense()` → `Agent.astep()` → `LocalConversation.arun()` → `EventService.run()`. Every async method has a sync counterpart; base classes provide default delegations to sync so custom subclasses work without changes. Token callbacks use `AnyTokenCallbackType` (union of sync/async) with `_invoke_token_callback()` for transparent dispatch.
- Async LLM hard timeouts belong inside the tenacity-decorated attempt so each timeout is retryable; stream idle timeouts wrap each `__anext__()` call so receiving a chunk resets the idle deadline. Both become LiteLLM `Timeout` errors before retry classification and eventually map to `LLMTimeoutError` after retries are exhausted.
- `conversation.interrupt()` cancels in-flight `arun()` by cancelling the tracked `_arun_task`. `asyncio.CancelledError` propagates through all layers (LLM HTTP stream → agent step → conversation loop) without needing per-layer interrupt APIs, because LLM and Agent are frozen/stateless Pydantic models that may be shared across conversations. `arun()` catches `CancelledError`, sets status to `PAUSED`, and emits `InterruptEvent`. The agent-server exposes this via `EventService.interrupt()` → `ConversationService.interrupt_conversation()` → `POST /{conversation_id}/interrupt`.
- OpenHands provider LLMs keep public config as `openhands/<model>` and translate to `litellm_proxy/<model>` plus the OpenHands proxy `api_base` only at LiteLLM boundaries. Legacy persisted proxy payloads are migrated once at settings/profile load boundaries; do not add downstream UI reverse-mapping for the normal path.
- Programmatic settings live in `openhands-sdk/openhands/sdk/settings/`. Treat `AgentSettings` and `export_settings_schema()` as the canonical structured settings surface in the SDK, and keep that schema focused on neutral config structure and semantics; downstream apps should own client-specific ordering, icons, widgets, and slash-command presentation.
- `ACP_PROVIDERS` declares which ACP harnesses exist, not which ones a product offers. Product clients declare `SURFACED_ACP_PROVIDERS` independently and assert only `SURFACED_ACP_PROVIDERS ⊆ ACP_PROVIDERS`; this makes a rename or removal fail while an addition is a no-op. Do not require a complementary not-surfaced list or add deployment policy to `ACPProviderInfo`.
- ACP examples routed through the OpenHands LiteLLM proxy (`examples/01_standalone_sdk/40_acp_agent_example.py`, `examples/02_remote_agent_server/09_acp_agent_with_remote_runtime.py`) use a dedicated `ACP_LLM_API_KEY` whose LiteLLM key alias (`acp-llm-api-key`) is in the proxy's `ignored_key_aliases`. Claude Code sends its own non-OpenHands system prompt, which the proxy's identity guardrail would 403; the bypassed key opts out of the guardrail instead of making Claude Code pretend to be OpenHands. Real users running ACP against `api.anthropic.com` are unaffected. See saas-deploy PRs #1929 (staging) / #1930 (prod) and issue #6434.
- `SettingsFieldSchema` intentionally does not export a `required` flag. If a consumer needs nullability semantics, inspect the underlying Python typing rather than inferring from SDK defaults.
- `AgentSettings.tools` is part of the exported settings schema so the schema stays aligned with the settings payload that round-trips through `AgentSettings` and drives `create_agent()`.
- `mcp_config` is declared on `OpenHandsAgentSettings` and `ACPAgentSettings` (and inherited by `LLMAgentSettings`) — the variants of the `AgentSettingsConfig` union. It is a flat `dict[str, MCPServer]` server map, where `MCPServer` is the SDK's own model in `openhands-sdk/openhands/sdk/mcp/config.py` — not FastMCP's `MCPConfig`, and not the `.mcp.json`-style `{"mcpServers": {...}}` wrapper (the v4→v5 migration `_migrate_mcp_config_to_server_map` unwraps that key). FastMCP's typed `MCPConfig` is built only at the FastMCP boundary, in `_prepare_mcp_config()` in `openhands-sdk/openhands/sdk/mcp/utils.py`, from the payload produced by `to_fastmcp_mcp_config()` in `openhands-sdk/openhands/sdk/mcp/config.py`. Keep serialized MCP output compact with `exclude_none=True, exclude_defaults=True`, applied per-`MCPServer` in `openhands-sdk/openhands/sdk/mcp/config.py`.
- Persisted SDK settings should use the direct `model_dump()` shape with a top-level `schema_version`; avoid adding wrapped payload formats or legacy migration shims in `openhands/sdk/settings/model.py`.
- Persisted settings compatibility is enforced by `.github/scripts/check_persisted_settings_compat.py` plus `tests/sdk/persisted_settings_baselines/vN/` golden fixtures. When a versioned persisted settings shape changes incompatibly, bump the relevant schema version constant, add the migration step, and add a fixture for the old shape.
- The persisted-settings compatibility check builds its historical PyPI baseline environment with `uv` and a PyPI release-date `exclude-newer` cutoff, and golden fixtures may include a top-level `__expected__` map of dotted paths whose post-migration values must be preserved.

- Prefer removing genuinely temporary compatibility fields and serializers outright rather than carrying legacy settings shims in the SDK. Note that persisted settings now ship with a versioned migration chain plus golden fixtures and a CI gate, so check `tests/sdk/persisted_settings_baselines/` before deleting anything that looks like a shim — it may be load-bearing for an older persisted payload.
- Do not expose settings schema versions as public `CURRENT_PERSISTED_VERSION` class constants on `AgentSettings` or `ConversationSettings`; keep versioning internal to the `schema_version` field/defaults and private module constants.
- `ConversationSettings` owns the conversation-scoped confirmation controls directly (`confirmation_mode`, `security_analyzer`); keep those fields top-level on the model and grouped into the exported `verification` section via schema metadata rather than nested helper models, and prefer the direct settings-model constructor `create_request(...)` over separate request-wrapper helpers.
- Anthropic malformed tool-use/tool-result history errors (for example, missing or duplicated ``tool_result`` blocks) are intentionally mapped to a dedicated `LLMMalformedConversationHistoryError` and caught separately in `Agent.step()`, so recovery can still use condensation while logs preserve that this was malformed history rather than a true context-window overflow.
- LLM-specific behavior tweaks should start in `openhands-sdk/openhands/sdk/llm/utils/model_features.py` whenever they can be expressed as model/provider capabilities. Genuinely try to keep provider-specific criteria out of `openhands-sdk/openhands/sdk/llm/llm.py`; only touch `llm.py` when the behavior cannot be represented cleanly in the feature registry or a focused helper.
- AgentSkills progressive disclosure resolves in `AgentContext._resolve_dynamic_data()`, shared by `get_system_message_suffix()` and the prompt section registry (`openhands/sdk/context/prompts/sections/`), and lands in `<available_skills>`. `openhands.sdk.skills.to_prompt()` (`openhands-sdk/openhands/sdk/skills/skill.py`) truncates each prompt description to 1024 characters because the AgentSkills specification caps `description` at 1-1024 characters.
- Workspace-wide uv resolver guardrails belong in the repository root `[tool.uv]` table. When `exclude-newer` is configured there, `uv lock` persists it into the root `uv.lock` `[options]` section as both an absolute cutoff and `exclude-newer-span`, and `uv sync --frozen` continues to use that locked workspace state.
- PR code review is handled via OpenHands Cloud automation (not a GitHub Actions workflow). Repo-specific reviewer instructions live in `.agents/skills/custom-codereview-guide.md`. The automation triggers on `ready_for_review` (established contributors), when `all-hands-bot` is requested as a reviewer, and when the `review-this` label is added.
- Release PR reviewer guidance now requires checking the latest PR-specific `Run tests`, `Run Examples Scripts`, and `Run Integration Tests` results/comments before approval; if any are missing, stale, ambiguous, skipped, or failing, the bot should leave a COMMENT and defer to human maintainer review.
- Directory-based runnable examples under `examples/` should expose their entrypoint as `main.py`, and `tests/examples/test_examples.py` should explicitly list the example directory in `_TARGET_DIRECTORIES` so the non-recursive example workflow collects it without accidentally running helper modules.
- The duplicate-issue automation scripts should validate `owner/repo` arguments before interpolating GitHub API paths, handle per-issue auto-close failures without aborting the whole batch, and keep `app_conversation_id` paths unquoted because OpenHands conversation IDs are already canonicalized for those endpoints.
- GitHub Actions workflows pin external third-party actions to a full 40-character commit SHA, with the version tag in a trailing comment (`uses: owner/repo@<sha> # v1.2.3`); mutable tags or branches are not used for third-party actions. GitHub-authored (`actions/*`, `github/*`) and first-party (`OpenHands/*`) refs are currently exempt. Dependabot's `github-actions` ecosystem still bumps the pinned SHA (and the trailing comment), so pinning does not block security or version updates.

- `agent-server` now defaults `TMUX_TMPDIR` to a per-process directory under the system temp dir (`openhands-agent-server-<pid>`) when the environment variable is unset. This isolates tmux sockets/cleanup across concurrent server instances while still respecting an explicit `TMUX_TMPDIR` override.
- Conversation worktrees for git-backed local workspaces live under `/tmp/conversation-worktrees/<conversation_id>/<repo_root.name>`, and if the original workspace points at a subdirectory inside the repo, the active workspace should preserve that relative path inside the worktree.

- Agent-server Docker publish tags are defined centrally in `openhands-agent-server/openhands/agent_server/docker/build.py`; keep `server.yml` manifest publication derived from the emitted per-arch tags so SHA/branch/git-tag aliases stay in sync, while preserving the legacy `latest-<variant>` alias used by workspace defaults.
- The published agent-server Docker images in `.github/workflows/server.yml` must pass `OPENHANDS_BUILD_GIT_SHA` and `OPENHANDS_BUILD_GIT_REF` as explicit `docker/build-push-action` build args; the workflow only uses `docker/build.py` for context/tag generation, so those runtime env vars are otherwise left at the Dockerfile `unknown` defaults.
- The PyInstaller agent-server binary should copy OpenHands distribution metadata (`openhands-agent-server`, `openhands-sdk`, `openhands-tools`, `openhands-workspace`) in `openhands-agent-server/openhands/agent_server/agent-server.spec`, otherwise `/server_info` version lookups via `importlib.metadata` can fall back to `unknown` inside published binary images.
- Agent-server deferred init (warm-pool / dormant mode) is driven by `Config.deferred_init` (env `OH_DEFERRED_INIT`). The `InitService` in `openhands-agent-server/openhands/agent_server/init_router.py` owns the dormant→initializing→ready transition and is registered on `app.state.init_service` only when `deferred_init=True`; the `require_initialized` dependency, added to the `/api/*` router, returns 503 while not `ready`. Bootstrap auth for `POST /api/init` uses the existing `secret_key` (`X-Init-API-Key` header) — the orchestrator already holds this key for encryption, and it is overwritten when the per-user runtime config arrives in the init body. The agent-server's 5xx exception handler rewrites `detail` on 503s, so warm-pool orchestrators should rely on the HTTP status code (not the body) when probing dormant state.


- Auto-title generation should not re-read `ConversationState.events` from a background task triggered by a freshly received `MessageEvent`; extract message text synchronously from the incoming event and then reuse shared title helpers (`extract_message_text`, `generate_title_from_message`) to avoid persistence-order races.
- `RemoteConversation.generate_title()` now reconciles remote events and reuses the shared local `generate_conversation_title(...)` helper instead of calling the removed deprecated agent-server `/generate_title` REST route, so explicit remote title generation still works without a transport-only compatibility endpoint.


- Remote workspace git operations should call `/api/git/changes` and `/api/git/diff` via the `path` query parameter with slash-normalized strings; building those URLs with `pathlib.Path` leaks host-platform separators and breaks Windows paths. The grep tool now prefers `rg`, then system `grep`, then Python; both the real grep executor and the SDK's terminal-command compatibility fallback should keep that order. For grep parity, the Python fallback should hide dotfiles by default but still let explicit `include` globs surface files like `.env`, matching ripgrep. For glob parity, any symlink-preservation regression test should force the Python fallback path, because ripgrep availability changes whether the fallback implementation runs at all.
- Keep path helpers split by purpose: `is_absolute_path_source()` is for cross-platform source/wire syntax detection, while local filesystem writes/validation (for example, the file editor) should use host-native absolute-path semantics so POSIX does not silently accept Windows drive paths as creatable files.
- Tool availability filtering belongs in `openhands-sdk/openhands/sdk/tool/registry.py` via `list_usable_tools()`, which preserves registration order and defaults tools to usable unless they expose an `is_usable()` callable. Environment-specific checks like Chromium detection should live on the concrete tool class (`BrowserToolSet.is_usable()`), while agent-server surfaces such as `/server_info` should consume the registry helper rather than re-implement per-tool filtering.
- Pydantic secret field helpers live in `openhands-sdk/openhands/sdk/utils/pydantic_secrets.py`. `serialize_secret()` handles serialization (cipher / `expose_secrets` / default Pydantic masking); `validate_secret()` handles deserialization (cipher decryption, redacted/empty → `None`); `is_redacted_secret()` checks for the sentinel; `REDACTED_SECRET_VALUE` is the canonical sentinel string. For `dict[str, str]` fields whose values are all secrets, wrap each value in `SecretStr` and call `serialize_secret` per value (see `LookupSecret._serialize_secrets`). Do not hand-roll redaction logic in field serializers.
- Agent-server persistence detects whether settings or secret payloads contain values by exercising the real Pydantic secret-serialization pipeline with `_SecretProbeCipher` (`persistence/models.py`). Do not replace this with a hardcoded field list or a recursive `SecretStr` walk: fields such as `AgentContext.secrets` store plain strings and only become secret-bearing inside their serializers.

- The director (`openhands-agent-server/openhands/agent_server/director/`, routes in `director_router.py` under `/api/director`) keeps an org tree of agent teams and a blackboard in hidden git refs (`refs/agents/org`, `refs/agents/blackboard`) of the org's repository. That is the git repository of the workspace of the org's first conversation, wherever the server was started: `create_org` moves `DirectorService._repo_dir` there and writes it to `director.json` beside `Config.conversations_path`, which is read first on the next start. Until there is an org, the director looks in `Config.director_repo_path` (default `workspace_path`). A repository that has no org and is a git submodule stands for the nearest repository above it that has one (`DirectorService._load_from`), both at startup and when a first message arrives; `OrgSummary.repo_path` is the org's repository and `OrgSummary.directory` is where the server's own directory sits in it, which is where a message without a `path` is routed from. Tests that start an org through the app must point `conversations_path` at a temporary directory, or `director.json` leaks between them. `GitBlackboard` never touches the working tree or the user's index: it builds trees in a throwaway `GIT_INDEX_FILE` and lands them with a compare-and-swap `git update-ref`, retrying when another writer wins. `OrgStore` owns the derived `child_ids`, `descendant_counts` and `escalation_count` and is the only emitter of org events; mutate the tree through it. A stream subscriber follows one focus node plus the nodes it has expanded (or the whole subtree with `full`), and gets the list of escalations that wait on the human, for the whole tree whatever its focus, whenever that list changes. An escalation never goes straight to the human: it is held by the node above the one that raised it and moves only through `OrgStore.pass_up_escalation`, one node at a time with a reason, so `held_by` is `None` only after every node above has passed it up (the store rejects anything else). Keep it that way when adding ways to raise or route escalations. The wire types are mirrored by hand in `clients/typescript/src/models/director.ts` (client-ahead of the pinned release); change both together. `DirectorService` loads nothing until first use, so `create_app` stays free of git calls. An org's agents are ordinary conversations: `POST /api/director/org` starts the root IC from the caller's `StartConversationRequest`, and every other agent is derived from an existing one with `ConversationService.start_child_conversation` (a Manager from its team's IC, a sub-team's IC from the IC above), so model configuration and secrets stay in the conversation store and never reach the org ref. The director reaches conversations only through `PodLauncher` (`director/pods.py`), which tests replace with a fake. An agent that runs its own tools (an `ACPAgent`: Claude Code, Codex) can be any agent of an org, the root IC included. It cannot be given the director's tools in-process and its tool calls never pass `guard_action`, so `DirectorService._adapter` changes it as its conversation starts (the `adapt_agent` hook of `start_conversation` and `start_child_conversation`): `mcp_config` gains the stdio MCP server `director/mcp_shim.py` (standard library only, run by its path), which lists the tools from `GET /api/director/tools` and passes each call to `POST /api/director/tools/{name}`, and `acp_command` is prefixed with `<git dir>/director-shells/<conversation id>.acp.sh` (`sandbox.command_script`), which starts the agent's whole process in the team's sandbox with the team's `git` first on `PATH`. Those two routes are mounted outside the session-key dependency: the caller is known only by `X-Director-Agent-Token`, an HMAC of its conversation id (`director/acp.py`) with a key kept in `director.json`, so it survives restarts and never reaches the org ref. They run the same `pod_actions` as the in-process tools (`director/tools/remote.py`); add a tool to `DIRECTOR_TOOLS` and both ways get it. The shim needs the server's own address: `director_server_url`, or else the base URL of the director's last request. Without bubblewrap nothing holds an ACP agent's process, so its IC always gets its own worktree and what it commits outside its directory is refused at merge-back; `OrgNode.boundary_enforcement` says which holds per team (`sandbox`, `checks`, `merge`), and an ACP Manager is then held by its prompt alone. A split may give a sub-team another agent with `SplitChild.agent_profile`, a name from `DirectorSettings.allowed_profiles`; that team's IC is started from the stored profile with `PodLauncher.start_from_profile` as a child of the IC above, its Manager derived from it as always. The director's one-shot model calls (`summarize`, `ask`) are put to an ACP agent itself with `ask_agent`, since it has no model the server can call. Agents act on the org through the tools in `director/tools/`, which run in-process: a tool resolves its caller with `OrgStore.find_by_conversation`, `director/pod_actions.py` checks the caller's role and node, and anything that touches a conversation (starting agents, sending messages, saving the tree) is queued with `DirectorService._enqueue(effect, lane)`. Effects of one lane run in order and lanes run side by side: a team's node id for its starts and messages, `review:<id>` and `headline:<id>` for its model calls, `org` for saves. An effect that must follow another has to share its lane. Keep it that way: tools run on agent threads and must not wait on a conversation, and nothing may call back into the store from a store listener. A start the server refuses because it runs `max_concurrent_runs` conversations waits in `_when_there_is_room` (the node shows `waiting_to_start`) until a conversation stops running. `DirectorSettings` (`max_agents` for a team with every team below it, `max_depth`; `GET`/`PATCH /api/director/settings`, kept in `director.json`) are read only by `split.check_limits`, when a split is proposed and again when it is carried out, so a lowered limit refuses the next split and removes nothing. A node's `working`/`idle` status follows its agents' conversations through `ConversationService.add_execution_status_listener`; `done` and `blocked` are declared by the team. `save_org` leaves that running state out, so status flips do not become commits. A team is held to its boundary in code, not by its prompt. `DirectorService.guard_action` is asked before every action of every agent through `ConversationService.add_action_guard` and blocks a file tool (`director/boundary.py` lists them and the field that names the file) from writing outside the team's directory, into a sub-team's directory, or anything at all for a Manager; a new file-writing tool must be added to that list. Where bubblewrap works, `director/sandbox.py` also writes one shell script per agent into `<git dir>/director-shells/` and the agent's terminal is started with it (`tool_params` on `start_conversation`/`start_child_conversation`), so the shell's mount namespace has the org's checkouts read-only except the team's own directory and the git directory; `OrgSummary.boundary_enforcement` says which of the two holds. The script is read when a shell starts, so a split narrows an IC's shell only from its next start. `POST /api/director/org/messages` is the human's way in: it routes a message to the nearest team above a directory (`boundary.nearest_team`) and, without an org, takes up the one saved in the repository of the `conversation` it comes with or starts one there from the message. A client can therefore offer every new conversation's first message to it. Conversations the director starts itself (Managers and sub-team ICs, not the root IC) carry the tag `startedby=director` (`pods.STARTED_BY_DIRECTOR`, through `extra_tags` on `start_child_conversation`), which clients use to keep them out of the list of the human's own conversations. Team titles and `headline`s come from one separate model call per team (`director/headline.py`, through `PodLauncher.summarize`), never from the agents' own conversations, and a failed or slow call leaves the name the team has. A team's own `blocked` stays until the team declares otherwise; `OrgStore.set_agent` lifts only a block that came from a failed agent, when that agent recovers. A Manager removes a team directly below its own with `prune_team`: `DirectorService._prune` stops and deletes the conversations of the team and of every team below it, discards the uncommitted changes in the team's directory after saving them as a commit under `refs/agents/pruned/<node id>/<time>` (`git.save_and_discard`), appends an `Autopsy` to the boundary's `autopsies.json`, and either removes the node or, given a replacement task, starts a new IC and Manager on the same node. Autopsies of a boundary are in every briefing for it. Every IC's terminal is started through a script in `<git dir>/director-shells/`, confined or not, because that script also puts the `git` from `director/git_index.py` first on `PATH` (through a bash `--rcfile`, since `~/.bashrc` may reorder `PATH`). Inside the org's repository that `git` runs the real one on `<git dir>/agent-index-<conversation id>`, where everything outside the team's directory is reset to `HEAD`, marked skip-worktree and ignored, so `git add -A` and `git commit -a` take only the team's own files; it resyncs when `HEAD` or the team's sub-teams change, runs HEAD-moving commands one at a time under `flock`, and after a commit resets the committed paths in the checkout's own index. Do not give agents a bare `GIT_INDEX_FILE`: an index that is one commit behind reverts that commit when it is committed from. A team is told its own boundary's `spec.md` and two sets of contracts, nothing else of the teams above: `GitBlackboard.applies_to` returns the spec, the `contracts.md` of its own boundary (between its sub-teams) and the `contracts.md` of the boundary of the team above (between it and the teams beside it). The contracts of an approved split are written to `contracts.md`, not into the spec, so whatever a sub-team must respect has to be in them. Taught rules are still inherited from every boundary above. `GET /api/director/org/nodes/{id}/ownership` (`director/ownership.py`) lists the files of a node's checkout by the direct report that owns them and the files that conflict when the tips of the checkouts of the node and its reports are merged in memory with `git merge-tree --write-tree`; uncommitted changes are not compared. Work a team reports `done` (which `set_status` refuses while anything in its directory is uncommitted) is reviewed by the IC of the team above, not by a separate agent and not by a Manager: `DirectorService._request_review` first asks the team's own IC two questions on top of its conversation with `PodLauncher.ask_agent`, which records nothing in it: to write up what the team changed since `OrgNode.base_commit` as a pull request description, and to review it with OpenHands' public `/codereview` skill as instructions (`director/bugbot.py`; `find_review_instructions` is injected so tests never load public skills). No diff is sent to a model; the questions carry only git's list of changed files. Both answers are stored on `OrgNode.review` and sent to the IC above in a review request. A team that was `done` and gets a review request is `working` until it has decided. That IC decides with `review_team`. An accept merges the tip of a team with its own worktree into the checkout above (`director/merge_back.py`, under the lock the agents' `git` takes) and tells nobody further up unless the IC passed `needed_above`; work that conflicts or reaches outside the team's directory is sent back instead. A third rejection leaves the team `blocked` with its agents stopped and tells no Manager. `DirectorService.settled()` waits for effects that effects queue, so a test can assert right after it. The blackboard leaves the machine only when asked: `POST /api/director/org/blackboard/pull` fetches a remote's `refs/agents/blackboard` and merges it, and `.../push` pushes, pulling first only when the remote refuses the push (`director/blackboard_sync.py`). `GitBlackboard.merge` settles what both sides changed itself: rules and autopsies become the union, and of any other file the local text is kept and named in `kept_local`. Agents add rules to their own boundary with `record_rule`. A loop owned by `DirectorService` (`Config.director_consolidation_seconds`, 0 for off) has a model rewrite a boundary's rules into fewer (`director/memory.py`) only when it has more than 12 and a fifth more than at the last rewrite; `consolidated.json` beside `memory.json` records that state, so unchanged rules are never sent to a model again. A model call the director makes for itself, apart from any conversation, goes through `PodLauncher.ask` with its own `usage_id`.
- The director's unfinished work is kept outside git history, in the SQLite file `<git common dir>/director/state.db` (`director/run_state.py`): which agents were running, which teams wait for a run slot and in what order, the multi-step actions it began (intents) and the messages it has not delivered. A multi-step action goes through `DirectorService._begin`, which records it before its first step; `_carry_out` runs the `_do_<kind>` handler, which marks each finished step with `run.step_done` and skips marked steps, so every handler must be safe to run again. Messages to agents go through `_post`/`_send` (held, then delivered: at least once). `POST /api/director/org/pause` and `/resume` stop and continue all of it, `GET /org/recovery` says what a resume will do; a server that starts with unfinished work loads the org paused (`shutdown` or `crash`) unless `DirectorSettings.resume_on_start` is set. `director/recovery.py` drops run state that no longer fits the tree (another server's file, a team changed or removed since) and names what it dropped. Finished intents and delivered messages are deleted, so the file holds only what is open.
- An org is exchanged with a remote of its repository only when asked, as git does a branch: `POST /api/director/org/push` and `/org/pull` (`director/org_sync.py`) move `refs/agents/org`, `refs/agents/blackboard` and the branch of every team with a checkout of its own, kept on the remote as `refs/agents/teams/<node id>`. A pull is a merge (`director/org_merge.py`, a pure function over saved nodes): a team or a field changed on one side only is taken from that side; the same field changed on both, a team removed on one side and changed on the other, and a spec or contracts both rewrote in the same lines are conflicts. With conflicts nothing is written, the org stays paused, the merge is kept in the run state and `GET /org/merge`, `POST /org/merge/resolve` and `/org/merge/abort` settle it; finishing writes a commit with both parents. `worktree_path` and `agents` belong to the machine that runs a team and are never taken from the other side (`LOCAL_FIELDS`). Agents' conversations do not travel: after a pull into another clone every agent is `lost`, and `POST /org/nodes/{id}/start_again`, or a message that comes with a conversation, starts a team afresh in its own checkout.

- `LookupSecret` normalizes hostless URLs against `OH_INTERNAL_SERVER_URL` (set by `openhands-agent-server.__main__` from the bound host/port, rewriting wildcard binds to loopback) and otherwise falls back to `http://127.0.0.1:8000`, so relative secret URLs can safely target the current agent-server instance.




## Package-specific guidance
When reviewing or modifying code, read the closest AGENTS file for the
package(s) containing the changed files. If a PR spans multiple packages,
consult each relevant package-level AGENTS.md.

- SDK: [openhands-sdk/openhands/sdk/AGENTS.md](openhands-sdk/openhands/sdk/AGENTS.md)
- Subagents: [openhands-sdk/openhands/sdk/subagent/AGENTS.md](openhands-sdk/openhands/sdk/subagent/AGENTS.md)
- Tools: [openhands-tools/openhands/tools/AGENTS.md](openhands-tools/openhands/tools/AGENTS.md)
- Workspace: [openhands-workspace/openhands/workspace/AGENTS.md](openhands-workspace/openhands/workspace/AGENTS.md)
- Agent server: [openhands-agent-server/AGENTS.md](openhands-agent-server/AGENTS.md)

## API compatibility pointers

- For SDK Python API deprecation/removal policy, read
  [openhands-sdk/openhands/sdk/AGENTS.md](openhands-sdk/openhands/sdk/AGENTS.md).
  Public API removals require deprecation metadata with a removal target at
  least **5 minor releases** after `deprecated_in`, and breaking SDK API
  changes require at least a **MINOR** SemVer bump.
- The SDK API breakage checker should treat metadata-only changes to
  Pydantic `Field(...)` declarations as non-breaking, including adding,
  removing, or editing `description`, `title`, `examples`,
  `json_schema_extra`, and `deprecated` kwargs.
- Public SDK `Field(default=...)` changes are treated separately from removals/structural API breakages: the API breakage workflow should surface them as behavioral compatibility changes, auto-apply the green `release-note-required` label on PRs, and the release workflow should prepend those labeled PRs to generated GitHub release notes.
- For public SDK `Field(default=...)` changes, keep two views in the API breakage workflow: compare against the latest released PyPI baseline for compatibility reporting, but compare against the PR base ref before syncing the `release-note-required` label or PR comment so unrelated follow-up PRs are not re-labeled for already-merged unreleased defaults.


- The SDK API breakage checker compares stringified `Field(...)` values by
  parsing them as Python expressions after escaping literal newlines inside
  quoted strings; this avoids false positives on multiline descriptions that
  include embedded quotes like `'security_policy.j2'`.
- For public REST APIs, read
  [openhands-agent-server/AGENTS.md](openhands-agent-server/AGENTS.md).
  REST contract breaks need a deprecation notice and a runway of
  **5 minor releases** before removing the old contract or making an
  incompatible replacement mandatory.

<DEV_SETUP>
- Make sure you `make build` to configure the dependencies first
- We use pre-commit hooks `.pre-commit-config.yaml` that includes:
  - type check through pyright
  - linting and formatting with ruff (`ruff-check`, `ruff-format`) plus `pycodestyle`
  - `yamlfmt` for YAML
  - repo-specific gates: `check-import-rules`, `check-tool-registration`
- NEVER USE `mypy`!
- Do NOT commit ALL the file, just commit the relevant file you've changed!
- In every commit message, you should add "Co-authored-by: openhands <openhands@all-hands.dev>"
- You can run pytest with `uv run pytest`

# Instruction for fixing "E501 Line too long"

- If it is just code, you can modify it so it spans multiple lines.
- If it is a single-line string, you can break it into a multi-line string by doing "ABC" -> ("A"\n"B"\n"C")
- If it is a long multi-line string (e.g., docstring), add `# noqa: E501` AFTER the ending """. You should NEVER ADD IT INSIDE the docstring. E501 is a ruff lint rule, so `# noqa: E501` is what suppresses it — `# type: ignore` is a pyright directive and has no effect on E501.


</DEV_SETUP>

<PR_ARTIFACTS>
# PR-Specific Evidence Documents

The `.pr/` directory is intentionally temporary by repository policy: it holds
PR-only reviewer context. It stays on the PR branch through review and on
`main` after merge, until the next release PR removes it:
`.github/workflows/prepare-release.yml` deletes the tracked root `.pr/`
directory in the same commit as the version changes.

When working on a PR that requires design documents, scripts meant for development-only, or other temporary artifacts that should not be kept long-term, store them in a `.pr/` directory at the repository root.

You can also use the `.pr/` directory for evidence, such as logs, live-tests, live-test summary files.

## Usage

```bash
# Create the directory if it doesn't exist
mkdir -p .pr

# Add your PR-specific documents
.pr/
├── design.md       # Design decisions and architecture notes
├── analysis.md     # Investigation or debugging notes
└── notes.md        # Any other PR-specific content
```

## How It Works

1. **Notification**: When a PR adds or changes files under `.pr/`, the `PR Artifacts` workflow (`.github/workflows/pr-artifacts.yml`) posts a single comment to the PR conversation alerting reviewers
2. **Release cleanup**: `.pr/` accumulates on `main` until the next release PR removes it; release PR checks reject any `.pr/` files that remain

## Important Notes

- Do NOT put anything in `.pr/` that needs to be preserved
- The `.pr/` check passes (green ✅) during development - it only posts a notification, not a blocking error
- Do not open separate PRs to remove `.pr/` from `main`; release preparation removes it

## When to Use

- Complex refactoring that benefits from written design rationale
- Debugging sessions where you want to document your investigation
- Feature implementations that need temporary planning docs
- Temporary script that are intended to show reviewers that the feature works
- Any analysis that helps reviewers understand the PR but isn't needed long-term
</PR_ARTIFACTS>

<PR_DESCRIPTION_HUMAN_CHECK>
# Human-only PR description fields in PR template

When opening a PR, use the repository's PR template for the description. The
`HUMAN:` section in PR descriptions is reserved for human contributors only. AI
agents MUST NOT edit, move, or remove this field, only set the placeholder. If
the PR description CI fails because this field is missing
or empty, stop and ask the human user to update it in their own words. If
the field was already updated by a human, report the exact validator error rather
than editing it yourself.
</PR_DESCRIPTION_HUMAN_CHECK>


<REVIEW_HANDLING>
- Critically evaluate each review comment before acting on it. Not all feedback is worth implementing:
  - Does it fix a real bug or improve clarity significantly?
  - Does it align with the project's engineering principles (simplicity, maintainability)?
  - Is the suggested change proportional to the benefit, or does it add unnecessary complexity?
- It's acceptable to respectfully decline suggestions that add verbosity without clear benefit, over-engineer for hypothetical edge cases, or contradict the project's pragmatic approach.
- After addressing (or deciding not to address) inline review comments, mark the corresponding review threads as resolved.
- Before resolving a thread, leave a reply comment that either explains the reason for dismissing the feedback or references the specific commit (e.g., commit SHA) that addressed the issue.
- Prefer resolving threads only once fixes are pushed or a clear decision is documented.
- Use the GitHub GraphQL API to reply to and resolve review threads (see below).

## Resolving Review Threads via GraphQL

The CI check `Review Thread Gate/unresolved-review-threads` will fail if there are unresolved review threads. To resolve threads programmatically:

1. Get the thread IDs (replace `<OWNER>`, `<REPO>`, `<PR_NUMBER>`):
```bash
gh api graphql -f query='
{
  repository(owner: "<OWNER>", name: "<REPO>") {
    pullRequest(number: <PR_NUMBER>) {
      reviewThreads(first: 20) {
        nodes {
          id
          isResolved
          comments(first: 1) {
            nodes { body }
          }
        }
      }
    }
  }
}'
```

2. Reply to the thread explaining how the feedback was addressed:
```bash
gh api graphql -f query='
mutation {
  addPullRequestReviewThreadReply(input: {
    pullRequestReviewThreadId: "<THREAD_ID>"
    body: "Fixed in <COMMIT_SHA>"
  }) {
    comment { id }
  }
}'
```

3. Resolve the thread:
```bash
gh api graphql -f query='
mutation {
  resolveReviewThread(input: {threadId: "<THREAD_ID>"}) {
    thread { isResolved }
  }
}'
```

4. Get the failed workflow run ID and rerun it:
```bash
# Find the run ID from the failed check URL, or use:
gh run list --repo <OWNER>/<REPO> --branch <BRANCH> --limit 5

# Rerun failed jobs
gh run rerun <RUN_ID> --repo <OWNER>/<REPO> --failed
```
</REVIEW_HANDLING>


<CODE>
- Avoid hacky trick like `sys.path.insert` when resolving package dependency
- Use existing packages/libraries instead of implementing yourselves whenever possible.
- Avoid using # type: ignore. Treat it only as a last resort. In most cases, issues should be resolved by improving type annotations, adding assertions, or adjusting code/tests—rather than silencing the type checker.
  - Please AVOID using # type: ignore[attr-defined] unless absolutely necessary. If the issue can be addressed by adding a few extra assert statements to verify types, prefer that approach instead!
  - For issue like # type: ignore[call-arg]: if you discover that the argument doesn’t actually exist, do not try to mock it again in tests. Instead, simply remove it.
- Avoid doing in-line imports unless absolutely necessary (e.g., circular dependency).
- Avoid getattr/hasattr guards and instead enforce type correctness by relying on explicit type assertions and proper object usage, ensuring functions only receive the expected Pydantic models or typed inputs. Prefer type hints and validated models over runtime shape checks.
- Prefer accessing typed attributes directly. If necessary, convert inputs up front into a canonical shape; avoid purely hypothetical fallbacks.
- Use real newlines in commit messages; do not write literal "\n".
- Comments policy: write comments sparingly and strategically. Prefer making the code self-explanatory through clear naming and structure over adding prose.
  - Do NOT add comments that restate what the code already says, summarize the surrounding diff/PR, or narrate the change history ("previously we did X, now we do Y"). That kind of context belongs in the PR description or commit message, not in the source — `git blame` and the PR are the source of truth for *why* a change was made.
  - Do NOT describe non-local parts of the system (other modules, callers, downstream behavior) in a comment unless there is a mechanism to keep that description in sync. Such comments drift and become misleading.
  - DO add a comment when the code expresses something genuinely unintuitive: a non-obvious invariant, a workaround for an external bug, a subtle ordering/locking requirement, or a deliberate trade-off the reader cannot infer from the code itself.
  - When in doubt, prefer restructuring or renaming over commenting. A 3-line change should not produce 19 lines of comments — if it feels like it needs that much narration, the explanation belongs in the PR description.

</CODE>

<TESTING>
- AFTER you edit ONE file, you should run pre-commit hook on that file via `uv run pre-commit run --files [filepath]` to make sure you didn't break it.
- For each behavior you add or change, test at least one failure or edge input it must handle (invalid or missing input, legacy or persisted data, an error from a dependency, cancellation), not only the success path. Keep it focused: one test per case is enough.
- Take expected values from the issue, the spec, or the documented contract, not from what the code returns today. A test whose expectation was copied from the implementation passes on the bug.
- For a bug fix, the regression test must fail without the fix.
- Check how we perform tests in .github/workflows/tests.yml
- Put unit tests under the corresponding domain folder in `tests/` (e.g., `tests/sdk`, `tests/tools`, `tests/workspace`). For example, changes to `openhands-sdk/openhands/sdk/tool/tool.py` should be covered in `tests/sdk/tool/test_tool.py`.
- DON'T write TEST CLASSES unless absolutely necessary!
- If you find yourself duplicating logics in preparing mocks, loading data etc, these logic should be fixtures in conftest.py!
- Please test only the logic implemented in the current codebase. Do not test functionality (e.g., BaseModel.model_dumps()) that is not implemented in this repository.
- Assert observable behavior rather than source text, static implementation lists, private helpers or state, generic framework behavior, exhaustive default/export mirrors, or mock wiring. Tests should survive behavior-preserving refactors.
- For changes to prompt templates, tool descriptions, or agent decision logic, add the `integration-test` label to trigger integration tests and verify no unexpected impact on benchmark performance.

# Stress Tests

`tests/agent_server/stress/` contains an opt-in stress/scale suite for the agent-server, excluded from default collection via the `stress` pytest marker. Run with `uv run pytest -m stress`. For full details on running, infrastructure, and adding new stress tests, see [openhands-agent-server/AGENTS.md](openhands-agent-server/AGENTS.md).

# Behavior Tests

Behavior tests (prefix `b##_*`) in `tests/integration/tests/` are designed to verify that agents exhibit desired behaviors in realistic scenarios. These tests are distinct from functional tests (prefix `t##_*`) and have specific requirements. The same directory also holds `a##_*` (malformed tool-use/tool-result history) and `c##_*` (condenser) tests; only the `b`/`t` prefixes drive the behavior-vs-integration classification described in `BEHAVIOR_TESTS.md`.

Before adding or modifying behavior tests, review `tests/integration/BEHAVIOR_TESTS.md` for the latest workflow, expectations, and examples.
</TESTING>

<AGENT_TMP_DIRECTORY>
# Agent Temporary Directory Convention

When tools need to store observation files (e.g., browser session recordings, task tracker data), use `.agent_tmp` as the directory name for consistency.

The browser session recording tool saves recordings to `.agent_tmp/browser_observations/recording-{timestamp}/` (see `BROWSER_RECORDING_OUTPUT_DIR` in `openhands-tools/openhands/tools/browser_use/definition.py`).

This convention ensures tool-generated observation files are stored in a predictable location that can be easily:
- Added to `.gitignore`
- Cleaned up after agent sessions
- Identified as agent-generated artifacts

Note: This is separate from `persistence_dir` which is used for conversation state persistence.
</AGENT_TMP_DIRECTORY>

<REPO>
<PROJECT_STRUCTURE>
- This is a monorepo with a `uv`-managed Python workspace (single `uv.lock` at repo root) and a separately managed npm package under `clients/typescript/`. The Python distributable packages are: `openhands-sdk/` (SDK), `openhands-tools/` (built-in tools), `openhands-workspace/` (workspace impls), and `openhands-agent-server/` (server runtime).
- `clients/typescript/` contains the browser-compatible TypeScript client, with its own `package-lock.json`, npm scripts, and Vitest tests.
- `examples/` contains runnable patterns; `tests/` is split by domain (`tests/sdk`, `tests/tools`, `tests/workspace`, `tests/agent_server`, etc.).
- Python namespace is `openhands.*` across packages; keep new modules within the matching package and mirror test paths under `tests/`.
</PROJECT_STRUCTURE>

<QUICK_COMMANDS>
- Set up the dev environment: `make build` (runs `uv sync --dev` and installs pre-commit; requires uv >= 0.8.13)
- Lint/format: `make lint`, `make format`
- Run Python tests: `uv run pytest`
- Set up and validate the TypeScript client: `cd clients/typescript && npm ci && npm run lint && npm run build && npm test`
- Run agent-server stress tests: `uv run pytest -m stress` (see [openhands-agent-server/AGENTS.md](openhands-agent-server/AGENTS.md))
- Build agent-server: `make build-server` (output: `dist/agent-server/`)
- Clean caches: `make clean`
- Run SDK examples: see [openhands-sdk/openhands/sdk/AGENTS.md](openhands-sdk/openhands/sdk/AGENTS.md).
- The example workflow runs `uv run pytest tests/examples/test_examples.py --run-examples`; each successful example must print an `EXAMPLE_COST: ...` line to stdout (use `EXAMPLE_COST: 0` for non-LLM examples).
- Linear walkthroughs in `examples/` should use top-level code flow (e.g. `with` blocks, bare statements) rather than wrapping the whole example in `def main()`. CLI-style examples with argument-driven branches may use a `main()` entrypoint.
- Conversation plugins passed via `plugins=[...]` are lazy-loaded on the first `send_message()` or `run()`, so example code should inspect plugin-added skills or `resolved_plugins` only after that first interaction.
</QUICK_COMMANDS>

<REPO_CONFIG_NOTES>
- Ruff: `line-length = 88`, `target-version = "py313"` (see `pyproject.toml`).
- Ruff ignores `ARG` (unused arguments) under `tests/**/*.py` to allow pytest fixtures.
- Repository guidance lives in the project root AGENTS.md (loaded as a third-party skill file).
</REPO_CONFIG_NOTES>

<KNOWN_RACES_AND_GOTCHAS>
- **`RemoteConversation._wait_for_run_completion` and stop hooks**: Per-field WebSocket `FINISHED` status events are *hints*, not authoritative termination. The server-side `LocalConversation.run` loop releases its state lock at the end of each iteration, so a `FINISHED` status set by `agent.step()` is visible to clients before the *next* loop iteration runs stop hooks (`hook_processor.run_stop`). If a stop hook returns rc=2 (denying the stop), status flips back to RUNNING and the agent gets another iteration. The client's `_wait_for_run_completion` therefore must **not** return on the first WS-delivered FINISHED. Instead, post-run full-state WebSocket snapshots are authoritative; if that snapshot is missing, the time-based hard-fallback path (`TERMINAL_HARD_FALLBACK_SECS = 30.0`) accepts REST-confirmed terminal status after 30 continuous seconds. ERROR/STUCK still raise immediately through `_handle_conversation_status`. Empirically this caused agents to consume just 0–1 iterations after a hook block on programbench retry-16; fix shipped in `feat/programbench`.
- **Hook events vs `state.events`**: `HookExecutionEvent` is emitted via `hook_processor.original_callback` (the chained `_on_event`), so it *should* land in `state.events` when the run is allowed to complete. But because the WS-FINISHED race above used to make the client snapshot `list(conversation.state.events)` *before* the server-side hook eval ran, `output.jsonl` history could miss hook events while on-disk persisted events under `/workspace/conversations/.../events/` had them — useful as a forensic signal that the race fired.
</KNOWN_RACES_AND_GOTCHAS>

</REPO>

## Dependency release-age policy

- Wait at least seven days after publication before adding or updating a
  third-party npm or PyPI version, including transitive lockfile changes.
  A weekly update schedule alone does not enforce this minimum age.
- Keep Dependabot `cooldown.default-days: 7` for npm and Python version
  updates, and preserve `tool.uv.exclude-newer = "7 days"` where configured.
  Existing named first-party exceptions do not exempt their third-party
  dependencies. Do not add broad exemptions or disable the cutoff to make a
  resolver succeed.
- Verify the exact version's publication/upload time from npm or PyPI metadata
  (or artifact `upload-time` in `uv.lock`), not the package's creation date.
  If age cannot be established, report the missing evidence rather than guess.
- Security updates can bypass Dependabot cooldown. An urgent exception needs
  explicit maintainer approval recorded in the PR with the package, version,
  advisory/reason, and publication time. Agents must not grant themselves an
  exception.
- Use the committed lock for reproducible installs; review newly resolved
  artifacts before committing a lock update.
