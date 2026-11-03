# Local Chat Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已验证的 G6＋新清晰度路由接入正式入口，提供本地单用户 React＋FastAPI 聊天工作台。

**Architecture:** 复用 ChatService、LangGraph、MemoryManager 和现有 RAG。FastAPI 管理持久化会话、幂等受管任务与公开 SSE；React 只读取公开视图，不持有模型密钥。采用阶段更新＋最终回答，不改造模型 token 流式接口。

**Tech Stack:** Python 3.12、unittest、FastAPI、Uvicorn、httpx、SQLite；React、TypeScript、Vite、Vitest、Testing Library、react-markdown。

**Spec:** docs/superpowers/specs/2026-10-22-local-chat-workbench-design.md

## Global Constraints

- 默认后端仅监听 `127.0.0.1:8000`，开发前端为 `127.0.0.1:5173`，通过 Vite 代理 `/api`。
- 第一版只支持单 worker；服务最多允许两个会话任务同时运行；同一会话同一时刻最多一个运行中的轮次。不无限排队。
- 稳定配置 `routing-g6-clarity-v3`；阈值保持清晰度 0.75、业务相关性 0.8，6/5 分批方式保持不变。
- 最近五轮完整用户问题和最终回答，不含工具轨迹或专业 Agent 私有执行记录；不采用 `business_probe.CONTRASTS`。
- 消息非空，沿用 16000 字符上限；发送体只接受 `message` 与 `turn_id`，拒绝额外字段；浏览器不能选择身份。
- demo 不调用外部模型，界面醒目标识为离线演示；live 配置错误不静默降级为 demo。启动不产生模型费用或重建索引。
- 不修改历史评测数据、已有原文、索引、数据库内容和密钥；仅做新增兼容表及指定非敏感配置更新。
- 不展示内部思维链，不提供登录、上传、管理后台、批量删除、公网发布或付费自动重试。
- 未经另行确认不做真实模型冒烟；全部默认验证离线运行。本目录不是 Git 仓库，采用文件检查点，不初始化仓库、不提交到父目录。

## Review Focus

1. POST 响应丢失后同 ID 重发、SSE 重连或刷新：只执行一次；同 ID 换内容必须 409（任务 4、5、6）。
2. 两个页面同时发送、第三会话发送、取消或持久化失败：容量不泄漏，不把未落库回答标成完成（任务 4、5）。
3. 服务重启时遗留 running 记录：显示 interrupted，不重放工具；demo 与 live 数据隔离（任务 2、3、4）。
4. 恶意 Markdown、伪造身份、跨站写请求、证据磁盘路径：禁止泄露或执行（任务 3、5、7）。
5. 中文输入法 Enter、窄屏、切换会话时旧 SSE 回调：不误发、不串会话，保留用户输入与部分结果（任务 6、7、8）。

## File Map

- `intent/stable_routing_prompts.py`：稳定提示定义；`intent/jev.py` 和 `intent/ensemble.py`：显式 profile 传递与校验。
- `runtime/factory.py`、`runtime/demo.py`：共享构建/关闭与离线依赖；`scripts/chat.py`：复用工厂。
- `web/contracts.py`、`web/public_views.py`：浏览器 DTO 与公开字段过滤；`web/store.py`：会话、任务元数据。
- `runtime/memory.py`：在既有 SQLite 锁下提供安全查询/事务入口，保留已有记录语义。
- `web/jobs.py`：任务幂等、容量、执行与订阅；`web/app.py`：API、安全中间件、生命周期、静态目录。
- `scripts/serve.py`：显式 demo/live 的单 worker 本地启动；`environment.yml`：后端依赖。
- `frontend/`：React UI、API 类型、状态恢复、样式、测试和生产构建。
- `tests/test_stable_routing_profile.py`、`test_runtime_factory.py`、`test_web_store.py`、`test_web_jobs.py`、`test_web_api.py`：离线验收。
- `docs/local-chat-workbench.md`：启动、模式、限制；`docs/superpowers/plans/2026-10-22-local-chat-workbench-progress.md`：任务证据。

以下 `src/` 路径均位于 `src/submission_assistant/`；命令在项目根目录执行。Python 测试使用现有 `.venv-evaluation/Scripts/python.exe -X utf8`（下文记作 `PY`，执行时用完整命令替换，不新增环境变量）。npm 当前不在 PATH：实施时定位可用 npm CLI，找不到则安装/补齐包管理器后记录准确路径，不把 Node 可用当作 npm 可用。

### Task 1: Promote the frozen stable routing profile

**Files:** Create `src/submission_assistant/intent/stable_routing_prompts.py`, `tests/test_stable_routing_profile.py`; modify `intent/jev.py`, `intent/ensemble.py` at request construction/prevalidation only.

**Interfaces:** Consumes existing `build_requests(message, history=None, *, model, provider, prompt_version)`; produces `STABLE_ROUTING_PROMPT_VERSION = 'routing-g6-clarity-v3'` and `stable_questions() -> dict[str, dict]`. Add `prompt_version: str = 'baseline'` to `JevIntentRecognizer.__init__`, store it and use it in actual requests and ensemble prevalidation.

- [x] Write failing unittest tests `test_stable_payload_matches_frozen_candidate`, `test_stable_history_keeps_five_complete_pairs`, `test_legacy_profiles_unchanged`: assert both request bodies equal saved successful candidate fields, question counts `[6, 5]`, thresholds unchanged, 12 history messages reduce to last 10 without private messages. Snapshot source: `evaluation/agent/clarity-probes/20261022-paired-01/plan.json` and `cases.json`; inspect their actual structure, select a fixed case and compare full bodies, not only text fragments.
- [x] RED: `PY -m unittest discover -s tests -p test_stable_routing_profile.py -v`; expect missing stable profile failures before product edits.

  Core assertions: `self.assertEqual(actual_bodies, frozen_candidate_bodies)`; `self.assertEqual([len(b['questions']) for b in actual_bodies], [6, 5])`. Use actual snapshot/provider field names if its batch body wraps questions; compare the full serialized body after only documented model normalization.
- [x] Extract exact successful definitions from `evaluation/agent/ablation_design.py` G6 and `clarity_probe.py` CANDIDATE into the production module; allowlist the new profile, preserve old defaults and options. Application imports must never reference `evaluation/`; legacy tests remain meaningful.
- [x] GREEN: run new tests plus `test_decisions_provider.py`, `test_routing_prompt_examples.py`, `test_routing_history_boundary.py` separately using discover patterns. Check the prevalidation and request paths select the same profile.
- [x] Record changed files and actual commands/results in progress document; do not rerun paid evaluations.

### Task 2: Share runtime construction and lifecycle

**Files:** Create `runtime/factory.py`, `runtime/demo.py`, `tests/test_runtime_factory.py`; modify `scripts/chat.py`, `.env` (only three approved nonsecret values), `environment.yml` when API dependency is introduced in task 5.

**Interfaces:** Produces `RuntimeBundle(service: ChatService, mode: Literal['demo','live'], prompt_version: str)` with `async aclose() -> None`; `async create_runtime(project: Path, *, mode: Literal['demo','live'], data_dir: Path | None = None, no_multi_query: bool = False) -> RuntimeBundle`. CLI explicitly selects stable profile in live. Web chooses separate persistent demo data directory; CLI demo retains temporary-directory behavior.

- [x] Tests: `test_demo_never_opens_network` uses patched external clients and asserts call count `0`; `test_live_factory_uses_stable_profile` inspects mocked recognizer arguments; `test_invalid_live_does_not_become_demo` asserts a sanitized configuration exception; `test_aclose_orders_shutdown` asserts service drain before SQLite/client/Redis close and idempotent double-close. Also assert demo storage path differs from `data/memory/memory.sqlite3`.
- [x] RED: `PY -m unittest discover -s tests -p test_runtime_factory.py -v`; expect unavailable factory failures.

  Core assertions: `self.assertEqual(external_send.call_count, 0)`; `self.assertEqual(close_order, ['service', 'memory', 'http', 'redis'])` for the all-resources-owned fixture; `self.assertNotEqual(demo_db, live_db)`.
- [x] Extract existing wiring with no retrieval-policy changes. Validate local configuration and existing index paths without sending requests. Inspect HTTP ownership in provider/backend and close only owned resources; unwind partially constructed resources on startup failure. Expose sanitized dependency flags, not settings objects.
- [x] Change only `JEV_PROVIDER=decisions`, `JEV_BASE_URL=https://api.perplexity.ai/v1/decisions`, `JEV_MODEL=pplx-decider-flash-v1.2`; preserve key values, encoding and unrelated lines. Test config rewrite with a temporary fixture and assert secret sentinel unchanged; never print actual `.env`.
- [x] GREEN: factory tests and `PY scripts/chat.py --demo`; expect clearly labeled offline final answer and clean exit, no external requests. Record checkpoint.

### Task 3: Persist conversations and safe public views

**Files:** Create `web/__init__.py`, `web/contracts.py`, `web/public_views.py`, `web/store.py`, `tests/test_web_store.py`; extend `runtime/memory.py` with scoped query/transaction methods rather than opening an unsynchronized second writer.

**Interfaces:** Public DTOs: `ConversationSummary(id: str, title: str, updated_at: str)`, `TurnView(turn_id: str, message: str, state: TurnState, answer: str | None, routing: dict | None, agent_results: list[dict], status: list[dict], error_code: str | None)`, `ConversationView(id: str, title: str, turns: list[TurnView])`. `TurnState` is Literal `pending|running|completed|incomplete|failed|cancelled|interrupted`. `TurnPatch` has optional state/routing/agent_results/status/error_code fields, distinguishes unset from null and cannot write the core answer. `ConversationStore(memory, user_id: str)` supplies `create() -> ConversationSummary`, `list() -> list[ConversationSummary]`, `get(conv_id: str) -> ConversationView`, `get_turn(conv_id: str, turn_id: str) -> TurnView`, `reserve(conv_id: str, turn_id: str, message: str) -> tuple[TurnView, bool]`, `update_public(conv_id: str, turn_id: str, patch: TurnPatch) -> TurnView`, `interrupt_pending() -> int`. Reserve's bool means newly created. Missing resources raise `ResourceNotFound`; ID/message mismatch raises `TurnConflict` (both defined in `web/store.py`).

- [x] Tests `test_scoped_conversations_and_legacy_history`, `test_empty_conversation_and_title`, `test_id_collision`, `test_public_view_redacts_private_records`, `test_restart_marks_only_local_jobs_interrupted`: assert other-user IDs return not found; fresh title is “新会话”; first-message title deterministically uses at most 30 characters; old dialogue remains unchanged; private execution/prompt/secret sentinels and filesystem paths absent from serialized DTOs.
- [x] RED: `PY -m unittest discover -s tests -p test_web_store.py -v`; expect missing store/DTO failures.

  Core assertions: `self.assertEqual(store.create().title, '新会话')`; `self.assertLessEqual(len(view.title), 30)`; `self.assertNotIn('PRIVATE_TRACE_SENTINEL', view.model_dump_json())`; `with self.assertRaises(ResourceNotFound): other_user_store.get(conv_id)`.
- [x] Add `web_conversations` and `web_turns` metadata tables keyed by trusted user/conv/turn using the existing database lock and transactions. Keep core `turns`/dialogue as answer source of truth; store only public routing/status/result metadata needed across refresh. Import existing local conversations lazily without modifying their raw records. `reserve` registers web metadata, not a second core `begin_turn` or `claim_turn` implementation.
- [x] Implement `public_result(raw: dict) -> dict` and `public_status(raw: dict) -> dict` with explicit allowlists and safe HTTP/HTTPS source validation. Results expose answer, safe evidence, role, intent IDs, terminal state/reason and unresolved items; never whole graph state or raw error objects. Conversation detail displays full public history, not the five-pair prompt window.
- [x] GREEN: store tests plus existing memory/chat tests; verify failed, empty and interrupted turns stay visible and are not fabricated assistant answers. Record checkpoint.

### Task 4: Manage idempotent jobs and real stage events

**Files:** Create `web/jobs.py`, `tests/test_web_jobs.py`; minimally adjust `runtime/chat.py` only if a public event hook is necessary (prefer consuming LangGraph update events).

**Interfaces:** `JobController(bundle: RuntimeBundle, store: ConversationStore, *, max_active: int = 2)`. Produces `async submit(conv_id: str, turn_id: str, message: str) -> TurnView`, `get(conv_id: str, turn_id: str) -> TurnView`, `async events(conv_id: str, turn_id: str) -> AsyncIterator[dict]`, `async aclose() -> None`; defines `ConversationBusy` and `CapacityBusy`. Event envelopes `{type: 'stage'|'result'|'error', turn: TurnView}` are snapshots, not token deltas. New subscribers immediately receive current snapshot; terminal snapshot ends stream.

- [x] Tests `test_duplicate_post_runs_once`, `test_same_id_different_message_conflicts`, `test_parallel_capacity_and_release`, `test_disconnect_does_not_cancel_job`, `test_finish_requires_persistence`, `test_restart_does_not_replay`, `test_slow_subscriber_is_bounded`: assert graph execution counts `1`, same conv second ID conflicts, two distinct conv jobs start, third is rejected, completion/failure/cancellation release slots. Persist-failure must not produce a completed event.
- [x] RED: `PY -m unittest discover -s tests -p test_web_jobs.py -v`; expect missing controller failures.

  Core assertions after duplicate submit/reconnect: `self.assertEqual(fake_graph.calls, 1)`; third concurrent conversation uses `with self.assertRaises(CapacityBusy)`; persistence-failure snapshots use `self.assertNotIn('completed', [e['turn'].state for e in seen])`.
- [x] Guard reserve/capacity/task creation with one asyncio lock; check existing ID before capacity, no await between accepted reservation and task registration. Invoke the existing graph once with a backend-created `Request(user_id=store.user_id, ...)`; consume public node updates, persist filtered metadata, then publish. Capture live status via scoped safe status reads; never relay raw updates. Terminal answer must already be committed by graph; catch faults into sanitized failed state without replay.
- [x] Use bounded snapshot queues (one latest snapshot per subscriber), heartbeat on idle connections, no unbounded event log. Browser disconnect removes only its subscription. On shutdown reject new jobs, cancel/drain tasks in a bounded period and mark leftovers interrupted; do not close memory ahead of tasks.
- [x] GREEN: jobs tests using controlled fake graph barriers and temporary SQLite; assert no leaked tasks/subscribers after teardown. Record checkpoint.

### Task 5: FastAPI API, security and local launcher

**Files:** Create `web/app.py`, `scripts/serve.py`, `tests/test_web_api.py`; modify `environment.yml` to add `fastapi>=0.115,<1`, `uvicorn>=0.30,<1`; preserve existing dependency constraints.

**Interfaces:** `create_app(project: Path, *, mode: Literal['demo','live'], port: int = 8000, runtime_factory=create_runtime) -> FastAPI`. Lifespan owns bundle/store/jobs; trusted identity fixed to `local-user`. Launcher accepts required mutually exclusive `--demo|--live`, optional `--port` (default 8000), binds only `127.0.0.1`, forces one worker. Implements the seven API endpoints in the spec verbatim.

- [x] Install declared API dependencies in the existing environment; do not upgrade unrelated packages. Tests `test_request_validation_and_ownership`, `test_idempotent_post_and_reconnect`, `test_busy_status_codes`, `test_host_origin_and_static_boundary`, `test_public_errors_and_mode`: assert blank/16001/extra user_id ->422, unknown ->404, mismatched or same-conv conflict ->409, capacity ->429; exact duplicate ->existing turn; valid 16000-char input accepted. Count fake graph calls across multiple SSE subscriptions and GETs: `1`.
- [x] RED: `PY -m unittest discover -s tests -p test_web_api.py -v`; expect absent routes/app failures, not missing test dependencies.

  Core assertions: `self.assertEqual(extra_identity.status_code, 422)`; `self.assertEqual(conflict.status_code, 409)`; `self.assertEqual(over_capacity.status_code, 429)`; `self.assertEqual(private_file.status_code, 404)`; `self.assertNotIn('SECRET_SENTINEL', response.text)`.
- [x] Strict Pydantic send body; validate turn IDs with existing identifier constraints. Write policy permits `http://127.0.0.1:{port}` and development proxy origin `http://127.0.0.1:5173`, plus absent Origin for local CLI; reject unrelated Origin and Host, no wildcard CORS. Vite proxy preserves backend Host. Use normal errors with stable codes, never remote bodies. API methods derive identity solely server-side. Health exposes mode/profile/dependency availability flags only, with “未探测” rather than false claims of live reachability.
- [x] Add SSE `text/event-stream`, snapshot events and heartbeat; no implicit execution on GET. Only serve `frontend/dist` assets when present; unknown `/api/*` remains 404, not SPA fallback. Traversal, `.env`, SQLite and source files return404. Use explicit routes for SPA entry and legitimate assets, not arbitrary project serving.
- [x] GREEN: API tests via httpx ASGI transport and explicit lifespan; launch demo and verify HTTP health, POST, history and SSE with an offline client. Assert no external network calls. Record checkpoint.

### Task 6: React client and recoverable conversation state

**Files:** Create `frontend/package.json`, lockfile, `index.html`, `tsconfig*.json`, `vite.config.ts`, `src/main.tsx`, `src/api.ts`, `src/types.ts`, `src/useWorkbench.ts`, `src/useWorkbench.test.tsx`, `src/test/setup.ts`; create/update project `.gitignore` to ignore only generated dependencies/build caches, never user content.

**Interfaces:** TypeScript DTOs mirror task 3. API exports `listConversations()`, `createConversation()`, `getConversation(id)`, `submitTurn(id, {message, turn_id})`, `getTurn(id, turnId)`, `subscribeTurn(id, turnId, onSnapshot, onError): () => void`. Hook `useWorkbench()` exposes conversations, selectedId, turns, activeTurn, draft, error and create/select/send/setDraft actions.

- [x] Obtain npm CLI without altering global policy; use React/ReactDOM, TypeScript, Vite with React plugin, Vitest/jsdom/Testing Library and react-markdown; lock resolved dependencies. Scripts: `dev`, `test` (vitest run), `build` (typecheck then vite build). Proxy `/api` to127.0.0.1:8000 and listen127.0.0.1:5173 with strictPort. API base remains same-origin, no credentials in Vite env.
- [x] Tests `reconnect_does_not_post_again`, `lost_post_response_keeps_turn_id`, `switch_ignores_stale_event`, `reload_restores_running_turn`, `server_failure_preserves_draft`: assert explicit retry of uncertain POST reuses identical ID/message; editing draft creates a new turn only after prior status resolved; event for old conv cannot overwrite current; only one POST for initial send and no POST on mount/reconnect. Generate IDs with `crypto.randomUUID()`.
- [x] RED: `npm run test -- --reporter=verbose` in frontend; expect missing client/hook behavior failures. Use located npm executable if not on PATH; record its real command.

  Core assertions: `expect(submitTurn).toHaveBeenCalledTimes(1)` after reconnect; `expect(retryPayload.turn_id).toBe(initialPayload.turn_id)`; `expect(result.current.selectedId).toBe(newConversation.id)` after an old subscription callback.
- [x] Implement bounded EventSource lifecycle and GET recovery. Persist only selected conversation and pending conv/turn/message as needed for ambiguous response recovery; no model settings or keys. Clear pending after terminal confirmation. Surface failures without automatic paid resubmission; unsubscribe on switch/unmount and ignore stale async results.
- [x] GREEN: frontend unit tests and `npm run build`; confirm DTO and event types align with API. Record checkpoint.

### Task 7: Chat workbench UI and safe evidence display

**Files:** Create `frontend/src/App.tsx`, `components/ConversationList.tsx`, `ChatPanel.tsx`, `EvidencePanel.tsx`, `SafeMarkdown.tsx`, `styles.css`, `App.test.tsx`, `components/SafeMarkdown.test.tsx`.

**Interfaces:** `App` consumes task 6 hook; child components accept typed public DTOs and callbacks only. `SafeMarkdown({text: string})` disables raw HTML and permits HTTP/HTTPS link targets only; code copying uses browser clipboard with visible failure state.

- [x] UI tests `demo_badge_is_always_visible`, `composition_enter_does_not_send`, `empty_message_disabled`, `partial_result_and_reason_visible`, `unsafe_markdown_links_removed`, `missing_evidence_is_explicit`: input Enter sends only when not composing, Shift+Enter inserts newline; raw `<script>` and javascript/file URLs never become executable links; incomplete answer shows reason/unresolved items, no evidence shows “暂无证据”.
- [x] RED: `npm run test`; expect missing components/behavior failures.

  Core assertions: `expect(onSend).not.toHaveBeenCalled()` on composing Enter; `expect(screen.getByText('离线演示')).toBeVisible()`; `expect(container.querySelector('a[href^="javascript:"]')).toBeNull()`; `expect(screen.getByText('暂无证据')).toBeVisible()`.
- [x] Implement approved three-column responsive light UI: list/new/switch; safe Markdown user/assistant messages, loading/error, draft and send; collapsible real route/roles/evidence/public status. Desktop columns collapse to accessible controls on narrow screens; focus labels, readable contrast and scroll behavior. No fake token streaming, no invented route/stage/evidence and no internal prompt inspector.
- [x] GREEN: full frontend tests/build. Ensure linked sources use safe external URLs with appropriate target protections, local files show title/page only. Record checkpoint.

### Task 8: Offline end-to-end acceptance and handoff

**Files:** Create `docs/local-chat-workbench.md`; finish progress document; adjust only defects found in preceding task files. No frozen report changes.

**Interfaces:** Start backend `PY scripts/serve.py --demo`; frontend development `npm run dev`, or production `npm run build` then open backend root. Live command is documented separately and never automatically executed for QA.

- [x] Run `PY -m unittest discover -s tests -q` and save actual output/count to a new local verification log. Run frontend `npm run test` and `npm run build`; record real versions and commands, not prior pass counts. Missing dependencies or failures must be resolved or explicitly reported.
- [x] Read the Sites skill local-preview guidance and computer-use skill before UI QA. Start demo hidden/background with logs in `.cache/`, inspect existing listening processes before choosing/reusing ports, never kill unrelated services. Browser-test create/switch/send/stage/final/evidence/refresh and 390px narrow layout; use injected offline failure fixture for error/incomplete presentation, never a live paid request.
- [x] Check refresh during a controlled running demo turn returns the same turn and one persisted assistant answer; switching conversation does not mix stage/evidence. Check direct `/api/health`, invalid payloads, Host/Origin denial and private-file404 separately. Inspect browser console and network for frontend failures or exposed secrets.
- [ ] Perform an independent final code review using requesting-code-review skill, with no external requests allowed. Address material findings, rerun owning tests, then full verification as needed. No whole-project refactor or fresh prompting experiment.
- [x] Document prerequisites (including actual npm path if needed), install/start/stop commands, localhost URL, demo/live distinction, no-token-stream limit, one-worker/two-job limits, failed/interrupted recovery and preserved storage. Provide a concise changed-file list and verification evidence; disclose if live model behavior remains untested.

## Self-review / execution gate

- Spec coverage: stable promotion→1; shared lifecycle/config→2; identity/history/public DTO→3; idempotency/concurrency/restart→4; endpoints/security/static→5; resume/client→6; UI/safety→7; tests/docs/review→8.
- Interface consistency: the store owns persisted public snapshots; jobs own execution; app owns HTTP status mapping; frontend consumes exactly those DTOs. Graph remains the only Agent executor and core final-answer writer.
- All five Review Focus classes have owning test names above. Tests use temporary databases/fake models, never the user's production DB for destructive assertions.
- Plan intentionally excludes hosted Sites, paid testing, full authentication and historical evaluation rewrites.
- User approved native execution. Implementation and offline acceptance complete; independent reviewer findings fixed, but final independent verdict unavailable due account limit. See persistent progress ledger.
