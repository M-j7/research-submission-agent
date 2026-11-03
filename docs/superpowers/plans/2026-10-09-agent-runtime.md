# Agent Runtime Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. User requested direct execution without further confirmation; implement inline.

**Goal:** 把已确认的 Agent、记忆、工具和 RAG 设计连接成可测试的问答运行时。

**Architecture:** 保留旧识别/路由预览图，新建可执行聊天图。MemoryManager 是原文与投影权威，PromptManager 只读取组装；所有角色复用带预算的原生工具循环。RAG 通过适配器复用现有索引，不迁移原件。

**Tech Stack:** Python 3.12 / SQLite / Pydantic / httpx / LangGraph / existing Chroma and LlamaIndex.

**Spec:** ../specs/2026-10-08-context-memory-design.md；../specs/2026-10-08-virtual-filesystem-design.md；../specs/2026-10-08-agent-tools-design.md；../specs/2026-10-08-langgraph-flow-design.md。

## Global Constraints

- 公共按 user/conversation 隔离，私有按 run 隔离；无独立黑板。
- 原文不覆盖；短期摘要＋五轮；先旧 tool 正文、再状态描述；70% 异步触发。
- 原生 tool calling，后台严格 Schema 和白名单；普通文本 JSON 不执行。
- 步数默认8、最后一步禁用工具；工具并发默认2；取消传播。
- 不改现有索引、不打印密钥、不建立 Git 仓库；未定的人工接管/监测不接外部服务。

## Review Focus

- 同一 turn 重放/并发：不得重复调用、发布或计数。
- 压缩中追加/结束：不得丢新消息或提交迟到摘要。
- 恶意路径/符号链接/游标跨用户：拒绝且不泄漏宿主机信息。
- 多工具失败/乱序/取消：每个 ID 成对记录，不假报成功。
- 改写失效/后端局部失败：原问题保留，明确降级，不重复算证据。

## Task 1: Memory and Prompt

Files: `runtime/contracts.py`, `runtime/memory.py`, `runtime/prompt.py`, `tests/test_runtime_memory.py`.

Interfaces: Request(user_id,conv_id,turn_id,message); MemoryManager.begin_turn/snapshot/append/publish_result/finish_turn; PromptManager.build(public,instant,role,tools,skills).

- [x] 写并运行隔离、幂等五轮、CAS压缩、固定前缀和硬预算失败测试（RED）。
- [x] 实现 SQLite 追加记录、私有投影、来源摘要、用户画像版本接口及 Prompt 预算（GREEN）。
- [x] 验证本任务和原有94项测试。

## Task 2: Workspace and tools

Files: `runtime/workspace.py`, `runtime/tools.py`, `runtime/skills.py`, `tests/test_runtime_tools.py`.

Interfaces: WorkspaceManager scoped virtual resolve; ToolRegistry.schemas/execute; SkillCatalog metadata only.

- [x] 写并运行路径越权、严格输入、完整分页及白名单测试（RED）。
- [x] 实现 glob/grep/read/query_memory/search_knowledge、受控游标、原文文件映射（GREEN）。
- [x] 部署现有六份 Skill 草稿；不新增政策内容。
- [x] 验证任务测试和全套。

## Task 3: Agent loop and outer graph

Files: `runtime/loop.py`, `runtime/chat.py`, `runtime/provider.py`, `scripts/chat.py`, `tests/test_runtime_chat.py`; update role whitelists and existing tests.

Interfaces: model.complete(messages,tools,allow_tools); AgentLoop.run(Request,prepared,public); build_chat_graph(service).

- [x] 写并运行原生工具配对、拒绝非法工具、步数收尾、并行失败、多角色仅一最终回答测试（RED）。
- [x] 实现共享循环、角色执行/结果发布/汇总、持久化后返回和受控结束（GREEN）。
- [x] 保留旧预览入口；用新 CLI 提供不联网 demo 和显式 live 模式。
- [x] 验证图级回归和全套。

## Task 4: RAG Multi-Query and cache

Files: `runtime/rag.py`, `runtime/cache.py`, `tests/test_runtime_rag.py`.

Interfaces: MultiQueryRAG.search(original,top_k); existing child retrieval -> cross-list RRF -> original-query rerank -> parents; optional cache backend.

- [x] 写并运行漂移拒绝、原查询回退、ID去重、原查询重排与缓存失效测试（RED）。
- [x] 实现最多3候选、最多2次改写、JEV守门、读取原索引适配器和可选 Redis 适配（GREEN）。
- [x] 验证离线及已存知识库回归，不自动付费批量测试。

## Task 5: Review and handoff

- [x] 全套测试与 CLI demo，检查来源/数据未被迁移或删除。
- [x] 记录实现/未实现边界和运行方法，更新项目入口。
- [x] 按技能做最终代码审查，重要问题用 RED→GREEN 修复；不声称离线测试证明模型质量。

## Execution ledger

- 2026-10-09 基线：`python -m unittest discover -s tests -q`，94项通过。
- Ruling: 无 Git 仓库，直接在用户指定工作区实现，不运行 Git worktree/commit脚本；不删除执行台账。
- Ruling: 本次不用新的实现子 Agent；用户明确要求直接实施并免逐项确认，按已有设计推进。
- Pre-flight: 1→2 共用后端身份及记忆查询；2→3共用工具Schema/结果；1→3共用快照/状态；4→2通过search_knowledge回调绑定，无额外角色RAG接口。
- Task 1–4: 实现基线完成；新增 memory/tools/chat/rag/maintenance/skills 测试按任务 RED→GREEN，审查前主测试123项通过，原父子检索16项通过。
- Final review: 一次独立只读整体审查，确认6项 Important、无已确认 Critical；不再分派实现或重复审查。
- Final: fixed 画像按完成顺序推进水位 — test_profile_batch_uses_completion_not_receipt_order RED→GREEN。
- Final: fixed 画像/短期摘要跨连接CAS写事务 — test_profile_and_summary_cas_are_atomic_across_connections RED→GREEN。
- Final: fixed 本地扫描协作式取消与整体截止 — test_regex_scan_yields_to_external_deadline RED→GREEN。
- Final: fixed 汇总来源继承、证据ID去重与失败覆盖 — test_composer_preserves_evidence_identity_and_failed_coverage RED→GREEN。
- Final: fixed 输出不能越过估算累计预算 — test_final_response_cannot_bypass_token_budget RED→GREEN；远程输出预留后仍校验，不声称严格多服务计费上限。
- Final: fixed 范围读取起始行不污染续页身份 — test_ranged_read_cursor_can_continue_without_start_line RED→GREEN。
- Final: fixed 画像来源不能跨提取截止范围 — test_profile_cannot_cite_sources_after_extraction_cutoff RED→GREEN。
- Final: fixed 终态计数与压缩投影事务 — 两个专门回归 RED→GREEN；最终主suite132/132（10.997s）。
- Final verification: 父子检索16/16、清洗6/6、切分5/5；compileall成功；CLI demo退出0，两个专业角色各2步/1工具，汇总1步/0工具，唯一最终答复。
- Final verification: 只读RAG适配器加载966父块/2737子块，Chroma沿用ASCII工作区兄弟目录；没有重建索引。
- Final verification: 真实合成探针先受网络权限10013限制，经正常授权后2次Qwen请求原生调用/Schema/结果配对/最终文本通过；未发送既有记忆，不是完整真实链路或质量评测。
- Ruling: 保留旧preview与索引，增加执行入口和异步RAG适配；代价是需要区分两个入口，RAG尚非独立LangGraph子图。
- Ruling: 二级自然语言状态压缩、Redis实接、人工接管、恢复、监测和质量评测不冒充完成；当前状态栏短小且仅最新版入Prompt，预算仍为实验基线；代价是需继续联调与补齐。
- Final: minor (deferred): 锁表容量回收、分页警告延续/目录新增失效、逐工具返回data强类型；完整清单与安全强化边界见 docs/Agent运行时实现与验证.md。
- Final: Ruling: live质量、Windows junction竞态、生产鉴权/崩溃恢复不在这次离线审查证明范围；不能宣称生产就绪，代价是上线前仍需安全与联调验证。
- Final handoff: 无Git分支可合并/提交，不初始化仓库、不删除台账；历史10-08规格仍保持设计稿，最新入口及10-09实现报告已更新。
