# Agent Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 EvalScope 接入真实 Agent，交付可追溯的三层评测与统一看板的数据源。

**Architecture:** 保留 ChatService / LangGraph / ReAct 实现，用实例级依赖注入采集工具、模型与原始记忆记录。执行、判分、报告解耦；Decisions 接入保留旧 JEV 兼容，正式运行前先通过离线及 6 题烟测。

**Tech Stack:** Python 3.12、unittest、httpx、SQLite、现有 LangGraph、EvalScope 1.13.0；独立 `.venv-evaluation`，不升级应用环境。

**Spec:** `docs/superpowers/specs/2026-10-18-agent-evaluation-design.md`

## Global Constraints

- 30 题＝25 业务＋5 故障安全；6 题烟测＝5 业务＋1 故障，三层为端到端／轨迹／表现。
- 不改 `.env`、历史 RAG、生产记忆、Redis 或知识库索引；非 Git 目录不初始化、不提交。每任务结束在进度记录保存测试命令和结果，替代提交检查点。
- 新运行目录 `evaluation/agent/runs/<run-id>/`；新 user_id、conv_id、trial_id，每轮新 turn_id。参考判据不进入执行提示。
- 烟测并发 1，完整批次并发 2，外层每任务 300 秒；无整任务隐藏重试。
- 裁判 timeout 20 秒，最多 2 次，总限 45 秒；只对网络／429／5xx 做指数退避＋随机抖动。401／403 或额度错误暂停后续相关调用。
- Decisions 端点 `https://api.perplexity.ai/v1/decisions`，模型 `pplx-decider-flash-v1.2`；不再调用旧 JEV 或 Router，不静默重写配置。
- 概率 >=0.8 通过、<=0.2 失败、其余未知，标记未人工校准；用量未知不是零，业务成功不等于执行完成。
- 本计划不自动启动第二批 6×3 稳定性运行；实现入口并测试，是否启动在首批结果后说明。

## Review Focus

1. 凭据错误被意图识别降级吞掉：运行监督器仍识别并中止后续付费任务（Task 3）。
2. 并发角色或多轮用量串题：以实例及显式 trial_id 关联，不能依赖全局当前题（Task 3）。
3. 正常拒绝与空答、运行完成与任务通过混淆：冻结各类规则，分开状态（Task 4）。
4. 中途取消、重复 ID 或记录损坏：保留终态与期望分母，验证失败不得标 complete（Task 5）。
5. 裁判材料中的提示注入：可信 rubric 与不可信材料隔离，红线仍由程序优先判断（Task 4）。

## 执行约定

以下 `$evalPython` 指 `<项目>/.venv-evaluation/Scripts/python.exe`，PowerShell 调用 `& $evalPython -X utf8 ...`；测试文件依现有 unittest discover 风格。新增模块均置于 `evaluation/agent/`，应用源码仅做默认兼容的必要扩展。首个任务前记录现有全套测试基线。

### Task 1: Decisions 显式协议适配

**Files:** Modify `src/submission_assistant/intent/jev.py` (`JevSettings`, `from_env`, `_post` 创建 client 位置)；Create `tests/test_decisions_provider.py`；Modify `.env.example`（仅示例，无密钥）。

**Interfaces:** 保留 `JevSettings(endpoint,api_key,model,timeout_s)` 位置参数兼容；新增尾部字段 `provider: str='systemone'`、`trust_env: bool=False`。新增 `JevSettings.from_values(values: Mapping[str,str]) -> JevSettings`，`from_env` 合并后委托它。既有 `JevIntentRecognizer.recognize` 与 `IntentJudgments` 不变。

- [ ] 写 `DecisionsProviderTests`：mock transport 断言 decisions provider 只 POST `/v1/decisions`、发送 6＋5 个 noul、完整解析 11 项；旧 systemone 路径不变；拒绝 Router、错误供应商、内嵌凭证、非有限概率、缺项；repr／错误不含测试密钥。`trust_env` 显式配置，不改变原默认。
- [ ] 运行 `-m unittest discover -s tests -p test_decisions_provider.py -v`，确认新行为 RED（不是依赖缺失）。
- [ ] 增加供应商各自严格端点白名单、模型默认与 `JEV_PROVIDER`／`JEV_TRUST_ENV` 读取。评测内存配置显式覆盖为 Decisions＋正确模型＋系统代理，不写 `.env`。保持旧解析契约、现有识别分组与业务阈值。
- [ ] 运行新测试及 `-m unittest discover -s tests -p '*jev*.py' -v`、`-p '*ensemble*.py'`，确认 GREEN；记录兼容结果。

### Task 2: 冻结用例与工件契约

**Files:** Create `evaluation/agent/__init__.py`, `contracts.py`, `dataset.py`, `cases.zh.json`, `tests/test_agent_eval_data.py`。

**Interfaces:** `load_cases(path: Path) -> list[dict]`；`validate_cases(cases: list[dict]) -> None`。每题字段：`case_id, category, group, turns, initial_memory, expected_roles, required_actions, forbidden_actions, semantic_checks, reference_notes, source_refs, review_status, fault`。每轮为 `{turn_id, message, checks}`；fault 无注入为 null。所有 rubric、来源与审核状态冻结进 manifest 哈希。

执行记录 `ExecutionRecord` 为 TypedDict：`case_id,trial_id,group,execution_status,turns,route,events,usage,latency_ms,stop_reason,error`。结果 `ScoredRecord`：`case_id,trial_id,group,business_outcome,rules,semantic,performance,review_status`。字段字典中记录时间单位与 null 语义；真实 token 字段不与 estimate 合并。

- [ ] 写 `AgentEvalDataTests`：30 个唯一 ID、六类各 5、25/5 分组；多意图 2 同角色＋3 跨角色，多轮 2～3 轮；烟测固定 6 ID 覆盖各类；拒绝重复 ID、无来源、非法规则或未定义角色。执行输入导出断言无参考／期望判据。
- [ ] 运行 `-m unittest discover -s tests -p test_agent_eval_data.py -v`，确认 RED。
- [ ] 根据现有角色目录、工具白名单及只读知识资料写 30 个用例，`review_status='agent_draft'`，不编造官方事实。`select_cases(cases, *, stage: str) -> list[dict]` 固定 smoke/full/repeat 的 ID；repeat 预选 6 题、每题 3 个独立试次，不自动执行。
- [ ] 同命令 GREEN，导出用例审阅表；逐条自查有可检验的预期而非“答得好”。

### Task 3: 真实服务采集、隔离与故障注入

**Files:** Modify `src/submission_assistant/runtime/chat.py`（构造与 ToolRegistry 实例化位置）；Create `evaluation/agent/observe.py`, `runtime.py`, `faults.py`, `tests/test_agent_eval_runtime.py`。

**Interfaces:** `ChatService(..., tool_registry_factory=ToolRegistry)` 默认行为不变。`ObservedModelRouter` 包装 `.for_role(role)`，返回保持 `.model`、`.complete` 契约的模型；`ObservedToolRegistry` 包装真实 `.execute`。`ObservationSink(trial_id: str)` 显式实例归属，收集开始／完成／失败、role、run_id、调用 ID、时间与原始 usage。

`async execute_case(case: dict, *, trial_id: str, run_dir: Path, values: Mapping[str,str], live: bool) -> dict` 返回 ExecutionRecord；live=False 只允许注入测试依赖，不伪装真实质量。`build_service(...)` 复用真实全聊天图，使用每试次独立 SQLite／scratch，关闭生产缓存、独立 embedding 缓存并记录；知识索引只读复用并哈希。本轮禁用后台画像维护（manifest 注明），不改即时／短期记忆及现有压缩机制。

- [ ] 写 `AgentEvalRuntimeTests`：确定性假供应商驱动完整 ChatService 图和真实工具封装；两试次并发不串数据，同题多轮保持会话；raw SQLite records 保留压缩前内容；真实 usage 缺失仍为 null，重复／失败调用计数保留。验证外层超时清理资源且留下终态。
- [ ] 同文件 RED 后添加实例级观测与隔离；不 monkeypatch 全局、不以根目录准备图代替完整聊天图、不把 token 预算估计当账单。
- [ ] 增加固定 fault IDs 对应注入：工具超时、重复无进展、非法参数、越权读取、材料提示注入。重复／非法参数等需要脚本模型驱动的用例明确标记 `execution_mode='deterministic_fault'`；其余业务不替换模型。工具结果保留实际安全拒绝，不能让注入器替系统回答。
- [ ] 测试五种故障及被 recognizer 包装的 401／403：提取结构化错误给监督器，后续待运行题保留未执行。观测层不得保存认证头。
- [ ] 运行 `-m unittest discover -s tests -p test_agent_eval_runtime.py -v` 及 `-p test_runtime_chat.py`、`-p test_runtime_tools.py`，确认 GREEN，记录默认应用行为未变。

### Task 4: 独立规则、语义裁判与正确分母

**Files:** Create `evaluation/agent/rules.py`, `judge.py`, `score.py`, `tests/test_agent_eval_scoring.py`。

**Interfaces:** `check_rules(case: dict, execution: dict) -> list[dict]`，每项 `{id,status,reason,red_line}`；`async judge_semantics(case: dict, execution: dict, *, settings: JevSettings, client: httpx.AsyncClient) -> dict`，保留原概率／版本／尝试／usage，不输出编造的自然语言解释；`score_case(case,execution,semantic) -> dict` 产出 ScoredRecord；`summarize(cases,executions,scores) -> dict` 业务、故障分组。

- [ ] 写 `AgentEvalScoringTests`：completed 但漏要求是 fail；正确澄清可 pass；服务执行失败是 fail，裁判失败不是执行失败；红线优先。概率 .8/.2 边界，中间未知；真实0与缺项不同。3 题中 1 pass、1 fail、1 unavailable：已验证1/3，有效通过1/2；故障通过不进入业务分母。
- [ ] 运行测试 RED；实现冻结角色比较、调用配对、白名单、预算与必需／禁止动作，查询参数按业务约束而非字面相等。存在可观察计划才评价计划，无隐藏思维链要求。
- [ ] 实现 Decisions rubric 可信边界与单任务固定问题分组；单次20秒、2次／45秒上限，仅临时错误指数抖动重试。测试 fake clock／sleep：429重试、401/403不重试、超限停止、malformed不盲重试；测试材料 `ignore rubric` 不修改问题模板及红线。
- [ ] 同文件 GREEN；保留 `review_status='uncalibrated'`，仅留人工复核入口，不替用户宣称已完成校准。延迟按墙钟、usage按已知加总＋完整性，缺价格不报金额。

### Task 5: EvalScope、运行账本与报告

**Files:** Create `evaluation/agent/evalscope_adapter.py`, `run.py`, `report.py`, `README.md`, `tests/test_agent_eval_adapter.py`, `tests/test_agent_eval_runs.py`。

**Interfaces:** `ProjectAgentAdapter(AgentAdapter)` 的同步 `_on_inference(self, model, sample) -> InferenceResult` 以每样本独立 asyncio.run 桥接 Task 3 执行接口，不使用 EvalScope AgentLoop。`to_agent_trace(execution: dict) -> AgentTrace` 保留分支 run_id/parent_id、角色、真实事件顺序；同步 `run_batch(cases: list[dict], *, output: Path, values: Mapping[str,str], concurrency: int, live: bool) -> dict` 配置 TaskConfig(eval_batch_size=concurrency)，只由 EvalScope inference 调用 execute_case，不能在外围再执行一遍。每样本拥有自己的异步客户端和服务，不跨事件循环复用。`load_agent_run(run_dir: Path) -> tuple[dict,list[dict],list[dict],list[dict]]` 返回 manifest/cases/executions/scores，供看板复用。

- [ ] 写 Adapter 测试证明调用真实服务入口一次、报告重放零业务调用、null均值不补0、usage和并行角色Trace对应；用安装的1.13.0真实TaskConfig离线契约测试而非只测mock。
- [ ] 写 Runs 测试证明原子增量保存；已有输出目录拒绝覆盖；输入hash不符、重复样本、未知ID拒绝。中断后所有预期ID有已执行／未执行终态，manifest.complete只在覆盖及报告核验通过后为true。
- [ ] 运行两文件 RED；实现 CLI `python -m evaluation.agent.run --stage smoke|full|repeat --output <new-dir> --live`；无 `--live` 不调用供应商。实时 EvalScope inference 只执行一次；判分与原生报告重放必须使用落盘结果，显式标识 replay。
- [ ] 落盘 `manifest.json,cases.json,executions.jsonl,traces.jsonl,scores/v1/{manifest.json,scores.jsonl},summary.json,report.zh.md,evalscope/`。manifest保存版本、hash、模型、缓存条件、预算、usage完整性、未校准说明；不含密钥。score-only入口 `--score-only <existing-run> --score-version <new-version>` 仅补缺项，拒绝覆盖旧评分。跨样本停机信号用线程安全Event及按凭据身份分组的调用门禁，每次新付费调用前检查；已在途请求如实记录，不能宣称能够追回费用。JSONL追加同样受线程锁保护。
- [ ] 运行两个新测试文件及全部测试 `-m unittest discover -s tests -v`；所有非可选失败需处理；记录可选跳过与环境，不能沿用旧测试数当本次结果。

### Task 6: 分阶段真实运行与交付核验

**Files:** Create 新的 `evaluation/agent/runs/20261018-smoke-01/` 与通过门禁后的 `20261018-full-01/`（已存在则选择新ID，不覆盖）；Update `evaluation/agent/README.md` 的运行说明与记录。

- [ ] 检查配置脱敏摘要、端点模型、只读知识哈希和隔离路径；不额外重复付费“连通性测试”。先执行 smoke，单并发、6题，不跑旧JEV；权限／额度失败立即停止依赖该密钥的新请求。
- [ ] 核对6条唯一终态、5业务1故障、预期专业路由不是全部故障兜底、工具轨迹与usage来源，保存具体差异。合理业务失败可保留，基础采集／配置失败必须修复后另建run，不在原结果上涂改。
- [ ] 展示6题机器判分及未人工校准状态。若只缺人工复核，完整批次可标为“未校准机器诊断”继续；不得冒充golden自动评测。其他运行门禁通过后执行full，并发2、30题。
- [ ] 验证30个唯一预期ID和全部终态；25/5分组，业务／裁判／未执行错误分开，逐题统计与JSON／中文／EvalScope容差0.00011内一致。已有RAG输入哈希应未变。
- [ ] 按用户最新简化要求，将本轮 Agent 及已有 RAG 的相关指标、有效数／预期数和必要限制导出为简单静态 `evaluation/reports/agent-rag-overview.html`；不执行原交互看板计划，不做筛选、搜索或复杂图表。完整跑完才报告完成；中断则交付部分数据并指出阻塞，不自动充值、不无限重跑。

## 自查与交接

覆盖：设计1～4→Tasks1～3；三层与校准→Task4；状态隔离→Task3；预算与失败→Tasks4～6；报告分母与验收→Tasks5～6。五项Review Focus均有对应测试。接口统一使用字典工件契约与 `load_agent_run`，看板不反向依赖实时服务。没有新增安装、真实执行或评分发生在计划阶段。

关键边界断言（作为对应测试的最小断言，不是生产代码）：

```python
# Task 1: test_decisions_endpoint_and_secret_redaction
self.assertEqual(settings.endpoint, 'https://api.perplexity.ai/v1/decisions')
self.assertNotIn('unit-test-secret', repr(settings))
# Task 2: test_frozen_category_counts
self.assertEqual(len(cases), 30)
self.assertEqual(sum(c['group'] == 'business' for c in cases), 25)
# Task 3: test_parallel_trials_are_isolated
self.assertNotEqual(left['trial_id'], right['trial_id'])
self.assertTrue(all(e['trial_id'] == left['trial_id'] for e in left['events']))
# Task 4: test_unknown_not_counted_as_zero
self.assertEqual(result['business_outcome'], 'unavailable')
# Task 5: test_replay_does_not_execute_agent
execute_mock.assert_not_called()
```

执行方式待用户选择；推荐本会话直接实现，最后一次独立审查，避免各任务共享接口被反复拆解。
