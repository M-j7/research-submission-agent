# SDD ledger — plan: docs/superpowers/plans/2026-10-07-intent-to-agent.md

用户明确要求无需确认、直接实施。主Agent在本会话顺序执行，最终独立审查。

Pre-flight: Task1 IntentJudgments -> Task2 judgments -> Task3 route_intents -> Task4 prepare -> Task5 graph，字段及调用顺序一致。

Ruling: 本目录非Git仓库，原地按文件限定修改并保留此台账，不创建worktree或使用依赖Git的台账脚本；不提交、不删除台账 — 用户指定此工作区 — 代价是没有Git提交回退。

Ruling: 保留旧IntentResult/build_request/parse_response作历史离线兼容，新在线recognize与CLI只走build_requests和IntentJudgments，路由拒绝旧契约 — 保留已有协议及诊断测试，不把旧单选模拟成多意图 — 代价是暂留少量显式标注的兼容代码。

Baseline: 主tests目录64项，60通过、4报错，均是辅助证据字段缺失。四项必须修复，未授权删减或skip。

## Tasks

- Task 1: complete — test_independent_jev.py六项RED→GREEN；旧配置/输入/安全与历史协议测试保留通过，在线契约测试迁移为两批noul。
- Task 2: complete — test_independent_ensemble.py三项RED→GREEN；原四项辅助接口错误全部修复、保留并通过；CLI同步两批预览。
- Task 3: complete — test_routing.py八项RED→GREEN，同角色去重、跨角色协作、五种兜底及阈值边界通过。
- Task 4: complete — test_agents.py五项RED→GREEN，统一继承、静态/动态分离、工具白名单和注册表过滤通过。
- Task 5: complete — test_agent_graph.py最初七项RED→GREEN，真实LangGraph与受控CLI通过；小量真实模型三例共六次POST完成。

Ruling: 增加--demo、--jev-only与拒绝覆盖的--output入口 — 用户无需手工构造JSON即可检查图，真实烟雾可以控制Embedding费用 — 代价是CLI多三个可选参数，输出明确标记受控与真实来源。

Dependency evidence: langgraph1.2.14、langchain-core1.6.7安装到原Conda环境，配套依赖websockets17.2→16.1.1；pip check无冲突，旧RAG测试通过。未升级其他现有RAG包。

Real smoke: 单业务C02=.97→规范；跨角色C02=.88/C07=.93→规范+费用；模糊稿件clarity=.12→通用。JEV总15908输入/615输出，Embedding首次75条模板+查询1871tokens。不能推导整体准确率或费用金额。

Final review: 一位新上下文只读reviewer审查，Critical0、Important1、Minor1，Declined to judge为空。

Final: fixed BM25异常阻断JEV — test_bm25_runtime_failure_cannot_prevent_complete_jev_routing先失败（general而非fees），隔离辅助异常后通过；全主suite94/94。取消仍传播，异常正文脱敏。

Final: 源码旧说明问题在审查反馈到达前已同步随新接口更新；无遗留minor，不发起重复审查。

Final verification: 主测试94、简单清洗6、语义分割5、父子检索16，合计121项通过；pip check通过。受控六例运行正常，三份实测报告无api_key/Authorization/Bearer字段。资料库、Chroma与.env未改；新意图示例缓存属于正常调用产物。

No Git integration: 本项目非Git仓库，保留原地源码、测试、设计计划与台账，未创建提交、PR或删除工作区。
