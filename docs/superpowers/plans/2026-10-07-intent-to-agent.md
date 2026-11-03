# JEV 独立判断、路由及多 Agent 协作判定 Implementation Plan

执行状态：用户明确要求“不用我确认，你直接写”，已在当前会话原地完成；详细RED/GREEN、兼容性裁定、独立审查与验证记录见 `2026-10-07-intent-to-agent-progress.md`。下面是执行前计划，不再等待确认。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将十类独立 JEV 判断接入确定性路由，判断是否需要跨角色协作，并让选中的 Agent 准备输入。

**Architecture:** JEV 主判，Embedding/BM25只提供辅助证据；LangGraph串联识别、路由和Agent输入准备。所有五个角色直接继承同一个BaseAgent，专业角色仅改变提示词与工具权限。

**Tech Stack:** Python 3.12、httpx、unittest、LangGraph，复用现有 Miniconda 环境。

**Spec:** `docs/superpowers/specs/2026-10-07-intent-to-agent-design.md`

## Global Constraints

- 工作目录为本项目；不是Git仓库，不创建仓库、不伪称提交。
- 只使用 `C:\Users\amorous\miniconda3\envs\ieee-submission-assistant\python.exe`，不另建环境。
- 十类各自返回0–1独立相关性概率，不要求总和为1；另有独立clarity。
- 两批请求：clarity与C01–C05共6题，C06–C10共5题；共用本轮消息和最近六条历史，state不超过8000字符。
- 不自动重试；取消传播；不输出密钥、认证头或原始错误响应正文。
- 默认clarity_threshold=0.75，relevance_threshold=0.80，均可配置且明确未校准。
- General兜底；跨专业角色才触发多Agent；同角色多意图去重。
- 保留三路原始证据；0.7/0.2/0.1加权分只作诊断，不用于路由。
- 本次不执行回答LLM、RAG工具或多Agent答案汇总；未注册工具不冒充可执行工具。
- 不改.env密钥、原始资料库、解析数据或Chroma索引。

## Review Focus

1. 一批失败而另一批成功：保留批次状态和已知用量，但必须通用兜底，不拼成完整成功。
2. 问候后接业务问题以及退款背景中提到撤稿：按实际诉求判断，不因关键词误触发协作。
3. 同角色多意图、不同角色多意图、C10与业务同时高分：分别单Agent、多Agent、冲突兜底。
4. 两路辅助全零、并列或失败：不制造确定意图、不提高JEV概率；JEV完整时仍可路由。
5. 历史和角色工具权限：不修改调用方消息；未注册及越权工具不可进入准备结果中的实际工具列表。

## 文件与边界

- 修改 `intent/catalog.py`：十项独立相关性说明；统一C10排除“混合业务消息”的标准。
- 新建 `intent/contracts.py`：独立判断和批次状态数据契约。
- 修改 `intent/jev.py`：构建两批、并发调用、严格解析和统计；保留现有安全配置与HTTP错误处理。
- 修改 `intent/ensemble.py`：JEV主判＋辅助证据，保留加权诊断。
- 新建 `routing.py`：角色映射、参数和确定性协作判断，不调用模型。
- 新建 `agents/base.py`、`agents/roles.py`、`agents/__init__.py`：统一输入准备、角色配置、注册表过滤。
- 新建 `graph.py`：LangGraph三个节点，只到Agent接收。
- 修改两个现有JEV/三路CLI，新增 `scripts/route_to_agents.py`：预览、离线受控结果、真实识别。
- 修改/新增测试与说明文档；现有RAG相关脚本不变。

## Task 1：独立 JEV 契约及两批调用

**Files:** `intent/contracts.py`、`intent/catalog.py`、`intent/jev.py`、`tests/test_jev_intent.py`。

**Interfaces:**
- `BatchStatus(batch_id: str, request_id: str, status: str, usage: dict[str,int], error: str | None)`。
- `IntentJudgments(probabilities: dict[str,float], clarity_probability: float | None, model: str, request_id: str, batches: list[BatchStatus], latency_ms: float)`；`complete`只在十项、clarity及两批全部有效时为true；`to_dict()`输出score_type=independent_relevance_probability。
- `build_requests(message: str, history: list[dict[str,str]] | None = None, *, model: str = 'jev-latest') -> list[dict]`。
- `JevIntentRecognizer.recognize(message, history=None) -> IntentJudgments`。

- [ ] 写测试并确认RED：两批题目严格为6/5，十一题全为noul，共享state且不修改history；如C02=.95与C07=.94都可返回，不归一化。
- [ ] 写故障测试：丢题、bool、NaN、越界、错误类型拒绝；一批502而另一批成功则complete=false且保留已知用量；两批请求ID不同；无自动重试；取消终止请求。
- [ ] 实现两批严格解析及安全错误状态；总usage按已知批次相加，失败批次用量未知不能声称为零。成功批次模型不一致判协议失败。
- [ ] 替换旧九选一HTTP契约及构造测试，保留原有输入限制、设置校验和不泄密断言；不再捏造单选confidence、dialogue或top_two_gap。
- [ ] 验证：`python -m unittest discover -s tests -p test_jev_intent.py -v` 全部PASS。

## Task 2：JEV主判与两路辅助证据

**Files:** `intent/ensemble.py`、`tests/test_intent_ensemble.py`、`tests/test_non_business_templates.py`、`scripts/judge_intent.py`、`scripts/judge_all_intents.py`。

**Interfaces:**
- `fuse_results(*, jev: IntentJudgments | None = None, embedding=None, bm25=None, source_errors=None, latency_ms=0.0) -> dict`。
- 返回 `judgments`、`auxiliary_evidence`、`diagnostic_fusion`、`sources`、`source_errors`；下游只读取judgments做路由。
- `EnsembleIntentRecognizer.recognize(message, history=None) -> dict` 保留现有异步并发入口。

- [ ] 写RED测试：辅助匹配第一名不能覆盖JEV；正分并列输出tied_candidates，全零no_match；原始独立概率保持不变；缺少JEV时candidate_only=true。
- [ ] 实现辅助证据中的top_intents、top_matches（含原始分和示例）及支持/替代/并列/无匹配关系；多标签情形按JEV业务候选集合比较，不依赖唯一主意图。
- [ ] 加权诊断保留九业务类的旧公式与手算断言，移入diagnostic_fusion；C10原始检索证据保留，不用旧加权分作非业务判定。
- [ ] 迁移此前四项未完成辅助证据测试到新多标签契约，保留它们各自的核心断言，不删测试、skip或xfail；其他服务失败与取消测试改为两批。
- [ ] CLI dry-run输出两批、独立分数契约、75条模板及真实Embedding批次大小，不读取密钥；真实模式保留批次用量。
- [ ] 验证：`python -m unittest discover -s tests -p 'test_*intent*.py' -v` 与 `test_non_business_templates.py` 全部PASS。

## Task 3：确定性路由与协作条件

**Files:** 新建 `src/submission_assistant/routing.py`、`tests/test_routing.py`。

**Interfaces:**
- `AgentType`枚举：general、submission_requirements、editorial_workflow、publication_policy、publication_fees。
- `RoutingSettings(clarity_threshold: float=0.75, relevance_threshold: float=0.80)`：拒绝bool、非有限值及0–1范围外配置。
- `RoutingDecision(agent_types: list[AgentType], agent_intents: dict[str,list[str]], matched_intents: list[str], mode: str, multi_agent: bool, fallback_reason: str | None, collaboration_reason: str | None, settings: dict)`；`to_dict()`可JSON序列化。
- `route_intents(judgments: IntentJudgments | None, settings: RoutingSettings | None = None) -> RoutingDecision`。

- [ ] 写RED：C02=.95、clarity=.90→requirements；C02+C03→一个requirements、multi_agent=false；C02+C07→requirements+fees、multi_agent=true并保存各角色负责意图。
- [ ] 写RED：C05+C06+C09只选workflow；C04+C08只选policy；四专业角色同时命中时恰好四个，不加入General。
- [ ] 写RED：不完整→recognition_failed；C10独高且clarity低→non_business；C10和C02高→conflicting_judgments；清晰度低→unclear_request；清晰但无高业务→uncertain_intent。所有兜底multi_agent=false。
- [ ] 实现设计规定的顺序与>=门槛边界；角色稳定排序，同角色去重，不用辅助分差或检索第一名触发协作。
- [ ] 验证：`python -m unittest discover -s tests -p test_routing.py -v` 全部PASS，覆盖恰好门槛及非法设置。

## Task 4：五类Agent接收输入

**Files:** `agents/base.py`、`agents/roles.py`、`agents/__init__.py`、`tests/test_agents.py`。

**Interfaces:**
- `AgentRequest(message: str, history: list[dict[str,str]], request_id: str)`，不引入user_id与画像。
- `AgentProfile(agent_type: AgentType, system_prompt: str, allowed_tool_names: tuple[str,...])`。
- `BaseAgent.prepare(req: AgentRequest, decision: RoutingDecision, tool_registry: Mapping[str,object] | None = None) -> dict`。
- `create_agent(agent_type: AgentType) -> BaseAgent` 工厂；五个角色直接继承BaseAgent，无各自决策循环。

- [ ] 写RED：五个类直接继承BaseAgent且prepare共用；静态system_prompt跨请求不变；message/history单独保存且不修改输入。
- [ ] 写RED：跨角色复合请求每个Agent只标注自身负责的intent_ids，保留完整用户消息；通用输入带fallback_reason及接待/澄清/说明服务失败的指导。
- [ ] 实现角色提示与允许工具配置。未接入的检索工具列unbound_tools；只有同时注册且角色允许的工具进入registered_tools；工具对象不直接写入JSON输出。
- [ ] 验证未注册、越权工具过滤及不执行handler；`python -m unittest discover -s tests -p test_agents.py -v` 全部PASS。

## Task 5：LangGraph入口、CLI及端到端验收

**Files:** `graph.py`、`scripts/route_to_agents.py`、`tests/test_agent_graph.py`、`environment.yml`、`docs/JEV需求清晰度与意图分类.md`、`docs/三路意图识别与融合.md`、新增 `docs/意图路由与多Agent协作.md`。

**Interfaces:**
- `build_agent_graph(recognizer, *, settings: RoutingSettings | None = None, tool_registry=None)` 返回编译图；节点START→recognize_intents→route_agents→prepare_selected_agents→END。
- state包含request、recognition、routing_decision、prepared_agents，均可检查，不包含回答或工具执行轨迹。
- `route_to_agents.py --message TEXT [--history-file PATH] [--judgments-file PATH | --dry-run] [--env-file PATH]`；judgments-file复用Task1的完整JSON契约并校验，不读.env、不联网。

- [ ] 在既有Conda环境安装langgraph，记录实际验证版本到environment.yml；先检查依赖冲突，不升级无关RAG依赖；运行`python -m pip check`。
- [ ] 写RED：用假recognizer但真实编译图验证single/multiple/fallback、history、工具未绑定、无生成模型调用；调用者取消传播。
- [ ] 实现图与CLI；离线模式使用受控结果，不伪装模型实测；真实模式明确产生JEV两批和Embedding调用。输入错误使用非零退出码；正常通用兜底是合法路由结果。
- [ ] 全回归：`python -m unittest discover -s tests -v`；记录实际通过/失败总数。修改后的测试不得保留原有四个报错，也不得通过删除它们变绿。
- [ ] controlled fixture至少演示格式、格式+材料、格式+费用、问候、模糊业务、判断失败六类，输出角色、matched_intents、multi_agent、原因及prepared_agents。
- [ ] 小量真实JEV烟雾测试：明确单业务、跨角色双业务、模糊业务各一个（最多6次HTTP POST、不自动重试）；保存脱敏结果和用量，不据三例宣称准确率。网络不可用则报告限制，不能替换为mock后声称实测。
- [ ] 文档更新当前接口、五角色表、协作条件、命令与未实现的回答/RAG/汇总边界；给用户可直接运行的Windows命令。

## 自检与执行交接

已核对：设计中的JEV、辅助证据、五角色、顺序兜底、协作判断、图和CLI均有任务及测试；无共享内存或真正回答执行的范围扩张。接口变更必须同步迁移CLI与测试，不把旧互斥单选当作独立多标签。

建议在当前会话由主Agent顺序实施：五个任务互相依赖，减少并行接口漂移。计划待用户确认；确认前不安装依赖、不改产品代码。
