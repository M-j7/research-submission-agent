# 意图识别 → 路由 → Agent接收：实现与验证

日期：2026-10-07。JEV独立判断、确定性路由与多Agent协作判断已实现。**本次终点是准备Agent输入，不是完整聊天回答。**

## 1. 现在实际运行什么

```text
本轮消息＋最近六条历史
  → JEV（clarity＋十类相关性，两批6/5）
    同时：Embedding＋BM25匹配75条示例，提供辅助证据
  → 程序路由（不再调用一个LLM来选Agent）
  → 一个或多个Agent实例prepare()
  → 返回角色、原因、负责意图、静态提示、动态输入、工具绑定状态
```

对应代码：

- `src/submission_assistant/intent/jev.py`：`build_requests()`、`recognize()`与两批noul严格解析。
- `src/submission_assistant/intent/contracts.py`：`IntentJudgments`、`BatchStatus`、完整性与已知用量。
- `src/submission_assistant/intent/ensemble.py`：三路入口，JEV主判、辅助关系与诊断分。
- `src/submission_assistant/routing.py`：实际使用的意图映射、阈值与协作判断。
- `src/submission_assistant/agents/`：统一BaseAgent、五种角色配置与工具权限过滤。
- `src/submission_assistant/graph.py`：LangGraph三个节点。
- `scripts/route_to_agents.py`：预览、受控演示、已有判断文件与真实入口。

继续使用 `ieee-submission-assistant` Miniconda环境；已安装LangGraph1.2.14、langchain-core1.6.7并记录到environment.yml。配套依赖将websockets17.2调整到16.1.1，`pip check`通过，旧RAG测试通过。没有修改.env密钥、原始资料或Chroma索引；复合问题实测创建了新的75条意图模板Embedding缓存，这是意图示例缓存，不是RAG库重建。

## 2. 五个Agent，不是十个Agent

| Agent类 | 职责／意图 | 允许工具标识（尚未绑定） |
|---|---|---|
| GeneralAgent | 接待、澄清、范围说明、技术失败兜底 | 无 |
| SubmissionRequirementsAgent | C01范围、C02格式、C03材料 | search_submission_requirements |
| EditorialWorkflowAgent | C05平台、C06审稿返修、C09撤稿转投申诉 | search_editorial_workflow |
| PublicationPolicyAgent | C04伦理署名、C08出版版权与发表后更正撤回 | search_publication_policy |
| PublicationFeesAgent | C07费用减免退款 | search_publication_fees |

五个角色直接继承BaseAgent，共用prepare逻辑，专业类只改变profile里的提示词和允许工具。General也是BaseAgent的子类，不是专业角色的父类，避免专业Agent继承不适合的接待话术。

`system_prompt`为静态角色与证据约束，用户message、history、intent_ids、fallback_reason在独立动态input里。角色允许某工具并不代表它已经可用：只有同时在注册表和白名单中才能进入registered_tools。当前检索工具还没封装，因此正常专业角色显示registered_tools=[]、对应unbound_tools，不能声称已经查过库。

## 3. 什么时候多Agent协作

当前实验门槛：clarity >= .75、单项相关性 >= .80，CLI可以修改，输出保存实际门槛。它们未校准，不是正确率或行业标准。

按顺序判断：

1. 缺少完整JEV或任何必需批次失败：通用Agent，recognition_failed。
2. C10高且无高业务项：通用Agent，non_business；即便clarity低也先自然接待／说明范围。
3. C10与业务项同时高：通用Agent，conflicting_judgments。
4. clarity低：通用Agent，unclear_request；不靠BM25猜“他其实想问什么”。
5. clarity够高但业务项都不到门槛：通用Agent，uncertain_intent。
6. 其余明确业务：映射角色，同角色去重；去重后的专业角色数量 > 1 才multi_agent=true。

| 问题 | 意图 | 路由 | 协作 |
|---|---|---|---|
| 摘要最多多少词 | C02 | 规范Agent | 否 |
| 摘要多少词＋提交哪些材料 | C02＋C03 | 一个规范Agent处理两项 | 否 |
| 摘要多少词＋发表费用多少 | C02＋C07 | 规范＋费用Agent | 是 |
| 审稿状态＋返修＋撤稿条件 | C06＋C09 | 一个流程Agent | 否 |
| 你好／能力咨询／范围外任务 | C10 | 通用Agent | 否 |
| 我的稿件出问题了 | 可能有业务候选，但clarity低 | 通用Agent先澄清 | 否 |

每个专业Agent收到完整原始问题，但intent_ids只列自己负责的事项。没有另付费拆成子问题；本阶段协作输出仅表示“需要这些角色共同处理”，不是已经并发完成工作、解决依赖顺序或合并答案。

缺少RAG证据不是需求不清，不能据此改通用兜底。费用问题只是背景提到撤稿，也不应仅凭“撤稿”一词再加流程Agent；这一语义由JEV判断，程序只映射明确命中。

## 4. 三路现在分别干什么

JEV是主判，能读取历史并独立识别复合诉求。Embedding/BM25只是模板对照证据：support、alternative、tie、no_match、unavailable等，不能抬高JEV概率、覆盖JEV、单独发起协作或替缺失批次补题。

两个网络服务并发；JEV内部又并发两批。BM25虽然本地执行，也按来源隔离异常；任何辅助路径失败只标记降级，完整JEV仍能路由。调用者取消传播，不当成通用Agent正常回应。服务错误只输出脱敏说明，没有自动重试。

`diagnostic_fusion`及兼容顶层scores保存旧0.7/0.2/0.1九业务类正分占比公式，**路由不用它**。`primary_candidates`按默认.80仅作展示；真实角色以routing_decision为准，尤其当CLI修改门槛时不要把展示候选当作路由结果。原始独立概率总和可以大于1，无单选confidence或总和为1的约束。

## 5. 直接运行

```powershell
conda activate ieee-submission-assistant
cd "C:\Users\amorous\Documents\AI_Learning\求职\投稿咨询助手项目"

# 六类受控演示，不读.env、不联网，不能当模型测试。
python .\scripts\route_to_agents.py --demo

# 只看请求与映射，也不联网。
python .\scripts\route_to_agents.py --message "摘要最多多少词，发表费用多少？" --dry-run

# 完整三路真实识别，产生JEV两批和Embedding调用费用。
python .\scripts\route_to_agents.py --message "IEEE Access 摘要最多多少词，发表需要支付多少费用？"

# 单测JEV接路由，不付Embedding费；只缺辅助不会阻止完整JEV路由。
python .\scripts\route_to_agents.py --message "我的稿件出问题了，帮我看看。" --jev-only

# 读取已有报告中的judgments可另存为符合契约的JSON，受控复放，不读.env。
python .\scripts\route_to_agents.py --message "摘要和费用？" --judgments-file .\judgments.json

# 自定义实验门槛；不是生产校准建议。
python .\scripts\route_to_agents.py --message "摘要和费用？" --relevance-threshold 0.85 --clarity-threshold 0.80

# 保存生成报告；拒绝覆盖已有文件。
python .\scripts\route_to_agents.py --demo --output .\docs\实测结果\新的离线演示.json

python -X utf8 -m unittest discover -s tests -v
python -m pip check
```

有历史时添加 `--history-file .\history.json`，内容为UTF-8 JSON的role/content数组，只接受user/assistant。入口在读密钥与付费请求前验证输入。CLI调用图时关闭LangSmith追踪上下文，不额外将消息送往追踪服务、不改全局环境。

正常通用兜底也是合法图结果，路由CLI退出0；不要只凭退出码认定JEV成功，应检查recognition.judgments.complete和fallback_reason。JEV单独CLI的不完整结果退出1；三路CLI降级时退出1。命令行非法输入／配置／文件读写失败退出1。

## 6. 真实烟雾验证（不是总体准确率）

配置模型实际返回jev-1.13.0。共三个合成问题、六次JEV POST，无重试：

| 问题 | clarity | 主要独立相关分 | 路由 | JEV输入／输出tokens |
|---|---:|---|---|---:|
| IEEE Access摘要最多多少词？ | .95 | C02=.97，C10=.05 | 规范Agent，single | 5294／205 |
| IEEE Access摘要最多多少词，发表需要支付多少费用？ | .95 | C02=.88，C07=.93，C10=.04 | 规范＋费用Agent，multiple | 5316／205 |
| 我的稿件出问题了，帮我看看。 | .12 | C10=.09；具体诉求不清 | 通用Agent，unclear_request | 5298／205 |

JEV合计15908输入、615输出tokens；不推算实际金额，扣费以服务后台为准。复合问题同时实测了完整三路，Embedding含75条模板首次初始化＋本轮查询，共1871prompt/total tokens。两路最高候选都为C02、关系supports_primary；费用C07来自独立JEV，不因检索只偏向格式而漏掉第二角色。

完整报告：

- [单业务](实测结果/20261007-jev-independent-single.json)
- [三路复合业务](实测结果/20261007-jev-independent-compound.json)
- [模糊业务](实测结果/20261007-jev-independent-unclear.json)

两批使用相同state但分两次发送，会重复部分输入；本轮每题描述也较长，约5300输入tokens/问题，这是当前成本，不宣称比旧三题省钱。后续可评测是否先筛候选或压缩标准，但不能为省钱跳过C10、清晰度或复合诉求必要判断。

## 7. 验证与尚未实现

主tests目录94项通过；规则清洗6、语义分割5、父子检索16项通过，合计121项。原先4个辅助字段报错测试已补齐功能，保留且通过，没有删减、skip或xfail。覆盖两批完整性、非法概率、未知用量、服务失败、取消、并列/无匹配、辅助不覆盖主判、同角色去重、跨角色协作、所有通用原因、静态/动态输入隔离、工具权限和真实图入口。

一次独立只读审查发现BM25异常可能阻断JEV，在新增图级回归中先复现失败，再修复来源隔离，全套主测试94项通过。没有用三条烟雾结果宣称准确率，尚需独立多标签评测集、清晰度校准、边界与带历史用例。

下一步尚未做：BaseAgent调用回答LLM的共享工具循环；已有RAG封装与绑定；查询改写＋JEV守门；多Agent执行与有证据的答案汇总；Ragas端到端回答评测；记忆重设计。现在的Agent是共用基类的输入准备骨架，不要在面试里说已经跑通完整多Agent回答系统。

保留旧choice构造/解析作离线历史兼容，旧实验文档增加历史标记，当前CLI与路由不使用它们。项目不是Git仓库，未创建worktree／提交；限定修改源码、测试、文档、依赖记录，未改用户密钥和资料库。
