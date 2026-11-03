# JEV：十项独立相关性 + 需求清晰度（当前接口）

2026-10-07 最新实现：已从互斥choice切换为 C01–C09九个业务相关性＋C10非业务相关性，另问clarity。十一题全是noul是非题，按6/5分成两个并发HTTP请求。**下面旧版本的“三问题、九类单选”等内容仅为历史记录，不是当前线上接口。**

每题返回“是”的独立概率，例如同一句“摘要要求和发表费用”可以同时C02=.88、C07=.93；总和不必为1，不计算虚假的单选confidence。C10只指不含业务诉求的非业务交流／范围外请求，不包括“业务不清楚”。两批共用当前消息与最近六条消息，只用历史补全指代，不把旧诉求重复算作当前意图。

当前函数为 `build_requests()`、`JevIntentRecognizer.recognize()` → `IntentJudgments`。旧 `build_request()`、`parse_response()`、`IntentResult`仅作历史离线兼容，路由不接受它们。两个CLI预览都输出 `jev_requests` 数组，而非单个旧choice请求。

`IntentJudgments`保存十项 `probabilities`、独立 `clarity_probability`、`complete`、模型、父请求ID、两批请求ID与状态、已知用量、耗时。缺题、非法概率、模型批次不一致或服务失败均令判断不完整；不补零、不重试、不伪造用户意图不清。失败批次的计费用量可能未知，已知用量求和以 `usage_complete` 区分。

已实现路由与Agent输入准备，见 [意图路由与多Agent协作](意图路由与多Agent协作.md)。当前三例真实烟雾测试共6次JEV POST，15908输入／615输出tokens；格式单问题、格式＋费用双问题、模糊稿件问题分别进入一个专业Agent、两个专业Agent、通用Agent。不是准确率评测。

```powershell
conda activate ieee-submission-assistant
cd "C:\Users\amorous\Documents\AI_Learning\求职\投稿咨询助手项目"
python .\scripts\judge_intent.py --message "摘要要求和发表费用？" --dry-run
python .\scripts\judge_intent.py --message "摘要要求和发表费用？"
python .\scripts\route_to_agents.py --demo
```

## 历史记录：以下是旧choice版本，保留供对照

日期：2026-10-06。当前实现替代旧的“九类加不明确”单选判断。仅修改 M1 的判断接口；不实现实体抽取、LangGraph 路由、RAG 工具或答案生成。

2026-10-07 补充：九类各五条共享示例已建立；JEV 每类判断标准现附一条代表性示例。三路与融合评分的当前入口见 [三路意图识别与融合](三路意图识别与融合.md)。本页第6节保留10月6日未附示例时的实测，不代表更新后模型结果。

2026-10-07 当前接口：增加独立对话性质判断，将社交接待、范围外请求与真正不明分开。当前一次请求三个问题；8条基线与2条历史复测见 [对话性质实测](JEV对话性质实测-20261007.md)。后文第6节仍保留旧版两问题的历史结果。

## 1. 一次请求，三个问题

同一个 `state`（当前消息 + 最近六条历史消息），一次 POST `/v1/systemone`，发送三个独立问题：

| 问题 ID | 类型 | 判断内容 | 服务返回 |
|---|---|---|---|
| dialogue | choice | 本轮是业务咨询、社交接待、范围外请求，还是性质不明？ | BUSINESS / SOCIAL / OUT_OF_SCOPE / UNCERTAIN 的竞争概率、主选项与 confidence |
| clarity | noul，是非题 | 用户这次想咨询或解决的具体事项是否清晰？ | `noul`：是的概率，0–1 |
| intent | choice | 诉求属于九类业务中的哪类？ | 主类别、九项竞争概率、分类 confidence |

不是三次串行模型调用；三个问题共同提交，业务分类不会读取其他问题的结果。程序先限制业务候选是否适用；具体清晰度阈值和 Agent 路由尚未实现。

`UNCLEAR` 已从九类的选项中移除。OPTIONS 只含 C01–C09；不明确不再与业务类别争夺同一份概率。

## 2. 是与否的标准

问题：结合当前消息与提供的历史，用户本轮想咨询或解决的具体事项是否清晰？

“是”：能够确定具体诉求、关键动作和必要指代。两个事项都讲清楚也是“是”，不要求确定哪一个更重要。未知答案、缺少检索证据或需要补充不改变诉求的业务细节，不影响“是”。

“否”：没有提出具体诉求；或关键对象、动作、指代缺失，让同一个请求存在不同事项的合理解释，历史也不能消除，必须先问清用户究竟在问什么。

边界：

- “摘要最多多少词？上传文件报错怎么排查？”：两个明确事项，应清晰。
- “之前提交的那个，现在还能退吗？”且无历史：钱款退回与稿件撤回都是合理解释，应先消歧。
- “摘要最多多少词？”没有期刊：知道用户在问摘要长度，缺的是回答条件，不是意图。
- 有必要历史时，“这个怎么回复？”可以变清晰；不能不看历史就判否。
- “明天南京会不会下雨？”本身是清晰请求，但 dialogue 应为 OUT_OF_SCOPE；清晰度不是域内判定。
- “你好”“谢谢”“你能做什么”：SOCIAL，不应因没有具体业务诉求就追问用户。SOCIAL 下 clarity 不是触发业务澄清的依据。
- “我的稿件出问题了”：BUSINESS，但具体诉求可能不清楚。
- 无历史的“这个怎么办”：UNCERTAIN；有最近唯一未解决事项且本轮明确承接时，可以补全为 BUSINESS。
- “你好，摘要最多多少词”：BUSINESS，实际业务优先于开头寒暄。

九个业务定义保持不变，见 `src/submission_assistant/intent/catalog.py`。

## 3. 请求结构

```json
{
  "model": "jev-latest",
  "state": {
    "domain": "IEEE Access 投稿咨询",
    "current_message": "本轮用户消息",
    "history": []
  },
  "questions": {
    "dialogue": {
      "type": "choice",
      "instructions": "本轮对话性质，区分业务、社交、范围外和性质不明",
      "criteria": {"BUSINESS": "论文投稿业务，具体需求可不清楚", "SOCIAL": "问候感谢或能力介绍", "OUT_OF_SCOPE": "明确的无关请求", "UNCERTAIN": "无法确定在说什么"}
    },
    "clarity": {
      "type": "noul",
      "instructions": "需求是否清晰及其判断边界",
      "criteria": {"true": "是的标准", "false": "否的标准"}
    },
    "intent": {
      "type": "choice",
      "instructions": "只判断本轮业务类别，不判断需求清晰度",
      "criteria": {"C01": "范围与适配", "C02": "格式规范", "C03": "投稿材料", "C04": "伦理署名", "C05": "平台操作", "C06": "审稿返修", "C07": "费用退款", "C08": "出版版权", "C09": "撤稿转投申诉"}
    }
  }
}
```

上面是结构简写，不是生产标准全文。实际指令和描述在 `catalog.py` 的 `CLARITY_INSTRUCTIONS`、`CLARITY_CRITERIA`、`INSTRUCTIONS`、`OPTIONS`。

服务文档：[当前网关文档](https://jevmodel.org/docs)。支持每次1–8个问题；noul 支持 true/false 说明，choice 支持2–20个选项。当前三个问题，choice 分别四个和九个选项，均在范围内。共享 state 不意味着费用等同旧版；额外标准会增加输入 tokens，实际扣费以后台为准。

## 4. 输出与迁移

保留九项原始 `probabilities`、业务分类 `confidence`、排名、分差、模型、请求 ID、token 用量和耗时，但它们可能只是被迫九选一的诊断数据。`dialogue_type` 非 BUSINESS 或未知时，公开 `to_dict()` 的 `selected_intent`、`selected_name`、`selected_probability` 都为 null；原始选择单列为 `business_candidate`。内部 dataclass 的 `selected_intent` 仍保存原始业务 choice，消费者不得绕过 `to_dict()` 和适用性判断用于路由。

新增：

- `clarity_probability` = clarity.noul = P(需求清晰／是)。
- `unclear_probability` = 1 - clarity_probability = P(需求不清晰／否)。这是是非题的互补概率，不是分类 confidence。
- `dialogue_type`、`dialogue_probabilities`、`dialogue_confidence`：独立的对话性质结果。
- `business_intent_applicable`：主对话类型为 BUSINESS 时 true，否则 false；旧直接构造结果缺少性质时为 null。它不代表具体需求清楚，不是 Agent 路由决定，也不是校准后可信度门槛。

移除旧的 `needs_clarification` 布尔输出：它以前依赖 choice 是否选中 UNCLEAR，已经不适用。暂时不按0.5或其他未校准阈值生成自动澄清／路由决定；接入 LangGraph 时再定义允许直接处理、需要澄清和需要复核的区间。

低清晰度时，九选一仍会被迫返回某个类别。哪怕 C09 得分93%、分类 confidence=0.92，也不能覆盖清晰度仅49%的事实，更不能单凭分类最高分就进入专业 Agent。当前只提供判断数据，不决定路由。

新响应必须同时含 `answers.dialogue`（type=choice，完整四项有效概率）、`answers.clarity`（type=noul，noul为有限0–1数值）和 `answers.intent`（type=choice，完整九项有效概率）。缺失任一问题不补默认值、不从其他判断推导。两问题历史响应与旧十项响应拒绝混用；本轮不存在已上线的路由消费者需要迁移。融合入口在 JEV 失败时保留辅助分，但不发布有效业务类别，也不伪造 UNCERTAIN；服务不可用与模型判断“不明”不同。

## 5. 运行

仍使用 `ieee-submission-assistant` Miniconda 环境，项目根目录 `.env` 中的 JEV 配置不变，无新模型或依赖安装。

```powershell
cd "C:\Users\amorous\Documents\AI_Learning\求职\投稿咨询助手项目"
conda activate ieee-submission-assistant

# 不联网预览三个问题，不读密钥。
python .\scripts\judge_intent.py --message "之前提交的那个，现在还能退吗？" --dry-run

# 真实调用：可能消耗服务余额。
python .\scripts\judge_intent.py --message "之前提交的那个，现在还能退吗？"

# history.json 是 UTF-8 JSON role/content 消息数组，仅取最后六条。
python .\scripts\judge_intent.py --message "这个怎么回复？" --history-file .\history.json

# 离线模块契约测试，不调用模型。
python -X utf8 -m unittest discover -s tests -v
```

## 6. 本轮真实验证

使用同一当前配置，模型实际解析为 `jev-1.13.0`。无历史，各一次请求，每次同时询问两个问题：

| 问题 | P(需求清晰) | 分类候选 | 分类 confidence | 输入/输出 tokens |
|---|---:|---|---:|---:|
| 之前提交的那个，现在还能退吗？ | 0.49 | C09=0.93、C07=0.05、C05=0.02 | 0.92 | 1894 / 117 |
| 我准备向 IEEE Access 投稿，摘要最多允许多少词？另外，投稿系统上传文件时一直报错，应该怎么排查？ | 0.94 | C05=0.67、C02=0.32、C03=0.01 | 0.62 | 1924 / 117 |

第一条请求 ID `0296fe7e-50d1-4ef6-9847-cb6eb9904a0f`，耗时2322.3ms；第二条 `8ba0e8a6-26a9-4518-85b0-57898a1a92c3`，耗时3135.52ms。两次合计3818输入、234输出 tokens。

明确双意图没有被当成不清晰；但歧义例的清晰度49%接近五五开，尚不能称为稳定识别成功。对比结果也不能证明中文歧义问题整体准确率或清晰度校准情况。没有通过阈值或程序强行改结果，没有自动追加评测调用。

旧版提示词的成功与失败实验保留在 [历史对比记录](JEV提示词优化对比-20261006.md)。后续优先评测单诉求、复合诉求、真正指代不明、历史消歧、缺少回答条件和域外清晰请求，不能只优化一个“退”字样本。

人工预期问题集 `tests/data/intent_cases.json` 已同步为29条：独立 `expected_clarity` 是非标注，`expected_intents` 仅用 C01–C09。不清晰样本与域外样本不强求被迫九选一的候选正确；域外样本即使清晰也不能计入九类业务准确率。它们是待评测的人工预期，不是模型实测结果，本轮没有批量调用。
