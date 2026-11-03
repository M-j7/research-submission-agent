# JEV 意图识别：九类业务 + 不明确

> 历史版接口记录，现已由 [独立需求清晰度 + 九类意图](JEV需求清晰度与意图分类.md) 替代。当前代码不再发送十选项，不再输出旧 `needs_clarification`；下文保留实现过程，不作当前运行契约。

日期：2026-10-06。范围：只实现 M1 的意图判断，不做实体提取、Agent 路由、RAG 调用或回复生成。

## 1. 当前实现

当前消息 + 调用方传入的最近历史 → 一个 JEV `choice` 问题（十个选项）→ 校验响应 → 输出完整概率分布，供未来 LangGraph 节点消费。

不是十次请求，也不是九个独立的是否判断。每次 `recognize()` 最多发送一次请求，不自动重试。没有余额时返回服务错误，不生成模拟结果。既有 RAG 代码和索引未改动。

代码入口：

- `src/submission_assistant/intent/catalog.py`：十个选项名称、判断标准和公共指令。
- `src/submission_assistant/intent/jev.py`：环境配置、请求构造、异步 HTTP 调用、响应校验、结果结构。
- `scripts/judge_intent.py`：命令行入口，支持不联网的请求预览。
- `tests/test_jev_intent.py`：本地接口契约测试；HTTP 边界使用测试响应，不证明模型分类准确率。
- `tests/data/intent_cases.json`：待真实服务验证的人工预期问题集，不是已跑出的模型结果。

## 2. 十个选项

| 标识 | 名称 |
|---|---|
| C01 | 期刊范围与适配 |
| C02 | 稿件格式与写作规范 |
| C03 | 投稿材料与声明 |
| C04 | 学术伦理与作者署名 |
| C05 | 投稿平台操作与故障 |
| C06 | 审稿流程、状态与返修 |
| C07 | 费用、减免与退款 |
| C08 | 录用后出版与版权许可 |
| C09 | 撤稿、转投与申诉 |
| UNCLEAR | 不明确 |

实际咨询诉求决定类别，背景关键词不触发。例如“上传投稿信失败”主要是 C05，不因“投稿信”三个字就判 C03。

“不明确”表示结合提供的历史仍不知道用户在问什么，不表示缺少可回答的证据。明确问“摘要多少词”但未说明期刊，是 C02；明确同时问格式和上传故障，也不是“不明确”。

当前提示词有已知局限：无历史的“之前提交的那个，现在还能退吗？”仍高概率猜为 C09；尝试加强歧义定义后，歧义样本通过但明确双意图回归失败，因此候选已撤回，详见[提示词实验记录](JEV提示词优化对比-20261006.md)。下面的设计意图不是已经由模型稳定保证的行为。

“其他／范围外”与“不明确”不同。当前按用户确定的十个选项实现，没有第十一个范围外标签；因此本模块只承诺在投稿咨询域内工作。天气等明确域外请求不应依据这十类强制路由，后续接待／范围检查需要单独设计。问题集中保留域外样本，但不预设它应该被判成 UNCLEAR。

## 3. 请求与历史

`state` 是结构化输入：

```json
{
  "domain": "IEEE Access 投稿咨询",
  "current_message": "这个应该怎么回复？",
  "history": [
    {"role": "user", "content": "编辑要求我逐条回复大修意见。"}
  ]
}
```

历史仅接受 `user/assistant` 的 `role/content`，取最后六条消息（不是保证完整的三轮）；不读取长期用户画像，不自动存取会话，不修改调用方的历史。无历史时是空数组。历史用于补全当前诉求，不把前面已经结束的问题当作当前意图。

没有历史的“之前提交的那个，现在还能退吗？”应判不明确；明确历史说明正在询问已支付的发表费用后，同一句可能归 C07。

当前服务限制：序列化 state ≤ 8000 字符，criteria ≤ 2000 字符，instructions ≤ 1800 字符。超过限制时本地报错，不悄悄截断用户问题。limits 按当前网关文档实现，不冒充 TypeSafe 原生服务的全局限制。

## 4. 配置与运行

继续使用项目根目录 `.env`：`JEV_BASE_URL`、`JEV_API_KEY`、`JEV_MODEL`；可选 `JEV_TIMEOUT_S`，默认 30 秒。已将原先空白的模型名补为该服务文档给出的 `jev-latest`；没有改你的地址、密钥或 Qwen 配置。`JEV_PROVIDER=jevmodel` 作为服务标记，目前没有多供应商切换逻辑。

配置地址可以是 `https://jevmodel.org`、`https://jevmodel.org/v1` 或完整 `/v1/systemone`；统一 POST 到一次判断接口。禁止 URL 内嵌密钥、HTTP 和自动跳转。不打印服务错误原文，避免异常正文泄露凭证。

在 PowerShell 中：

```powershell
cd "C:\Users\amorous\Documents\AI_Learning\求职\投稿咨询助手项目"
conda activate ieee-submission-assistant

# 只查看发送内容：不读密钥，不联网，不消耗次数。
python .\scripts\judge_intent.py --message "IEEE Access 摘要最多多少词？另外上传文件失败怎么办？" --dry-run

# 真实调用：会使用 .env 的 JEV 服务，可能消耗服务余额。
python .\scripts\judge_intent.py --message "IEEE Access 摘要最多多少词？另外上传文件失败怎么办？"

# history.json 必须是 UTF-8 JSON role/content 数组，格式见上节。
python .\scripts\judge_intent.py --message "这个怎么回复？" --history-file .\history.json

# 不调用模型的模块测试。
python -X utf8 -m unittest discover -s tests -v
```

现有 Miniconda 环境已有 httpx 和 python-dotenv；本次未安装额外模型或调用 Qwen。依赖版本已写入 `environment.yml`。

## 5. 输出含义

以下仅为本地接口测试的示例值，不是本次真实模型输出，也不能当成十选项实际效果：

```json
{
  "selected_intent": "C05",
  "selected_name": "投稿平台操作与故障",
  "selected_probability": 0.50,
  "confidence": 0.41,
  "probabilities": {
    "C01": 0.00, "C02": 0.46, "C03": 0.04, "C04": 0.00,
    "C05": 0.50, "C06": 0.00, "C07": 0.00, "C08": 0.00,
    "C09": 0.00, "UNCLEAR": 0.00
  },
  "top_two_gap": 0.04,
  "needs_clarification": false
}
```

实际还有十项排序列表 `ranked_options`、模型标识、token 用量、请求 ID、耗时。

- `selected_probability`：服务所选类别的概率。
- `confidence`：服务另行返回的置信度，不用最高概率替代，不自行编造。
- 十项概率竞争，和约为 1，不是十个独立标签的成立概率，更不是实测正确率。
- 多个类别得分较高，既可能是复合问题，也可能是模型分不清；保留第二、第三候选，但不靠固定阈值自动宣布多意图或多 Agent。
- `needs_clarification` 目前只表示服务选中了 UNCLEAR；尚未加入基于低置信度／分差的拒判策略。
- 缺选项、非法分数、概率和错误、choice 不是最高分、缺置信度等都报协议错误，不自动补零或归一化。
- 网络失败、401、402、422、429、502 与意图不明是不同情况，不应统一走澄清话术。

## 6. 后续接 LangGraph 的位置

可在意图节点调用 `await recognizer.recognize(state["message"], state.get("history"))`，再返回 `{"intent_result": result.to_dict()}`。当前还没有安装／实现整张 Agent 图，不能声称已经集成。

后续顺序：验证真实十选项效果 → 讨论拒判与复合意图策略 → 确定业务类别到 Agent 的映射 → 接入 LangGraph。不要把 C01–C09 直接当作九个 Agent。

## 7. 服务依据与验证边界

已阅读当前配置服务的 [JEV API 文档](https://jevmodel.org/docs)，核对 `POST /v1/systemone`、Bearer 认证、`state/questions/criteria` 请求及 `answers/choice/probabilities/confidence` 响应。该网关公开支持 2–20 个 choice 选项，十选项在文档限制内；一次只有一个问题，也没有触及八问题上限。

TypeSafe 本体及其他转发服务的限额、密钥和地址不能互换；保留当前配置，未把密钥发送给其他域名。

初次实现只做了离线契约测试和请求预览。随后用户授权了真实调用，当前地址和密钥已成功返回结果；针对一个明确双意图样本完成了提示词修改前后对比，记录见 [JEV 提示词优化对比](JEV提示词优化对比-20261006.md)。这不代表账户余额充足或整体分类质量达标。已知模糊退款例的历史依赖、多意图概率稀释、选项边界混淆及选项顺序偏差仍需实际评测。
