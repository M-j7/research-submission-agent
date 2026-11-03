# RAG 检索评测

只评检索，不生成业务回答，不运行 RAGAS。保留整个 IEEE 知识库和现有 Chroma 子块索引，以最终返回的父块 ID 评分。

## 数据从哪里来

`cases.json` 保存 15 个业务问题的中英文版本、参考说明、明确的父块 ID 和支持原文。它是来源核验过的 Agent 草稿，不是人工金标准。

`data/ieee-bilingual-v1/` 是生成的数据集：

| 文件 | 内容 |
|---|---|
| corpus.jsonl | 全部 966 个父块及来源元数据，包含未标注的干扰文档 |
| queries.jsonl | 15 组中英文等义问题，共 30 条查询 |
| qrels.jsonl | 问题 ID、相关父块 ID、二元相关性分数 1 |
| annotations.jsonl | 问题、支持摘录、来源、参考说明及 agent_draft 审核状态 |
| manifest.json | 计数、评分粒度、文件哈希及已知局限 |

qrels 的一行如下：

```json
{"query-id":"file_size.zh","corpus-id":"11f8608b845b3651:00002","score":1}
```

这表示该父块与此问题相关，不表示它是唯一正确文档，也不表示它必然独立覆盖问题的所有条件。中英文版本共享证据标注。

构建时校验：ID 不重复、支持父块存在、原文摘录确实出现、中英文查询完整、中文没有被 Windows 编码破坏。标注错误会中止导出，不能悄悄制造标签。

已经导出的数据集不允许覆盖：完全相同的输入再次构建不会改写文件；修改问题、标注或语料后，必须通过 `--output` 指定新的版本目录。采集时通过 `--dataset` 选择相应目录，确保旧实验还能读取原版本。各目录的 `dataset_version` 目前标记同一 schema 版本，具体数据版本由目录和 manifest 哈希区分。

## 四组对照

| 基线 | 实际流程 |
|---|---|
| B1 | 原问题向量召回 → 父块回溯 |
| B2 | 原问题向量＋BM25 → RRF → 父块回溯 |
| B3 | B2 → 原问题 Reranker → 父块回溯 |
| B4 | 生产 Multi-Query＋JEV → 多查询双路召回 → RRF → 原问题重排 → 父块回溯 |

复用生产 `MultiQueryRAG._search`，仅在缓存边界之下调用；不读取或清空 Redis 的最终结果缓存。B1–B3 共享一次原问题召回，减少付费请求；因此不能拿三组的处理时间比较独立端到端延迟。

固定每路子块召回 30、RRF 常数 60、融合候选 60、重排候选 30、最终父块 5。记录每个问题的证据、查询改写、警告、状态和耗时。失败题保留空排名；降级不冒充完整链路成功。

## 如何运行

在项目根目录运行。应用环境负责采集；独立的 `.venv-evaluation` 负责 EvalScope/MTEB 计分，避免升级应用依赖。

```powershell
# 不联网：生成并验证数据集
& 'C:/Users/amorous/miniconda3/envs/ieee-submission-assistant/python.exe' -X utf8 -m evaluation.rag.dataset

# 联网、有模型调用费用：输出目录必须是新的，防止覆盖历史实验
& 'C:/Users/amorous/miniconda3/envs/ieee-submission-assistant/python.exe' -X utf8 -m evaluation.rag.collect --live --output 'evaluation/rag/runs/my-run'

# 不调用业务模型：对真实排名回放计分，生成 EvalScope 报告
& '.venv-evaluation/Scripts/python.exe' -X utf8 -m evaluation.rag.score --run 'evaluation/rag/runs/my-run' --evalscope

# 评测模块测试，含官方框架集成
& '.venv-evaluation/Scripts/python.exe' -X utf8 -m unittest discover -s tests -p 'test_rag_evaluation*.py' -v
```

`--limit 2` 仅用于冒烟测试，会在 manifest 中保留实际查询范围；小规模结果不能冒充完整数据集结果。

## 如何看结果

`REPORT.md` 是中文摘要；`summary.json` 是分基线、分语言的汇总；`scores.jsonl` 是逐题指标；`rankings.jsonl` 是真实排名和证据。`evalscope/` 中是 EvalScope 原生报告、预测回放和评分记录，可以在其 Dashboard 查看。

MTEB 官方 retrieval_metrics 与 pytrec_eval 计算 Recall、NDCG、Precision、MRR、HitRate 的 @1/@3/@5；没有自己重写公式。EvalScope 使用自定义检索 benchmark 回放并统一报告，不使用其默认 Embedding 流程替换本项目的检索器。这里的 replay/model 名称只是框架接口适配，绝不代表又调用了一次生成模型。

主看 Recall@5 和 NDCG@5；失败问题按零分纳入，不从分母删除。状态计数用于区分无结果、降级和正常执行。当前 NDCG 使用二元相关性，不冒充分级相关性标注。

## 使用边界与下一步

### 用户授权的无 JEV 对照实验

JEV 余额不足时，可以仅在评测中运行 `--skip-jev --baselines B4`：保留原问题，将最多三个格式合法且去重后的改写全部参与召回，再走原问题重排。此模式完全不创建 JEV 客户端、不发送 JEV 请求，也不伪造“校验通过”判断；生产默认路径保持校验失败时不放行改写。

```powershell
& 'C:/Users/amorous/miniconda3/envs/ieee-submission-assistant/python.exe' -X utf8 -m evaluation.rag.collect --live --skip-jev --baselines B4 --output 'evaluation/rag/runs/my-no-jev-run'
& '.venv-evaluation/Scripts/python.exe' -X utf8 -m evaluation.rag.score --run 'evaluation/rag/runs/my-no-jev-run' --evalscope
```

新结果通过 `semantic_validation=skipped_by_user` 和 `experimental_multiquery_no_jev` 标记；可与旧 B3 比较，但不能当成“JEV 校验版 Multi-Query”的成绩。改写仍保留格式、长度和数量检查，跳过的是语义相关性判断。

- 问题从现有资料构造，尚非真实用户分布；15 组双语问题不能视作 30 个独立业务问题。
- qrels 未穷尽全部等价证据；未标注项会被标准指标当作不相关，可能造成评分偏差。
- 正式对外报告前应审阅低分题和不同基线的候选池，补标等价证据、确认语言等义性，冻结一个人工审核版本。不能因某结果被当前系统返回，就自动把它加入正例。
- 现有资料按采集快照评测，不保证其中金额或政策为官网最新版本。
- 保留第一次运行结果；不要为了更好看的成绩修改标签或问题后覆盖原实验。

## 生成评测（官方 RAGAS）

生成阶段固定以上真实检索证据，不重新检索，不拿 qrels 正例替换漏检结果，不调用 JEV。
两组各 30 条：原问题 B3 与实验性无 JEV Multi-Query B4。`--smoke` 为 APC 问题中英文 × 两组，共 4 条。

```powershell
# 仅向独立评测环境安装追加依赖；不升级应用环境
& '.venv-evaluation/Scripts/python.exe' -X utf8 -m pip install -r evaluation/requirements-generation.txt

# 付费回答生成；新目录，不覆盖历史运行
& 'C:/Users/amorous/miniconda3/envs/ieee-submission-assistant/python.exe' -X utf8 -m evaluation.rag.generate --live --smoke --output evaluation/rag/generation-runs/my-smoke

# 付费官方 RAGAS 评分，和不付费的分数回放可视化
& '.venv-evaluation/Scripts/python.exe' -X utf8 -m evaluation.rag.ragas_score --live --run evaluation/rag/generation-runs/my-smoke
& '.venv-evaluation/Scripts/python.exe' -X utf8 -m evaluation.rag.generation_report --run evaluation/rag/generation-runs/my-smoke --native

# 全量：生成命令去掉 --smoke，另选新目录，评分/回放命令对应修改目录
```

`inputs.jsonl` 保存原问题、实际上下文、独立参考答案及来源哈希；生成提示仅允许问题、语言和实际证据，绝不包含 reference。
`answers.jsonl` 逐条保存回答、结束原因、生成状态、耗时与已知用量。
`ragas/ragas-inputs.jsonl` 是评分四字段；`ragas/scores.jsonl` 包含每题五指标及官方裁判结构化输出；`summary.json`、`report.zh.md` 和 `evalscope/` 为汇总及原生可视化。
自定义评分目录用评分端 `--output`、报告端 `--scores` 指定，旧评分保留不覆盖。

主看 Faithfulness 与 AnswerCorrectness，辅助看 AnswerRelevancy；ContextPrecision（with reference）与 ContextRecall 诊断实际上下文。
采用 RAGAS 0.4.3 官方 legacy 指标实现并固定依赖（不是自写近似公式）：Correctness 默认事实/语义权重 0.75/0.25，Relevancy strictness=3，以三个独立 n=1 请求适配现有服务。
裁判提示采用同一官方英文模板，支持双语输入；模板哈希、模型、版本和参数写入评分 manifest。

真实零分参与均值；异常、NaN 和未成功生成保留为 null，显示总数、有效数、不可用数。EvalScope 仅回放既有数值，缺失质量键不补零；不可用率另列诊断，完整分母以中文报告为准。
所有生成/评分输出目录不可复用。缺失已知用量不代表没有费用，Embedding 用量目前不可得。
回答生成最多 2 次尝试、单次 60 秒和总时限 125 秒；裁判 SDK 无重试，RAGAS 仅 1 次尝试，关闭隐藏格式修复调用，单请求 60 秒、每指标 180 秒。
独立实例隔离裁判可变采样设置，限制同时评分的样本数为 2。框架的旧 wrapper 弃用告警是已知版本约束，不代表使用了其他评分指标。

注意：这只是固定上下文 RAG，不是完整 Agent 评测；参考答案为待人工复核的 Agent 草稿；同一千问模型兼任生成器与裁判存在自偏好；B4 改写未经 JEV 语义校验。小样本不能直接作为简历准确率。
