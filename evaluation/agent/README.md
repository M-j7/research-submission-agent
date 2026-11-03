# Agent 三层评测

复用现有完整 ChatService/LangGraph，EvalScope AgentAdapter 仅组织执行和报告，不替换原工具循环。

- 端到端：独立业务判分（澄清／拒绝可以是正确处理），不把流程 completed 当成功。
- 轨迹：角色路由、工具调用配对、权限与格式拒绝、预算与无进展退出。
- 表现：实际墙钟延迟、模型／工具尝试与供应商已知usage；未知不是0，不估算缺价格的费用。

数据：30道中文草稿题，25业务＋5确定性故障检查；故障由脚本驱动，不代表LLM抗注入或自然故障恢复概率。未完成人工校准，Decisions概率阈值是实验值。

运行使用 `.venv-evaluation/Scripts/python.exe -X utf8 -m evaluation.agent.run --stage smoke --output evaluation/agent/runs/<new-id> --live`。smoke 6题并发1；full 30题并发2；repeat显式运行6题×3，非默认。

运行时使用用户本地凭据，但在内存配置中显式选择 Perplexity Decisions 原生端点与 Flash 1.2，不访问Router或旧JEV，不改写.env。认证／额度错误中止新依赖请求，不自动充值。每题使用独立SQLite、工作区和缓存；知识库Chroma复制到ASCII快照，生产索引不写，生产Redis不使用。后台长期画像维护关闭，即时压缩机制仍保留。

执行、轨迹、判分分别保留JSONL；第一次live阶段原生报告没有独立判分，不可当最终成绩；以 `evalscope/replay` 和 `report.zh.md` 为最终判分报告。报告回放不调用业务Agent。评分失败不应重新跑业务题。

静态汇总：`python -X utf8 -m evaluation.html_report --agent evaluation/agent/runs/<id>`，输出 `evaluation/reports/agent-rag-overview.html`，不发模型请求。

## 已完成的首轮诊断

正式结果 `runs/20261018-full-01/`：30题全部有执行和判分记录。25道业务题为6通过、17失败、2未定；已验证完成比例6/25，有效判分通过率6/23。5项脚本故障保护检查均通过，不能视为真实模型鲁棒性成绩。主要暴露输入预算退出与角色路由偏差，2项语义判分处于实验阈值未知区间。未完成人工校准，不建议直接写成项目上线准确率。

报告已包含RAG及独立联调分区。无付费重跑的重建命令：`python -X utf8 -m evaluation.html_report --agent evaluation/agent/runs/20261018-full-01 --agent evaluation/agent/runs/20261018-smoke-02`。

## 2026-10-20 上下文与瞬时压缩修复

正式 live 链路及 chat CLI 统一使用 `runtime/context.py`：已核实的 `qwen3.8-flash` 上下文为 1,000,000 token，输入上限取70%（700,000），60%窗口（600,000）开始后台压缩，不再把两个70%相乘。累计执行预算与单次上下文预算分开管理。故障脚本与注入的离线依赖仍沿用测试自身预算。

本地计数使用公开 `Qwen/Qwen3.8-Flash-Next` 分词器，对序列化输入加10%估算余量；并非闭源API模型的精确分词或计费结果。真实用量仍取供应商usage。缺少本地分词器时live启动明确报错，不自动下载或退回字节计数。`AGENT_TOKENIZER_PATH` 可指定文件；其他模型必须显式声明其上下文窗口。

本次分词器只下载JSON，不含模型权重或可执行远程代码：

- 来源：`https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer.json`
- 本地：`.cache/tokenizers/qwen3.8-flash-next/tokenizer.json`
- SHA256：`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`

瞬时压缩保留最近两组完整工具交互，仅分批处理尚未压缩的较早工具正文；单条超长原文也无损分段。共享压缩截止时间，超限时最多等待/处理4批后重新检查；任务已有压缩则复用，提交使用短事务与版本校验。失败原因保存在独立compression记录，原文不覆盖。短期“历史摘要＋最近五轮”和长期画像仍走独立流程；评测中后台画像维护依旧关闭。

新运行的manifest记录窗口、计数方式、分词器哈希和执行预算。与旧字节估算预算不同，比较结果时必须注明是组合修复，不能把提升归因于单独扩大窗口；题目、路由期待和评分阈值不修改。
