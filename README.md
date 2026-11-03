# 科研投稿咨询 Agent

面向 IEEE Access 投稿咨询的多 Agent 应用，包含父子块 RAG、会话记忆、受控工具调用、自动化评测以及 React 聊天界面。

这是一个本地单用户研究与工程验证项目。支持不需要 API key 的离线演示；真实模型模式需要自行配置服务、tokenizer 和知识索引。

## 项目功能

* **意图识别与协作**：结合最近五轮问答判断需求清晰度和业务相关性，将任务交给投稿规范、审稿流程、出版政策、费用四类专业 Agent，必要时汇总多个角色的结果。
* **Agent Harness**：LangGraph 管理外层流程与内层 ReAct 工具循环，各 Agent 使用私有执行上下文，通过正式结果与证据协作。
* **知识检索**：LlamaIndex 组织清洗与切分，ChromaDB 保存子块向量；向量和 BM25 双路召回，经 RRF、重排、父块回溯形成可追溯证据。Multi Query 改写支持 JEV 语义校验。
* **记忆与上下文**：瞬时、短期、长期记忆分层，短期保留历史摘要和最近五轮问答。压缩后保留原文，通过版本校验提交摘要；Skill 渐进式披露，动态记录与状态后置。
* **执行控制**：工具白名单、Pydantic 参数校验、资源权限隔离、执行预算、超时及无进展检测；长工具结果保存完整原文并分页读取。
* **缓存**：Redis 旁路缓存 RAG 结果，使用 TTL、随机扰动和按键互斥；Redis 不可用时保留本地缓存或原始检索路径。
* **评测与观察**：RAG 检索与生成分开评测；Agent 按任务结果、执行轨迹、运行表现评测。支持规则检查、模型裁判、消融与配对实验、静态 HTML 报告。
* **交互界面**：React、TypeScript、Vite 前端，FastAPI 后端；SSE 展示实际执行阶段、Agent 状态及工具调用状态。当前回答为最终整段返回，不是模型 token 流式输出。

意图识别以 JEV/Decisions 的受检判断为主，Embedding 与 BM25 示例匹配用于对照诊断，不是三路投票决定路由。

## 技术栈

Python 3.12、LangGraph、LlamaIndex、ChromaDB、Redis、SQLite、FastAPI、React、TypeScript、Vite、EvalScope、RAGAS。

## 快速运行离线演示

需要 Python 3.12、Node.js 22.12 或更新版本，以及 pnpm。以下以 Windows PowerShell 为例，在仓库根目录执行。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

Set-Location frontend
pnpm install --frozen-lockfile
pnpm run build
Set-Location ..

.venv/Scripts/python.exe -X utf8 scripts/serve.py --demo
```

浏览器访问 **http://127.0.0.1:8000/**。离线演示使用受控模型和独立数据库，不会调用付费接口，也不能作为真实期刊政策答复。

Linux/macOS 将 `.venv/Scripts/python.exe` 替换成 `.venv/bin/python`。也可以通过 `environment.yml` 创建 Conda 环境。

## 真实模型模式

1. 将 `.env.example` 复制为 `.env`，填写生成、意图判断、向量与重排服务配置。示例中的模型名和上下文窗口需要根据自己账号实际可用服务核实。
2. 准备与所用模型匹配的本地 tokenizer，并设置 `AGENT_TOKENIZER_PATH` 和 `AGENT_CONTEXT_WINDOW_TOKENS`。程序不会自动下载 tokenizer 或模型权重。
3. 自行准备有权使用的知识资料。清洗、切分、父子块与索引脚本位于 `解析试验/`。现有适配器读取 `解析试验/父子检索/结果/20261006-qwen-child-128/` 下的 `parents.jsonl` 和 `children.jsonl`，Chroma 索引位置见 `persist_chroma_parent_child.py` 的 `chroma_dir_for_project`。
4. 启动：`.venv/Scripts/python.exe -X utf8 scripts/serve.py --live`。发送问题会调用配置的外部模型，可能产生费用。

原始第三方文档、处理后的完整语料、向量数据库、用户对话、API key、运行缓存与大批量模型输出不随仓库发布。此仓库下载后即可准备离线演示；真实检索需要先完成上述资料与索引配置。

## 测试与评测

```powershell
.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -q
Set-Location frontend
pnpm test
pnpm run build
```

单元测试使用离线夹具，不需要真实 key；部分可选评测框架相关测试在对应依赖未安装时会跳过。检索评测附加依赖见 `evaluation/requirements-retrieval.txt`，生成评测见 `evaluation/requirements-generation.txt`，建议安装在独立环境。

评测入口与实现：

| 路径 | 内容 |
| --- | --- |
| `evaluation/rag/` | 检索数据契约、召回采集、指标计算、固定上下文生成与 RAGAS 评分 |
| `evaluation/agent/run.py` | Agent smoke、完整任务和重复运行入口，真实调用需要显式 `--live` |
| `evaluation/agent/rules.py` | 路由、工具调用、权限和执行预算规则检查 |
| `evaluation/agent/judge.py` | 业务语义模型裁判 |
| `evaluation/html_report.py` | 读取已有结果并导出静态 HTML 汇总 |

报告导出只读取已有结果；仓库不包含本机全部历史运行数据。已核对的诊断结果和适用边界见 [评测结果说明](docs/evaluation-summary.md)。

## 目录

```text
src/submission_assistant/
  intent/       意图判断与示例检索
  agents/       角色定义
  runtime/      Agent 循环、记忆、上下文、工具、RAG、缓存
  web/          FastAPI、会话及 SSE
frontend/       React 交互界面
resources/skills/  按需加载的 Skill
evaluation/     RAG 和 Agent 评测
解析试验/       清洗、切分、父子块及索引脚本
tests/          离线测试
docs/           设计与运行说明
```

## 使用边界

当前服务仅绑定 `127.0.0.1`，使用本地身份与单 worker，未提供公网登录、多租户认证或生产部署方案。不要仅修改监听地址就公开后端。GitHub 托管的是源码，GitHub Pages 无法运行本项目的 Python Agent 后端。

项目中的第三方资料引用和依赖保留其原有权利；本仓库未附带第三方知识库全文或授权再分发它们。
