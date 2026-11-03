# 本地聊天工作台

## 已接入内容

React＋TypeScript＋Vite 前端，FastAPI 后端，复用现有 ChatService、LangGraph、记忆、RAG 与可选 Redis。左侧新建/切换会话，中间公开问答，右侧实际路由、公开运行状态与来源证据；窄屏侧栏可展开。

正式 CLI 和网页 live 模式选用 `routing-g6-clarity-v3`：G6 业务判断＋成功的新清晰度定义，最近五轮完整问答，阈值0.75/0.8。未采用后来退步的背景补充示例。旧评测构造器默认值与冻结结果不变。

## 直接启动

以下命令在项目根目录运行。已有 `.venv-evaluation` 环境为本次验证环境。

离线演示（不产生模型调用，单独保存到 `data/web-demo/memory.sqlite3`）：

```powershell
.venv-evaluation/Scripts/python.exe -X utf8 scripts/serve.py --demo
```

真实运行（读取已有 `.env`，发送问题会调用已配置模型并产生费用）：

```powershell
.venv-evaluation/Scripts/python.exe -X utf8 scripts/serve.py --live
```

访问 http://127.0.0.1:8000/ 。必须使用这个地址而非 localhost，后端按回环 Host 白名单保护。两种模式不要同时占用同一端口；按 Ctrl+C 停止前一个服务，也可用 `--port 8001` 显式换端口。只运行一个 worker，不使用多进程部署。

真实会话使用现有 `data/memory/memory.sqlite3`，可信本地身份 `local-user`；网页无身份选择项。旧公开会话可以恢复，不读取其他身份私有记录。不因 live 配置错误偷偷切为演示。

后端增量依赖已写入 environment.yml。本次安装版本 FastAPI0.159.1、Uvicorn0.54.0；如重建环境：

```powershell
.venv-evaluation/Scripts/python.exe -m pip install "fastapi>=0.115,<1" "uvicorn>=0.30,<1"
```

## 前端开发与重新构建

Node24可用。此机器 npm 不在 PATH，使用已有 bundled pnpm；也可以在正常配置的机器使用自己的 pnpm。只批准 esbuild 安装构建脚本，锁文件为 `frontend/pnpm-lock.yaml`。

```powershell
Set-Location frontend
$workbenchPnpm = 'C:/Users/amorous/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pnpm/bin/pnpm.cjs'
node $workbenchPnpm install --frozen-lockfile
node $workbenchPnpm test
node $workbenchPnpm run build
```

build包含 TypeScript 类型检查。构建产物 `frontend/dist/` 由 FastAPI 同源托管；仅暴露入口和 assets，不暴露项目目录。如果首次构建前已启动后端，请构建后重启后端，以挂载新增 assets 目录。

开发前端可运行 `node $workbenchPnpm dev`，访问 http://127.0.0.1:5173/ ，代理 `/api` 至8000；此时后端也需启动。更换后端端口需要同步 Vite 代理地址。

## 运行行为

- 明显显示“离线演示”或“真实模型”。演示返回不代表真实期刊政策，没有证据时明确显示“暂无证据”。
- 第一版是实际执行阶段＋最终整段回答，**不是模型 token 流式输出**，也不模拟逐字输出。
- 每条消息有客户端轮次ID；重复提交同ID/同消息恢复原任务，不再次执行。不同消息复用ID报冲突。任务最多两个不同会话并发，同一会话一次一轮；超出容量明确报忙，不无限排队。
- 刷新、切换、SSE重连只是恢复状态。浏览器断线不取消后台任务；服务重启后未完成的网页任务显示中断，不自动重放工具。遇到“发送结果尚未确认”可点击“恢复状态”，先查已有任务，确认不存在才用原ID重发。
- 页面将请求绑定到启动时确认的 demo/live 模式；后端切换模式后，旧页面拒绝继续提交，必须刷新确认，避免旧“离线演示”页误调用真实模型。慢请求返回不会清除期间新编辑的草稿，旧会话加载也不能覆盖更新的页面状态。
- 对话和最终回答以SQLite原文为准；新增 `web_conversations`/`web_turns` 仅存网页元数据。前端仅保存所选会话及待确认轮次，不保存模型密钥。
- 支持 Markdown、代码复制，禁用原始HTML、危险链接和远程图片。证据仅显示安全HTTP/HTTPS来源，本地文献不提供磁盘下载路径。
- 公开状态只含允许字段，不包含系统提示、工具参数、内部思维链或私有轨迹。密钥留在后端。启动仅验证本地配置/索引；health中 configured_not_probed 不代表远程服务已成功访问。
- 正常关闭会取消并收尾受管任务；任务创建后尚未开始也会标记中断。若工具在5秒期限内仍不响应取消，则报告关闭异常并保留仍在使用的存储，避免在任务运行时强关SQLite。无网页任务记录的旧CLI运行中轮次显示为不可恢复的中断，但不改写旧原文。

## 验证与限制

```powershell
.venv-evaluation/Scripts/python.exe -X utf8 -m unittest discover -s tests -q
```

新增测试覆盖冻结提示、运行工厂、会话归属、幂等、并发、SSE、私有文件保护、Origin/Host检查、Windows JS MIME兼容，以及前端断线恢复/中文输入法/安全Markdown。

开发与验收默认全离线，未自动进行真实模型端到端收费验证；不宣称新的线上识别准确率。此版本不含登录、公网部署、文件上传、知识库管理或批量删除。需要远程/多用户使用时必须另行增加认证和部署设计，不能简单改监听地址。

本次最终结果：291项后端测试、15项前端测试通过，TypeScript检查与生产构建通过；桌面/390px窄屏、新建/切换/发送/刷新、实际路由和状态、部分完成/证据/失败场景已做离线浏览器验收。独立审查反馈的问题已逐项复现并修复，但审查者受额度限制未给出最终完整结论，剩余检查为作者自查。

2026-10-22真实运行修复：路由及查询改写校验使用独立HTTP连接，遵循 `JEV_TRUST_ENV`；千问等原有直连策略不变。提交节点改为异步，在运行中的事件循环启动记忆维护，避免答复已保存后误报失败。新增4项离线回归，后端共295项测试通过；没有为此重跑收费测评，也没有自动重发旧问题。旧失败反馈仍保留，请刷新后新发问题。

2026-10-22执行进度展示：回答区域和右侧运行观察复用实际执行状态，展示意图识别、各 Agent 准备上下文/思考与生成、RAG 检索、具体工具调用、结果汇总及结束状态。并行工具分别记录等待、调用中、完成、失败、超时、取消，仅公开已知工具名称和状态，不公开参数、返回正文或内部思维链。沿用现有 SSE 订阅，运行期间约每秒读取实际状态；极快的中间步骤可能合并，不伪造进度百分比。旧会话缺少细化状态时兼容显示，最终轮次状态优先，避免失败后仍显示调用中。

进度卡片展开时仅对停留底部的阅读者跟随；上翻历史不强制滚回，只有计数刷新的心跳不滚动。验证：298项后端测试、25项前端测试及生产构建通过；独立8002离线验收覆盖识别、并行角色思考/RAG/工具、汇总、完成与390px窄屏，无新增真实模型请求。手工验收入口为 `tests/web_progress_preview.py`，数据独立位于 `.cache/progress-browser-qa`，不用于真实服务。
