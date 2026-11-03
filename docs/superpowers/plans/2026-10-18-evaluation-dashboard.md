# Unified Evaluation Dashboard Implementation Plan

> **已按用户要求缩减：下方原看板实施任务取消，不再执行。** 仅在评测结果落盘后复用已有汇总数据，导出 `evaluation/reports/agent-rag-overview.html`。页面按 RAG 检索、RAG 生成、Agent 三层分区放静态指标表，标明有效样本数／预期数、缺失原因和限制，保留来源链接，做简单清晰的排版即可。不做独立 dashboard 子系统、筛选／搜索／逐题交互或复杂图表，不新增依赖。验收只核对指标及分母与来源一致、缺失不补零、HTML文本安全转义和文件可打开；导出过程不调用模型、不重跑RAG。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已有 RAG 和新 Agent 的汇总、逐题结果及限制汇成一个漂亮的离线 HTML。

**Architecture:** 只读加载校验过的历史工件，转换成小型统一展示契约，再嵌入自包含 HTML。不启动业务或裁判；无 Agent 数据也能展示 RAG 并明确待运行。

**Tech Stack:** Python 标准库、unittest、内联 HTML/CSS/JavaScript/SVG，复用已有 RAG 校验器与新 Agent 结果校验器。

**Spec:** `docs/superpowers/specs/2026-10-18-evaluation-dashboard-design.md`

## Global Constraints

- 输出 `evaluation/reports/agent-rag-overview.html`，可离线单文件打开，无CDN、服务或网络fetch；原始数据只读。
- 覆盖检索／生成／Agent端到端、轨迹、表现；历史烟测和正式批次不能混算。
- 缺失评分不补0，均值展示有效数／预期数，Agent业务与故障分组。
- 不泄漏凭据／环境变量／生产记忆；所有待评文本作为不可信数据展示。
- 草稿标签、同模型裁判、no-JEV、新裁判未校准及历史降级实验限制保持可见。
- 与Agent计划独立交付，后续仅重建看板即可加入新运行，不重跑评测。非Git目录用文件清单及测试记录作检查点。

## Review Focus

1. 整项缺失与真实0分：保留null，显示—及原因（Task1）。
2. 同一题多个实验／语言重名：用kind/run/config/sample复合ID，不覆盖（Task1）。
3. HTML标签、`</script>`与恶意链接：只作文本，链接限本地产物（Task2）。
4. 筛选后分母和图表不一致：全部从同一selected集合计算（Task2）。
5. 损坏或缺失来源：显示告警与未覆盖数，不悄悄跳过（Task1）。

### Task 1: 只读数据目录与统一展示契约

**Files:** Create `evaluation/dashboard/__init__.py`, `load.py`, `sources.json`, `tests/test_evaluation_dashboard_data.py`。

**Interfaces:** `load_dashboard(sources_path: Path) -> dict` 返回 `{schema_version,generated_at,sources,warnings,runs,samples}`；run含kind/id/status/conditions/limitations/metrics；sample含复合id、run_id、config_id、language、category、status、question、answer、evidence、reference、events、metrics、errors。metric为 `{value,valid,expected,unavailable,unit}`；未执行不伪装不可用评分。

`sources.json` 显式配置 retrieval/generation/agent路径与primary标志；主来源为两个检索run `20261016-pilot-01`、`20261016-multiquery-no-jev-01` 和生成run `20261016-full-01`，历史smoke单列；实现时枚举现有全部evaluation运行目录，清单逐项登记。Agent完整结果若未生成则pending，不能假造分数。路径由清单相对项目根解析，不从HTML输入任意读文件。

- [ ] 写 `DashboardDataTests`：0/null区分；同题多配置不覆盖；RAGAS每指标有效数不同仍正确；Agent业务1pass/1fail/1unknown分母3与2并存；缺Agent不崩；损坏来源显示warning。不同run不混算，同模型／草稿限制保留。
- [ ] 运行 `-m unittest discover -s tests -p test_evaluation_dashboard_data.py -v`，确认RED。
- [ ] 实现读取，复用 `evaluation.rag.score.load_run`、`evaluation.rag.generate.load_generation_run`、`evaluation.rag.ragas_score.load_scores` 和Agent `load_agent_run`；Agent导入延迟，以支持先做RAG。缺失可选source标pending，校验异常标invalid及错误，不标正常完成。摘要从逐题重算并核对原summary，记录来源hash和不一致。
- [ ] 同命令GREEN；列出清单中每个真实运行的覆盖数，确认没有旧降级B4覆盖真实no-JEV实验。

### Task 2: 自包含 HTML 与交互

**Files:** Create `evaluation/dashboard/render.py`, `dashboard.html`, `build.py`, `tests/test_evaluation_dashboard_html.py`。

**Interfaces:** `render_dashboard(data: dict) -> str`；`build_report(sources_path: Path, output: Path) -> Path`；CLI `python -m evaluation.dashboard.build --sources evaluation/dashboard/sources.json --output evaluation/reports/agent-rag-overview.html`。输出旁保存脱敏 `agent-rag-overview.data.json` 供校验；HTML本身嵌入全部所需数据，不依赖旁文件。

- [ ] 写 `DashboardHtmlTests`：输入 `</script><script>alert(1)</script>`、中文、引号、Unicode行分隔符不形成执行标签；JS不对输入使用innerHTML，禁止外部script/style/fetch；JSON往返一致，原生报告链接只能指向已验证本地产物。白名单输出不含测试凭据字段。
- [ ] 测试RED；实现转义与白名单，内嵌JSON、CSS、JS和SVG。深侧栏浅内容、蓝绿强调色、中文标题与指标解释；概览卡、三主区、分组图表、逐题详情。使用textContent渲染数据、语义button与label，不使用外部字体／图库。
- [ ] 提供运行、基线、语言、类型、状态筛选及搜索、重置；所有卡片／表格／图表从同一过滤子集重算；零结果显示空态。可展开问题、答案、参考、证据、事件与失败原因，默认收起长文本；不把质量均值命名为准确率。
- [ ] 同命令GREEN，运行build CLI生成真实HTML；打开前核对生成数据与来源的分数／计数逐项一致，容差0.00011，null不参加质量均值。

### Task 3: 浏览器验收与最终交付

**Files:** Create `evaluation/reports/README.md`；Generate `evaluation/reports/agent-rag-overview.html`, `.data.json`；验收记录放 `evaluation/reports/verification.md`。

- [ ] 使用可用浏览器工具打开真实产物；按computer-use技能读取必要指令。实际检查约1440px桌面和390px窄屏；不能只检查模板截图。
- [ ] 实测中文搜索、no-JEV筛选、不可用评分筛选、逐题展开、多轮Agent轨迹、重置与EvalScope链接；同一筛选前后核对卡片／图表／表格数量与已知case值。查看控制台或工具可得错误，不宣称未经检验的零错误。
- [ ] 离线条件下检查无外网依赖、HTML单文件可用；记录字体、遮挡、表格滚动及键盘展开检查，修复后重建重验。浏览器或改viewport能力受限则报告具体未验收项，不编造视觉通过。
- [ ] 两组新测试及全套unit测试GREEN后，给用户最终HTML链接、三分区说明、已执行数／有效评分数与限制；只有Agent完成核验才说所有评测完成。附重建命令，不附密钥。

## 自查与交接

设计范围／数据正确性→Task1，布局交互安全→Task2，真实离线视觉验收→Task3；Review Focus全部对应测试或浏览器检查。该计划不调用实时模型，输入接口固定，可在Agent结果落盘后单独重建。

关键边界断言（Task1/2，供fixture实现时固定语义）：

```python
# test_zero_and_unavailable_have_different_denominators
self.assertEqual(metric['value'], 0.0)  # fixture仅一个真实0和一个null
self.assertEqual((metric['valid'], metric['expected'], metric['unavailable']), (1, 2, 1))
# test_script_payload_remains_data
self.assertNotIn('</script><script>alert(1)</script>', html)
self.assertNotIn('unit-test-secret', html)
```
