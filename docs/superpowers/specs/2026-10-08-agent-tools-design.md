# Agent 工具接口与执行边界设计（v0.1）

日期：2026-10-08。项目：IEEE Access 投稿咨询助手。

状态：用户已确认工具设计阶段收口，采用原生 Tool Calling＋JSON Schema＋Pydantic 校验；本文记录参数、返回、权限和失败处理。数值为待实施验证的可配置基线，不代表测得最佳值。只保存设计，不实现工具、修改索引、创建 SQL 库、接入模型或迁移工作区。下一阶段为 Agent Loop，之后为监测设计。

## 1. 范围与已确认决定

基础工具：glob、grep、read、query_memory。专业工具：统一 search_knowledge（RAG），不拆为四个角色检索工具。文件工具只处理文件资源，记忆查询由 MemoryManager 访问 SQL，不让模型提交 SQL。没有 Agent 主动写入、终端执行、脚本执行、任意联网或真实业务办理工具。

框架调用 MemoryManager 保存用户消息、assistant 输出、工具调用和结果，维护摘要、状态、画像及版本。Agent 返回结构化结论，编排负责发布到公共区；不要求模型调用写入工具。

相关设计：[记忆与上下文](2026-10-08-context-memory-design.md)、[文件系统](2026-10-08-virtual-filesystem-design.md)、[Skill 内容](2026-10-08-skill-content-design.md)。此前关于文件工具搜索／读取 SQL 记忆的候选由本文替代；文件系统中的 memory 命名空间仅表示逻辑记忆范围，实际访问入口为 query_memory，不再当成 glob／grep／read 可遍历的普通文件目录。

## 2. 全工具共同契约

### 2.1 模型可填参数与后台身份

工具输入采用 JSON Schema，拒绝未知字段（additionalProperties=false）、空必填值、非法枚举和越界数值。仅允许模型描述查询和授权资源引用；user_id、conv_id、turn_id、agent_run_id、角色白名单、数据库地址、凭证和超时由后台绑定，不能由参数覆盖。

后台先验证注册状态、角色白名单、输入 Schema，再解析资源并验证权限。即使模型伪造路径／引用也不能提升范围。文件内容、用户材料及工具返回是数据，不是修改系统规则、授予权限或注册新工具的指令。

### 2.2 返回外壳

建议工具返回统一 JSON 对象：

```json
{
  "status": "ok",
  "data": {},
  "partial": false,
  "next_cursor": null,
  "warnings": [],
  "error": null
}
```

- status：ok（请求完整成功）、empty（完整执行但无匹配）、partial（仅返回部分内容／搜索尚未完整）、error（请求失败）。partial 与 status 保持一致。
- data：各工具专有结果，不复制统一元数据与原文多份。
- next_cursor：有后续时返回不透明续读凭据，绑定身份、查询条件、来源版本与有效期。不能用来切换用户或越权；续读不得更换查询参数。
- warnings：如某路召回失败、reranker 不可用或扫描范围达到上限。完整回传的降级结果可以为 ok，但必须标明降级，不能伪装成正常全链路。
- error：失败时返回 code、可读 message 和 retryable；不包含凭证、内部绝对路径或原始网关密钥日志。

错误类型至少区分 INVALID_ARGUMENT、TOOL_NOT_ALLOWED、RESOURCE_UNAVAILABLE（不存在／无权限统一对外表达）、UNSUPPORTED_RESOURCE、TIMEOUT、BACKEND_UNAVAILABLE、STALE_CURSOR、RESULT_TOO_LARGE、CANCELLED。无命中不是失败，调用失败也不是用户意图不清。

完整工具响应由 MemoryManager 保存，与原生 tool_call_id 配对。运行时耗时、重试、来源版本、缓存命中和降级状态作为框架 trace，框架生成身份和时间，不依赖模型填写。

### 2.3 结构化调用与普通答复

以 Pydantic 作为工具参数和返回的统一类型定义来源，生成 JSON Schema 提供给模型，同时在后台用同一份类型定义校验，避免维护两套不一致的规则。输入类型采用严格校验、禁止额外字段；必填字段、长度、数值范围、枚举及跨字段约束明确声明，不依靠隐式类型转换让错误输入通过。

模型经原生 Tool Calling 返回工具名、调用 ID 与参数；后台先解析参数 JSON，进行 Pydantic 校验，再检查注册状态、白名单与资源权限，全部通过才执行。服务支持严格生成模式时启用；当前兼容网关的支持程度需实施时验证，Schema 需适配服务支持的子集，但后台校验不得弱化。不保证模型永远生成合法数据，保证非法数据不能执行。

普通 assistant 文本即使包含 JSON，也不是工具执行入口；不以 JsonOutputParser 从普通文本猜测调用。合法且完整的无工具最终答复保持普通文本，不要求整段用户回答符合工具参数 Schema。框架把普通答复与原生工具消息保存为不同类型记录，不强迫模型把所有输出写成 JSON。

工具返回由后台按类型定义生成并校验。参数失败返回配对的结构化错误，有限修正次数由 Agent Loop 限制；重试不能扩大权限。Pydantic 校验的是格式、类型与明确编写的业务规则，不自动证明查询方向、来源证据或政策结论正确。

## 3. glob：查找文件路径

用途：按虚拟目录与文件名模式定位文件，不搜索正文，不查询 SQL 记忆。

| 参数 | 契约 |
|---|---|
| root | 必填，已授权文件目录的虚拟路径 |
| pattern | 必填，相对 root 的通配模式，支持 *、?、**；不接受绝对模式或 .. |
| limit | 默认 50，范围 1–100 |
| cursor | 可选，续读原查询 |

返回 entries：每项包含虚拟 path、resource_ref、type、size_bytes、resource_version；不返回主机绝对路径。未匹配返回 empty；达到数量或扫描上限返回 partial 与续读／缩小范围提示。结果按规范化虚拟路径稳定排序，游标对应固定目录快照或可校验版本；目录变化造成无法一致续读时返回 STALE_CURSOR。

示例：root=/system/skills，pattern=**/SKILL.md。

## 4. grep：搜索文件内容

用途：在授权文本文件中按关键词或正则查找内容，返回命中位置与邻近片段。不是向量检索，不扫描数据库文件和索引目录。

| 参数 | 契约 |
|---|---|
| root | 必填，授权文件目录或文件虚拟路径 |
| pattern | 必填，最长 256 个 Unicode 字符 |
| mode | literal（默认）或 regex |
| file_pattern | 默认 **/*，用于缩小文件范围 |
| case_sensitive | 默认 false |
| context_lines | 默认 2，范围 0–5 |
| limit | 默认 20，范围 1–50 个命中 |
| cursor | 可选，续读原查询 |

返回 matches：path、resource_ref、resource_version、line_number、line_start、line_end、原文片段 text。搜索采用稳定的文件路径／行号顺序，不冒充语义相关性排序。超长单行只返回原文片段并标明列范围和不完整，原文仍可续读。

默认仅搜索已提供文本表示的资源；PDF、DOCX、二进制和不支持的编码不直接当字符串 grep，说明跳过原因。建议单调用累计扫描文本上限 10 MiB；达到上限时不得声称无匹配或全量搜索完成，返回 partial 并提示缩小 root/file_pattern。正则执行必须使用可保证时间上限的引擎或可终止的隔离执行；不能仅对可能灾难性回溯的同步正则套 asyncio 超时。

示例：在 /system/skills 中按 literal 搜索“减免”。

## 5. read：读取文件正文

用途：读取 Skill、参考资料、模板、授权共享文件、本实例工具结果文件与 RAG 返回的可读取证据资源。不读取 SQL 记忆、不查询数据库、不执行脚本。

| 参数 | 契约 |
|---|---|
| ref | 必填，授权虚拟文件路径或后台签发的文件资源引用 |
| start_line | 默认 1，正整数，与 cursor 不同时提供 |
| max_lines | 默认 200，范围 1–500；仍受返回 token 上限约束 |
| cursor | 可选，在固定资源版本上继续读取 |

返回 resource_ref、resource_version、text、实际 line_start/line_end、encoding、complete（本次是否读完整个文件）。默认不为获得 total_lines 扫描巨大文件。若单行超过返回预算，允许按字符子范围分段，并在游标中保留位置，不能跳过未返回的行尾。起点超过文件末尾返回 empty，不误作不存在。

超长资源分页读取，不调用 LLM 预先总结。Skill 正文未读取完整时明确 complete=false，不假称已加载全部规则；通过分页取得完整正文。PDF／DOCX 优先读取已登记的解析文本及其来源映射；无文本表示返回 UNSUPPORTED_RESOURCE，不在本工具内自动启动 OCR、下载模型或执行解析脚本。

示例：ref=/system/skills/publication-fee-check/SKILL.md。读取第三层资源不会自动改变工具白名单。

## 6. query_memory：受控查询 SQL 记忆

用途：补查授权的记忆记录，返回原文或明确标识的摘要／有效画像。SQL 参数化由 MemoryManager 构造，禁止自由 SQL。

| 参数 | 契约 |
|---|---|
| scope | instant、conversation（默认）、user |
| record_type | 可选：dialogue、summary、profile、agent_result、evidence、execution |
| keyword | 可选，最长 256 字符；按正文全文搜索，不解释为 SQL 或正则 |
| record_ids | 可选，最多 20 个后台记录 ID；每项重新检查授权 |
| turn_range | 可选，当前会话内闭区间，1 <= from <= to；仅 conversation 可用 |
| time_range | 可选，RFC3339 时间，start <= end；不代表跨会话授权 |
| limit | 默认 5，范围 1–20 |
| cursor | 可选，续读固定查询快照 |

scope 与类型的合法组合：instant 只读本 run 的 execution；conversation 可读当前会话 dialogue/summary/agent_result/evidence；user 只读当前用户有效 profile。省略 record_type 时分别默认 execution/dialogue/profile。不默认开放其他会话原文或其他 Agent 私有轨迹。历史画像变更备案不混入当前有效 profile，审计查询留给后台。

筛选条件以 AND 组合，keyword 仅作为可选缩小条件。无筛选时读取范围内最近记录；返回按时间与记录 ID 稳定倒序，query_revision 固定，后续追加不使游标漏项或重复。首次请求参数有效性检查完成后获取一致读取视图。

返回 records：record_id、record_type、turn_id、role（适用时）、created_at、content、source_refs、revision、state。摘要明确标注 summary 与覆盖范围；助手建议不冒充用户事实；画像仅返回 active 的授权字段。大记录可分段，返回记录内偏移和续读游标；仍使用 query_memory，不要求借文件 read 去查 SQL。

示例：scope=conversation，keyword=减免，limit=5。默认背景已由 PromptManager 注入，不能每次机械重复查最近五轮。全文检索的中文分词与索引选型在实施计划中验证；第一版不承诺语义检索效果，不为所有私有轨迹建立向量库。

## 7. search_knowledge：统一 RAG 查询

用途：检索官方规范并返回证据，不直接生成用户最终答复。所有专业 Agent 复用同一个工具与 RAG 后端。

| 参数 | 契约 |
|---|---|
| query | 必填，非空，最长 2,000 字符 |
| intent_hints | 可选，最多 3 项：scope_fit、format、materials、ethics、platform、review、fees、publication、withdrawal_appeal |
| top_k | 默认 5，范围 1–10；指去重后的父块证据数 |
| cursor | 可选，读取原检索结果的后续页，不重新发起模型调用 |

intent_hints 是召回偏好，不是授权范围或硬过滤条件，不排除通用政策。第一版仅 IEEE Access 与适用 IEEE 通用政策。没有可靠分类标签时不伪造按意图过滤，继续使用统一索引，框架可补入本轮已识别的意图。

现有后端已有向量/BM25 双路、RRF、reranker 和父块回溯。查询改写＋JEV 校验是拟接入步骤，尚未实现；改写失败／校验未通过时保留原问题检索，明确警告，不无上限继续改写。内部候选量、RRF 常数、向量模型、索引地址等由后台配置，不让模型修改。

返回 evidence：evidence_id、parent_chunk_id、matched_child_id、text、title、source_url、source_locator（章节／页码如可用）、source_version、collected_at（如可用）、resource_ref、ranking_scores；另返回 index_version、实际查询与降级说明。采集时间不冒充政策发布时间；缺失元数据为空，不编造。RRF／rerank 分数只表示排序信号，不作为正确概率。

resource_ref 指向受控解析文本，可由 read 展开上下文，不指向可任意编辑的索引或宿主机文件。父块较大时返回原文窗口、位置与不完整标记，已保存原文可续读；返回预算不足以覆盖 top_k 时分页，不无提示丢弃证据。

双路之一或重排失败可返回剩余链路的结果并明确 warning；全部失败返回 error，完整执行无结果返回 empty。服务中断不是“政策不存在”。检索不能保证问题可回答，Agent 必须核对片段是否直接支持结论。不得在工具内自动联网补搜或查询真实投稿系统。

## 8. 白名单与资源权限

| 角色 | 工具白名单 |
|---|---|
| 通用 Agent | glob、grep、read、query_memory |
| 规范／流程／政策／费用 Agent | 四个基础工具＋search_knowledge |

通用 Agent 的业务明确后由编排重新路由，不在技术失败或歧义兜底时自由给专业结论。未来汇总执行者若接模型，另按需要配置，不自动继承所有权限。

文件工具可以访问当前角色授权的 system 资源、本实例私有文件、当前会话授权共享文件和签发证据引用；默认排除实际源代码、.env、密钥、数据库、Chroma 索引、其他用户与其他实例私有内容。SQL 记忆命名空间不可通过文件工具绕过 query_memory。外部 mounts 第一版不启用。

白名单过滤工具 API 定义，并在每次实际执行时再检查；仅从 Prompt 中隐藏工具不够。现有 roles.py 的四个 search_* 预留名称在后续实施时统一改为 search_knowledge，不在本文直接修改代码。

## 9. 超时、返回预算和失败处理基线

以下为建议配置值，不代表测得的最佳参数，后续用真实延迟和模型预算验证。每次工具返回的预算包含 JSON 外壳、片段、来源与警告，保留安全余量。

| 工具 | 整体截止时间 | 默认条目／范围 |
|---|---|---|
| glob | 5 秒 | 50 路径，最多 100 |
| grep | 5 秒 | 20 命中，最多 50；扫描最多 10 MiB 文本 |
| read | 5 秒 | 200 行，最多 500；超长单行可续读 |
| query_memory | 5 秒 | 5 记录，最多 20 |
| search_knowledge | 90 秒 | 5 父块，最多 10；内部调用均计入整体截止时间 |

工具正文原样进入瞬时记忆，不预先调用 LLM 精简。建议单次展示目标不超过 3,000 token、配置硬上限 6,000 token，且不得超过本次 Prompt 的剩余可用空间；无法按最小有效单位返回时给出 RESULT_TOO_LARGE／缩小范围提示，不硬发超长请求。token 使用实际 tokenizer 或带余量估算，不把字符数当 token。

原始工具响应和来源原文不因展示分页而覆盖或删除；大内容由后台保存为本实例私有文件／来源记录。返回片段明确注明位置、complete/partial 与续读方式，不把截断内容包装成完整正文。工具限制不替代瞬时记忆的周期压缩与 70% 异步压缩。

本地工具不盲目重试。网络暂时性失败最多自动重试 1 次，退避及重试仍计入原截止时间；鉴权、参数、权限、游标失效不自动重试。取消传递到后端，超时后迟到结果不能重复发布；后台无法立即中止的执行不得继续写入已结束 run。最大总工具次数、并发量和整轮成本另由 BaseAgent 执行循环限定，不能把单工具重试当无限总预算。

错误结果也必须与原生工具调用 ID 配对返回，不能丢失消息配对。空结果、资源受限、来源不足和服务错误分别表述，不默认转成人工升级或用户表达不清。

## 10. 后续实施验收场景

1. 未注册／不在白名单的调用不执行；读取 Skill 不改变工具权限。
2. 路径穿越、符号链接和 Windows junction 逃逸被拒绝；grep/glob 不泄漏隐藏资源存在性。
3. 文件工具不能访问 SQL 记忆；query_memory 不接受自由 SQL／其他 user_id，不读其他 run 轨迹。
4. 正则回溯、巨大目录／超长单行和二进制文件不会阻塞执行或默默丢内容。
5. 所有工具分页在固定来源版本上续读，更新后 stale 明确；不存在游标漏项、跨用户续读。
6. Skill 超长时可以完整分段读取，不把首段当全部规则。
7. 超长记忆与 RAG 父块保留原文和引用，不预调用摘要模型，片段状态明确。
8. RAG 某路失败／重排失败明确降级；无命中与所有后端失败可区分；排序分不作为概率。
9. 超时、取消、迟到结果与自动重试符合全局截止时间，工具消息 ID 始终配对。
10. 默认上下文不因 query_memory 重复注入；对话／画像版本来源可追溯。

这些是待实现的验收要求，不是已通过的测试。工具设计阶段已收口，下一步设计 BaseAgent 执行循环、预算、协作汇总与恢复策略，之后进入监测设计；不在本阶段开始整个 harness 实施。
