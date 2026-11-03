# 父子块检索：GitHub 项目经验

研究日期：2026-10-06。用于 IEEE Access 投稿咨询助手的检索设计讨论。

本次核对了 Dify、QAnything、RAGFlow 的具体代码或修复记录，以及 LlamaIndex 的维护者答复。链接中的固定提交和历史分支代表被查看的版本；历史问题不能直接当作当前版本仍存在的问题。以下设计建议尚未在本项目上做检索评测。

## 1. 从应用源码看到的共同流程

先确定父块，再在每个父块内部切子块。子块参与检索，通过 parent_id 回到父块。多个命中子块可能指向同一个父块，因此需要按父 ID 去重，并明确定义分数聚合。送入 LLM 前再控制父块数量和总 tokens。

```text
来源文档
  └─ 父块：一个小节或一组相邻段落
       ├─ 子块 A：用于检索
       ├─ 子块 B：用于检索
       └─ 子块 C：用于检索

问题 → 检索子块 → 按 parent_id 分组 → 取回父块 → 控制上下文 → 回答
```

父块用于提供上下文。它可以是段落、小节，也可以是足够短的完整文档。检索块大小和最终上下文大小是两项独立设置。

## 2. Dify：父块边界和子块边界分别配置

核对来源：

- [ParentChildIndexProcessor 源码，59dc7c88](https://github.com/langgenius/dify/blob/59dc7c88/api/core/rag/index_processor/processor/parent_child_index_processor.py)
- [父子分块说明，a958852d](https://github.com/langgenius/dify-docs/blob/a958852d/en/use-dify/knowledge/create-knowledge/chunking-and-cleaning-text.mdx)

`transform()` 支持 `ParentMode.PARAGRAPH` 和 `ParentMode.FULL_DOC`。段落模式先按父级分隔符、长度和 overlap 切父块，再调用 `_split_child_nodes()` 使用另一组设置切子块；全文模式把本次文档内容组成一个父块，再切子块。`load()` 把 children 写入向量库。

对应文档明确说明：命中子块后，返回整个父块。段落模式适用于长文档中可独立理解的段落或章节；全文模式适用于短且主题一致的文档。

可以借鉴：HTML 已有标题路径，可以以小节建立父块，单独设置子块大小。短 FAQ 可保持问题与答案的完整单元。全文父块的大小必须有约束，不能把长手册全文作为每次命中的上下文。

## 3. QAnything：父块在 MySQL，子块在向量库

核对来源：

- [parent_retriever.py，qanything-v2 分支](https://github.com/netease-youdao/QAnything/blob/qanything-v2/qanything_kernel/core/retriever/parent_retriever.py)
- [model_config.py，qanything-v2 分支](https://github.com/netease-youdao/QAnything/blob/qanything-v2/qanything_kernel/configs/model_config.py)

这个分支基于 LangChain 的 `ParentDocumentRetriever` 自定义了 `SelfParentRetriever`，可以借鉴其数据组织，不需要为此更换我们使用的 LlamaIndex。

代码中的关键步骤：

1. 用递归分隔符切父块，子块继续在每个父块内部切分。
2. 配置的默认父块长度为 800、子块为 400，长度函数是 `num_tokens_embed`，不能把这两个数当作字符数。
3. 父块 overlap 为 0；默认子块 overlap 为子块大小的四分之一。
4. 子块 metadata 保存父 ID，子块写入 Milvus；父块通过 `MysqlStore` 保存。
5. 检索时先取子块，按父 ID 去重，再用 `docstore.amget(ids)` 批量取回父块。
6. `has_table` 的输入在父级切分阶段有特殊处理，保留为一个父级单元；子级仍会继续切分。

可以借鉴：父子关系和父级正文需要独立保存；表格等有内部结构的内容需要专门的边界规则。overlap 可以只设置在子级，避免把大量重复内容带入生成上下文。

### 一个不能照搬的分数细节

这个分支的 `_aget_relevant_documents()` 先对父 ID 去重，再用 `scores[i]` 给取回的父块赋分。按代码可以推导出分数错位风险：

```text
命中的子块：A1→父A，0.95；A2→父A，0.90；B1→父B，0.80
去重后的父块：[父A，父B]
按 scores[i] 赋分：[0.95，0.90]
父B 应关联的命中分数是 0.80，却拿到了父A另一个子块的 0.90。
```

这是对该分支代码的推导，不是引用维护者确认的缺陷报告。我们的实现应按 `parent_id` 聚合分数，可从最高子块分数开始，再通过评测比较平均分或父级 rerank。重排整个父块时，应核对 reranker 输入上限。

## 4. RAGFlow：父级回填必须进入所有检索入口

核对来源：

- [retrieval_by_children 源码，74b44e1a](https://github.com/infiniflow/ragflow/blob/74b44e1a/rag/nlp/search.py)
- [SDK 检索入口，74b44e1a](https://github.com/infiniflow/ragflow/blob/74b44e1a/api/apps/sdk/doc.py)
- [Issue #12783：网页端返回父块，SDK 返回子块](https://github.com/infiniflow/ragflow/issues/12783)
- [PR #12789：修复缺失父块，已合并](https://github.com/infiniflow/ragflow/pull/12789)

`retrieval_by_children()` 使用 `mom_id` 对命中子块分组，从存储中取回父正文和位置；父级 similarity 使用该组命中子块分数的平均值，再重新排序。这个分数是子块匹配分数的聚合，不代表父块经过了一次独立 rerank。

Issue 作者报告：在 v0.23.1 中，网页端能返回父块，SDK 只能返回子块。对应 PR #12789 已合并；作者后来在 v0.25.1 测试并确认解决。这里采用作者的复现、关联 PR 和最终验证作为证据，不以 Issue 中的机器人答复作为唯一依据。

可以借鉴：仅建立 parent_id 不足以完成父子检索。聊天接口、检索调试接口和 SDK 必须经过相同的父级回填逻辑，并核验真正交给 LLM 的内容。调试数据应同时保留命中子块和最终父块，便于定位相关性来源。

## 5. LlamaIndex：向量持久化与父节点持久化分别检查

核对来源：

- [Issue #14239：AutoMergingRetriever 在 Chroma 重载后找不到父节点](https://github.com/run-llama/llama_index/issues/14239)
- [Issue #14574：Chroma 持久化后 docstore 为空，含维护者答复](https://github.com/run-llama/llama_index/issues/14574)
- [AutoMergingRetriever 源码](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py)

#14239 是旧版用户报告：初始化过程可以查询，另一个进程从 Chroma 重新加载后，自动合并找不到父节点。不能据此宣称当前 LlamaIndex 或 Chroma 普遍不支持父子检索。

#14574 中维护者说明：使用许多第三方向量库时，默认不会再向 docstore 重复保存节点。需要手动填充、持久化 docstore，或使用适当的配置保存节点。

可以借鉴：子块向量、父块正文、节点关系都需要持久化。应在新进程重新加载后，验证每个子块的父 ID 能解析到真实父正文，不能只检查“能搜到向量”。

AutoMergingRetriever 根据同父节点下命中的子块比例决定是否替换成父级；显式父子回填则在命中子块后直接沿引用取回父级。两种策略可以分别评测。

## 6. 评测仓库给出的边界：功能列表不等于性能证明

核对来源：[satyaborg/rag](https://github.com/satyaborg/rag)。

该仓库基于 LlamaIndex，支持 base、sentence-window 和 child-to-parent 等分块方式，也支持 BM25、向量和 hybrid。可借鉴其把不同方案做成可切换配置的思路。

但是，本次读取的 README 公开结果表中列出的 chunking strategy 都是 `base`；不能从中得出父子检索优于普通分块的结论。仓库还说明回答评测因费用只取前 10 条，并使用模型生成的问答集。这些结果可作为实验方法参考，不能直接套用为我们项目的效果。

## 7. 对 IEEE Access 投稿咨询助手的建议

先试两层父子关系：

- HTML：以已有标题路径下的小节建立父块，小节内部建立短子块；过长小节继续限制父块大小。
- PDF / Word：现有语义块可以作为候选父块；检查其边界和长度后，在父块内部用 SentenceSplitter 切短子块。这种组合保留语义父级，也能缩小向量检索单元。
- 初始对比子块 128 / 256 tokens；父级优先以完整规则段落或小节为边界。此处是待评测参数，不是行业标准。
- 只为子块建立检索索引，父块正文与父子关系单独保存。子块保留来源、页码或标题路径、父 ID 和序号。
- 检索后按父 ID 去重，明确定义父级分数；保留命中子块供解释与引用。引用定位应基于实际来源跨度，不能让每个子块无条件继承父块的全部页码。
- 控制父块数量和最终上下文 tokens。短 FAQ / 单条规则本身完整时，允许保持独立单元。
- 与当前平铺切块做对照：看相关规则能否命中、限制与例外是否齐全、无答案问题是否误召回、送给 LLM 的上下文长度及耗时。

当前这一步只收集设计经验，尚未修改切块输出或建立检索索引。
