"""Prototype small-child retrieval with parent-context return over prior chunks."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_SCRIPT_DIR = PROJECT_ROOT / "解析试验" / "语义分割" / "scripts"
sys.path.insert(0, str(SPLIT_SCRIPT_DIR))

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode, TextNode

from split_documents import _load_embedding_model, load_env_file


SOURCE_DIR = (
    PROJECT_ROOT
    / "解析试验"
    / "语义分割"
    / "结果"
    / "20261006-qwen-text-embedding-flash"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "解析试验"
    / "父子检索"
    / "结果"
    / "20261006-qwen-child-128"
)
CHILD_SIZE = 128
CHILD_OVERLAP = 24
RETRIEVAL_TOP_K = 10

TRIAL_QUERIES = [
    {
        "question": "IEEE Access 投稿时，Word 或 LaTeX 源文件和 PDF 文件有什么要求？",
        "expected_title_fragments": ["Submission Guidelines for Authors"],
    },
    {
        "question": "For an IEEE Access submission, must the Word or LaTeX source match the PDF exactly?",
        "expected_title_fragments": ["Submission Guidelines for Authors"],
    },
    {
        "question": "IEEE Access 论文摘要需要多少词？",
        "expected_title_fragments": ["Microsoft Word", "Preparing Your Article"],
    },
    {
        "question": "投稿后还能修改作者名单吗，谁来决定？",
        "expected_title_fragments": ["Submission Guidelines for Authors"],
    },
    {
        "question": "How does IEEE handle corrections and retractions after publication?",
        "expected_title_fragments": ["IEEE 出版服务操作手册"],
    },
    {
        "question": "IEEE Access 论文图片的格式和分辨率要求是什么？",
        "expected_title_fragments": ["Create Graphics for Your Article"],
    },
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _token_count(splitter: SentenceSplitter, text: str) -> int:
    return len(splitter._tokenizer(text))


def _build_parent_and_child_nodes(
    parent_records: list[dict[str, Any]], splitter: SentenceSplitter
) -> tuple[list[TextNode], list[IndexNode], list[dict[str, Any]]]:
    parent_nodes: list[TextNode] = []
    child_index_nodes: list[IndexNode] = []
    child_records: list[dict[str, Any]] = []

    for parent_record in parent_records:
        parent_id = parent_record["chunk_id"]
        metadata = dict(parent_record["metadata"])
        parent = TextNode(id_=parent_id, text=parent_record["text"], metadata=metadata)
        parent_nodes.append(parent)

        child_texts = splitter.split_text(parent_record["text"])
        for child_number, child_text in enumerate(child_texts, start=1):
            child_id = f"{parent_id}:child:{child_number:04d}"
            child_node = TextNode(
                id_=child_id,
                text=child_text,
                metadata={"parent_chunk_id": parent_id},
                excluded_embed_metadata_keys=["parent_chunk_id"],
                excluded_llm_metadata_keys=["parent_chunk_id"],
            )
            child_index_node = IndexNode.from_text_node(child_node, parent_id)
            child_index_node.id_ = child_id
            child_index_node.excluded_embed_metadata_keys = ["parent_chunk_id"]
            child_index_node.excluded_llm_metadata_keys = ["parent_chunk_id"]
            child_index_nodes.append(child_index_node)

            child_metadata = dict(metadata)
            child_metadata.update(
                {
                    "parent_chunk_id": parent_id,
                    "child_index": child_number,
                    "parent_chars": len(parent_record["text"]),
                    "parent_tokens": _token_count(splitter, parent_record["text"]),
                    "child_tokens": _token_count(splitter, child_text),
                }
            )
            child_records.append(
                {
                    "chunk_id": child_id,
                    "text": child_text,
                    "metadata": child_metadata,
                }
            )

    return parent_nodes, child_index_nodes, child_records


def _run_retrieval(
    parents: list[TextNode], children: list[IndexNode], output_dir: Path
) -> list[dict[str, Any]]:
    embedding = _load_embedding_model()
    index = VectorStoreIndex(children, embed_model=embedding, show_progress=True)
    vector_retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
    node_dict = {node.node_id: node for node in parents}
    retriever = RecursiveRetriever(
        root_id="child_vector_index",
        retriever_dict={"child_vector_index": vector_retriever},
        node_dict=node_dict,
        verbose=False,
    )

    results: list[dict[str, Any]] = []
    for trial_query in TRIAL_QUERIES:
        parent_hits = retriever.retrieve(trial_query["question"])
        seen: set[str] = set()
        hits: list[dict[str, Any]] = []
        for hit in parent_hits:
            parent = hit.node
            if parent.node_id in seen:
                continue
            seen.add(parent.node_id)
            metadata = dict(parent.metadata)
            title = str(metadata.get("title", ""))
            hits.append(
                {
                    "parent_chunk_id": parent.node_id,
                    "title": title,
                    "file_type": metadata.get("file_type"),
                    "page_start": metadata.get("page_start"),
                    "page_end": metadata.get("page_end"),
                    "score": hit.score,
                    "parent_chars": len(parent.text),
                    "parent_tokens": _token_count(
                        SentenceSplitter(chunk_size=CHILD_SIZE, chunk_overlap=CHILD_OVERLAP),
                        parent.text,
                    ),
                    "text_preview": parent.text[:700],
                }
            )

        target_fragments = trial_query["expected_title_fragments"]
        expected_hit = any(
            any(fragment.casefold() in hit["title"].casefold() for fragment in target_fragments)
            for hit in hits
        )
        results.append(
            {
                "question": trial_query["question"],
                "expected_title_fragments": target_fragments,
                "expected_source_in_top_k": expected_hit,
                "unique_parent_count": len(hits),
                "hits": hits,
            }
        )
    return results


def _write_readme(
    output_dir: Path,
    parent_records: list[dict[str, Any]],
    child_records: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
) -> None:
    parent_sizes = [record["metadata"]["parent_tokens"] for record in child_records]
    child_sizes = [record["metadata"]["child_tokens"] for record in child_records]
    type_counts = Counter(record["metadata"].get("file_type", "unknown") for record in parent_records)
    eval_hits = sum(result["expected_source_in_top_k"] for result in retrieval_results)

    lines = [
        "# 父子块检索试验",
        "",
        "父块直接复用上一轮输出的 966 个切块；每个父块内部用 LlamaIndex `SentenceSplitter` 切为 128-token 子块，overlap=24。",
        "向量只建立在子块上；LlamaIndex `RecursiveRetriever` 根据 `IndexNode.index_id` 回取父块。本试验没有持久化向量索引。",
        "",
        f"- 父块：{len(parent_records)}（" + "、".join(f"{kind} {count}" for kind, count in sorted(type_counts.items())) + ")",
        f"- 子块：{len(child_records)}",
        f"- 子块 token 数：最小 {min(child_sizes)} / 中位 {median(child_sizes):.0f} / 最大 {max(child_sizes)}",
        f"- 父块 token 数：最小 {min(parent_sizes)} / 中位 {median(parent_sizes):.0f} / 最大 {max(parent_sizes)}",
        f"- 试跑查询：{len(retrieval_results)}；预期来源进入 top-{RETRIEVAL_TOP_K}：{eval_hits}/{len(retrieval_results)}",
        "- 试跑包含中英文问题，但目标来源匹配仅作人工标注的粗略 hit@k，不等价于回答正确率。",
        "- `parents.jsonl` 和 `children.jsonl` 保留来源元数据；`检索试跑.jsonl` 记录父级结果和片段预览。",
        "",
        "## 注意",
        "",
        "1. 当前父块就是上一轮已经截成 512-token 上限的块，试验没有回到原始语义分割边界；PDF/Word 父块可能带原切分 overlap。",
        "2. HTML 父块按标题结构生成，父级长度不一定统一；极长标题区仍可能返回较长上下文。",
        "3. 同一父块可命中多个子块，递归检索会按父节点 ID 去重；必须对最终父块数量和总 token 数设上限。",
        "4. 父块继承已有页码范围。若子块边界需精确引用，必须另算子块实际页码，不能把父级页码范围误当成子块范围。",
        "5. 六条问题是小样本试跑；不能据此证明父子检索一定优于平铺切块。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    preview = ["# 父子块检索试跑预览", ""]
    for result in retrieval_results:
        preview.extend(
            [
                f"## 问题：{result['question']}",
                "",
                f"预期来源进入 top-{RETRIEVAL_TOP_K}：{'是' if result['expected_source_in_top_k'] else '否'}；去重后父块数：{result['unique_parent_count']}",
                "",
            ]
        )
        for rank, hit in enumerate(result["hits"], start=1):
            preview.extend(
                [
                    f"### {rank}. {hit['title']} — {hit['parent_chunk_id']}",
                    "",
                    f"类型：{hit['file_type']}；score：{hit['score']:.4f}；父块：{hit['parent_tokens']} tokens / {hit['parent_chars']} 字符；页码：{hit['page_start']}–{hit['page_end']}",
                    "",
                    hit["text_preview"],
                    "",
                ]
            )
    (output_dir / "检索试跑预览.md").write_text("\n".join(preview), encoding="utf-8")


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Parent chunk source directory is missing: {SOURCE_DIR}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {OUTPUT_DIR}")

    load_env_file(PROJECT_ROOT / ".env")
    parent_records = _read_jsonl(SOURCE_DIR / "chunks.jsonl")
    splitter = SentenceSplitter(chunk_size=CHILD_SIZE, chunk_overlap=CHILD_OVERLAP)
    parents, children, child_records = _build_parent_and_child_nodes(parent_records, splitter)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _write_jsonl(OUTPUT_DIR / "parents.jsonl", parent_records)
    _write_jsonl(OUTPUT_DIR / "children.jsonl", child_records)
    retrieval_results = _run_retrieval(parents, children, OUTPUT_DIR)
    _write_jsonl(OUTPUT_DIR / "检索试跑.jsonl", retrieval_results)
    _write_readme(OUTPUT_DIR, parent_records, child_records, retrieval_results)
    print(
        f"Built {len(parent_records)} parent chunks and {len(child_records)} child chunks; "
        f"results: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
