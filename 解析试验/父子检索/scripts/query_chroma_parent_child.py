"""Query the persistent Chroma child index and print parent-context results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_SCRIPT_DIR = PROJECT_ROOT / "解析试验" / "语义分割" / "scripts"
sys.path.insert(0, str(SPLIT_SCRIPT_DIR))

from split_documents import _load_embedding_model, load_env_file  # noqa: E402
from persist_chroma_parent_child import (  # noqa: E402
    CHROMA_DIR,
    CHILDREN_PATH,
    COLLECTION_NAME,
    PARENTS_PATH,
    read_jsonl,
    retrieve_dense_child_hits,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+", re.IGNORECASE)
TERM_GLOSSARY_PATH = Path(__file__).resolve().parents[1] / "术语对照表.csv"


def emit_console_text(text: str, stream: Any = None) -> None:
    """Print without crashing when Windows' active code page cannot encode a glyph."""
    output = stream if stream is not None else sys.stdout
    reconfigure = getattr(output, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")
    print(text, file=output)


def tokenize_for_bm25(text: str) -> list[str]:
    """Keep exact Latin/digit terms and contiguous CJK runs for lexical retrieval."""
    return TOKEN_PATTERN.findall(text.casefold())


def load_term_glossary(path: Path = TERM_GLOSSARY_PATH) -> dict[str, list[str]]:
    """Read Chinese terms and their English equivalents from the project CSV."""
    aliases: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as glossary_file:
        reader = csv.DictReader(glossary_file)
        required_columns = {"中文术语", "英文术语", "英文别名"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"术语表必须包含列：{', '.join(sorted(required_columns))}")
        for row in reader:
            chinese = (row.get("中文术语") or "").strip()
            if not chinese:
                continue
            english_terms = [
                term.strip()
                for column in ("英文术语", "英文别名")
                for term in (row.get(column) or "").split("|")
                if term.strip()
            ]
            aliases[chinese] = list(dict.fromkeys(english_terms))
    return aliases


def _term_occurs(term: str, query: str) -> bool:
    if re.search(r"[a-z0-9]", term, re.IGNORECASE):
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", query, re.IGNORECASE) is not None
    return term in query


def expand_query_for_bm25(query: str, aliases: dict[str, list[str]]) -> str:
    """Add curated cross-language equivalents without rewriting the user's query."""
    additions: list[str] = []
    for chinese, english_terms in aliases.items():
        terms = [chinese, *english_terms]
        if any(_term_occurs(term, query) for term in terms):
            additions.extend(term for term in terms if not _term_occurs(term, query))
    return " ".join([query, *dict.fromkeys(additions)]).strip()


def bm25_rank_children(
    query: str,
    children: list[dict[str, Any]],
    *,
    top_k: int,
    term_aliases: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return matching child IDs ranked by BM25; omit zero-score noise candidates."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as error:
        raise RuntimeError(
            "BM25 dependency is missing; update the ieee-submission-assistant Conda environment."
        ) from error

    aliases = term_aliases if term_aliases is not None else load_term_glossary()
    query_tokens = tokenize_for_bm25(expand_query_for_bm25(query, aliases))
    tokenized_children = [tokenize_for_bm25(str(child.get("text", ""))) for child in children]
    valid_rows = [
        (str(child.get("chunk_id", "")).strip(), tokens)
        for child, tokens in zip(children, tokenized_children)
        if str(child.get("chunk_id", "")).strip() and tokens
    ]
    if not query_tokens or not valid_rows:
        return []

    child_ids = [child_id for child_id, _ in valid_rows]
    bm25 = BM25Okapi([tokens for _, tokens in valid_rows])
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(
        ((float(score), index, child_id) for index, (score, child_id) in enumerate(zip(scores, child_ids))),
        key=lambda row: (-row[0], row[1]),
    )
    return [child_id for score, _, child_id in ranked if score > 0][:top_k]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    rank_constant: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse ranked child IDs by reciprocal rank, independent of source score scales."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 when provided")

    ranks_by_id: dict[str, list[int | None]] = {}
    first_seen: dict[str, int] = {}
    for list_index, ranked_ids in enumerate(ranked_lists):
        seen_in_list: set[str] = set()
        for rank, raw_child_id in enumerate(ranked_ids, start=1):
            child_id = str(raw_child_id).strip()
            if not child_id or child_id in seen_in_list:
                continue
            seen_in_list.add(child_id)
            if child_id not in ranks_by_id:
                ranks_by_id[child_id] = [None] * len(ranked_lists)
                first_seen[child_id] = len(first_seen)
            ranks_by_id[child_id][list_index] = rank

    fused: list[dict[str, Any]] = []
    for child_id, ranks in ranks_by_id.items():
        score = sum(1.0 / (rank_constant + rank) for rank in ranks if rank is not None)
        fused.append({"child_id": child_id, "rrf_score": score, "ranks": tuple(ranks)})
    fused.sort(
        key=lambda item: (
            -item["rrf_score"],
            min(rank for rank in item["ranks"] if rank is not None),
            first_seen[item["child_id"]],
        )
    )
    return fused[:top_k] if top_k is not None else fused


def retrieve_hybrid_child_chunks(
    query: str,
    children: list[dict[str, Any]],
    persist_dir: Path,
    embed_model: Any,
    *,
    collection_name: str = COLLECTION_NAME,
    candidate_k: int = 30,
    rank_constant: int = 60,
    term_aliases: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Run dense and BM25 child retrieval, then return RRF-ranked child records."""
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1")
    dense_hits = retrieve_dense_child_hits(
        query,
        persist_dir,
        embed_model,
        collection_name=collection_name,
        top_k=candidate_k,
    )
    dense_ids = [str(hit["child_id"]) for hit in dense_hits]
    sparse_ids = bm25_rank_children(query, children, top_k=candidate_k, term_aliases=term_aliases)
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids], rank_constant=rank_constant)
    children_by_id = {str(child["chunk_id"]): child for child in children}

    candidates: list[dict[str, Any]] = []
    for item in fused:
        child = children_by_id.get(item["child_id"])
        if child is None:
            raise ValueError(f"Retriever returned child ID absent from children.jsonl: {item['child_id']!r}")
        candidates.append(
            {
                **child,
                "rrf_score": item["rrf_score"],
                "dense_rank": item["ranks"][0],
                "bm25_rank": item["ranks"][1],
            }
        )
    return candidates


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Reranker 请求失败（HTTP {error.code}）。请检查模型名、API 地址及服务权限。") from error
    except urllib.error.URLError as error:
        raise RuntimeError("Reranker 网络请求失败；请检查网络和 API 地址。") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Reranker 返回的内容不是有效 JSON。") from error
    if not isinstance(decoded, dict):
        raise ValueError("Reranker 响应顶层必须是 JSON 对象")
    return decoded


def rerank_child_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int = 30,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 30.0,
    request_json: Any = None,
) -> list[dict[str, Any]]:
    """Rerank leading RRF children through Qwen's DashScope text-rerank API."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not candidates:
        return []
    base_url = (base_url or os.getenv("RERANKER_BASE_URL", "")).strip().rstrip("/")
    api_key = api_key or os.getenv("RERANKER_API_KEY", "").strip()
    model = (model or os.getenv("RERANKER_MODEL", "")).strip()
    if not base_url or not api_key or not model:
        raise RuntimeError("请在项目 .env 中配置 RERANKER_BASE_URL、RERANKER_API_KEY 和 RERANKER_MODEL。")

    base_parts = urllib.parse.urlsplit(base_url)
    if base_parts.scheme not in {"http", "https"} or not base_parts.netloc:
        raise ValueError("RERANKER_BASE_URL 必须是完整的 HTTP(S) 地址")
    endpoint = urllib.parse.urlunsplit(
        (base_parts.scheme, base_parts.netloc, "/api/v1/services/rerank/text-rerank/text-rerank", "", "")
    )

    selected = candidates[:top_k]
    tail = candidates[top_k:]
    request = request_json or _post_json
    response = request(
        endpoint,
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {
            "model": model,
            "input": {"query": query, "documents": [str(item.get("text", "")) for item in selected]},
            "parameters": {
                "top_n": len(selected),
                "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
            },
        },
        timeout,
    )
    output = response.get("output")
    results = output.get("results") if isinstance(output, dict) else response.get("results")
    if not isinstance(results, list):
        raise ValueError("Reranker 响应缺少 results 列表")
    if len(results) != len(selected):
        raise ValueError("Reranker 没有为每个候选返回分数；拒绝静默沿用 RRF 顺序")

    scored: list[tuple[float, int, dict[str, Any]]] = []
    seen_indexes: set[int] = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Reranker results 中包含无效条目")
        index = result.get("index")
        score = result.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(selected):
            raise ValueError("Reranker 返回了超出候选范围的 index")
        if index in seen_indexes:
            raise ValueError("Reranker 对同一候选返回了重复 index")
        if isinstance(score, bool) or not isinstance(score, (float, int)):
            raise ValueError("Reranker relevance_score 必须是数字")
        if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise ValueError("Reranker relevance_score 必须是 0 到 1 之间的有限数值")
        seen_indexes.add(index)
        scored.append((float(score), index, {**selected[index], "rerank_score": float(score)}))
    scored.sort(key=lambda item: (-item[0], item[1]))
    reranked = [item for _, _, item in scored]
    reranked.extend(
        {**item, "rerank_score": None}
        for index, item in enumerate(selected)
        if index not in seen_indexes
    )
    return reranked + tail


def expand_fused_candidates_to_parents(
    candidates: list[dict[str, Any]],
    parents_by_id: dict[str, dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Expand RRF-ranked child evidence to unique parent context in rank order."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    hits: list[dict[str, Any]] = []
    seen_parent_ids: set[str] = set()
    for child in candidates:
        metadata = child.get("metadata") or {}
        parent_id = str(metadata.get("parent_chunk_id", ""))
        if not parent_id:
            raise ValueError(f"Retrieved child {child.get('chunk_id')!r} has no parent_chunk_id metadata")
        if parent_id in seen_parent_ids:
            continue
        parent = parents_by_id.get(parent_id)
        if parent is None:
            raise ValueError(f"Retrieved child references missing parent {parent_id!r}")
        seen_parent_ids.add(parent_id)
        hits.append(
            {
                "parent_chunk_id": parent_id,
                "text": parent["text"],
                "metadata": parent.get("metadata", {}),
                "rrf_score": float(child["rrf_score"]),
                "dense_rank": child.get("dense_rank"),
                "bm25_rank": child.get("bm25_rank"),
                "rerank_score": child.get("rerank_score"),
                "matched_child_id": child["chunk_id"],
                "matched_child_text": child["text"],
            }
        )
        if len(hits) >= top_k:
            break
    return hits


def format_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "未检索到相关父块。"

    sections: list[str] = []
    for rank, hit in enumerate(hits, start=1):
        metadata = hit.get("metadata") or {}
        title = metadata.get("title") or "未命名来源"
        source_file = metadata.get("source_file") or "来源路径未知"
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        page = ""
        if page_start is not None:
            page = f"；页码 {page_start}–{page_end or page_start}"
        source_url = metadata.get("source_url")
        score = float(hit.get("rrf_score", hit.get("score", 0.0)))
        if "rrf_score" in hit:
            dense_rank = hit.get("dense_rank") or "未命中"
            bm25_rank = hit.get("bm25_rank") or "未命中"
            rerank_score = hit.get("rerank_score")
            rerank_label = f"Reranker {rerank_score:.4f}" if rerank_score is not None else "Reranker 未评分"
            rank_label = f"RRF {score:.6f}；{rerank_label}；向量名次 {dense_rank}；BM25 名次 {bm25_rank}"
        else:
            rank_label = f"相似度分数 {score:.4f}"

        section = [
            f"## {rank}. {title}（{rank_label}）",
            f"来源：{source_file}{page}",
        ]
        if source_url:
            section.append(f"网址：{source_url}")
        if hit.get("matched_child_text"):
            section.extend(["", "命中子块：", str(hit["matched_child_text"])])
        section.extend(["", str(hit["text"])])
        sections.append("\n".join(section))
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用向量与 BM25 双路召回、RRF 融合和 Reranker 重排后返回父块上下文。"
    )
    parser.add_argument("--query", required=True, help="要检索的问题")
    parser.add_argument("--top-k", type=int, default=10, help="最多返回的父块数，默认 10")
    parser.add_argument(
        "--candidate-k", type=int, default=30,
        help="向量路和 BM25 路各自召回的候选子块数，默认各 30",
    )
    parser.add_argument(
        "--rerank-k", type=int, default=30,
        help="RRF 排名前多少个子块送入 Reranker，默认 30",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k 必须大于 0")
    if args.candidate_k < 1:
        parser.error("--candidate-k 必须大于 0")
    if args.rerank_k < 1:
        parser.error("--rerank-k 必须大于 0")

    if not PARENTS_PATH.is_file() or not CHILDREN_PATH.is_file() or not CHROMA_DIR.is_dir():
        raise FileNotFoundError(
            "找不到 Chroma 索引、parents.jsonl 或 children.jsonl；请先运行 persist_chroma_parent_child.py。"
        )

    load_env_file(PROJECT_ROOT / ".env")
    embedding = _load_embedding_model()
    children = read_jsonl(CHILDREN_PATH)
    parents = read_jsonl(PARENTS_PATH)
    parents_by_id = {str(record["chunk_id"]): record for record in parents}
    candidates = retrieve_hybrid_child_chunks(
        args.query,
        children,
        CHROMA_DIR,
        embedding,
        collection_name=COLLECTION_NAME,
        candidate_k=args.candidate_k,
    )
    candidates = rerank_child_candidates(args.query, candidates, top_k=args.rerank_k)
    hits = expand_fused_candidates_to_parents(candidates, parents_by_id, top_k=args.top_k)
    emit_console_text(format_hits(hits))


if __name__ == "__main__":
    main()
