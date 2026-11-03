"""Create reversible, source-aware chunks from the approved cleaned corpus."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any

from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.node_parser import (
    HTMLNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "解析试验" / "简单清洗" / "结果" / "20261006-final"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "解析试验"
    / "语义分割"
    / "结果"
    / "20261006-qwen-text-embedding-flash"
)
PDF_PAGE_RE = re.compile(r"(?m)^## PDF page (\d+)\s*$")
HEADER_RE = re.compile(r"^#\s+(.+?)\s*$")
FIELD_RE = re.compile(r"^-\s+([^:]+):\s*(.*?)\s*$")
DROP_NAME_RE = re.compile(
    r"cookie|consent|osano|breadcrumb|page[_-]?anchors|subsite[_-]?menu|"
    r"main[_-]?site[_-]?menu|(?:site|global)[_-]?(?:nav|menu|header|footer)|"
    r"utility[_-]?nav|skip[_-]?(?:to[_-]?)?content|skip[_-]?link",
    re.IGNORECASE,
)
DROP_TAGS = {"script", "style", "noscript", "svg", "iframe", "nav", "footer"}
HTML_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"]


def load_env_file(path: Path) -> None:
    """Load simple KEY=value entries without printing or overwriting process env."""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def parse_cleaned_markdown(text: str) -> tuple[dict[str, str], str]:
    """Separate the cleaning report header from the preserved document body."""
    lines = text.splitlines()
    metadata: dict[str, str] = {}
    if lines:
        title_match = HEADER_RE.match(lines[0])
        if title_match:
            metadata["title"] = title_match.group(1).strip()

    field_names = {
        "Source file": "source_file",
        "Type": "file_type",
        "SHA-256": "source_sha256",
        "Source URL": "source_url",
        "Source final URL": "final_url",
        "Source acquisition time": "retrieved_at",
        "Source HTTP Last-Modified": "http_last_modified",
    }
    cleaning_line = None
    for index, line in enumerate(lines):
        field_match = FIELD_RE.match(line)
        if not field_match:
            continue
        name, value = field_match.groups()
        if name == "Cleaning":
            cleaning_line = index
            break
        target = field_names.get(name)
        if target:
            metadata[target] = value.strip().strip("`")

    if cleaning_line is None:
        raise ValueError("cleaned Markdown is missing its '- Cleaning:' header")
    body = "\n".join(lines[cleaning_line + 1 :]).strip()
    if not body:
        raise ValueError("cleaned Markdown has an empty body")
    return metadata, body


def split_pdf_pages(body: str) -> list[tuple[int, str]]:
    """Return page-numbered text blocks from the cleaner's explicit page markers."""
    matches = list(PDF_PAGE_RE.finditer(body))
    if not matches:
        raise ValueError("PDF body is missing '## PDF page N' markers")
    pages: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        page_text = body[match.end() : end].strip()
        if page_text:
            pages.append((int(match.group(1)), page_text))
    return pages


def _cleaned_html_fragment(html: str) -> str:
    """Apply the same conservative site-chrome removals used by the clean trial."""
    soup = BeautifulSoup(html, "html.parser")
    for element in list(soup.find_all(True)):
        if element.parent is None:
            continue
        name = (element.name or "").lower()
        identity = " ".join(
            [
                str(element.get("id") or ""),
                " ".join(map(str, element.get("class") or [])),
                str(element.get("role") or ""),
            ]
        )
        is_site_header = name == "header" and (
            bool(element.find("nav")) or bool(DROP_NAME_RE.search(identity))
        )
        if name in DROP_TAGS or is_site_header or DROP_NAME_RE.search(identity):
            element.decompose()

    content = soup.body or soup
    candidates = content.find_all(["main", "article"])
    if candidates:
        content = max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))
    return str(content)


def split_html_structurally(
    html: str, metadata: dict[str, Any]
) -> list[Document]:
    """Use HTMLNodeParser, grouping leaf elements under their nearest headings."""
    cleaned_fragment = _cleaned_html_fragment(html)
    source = Document(text=cleaned_fragment, metadata=dict(metadata))
    parsed = HTMLNodeParser(tags=HTML_TAGS).get_nodes_from_documents([source])

    groups: list[dict[str, Any]] = []
    heading_stack: list[tuple[int, str]] = []
    for node in parsed:
        tag = str(node.metadata.get("tag", "")).lower()
        text = node.text.strip()
        if not text:
            continue
        if re.fullmatch(r"h[1-6]", tag):
            level = int(tag[1])
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            continue

        header_path = "/".join(title for _, title in heading_stack)
        if not groups or groups[-1]["header_path"] != header_path:
            groups.append({"header_path": header_path, "blocks": [], "tags": []})
        group = groups[-1]
        # HTMLNodeParser can see nested paragraph/list/table text more than once.
        # Ignore only an immediately repeated identical node; don't collapse
        # repeated wording elsewhere in the source.
        if not group["blocks"] or group["blocks"][-1] != text:
            group["blocks"].append(text)
            group["tags"].append(tag)

    documents: list[Document] = []
    for group in groups:
        header_path = str(group["header_path"])
        body = "\n\n".join(group["blocks"]).strip()
        if header_path:
            body = f"[{header_path}]\n\n{body}"
        node_metadata = dict(metadata)
        node_metadata.update(
            {
                "header_path": header_path,
                "html_tags": ",".join(group["tags"]),
                "split_method": "HTMLNodeParser",
            }
        )
        documents.append(Document(text=body, metadata=node_metadata))
    return documents


def _load_embedding_model() -> OpenAIEmbedding:
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    api_base = os.getenv("EMBEDDING_BASE_URL", "").strip()
    model_name = os.getenv("EMBEDDING_MODEL", "").strip()
    missing = [
        name
        for name, value in (
            ("EMBEDDING_API_KEY", api_key),
            ("EMBEDDING_BASE_URL", api_base),
            ("EMBEDDING_MODEL", model_name),
        )
        if not value
    ]
    if missing:
        raise ValueError("Missing embedding configuration: " + ", ".join(missing))
    return OpenAIEmbedding(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        # Qwen's embedding endpoint rejects batches larger than 20 inputs.
        embed_batch_size=20,
        timeout=90.0,
    )


def _pdf_page_range(text: str, current_page: int | None) -> tuple[int | None, int | None, int | None]:
    page_numbers = [int(value) for value in PDF_PAGE_RE.findall(text)]
    if page_numbers:
        return current_page or page_numbers[0], page_numbers[-1], page_numbers[-1]
    return current_page, current_page, current_page


def _cap_chunk_size(
    text: str, metadata: dict[str, Any], splitter: SentenceSplitter
) -> list[TextNode]:
    return [TextNode(text=part, metadata=dict(metadata)) for part in splitter.split_text(text)]


def _to_chunk_records(
    source_id: str,
    nodes: list[Any],
    cap_splitter: SentenceSplitter,
    page_aware: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_page: int | None = None
    for node in nodes:
        text = node.text.strip()
        metadata = dict(node.metadata)
        if page_aware:
            page_start, page_end, current_page = _pdf_page_range(text, current_page)
            metadata["page_start"] = page_start
            metadata["page_end"] = page_end
            text = PDF_PAGE_RE.sub("", text).strip()
        if not text:
            continue

        for subnode in _cap_chunk_size(text, metadata, cap_splitter):
            records.append(
                {
                    "chunk_id": f"{source_id}:{len(records) + 1:05d}",
                    "text": subnode.text,
                    "metadata": dict(subnode.metadata),
                }
            )
    return records


def split_corpus(
    input_dir: Path,
    project_root: Path,
    embedding: OpenAIEmbedding | None,
    allowed_types: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cleaned_files = sorted(input_dir.glob("*.cleaned.md"), key=lambda p: p.name.casefold())
    if not cleaned_files:
        raise FileNotFoundError(f"No *.cleaned.md inputs in {input_dir}")

    semantic_splitter = None
    cap_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)
    all_chunks: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []

    for cleaned_path in cleaned_files:
        metadata, body = parse_cleaned_markdown(cleaned_path.read_text(encoding="utf-8"))
        metadata["cleaned_file"] = cleaned_path.name
        source_file = metadata.get("source_file", "")
        if not source_file:
            raise ValueError(f"Missing source file metadata in {cleaned_path.name}")
        source_id = metadata.get("source_sha256", "")[:16] or cleaned_path.stem
        file_type = metadata.get("file_type", "").upper()
        if allowed_types and not any(
            file_type == allowed_type or file_type.startswith(allowed_type)
            for allowed_type in allowed_types
        ):
            continue

        if file_type == "HTML":
            source_path = (project_root / Path(*PurePosixPath(source_file).parts)).resolve()
            if not source_path.is_relative_to(project_root.resolve()):
                raise ValueError(f"Source escapes project root: {source_file}")
            html = source_path.read_text(encoding="utf-8", errors="replace")
            structural_docs = split_html_structurally(html, metadata)
            if not structural_docs:
                raise ValueError(f"HTMLNodeParser produced no body nodes for {source_file}")
            nodes = []
            for section in structural_docs:
                nodes.extend(cap_splitter.get_nodes_from_documents([section]))
            split_method = "HTMLNodeParser + SentenceSplitter"
            page_aware = False
        elif file_type == "PDF":
            if embedding is None:
                raise ValueError("PDF semantic splitting requires an embedding model")
            if semantic_splitter is None:
                semantic_splitter = SemanticSplitterNodeParser(
                    embed_model=embedding,
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                )
            pages = split_pdf_pages(body)
            text = "\n\n".join(
                f"## PDF page {page_number}\n\n{page_text}"
                for page_number, page_text in pages
            )
            document = Document(text=text, metadata=metadata)
            nodes = semantic_splitter.get_nodes_from_documents([document])
            split_method = "SemanticSplitterNodeParser + SentenceSplitter cap"
            page_aware = True
        elif file_type.startswith("DOCX"):
            if embedding is None:
                raise ValueError("DOCX semantic splitting requires an embedding model")
            if semantic_splitter is None:
                semantic_splitter = SemanticSplitterNodeParser(
                    embed_model=embedding,
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                )
            document = Document(text=body, metadata=metadata)
            nodes = semantic_splitter.get_nodes_from_documents([document])
            split_method = "SemanticSplitterNodeParser + SentenceSplitter cap"
            page_aware = False
        else:
            raise ValueError(f"Unsupported cleaned document type {file_type!r}: {cleaned_path.name}")

        source_chunks = _to_chunk_records(source_id, nodes, cap_splitter, page_aware)
        for index, chunk in enumerate(source_chunks, start=1):
            chunk["metadata"]["split_method"] = split_method
            chunk["metadata"]["source_chunk_index"] = index
        all_chunks.extend(source_chunks)
        lengths = [len(chunk["text"]) for chunk in source_chunks]
        report_rows.append(
            {
                "source_file": source_file,
                "title": metadata.get("title", cleaned_path.stem),
                "file_type": file_type,
                "chunks": len(source_chunks),
                "min_chars": min(lengths) if lengths else 0,
                "median_chars": round(median(lengths)) if lengths else 0,
                "max_chars": max(lengths) if lengths else 0,
                "source_id": source_id,
                "split_method": split_method,
            }
        )
    if not report_rows:
        raise ValueError("No cleaned documents matched the requested types")
    return all_chunks, report_rows


def write_results(
    output_dir: Path,
    chunks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    embedding_model: str,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "chunks.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    type_counts = Counter(row["file_type"] for row in rows)
    split_methods = sorted({row["split_method"] for row in rows})
    lines = [
        "# IEEE Access 语义/结构切分试验",
        "",
        "输入为已确认的 `20261006-final` 清洗副本；没有修改原始 HTML/PDF/DOCX，也没有生成或写入向量。",
        f"语义模型：`{embedding_model or '未调用（仅结构切分）'}`。语义边界阈值 95，buffer_size=1；最终块上限目标 512 tokens、overlap 80。",
        "本批实际切分方式：" + "；".join(f"`{method}`" for method in split_methods) + "。",
        "`chunks.jsonl` 不包含 embedding 向量或 API 密钥，保留来源哈希、URL、获取时间及可用的页码/标题路径。",
        "",
        f"- 文档数：{len(rows)}（" + "、".join(f"{key} {value}" for key, value in sorted(type_counts.items())) + ")",
        f"- 切块总数：{len(chunks)}",
        f"- 输出：`{jsonl_path.name}`",
        "",
        "## 各文档切块统计",
        "",
        "| 文档 | 类型 | 块数 | 最短字符 | 中位字符 | 最长字符 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['title']} | {row['file_type']} | {row['chunks']} | "
            f"{row['min_chars']} | {row['median_chars']} | {row['max_chars']} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sample_selectors = [
        ("HTML", "Submission Guidelines for Authors"),
        ("PDF", "IEEE Editorial Style Manual for Authors"),
        ("DOCX", "Microsoft Word"),
    ]
    sample_chunks: list[dict[str, Any]] = []
    for file_type, title_fragment in sample_selectors:
        source_row = next(
            (
                row
                for row in rows
                if row["file_type"].startswith(file_type)
                and (
                    title_fragment in row["title"]
                    or title_fragment in row["source_file"]
                )
            ),
            None,
        )
        if source_row:
            selected = [
                chunk
                for chunk in chunks
                if chunk["metadata"].get("source_sha256", "").startswith(source_row["source_id"])
            ][:3]
            sample_chunks.extend(selected)
    sample_lines = ["# 切块抽样预览", ""]
    for chunk in sample_chunks:
        metadata = chunk["metadata"]
        sample_lines.extend(
            [
                f"## {metadata.get('title', '')} — {chunk['chunk_id']}",
                "",
                f"- 类型：{metadata.get('file_type', '')}",
                f"- 页码：{metadata.get('page_start', '—')}–{metadata.get('page_end', '—')}",
                f"- HTML 标题路径：{metadata.get('header_path') or '—'}",
                "",
                chunk["text"],
                "",
            ]
        )
    (output_dir / "抽样预览.md").write_text("\n".join(sample_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    embedding_name = os.getenv("EMBEDDING_MODEL", "").strip()
    embedding = _load_embedding_model()
    chunks, rows = split_corpus(args.input_dir.resolve(), PROJECT_ROOT, embedding)
    write_results(args.output_dir.resolve(), chunks, rows, embedding_name)
    print(f"Processed {len(rows)} documents into {len(chunks)} chunks.")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
