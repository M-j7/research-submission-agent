"""Conservative, reversible cleaning trial for the IEEE Access source set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


DROP_NAME_RE = re.compile(
    r"cookie|consent|osano|breadcrumb|page[_-]?anchors|subsite[_-]?menu|"
    r"main[_-]?site[_-]?menu|(?:site|global)[_-]?(?:nav|menu|header|footer)|"
    r"utility[_-]?nav|skip[_-]?(?:to[_-]?)?content|skip[_-]?link",
    re.IGNORECASE,
)
DROP_TAGS = {"script", "style", "noscript", "svg", "iframe", "nav", "footer"}
BLOCK_TAGS = {
    "address", "blockquote", "dd", "div", "dl", "dt", "figcaption", "figure",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "ol", "p", "pre", "section",
    "table", "ul",
}
PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?\d+(?:\s+(?:of|/)\s+\d+)?$", re.IGNORECASE)
PDF_PAGE_RE = re.compile(r"(?m)^## PDF page (\d+)\s*$")


def normalize_plain_text(text: str) -> str:
    """Normalize whitespace without rewriting sentence or list boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _inline_text(node: Tag | NavigableString) -> str:
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node))
    if not isinstance(node, Tag):
        return ""
    if node.name == "br":
        return " "
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, Tag) and child.name in {"script", "style", "noscript"}:
            continue
        if isinstance(child, Tag) and child.name == "a":
            label = _inline_text(child).strip()
            href = (child.get("href") or "").strip()
            if label and href and not href.startswith(("#", "javascript:")):
                parts.append(f"[{label}]({href})")
            elif label:
                parts.append(label)
            continue
        parts.append(_inline_text(child))
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()


def _render_table(table: Tag) -> list[str]:
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            _inline_text(cell)
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _render_node(node: Tag | NavigableString) -> list[str]:
    if isinstance(node, Comment):
        return []
    if isinstance(node, NavigableString):
        text = re.sub(r"\s+", " ", str(node)).strip()
        return [text] if text else []
    if not isinstance(node, Tag):
        return []

    name = (node.name or "").lower()
    if name in {"script", "style", "noscript", "svg", "iframe"}:
        return []
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        text = _inline_text(node)
        return [f"{'#' * int(name[1])} {text}"] if text else []
    if name == "p" or name == "blockquote" or name == "pre":
        text = _inline_text(node)
        if not text:
            return []
        return [f"> {text}" if name == "blockquote" else text]
    if name == "li":
        text = _inline_text(node)
        return [f"- {text}"] if text else []
    if name == "table":
        return _render_table(node)
    if name in {"ul", "ol", "dl"}:
        lines: list[str] = []
        for child in node.find_all(["li", "dt", "dd"], recursive=False):
            text = _inline_text(child)
            if text:
                lines.append(f"- {text}" if child.name == "li" else text)
        return lines

    lines: list[str] = []
    inline_parts: list[str] = []
    for child in node.children:
        if isinstance(child, Tag) and child.name in BLOCK_TAGS:
            if inline_parts:
                inline = re.sub(r"\s+", " ", " ".join(inline_parts)).strip()
                if inline:
                    lines.append(inline)
                inline_parts = []
            lines.extend(_render_node(child))
        else:
            inline_parts.append(_inline_text(child))
    if inline_parts:
        inline = re.sub(r"\s+", " ", " ".join(inline_parts)).strip()
        if inline:
            lines.append(inline)
    return lines


def clean_html(html: str) -> tuple[str, list[str]]:
    """Remove obvious site chrome and render the remaining structure as text."""
    soup = BeautifulSoup(html, "html.parser")
    removed: list[str] = []

    for element in list(soup.find_all(True)):
        if element.parent is None:
            continue
        name = (element.name or "").lower()
        identity = " ".join(
            [str(element.get("id") or ""), " ".join(map(str, element.get("class") or [])),
             str(element.get("role") or "")]
        )
        is_site_header = name == "header" and (
            bool(element.find("nav")) or bool(DROP_NAME_RE.search(identity))
        )
        if name in DROP_TAGS or is_site_header or DROP_NAME_RE.search(identity):
            snippet = _inline_text(element)[:120]
            removed.append(f"<{name}> {snippet}".strip())
            element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    content: Tag | BeautifulSoup = soup.body or soup
    candidates = content.find_all(["main", "article"])
    if candidates:
        content = max(candidates, key=lambda element: len(element.get_text(" ", strip=True)))

    lines = _render_node(content)
    cleaned = normalize_plain_text("\n\n".join(lines))
    has_content_heading = bool(content.find(re.compile(r"^h[1-6]$")))
    if title and not has_content_heading and title.casefold() not in cleaned[:300].casefold():
        cleaned = normalize_plain_text(f"# {title}\n\n{cleaned}")
    return cleaned, removed


def clean_pdf_pages(pages: Iterable[str]) -> tuple[list[str], list[str]]:
    """Normalize pages and remove only page-edge lines repeated across pages."""
    normalized = [normalize_plain_text(page) for page in pages]
    edge_counts: Counter[str] = Counter()
    for page in normalized:
        lines = [line for line in page.splitlines() if line.strip()]
        edge_counts.update({
            line.casefold()
            for line in lines[:2] + lines[-2:]
            if len(line) >= 4 and not PAGE_NUMBER_RE.fullmatch(line.strip())
        })

    threshold = max(2, math.ceil(len(normalized) * 0.60))
    repeated_edges = {line for line, count in edge_counts.items() if count >= threshold}
    removed: list[str] = []
    cleaned_pages: list[str] = []

    for page in normalized:
        lines = page.splitlines()
        nonblank_positions = [index for index, line in enumerate(lines) if line.strip()]
        edge_positions = set(nonblank_positions[:2] + nonblank_positions[-2:])
        kept: list[str] = []
        for index, line in enumerate(lines):
            candidate = line.strip()
            if not candidate:
                kept.append("")
                continue
            if index in edge_positions and candidate.casefold() in repeated_edges:
                removed.append(candidate)
                continue
            if index in edge_positions:
                page_match = PAGE_NUMBER_RE.fullmatch(candidate)
                if page_match:
                    number_match = re.search(r"\d+", candidate)
                    if number_match and int(number_match.group()) <= len(normalized):
                        removed.append(candidate)
                        continue
            kept.append(line)
        cleaned_pages.append(normalize_plain_text("\n".join(kept)))

    return cleaned_pages, removed


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_source_metadata(entry: dict | None) -> list[str]:
    """Render recorded provenance; never substitute filesystem dates."""
    aliases = (entry or {}).get("source_aliases") or []
    if not aliases:
        return [
            "- Source URL: not recorded in the official source manifest",
            "- Final URL: not recorded in the official source manifest",
            "- Acquisition time: not recorded in the official source manifest",
            "- HTTP Last-Modified: not recorded in the official source manifest",
        ]

    lines: list[str] = []
    for index, alias in enumerate(aliases, start=1):
        label = f"Source {index}" if len(aliases) > 1 else "Source"
        lines.append(f"- {label} URL: `{alias.get('url') or 'not recorded'}`")
        lines.append(f"- {label} final URL: `{alias.get('final_url') or 'not recorded'}`")
        lines.append(f"- {label} acquisition time: `{alias.get('retrieved_at_utc') or 'not recorded'}`")
        lines.append(f"- {label} HTTP Last-Modified: `{alias.get('last_modified_http') or 'not recorded'}`")
    return lines


def _visible_html_char_count(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.find_all(["script", "style", "noscript", "svg", "iframe"]):
        element.decompose()
    return len(soup.get_text(" ", strip=True))


def _extract_pdf_pages(preview: str) -> tuple[list[tuple[str, str]], str]:
    matches = list(PDF_PAGE_RE.finditer(preview))
    pages: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(preview)
        pages.append((match.group(1), preview[match.end():end]))
    return pages, preview


def _extract_preview_body(preview: str) -> str:
    marker = "Source hash unchanged: True"
    if marker in preview:
        return preview.split(marker, 1)[1].lstrip("\n")
    return preview


def _write_document(path: Path, title: str, source: Path, project_root: Path,
                    file_type: str, source_hash: str, cleaning_note: str,
                    body: str, source_metadata: list[str]) -> None:
    relative_source = source.relative_to(project_root).as_posix()
    content = (
        f"# {title}\n\n"
        f"- Source file: `{relative_source}`\n"
        f"- Type: {file_type}\n"
        f"- SHA-256: `{source_hash}`\n"
        + "\n".join(source_metadata) + "\n"
        f"- Cleaning: {cleaning_note}\n\n"
        f"{body.strip()}\n"
    )
    path.write_text(content, encoding="utf-8")


def run_cleaning(project_root: Path, output_dir: Path) -> Path:
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_base = project_root / "官方原始资料库" / "IEEE Access"
    official_root = project_root / "官方原始资料库"
    manifest_path = official_root / "来源清单.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_entries = {
        str(entry.get("file", "")).replace("\\", "/"): entry
        for entry in manifest.get("entries", [])
    }

    def source_entry(source: Path) -> dict | None:
        key = source.relative_to(official_root).as_posix()
        return manifest_entries.get(key)

    def source_hash_and_metadata(source: Path) -> tuple[str, list[str], bool]:
        digest = _source_hash(source)
        entry = source_entry(source)
        expected = (entry or {}).get("sha256")
        if expected and expected.lower() != digest:
            raise ValueError(f"Source manifest SHA-256 does not match {source}")
        return digest, format_source_metadata(entry), bool(entry and entry.get("source_aliases"))

    html_dir = source_base / "网页"
    doc_dir = source_base / "文档"
    preview_dir = project_root / "SimpleDirectoryReader试读预览" / "LlamaIndex输出"
    records: list[dict[str, str | int]] = []

    for source in sorted(html_dir.glob("*.html"), key=lambda path: path.name.casefold()):
        raw = source.read_text(encoding="utf-8", errors="replace")
        cleaned, removed = clean_html(raw)
        output = output_dir / f"{source.stem}.cleaned.md"
        title = source.stem
        source_soup = BeautifulSoup(raw, "html.parser")
        if source_soup.title and source_soup.title.get_text(" ", strip=True):
            title = source_soup.title.get_text(" ", strip=True)
        note = f"HTML boilerplate removed ({len(removed)} elements); structure and visible links retained"
        source_digest, source_metadata, has_provenance = source_hash_and_metadata(source)
        _write_document(output, title, source, project_root, "HTML", source_digest, note,
                        cleaned, source_metadata)
        records.append({
            "name": source.name, "type": "HTML", "output": output.name,
            "input_chars": _visible_html_char_count(raw), "output_chars": len(cleaned),
            "removed": len(removed), "details": "; ".join(removed[:8]) or "none",
            "has_provenance": has_provenance,
        })

    pdf_sources = sorted(doc_dir.glob("*.pdf"), key=lambda path: path.name.casefold())
    for source in pdf_sources:
        preview_path = preview_dir / f"{source.stem}__pdf.md"
        if not preview_path.exists():
            raise FileNotFoundError(f"Missing SimpleDirectoryReader PDF output: {preview_path}")
        preview = preview_path.read_text(encoding="utf-8-sig", errors="replace")
        page_records, _ = _extract_pdf_pages(preview)
        page_numbers = [number for number, _ in page_records]
        cleaned_pages, removed = clean_pdf_pages(page for _, page in page_records)
        body = "\n\n".join(
            f"## PDF page {number}\n\n{text}" for number, text in zip(page_numbers, cleaned_pages)
        )
        output = output_dir / f"{source.stem}.cleaned.md"
        note = f"Whitespace normalized; removed {len(removed)} repeated page-edge/header/footer items"
        source_digest, source_metadata, has_provenance = source_hash_and_metadata(source)
        _write_document(output, source.stem, source, project_root, "PDF", source_digest, note,
                        body, source_metadata)
        records.append({
            "name": source.name, "type": "PDF", "output": output.name,
            "input_chars": sum(len(page) for _, page in page_records), "output_chars": len(body),
            "removed": len(removed), "details": "; ".join(dict.fromkeys(removed)) or "none",
            "has_provenance": has_provenance,
        })

    docx_sources = sorted(doc_dir.glob("*.docx"), key=lambda path: path.name.casefold())
    for source in docx_sources:
        preview_path = preview_dir / f"{source.stem}__docx.md"
        if not preview_path.exists():
            raise FileNotFoundError(f"Missing SimpleDirectoryReader DOCX output: {preview_path}")
        preview = preview_path.read_text(encoding="utf-8-sig", errors="replace")
        body = normalize_plain_text(_extract_preview_body(preview))
        output = output_dir / f"{source.stem}.cleaned.md"
        note = "Whitespace normalized; template examples and placeholders preserved"
        source_digest, source_metadata, has_provenance = source_hash_and_metadata(source)
        _write_document(output, source.stem, source, project_root, "DOCX template",
                        source_digest, note, body, source_metadata)
        records.append({
            "name": source.name, "type": "DOCX template", "output": output.name,
            "input_chars": len(_extract_preview_body(preview)), "output_chars": len(body),
            "removed": 0, "details": "template examples and placeholders retained",
            "has_provenance": has_provenance,
        })

    expected_count = len(list(html_dir.glob("*.html"))) + len(pdf_sources) + len(docx_sources)
    if len(records) != expected_count:
        raise RuntimeError(f"Processed {len(records)} sources; expected {expected_count}")

    type_counts = Counter(str(row["type"]) for row in records)
    provenance_count = sum(bool(row.get("has_provenance")) for row in records)
    report_lines = [
        "# 简单规则清洗试验",
        "",
        "本试验只生成清洗副本；未修改原始 HTML、PDF、DOCX，也未覆盖 SimpleDirectoryReader 试读稿。",
        "已删除的 7 份网页打印 PDF 不在当前原件目录，因此其旧试读稿未复制到本次结果。",
        "",
        f"- 输入：{len(records)} 份（" + "、".join(f"{kind} {count} 份" for kind, count in type_counts.items()) + ")",
        f"- 来源清单：{provenance_count} 份带原始 URL/采集时间记录；其余 {len(records) - provenance_count} 份明确标记为未记录，不以文件系统时间代替。",
        "- HTML：移除 script/style/noscript/svg/iframe、nav/footer、明显的 Cookie/Consent/面包屑/全站导航容器；保留正文标题、段落、列表、表格行与可见链接。",
        "- PDF：只规范空白；只删除页首/页尾独立页码，以及至少 60% 页数重复出现的页边行。内部换行、条款及正文均保留。",
        "- DOCX：只规范空白；模板示例和占位内容保留，并标记为模板，不作为政策文件处理。",
        "- 本轮没有调用 LLM/OCR，没有切块、生成 embedding 或写向量库。",
        "",
        "## 文件级结果",
        "",
        "| 输入文件 | 类型 | 输入字符 | 清洗后字符 | 删除项数 | 结果文件 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in records:
        report_lines.append(
            f"| {row['name']} | {row['type']} | {row['input_chars']} | {row['output_chars']} | "
            f"{row['removed']} | [{row['output']}]({row['output']}) |"
        )
    report_lines.extend(["", "## 删除项摘要", ""])
    for row in records:
        if row["removed"]:
            report_lines.append(f"- **{row['name']}**：{row['details']}")
    (output_dir / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        args.project_root / "解析试验" / "简单清洗" / "结果" /
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    print(run_cleaning(args.project_root, output_dir))


if __name__ == "__main__":
    main()
