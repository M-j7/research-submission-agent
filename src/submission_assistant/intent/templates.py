"""One example library for BM25, embedding and JEV few-shot criteria."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import OPTIONS, OPTION_BY_ID, NON_BUSINESS_ID, RETRIEVAL_OPTION_BY_ID

# 三路共用同一份示例库，避免各路的“C02”实际拿了不同定义或不同版本的材料。
TEMPLATE_PATH = Path(__file__).parent / "data" / "templates.json"
NON_BUSINESS_TEMPLATE_PATH = Path(__file__).parent / "data" / "non_business_templates.json"


def load_templates(path: Path = TEMPLATE_PATH) -> dict[str, tuple[str, ...]]:
    # 完整九类、每类五条、全局去重。例子用于匹配与提示，不是独立评测集。
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise ValueError("无法读取意图模板，请检查 UTF-8 JSON 文件。") from None
    if not isinstance(data, dict) or set(data) != set(OPTION_BY_ID):
        raise ValueError("模板必须完整包含 C01–C09，不能有未知类别。")
    output = {}
    seen = set()
    for option in OPTIONS:
        texts = data[option.id]
        if not isinstance(texts, list) or len(texts) != 5:
            raise ValueError("当前每个意图需要五条不同的示例。")
        clean = []
        for text in texts:
            if not isinstance(text, str) or not text.strip() or text.strip().casefold() in seen:
                raise ValueError("模板不能为空，也不能在同类或跨类别重复。")
            clean.append(text.strip())
            seen.add(text.strip().casefold())
        output[option.id] = tuple(clean)
    return output


def load_retrieval_templates(path: Path = TEMPLATE_PATH, *,
                             non_business_path: Path = NON_BUSINESS_TEMPLATE_PATH) -> dict[str, tuple[str, ...]]:
    # JEV 暂时仍读取九类业务模板；两路检索共同读取“业务45条＋非业务对照示例”。
    # 分文件保留边界，不能把扩充模板误认为十项 JEV 独立判断已经实现。
    business = load_templates(path)
    try:
        data = json.loads(non_business_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise ValueError("无法读取非业务示例，请检查 UTF-8 JSON 文件。") from None
    if not isinstance(data, dict) or set(data) != {NON_BUSINESS_ID}:
        raise ValueError("非业务示例文件必须且只能包含 C10。")
    texts = data[NON_BUSINESS_ID]
    if not isinstance(texts, list) or not texts:
        raise ValueError("非业务示例必须是非空字符串列表。")
    seen = {text.casefold() for group in business.values() for text in group}
    clean = []
    for text in texts:
        if not isinstance(text, str) or not text.strip() or text.strip().casefold() in seen:
            raise ValueError("非业务示例不能为空，不能与业务示例或自身重复。")
        clean.append(text.strip())
        seen.add(text.strip().casefold())
    return {**business, NON_BUSINESS_ID: tuple(clean)}


def validate_message(message: str) -> str:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("当前消息必须是非空字符串。")
    return message.strip()


@dataclass(frozen=True)
class TemplateIntentResult:
    source: str
    score_type: str
    scores: dict[str, float]
    matched_examples: dict[str, str | None]
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # 这里的 selected_intent 只是“模板匹配第一名”，语义不同于融合输出的有效候选。
        # 是否为业务、是否要澄清，应检查 JEV 对话性质与清晰度，不能只读这个字段。
        ranked = sorted(self.scores, key=lambda key: (-self.scores[key], key))
        # Zero/negative similarity is not a positive match. Positive scores are
        # still only candidates; this is not an in-domain or clarity decision.
        selected = ranked[0] if self.scores[ranked[0]] > 0 else None
        return {
            "source": self.source, "score_type": self.score_type,
            "selected_intent": selected,
            "selected_name": RETRIEVAL_OPTION_BY_ID[selected].name if selected else None,
            "scores": dict(self.scores), "matched_examples": dict(self.matched_examples),
            "ranked_options": [{"intent": key, "name": RETRIEVAL_OPTION_BY_ID[key].name,
                                "score": self.scores[key], "example": self.matched_examples[key]} for key in ranked],
            "latency_ms": self.latency_ms, **self.metadata,
        }
