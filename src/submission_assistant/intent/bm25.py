"""Lexical template retrieval, not a hand-written business keyword router."""

from __future__ import annotations

import re
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from .templates import (TEMPLATE_PATH, NON_BUSINESS_TEMPLATE_PATH, TemplateIntentResult,
                        load_retrieval_templates, validate_message)


def tokenize(text: str) -> list[str]:
    # 不手写“退款→费用类”规则，而是把用户文本和示例都变成相同粒度的词项。
    # 中文连续片段按相邻双字切开，例如“投稿费用”→“投稿”“稿费”“费用”。
    # 英文和数字按连续词项保留；此做法轻量，但跨词边界的双字片段也可能带来噪声。
    # Without a new word-segmentation dependency, CJK bigrams allow partial
    # matching inside a sentence; Latin terms (APC, CC, PDF...) stay intact.
    tokens = []
    for part in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", text.casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


class BM25IntentRecognizer:
    def __init__(self, template_path: Path = TEMPLATE_PATH, *,
                 non_business_path: Path = NON_BUSINESS_TEMPLATE_PATH):
        # 业务45条＋非业务30条共同建索引；不调用模型，也不检索 RAG 知识库正文。
        self.templates = load_retrieval_templates(template_path, non_business_path=non_business_path)
        self.rows = [(key, text) for key, texts in self.templates.items() for text in texts]
        self._index = BM25Okapi([tokenize(text) for _, text in self.rows])

    def recognize(self, message: str) -> TemplateIntentResult:
        # 此路只看当前消息，不带历史；“这个怎么办”没有稳定的字面分类证据。
        message = validate_message(message)
        started = time.perf_counter()
        values = self._index.get_scores(tokenize(message))
        scores = {key: 0.0 for key in self.templates}
        matches = {key: None for key in self.templates}
        for (key, text), value in zip(self.rows, values):
            # 每类取最高正分，不求和，避免非业务示例较多就因数量直接累积高分。
            # 但更多示例仍可能提高最大命中的机会，后续需评测数量偏差。
            # 未超过0的类别保留零分与空示例；无匹配不代表它一定是寒暄或范围外。
            if value > scores[key]:
                scores[key], matches[key] = float(value), text
        return TemplateIntentResult("bm25", "bm25", scores, matches,
                                    round((time.perf_counter() - started) * 1000, 2))
