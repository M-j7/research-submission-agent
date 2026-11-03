"""十项独立判断契约；不负责阈值、路由或生成答案。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .catalog import RETRIEVAL_OPTION_BY_ID

BATCH_QUESTIONS = (
    ('clarity', 'C01', 'C02', 'C03', 'C04', 'C05'),
    ('C06', 'C07', 'C08', 'C09', 'C10'),
)


def probability(value: Any) -> float:
    # 不接受布尔、百分数、字符串或NaN；不帮服务自动修正分数。
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError('判断概率必须是0–1范围的有限数值。')
    return float(value)


def known_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in ('input_tokens', 'output_tokens')
            if isinstance(value.get(key), int) and not isinstance(value[key], bool) and value[key] >= 0}


@dataclass(frozen=True)
class BatchStatus:
    batch_id: str
    request_id: str
    status: str
    usage: dict[str, int]
    error: str | None = None

    def __post_init__(self):
        if self.batch_id not in ('batch_1', 'batch_2') or self.status not in ('success', 'failed'):
            raise ValueError('非法JEV批次状态。')
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError('批次缺少请求ID。')
        if not isinstance(self.usage, dict) or known_usage(self.usage) != self.usage:
            raise ValueError('非法批次用量。')
        if self.status == 'success' and self.error is not None:
            raise ValueError('成功批次不能带失败原因。')
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError('批次错误必须是脱敏文字。')
        object.__setattr__(self, 'usage', dict(self.usage))

    def to_dict(self):
        return {'batch_id': self.batch_id, 'request_id': self.request_id,
                'status': self.status, 'usage': dict(self.usage), 'error': self.error}


@dataclass(frozen=True)
class IntentJudgments:
    probabilities: dict[str, float]
    clarity_probability: float | None
    model: str
    request_id: str
    batches: list[BatchStatus]
    latency_ms: float = 0.0

    def __post_init__(self):
        if not isinstance(self.probabilities, dict) or not set(self.probabilities) <= set(RETRIEVAL_OPTION_BY_ID):
            raise ValueError('独立判断包含未知意图。')
        values = {key: probability(value) for key, value in self.probabilities.items()}
        if self.clarity_probability is not None:
            probability(self.clarity_probability)
        if not isinstance(self.model, str) or not self.model.strip() or not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError('判断缺少模型或请求标识。')
        if not isinstance(self.batches, list) or any(not isinstance(b, BatchStatus) for b in self.batches):
            raise ValueError('判断缺少批次状态。')
        if len(self.batches) != 2 or [b.batch_id for b in self.batches] != ['batch_1', 'batch_2']:
            raise ValueError('判断必须包含按序的两个批次状态。')
        if len({b.request_id for b in self.batches}) != 2:
            raise ValueError('两个批次必须使用不同请求ID。')
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int, float)) or not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError('非法判断耗时。')
        # 全成功时必须完整；失败时允许保留成功批次的原分，但不能补零。
        if all(b.status == 'success' for b in self.batches) and (
            set(values) != set(RETRIEVAL_OPTION_BY_ID) or self.clarity_probability is None
        ):
            raise ValueError('成功判断必须完整包含十项相关性和清晰度。')
        object.__setattr__(self, 'probabilities', values)
        object.__setattr__(self, 'batches', list(self.batches))

    @property
    def complete(self) -> bool:
        return (len(self.probabilities) == 10 and self.clarity_probability is not None
                and all(b.status == 'success' for b in self.batches))

    def to_dict(self) -> dict:
        # usage是已知用量之和。失败/漏usage不能假定免费，usage_complete显式区分。
        usage = {key: sum(b.usage.get(key, 0) for b in self.batches)
                 for key in ('input_tokens', 'output_tokens') if any(key in b.usage for b in self.batches)}
        return {'score_type': 'independent_relevance_probability', 'complete': self.complete,
                'probabilities': dict(self.probabilities), 'clarity_probability': self.clarity_probability,
                'model': self.model, 'request_id': self.request_id, 'latency_ms': self.latency_ms,
                'batches': [b.to_dict() for b in self.batches], 'usage': usage,
                'usage_complete': all(set(b.usage) == {'input_tokens', 'output_tokens'} for b in self.batches)}

    @classmethod
    def from_dict(cls, value: dict) -> IntentJudgments:
        """离线CLI使用同一契约；不把用户提供的complete标志当作事实。"""
        try:
            if value['score_type'] != 'independent_relevance_probability':
                raise ValueError
            batches = [BatchStatus(b['batch_id'], b['request_id'], b['status'], b['usage'], b.get('error'))
                       for b in value['batches']]
            return cls(value['probabilities'], value['clarity_probability'], value['model'],
                       value['request_id'], batches, value.get('latency_ms', 0.0))
        except (KeyError, TypeError, AttributeError, ValueError):
            raise ValueError('判断文件不符合十项独立JEV结果契约。') from None
