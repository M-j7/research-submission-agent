"""JEV主判、两路辅助；加权分只作为诊断，路由另有独立程序。"""

# 当前实现：JEV独立判断作为主判；Embedding/BM25输出辅助命中与关系。
# JEV 理解当前消息与历史；另两路用本轮消息匹配示例，不能补出缺失的指代。
# 加权九类分仅是诊断基线；新路由只读取完整judgments，不读取scores或selected_intent。

from __future__ import annotations

import asyncio
import math
import time

from .bm25 import BM25IntentRecognizer
from .catalog import OPTION_BY_ID, NON_BUSINESS_ID, RETRIEVAL_OPTION_BY_ID
from .embedding import (EmbeddingIntentRecognizer, EmbeddingProtocolError, EmbeddingServiceError)
from .jev import (IntentInputError, IntentResult, JevIntentRecognizer,
                  JevProtocolError, JevServiceError, build_request)
from .templates import TemplateIntentResult, load_retrieval_templates
from .contracts import IntentJudgments

# 权重是约定的实验起点，不代表已评测出最优值，也不保证辅助分不会改变业务排名。
WEIGHTS = {"jev": 0.7, "embedding": 0.2, "bm25": 0.1}


def _positive_shares(scores: dict[str, float]) -> dict[str, float]:
    # BM25 分、余弦相似度与 JEV 概率的尺度不同，不能直接相加。
    # 本基线先保留正分，再求每类在本路中的占比；全零则不产生贡献。
    # 例：[2, 6, 0] → [0.25, 0.75, 0]。这种占比只是相对分，不是正确概率。
    if set(scores) != set(OPTION_BY_ID):
        raise ValueError("融合输入必须完整包含 C01–C09 的分数。")
    positive = {}
    for key, score in scores.items():
        try:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError
            positive[key] = max(0.0, float(score))
        except (ValueError, OverflowError):
            raise ValueError("融合分数必须是有限数值。") from None
    # Scale by maximum first to avoid overflow when adding finite large scores.
    largest = max(positive.values())
    if largest == 0:
        return positive
    scaled = {key: value / largest for key, value in positive.items()}
    total = math.fsum(scaled.values())
    return {key: value / total for key, value in scaled.items()}


def fuse_results(*, jev: IntentResult | IntentJudgments | None = None,
                 embedding: TemplateIntentResult | None = None,
                 bm25: TemplateIntentResult | None = None,
                 source_errors: dict[str, str] | None = None, latency_ms: float = 0.0) -> dict:
    # sources 保留原分、命中示例与用量，normalized / contributions 解释融合分怎么来。
    # 因此后续看到排名异常时，可以回看是哪路、哪个示例产生了贡献。
    results = {"jev": jev, "embedding": embedding, "bm25": bm25}
    zero = {key: 0.0 for key in OPTION_BY_ID}
    normalized = {}
    sources = {}
    non_business_evidence = {}
    independent = isinstance(jev, IntentJudgments)
    # primary_candidates是展示用候选（默认.80），不是路由决定。
    primary_candidates = ([key for key in OPTION_BY_ID if jev.probabilities.get(key, 0) >= .8]
                          if independent and jev.complete else [])
    for source, result in results.items():
        if result is None:
            normalized[source] = dict(zero)
            continue
        values = result.probabilities if source == "jev" else result.scores
        if source == 'jev' and independent:
            # 不完整批次不产生JEV诊断贡献；原始partial分仍保留在sources供排查。
            values = {key: result.probabilities[key] for key in OPTION_BY_ID} if result.complete else dict(zero)
        if source != "jev" and set(values) == set(RETRIEVAL_OPTION_BY_ID):
            # 两路提供十类检索结果，C10另保留原始证据。
            # 九业务类的旧公式仅供诊断，不拿它比较业务/非业务可信度。
            score = values[NON_BUSINESS_ID]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("非业务检索分必须是有限数值。")
            non_business_evidence[source] = {
                "score": score, "score_type": result.score_type,
                "example": result.matched_examples.get(NON_BUSINESS_ID),
                "note": "仅为对照示例匹配证据，不是概率或自动路由决定。",
            }
            values = {key: values[key] for key in OPTION_BY_ID}
        normalized[source] = _positive_shares(values)
        sources[source] = result.to_dict()
    contributions = {source: {key: WEIGHTS[source] * share for key, share in shares.items()}
                     for source, shares in normalized.items()}
    scores = {key: math.fsum(contributions[source][key] for source in WEIGHTS) for key in OPTION_BY_ID}
    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    evidence_top_intent = ranked[0] if scores[ranked[0]] > 0 else None
    # 检索匹配不能制造诉求或覆盖主判；旧dialogue只供历史离线结果兼容。
    if independent:
        applicable = (bool(primary_candidates) and jev.probabilities['C10'] < .8) if jev.complete else None
        selected = max(primary_candidates, key=lambda key: jev.probabilities[key]) if applicable else None
    else:
        applicable = jev.business_intent_applicable if jev is not None else None
        selected = jev.selected_intent if applicable is True else None
    # 辅助路线仅解释匹配。无匹配不算反对；并列不强行用字典排序选一个。
    primary_set = set(primary_candidates if independent else ([selected] if selected else []))
    auxiliary_evidence, review_flags = {}, []
    for source, result in (('embedding', embedding), ('bm25', bm25)):
        if result is None:
            auxiliary_evidence[source] = {'relation': 'unavailable', 'top_intents': [], 'top_matches': []}
            continue
        # values此前已作完整性/有限值验证；检索含C10，匹配排行也必须含C10。
        positive = {key: float(value) for key, value in result.scores.items() if value > 0}
        highest = max(positive.values(), default=0)
        top = sorted(key for key, value in positive.items() if math.isclose(value, highest, rel_tol=1e-9, abs_tol=1e-12))
        relation = ('no_match' if not top else 'tied_candidates' if len(top) > 1 else
                    'supports_primary' if top[0] in primary_set else 'alternative_candidate' if primary_set else 'candidate_only')
        auxiliary_evidence[source] = {'relation': relation, 'top_intents': top,
            'score_type': result.score_type,
            'top_matches': [{'intent': key, 'score': result.scores[key], 'example': result.matched_examples.get(key)}
                            for key in sorted(positive, key=lambda key: (-positive[key], key))[:3]]}
        if relation == 'alternative_candidate':
            review_flags.append(f'{source}_alternative_candidate')
    if evidence_top_intent is not None and primary_set and evidence_top_intent not in primary_set:
        review_flags.append('evidence_ranking_differs_from_primary')
    missing = [source for source, result in results.items() if result is None]
    source_errors = dict(source_errors or {})
    if independent and not jev.complete:
        source_errors['jev'] = '；'.join(b.error or 'JEV判断未完成。' for b in jev.batches if b.status == 'failed')
    return {
        "selected_intent": selected,
        "selected_name": OPTION_BY_ID[selected].name if selected else None,
        "dialogue_type": jev.dialogue_type if jev is not None and not independent else None,
        "dialogue_probabilities": dict(jev.dialogue_probabilities) if jev is not None and not independent else None,
        "dialogue_confidence": jev.dialogue_confidence if jev is not None and not independent else None,
        "business_intent_applicable": applicable,
        "score_type": "weighted_fusion_score_not_probability",
        "scores": scores,
        "ranked_options": [{"intent": key, "name": OPTION_BY_ID[key].name, "score": scores[key]} for key in ranked],
        "top_two_gap": scores[ranked[0]] - scores[ranked[1]],
        # BUSINESS 也可能需求不清晰；保留这个独立判断，暂不拍定自动澄清阈值。
        "clarity_probability": jev.clarity_probability if jev is not None else None,
        "unclear_probability": 1 - jev.clarity_probability if jev is not None and jev.clarity_probability is not None else None,
        "weights": dict(WEIGHTS), "normalized_scores": normalized,
        "contributions": contributions, "sources": sources,
        "non_business_evidence": non_business_evidence,
        'judgments': jev.to_dict() if independent else None,
        'primary_candidates': primary_candidates,
        'candidate_threshold_for_display': .8,
        'selection_source': 'jev' if selected else None, 'candidate_only': selected is None,
        'evidence_top_intent': evidence_top_intent,
        'auxiliary_evidence': auxiliary_evidence, 'review_flags': review_flags,
        'diagnostic_fusion': {'scores': scores, 'weights': dict(WEIGHTS), 'score_type': 'weighted_fusion_score_not_probability'},
        "missing_sources": missing, "source_errors": source_errors,
        # 一路失败，其贡献归零，但不将剩余权重扩大为1。
        # 例：只剩 BM25 时总贡献最多0.1。degraded 表示路径缺失，不是答案质量。
        "degraded": bool(missing) or (independent and not jev.complete), "score_mass": math.fsum(scores.values()),
        "latency_ms": latency_ms,
    }


class EnsembleIntentRecognizer:
    def __init__(self, jev: JevIntentRecognizer, embedding: EmbeddingIntentRecognizer,
                 *, bm25: BM25IntentRecognizer | None = None):
        self.jev = jev
        self.embedding = embedding
        self.bm25 = bm25 if bm25 is not None else BM25IntentRecognizer()
        # A caller cannot accidentally blend scores based on different catalogs.
        self._check_templates()

    def _check_templates(self):
        # 三路仍共用默认业务示例；两路检索还必须共用默认 C10 对照示例。
        # 自定义或过期快照不能静默混入；独立检索入口可以自行传入示例路径。
        current = load_retrieval_templates()
        if self.embedding.templates != current or self.bm25.templates != current:
            raise ValueError("三路意图模板必须一致；模板发生变化后请重新创建识别器。")

    async def recognize(self, message: str, history: list[dict[str, str]] | None = None) -> dict:
        # Validate JEV input limits before any paid call to either service.
        from .jev import build_requests
        build_requests(message, history, model=self.jev.settings.model, provider=self.jev.settings.provider,
                       prompt_version=getattr(self.jev, 'prompt_version', 'baseline'))
        self._check_templates()
        started = time.perf_counter()
        errors = {}
        try:
            lexical = self.bm25.recognize(message)
        except Exception:
            # 本地辅助也可能失败，不能因此阻断尚未调用的主判。
            # 不打印未知异常正文；CancelledError不属于Exception，照常传播。
            lexical = None
            errors['bm25'] = 'BM25辅助路径出现内部错误，未生成匹配分。'
        # BM25 本地执行；两个网络服务并发，减少串行等待。
        # return_exceptions=True 使某一路失败时还能保留另一路结果，随后显式标记降级。
        # history 只给 JEV；Embedding 与 BM25 当前并不具备历史消歧能力。
        responses = await asyncio.gather(self.jev.recognize(message, history),
                                         self.embedding.recognize(message), return_exceptions=True)
        results = {}
        for source, response in zip(("jev", "embedding"), responses):
            if isinstance(response, BaseException):
                # Cancellation is a control signal, not a degraded prediction.
                if isinstance(response, asyncio.CancelledError):
                    raise response
                safe_types = (JevServiceError, JevProtocolError, IntentInputError,
                              EmbeddingServiceError, EmbeddingProtocolError)
                # 仅公开本项目已经脱敏的已知异常；未知异常可能含地址、密钥等敏感信息。
                errors[source] = str(response) if isinstance(response, safe_types) else "该识别路径出现内部错误，未生成分数。"
                results[source] = None
            else:
                results[source] = response
        return fuse_results(jev=results["jev"], embedding=results["embedding"], bm25=lexical,
                            source_errors=errors, latency_ms=round((time.perf_counter() - started) * 1000, 2))
