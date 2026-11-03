"""确定性路由：JEV判断业务，程序按职责映射；不再次询问LLM。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .intent.contracts import IntentJudgments, probability


class AgentType(str, Enum):
    GENERAL = 'general'
    SUBMISSION_REQUIREMENTS = 'submission_requirements'
    EDITORIAL_WORKFLOW = 'editorial_workflow'
    PUBLICATION_POLICY = 'publication_policy'
    PUBLICATION_FEES = 'publication_fees'


INTENT_ROUTING = {
    'C01': AgentType.SUBMISSION_REQUIREMENTS, 'C02': AgentType.SUBMISSION_REQUIREMENTS,
    'C03': AgentType.SUBMISSION_REQUIREMENTS,
    'C04': AgentType.PUBLICATION_POLICY, 'C08': AgentType.PUBLICATION_POLICY,
    'C05': AgentType.EDITORIAL_WORKFLOW, 'C06': AgentType.EDITORIAL_WORKFLOW,
    'C09': AgentType.EDITORIAL_WORKFLOW, 'C07': AgentType.PUBLICATION_FEES,
}


@dataclass(frozen=True)
class RoutingSettings:
    # 只是未校准的实验基线，不是模型准确率，也不是业内统一门槛。
    clarity_threshold: float = .75
    relevance_threshold: float = .80

    def __post_init__(self):
        probability(self.clarity_threshold)
        probability(self.relevance_threshold)

    def to_dict(self):
        return {'clarity_threshold': self.clarity_threshold, 'relevance_threshold': self.relevance_threshold}


@dataclass(frozen=True)
class RoutingDecision:
    agent_types: list[AgentType]
    agent_intents: dict[str, list[str]]
    matched_intents: list[str]
    mode: str
    multi_agent: bool
    fallback_reason: str | None
    collaboration_reason: str | None
    settings: dict

    def to_dict(self):
        return {'agent_types': [a.value for a in self.agent_types],
                'agent_intents': {a: list(ids) for a, ids in self.agent_intents.items()},
                'matched_intents': list(self.matched_intents), 'mode': self.mode,
                'multi_agent': self.multi_agent, 'fallback_reason': self.fallback_reason,
                'collaboration_reason': self.collaboration_reason, 'settings': dict(self.settings)}


def route_intents(judgments: IntentJudgments | None, settings: RoutingSettings | None = None) -> RoutingDecision:
    settings = settings or RoutingSettings()
    def fallback(reason, matched=None):
        # 不清楚与服务失败要分开：前者澄清用户，后者说明技术状态。
        return RoutingDecision([AgentType.GENERAL], {'general': []}, list(matched or []),
                               'fallback', False, reason, None, settings.to_dict())
    if not isinstance(judgments, IntentJudgments):
        return fallback('recognition_failed')
    try:
        # dataclass frozen并不冻结内部dict；边界重验，防止被调用方修改后误路由。
        checked = IntentJudgments.from_dict(judgments.to_dict())
    except ValueError:
        return fallback('recognition_failed')
    if not checked.complete:
        return fallback('recognition_failed')
    matched = sorted(key for key in INTENT_ROUTING if checked.probabilities[key] >= settings.relevance_threshold)
    non_business = checked.probabilities['C10'] >= settings.relevance_threshold
    # 先处理非业务，否则“你好”的低clarity会导致莫名其妙追问。
    if non_business and not matched:
        return fallback('non_business')
    if non_business and matched:
        return fallback('conflicting_judgments', matched)
    if checked.clarity_probability < settings.clarity_threshold:
        return fallback('unclear_request', matched)
    if not matched:
        return fallback('uncertain_intent')
    owners = {role.value: [key for key in matched if INTENT_ROUTING[key] == role]
              for role in AgentType if role != AgentType.GENERAL}
    owners = {role: ids for role, ids in owners.items() if ids}
    agents = [AgentType(role) for role in owners]
    # 多意图≠多Agent。只有清晰诉求跨越去重后的专业角色才要协作。
    multi = len(agents) > 1
    return RoutingDecision(agents, owners, matched, 'multiple' if multi else 'single', multi, None,
                           'clear_business_intents_span_roles' if multi else None, settings.to_dict())
