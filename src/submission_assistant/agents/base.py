"""共享Agent骨架；本阶段只准备输入，不执行模型或工具。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..routing import AgentType, RoutingDecision


@dataclass(frozen=True)
class AgentRequest:
    message: str
    history: list[dict[str, str]]
    request_id: str

    def __post_init__(self):
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError('Agent当前消息必须是非空字符串。')
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError('Agent输入缺少请求ID。')
        if not isinstance(self.history, list) or any(
            not isinstance(item, dict) or item.get('role') not in ('user','assistant')
            or not isinstance(item.get('content'), str) for item in self.history
        ):
            raise ValueError('Agent历史必须是user/assistant的role/content列表。')
        object.__setattr__(self, 'history', [{'role': x['role'], 'content': x['content']} for x in self.history])


@dataclass(frozen=True)
class AgentProfile:
    agent_type: AgentType
    system_prompt: str
    allowed_tool_names: tuple[str, ...]


FALLBACK_GUIDANCE = {
    'non_business': '根据当前消息进行自然接待；对能力咨询说明范围，对明确范围外请求说明边界，不因低清晰度强行追问。',
    'unclear_request': '澄清用户具体想解决的事项和缺失指代；不猜业务答案，一次优先确认关键缺失信息。',
    'uncertain_intent': '系统暂未可靠确定该由哪类专业服务处理；不等于用户没说清楚。结合历史简短确认本轮具体事项，不给专业答案。',
    'conflicting_judgments': '业务与非业务判断冲突；结合本轮消息澄清，不让用户承担模型分类术语。',
    'recognition_failed': '意图服务出现技术失败；说明服务状态，不宣称用户没有说清楚，不假称查过知识库。',
}


class BaseAgent:
    profile: AgentProfile

    def prepare(self, req: AgentRequest, decision: RoutingDecision,
                tool_registry: Mapping[str, object] | None = None) -> dict:
        role = self.profile.agent_type
        if role not in decision.agent_types:
            raise ValueError('不能为本轮未选中的Agent准备业务输入。')
        registry = tool_registry or {}
        # 允许范围和实际可用工具是两回事。不存在的检索工具只能标未绑定。
        # 这个过滤是程序约束，不是仅在prompt里写“不要越权”。
        registered = [name for name in self.profile.allowed_tool_names
                      if name in registry and registry[name] is not None]
        unbound = [name for name in self.profile.allowed_tool_names if name not in registered]
        guidance = FALLBACK_GUIDANCE.get(decision.fallback_reason, '') if role == AgentType.GENERAL else (
            '只负责intent_ids对应事项；其他角色负责的诉求不重复作答。'
            '需要证据时使用实际已注册且允许的工具；输入准备本身不执行回答，后续由统一循环处理。')
        return {'agent_type': role.value, 'system_prompt': self.profile.system_prompt,
                'input': {'request_id': req.request_id, 'message': req.message,
                          'history': [{'role': x['role'], 'content': x['content']} for x in req.history],
                          'intent_ids': list(decision.agent_intents.get(role.value, [])),
                          'fallback_reason': decision.fallback_reason, 'guidance': guidance},
                'allowed_tool_names': list(self.profile.allowed_tool_names),
                'registered_tools': registered, 'unbound_tools': unbound,
                'status': 'input_prepared', 'executed': False}
