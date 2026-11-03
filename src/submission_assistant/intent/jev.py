"""Native SystemOne independent noul client; no LLM JSON generation or RAG dependency."""

# 阅读顺序：build_requests组装6/5两批 → recognize并发POST → parse_independent_batch
# → IntentJudgments.to_dict输出给下游。这个模块只判断，不选择Agent、不生成答案。
# build_request/parse_response/IntentResult仅保留历史单选的离线兼容，不是线上路径。
# 注意：用户“没说清楚”是正常判断结果；配置错误、服务失败、响应异常是异常，不能混为一谈。

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

from .catalog import (CLARITY_CRITERIA, CLARITY_INSTRUCTIONS, DIALOGUE_CRITERIA,
                      DIALOGUE_INSTRUCTIONS, INSTRUCTIONS, OPTIONS, OPTION_BY_ID)
from .templates import load_templates
from .contracts import BATCH_QUESTIONS, BatchStatus, IntentJudgments, known_usage
from .catalog import RETRIEVAL_OPTION_BY_ID, RELEVANCE_INSTRUCTIONS
from .routing_prompts import build_relevance_question, ROUTING_PROMPT_VERSION


class IntentInputError(ValueError):
    """Input cannot be judged under the configured service limits."""


class JevConfigurationError(ValueError):
    """Invalid local configuration; no request was sent."""


class JevProtocolError(ValueError):
    """Response does not satisfy the dialogue, clarity and intent contract."""


class JevServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class JevSettings:
    endpoint: str
    api_key: str = field(repr=False)
    # repr=False 避免打印配置对象时泄露密钥；调用时仍正常使用该字段。
    model: str = "jev-latest"
    timeout_s: float = 30.0
    provider: str = "systemone"
    trust_env: bool = True

    def __post_init__(self):
        if self.provider not in ('systemone', 'decisions'):
            raise JevConfigurationError('JEV_PROVIDER 必须为 systemone 或 decisions。')
        if not isinstance(self.trust_env, bool):
            raise JevConfigurationError('JEV_TRUST_ENV 必须是布尔值。')
        try:
            parsed = urlsplit(self.endpoint.strip())
            # Accessing .port also rejects nonnumeric and out-of-range ports.
            port = parsed.port
            valid = (
                parsed.scheme == "https" and bool(parsed.hostname)
                and port != 0
                and not parsed.username and not parsed.password
                and not parsed.query and not parsed.fragment
                and parsed.path.rstrip("/") in ("", "/v1", "/v1/decisions" if self.provider == 'decisions' else "/v1/systemone")
                and (self.provider != 'decisions' or parsed.hostname == 'api.perplexity.ai')
            )
            if not valid:
                raise ValueError
            path = parsed.path.rstrip("/")
            suffix = 'decisions' if self.provider == 'decisions' else 'systemone'
            endpoint = f"https://{parsed.netloc}/v1/{suffix}"
            if path == "/v1/systemone":
                endpoint = self.endpoint.strip().rstrip("/")
        except (AttributeError, ValueError):
            raise JevConfigurationError("JEV_BASE_URL 必须是供应商有效 HTTPS 根地址、/v1 或原生端点；不允许 Router、内嵌凭证、查询参数。") from None
        if not isinstance(self.api_key, str) or not self.api_key.strip() or any(c.isspace() for c in self.api_key):
            raise JevConfigurationError("请在本地 .env 中配置有效的 JEV_API_KEY。")
        if not isinstance(self.model, str) or not self.model.strip():
            raise JevConfigurationError("JEV_MODEL 不能为空。")
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) or not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise JevConfigurationError("JEV_TIMEOUT_S 必须是有限正数。")
        object.__setattr__(self, "endpoint", endpoint)

    @classmethod
    def from_env(cls, env_file: Path | None = None, *, environ: Mapping[str, str] | None = None):
        # 本地 .env 提供基础配置，显式环境变量可覆盖；只读取，不改写全局环境或文件。
        values = {}
        if env_file is not None:
            if not env_file.is_file():
                raise JevConfigurationError("找不到项目 .env 文件。")
            # Local-only values; no interpolation into other secrets or global env writes.
            values.update(dotenv_values(env_file, encoding="utf-8-sig", interpolate=False))
        values.update(os.environ if environ is None else environ)
        return cls.from_values(values)

    @classmethod
    def from_values(cls, values: Mapping[str, str]):
        def setting(name, default=""):
            return (values.get(name) or "").strip() or default
        try:
            timeout = float(setting("JEV_TIMEOUT_S", "30"))
        except ValueError:
            raise JevConfigurationError("JEV_TIMEOUT_S 必须是有限正数。") from None
        provider = setting('JEV_PROVIDER', 'systemone')
        proxy = setting('JEV_TRUST_ENV', 'true').lower()
        if proxy not in ('true', 'false'):
            raise JevConfigurationError('JEV_TRUST_ENV 必须为 true 或 false。')
        model = 'pplx-decider-flash-v1.2' if provider == 'decisions' else 'jev-latest'
        return cls(setting("JEV_BASE_URL"), setting("JEV_API_KEY"), setting("JEV_MODEL", model), timeout,
                   provider=provider, trust_env=proxy == 'true')


def build_request(message: str, history: list[dict[str, str]] | None = None, *, model: str = "jev-latest") -> dict[str, Any]:
    """历史九选一请求构造器，仅供离线历史测试；在线入口使用build_requests。"""
    # 所有输入检查在联网前执行，避免为无效输入产生付费调用。
    if not isinstance(message, str) or not message.strip():
        raise IntentInputError("当前消息必须是非空字符串。")
    if history is not None and not isinstance(history, list):
        raise IntentInputError("history 必须是 role/content 消息列表。")
    recent = []
    # 六条“消息”不是六轮对话；history 用来解释“这个怎么办”等省略句。
    # 拷贝需要的字段而非修改原列表，防止一次请求污染另一请求的历史。
    for item in (history or [])[-6:]:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant") or not isinstance(item.get("content"), str):
            raise IntentInputError("历史只接受 user/assistant 角色和字符串 content。")
        recent.append({"role": item["role"], "content": item["content"]})
    state = {"domain": "IEEE Access 投稿咨询", "current_message": message.strip(), "history": recent}
    # state 是三个问题共用的判断材料；domain 只是背景，不能当作缺失指代的证据。
    examples = load_templates()
    # 模板共九类×五条；JEV 每类只附一条例子以控制输入长度。
    # 另外两路使用全部示例。例子帮助解释类别，但不能限定类别只能匹配这些问法。
    criteria = {option.id: f"{option.name}：{option.description} 典型示例：{examples[option.id][0]}" for option in OPTIONS}
    # Limits of the currently configured jevmodel.org gateway, not upstream TypeSafe.
    if len(json.dumps(state, ensure_ascii=False)) > 8000:
        raise IntentInputError("当前消息加最近历史超过 JEV 的 8000 字符 state 限制；请先缩减历史，不会静默截断问题。")
    questions = {
        # 对话性质：业务 / 社交 / 明确范围外 / 连性质都无法判断。
        # “我的稿件出问题了”属于业务，即使还不知道具体哪里出了问题。
        "dialogue": {"type": "choice", "instructions": DIALOGUE_INSTRUCTIONS, "criteria": dict(DIALOGUE_CRITERIA)},
        # 需求清晰度：知道用户究竟想问什么吗？不是“已经有证据能够回答了吗”。
        # 寒暄没有具体业务诉求，不应单凭这里的低分追问用户。
        "clarity": {"type": "noul", "instructions": CLARITY_INSTRUCTIONS, "criteria": dict(CLARITY_CRITERIA)},
        # 业务类别：九选一的竞争分布；非业务输入也会被迫选一个，后面必须检查适用性。
        # 多个类别都有分数，不等于获得了各类别彼此独立的多标签概率。
        "intent": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria},
    }
    for question in questions.values():
        if len(json.dumps(question["criteria"], ensure_ascii=False)) > 2000 or len(question["instructions"]) > 1800:
            raise IntentInputError("判断定义超过当前 JEV 服务的 criteria/instructions 限制。")
    return {"model": model, "state": state, "questions": questions}


def build_requests(message: str, history: list[dict[str, str]] | None = None, *,
                   model: str = 'jev-latest', provider: str = 'systemone',
                   prompt_version: str = 'baseline') -> list[dict]:
    # 在线入口不再复用旧九选一的六条消息窗口/网关限制。
    # few-shot v2 increased false positives in the fixed live comparison;
    # online defaults stay on baseline. Experiments must opt in explicitly.
    from .stable_routing_prompts import STABLE_ROUTING_PROMPT_VERSION, stable_questions
    if prompt_version not in ('baseline', ROUTING_PROMPT_VERSION, STABLE_ROUTING_PROMPT_VERSION):
        raise IntentInputError('未知路由提示版本；未发送请求。')
    if not isinstance(message, str) or not message.strip():
        raise IntentInputError('当前消息必须是非空字符串。')
    if history is not None and not isinstance(history, list):
        raise IntentInputError('history 必须是已完成问答的 role/content 列表。')
    history = [] if history is None else history
    if len(history) % 2 or any(
        not isinstance(item, dict) or item.get('role') != ('user' if i % 2 == 0 else 'assistant')
        or not isinstance(item.get('content'), str) or not item['content'].strip()
        for i, item in enumerate(history)
    ):
        raise IntentInputError('历史必须按 user/assistant 成对提供已完成问答，不接受工具轨迹或未完成消息。')
    state = {'domain': 'IEEE Access 投稿咨询', 'current_message': message.strip(),
             'history': [{'role': x['role'], 'content': x['content']} for x in history[-10:]]}
    rendered = json.dumps(state, ensure_ascii=False)
    if provider == 'systemone':
        if len(rendered) > 8000:
            raise IntentInputError('当前消息加最近五轮问答超过 SystemOne 网关的 8000 字符限制；未截断。')
    elif provider == 'decisions':
        # 本地保守字节保护，不声称等于供应商的 256K token 上限。
        if len(rendered.encode('utf-8')) > 256 * 1024:
            raise IntentInputError('路由 state 超过本地 256 KiB UTF-8 保护上限；未截断。')
    else:
        raise IntentInputError('未知意图服务 provider。')
    questions = {'clarity': {'type': 'noul', 'instructions': CLARITY_INSTRUCTIONS,
                             'criteria': dict(CLARITY_CRITERIA)}}
    examples = load_templates() if prompt_version == 'baseline' else None
    for key in RETRIEVAL_OPTION_BY_ID:
        if prompt_version == STABLE_ROUTING_PROMPT_VERSION:
            continue
        if prompt_version == ROUTING_PROMPT_VERSION:
            questions[key] = build_relevance_question(key)
            continue
        option = RETRIEVAL_OPTION_BY_ID[key]
        example = examples[key][0] if key in examples else '你好；谢谢；你能帮助我做什么；今天的天气如何。'
        positive = f'本轮包含{option.name}的实际诉求。范围与边界：{option.description} 示例：{example}'
        negative = '本轮不包含上述实际诉求；只是背景提及，或当前消息与历史不足以确认此类诉求。'
        if key == 'C10':
            positive = f'本轮只有明确的非业务交流或范围外请求，不包含投稿业务诉求。{option.description} 示例：{example}'
            negative = '本轮包含任何投稿业务诉求，或业务具体需求不清、指代不明、难以理解；含问候的业务问题也判否。'
        questions[key] = {'type': 'noul', 'instructions': RELEVANCE_INSTRUCTIONS,
                          'criteria': {'true': positive, 'false': negative}}
    if prompt_version == STABLE_ROUTING_PROMPT_VERSION:
        questions = stable_questions()
    for question in questions.values():
        if len(json.dumps(question['criteria'], ensure_ascii=False)) > 2000 or len(question['instructions']) > 1800:
            raise IntentInputError('独立判断定义超过JEV服务限制。')
    return [{'model': model, 'state': json.loads(json.dumps(state)),
             'questions': {key: questions[key] for key in keys}} for keys in BATCH_QUESTIONS]


def parse_independent_batch(data: Any, expected_keys: tuple[str, ...]) -> tuple[dict[str, float], str]:
    if not isinstance(data, dict) or not isinstance(data.get('answers'), dict) or set(data['answers']) != set(expected_keys):
        raise JevProtocolError('JEV批次必须完整且仅返回所请求的独立判断。')
    model = data.get('model')
    if not isinstance(model, str) or not model.strip():
        raise JevProtocolError('JEV批次缺少模型标识。')
    values = {}
    for key in expected_keys:
        answer = data['answers'][key]
        if not isinstance(answer, dict) or answer.get('type') != 'noul':
            raise JevProtocolError('JEV独立判断必须返回noul类型。')
        values[key] = _probability(answer.get('noul'))
    return values, model


@dataclass(frozen=True)
class IntentResult:
    """历史互斥单选结果，保留离线诊断兼容；不可用于新Agent路由。"""
    selected_intent: str  # 服务原始九选一结果；下游应读 to_dict()，不要直接据此路由。
    probabilities: dict[str, float]
    confidence: float
    clarity_probability: float
    model: str
    usage: dict[str, int]
    request_id: str
    latency_ms: float = 0.0
    # None 仅兼容旧的进程内直接构造；真实 HTTP 缺失 dialogue 会报错，不补默认值。
    dialogue_type: str | None = None
    dialogue_probabilities: dict[str, float] = field(default_factory=dict)
    dialogue_confidence: float | None = None

    @property
    def business_intent_applicable(self) -> bool | None:
        # True：主对话类型是业务；False：不是业务；None：没有性质判断。
        # 这只说明业务候选是否适用，不证明诉求清晰，也没有做置信度阈值判断。
        return None if self.dialogue_type is None else self.dialogue_type == "BUSINESS"

    def to_dict(self) -> dict[str, Any]:
        # 分数相同时优先展示服务实际选中的类别，避免显示顺序与 choice 不一致。
        ranked = sorted(self.probabilities, key=lambda key: (-self.probabilities[key], key != self.selected_intent))
        return {
            "request_id": self.request_id,
            # 只有明确 BUSINESS 才发布有效业务候选；性质未知也不能默认当业务。
            "selected_intent": self.selected_intent if self.business_intent_applicable is True else None,
            "selected_name": OPTION_BY_ID[self.selected_intent].name if self.business_intent_applicable is True else None,
            "selected_probability": self.probabilities[self.selected_intent] if self.business_intent_applicable is True else None,
            "business_candidate": self.selected_intent,
            # 原始九选一结果仍留在 business_candidate / probabilities 供诊断。
            # 例如“这个怎么办”可能给 C05 很高分，但性质不明时 selected_intent 仍为 null。
            "business_intent_applicable": self.business_intent_applicable,
            "dialogue_type": self.dialogue_type,
            "dialogue_probabilities": dict(self.dialogue_probabilities),
            "dialogue_confidence": self.dialogue_confidence,
            "confidence": self.confidence,
            # confidence 是服务的业务分类置信度，不等于所选类别概率或需求清晰度。
            "clarity_probability": self.clarity_probability,
            "unclear_probability": 1.0 - self.clarity_probability,
            "probabilities": dict(self.probabilities),
            "ranked_options": [{"intent": key, "name": OPTION_BY_ID[key].name, "probability": self.probabilities[key]} for key in ranked],
            "top_two_gap": self.probabilities[ranked[0]] - self.probabilities[ranked[1]],
            "model": self.model, "usage": dict(self.usage), "latency_ms": self.latency_ms,
        }


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1 or not math.isfinite(value):
        raise JevProtocolError("JEV 概率和置信度必须是 0–1 范围的有限数值。")
    return float(value)


def parse_response(data: Any, *, request_id: str = "", latency_ms: float = 0.0) -> IntentResult:
    # 校验三个独立问题的响应契约，不在这里纠正模型判断，也不自动决定是否追问。
    # 丢字段、非法概率等应暴露为协议异常，不能伪装成“用户意图不明”。
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise JevProtocolError("JEV 响应缺少 answers 对象。")
    dialogue = data["answers"].get("dialogue")
    if not isinstance(dialogue, dict) or dialogue.get("type") != "choice":
        raise JevProtocolError("JEV 响应缺少 dialogue choice 判断。")
    raw_dialogue = dialogue.get("probabilities")
    # 必须完整包含四类，避免服务漏返一类却被当成零分继续使用。
    if not isinstance(raw_dialogue, dict) or set(raw_dialogue) != set(DIALOGUE_CRITERIA):
        raise JevProtocolError("JEV 必须返回全部四个对话类型的概率。")
    dialogue_probabilities = {key: _probability(raw_dialogue[key]) for key in DIALOGUE_CRITERIA}
    if not math.isclose(sum(dialogue_probabilities.values()), 1.0, abs_tol=0.001):
        raise JevProtocolError("JEV 对话类型概率之和不约等于 1。")
    dialogue_type = dialogue.get("choice")
    if not isinstance(dialogue_type, str) or dialogue_type not in dialogue_probabilities or dialogue_probabilities[dialogue_type] + 1e-8 < max(dialogue_probabilities.values()):
        raise JevProtocolError("JEV dialogue choice 必须是最高概率的已知对话类型。")
    dialogue_confidence = _probability(dialogue.get("confidence"))
    clarity = data["answers"].get("clarity")
    if not isinstance(clarity, dict) or clarity.get("type") != "noul":
        raise JevProtocolError("JEV 响应缺少 clarity noul 判断。")
    clarity_probability = _probability(clarity.get("noul"))
    # noul 返回“是”的概率，即需求清晰概率；不可从业务分类的分差推导它。
    answer = data["answers"].get("intent")
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise JevProtocolError("JEV 响应缺少 intent choice 判断。")
    raw = answer.get("probabilities")
    if not isinstance(raw, dict) or set(raw) != set(OPTION_BY_ID):
        raise JevProtocolError("JEV 必须返回全部九个已知业务选项的概率。")
    probabilities = {option.id: _probability(raw[option.id]) for option in OPTIONS}
    # choice 的九项分数应约等于1；只允许小数舍入误差，不替服务做归一化。
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.001):
        raise JevProtocolError("JEV 选项概率之和不约等于 1；不会自动归一化或猜测百分数。")
    selected = answer.get("choice")
    if not isinstance(selected, str) or selected not in probabilities or probabilities[selected] + 1e-8 < max(probabilities.values()):
        raise JevProtocolError("JEV choice 必须是最高概率的已知选项。")
    confidence = _probability(answer.get("confidence"))
    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        raise JevProtocolError("JEV 响应缺少模型标识。")
    raw_usage = data.get("usage", {})
    usage = {}
    if isinstance(raw_usage, dict):
        for key in ("input_tokens", "output_tokens"):
            value = raw_usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[key] = value
    return IntentResult(
        selected_intent=selected, probabilities=probabilities, confidence=confidence,
        clarity_probability=clarity_probability, model=model, usage=usage,
        request_id=request_id, latency_ms=latency_ms,
        dialogue_type=dialogue_type, dialogue_probabilities=dialogue_probabilities,
        dialogue_confidence=dialogue_confidence,
    )


class JevIntentRecognizer:
    def __init__(self, settings: JevSettings, *, client: httpx.AsyncClient | None = None,
                 prompt_version: str = 'baseline'):
        self.settings = settings
        self._client = client
        self.prompt_version = prompt_version

    async def recognize(self, message: str, history: list[dict[str, str]] | None = None) -> IntentJudgments:
        # 十类相关性+clarity分成6/5两批，两个请求共享相同判断材料。
        # 每批有独立请求ID；不重试；失败保留状态，绝不用辅助分补成成功。
        bodies = build_requests(message, history, model=self.settings.model, provider=self.settings.provider,
                                prompt_version=self.prompt_version)
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        if self._client is not None:
            results = await self._run_batches(self._client, bodies)
        else:
            async with httpx.AsyncClient(follow_redirects=False, trust_env=self.settings.trust_env) as client:
                results = await self._run_batches(client, bodies)
        values, batches, models = {}, [], []
        for batch, scores, model in results:
            batches.append(batch)
            values.update(scores)
            if model:
                models.append(model)
        if len(set(models)) > 1:
            # 不能将两个不同模型的批次伪装成一次完整判断；保留已知用量。
            batches[1] = BatchStatus('batch_2', batches[1].request_id, 'failed', batches[1].usage,
                                     'JEV两个批次返回不同模型，判断未完成。')
        clarity = values.pop('clarity', None)
        return IntentJudgments(values, clarity, models[0] if models else self.settings.model,
                               request_id, batches, round((time.perf_counter() - started) * 1000, 2))

    async def _run_batches(self, client, bodies):
        async def run(index, body):
            request_id = str(uuid.uuid4())
            data = None
            try:
                data = await self._post(client, body, request_id)
                values, model = parse_independent_batch(data, BATCH_QUESTIONS[index])
                status = BatchStatus(f'batch_{index+1}', request_id, 'success', known_usage(data.get('usage')))
                return status, values, model
            except (JevServiceError, JevProtocolError) as exc:
                usage = known_usage(data.get('usage')) if isinstance(data, dict) else {}
                return BatchStatus(f'batch_{index+1}', request_id, 'failed', usage, str(exc)), {}, None
        # 外部取消会取消两个子任务；不把取消记成一次业务判断失败。
        return await asyncio.gather(*(run(index, body) for index, body in enumerate(bodies)))

    async def _post(self, client: httpx.AsyncClient, body: dict[str, Any], request_id: str):
        # 不自动重试：失败是否被服务计费未必可知，也不能假定重复调用没有成本。
        # 出错只返回经过筛选的信息，不把服务响应正文或认证头打印出来。
        try:
            response = await client.post(
                self.settings.endpoint, json=body,
                headers={"Authorization": f"Bearer {self.settings.api_key}", "Idempotency-Key": request_id},
                timeout=self.settings.timeout_s, follow_redirects=False,
            )
        except httpx.TimeoutException:
            raise JevServiceError("JEV 请求超时；未自动重试，未生成意图。") from None
        except httpx.RequestError:
            raise JevServiceError("无法连接 JEV 服务；请检查网络、代理和服务地址。") from None
        if response.status_code != 200:
            if response.extensions.get('evaluation_billing_error'):
                raise JevServiceError('供应商额度不足；暂停付费调用。', 402)
            hints = {401: "密钥无效或未授权", 402: "余额不足", 422: "请求未通过服务校验", 429: "请求限流", 502: "上游模型失败"}
            hint = hints.get(response.status_code, "服务未返回成功响应")
            raise JevServiceError(f"JEV HTTP {response.status_code}：{hint}；未自动重试。", response.status_code)
        try:
            return response.json()
        except ValueError:
            raise JevProtocolError("JEV 未返回有效 JSON。") from None
