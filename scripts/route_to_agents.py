"""预览、受控离线演示或真实识别 -> 路由 -> Agent准备。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from submission_assistant.agents import AgentRequest
from submission_assistant.graph import build_agent_graph
from submission_assistant.intent import BatchStatus, IntentJudgments, JevIntentRecognizer, JevSettings, build_requests
from submission_assistant.intent.embedding import EmbeddingIntentRecognizer, EmbeddingSettings
from submission_assistant.intent.ensemble import EnsembleIntentRecognizer, fuse_results
from submission_assistant.routing import INTENT_ROUTING, RoutingSettings


class ControlledRecognizer:
    """明确标记的离线输入，不伪造模型实测。"""
    def __init__(self, judgments):
        self.judgments = judgments

    async def recognize(self, message, history=None):
        return fuse_results(jev=self.judgments)


async def invoke(recognizer, message, history, settings, *, source_mode):
    req = AgentRequest(message, history or [], str(uuid.uuid4()))
    # 防止环境中已有的LangSmith开关把用户消息送往额外服务；不改全局环境变量。
    from langsmith import tracing_context
    with tracing_context(enabled=False):
        result = await build_agent_graph(recognizer, settings=settings).ainvoke({'request': req})
    return {'source_mode': source_mode, 'request': asdict(req),
            'recognition': result['recognition'], 'routing_decision': result['routing_decision'],
            'prepared_agents': result['prepared_agents']}


async def demo(settings):
    cases = [
        ('格式单业务', 'IEEE Access摘要最多多少词？', {'C02':.95}, .92),
        ('同角色多意图', '摘要最多多少词，首次投稿要交哪些材料？', {'C02':.95,'C03':.94}, .92),
        ('跨角色多意图', '摘要最多多少词，发表需要支付多少费用？', {'C02':.95,'C07':.94}, .92),
        ('非业务接待', '你好，你能帮助我做什么？', {'C10':.99}, .10),
        ('需求不清', '我的稿件出问题了，帮我看看。', {'C05':.91}, .10),
        ('技术失败', 'IEEE Access发表费用多少？', None, None),
    ]
    outputs = []
    for name, message, scores, clarity in cases:
        values = {f'C{i:02d}':.01 for i in range(1,11)} if scores is not None else {}
        values.update(scores or {})
        status = 'success' if scores is not None else 'failed'
        batches = [BatchStatus(f'batch_{i}', f'controlled-{i}', status, {},
                               None if status == 'success' else '受控模拟服务失败。') for i in (1,2)]
        judgments = IntentJudgments(values, clarity, 'controlled-not-model', 'controlled', batches)
        result = await invoke(ControlledRecognizer(judgments), message, [], settings, source_mode='controlled_judgments')
        outputs.append({'case': name, **result})
    return {'source_mode': 'controlled_demo_not_model_test', 'cases': outputs}


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='JEV与两路辅助 -> 确定性路由 -> Agent接收，不生成业务答案。真实模式产生服务费用。')
    parser.add_argument('--message')
    parser.add_argument('--history-file', type=Path)
    parser.add_argument('--env-file', type=Path, default=PROJECT_ROOT / '.env')
    parser.add_argument('--cache-dir', type=Path, default=PROJECT_ROOT / '.cache' / 'intent_embeddings')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--judgments-file', type=Path, help='已有十项独立判断JSON，不读.env、不联网')
    modes.add_argument('--dry-run', action='store_true', help='只展示两批请求和路由映射，不联网')
    modes.add_argument('--demo', action='store_true', help='六类受控离线演示，非模型测试')
    parser.add_argument('--jev-only', action='store_true', help='真实模式只测JEV，不调用Embedding；辅助缺失不阻止完整JEV路由')
    parser.add_argument('--clarity-threshold', type=float, default=.75)
    parser.add_argument('--relevance-threshold', type=float, default=.80)
    parser.add_argument('--output', type=Path, help='将本次生成的JSON报告另存到文件，拒绝覆盖已有文件')
    args = parser.parse_args()
    try:
        settings = RoutingSettings(args.clarity_threshold, args.relevance_threshold)
        if args.output and args.output.exists():
            raise ValueError('输出文件已存在；请选择新文件名，避免覆盖数据。')
        if args.demo:
            output = asyncio.run(demo(settings))
        else:
            if not args.message:
                raise ValueError('请提供--message，或使用--demo。')
            history = json.loads(args.history_file.read_text(encoding='utf-8-sig')) if args.history_file else []
            if args.dry_run:
                # 仅结构预览；真实调用由配置的 provider 在任何付费请求前校验。
                preview = build_requests(args.message, history, provider='decisions')
                output = {'source_mode': 'dry_run', 'jev_requests': preview,
                          'routing_settings': settings.to_dict(),
                          'intent_routing': {key:role.value for key,role in INTENT_ROUTING.items()},
                          'note': '清晰且高相关业务跨角色才需协作；本阶段仅准备Agent输入。'}
            else:
                if args.judgments_file:
                    judgments = IntentJudgments.from_dict(json.loads(args.judgments_file.read_text(encoding='utf-8-sig')))
                    recognizer, source = ControlledRecognizer(judgments), 'controlled_judgments'
                else:
                    jev = JevIntentRecognizer(JevSettings.from_env(args.env_file))
                    recognizer = jev if args.jev_only else EnsembleIntentRecognizer(jev,
                        EmbeddingIntentRecognizer(EmbeddingSettings.from_env(args.env_file), cache_dir=args.cache_dir))
                    source = 'live_jev_only' if args.jev_only else 'live_three_sources'
                output = asyncio.run(invoke(recognizer, args.message, history, settings, source_mode=source))
        rendered = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as handle:
                handle.write(rendered + '\n')
        print(rendered)
        # 正常通用兜底也是合法结果；technical_failure体现在fallback_reason，不混淆用户输入错误。
        return 0
    except (ValueError, OSError) as exc:
        # JSON解码异常的正文可能包含文件内容，不直接向终端打印。
        if isinstance(exc, json.JSONDecodeError):
            print('输入文件不是合法UTF-8 JSON。', file=sys.stderr)
        elif isinstance(exc, OSError):
            print('本地文件读写失败，请检查路径与权限。', file=sys.stderr)
        else:
            print(f'路由入口未启动：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
