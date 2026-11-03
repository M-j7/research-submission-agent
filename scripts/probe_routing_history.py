"""Small fixed live diagnostic, not a replacement for the frozen Agent benchmark.

Eight routing-only calls (two batches each), then two feedback-only generations.
No retries, RAG, production memory, index or Redis access. Outputs never overwrite.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src')]
import httpx
from dotenv import dotenv_values
from langsmith import tracing_context
from submission_assistant.agents import AgentRequest
from submission_assistant.graph import build_agent_graph
from submission_assistant.intent.jev import JevIntentRecognizer, JevSettings, JevServiceError
from submission_assistant.runtime.chat import ChatService, build_chat_graph
from submission_assistant.runtime.contracts import Request
from submission_assistant.runtime.memory import MemoryManager
from submission_assistant.runtime.provider import RoleModelRouter
from submission_assistant.runtime.loop import LoopSettings
from evaluation.agent.observe import ObservationSink, ObservedModelRouter, PaidCallGate
from evaluation.agent.runtime import decisions_values


def cases():
    old = {r['case_id']: r for r in (
        json.loads(s) for s in (PROJECT / 'evaluation/agent/runs/20261020-context-fix-full-01/executions.jsonl')
            .read_text(encoding='utf-8').splitlines() if s.strip())}
    rows = []
    for key in ('conversation-01', 'conversation-05', 'multi-04', 'multi-05'):
        turns = old[key]['turns']
        history = [m for t in turns[:-1] for m in (
            {'role': 'user', 'content': t['message']}, {'role': 'assistant', 'content': t['answer']})]
        rows.append({'id': key, 'message': turns[-1]['message'], 'history': history,
            'expected_roles': ['submission_requirements'] if key.startswith('conversation') else
                ['editorial_workflow', 'publication_policy'] if key == 'multi-04' else
                ['submission_requirements', 'publication_fees'],
            'previous_route': turns[-1]['route']})
    five = [
        ('IEEE Access投稿补充视频最大文件大小是多少？', '视频作为投稿附件，具体限额须查证。'),
        ('谢谢，摘要长度问题已经确认了。', '好的。'),
        ('投稿信已经准备完了。', '好的。'),
        ('作者简介我也处理好了。', '好的。'),
        ('今天先不讨论费用。', '好的。')]
    for identifier, padded in [('five-round-reference', False), ('long-five-round-reference', True)]:
        history = [m for q, a in five for m in (
            {'role': 'user', 'content': q},
            {'role': 'assistant', 'content': a + ('\n历史回答的资料摘录，不是新诉求。' * 150 if padded else '')})]
        rows.append({'id': identifier, 'message': '回到刚才的补充视频，它的大小限制呢？',
                     'history': history, 'expected_roles': ['submission_requirements']})
    rows.extend([
        {'id': 'vague-control', 'message': '这个怎么办？', 'history': [], 'expected_roles': ['general'],
         'expected_fallback': 'unclear_request'},
        {'id': 'nonbusiness-control', 'message': '你好，今天的天气怎么样？', 'history': [],
         'expected_roles': ['general'], 'expected_fallback': 'non_business'}])
    return rows


async def run(home):
    home.mkdir(parents=True, exist_ok=False)
    values = dict(dotenv_values(PROJECT / '.env', encoding='utf-8-sig', interpolate=False))
    # Same explicit native provider/model as the frozen full evaluation. Some
    # local .env files retain an old provider label; never rewrite user config.
    settings = JevSettings.from_values(decisions_values(values))
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'complete': False,
        'scope': '8 routing-only probes + 2 General feedback generations; no retry, no RAG',
        'provider': settings.provider, 'model': settings.model, 'batch_sizes': [6, 5],
        'configuration_source': 'same native Decisions overrides as prior full evaluation; credentials from local .env, file unchanged',
        'thresholds': {'clarity': .75, 'relevance': .80}, 'expected_route_samples': 8,
        'expected_feedback_samples': 2, 'caveat': 'Targeted agent-draft diagnostics, not overall accuracy.'}
    def save(name, data):
        (home / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    sink, gate = ObservationSink('routing-history-probes'), PaidCallGate()
    def save_telemetry():
        save('telemetry.json', {'events': sink.events, 'generation_usage': sink.totals(),
                               'all_http_usage': sink.totals('http_result')})
    save('manifest.json', manifest)
    rows, feedback = [], []
    async with httpx.AsyncClient(trust_env=settings.trust_env, follow_redirects=False,
                                event_hooks=sink.http_hooks(gate)) as client:
        recognizer = JevIntentRecognizer(settings, client=client)
        graph = build_agent_graph(recognizer)
        for case in cases():
            sink.turn_id = case['id']
            out = await graph.ainvoke({'request': AgentRequest(case['message'], case['history'], case['id'])})
            route = out['routing_decision']
            row = {**case, 'recognition': out['recognition'], 'route': route,
                'matches_expected_roles': set(route['agent_types']) == set(case['expected_roles']),
                'matches_expected_fallback': ('expected_fallback' not in case or
                    route['fallback_reason'] == case['expected_fallback'])}
            rows.append(row)
            save('routing.json', rows)
            save_telemetry()
            print(case['id'], route['agent_types'], route['fallback_reason'], flush=True)
            # Authentication/balance errors must not fan out into more paid calls.
            errors = json.dumps(out['recognition'].get('source_errors', {}))
            if gate.blocked or any(f'HTTP {status}' in errors for status in (401, 402, 403, 429)):
                manifest['stopped_reason'] = 'credential_or_rate_limit'
                save('manifest.json', manifest)
                return
        vague = next(r for r in rows if r['id'] == 'vague-control')
        class Captured:
            async def recognize(self, message, history=None):
                return vague['recognition']
        class Unavailable:
            async def recognize(self, message, history=None):
                raise JevServiceError('意图判断服务暂不可用。')
        router = ObservedModelRouter(RoleModelRouter(values, client=client), sink)
        for identifier, message, source in (
            ('vague-feedback', '这个怎么办？', Captured()),
            ('technical-feedback', 'IEEE Access投稿需要哪些材料？', Unavailable()),
        ):
            if identifier == 'vague-feedback' and vague['route']['agent_types'] != ['general']:
                feedback.append({'id': identifier, 'skipped_reason': 'control_did_not_route_to_general'})
                save('feedback.json', feedback)
                continue
            sink.turn_id = identifier
            memory = MemoryManager(home / f'{identifier}.sqlite3')
            # One generation per feedback check, even if a provider ignores tool_choice=none.
            service = ChatService(home / 'isolated-workspace', memory, source, router,
                                  settings=LoopSettings(max_steps=1))
            try:
                req = Request(user_id='probe', conv_id=identifier, turn_id='t1', message=message)
                out = await build_chat_graph(service).ainvoke({'request': req})
                feedback.append({'id': identifier, 'message': message, 'answer': out['answer'],
                    'route': out['routing_decision'], 'result': out['final_result']})
                save('feedback.json', feedback)
                save_telemetry()
                print(identifier, out['final_result']['execution_status'], flush=True)
            finally:
                await service.close()
                memory.close()
            if gate.blocked:
                manifest['stopped_reason'] = 'credential_or_rate_limit'
                save('manifest.json', manifest)
                return
    manifest.update(complete=len(rows) == 8 and len(feedback) == 2 and not any('skipped_reason' in r for r in feedback),
        finished_at=datetime.now(timezone.utc).isoformat(),
        route_matches=sum(r['matches_expected_roles'] and r['matches_expected_fallback'] for r in rows))
    save('manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Explicitly enable the fixed paid diagnostic')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.live:
        parser.error('--live is required; diagnostic uses paid Decisions and Qwen calls')
    with tracing_context(enabled=False):
        asyncio.run(run(args.output))
