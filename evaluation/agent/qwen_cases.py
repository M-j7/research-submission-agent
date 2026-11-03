"""Four bounded Qwen requests produce candidates, never automatically gold labels."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
import httpx
from dotenv import dotenv_values
from submission_assistant.intent.jev import build_requests
from submission_assistant.runtime.provider import OpenAICompatibleModel
from submission_assistant.routing import INTENT_ROUTING
from .observe import ObservationSink, PaidCallGate
from .prompt_probe import write_json

# An independent task taxonomy, not the experimental boundary text, examples,
# previous data, failures, scores or candidate rankings.
TAXONOMY = {
    'C01': '期刊征稿范围、研究适配与选刊', 'C02': '论文写作、结构、格式与排版',
    'C03': '投稿需要准备的材料、附件与作者信息的要求', 'C04': '伦理合规、作者资格与署名责任',
    'C05': '投稿平台的具体操作与技术故障', 'C06': '审稿流程、状态及普通返修',
    'C07': '发表费用、付款、优惠、退款', 'C08': '录用后出版、版权许可及发表后更正撤回',
    'C09': '未发表稿件撤回、转投、拒稿重投与申诉',
}
BRIEFS = {
    'single': '生成10条自然的单意图问题：C01至C09各1条，另1条明确非业务交流。全部history为空。',
    'boundary': '生成10条相邻类别的最小对照问法，5对、每对2条，改变诉求而非只换同义词；各题只含1个实际业务意图，history为空。覆盖写作/材料、材料/平台、审稿/申诉、伦理/版权、费用/平台等交界。',
    'multi': '生成10条自然多意图问题，每条2至3个实际业务类别，覆盖不同角色组合及同角色多类别。背景或被明确否定的事项不能算意图。history为空。不要固定模板。',
    'history': '生成10条：前6条为必须依赖历史才能理解的省略追问（history_kind=dependent，缺历史应澄清），包括两条相同当前问法但不同历史/意图；接着2条有历史但本轮明确换话题或更正（context_or_distractor）；最后2条无历史控制题，分别为明确非业务、无法理解指代而须澄清。历史1至3轮，助手仅确认或澄清，不编造期刊政策。',
}
SYSTEM = '''你是中文投稿咨询助手的测试用例设计者。请独立设计自然、多样的合成问题，只出题和候选分类，不回答业务问题。不要编造具体期刊政策、事实或费用数字作为判定依据，不使用真人个人信息。类别可多选，判断本轮实际诉求；历史只用于解释指代和更正，不把已结束的旧任务继续计入。没有业务诉求时：明确非业务用non_business；指代不清或业务需求无法确定用unclear_request，二者不可混同。业务答案未知不等于意图不清楚。
只输出合法JSON对象，不要代码围栏或解释。结构为{"cases":[...]}，恰好10项。每项字段：message（当前中文问题）、history（按user/assistant交替排列的完整问答列表，每项role/content）、history_kind（none/context_or_distractor/dependent）、expected_intents（C01至C09的无重复数组，非业务或无法分类为空数组）、expected_fallback（null/non_business/unclear_request）、reason（简短分类理由）。业务可清楚分类时fallback必须null。不要在真实message/history里写分类标签。
类别目录：''' + json.dumps(TAXONOMY, ensure_ascii=False)


def parse_candidates(content, group):
    body = json.loads(content)
    rows = body.get('cases') if isinstance(body, dict) else None
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError('expected exactly ten candidates')
    output = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError('invalid candidate')
        message, history = row.get('message'), row.get('history')
        labels, fallback = row.get('expected_intents'), row.get('expected_fallback')
        kind, reason = row.get('history_kind'), row.get('reason')
        if not isinstance(message, str) or not message.strip() or len(message) > 2000:
            raise ValueError('invalid message')
        if not isinstance(history, list) or len(history) % 2 or len(history) > 10:
            raise ValueError('history must be complete QA pairs')
        for i, item in enumerate(history):
            if not isinstance(item, dict) or set(item) != {'role', 'content'} or item['role'] != ('user' if i % 2 == 0 else 'assistant'):
                raise ValueError('invalid history role/order')
            if not isinstance(item['content'], str) or not item['content'].strip() or len(item['content']) > 2000:
                raise ValueError('invalid history content')
        if not isinstance(labels, list) or any(not isinstance(k, str) or k not in TAXONOMY for k in labels) or len(set(labels)) != len(labels):
            raise ValueError('invalid candidate labels')
        if fallback not in (None, 'non_business', 'unclear_request') or bool(labels) != (fallback is None):
            raise ValueError('inconsistent fallback and labels')
        if kind not in ('none', 'context_or_distractor', 'dependent') or bool(history) != (kind != 'none'):
            raise ValueError('inconsistent history kind')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('missing proposed rationale')
        build_requests(message, history, provider='decisions')
        output.append({'id': f'qwen-{group}-{index:02}', 'group': group, 'message': message,
                       'history': history, 'history_kind': kind, 'proposed_intents': sorted(labels),
                       'proposed_fallback': fallback, 'proposed_reason': reason, 'review_status': 'pending'})
    return output


def freeze_review(home):
    home = Path(home)
    names = ('reviewed-cases.json', 'disputed-cases.json', 'review-manifest.json')
    if any((home / name).exists() for name in names):
        raise FileExistsError('review export already frozen; refuse overwrite')
    read = lambda name: json.loads((home / name).read_text(encoding='utf-8'))
    candidates, review = read('candidates.json'), read('review-decisions.json')
    decisions = review['decisions']
    by = {r['id']: r for r in decisions}
    if len(by) != len(decisions) or set(by) != {r['id'] for r in candidates}:
        raise ValueError('every candidate must have exactly one review decision')
    accepted, disputed, changes = [], [], []
    for candidate in candidates:
        d = by[candidate['id']]
        if d['status'] not in ('accepted', 'disputed') or not d.get('reason'):
            raise ValueError('review status and rationale required')
        if d['status'] == 'disputed':
            disputed.append({'candidate': candidate, 'review': d})
            continue
        labels, fallback = d['intents'], d['fallback']
        if not isinstance(labels, list) or any(k not in TAXONOMY for k in labels) or len(set(labels)) != len(labels):
            raise ValueError('invalid reviewed labels')
        if fallback not in (None, 'non_business', 'unclear_request') or bool(labels) != (fallback is None):
            raise ValueError('invalid reviewed fallback')
        kind = d.get('history_kind', candidate['history_kind'])
        if kind not in ('none', 'context_or_distractor', 'dependent') or bool(candidate['history']) != (kind != 'none'):
            raise ValueError('invalid reviewed history type')
        accepted.append({'id': candidate['id'], 'group': candidate['group'], 'message': candidate['message'],
            'history': candidate['history'], 'history_kind': kind, 'expected_intents': sorted(labels),
            'expected_fallback': fallback, 'expected_roles': sorted({INTENT_ROUTING[k].value for k in labels}) or ['general'],
            'review_status': 'assistant_reviewed_synthetic', 'review_reason': d['reason']})
        if sorted(labels) != candidate['proposed_intents'] or fallback != candidate['proposed_fallback'] or kind != candidate['history_kind']:
            changes.append({'id': candidate['id'], 'review': d})
    if not accepted:
        raise ValueError('no accepted candidates')
    digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    manifest = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'candidates': len(candidates),
        'accepted': len(accepted), 'disputed': len(disputed), 'changes': changes,
        'candidates_sha256': digest(candidates), 'review_sha256': digest(review), 'cases_sha256': digest(accepted),
        'limitation': 'Qwen-generated, Codex assistant-reviewed before routing; not independent human gold'}
    write_json(home / names[0], accepted)
    write_json(home / names[1], disputed)
    write_json(home / names[2], manifest)


async def generate(home, *, continue_unattempted=False):
    home = Path(home)
    values = dotenv_values(PROJECT / '.env', encoding='utf-8-sig', interpolate=False)
    sink, gate = ObservationSink('qwen-independent-routing-cases-v1'), PaidCallGate()
    if continue_unattempted:
        read = lambda name: json.loads((home / name).read_text(encoding='utf-8'))
        manifest, old = read('manifest.json'), read('telemetry.json')
        if manifest['phase'] != 'stopped' or manifest['model'] != values.get('LLM_MODEL'):
            raise ValueError('only a stopped run with unchanged model can continue')
        if read('generation-prompts.json') != {'system': SYSTEM, 'groups': BRIEFS}:
            raise ValueError('generation prompts changed')
        sink.events = old['events']
        attempts = [e for e in sink.events if e['type'] == 'http_call']
        if manifest.get('credential_blocked') or manifest.get('stopped_reason') == 'credential_or_billing' or len(attempts) >= 4 or any(e.get('status_code') in (401, 402, 403, 429) for e in sink.events):
            raise ValueError('call cap exhausted or credential/rate error; cannot continue')
        attempted = {e['turn_id'] for e in attempts}
        if len(attempted) != len(attempts) or not attempted <= BRIEFS.keys():
            raise ValueError('invalid prior attempt ledger')
        candidates, raw = read('candidates.json'), read('raw-responses.json')
        groups = {k: v for k, v in BRIEFS.items() if k not in attempted}
        manifest.update(phase='running', continued_at=datetime.now(timezone.utc).isoformat())
    else:
        home.mkdir(parents=True, exist_ok=False)  # Refuse automatic replay.
        manifest = {'phase': 'running', 'complete': False, 'max_http_attempts': 4, 'retries': 0,
                'model': values.get('LLM_MODEL'), 'started_at': datetime.now(timezone.utc).isoformat(),
                'label_status': 'model proposals only; review required; not human gold',
                'input_scope': 'task taxonomy and group brief only; no old cases, prompts, results or credentials in messages'}
        candidates, raw, groups = [], [], BRIEFS
        write_json(home / 'generation-prompts.json', {'system': SYSTEM, 'groups': BRIEFS})
    def save():
        manifest['credential_blocked'] = gate.blocked
        write_json(home / 'manifest.json', manifest)
        write_json(home / 'raw-responses.json', raw)
        write_json(home / 'candidates.json', candidates)
        write_json(home / 'telemetry.json', {'events': sink.events, 'usage': sink.totals('http_result')})
    save()
    try:
        async with httpx.AsyncClient(trust_env=True, follow_redirects=False, event_hooks=sink.http_hooks(gate)) as client:
            model = OpenAICompatibleModel(values.get('LLM_BASE_URL', ''), values.get('LLM_API_KEY', ''),
                values.get('LLM_MODEL', ''), client=client, max_output_tokens=10000, timeout_s=120)
            for group, brief in groups.items():
                if sum(e['type'] == 'http_call' for e in sink.events) >= 4:
                    raise ValueError('global generation call cap reached')
                sink.turn_id = group
                try:
                    reply = await model.complete([{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': brief}], [], allow_tools=False)
                    raw.append({'group': group, **reply})
                    save()  # Keep raw response before parsing or reviewing.
                    if reply['finish_reason'] != 'stop':
                        raise ValueError('incomplete generation')
                    candidates.extend(parse_candidates(reply['message']['content'], group))
                except Exception as exc:
                    manifest.update(phase='stopped', stopped_reason='credential_or_billing' if gate.blocked else type(exc).__name__)
                    return
                save()
                print(f'Generated {group}: 10 candidates; total {len(candidates)}', flush=True)
                if gate.blocked:
                    manifest.update(phase='stopped', stopped_reason='credential_or_billing')
                    return
        manifest.update(phase='finished' if len(candidates) == 40 else 'finished_with_gaps', complete=len(candidates) == 40,
                        finished_at=datetime.now(timezone.utc).isoformat())
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--continue-unattempted', action='store_true', help='Never retry an attempted group; keep global four-call cap')
    parser.add_argument('--freeze-review', action='store_true', help='Offline export of reviewed labels; no config/API reads')
    args = parser.parse_args()
    if args.freeze_review:
        if args.continue_unattempted:
            parser.error('review export and paid generation are separate operations')
        freeze_review(args.output)
    else:
        asyncio.run(generate(args.output, continue_unattempted=args.continue_unattempted))
