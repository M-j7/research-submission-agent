"""Positive allowlists. No graph/private state is serialized to the browser."""
from urllib.parse import urlsplit
import re

ROLES = frozenset(('general', 'submission_requirements', 'editorial_workflow', 'publication_policy', 'publication_fees', 'composer'))
TOOLS = frozenset(('glob', 'grep', 'read', 'query_memory', 'search_knowledge'))
TOOL_STATES = frozenset(('waiting', 'running', 'completed', 'failed', 'timed_out', 'cancelled'))


def safe_text(value, limit=2000):
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return ''
    text = str(value)[:limit]
    text = re.sub(r'(?i)(?:[a-z]:[\\/]|file://)[^\s<>"\']+', '[本地资源]', text)
    return text


def safe_url(value):
    if not isinstance(value, str):
        return None
    try:
        url = urlsplit(value)
        if url.scheme in ('http', 'https') and url.hostname and not url.username and not url.password:
            return value
    except ValueError:
        pass
    return None


def public_status(raw: dict) -> dict:
    text_keys = ('phase', 'execution_status', 'stop_reason')
    number_keys = ('current_step', 'max_steps', 'steps_remaining', 'tool_calls_used', 'elapsed_ms', 'remaining_time_ms')
    out = {k: safe_text(raw[k], 100) for k in text_keys if isinstance(raw.get(k), str)}
    out.update({k: raw[k] for k in number_keys if type(raw.get(k)) in (int, float)})
    if raw.get('agent_type') in ROLES:
        out['agent_type'] = raw['agent_type']
    if isinstance(raw.get('tools'), list):
        out['tools'] = [{'name': t['name'], 'state': t['state']} for t in raw['tools'][:20]
                        if isinstance(t, dict) and isinstance(t.get('name'), str) and t['name'] in TOOLS
                        and isinstance(t.get('state'), str) and t['state'] in TOOL_STATES]
    return out


def public_routing(raw: dict) -> dict:
    return {'agent_types': [x for x in raw.get('agent_types', []) if x in ROLES],
            'matched_intents': [x for x in raw.get('matched_intents', []) if x in {f'C{i:02d}' for i in range(1, 11)}],
            'mode': safe_text(raw.get('mode'), 40),
            'fallback_reason': safe_text(raw.get('fallback_reason'), 80) or None}


def public_result(raw: dict) -> dict:
    evidence = []
    for row in raw.get('evidence', [])[:30]:
        if not isinstance(row, dict):
            continue
        evidence.append({'title': safe_text(row.get('title') or '参考资料', 200),
                         'source_url': safe_url(row.get('source_url')),
                         'page': row.get('page') if type(row.get('page')) is int else None,
                         'source_locator': safe_text(row.get('source_locator'), 300),
                         'text': safe_text(row.get('text'), 2000)})
    return {'agent_type': raw.get('agent_type') if raw.get('agent_type') in ROLES else 'general',
            'intent_ids': [x for x in raw.get('intent_ids', []) if x in {f'C{i:02d}' for i in range(1, 11)}],
            'execution_status': raw.get('execution_status') if raw.get('execution_status') in ('completed','incomplete','failed','cancelled') else 'failed',
            'stop_reason': safe_text(raw.get('stop_reason'), 100),
            'unresolved': [safe_text(x, 500) for x in raw.get('unresolved', [])[:20]],
            'evidence': evidence}
