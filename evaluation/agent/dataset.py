"""Frozen Chinese draft task set; expected behavior never enters runtime input."""
from copy import deepcopy
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROLES = {'general', 'submission_requirements', 'editorial_workflow', 'publication_policy', 'publication_fees'}
CATEGORIES = {'single', 'multi', 'unclear', 'conversation', 'boundary', 'fault'}
FAULTS = {'timeout', 'no_progress', 'invalid_argument', 'unauthorized', 'injection'}


def validate_cases(cases):
    ids = set()
    for c in cases:
        if not isinstance(c, dict) or not c.get('case_id') or c['case_id'] in ids:
            raise ValueError('duplicate or missing case ID')
        ids.add(c['case_id'])
        if c['category'] not in CATEGORIES or c['group'] != ('fault' if c['category'] == 'fault' else 'business'):
            raise ValueError('invalid category/group')
        if not c['turns'] or len(c['expected_roles']) != len(c['turns']):
            raise ValueError('missing per-turn expectations')
        if any(not roles or not set(roles) <= ROLES for roles in c['expected_roles']):
            raise ValueError('invalid expected roles')
        if len({t['turn_id'] for t in c['turns']}) != len(c['turns']) or any(not t['message'].strip() for t in c['turns']):
            raise ValueError('invalid turns')
        if not c['source_refs'] or c['review_status'] not in ('agent_draft', 'human_reviewed'):
            raise ValueError('missing source/review state')
        if not set(c['required_actions']) <= {'search_knowledge', 'read', 'query_memory', 'glob', 'grep'}:
            raise ValueError('unknown required action')
        if c['fault'] is not None and c['fault'] not in FAULTS:
            raise ValueError('unknown fault')


def load_cases(path=None):
    cases = json.loads(Path(path or Path(__file__).with_name('cases.zh.json')).read_text(encoding='utf-8'))
    if path is None:
        # Only new runs opt into v2. Explicit paths include frozen historical runs.
        harness = json.loads(Path(__file__).with_name('fault-harness.v2.json').read_text(encoding='utf-8'))
        for c in cases:
            if c['fault']:
                c['expected_roles'] = [[harness['role']] for _ in c['turns']]
                c['fault_harness_version'] = harness['version']
    validate_cases(cases)
    return cases


def select_cases(cases, *, stage):
    if stage == 'full':
        return deepcopy(cases)
    if stage not in ('smoke', 'repeat'):
        raise ValueError('unknown stage')
    selected = {'single-01', 'multi-03', 'unclear-02', 'conversation-02', 'boundary-05', 'fault-02'}
    return deepcopy([c for c in cases if c['case_id'] in selected])


def execution_input(case):
    return {'case_id': case['case_id'], 'turns': [{k: t[k] for k in ('turn_id', 'message')} for t in case['turns']],
            'initial_memory': deepcopy(case['initial_memory']), 'fault': case['fault']}
