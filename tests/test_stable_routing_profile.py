import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from submission_assistant.intent.jev import build_requests


class StableProfileTests(unittest.TestCase):
    def test_stable_payload_matches_frozen_candidate(self):
        plan = json.loads((ROOT / 'evaluation/agent/clarity-probes/20261022-paired-01/plan.json').read_text(encoding='utf-8'))
        for query in plan['queries']:
            if query['variant'] != 'candidate':
                continue
            state = query['bodies'][0]['state']
            actual = build_requests(state['current_message'], state['history'], model=plan['model'],
                                    provider='decisions', prompt_version='routing-g6-clarity-v3')
            self.assertEqual(actual, query['bodies'], query['case_id'])
            self.assertEqual([len(b['questions']) for b in actual], [6, 5])

    def test_stable_history_keeps_five_complete_pairs(self):
        history = [dict(role=role, content=f'{i}-{role}') for i in range(6) for role in ('user', 'assistant')]
        bodies = build_requests('继续', history, provider='decisions', prompt_version='routing-g6-clarity-v3')
        self.assertEqual(bodies[0]['state']['history'], history[2:])

    def test_legacy_profiles_unchanged(self):
        from submission_assistant.intent.routing_prompts import ROUTING_PROMPT_VERSION
        self.assertEqual(build_requests('摘要字数'), build_requests('摘要字数', prompt_version='baseline'))
        self.assertEqual(len(build_requests('摘要字数', prompt_version=ROUTING_PROMPT_VERSION)), 2)
