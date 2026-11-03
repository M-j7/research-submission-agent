import importlib
import json
from collections import Counter
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class AgentEvalDataTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec('evaluation.agent')
        self.assertIsNotNone(spec, 'Agent evaluation dataset missing')
        self.api = importlib.import_module('evaluation.agent.dataset')

    def test_frozen_cases_cover_six_groups_and_independent_smoke(self):
        cases = self.api.load_cases()
        self.assertEqual(len(cases), 30)
        self.assertEqual(set(Counter(c['category'] for c in cases).values()), {5})
        self.assertEqual(sum(c['group'] == 'business' for c in cases), 25)
        smoke = self.api.select_cases(cases, stage='smoke')
        self.assertEqual(len(smoke), 6)
        self.assertEqual(sum(c['group'] == 'fault' for c in smoke), 1)
        self.assertEqual(len({c['category'] for c in smoke}), 6)
        multi = [c for c in cases if c['category'] == 'multi']
        self.assertEqual(sum(len(c['expected_roles'][0]) > 1 for c in multi), 3)

    def test_duplicate_ids_and_invalid_roles_are_rejected(self):
        cases = self.api.load_cases()
        with self.assertRaises(ValueError):
            self.api.validate_cases(cases + [cases[0]])
        cases[0]['expected_roles'] = [['not_a_role']]
        with self.assertRaises(ValueError):
            self.api.validate_cases(cases)

    def test_new_fault_harness_is_versioned_without_migrating_frozen_v1(self):
        current = self.api.load_cases()
        frozen = self.api.load_cases(ROOT / 'evaluation/agent/cases.zh.json')
        self.assertEqual([c for c in current if not c['fault']], [c for c in frozen if not c['fault']])
        for c in current:
            if c['fault']:
                self.assertEqual(c['expected_roles'], [['submission_requirements']])
                self.assertEqual(c['fault_harness_version'], 'specialist-v2')
        for c in frozen:
            if c['fault']:
                self.assertEqual(c['expected_roles'], [['general']])

    def test_execution_inputs_do_not_include_reference_or_rubric(self):
        case = self.api.load_cases()[0]
        public = self.api.execution_input(case)
        self.assertEqual(set(public), {'case_id', 'turns', 'initial_memory', 'fault'})
        self.assertEqual(set(public['turns'][0]), {'turn_id', 'message'})
        self.assertNotIn('reference_notes', public)


if __name__ == '__main__':
    unittest.main()
