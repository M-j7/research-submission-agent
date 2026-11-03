import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


class AgentEvalAdapterTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.evalscope_adapter'), 'AgentAdapter missing')
        return importlib.import_module('evaluation.agent.evalscope_adapter')

    def test_replay_omits_unavailable_from_valid_pass_and_keeps_expected_denominator(self):
        api = self.api()
        self.assertEqual(api.replay_values({'business_outcome': 'pass'})['valid_task_pass'], 1)
        self.assertEqual(api.replay_values({'business_outcome': 'fail'})['valid_task_pass'], 0)
        missing = api.replay_values({'business_outcome': 'unavailable'})
        self.assertNotIn('valid_task_pass', missing)
        self.assertEqual(missing['verified_success'], 0)
        self.assertEqual(missing['unavailable'], 1)

    def test_trace_preserves_parallel_roles_and_unknown_usage(self):
        api = self.api()
        trace = api.to_agent_trace({'trial_id': 'trial', 'events': [
            {'type': 'model_result', 'timestamp': 1.0, 'role': 'publication_fees', 'run_id': 'r1',
             'parent_id': 'turn', 'call_id': 'c1', 'usage': {'input_tokens': None,'output_tokens': None}},
            {'type': 'tool_result', 'timestamp': 1.1, 'role': 'submission_requirements', 'run_id': 'r2', 'parent_id': 'turn'}]})
        self.assertEqual(trace.framework, 'project-langgraph')
        self.assertEqual(trace.events[0].payload['run_id'], 'r1')
        self.assertEqual(trace.events[1].payload['run_id'], 'r2')
        self.assertIsNone(trace.events[0].token_usage)


if __name__ == '__main__':
    unittest.main()
