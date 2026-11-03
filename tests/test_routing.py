"""Routing is program policy: no model/network dependency."""
import importlib
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_independent_jev import judgments_fixture


class RoutingTests(unittest.TestCase):
    def setUp(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.routing'), 'router missing')
        self.api = importlib.import_module('submission_assistant.routing')

    def decision(self, scores, clarity=.92):
        return self.api.route_intents(judgments_fixture(scores=scores, clarity=clarity)).to_dict()

    def test_one_intent_routes_professional(self):
        output = self.decision({'C02': .95})
        self.assertEqual(output['agent_types'], ['submission_requirements'])
        self.assertEqual(output['agent_intents'], {'submission_requirements': ['C02']})
        self.assertEqual(output['mode'], 'single')
        self.assertFalse(output['multi_agent'])

    def test_same_role_multi_intent_is_not_multi_agent(self):
        for scores, role in [({'C02':.95,'C03':.96},'submission_requirements'),
                             ({'C05':.99,'C06':.98,'C09':.97},'editorial_workflow'),
                             ({'C04':.95,'C08':.94},'publication_policy')]:
            with self.subTest(role=role):
                output = self.decision(scores)
                self.assertEqual(output['agent_types'], [role])
                self.assertEqual(output['matched_intents'], sorted(scores))
                self.assertFalse(output['multi_agent'])

    def test_cross_role_triggers_collaboration_with_owned_intents(self):
        output = self.decision({'C02':.95,'C03':.93,'C07':.94})
        self.assertEqual(output['agent_types'], ['submission_requirements','publication_fees'])
        self.assertEqual(output['agent_intents'], {'submission_requirements':['C02','C03'], 'publication_fees':['C07']})
        self.assertEqual(output['mode'], 'multiple')
        self.assertTrue(output['multi_agent'])
        self.assertEqual(output['collaboration_reason'], 'clear_business_intents_span_roles')

    def test_all_roles_dedup_and_no_general_for_clear_business(self):
        output = self.decision({f'C{i:02d}':.9 for i in range(1,10)})
        self.assertEqual(len(output['agent_types']), 4)
        self.assertNotIn('general', output['agent_types'])

    def test_fallback_reasons_do_not_confuse_social_ambiguity_and_failure(self):
        cases = [({'C10':.99}, .1, 'non_business'),
                 ({'C10':.99,'C02':.98}, .9, 'conflicting_judgments'),
                 ({'C02':.99,'C07':.98}, .1, 'unclear_request'),
                 ({'C02':.79}, .9, 'uncertain_intent')]
        for scores, clarity, reason in cases:
            with self.subTest(reason=reason):
                output = self.decision(scores, clarity)
                self.assertEqual(output['agent_types'], ['general'])
                self.assertEqual(output['fallback_reason'], reason)
                self.assertEqual(output['mode'], 'fallback')
                self.assertFalse(output['multi_agent'])
        self.assertEqual(self.api.route_intents(None).fallback_reason, 'recognition_failed')

    def test_partial_and_legacy_results_cannot_route_professionals(self):
        from submission_assistant.intent.contracts import BatchStatus, IntentJudgments
        partial = IntentJudgments({'C02':.99}, .99, 'test', 'request', [
            BatchStatus('batch_1','one','success',{}), BatchStatus('batch_2','two','failed',{},'HTTP 502')])
        self.assertEqual(self.api.route_intents(partial).fallback_reason, 'recognition_failed')
        self.assertEqual(self.api.route_intents({'probabilities': {'C02':.99}}).fallback_reason, 'recognition_failed')

    def test_threshold_boundary_and_config_validation(self):
        output = self.decision({'C07':.8}, .75)
        self.assertEqual(output['agent_types'], ['publication_fees'])
        for value in [True, math.nan, math.inf, -.01, 1.01, '0.8']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.api.RoutingSettings(relevance_threshold=value)
        configured = self.api.RoutingSettings(clarity_threshold=.6, relevance_threshold=.7)
        result = self.api.route_intents(judgments_fixture(scores={'C07':.71}, clarity=.61), configured)
        self.assertEqual(result.to_dict()['settings']['relevance_threshold'], .7)
        self.assertEqual(result.agent_types, [self.api.AgentType.PUBLICATION_FEES])

    def test_mutated_invalid_scores_fail_closed(self):
        result = judgments_fixture()
        result.probabilities['C02'] = math.nan
        self.assertEqual(self.api.route_intents(result).fallback_reason, 'recognition_failed')


if __name__ == '__main__':
    unittest.main()
