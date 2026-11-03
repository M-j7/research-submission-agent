import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_independent_jev import judgments_fixture
from submission_assistant.intent.ensemble import fuse_results
from submission_assistant.intent.templates import TemplateIntentResult


class IndependentEnsembleTests(unittest.TestCase):
    def test_auxiliaries_never_override_independent_business_candidates(self):
        jev = judgments_fixture(scores={'C02':.95,'C07':.94})
        scores = {f'C{i:02d}': float(i==5) for i in range(1,11)}
        candidate = TemplateIntentResult('bm25','bm25',scores,{key:'example' for key in scores})
        result = fuse_results(jev=jev, bm25=candidate)
        self.assertEqual(result['judgments']['probabilities']['C02'], .95)
        self.assertEqual(result['judgments']['probabilities']['C07'], .94)
        self.assertEqual(result['primary_candidates'], ['C02','C07'])
        self.assertEqual(result['auxiliary_evidence']['bm25']['relation'], 'alternative_candidate')
        self.assertEqual(result['sources']['jev']['probabilities'], jev.probabilities)
        self.assertNotIn('multi_agent', result)

    def test_nonbusiness_auxiliary_is_only_evidence(self):
        jev = judgments_fixture(scores={'C02':.95})
        scores = {f'C{i:02d}': float(i==10) for i in range(1,11)}
        result = fuse_results(jev=jev, bm25=TemplateIntentResult('bm25','bm25',scores,{key:None for key in scores}))
        self.assertEqual(result['primary_candidates'], ['C02'])
        self.assertEqual(result['auxiliary_evidence']['bm25']['top_intents'], ['C10'])
        self.assertEqual(result['non_business_evidence']['bm25']['score'], 1)

    def test_missing_jev_is_not_replaced_by_auxiliary_votes(self):
        scores = {f'C{i:02d}': float(i==2) for i in range(1,11)}
        result = fuse_results(bm25=TemplateIntentResult('bm25','bm25',scores,{key:None for key in scores}))
        self.assertIsNone(result['judgments'])
        self.assertTrue(result['candidate_only'])

