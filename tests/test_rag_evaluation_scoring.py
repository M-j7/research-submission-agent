import importlib
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_FRAMEWORKS = importlib.util.find_spec('mteb') is not None


@unittest.skipUnless(HAS_FRAMEWORKS, 'run in .venv-evaluation for MTEB scoring')
class ScoringTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module('evaluation.rag.score')
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, 'official retrieval scoring adapter is missing')

    def test_official_metrics_keep_no_results_in_denominator(self):
        result = self.api.score_rankings(
            {'q1': {'a': 1, 'b': 1}, 'q2': {'c': 1}},
            {'q1': ['x', 'a'], 'q2': []}, ks=(1, 3, 5))
        self.assertAlmostEqual(result['aggregate']['Recall@3'], .25)
        self.assertAlmostEqual(result['aggregate']['MRR@3'], .25)
        self.assertAlmostEqual(result['aggregate']['Precision@3'], 1 / 6, places=5)
        self.assertAlmostEqual(result['aggregate']['HitRate@3'], .5)
        self.assertAlmostEqual(result['per_query']['q1']['NDCG@3'], .3868528072)
        self.assertTrue(all(value == 0 for value in result['per_query']['q2'].values()))

    def test_no_hit_and_perfect_ranking_are_not_conflated(self):
        result = self.api.score_rankings(
            {'q1': {'a': 1}, 'q2': {'b': 1}}, {'q1': ['z'], 'q2': ['b']})
        self.assertEqual(result['aggregate']['HitRate@1'], .5)
        self.assertEqual(result['per_query']['q1']['MRR@1'], 0)
        self.assertEqual(result['per_query']['q2']['NDCG@1'], 1)

    def test_missing_query_duplicate_rank_and_empty_qrels_are_rejected(self):
        for qrels, rankings in [({'q': {'a': 1}}, {}),
                                ({'q': {'a': 1}}, {'q': ['a', 'a']}),
                                ({'q': {}}, {'q': []})]:
            with self.subTest(qrels=qrels, rankings=rankings):
                with self.assertRaises(ValueError):
                    self.api.score_rankings(qrels, rankings)

    def test_evalscope_adapter_scores_rankings_not_business_answers(self):
        try:
            adapter = importlib.import_module('evaluation.rag.evalscope_adapter')
        except ModuleNotFoundError:
            adapter = None
        self.assertIsNotNone(adapter, 'EvalScope replay adapter is missing')
        score = adapter.RetrievalReplayAdapter.match_score(
            None, original_prediction='{"parent_ids":["x","a"],"status":"degraded"}',
            filtered_prediction='{"parent_ids":["x","a"],"status":"degraded"}',
            reference='{"a":1,"b":1}', task_state=None)
        self.assertEqual(score.value['Recall@5'], .5)
        self.assertEqual(score.value['MRR@5'], .5)
        self.assertEqual(score.value['execution_failed'], 0)
        self.assertEqual(score.value['execution_degraded'], 1)

    def test_evalscope_cutoff_is_a_dimension_and_primary_metric_is_resolvable(self):
        adapter = importlib.import_module('evaluation.rag.evalscope_adapter')
        from evalscope.api.metric import AggScore
        from evalscope.api.metric.semantics import MetricSelector
        from evalscope.report.generator import ReportGenerator
        self.assertTrue(hasattr(adapter, 'canonicalize_aggregates'), 'structured cutoff mapping is missing')
        rows = adapter.canonicalize_aggregates([
            AggScore(metric_name='Recall@5', aggregation='mean', score=.5, num=2),
            AggScore(metric_name='NDCG@5', aggregation='mean', score=.4, num=2)])
        self.assertEqual(rows[0].identity.name, 'recall')
        self.assertEqual(rows[0].identity.dimensions, {'k': 5})
        semantics, selection = ReportGenerator._resolve_semantics(
            adapter.NAME, [r.identity for r in rows],
            MetricSelector(name='recall', aggregation='mean', dimensions={'k': 5}))
        self.assertEqual(selection.identity.name, 'recall')
        self.assertEqual(rows[0].score, .5)


if __name__ == '__main__':
    unittest.main()
