import importlib
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.rag.ragas_score import METRICS, metric_result


class GenerationReportTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module('evaluation.rag.generation_report')
        except ModuleNotFoundError:
            self.api = None
        self.assertIsNotNone(self.api, 'generation report replay missing')
        self.rows = [dict(sample_id=str(i), config_id='B4_multiquery_no_jev', language='zh',
                          generation_status='ok', metrics={k: metric_result(v) for k in METRICS})
                     for i, v in enumerate([1, 0, None])]

    def test_unavailable_is_omitted_from_quality_not_changed_to_zero(self):
        replay = [self.api.replay_values(r) for r in self.rows]
        self.assertEqual(replay[1]['faithfulness'], 0)
        self.assertNotIn('faithfulness', replay[2])
        self.assertEqual(replay[2]['faithfulness_unavailable'], 1)
        self.assertEqual(replay[0]['faithfulness_unavailable'], 0)
        self.assertEqual(replay[2]['sample_accounted'], 1)

    def test_chinese_summary_explicitly_shows_denominators_and_limits(self):
        report = self.api.render_summary(self.rows, {'limitations': ['same-model judge; agent-draft references']})
        self.assertIn('0.5000', report)
        self.assertIn('2/3', report)
        self.assertIn('B4_multiquery_no_jev', report)
        self.assertIn('same-model judge', report)

    @unittest.skipUnless(importlib.util.find_spec('evalscope'), 'optional EvalScope environment')
    def test_native_aggregation_uses_two_valid_not_three_expected(self):
        from evaluation.rag.generation_evalscope import aggregate_replay
        from evalscope.api.metric import SampleScore, Score
        aggregates = aggregate_replay([SampleScore(sample_id=r['sample_id'], score=Score(value=self.api.replay_values(r)))
                                       for r in self.rows])
        actual = next(a for a in aggregates if a.metric_name == 'faithfulness')
        self.assertEqual(actual.score, .5)
        self.assertEqual(actual.num, 2)
        self.assertEqual(actual.metadata['total_expected'], 3)
        self.assertEqual(actual.metadata['unavailable'], 1)

    @unittest.skipUnless(importlib.util.find_spec('evalscope'), 'optional EvalScope environment')
    def test_report_metric_semantics_resolve_before_native_render(self):
        from evaluation.rag.generation_evalscope import NAME, ALL_METRICS
        from evalscope.metrics.semantics.catalog import BENCHMARK_METRIC_OVERRIDES
        for name in ALL_METRICS:
            semantics = BENCHMARK_METRIC_OVERRIDES[(NAME, name)].resolve(name)
            self.assertIsNotNone(semantics)


if __name__ == '__main__':
    unittest.main()
