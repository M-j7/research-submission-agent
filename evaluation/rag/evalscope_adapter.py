"""EvalScope's native runner replays actual rankings, not simulated model answers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from .dataset import PROJECT
from .score import KS, load_run, score_rankings

os.environ.setdefault('MTEB_CACHE', str(PROJECT / '.cache/evaluation/mteb'))
os.environ.setdefault('EVALSCOPE_CACHE', str(PROJECT / '.cache/evaluation/evalscope'))
os.environ.setdefault('MODELSCOPE_CACHE', str(PROJECT / '.cache/evaluation/modelscope'))

from evalscope import TaskConfig, run_task  # noqa: E402
from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter  # noqa: E402
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample  # noqa: E402
from evalscope.api.messages import ChatMessageUser  # noqa: E402
from evalscope.api.metric import AggScore, Score  # noqa: E402
from evalscope.api.metric.semantics import MetricSelector  # noqa: E402
from evalscope.api.model import ModelAPI, ModelOutput  # noqa: E402
from evalscope.api.registry import register_benchmark  # noqa: E402
from evalscope.metrics.semantics.catalog import BENCHMARK_METRIC_OVERRIDES  # noqa: E402
from evalscope.metrics.semantics.entry import MetricEntry  # noqa: E402

NAME = 'ieee_parent_retrieval_replay'
METRICS = ['recall', 'ndcg', 'precision', 'mrr', 'hit_rate', 'execution_failed', 'execution_degraded']
# EvalScope 1.13 requires semantics for reportable custom metrics. Register only
# this benchmark's names in memory; never edit the installed framework catalog.
for name, label in [('ndcg', 'NDCG'), ('mrr', 'MRR'), ('hit_rate', 'Hit Rate')]:
    BENCHMARK_METRIC_OVERRIDES[(NAME, name)] = MetricEntry(baseline='quality.score.ratio', metric_name=label)
for name in ('execution_failed', 'execution_degraded'):
    BENCHMARK_METRIC_OVERRIDES[(NAME, name)] = MetricEntry(baseline='diagnostic.parse_status.ratio', display_name=name)


def canonicalize_aggregates(rows):
    """Expose K as a structured dimension, not an undeclared metric name."""
    output = []
    labels = {'recall': 'recall', 'ndcg': 'ndcg', 'precision': 'precision',
              'mrr': 'mrr', 'hit_rate': 'hit_rate', 'hitrate': 'hit_rate'}
    for row in rows:
        match = re.fullmatch(r'(recall|ndcg|precision|mrr|hit_rate|hitrate)[_@](\d+)', row.metric_name, re.I)
        if match:
            data = row.model_dump()
            data.update(metric_name=labels[match[1].lower()], dimensions={**row.dimensions, 'k': int(match[2])})
            row = AggScore.model_validate(data)
        output.append(row)
    return output


@register_benchmark(BenchmarkMeta(
    name=NAME, dataset_id=NAME, pretty_name='IEEE parent retrieval — recorded ranking replay',
    description='Offline grading of actual production retrieval rankings. Agent-draft, non-exhaustive qrels.',
    subset_list=['zh', 'en'], metric_list=METRICS,
    primary_metric=MetricSelector(name='recall', aggregation='mean', dimensions={'k': 5}),
    train_split=None, eval_split='test', few_shot_num=0,
    extra_params={'run_path': '', 'baseline': ''},
))
class RetrievalReplayAdapter(DefaultDataAdapter):
    def aggregate_scores(self, sample_scores):
        return canonicalize_aggregates(super().aggregate_scores(sample_scores))

    def load(self):
        _, queries, qrels, rows = load_run(Path(self.extra_params['run_path']))
        baseline = self.extra_params['baseline']
        subsets = {}
        for language in self.subset_list:
            samples = [Sample(input=[ChatMessageUser(content=row['query_id'])],
                              target=json.dumps(qrels[row['query_id']]),
                              metadata={'query_id': row['query_id'], 'question': queries[row['query_id']]['text'],
                                        'queries': row['queries'], 'baseline_description': row['baseline_description'],
                                        'semantic_validation': row.get('semantic_validation', 'jev'),
                                        'source_mode': 'recorded_actual_rankings_not_generation'})
                       for row in rows if row['baseline'] == baseline and row['language'] == language]
            dataset = MemoryDataset(samples, name=f'{baseline}.{language}')
            dataset.reindex()
            subsets[language] = dataset
        return DatasetDict(subsets), None

    def record_to_sample(self, record):
        raise NotImplementedError('This adapter uses validated in-memory replay records only.')

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        row = json.loads(filtered_prediction)
        scores = score_rankings({'query': json.loads(reference)}, {'query': row['parent_ids']})
        values = scores['per_query']['query']
        values.update(execution_failed=float(row['status'] == 'failed'),
                      execution_degraded=float(row['status'] == 'degraded'))
        return Score(value=values, prediction=original_prediction,
                     extracted_prediction=filtered_prediction, main_score_name='Recall@5',
                     metadata={'source_mode': 'recorded_actual_rankings_not_generation', 'execution_status': row['status'],
                               'semantic_validation': row.get('semantic_validation', 'jev')})


class RankingReplay(ModelAPI):
    def __init__(self, rows, baseline):
        self.rows = {r['query_id']: r for r in rows if r['baseline'] == baseline}
        unverified = any(r.get('semantic_validation') == 'skipped_by_user' for r in self.rows.values())
        super().__init__(model_name=f'{baseline}' + ('-no-JEV' if unverified else '') + '-recorded-retrieval')

    def generate(self, input, tools, tool_choice, config):
        query_id = input[-1].content
        row = self.rows[query_id]
        return ModelOutput.from_content(model=self.model_name, content=json.dumps({
            'parent_ids': row['parent_ids'], 'status': row['status'],
            'semantic_validation': row.get('semantic_validation', 'jev')}, ensure_ascii=False))


def run_evalscope(run_dir):
    manifest, _, _, rows = load_run(run_dir)
    results = {}
    for baseline in manifest['baselines']:
        config = TaskConfig(
            model=RankingReplay(rows, baseline), eval_type='mock_llm',
            datasets=[NAME], dataset_args={NAME: {'extra_params': {
                'run_path': str(run_dir.resolve()), 'baseline': baseline}}},
            work_dir=str((run_dir / 'evalscope' / baseline).resolve()),
            eval_batch_size=1,
        )
        results[baseline] = run_task(config)
    return results
