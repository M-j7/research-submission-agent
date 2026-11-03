"""Native EvalScope visualization: replay actual RAGAS numbers, not model calls."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .dataset import PROJECT
from .generation_report import replay_values
from .ragas_score import METRICS, load_scores

for key, directory in [('MTEB_CACHE', 'mteb'), ('EVALSCOPE_CACHE', 'evalscope'), ('MODELSCOPE_CACHE', 'modelscope')]:
    os.environ.setdefault(key, str(PROJECT / '.cache/evaluation' / directory))

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

NAME = 'ieee_ragas_generation_replay'
ALL_METRICS = list(METRICS) + [m + '_unavailable' for m in METRICS] + ['sample_accounted']
for name in METRICS:
    baseline = 'quality.judge_score.unbounded' if name == 'answer_relevancy' else 'quality.score.ratio'
    BENCHMARK_METRIC_OVERRIDES[(NAME, name)] = MetricEntry(baseline=baseline, metric_name=name)
for name in ALL_METRICS[len(METRICS):]:
    BENCHMARK_METRIC_OVERRIDES[(NAME, name)] = MetricEntry(baseline='diagnostic.parse_status.ratio', display_name=name)


def aggregate_replay(sample_scores):
    aggregates = []
    for name in ALL_METRICS:
        valid = [(s.sample_id, s.score.value[name]) for s in sample_scores if name in s.score.value]
        if valid:  # Entirely unavailable metrics do not acquire a fabricated 0.
            aggregates.append(AggScore(metric_name=name, aggregation='mean',
                                       score=sum(v for _, v in valid) / len(valid), num=len(valid),
                                       ids=[sid for sid, _ in valid],
                                       metadata={'total_expected': len(sample_scores),
                                                 'unavailable': len(sample_scores) - len(valid),
                                                 'source': 'official_ragas_saved_score_replay'}))
    return aggregates


@register_benchmark(BenchmarkMeta(
    name=NAME, dataset_id=NAME, pretty_name='IEEE RAGAS generation — saved-score replay',
    description='Fixed-context RAG, draft references, Qwen same-model judge. B4 has no JEV validation. '
                'Quality means use valid scores only; unavailable diagnostics and Chinese report give denominators.',
    subset_list=['zh', 'en'], metric_list=ALL_METRICS,
    primary_metric=MetricSelector(name='faithfulness', aggregation='mean'),
    train_split=None, eval_split='test', few_shot_num=0,
    extra_params={'run_path': '', 'score_path': '', 'config_id': ''},
))
class GenerationReplayAdapter(DefaultDataAdapter):
    def aggregate_scores(self, sample_scores):
        return aggregate_replay(sample_scores)

    def load(self):
        manifest, answers, rows = load_scores(Path(self.extra_params['run_path']), Path(self.extra_params['score_path']))
        inputs = {r['sample_id']: r for r in answers}
        config_id = self.extra_params['config_id']
        subsets = {}
        for language in self.subset_list:
            samples = [Sample(input=[ChatMessageUser(content=r['sample_id'])], target='recorded RAGAS scores',
                              metadata={'sample_id': r['sample_id'], 'question': inputs[r['sample_id']]['user_input'],
                                        'answer': inputs[r['sample_id']]['response'], 'config_id': config_id,
                                        'semantic_validation': inputs[r['sample_id']]['semantic_validation'],
                                        'limitations': manifest['limitations'],
                                        'metric_status': r['metrics']})
                       for r in rows if r['config_id'] == config_id and r['language'] == language]
            dataset = MemoryDataset(samples, name=f'{config_id}.{language}')
            dataset.reindex()
            subsets[language] = dataset
        return DatasetDict(subsets), None

    def record_to_sample(self, record):
        raise NotImplementedError('validated in-memory saved scores only')

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        row = json.loads(filtered_prediction)
        return Score(value=replay_values(row), prediction=original_prediction,
                     extracted_prediction=filtered_prediction, main_score_name='faithfulness',
                     metadata={'availability': row['metrics'], 'source': 'official_ragas_saved_score_replay'})


class SavedScoreReplay(ModelAPI):
    def __init__(self, rows, config_id):
        self.rows = {r['sample_id']: r for r in rows if r['config_id'] == config_id}
        super().__init__(model_name=config_id + '-RAGAS-score-replay')

    def generate(self, input, tools, tool_choice, config):
        row = self.rows[input[-1].content]
        return ModelOutput.from_content(model=self.model_name, content=json.dumps(row, ensure_ascii=False))


def run_evalscope(run_dir, score_dir):
    _, _, rows = load_scores(run_dir, score_dir)
    for config_id in sorted({r['config_id'] for r in rows}):
        config = TaskConfig(model=SavedScoreReplay(rows, config_id), eval_type='mock_llm',
                            datasets=[NAME], dataset_args={NAME: {'extra_params': {
                                'run_path': str(run_dir.resolve()), 'score_path': str(score_dir.resolve()),
                                'config_id': config_id}}}, eval_batch_size=1,
                            work_dir=str((score_dir / 'evalscope' / config_id).resolve()))
        run_task(config)
