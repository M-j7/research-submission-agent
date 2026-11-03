"""EvalScope drives the project's graph, then replays independent saved scores."""
import json
from pathlib import Path
import os
import logging

from .dataset import PROJECT
os.environ.setdefault('EVALSCOPE_CACHE', str(PROJECT/'.cache/evaluation/evalscope'))
from evalscope import TaskConfig, run_task
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters.agent_adapter import AgentAdapter
from evalscope.api.agent import AgentTrace, AgentTraceEvent, EventType
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.evaluator import InferenceResult
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import AggScore, Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.model import ModelAPI, ModelOutput
from evalscope.api.registry import register_benchmark
from evalscope.metrics.semantics.catalog import BENCHMARK_METRIC_OVERRIDES
from evalscope.metrics.semantics.entry import MetricEntry

NAME = 'ieee_project_agent'
RUNNERS = {}  # Run-scoped callback only; no serialized credentials or global current case.
METRICS = ['verified_success', 'valid_task_pass', 'unavailable', 'execution_completed', 'sample_accounted']
for name in METRICS:
    BENCHMARK_METRIC_OVERRIDES[(NAME, name)] = MetricEntry(baseline='quality.score.ratio', metric_name=name)


def replay_values(row):
    state = row['business_outcome']
    result = {'verified_success': int(state == 'pass'), 'unavailable': int(state == 'unavailable'), 'sample_accounted': 1}
    if state in ('pass', 'fail'):
        result['valid_task_pass'] = int(state == 'pass')
    return result


def to_agent_trace(execution):
    mapping = {'model_result': EventType.MODEL_GENERATE, 'tool_call': EventType.TOOL_CALL,
               'tool_result': EventType.TOOL_RESULT, 'tool_error': EventType.ERROR, 'model_error': EventType.ERROR,
               'turn_start': EventType.RUN_START, 'turn_end': EventType.RUN_END}
    trace = AgentTrace(framework='project-langgraph', strategy='existing-react', environment='isolated-evaluation',
                       max_steps=8, trial_id=execution['trial_id'])
    steps = {}
    for e in execution.get('events', []):
        if e['type'] not in mapping:
            continue
        key = (e.get('turn_id'), e.get('role'))
        if e['type'] == 'model_result':
            steps[key] = steps.get(key, 0) + 1
        usage = e.get('usage', {})
        tokens = {'input': usage['input_tokens'], 'output': usage['output_tokens']} if all(type(usage.get(k)) is int for k in ('input_tokens','output_tokens')) else None
        trace.events.append(AgentTraceEvent(step=max(0, steps.get(key, 1)-1), timestamp=e['timestamp'], type=mapping[e['type']],
            message_id=e.get('call_id'), latency_ms=e.get('latency_ms'), token_usage=tokens,
            payload={k: v for k,v in e.items() if k not in ('timestamp', 'usage')}))
    return trace


@register_benchmark(BenchmarkMeta(name=NAME, dataset_id=NAME, pretty_name='IEEE full Agent — draft diagnostic',
    description='Existing LangGraph, independent rules and uncalibrated Decisions rubric. Business/fault denominators separate.',
    subset_list=['business','fault'], metric_list=METRICS,
    primary_metric=MetricSelector(name='verified_success', aggregation='mean'),
    train_split=None, eval_split='test', few_shot_num=0, extra_params={'run_path': '', 'mode': 'replay'}))
class ProjectAgentAdapter(AgentAdapter):
    def load(self):
        home = Path(self.extra_params['run_path'])
        cases = json.loads((home/'cases.json').read_text(encoding='utf-8'))
        groups = {}
        for group in self.subset_list:
            samples = [Sample(input=[ChatMessageUser(content=c['case_id'])], target='independent saved task criteria',
                              metadata={'case_id': c['case_id'], 'review_status': c['review_status']})
                       for c in cases if c['group'] == group]
            if samples:
                ds = MemoryDataset(samples, name=group)
                ds.reindex()
                groups[group] = ds
        return DatasetDict(groups), None

    def record_to_sample(self, record):
        raise NotImplementedError('frozen in-memory dataset')

    def _on_inference(self, model, sample):
        home = Path(self.extra_params['run_path'])
        case_id = sample.metadata['case_id']
        if self.extra_params['mode'] == 'live':
            execution = RUNNERS[str(home.resolve())].invoke(case_id)
            scored = {'business_outcome': 'unavailable'}
        else:
            # Replay cannot reach runtime or invoke any paid model.
            executions = [json.loads(l) for l in (home/'executions.jsonl').read_text(encoding='utf-8').splitlines() if l]
            scores = [json.loads(l) for l in (home/'scores/v1/scores.jsonl').read_text(encoding='utf-8').splitlines() if l]
            execution = next(r for r in executions if r['case_id'] == case_id)
            scored = next(r for r in scores if r['case_id'] == case_id)
        row = {**scored, 'execution_completed': execution['execution_status'] == 'completed'}
        return InferenceResult(output=ModelOutput.from_content(model='project-agent-saved-artifact',
            content=json.dumps(row, ensure_ascii=False)), trace=to_agent_trace(execution))

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        row = json.loads(filtered_prediction)
        values = replay_values(row)
        values['execution_completed'] = int(row['execution_completed'])
        return Score(value=values, prediction=original_prediction, extracted_prediction=filtered_prediction,
                     main_score_name='verified_success', metadata={'outcome': row['business_outcome']})

    def aggregate_scores(self, sample_scores):
        result = []
        for name in METRICS:
            known = [(s.sample_id, s.score.value[name]) for s in sample_scores if name in s.score.value]
            if known:
                result.append(AggScore(metric_name=name, aggregation='mean', score=sum(v for _,v in known)/len(known),
                    num=len(known), ids=[sid for sid,_ in known], metadata={'expected':len(sample_scores),'unavailable':len(sample_scores)-len(known)}))
        return result


class DriverModel(ModelAPI):
    def __init__(self):
        super().__init__(model_name='project-langgraph-agent')
    def generate(self, input, tools, tool_choice, config):
        raise RuntimeError('adapter must own execution; no second model loop')


def run_evalscope(home, *, mode, concurrency=1, runner=None):
    key = str(home.resolve())
    if runner:
        RUNNERS[key] = runner
    try:
        groups = sorted({c['group'] for c in json.loads((home/'cases.json').read_text(encoding='utf-8'))})
        run_task(TaskConfig(model=DriverModel(), eval_type='mock_llm', datasets=[NAME],
            dataset_args={NAME: {'subset_list':groups, 'extra_params':{'run_path':key,'mode':mode}}},
            eval_batch_size=concurrency, work_dir=str(home/'evalscope'/mode)))
    finally:
        RUNNERS.pop(key, None)
        # EvalScope leaves its FileHandler open on Windows. Close only this run's files.
        for name in ['evalscope', *logging.Logger.manager.loggerDict]:
            if name != 'evalscope' and not name.startswith('evalscope.'):
                continue
            logger = logging.getLogger(name)
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve().is_relative_to(home.resolve()):
                    logger.removeHandler(handler)
                    handler.close()
