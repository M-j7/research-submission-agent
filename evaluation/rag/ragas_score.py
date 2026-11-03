"""Official RAGAS 0.4.x metrics over saved answers; no JEV or retrieval calls."""
from __future__ import annotations

import argparse
import asyncio
import copy
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time

from .dataset import PROJECT, file_hash, read_jsonl, unique_map, write_jsonl
from .generate import load_generation_run
from .generation_data import value_hash

# Set before importing optional frameworks. No external tracing of evaluation data.
os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_TRACING'] = 'false'
METRICS = ('faithfulness', 'answer_correctness', 'answer_relevancy', 'context_precision', 'context_recall')


def metric_result(value, reason=None):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return {'value': float(value), 'status': 'ok', 'reason': None}
    return {'value': None, 'status': 'unavailable', 'reason': reason or 'non_finite_or_missing_score'}


def summarize(rows):
    result = {}
    for config_id in sorted({r['config_id'] for r in rows}):
        selected = [r for r in rows if r['config_id'] == config_id]
        result[config_id] = {}
        for language in ['all'] + sorted({r['language'] for r in selected}):
            group = [r for r in selected if language == 'all' or r['language'] == language]
            result[config_id][language] = {}
            for metric in METRICS:
                values = [r['metrics'][metric]['value'] for r in group
                          if r['metrics'][metric]['status'] == 'ok' and r['metrics'][metric]['value'] is not None]
                result[config_id][language][metric] = {
                    'total': len(group), 'valid': len(values), 'unavailable': len(group) - len(values),
                    'mean': sum(values) / len(values) if values else None}
    return result


def make_metrics(judge, embeddings, *, initialize=True):
    # Pin the legacy official implementations: these preserve the requested five
    # metric definitions, including correctness's default factual/semantic weights.
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_correctness import AnswerCorrectness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.run_config import RunConfig

    if judge is not None and hasattr(judge, 'langchain_llm'):
        judge = copy.copy(judge)
        judge.langchain_llm = judge.langchain_llm.model_copy(deep=False)

    metrics = dict(zip(METRICS, [Faithfulness(llm=judge), AnswerCorrectness(llm=judge, embeddings=embeddings),
                                AnswerRelevancy(llm=judge, embeddings=embeddings),
                                LLMContextPrecisionWithReference(llm=judge), LLMContextRecall(llm=judge)]))
    descriptions = {}
    for name, metric in metrics.items():
        if initialize:
            # In RAGAS 0.4.3 max_retries is stop_after_attempt, so 1 is ONE attempt.
            metric.init(RunConfig(timeout=60, max_retries=1, max_wait=1, max_workers=2))
        prompts = metric.get_prompts()
        prompt_descriptions = {}
        for attr, prompt in prompts.items():
            prompt = copy.deepcopy(prompt)  # Never patch a framework-global prompt.
            prompt_descriptions[attr] = {'language': prompt.language, 'sha256': value_hash(str(prompt))}
            original = prompt.generate_multiple

            async def bounded(*args, _original=original, **kwargs):
                kwargs['retries_left'] = 0  # Disable hidden output-fix LLM calls.
                return await _original(*args, **kwargs)

            prompt.generate_multiple = bounded
            metric.set_prompts(**{attr: prompt})
        descriptions[name] = {'class': type(metric).__module__ + '.' + type(metric).__name__,
                              'required_fields': sorted(metric.required_columns['SINGLE_TURN']),
                              'prompts': prompt_descriptions, 'parser_retries': 0}
    descriptions['answer_correctness'].update(weights=[.75, .25], beta=1.0)
    descriptions['answer_relevancy'].update(strictness=3, scale='unclipped cosine similarity [-1,1]')
    return metrics, descriptions


def usage_callback():
    from langchain_core.callbacks import BaseCallbackHandler

    class Audit(BaseCallbackHandler):
        run_inline = True

        def __init__(self):
            self.calls = 0
            self.completed = 0
            self.known_usage = {}
            self.usage_complete = True
            self.decisions = []

        def on_chat_model_start(self, serialized, messages, **kwargs):
            self.calls += 1

        def on_llm_start(self, serialized, prompts, **kwargs):
            self.calls += 1

        def on_llm_end(self, response, **kwargs):
            self.completed += 1
            usage = (response.llm_output or {}).get('token_usage')
            if not isinstance(usage, dict) or not all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens')):
                self.usage_complete = False
                return
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if type(usage.get(key)) is int and usage[key] >= 0:
                    self.known_usage[key] = self.known_usage.get(key, 0) + usage[key]

        def on_llm_error(self, error, **kwargs):
            self.usage_complete = False

        def on_chain_end(self, outputs, **kwargs):
            # Official structured verdicts/reasons; no serialized client/config.
            def safe(value):
                if hasattr(value, 'model_dump'):
                    return value.model_dump()
                raise TypeError
            try:
                self.decisions.append(json.loads(json.dumps(outputs, default=safe, ensure_ascii=False)))
            except (TypeError, ValueError):
                pass

    return Audit()


async def score_records(records, *, judge, embeddings, on_row=None):
    unique_map(records, 'sample_id')
    semaphore = asyncio.Semaphore(2)

    async def one(record):
        async with semaphore:
            result = {k: record[k] for k in ('sample_id', 'query_id', 'config_id', 'language', 'input_sha256') if k in record}
            result.update(generation_status=record.get('generation_status'), metrics={}, judge_diagnostics={})
            reason = ('generation_' + str(record.get('generation_status')) if record.get('generation_status') != 'ok'
                      else 'missing_reference' if not record.get('reference', '').strip() else None)
            if reason:
                result['metrics'] = {name: metric_result(None, reason) for name in METRICS}
            else:
                from ragas.dataset_schema import SingleTurnSample
                sample = SingleTurnSample(**{k: record[k] for k in ('user_input', 'retrieved_contexts', 'response', 'reference')})
                metrics, _ = make_metrics(judge, embeddings)
                for name, metric in metrics.items():
                    callback = usage_callback()
                    start = time.monotonic()
                    try:
                        value = await metric.single_turn_ascore(sample, callbacks=[callback], timeout=180)
                        result['metrics'][name] = metric_result(float(value))
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        # Provider error strings may contain request data. Persist only type.
                        result['metrics'][name] = metric_result(None, type(error).__name__)
                    result['judge_diagnostics'][name] = {
                        'calls': callback.calls, 'completed': callback.completed,
                        'known_usage': callback.known_usage,
                        'usage_complete': callback.usage_complete and callback.calls == callback.completed,
                        'embedding_usage': None, 'elapsed_seconds': round(time.monotonic() - start, 3),
                        'official_outputs': callback.decisions}
                    print(json.dumps({'sample_id': record['sample_id'], 'metric': name,
                                      'status': result['metrics'][name]['status'],
                                      'reason': result['metrics'][name]['reason']}, ensure_ascii=False), flush=True)
            if on_row:
                on_row(result)
            return result
    return await asyncio.gather(*(one(r) for r in records))


def make_clients(values, sync_http, async_http):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from submission_assistant.runtime.provider import RoleModelRouter
    model = RoleModelRouter(values).for_role('general')
    chat = ChatOpenAI(model=model.model, base_url=model.base_url, api_key=model.api_key,
                      max_retries=0, timeout=60, max_tokens=4096, temperature=.01, n=1,
                      http_client=sync_http, http_async_client=async_http)
    embedding_url = values['EMBEDDING_BASE_URL'].rstrip('/')
    if embedding_url.endswith('/embeddings'):
        embedding_url = embedding_url[:-len('/embeddings')]
    embedding = OpenAIEmbeddings(model=values['EMBEDDING_MODEL'], base_url=embedding_url,
                                 api_key=values['EMBEDDING_API_KEY'], max_retries=0, request_timeout=60,
                                 check_embedding_ctx_length=False, chunk_size=16,
                                 http_client=sync_http, http_async_client=async_http)
    config = RunConfig(timeout=60, max_retries=1, max_wait=1, max_workers=2)
    return (LangchainLLMWrapper(chat, run_config=config, bypass_n=True), LangchainEmbeddingsWrapper(embedding, run_config=config),
            {'judge_model': model.model, 'embedding_model': values['EMBEDDING_MODEL'],
             'sdk_max_retries': 0, 'ragas_max_attempts': 1, 'parser_retries': 0,
             'max_judge_output_tokens': 4096, 'request_timeout_s': 60, 'metric_timeout_s': 180,
             'relevancy_sampling': '3 separate n=1 requests (endpoint does not support n=3)',
             'concurrency': 2, 'prompt_language': 'official English prompts for both languages',
             'embedding_usage': 'unavailable; not assumed free',
             'versions': {p: version(p) for p in ('ragas', 'langchain-openai', 'evalscope')}})


def load_scores(run_dir, score_dir):
    manifest, answers = load_generation_run(run_dir)
    score_dir = Path(score_dir)
    scoring = json.loads((score_dir / 'manifest.json').read_text(encoding='utf-8'))
    if not scoring.get('complete') or scoring['answers_sha256'] != manifest['answers_sha256']:
        raise ValueError('incomplete or mismatched scoring run')
    if scoring['scores_sha256'] != file_hash(score_dir / 'scores.jsonl'):
        raise ValueError('score hash mismatch')
    rows = read_jsonl(score_dir / 'scores.jsonl')
    mapping = unique_map(rows, 'sample_id')
    if set(mapping) != {r['sample_id'] for r in answers}:
        raise ValueError('score coverage mismatch')
    for answer in answers:
        row = mapping[answer['sample_id']]
        if any(row.get(k) != answer[k] for k in ('query_id', 'config_id', 'language', 'input_sha256', 'generation_status')):
            raise ValueError('score identity mismatch')
        if set(row['metrics']) != set(METRICS):
            raise ValueError('metric coverage mismatch')
        for metric in row['metrics'].values():
            if metric != metric_result(metric['value'], metric.get('reason')):
                raise ValueError('invalid score availability')
    return scoring, answers, rows


async def run(args):
    import httpx
    from dotenv import dotenv_values
    manifest, answers = load_generation_run(args.run)
    output = args.output or args.run / 'ragas'
    if output.exists():
        raise FileExistsError('scoring output exists; choose a new directory')
    values = {**dotenv_values(PROJECT / '.env', encoding='utf-8-sig', interpolate=False), **os.environ}
    with httpx.Client(trust_env=False, timeout=60) as sync_http:
        async with httpx.AsyncClient(trust_env=False, timeout=60) as async_http:
            judge, embeddings, settings = make_clients(values, sync_http, async_http)
            _, descriptions = make_metrics(judge, embeddings)
            output.mkdir(parents=True)
            write_jsonl(output / 'ragas-inputs.jsonl', [
                {'sample_id': r['sample_id'], **{k: r[k] for k in ('user_input', 'retrieved_contexts', 'response', 'reference')}}
                for r in answers])
            scoring = {'created_at': datetime.now(timezone.utc).isoformat(), 'complete': False,
                       'answers_sha256': manifest['answers_sha256'], 'settings': settings,
                       'metrics': descriptions, 'limitations': manifest['limitations'], 'jev_calls': 0}
            target = output / 'manifest.json'
            target.write_text(json.dumps(scoring, ensure_ascii=False, indent=2), encoding='utf-8')
            completed = 0
            def save(row):
                nonlocal completed
                with (output / 'scores.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                completed += 1
                print(json.dumps({'sample_id': row['sample_id'], 'completed': completed, 'total': len(answers),
                                  'valid_metrics': sum(m['status'] == 'ok' for m in row['metrics'].values())}), flush=True)
            rows = await score_records(answers, judge=judge, embeddings=embeddings, on_row=save)
            (output / 'summary.json').write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2), encoding='utf-8')
            scoring.update(complete=True, scores_sha256=file_hash(output / 'scores.jsonl'))
            target.write_text(json.dumps(scoring, ensure_ascii=False, indent=2), encoding='utf-8')
            load_scores(args.run, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if not args.live:
        parser.error('--live explicitly enables paid official RAGAS judging; no JEV')
    try:
        asyncio.run(run(args))
    except Exception as error:
        raise SystemExit('RAGAS scoring stopped: ' + type(error).__name__) from None


if __name__ == '__main__':
    main()
