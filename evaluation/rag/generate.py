"""Collect fixed-context RAG answers; references are never sent to the generator."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import sys
import time

from .dataset import PROJECT, file_hash, read_jsonl, unique_map, write_jsonl
from .generation_data import DEFAULT_SOURCES, PROMPT_VERSION, build_generation_messages, prepare_records, value_hash

sys.path.insert(0, str(PROJECT / 'src'))
from submission_assistant.runtime.provider import ModelServiceError, RoleModelRouter  # noqa: E402


def retryable(error):
    import httpx
    if isinstance(error, (TimeoutError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(error, ModelServiceError):
        match = re.search(r'HTTP\s*(\d{3})', str(error))
        return bool(match and int(match[1]) in {429, 500, 502, 503, 504})
    return False


async def collect_answer(record, model, *, deadline_s=125):
    start = time.monotonic()
    deadline = start + deadline_s
    output = {**record, 'response': '', 'generation_status': 'failed',
              'failure_reason': 'deadline', 'finish_reason': None, 'attempts': 0,
              'generator_model': model.model, 'usage': {}, 'usage_complete': True, 'retry_delays': []}
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        output['attempts'] += 1
        try:
            reply = await asyncio.wait_for(model.complete(build_generation_messages(record), [], allow_tools=False), min(60, remaining))
            usage = reply.get('usage')
            valid_usage = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
                                                       for k in ('prompt_tokens', 'completion_tokens'))
            output['usage_complete'] &= valid_usage
            if valid_usage:
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    if type(usage.get(key)) is int and usage[key] >= 0:
                        output['usage'][key] = output['usage'].get(key, 0) + usage[key]
            message = reply.get('message')
            finish = reply.get('finish_reason')
            output['finish_reason'] = finish
            if not isinstance(message, dict) or not isinstance(message.get('content'), str):
                output['failure_reason'] = 'invalid_response'
            else:
                output['response'] = message['content']
                if message.get('tool_calls') or finish == 'tool_calls':
                    output['failure_reason'] = 'unexpected_tool_call'
                elif finish == 'length':
                    output.update(generation_status='incomplete', failure_reason='truncated')
                elif not message['content'].strip():
                    output['failure_reason'] = 'empty_response'
                elif finish != 'stop':
                    output['failure_reason'] = 'unexpected_finish_reason'
                else:
                    output.update(generation_status='ok', failure_reason=None)
            break
        except asyncio.CancelledError:
            raise
        except Exception as error:
            output.update(failure_reason=type(error).__name__, usage_complete=False)
            if attempt == 1 or not retryable(error):
                break
            delay = random.uniform(0, min(4, 2 ** attempt))
            if time.monotonic() + delay >= deadline:
                break
            output['retry_delays'].append(delay)
            await asyncio.sleep(delay)
    if output['attempts'] == 0:
        output['usage_complete'] = False
    output['elapsed_seconds'] = round(time.monotonic() - start, 6)
    return output


def initialize_run(output, records, model):
    output = Path(output)
    if output.exists():
        raise FileExistsError('generation output already exists; choose a new directory')
    unique_map(records, 'sample_id')
    output.mkdir(parents=True)
    write_jsonl(output / 'inputs.jsonl', records)
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(),
                'mode': 'fixed_context_rag_generation_not_agent',
                'expected_sample_ids': [r['sample_id'] for r in records],
                'inputs_sha256': file_hash(output / 'inputs.jsonl'),
                'source_runs': {r['source_run']: r['source_manifest_sha256'] for r in records},
                'generator_model': model.model, 'prompt_version': PROMPT_VERSION,
                'system_prompt_sha256': value_hash(build_generation_messages(records[0])[0]),
                'parameters': {'temperature': .2, 'max_output_tokens': 2048, 'request_timeout_s': 60,
                               'max_attempts': 2, 'sample_deadline_s': 125, 'concurrency': 2},
                'jev_calls': 0, 'reference_status': 'agent_draft',
                'limitations': ['Fixed-context RAG only, not the complete Agent.',
                               'References are provisional agent drafts.',
                               'B4 rewrites were not semantically validated.',
                               'Same-model Qwen judging can have self-preference bias.'],
                'complete': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def load_generation_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise ValueError('generation run incomplete')
    for filename, key in [('inputs.jsonl', 'inputs_sha256'), ('answers.jsonl', 'answers_sha256')]:
        if file_hash(run_dir / filename) != manifest[key]:
            raise ValueError('generation file hash mismatch')
    for source, digest in manifest['source_runs'].items():
        if file_hash(Path(source) / 'manifest.json') != digest:
            raise ValueError('source manifest hash mismatch')
        from .score import load_run
        load_run(Path(source))  # Includes frozen annotation/reference hashes.
    inputs = unique_map(read_jsonl(run_dir / 'inputs.jsonl'), 'sample_id')
    rows = read_jsonl(run_dir / 'answers.jsonl')
    answers = unique_map(rows, 'sample_id')
    if set(inputs) != set(manifest['expected_sample_ids']) or set(answers) != set(inputs):
        raise ValueError('generation query coverage mismatch')
    for row in rows:
        original = inputs[row['sample_id']]
        if any(row.get(key) != value for key, value in original.items() if key != 'generation_status'):
            raise ValueError('answer input/reference drift')
        if row['input_sha256'] != value_hash(build_generation_messages(row)):
            raise ValueError('generation prompt hash mismatch')
    return manifest, rows


async def run(args):
    import httpx
    from dotenv import dotenv_values
    if args.output.exists():
        raise FileExistsError('generation output already exists')
    records = prepare_records(DEFAULT_SOURCES)
    if args.smoke:
        records = [r for r in records if r['query_id'] in {'apc_amount.zh', 'apc_amount.en'}]
    values = {**dotenv_values(PROJECT / '.env', encoding='utf-8-sig', interpolate=False), **os.environ}
    async with httpx.AsyncClient(trust_env=False) as client:
        model = RoleModelRouter(values, client=client).for_role('general')
        manifest = initialize_run(args.output, records, model)
        collected = []
        semaphore = asyncio.Semaphore(2)
        async def one(record):
            async with semaphore:
                result = await collect_answer(record, model)
                with (args.output / 'answers.jsonl').open('a', encoding='utf-8') as target:
                    target.write(json.dumps(result, ensure_ascii=False) + '\n')
                collected.append(result)
                print(json.dumps({'sample_id': result['sample_id'], 'status': result['generation_status'],
                                  'completed': len(collected), 'total': len(records)}, ensure_ascii=False), flush=True)
        await asyncio.gather(*(one(r) for r in records))
        manifest.update(complete=True, answers_sha256=file_hash(args.output / 'answers.jsonl'))
        (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        load_generation_run(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if not args.live:
        parser.error('--live explicitly enables paid Qwen generation; JEV is not used')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
