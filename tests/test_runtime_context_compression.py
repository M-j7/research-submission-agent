"""Real local memory/context regressions; model calls are controlled fixtures."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from submission_assistant.runtime.contracts import Request
from submission_assistant.runtime.memory import MemoryManager
from submission_assistant.runtime.prompt import PromptManager


class ContextCompressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.memory = MemoryManager(self.root / 'memory.sqlite3')
        self.req = Request(user_id='u', conv_id='c', turn_id='t', message='费用')
        self.memory.begin_turn(self.req)
        self.memory.create_run(self.req, 'r', 'general', [])

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def groups(self, bodies):
        for step, body in enumerate(bodies, 1):
            self.memory.append(self.req, 'r', 'assistant', {'content': '', 'tool_calls': [
                {'id': f'c{step}', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]}, step)
            self.memory.append(self.req, 'r', 'tool', {'tool_call_id': f'c{step}', 'content': body}, step)

    def test_compression_batches_make_progress_without_recompressing_committed_records(self):
        self.groups(['正文' * 100] * 4 + ['最近1', '最近2'])
        seen = []
        class Compressor:
            max_input_bytes = 900
            async def __call__(inner, records):
                if len(json.dumps(records, ensure_ascii=False).encode()) > 900:
                    raise ValueError('batch exceeds the real summarizer input bound')
                seen.extend(r['record_id'] for r in records)
                return {r['record_id']: '已压缩' for r in records}
        self.memory.compressor = Compressor()
        async def exercise():
            for _ in range(4):
                task = self.memory.trigger_compression(self.req, 'r', input_tokens=800, input_budget=1000)
                self.assertIsNotNone(task)
                self.assertTrue(await task)
            self.assertIsNone(self.memory.trigger_compression(self.req, 'r', input_tokens=800, input_budget=1000))
        asyncio.run(exercise())
        self.assertEqual(len(seen), 4)
        self.assertEqual(len(set(seen)), 4)
        texts = [r['content'] for r in self.memory.instant_snapshot(self.req, 'r')['messages'] if r['role'] == 'tool']
        self.assertEqual(texts, ['已压缩'] * 4 + ['最近1', '最近2'])
        raw = self.memory.db.execute("SELECT body FROM records WHERE kind='execution' AND role='tool'").fetchall()
        self.assertIn('正文' * 100, json.loads(raw[0]['body'])['content'])
        self.assertEqual(self.memory.public_snapshot(self.req)['profile'], {})

    def test_version_conflict_does_not_permanently_suppress_a_new_candidate(self):
        self.groups(['原文' * 100, '最近1', '最近2'])
        async def compress(rows):
            return {r['record_id']: '摘要' for r in rows}
        self.memory.compressor = compress
        async def exercise():
            first = self.memory.trigger_compression(self.req, 'r', input_tokens=800, input_budget=1000)
            with self.memory.db:
                self.memory.db.execute('UPDATE runs SET projection_rev=1 WHERE run=?', ('r',))
            self.assertFalse(await first)
            second = self.memory.trigger_compression(self.req, 'r', input_tokens=800, input_budget=1000)
            self.assertIsNotNone(second, 'a changed revision must allow a fresh candidate')
            self.assertTrue(await second)
        asyncio.run(exercise())

    def test_compression_failures_have_a_persisted_reason_without_replacing_raw(self):
        self.groups(['原文' * 100, '最近1', '最近2'])
        async def compress(rows): raise ValueError('private provider detail')
        self.memory.compressor = compress
        async def exercise():
            task = self.memory.trigger_compression(self.req, 'r', input_tokens=800, input_budget=1000)
            self.assertFalse(await task)
        asyncio.run(exercise())
        events = [json.loads(r['body']) for r in self.memory.db.execute("SELECT body FROM records WHERE kind='compression'")]
        self.assertTrue(any(e.get('reason') == 'compressor_error' for e in events))
        self.assertNotIn('private provider detail', str(events))
        self.assertEqual(self.memory.instant_snapshot(self.req, 'r')['messages'][1]['content'], '原文' * 100)

    def test_over_budget_loop_waits_for_the_compressor_deadline_not_ten_seconds(self):
        from submission_assistant.runtime.loop import AgentLoop
        from submission_assistant.runtime.tools import ToolRegistry
        from submission_assistant.runtime.workspace import WorkspaceManager
        self.groups(['旧结果' * 2000, '最近1', '最近2'])
        async def compress(rows):
            await asyncio.sleep(10.2)
            return {r['record_id']: '旧工具结果摘要' for r in rows}
        self.memory.compressor = compress
        self.memory.compression_timeout_s = 12
        prompt = PromptManager(input_budget=30000)
        public = self.memory.public_snapshot(self.req)
        initial = prompt.assemble(public, self.memory.instant_snapshot(self.req, 'r'), system_prompt='客服', tools=[], skills=[])
        prompt.input_budget = initial.input_tokens - 4000
        class Model:
            async def complete(inner, *args, **kwargs):
                return {'message': {'role': 'assistant', 'content': '根据摘要继续回答'}, 'finish_reason': 'stop'}
        tools = ToolRegistry(WorkspaceManager(self.root, self.req, 'r'), self.memory, self.req, 'r')
        prepared = {'agent_type': 'general', 'system_prompt': '客服', 'registered_tools': [], 'input': {'intent_ids': []}}
        result = asyncio.run(AgentLoop(self.memory, prompt, Model(), tools).run(self.req, 'r', prepared, public))
        self.assertEqual(result.execution_status, 'completed')
        self.assertEqual(result.answer_text, '根据摘要继续回答')

    def test_live_context_uses_seventy_percent_and_local_token_counting(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from submission_assistant.runtime.context import build_runtime_config
        tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, '你好': 1}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        path = self.root / 'tokenizer.json'
        tokenizer.save(str(path))
        config = build_runtime_config({'LLM_MODEL': 'qwen3.8-flash', 'AGENT_TOKENIZER_PATH': str(path)}, self.root)
        self.assertEqual(config.prompt.input_budget, 700000)
        self.assertEqual(config.prompt.compression_trigger_tokens, 600000)
        self.assertIsNotNone(config.prompt.token_counter)
        self.assertLess(config.prompt.count('你好 ' * 1000), 2000)
        self.assertEqual(config.metadata['context_window_tokens'], 1000000)
        self.assertIn('estimate', config.metadata['counting_mode'])
        self.assertGreater(config.settings.max_total_tokens_estimate, 700000)
        with self.assertRaises((ValueError, FileNotFoundError)):
            build_runtime_config({'LLM_MODEL': 'unverified-model', 'AGENT_TOKENIZER_PATH': str(path)}, self.root)

    def test_oversized_single_record_is_split_without_losing_unicode_tail(self):
        from submission_assistant.runtime.maintenance import ModelMemoryOps
        source = '正文😀\\"\n' * 300
        observed = []
        class Model:
            async def complete(inner, messages, tools, **kwargs):
                self.assertLessEqual(len(messages[1]['content'].encode()), 900)
                rows = json.loads(messages[1]['content'])
                observed.extend(rows)
                return {'message': {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'id': 'summary', 'type': 'function', 'function': {'name': 'memory_candidate',
                     'arguments': json.dumps({'summaries': [{'record_id': r['record_id'], 'content': '事实摘要'} for r in rows]})}}]},
                    'finish_reason': 'tool_calls'}
        result = asyncio.run(ModelMemoryOps(Model(), max_input_bytes=900)([{'record_id': 'record-1', 'content': source}]))
        self.assertEqual(''.join(r['content'] for r in observed), source)
        self.assertGreater(len(observed), 1)
        self.assertEqual(set(result), {'record-1'})
        self.assertTrue(result['record-1'])


if __name__ == '__main__':
    unittest.main()
