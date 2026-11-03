"""Offline regressions for bounded tool output, without recursive artifacts."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from submission_assistant.runtime.contracts import Request
from submission_assistant.runtime.loop import AgentLoop
from submission_assistant.runtime.memory import MemoryManager, dumps
from submission_assistant.runtime.prompt import PromptManager
from submission_assistant.runtime.tools import ToolRegistry
from submission_assistant.runtime.workspace import WorkspaceManager


def tool_call(identifier, name, args):
    return {'id': identifier, 'type': 'function',
            'function': {'name': name, 'arguments': dumps(args)}}


def reply(*calls):
    return {'message': {'role': 'assistant', 'content': '' if calls else '已获得证据',
                        **({'tool_calls': list(calls)} if calls else {})},
            'finish_reason': 'tool_calls' if calls else 'stop'}


class ToolPresentationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.req = Request(user_id='u', conv_id='c', turn_id='t', message='费用')
        self.memory = MemoryManager(self.root / 'memory.sqlite3')
        self.memory.begin_turn(self.req)
        self.memory.create_run(self.req, 'r', 'fees', [])
        self.workspace = WorkspaceManager(self.root, self.req, 'r')
        self.tools = ToolRegistry(self.workspace, self.memory, self.req, 'r')
        self.prompt = PromptManager(input_budget=20000)

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def run_loop(self, model, allowed):
        prepared = {'agent_type': 'fees', 'system_prompt': '费用咨询',
                    'registered_tools': allowed, 'input': {'intent_ids': ['fees']}}
        loop = AgentLoop(self.memory, self.prompt, model, self.tools)
        return asyncio.run(loop.run(self.req, 'r', prepared, self.memory.public_snapshot(self.req)))

    def test_oversized_read_returns_text_and_never_creates_another_artifact(self):
        ref = '/private/scratch/long.txt'
        original = '费用证据：2160，税费另计。' + '中文\\"' * 4000
        self.workspace.resolve(ref).write_text(original, encoding='utf-8')
        class Model:
            def __init__(self): self.inputs = []
            async def complete(inner, messages, tools, **kwargs):
                inner.inputs.append(messages)
                return reply(tool_call('read-1', 'read', {'ref': ref})) if len(inner.inputs) == 1 else reply()
        model = Model()
        outcome = self.run_loop(model, ['read'])
        self.assertEqual(outcome.execution_status, 'completed')
        tool = next(m for m in model.inputs[1] if m['role'] == 'tool')
        view = json.loads(tool['content'])
        self.assertTrue(view['data'].get('text'), 'read must show content, not another file wrapper')
        self.assertEqual(view['data']['resource_ref'], ref)
        self.assertTrue(original.startswith(view['data']['text']))
        self.assertTrue(view['partial'])
        self.assertTrue(view['next_cursor'])
        self.assertEqual(list(self.workspace.blobs.glob('*.json')), [])
        self.assertLessEqual(self.prompt.count(tool['content']), 12000)

    def test_large_search_then_read_keeps_one_raw_file_and_shows_usable_text(self):
        class Rag:
            async def search(inner, query, top_k):
                return {'evidence': [{'evidence_id': 'fees', 'text': '金额为2160。' + '政策正文' * 6000,
                                      'source_url': 'https://example.test/fees'}]}
        self.tools.rag = Rag()
        class Model:
            def __init__(self): self.inputs = []; self.ref = None
            async def complete(inner, messages, tools, **kwargs):
                inner.inputs.append(messages)
                if len(inner.inputs) == 1:
                    return reply(tool_call('search', 'search_knowledge', {'query': '费用'}))
                if len(inner.inputs) == 2:
                    view = json.loads(next(m for m in reversed(messages) if m['role'] == 'tool')['content'])
                    inner.ref = view['data']['resource_ref']
                    args = {'ref': inner.ref}
                    if view['next_cursor']: args['cursor'] = view['next_cursor']
                    return reply(tool_call('read', 'read', args))
                return reply()
        model = Model()
        outcome = self.run_loop(model, ['search_knowledge', 'read'])
        self.assertEqual(outcome.execution_status, 'completed')
        views = [json.loads(m['content']) for m in model.inputs[-1] if m['role'] == 'tool']
        self.assertIn('金额为2160', views[0]['data'].get('text', ''))
        self.assertTrue(views[1]['data'].get('text'), 'following a raw resource must expose its contents')
        self.assertEqual(views[1]['data']['resource_ref'], model.ref)
        self.assertEqual(len(list(self.workspace.blobs.glob('*.json'))), 1)
        self.assertEqual(outcome.tool_calls_used, 2)
        self.assertEqual(outcome.evidence[0]['evidence_id'], 'fees')

    def test_budgeted_continuation_is_lossless_for_unicode_quotes_and_newlines(self):
        ref = '/private/scratch/escaped.txt'
        original = ('甲乙😀\\"\t\n' * 850) + '结尾'
        self.workspace.resolve(ref).write_text(original, encoding='utf-8')
        async def exercise():
            args = {'ref': ref, 'max_lines': 500}
            pieces = []
            offset = 0
            for index in range(100):
                raw = await self.tools.execute('read', dumps(args), ['read'])
                # The continuation must also survive a change of available budget.
                limit = 1100 if index % 2 else 1500
                view, _ = self.tools.present('read', dumps(args), raw,
                                             limit=limit, count=self.prompt.count)
                self.assertIsNone(view.error)
                self.assertLessEqual(self.prompt.count(dumps(view.model_dump())), limit)
                self.assertEqual(view.data['resource_ref'], ref)
                self.assertEqual(view.data['character_start'], offset)
                self.assertEqual(view.data['line_start'], original[:offset].count('\n') + 1)
                self.assertTrue(view.data['text'])
                pieces.append(view.data['text'])
                offset += len(pieces[-1])
                self.assertEqual(view.data['character_end'], offset)
                self.assertEqual(view.data['line_end'], original[:offset].count('\n') + 1)
                if not view.next_cursor:
                    self.assertFalse(view.partial)
                    break
                self.assertTrue(view.partial)
                args = {'ref': ref, 'max_lines': 500, 'cursor': view.next_cursor}
            else:
                self.fail('cursor did not reach the end of the original resource')
            self.assertEqual(''.join(pieces), original)
        asyncio.run(exercise())
        self.assertEqual(list(self.workspace.blobs.glob('*.json')), [])

    def test_budgeted_cursor_still_rejects_wrong_resource_and_changed_file(self):
        ref = '/private/scratch/versioned.txt'
        self.workspace.resolve(ref).write_text('原文' * 8000, encoding='utf-8')
        async def exercise():
            raw = await self.tools.execute('read', dumps({'ref': ref}), ['read'])
            view, _ = self.tools.present('read', dumps({'ref': ref}), raw,
                                         limit=1000, count=self.prompt.count)
            self.assertTrue(view.next_cursor)
            wrong = await self.tools.execute('read', dumps({'ref': '/private/scratch/other.txt',
                                                           'cursor': view.next_cursor}), ['read'])
            self.assertEqual(wrong.error.code, 'STALE_CURSOR')
            self.workspace.resolve(ref).write_text('changed', encoding='utf-8')
            stale = await self.tools.execute('read', dumps({'ref': ref, 'cursor': view.next_cursor}), ['read'])
            self.assertEqual(stale.error.code, 'STALE_CURSOR')
        asyncio.run(exercise())

    def test_tiny_budget_stops_cleanly_instead_of_creating_zero_length_cursor(self):
        ref = '/private/scratch/long.txt'
        self.workspace.resolve(ref).write_text('原文' * 8000, encoding='utf-8')
        async def exercise():
            raw = await self.tools.execute('read', dumps({'ref': ref}), ['read'])
            before = set(self.tools._cursors)
            view, _ = self.tools.present('read', dumps({'ref': ref}), raw,
                                         limit=400, count=self.prompt.count)
            self.assertEqual(view.error.code, 'PRESENTATION_BUDGET')
            self.assertIsNone(view.next_cursor)
            self.assertEqual(set(self.tools._cursors), before)
        asyncio.run(exercise())
        self.assertEqual(list(self.workspace.blobs.glob('*.json')), [])

    def test_budget_exhaustion_pairs_all_calls_and_does_not_externalize_read_errors(self):
        ref = '/private/scratch/long.txt'
        self.workspace.resolve(ref).write_text('正文' * 10000, encoding='utf-8')
        class Model:
            def __init__(self): self.inputs = []
            async def complete(inner, messages, tools, **kwargs):
                inner.inputs.append(messages)
                out = reply(tool_call('large', 'read', {'ref': ref}),
                            tool_call('missing', 'read', {'ref': '/private/scratch/missing.txt'}))
                out['message']['content'] = '填' * 6500
                return out
        model = Model()
        outcome = self.run_loop(model, ['read'])
        self.assertEqual(outcome.stop_reason, 'input_budget')
        self.assertEqual(len(model.inputs), 1, 'no over-budget prompt may reach the model')
        messages = self.memory.instant_snapshot(self.req, 'r')['messages']
        paired = [m['tool_call_id'] for m in messages if m['role'] == 'tool']
        self.assertEqual(paired, ['large', 'missing'])
        self.assertEqual(list(self.workspace.blobs.glob('*.json')), [])

    def test_externalized_json_is_losslessly_read_and_original_search_cursor_is_preserved(self):
        class Rag:
            async def search(inner, query, top_k):
                return {'evidence': [{'evidence_id': 'first', 'text': '第一份\\"\n' * 100},
                                     {'evidence_id': 'second', 'text': '第二份'}]}
        self.tools.rag = Rag()
        self.tools.max_chars = 10000
        async def exercise():
            args = {'query': '费用', 'top_k': 1}
            raw = await self.tools.execute('search_knowledge', dumps(args), ['search_knowledge'])
            view, ref = self.tools.present('search_knowledge', dumps(args), raw,
                                           limit=1700, count=self.prompt.count)
            self.assertIsNone(view.error)
            self.assertEqual(view.data['continuation_tool'], 'read')
            self.assertEqual(view.data['original_next_cursor'], raw.next_cursor)
            pieces = [view.data['text']]
            end = view.data['character_end']
            while view.next_cursor:
                read_args = {'ref': ref, 'cursor': view.next_cursor}
                raw_read = await self.tools.execute('read', dumps(read_args), ['read'])
                view, _ = self.tools.present('read', dumps(read_args), raw_read,
                                             limit=1000, count=self.prompt.count)
                self.assertIsNone(view.error)
                self.assertLessEqual(self.prompt.count(dumps(view.model_dump())), 1000)
                self.assertEqual(view.data['character_start'], end)
                end = view.data['character_end']
                pieces.append(view.data['text'])
            self.assertEqual(json.loads(''.join(pieces)), raw.model_dump())
            following = await self.tools.execute('search_knowledge', dumps({**args, 'cursor': raw.next_cursor}),
                                                  ['search_knowledge'])
            self.assertEqual(following.data['evidence'][0]['evidence_id'], 'second')
        asyncio.run(exercise())
        self.assertEqual(len(list(self.workspace.blobs.glob('*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
