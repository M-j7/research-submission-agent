"""Real SQLite, snapshots and prompt assembly; no remote services."""
import asyncio
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.runtime'), 'runtime subsystem not implemented')
        self.api = importlib.import_module('submission_assistant.runtime.memory')
        self.contracts = importlib.import_module('submission_assistant.runtime.contracts')
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = self.api.MemoryManager(Path(self.tmp.name) / 'memory.sqlite3')
        self.req = self.contracts.Request(user_id='u', conv_id='c', turn_id='t', message='问题')

    def tearDown(self):
        if hasattr(self, 'memory'):
            self.memory.close()
        if hasattr(self, 'tmp'):
            self.tmp.cleanup()

    def test_completion_is_idempotent_and_only_final_answers_enter_five_rounds(self):
        for i in range(7):
            req = self.req.model_copy(update={'turn_id': f't{i}', 'message': f'问{i}'})
            self.memory.begin_turn(req)
            self.memory.create_run(req, f'r{i}', 'fees', ['C07'])
            self.memory.publish_result(req, f'r{i}', {'answer_text': '私有角色答复'})
            self.memory.finish_turn(req, f'答{i}')
            self.memory.finish_turn(req, f'答{i}')
        public = self.memory.public_snapshot(self.req)
        self.assertEqual(self.memory.completed_count('u'), 7)
        self.assertEqual([x['content'] for x in public['recent']], ['问2','答2','问3','答3','问4','答4','问5','答5','问6','答6'])
        self.assertEqual([x['content'] for x in public['older']], ['问0','答0','问1','答1'])
        self.assertNotIn('私有角色答复', str(public['recent']))
        with self.assertRaises(ValueError):
            self.memory.finish_turn(req, '不一致的第二份答案')

    def test_scope_isolation_and_turn_collision_are_rejected(self):
        self.memory.begin_turn(self.req)
        self.memory.create_run(self.req, 'run', 'fees', [])
        self.memory.append(self.req, 'run', 'assistant', {'content': 'secret'}, 1)
        other = self.req.model_copy(update={'user_id':'other'})
        self.assertEqual(self.memory.query(other, scope='conversation'), [])
        with self.assertRaises(PermissionError):
            self.memory.instant_snapshot(other, 'run')
        with self.assertRaises(ValueError):
            self.memory.begin_turn(self.req.model_copy(update={'message':'覆盖问题'}))

    def test_compression_preserves_raw_pairing_and_append_during_async_work(self):
        async def exercise():
            ready, release = asyncio.Event(), asyncio.Event()
            async def compressor(records):
                ready.set()
                await release.wait()
                return {r['record_id']: '旧结果摘要' for r in records}
            self.memory.compressor = compressor
            self.memory.begin_turn(self.req)
            self.memory.create_run(self.req, 'run', 'fees', [])
            for step in range(1,4):
                self.memory.append(self.req, 'run', 'assistant', {'content':'','tool_calls':[{'id':f'c{step}','type':'function','function':{'name':'read','arguments':'{}'}}]}, step)
                self.memory.append(self.req, 'run', 'tool', {'tool_call_id':f'c{step}','content':'原始结果'*200}, step)
            task = self.memory.trigger_compression(self.req, 'run', input_tokens=80, input_budget=100)
            self.assertIs(task, self.memory.trigger_compression(self.req, 'run', input_tokens=90, input_budget=100))
            await ready.wait()
            self.memory.append(self.req, 'run', 'assistant', {'content':'新增'}, 4)
            release.set()
            self.assertTrue(await task)
            snap = self.memory.instant_snapshot(self.req, 'run')
            self.assertEqual(snap['messages'][1]['content'], '旧结果摘要')
            self.assertEqual(snap['messages'][3]['content'], '原始结果'*200)
            self.assertEqual(snap['messages'][-1]['content'], '新增')
            raw = self.memory.query(self.req, scope='instant', run_id='run', limit=20)
            self.assertTrue(any('原始结果'*200 in r['content'] for r in raw))
            self.assertEqual(snap['messages'][0]['tool_calls'][0]['id'], 'c1')
            await self.memory.seal_run(self.req, 'run', 'completed')
        asyncio.run(exercise())

    def test_short_summary_cas_rejects_stale_and_keeps_sources(self):
        for i in range(6):
            req = self.req.model_copy(update={'turn_id':str(i)})
            self.memory.begin_turn(req)
            self.memory.finish_turn(req, '回答')
        snap = self.memory.public_snapshot(self.req)
        self.assertTrue(self.memory.commit_short_summary(self.req, snap, '历史条件摘要'))
        self.assertFalse(self.memory.commit_short_summary(self.req, snap, '迟到覆盖'))
        new = self.memory.public_snapshot(self.req)
        self.assertEqual(new['summary'], '历史条件摘要')
        self.assertEqual(len(new['older']), 0)
        self.assertEqual(len(self.memory.query(self.req, scope='conversation', limit=20)), 12)

    def test_profile_changes_require_user_sources_and_revision(self):
        self.memory.begin_turn(self.req)
        self.memory.finish_turn(self.req, '回答')
        source = self.memory.query(self.req, scope='conversation', keyword='问题')[0]['record_id']
        change = {'key':'response_style','value':'简短','operation':'add','source_ids':[source]}
        self.assertTrue(self.memory.commit_profile('u', [change], base_revision=0, cutoff=1))
        self.assertFalse(self.memory.commit_profile('u', [change], base_revision=0, cutoff=1))
        self.assertEqual(self.memory.public_snapshot(self.req)['profile']['response_style'], '简短')
        assistant = self.memory.query(self.req, scope='conversation', keyword='回答')[0]['record_id']
        with self.assertRaises(ValueError):
            self.memory.commit_profile('u', [{**change,'source_ids':[assistant]}], base_revision=1, cutoff=1)


class PromptTests(unittest.TestCase):
    def test_prefix_static_status_last_and_budget_fails_closed(self):
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.runtime'), 'runtime subsystem not implemented')
        api = importlib.import_module('submission_assistant.runtime.prompt')
        prompt = api.PromptManager(input_budget=1000)
        public = {'profile':{},'summary':'旧摘要','recent':[{'role':'user','content':'历史'}], 'older':[]}
        private = {'message':'本轮','intent_ids':['C07'],'messages':[], 'status':{'current_step':1}}
        one = prompt.build(public, private, system_prompt='角色', tools=[], skills=[])
        two = prompt.build(public, {**private, 'message':'新问题'}, system_prompt='角色', tools=[], skills=[])
        self.assertEqual(one.messages[0], two.messages[0])
        self.assertIn('状态栏', one.messages[-1]['content'])
        self.assertEqual(sum('本轮' in str(x) for x in one.messages), 1)
        with self.assertRaises(api.PromptBudgetError):
            api.PromptManager(input_budget=10).build(public, private, system_prompt='角色'*100, tools=[], skills=[])
