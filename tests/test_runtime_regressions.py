import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from submission_assistant.runtime.contracts import Request
from submission_assistant.runtime.memory import MemoryManager
from submission_assistant.runtime.prompt import PromptManager
from submission_assistant.runtime.loop import AgentLoop, LoopSettings
from submission_assistant.runtime.tools import ToolRegistry
from submission_assistant.runtime.workspace import WorkspaceManager


class RuntimeRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.memory = MemoryManager(self.root / 'm.sqlite3')
        self.req = Request(user_id='u', conv_id='c', turn_id='early', message='你好')
        self.memory.begin_turn(self.req)

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def registry(self, max_chars=12000):
        self.memory.create_run(self.req, 'r', 'general', [])
        ws = WorkspaceManager(self.root, self.req, 'r')
        return ToolRegistry(ws, self.memory, self.req, 'r', max_chars=max_chars)

    def test_profile_batch_uses_completion_not_receipt_order(self):
        later = self.req.model_copy(update={'turn_id': 'later'})
        self.memory.begin_turn(later)
        self.memory.finish_turn(later, 'later answer')
        self.assertTrue(self.memory.commit_profile('u', [], base_revision=0, cutoff=1))
        self.memory.finish_turn(self.req, 'early answer')
        self.assertEqual([r['turn_id'] for r in self.memory.profile_batch('u')['rows']], ['early'])

    def test_profile_and_summary_cas_are_atomic_across_connections(self):
        self.memory.finish_turn(self.req, 'answer')
        other = MemoryManager(self.root / 'm.sqlite3')
        try:
            for kind in ('profile', 'summary'):
                barrier = threading.Barrier(2)
                start = threading.Barrier(2)
                class Proxy:
                    def __init__(self, db): self.db = db
                    def __getattr__(self, name): return getattr(self.db, name)
                    def __enter__(self): self.db.__enter__(); return self
                    def __exit__(self, *args): return self.db.__exit__(*args)
                    def execute(self, sql, args=()):
                        result = self.db.execute(sql, args)
                        if sql.startswith('SELECT') and not self.db.in_transaction and ('FROM users' in sql or 'FROM summaries' in sql):
                            barrier.wait(timeout=2)
                        return result
                originals = [self.memory.db, other.db]
                managers = [self.memory, other]
                for manager, db in zip(managers, originals): manager.db = Proxy(db)
                results, errors = [], []
                def commit(manager):
                    try:
                        start.wait(timeout=2)
                        if kind == 'profile': results.append(manager.commit_profile('u', [], base_revision=0, cutoff=1))
                        else: results.append(manager.commit_short_summary(self.req, {'summary_revision':0, 'older_source_ids':['source']}, '摘要'))
                    except Exception as exc: errors.append(exc)
                threads = [threading.Thread(target=commit, args=(manager,)) for manager in managers]
                for thread in threads: thread.start()
                for thread in threads: thread.join(timeout=5)
                for manager, db in zip(managers, originals): manager.db = db
                self.assertEqual(errors, [])
                self.assertEqual(sorted(results), [False, True], kind)
        finally: other.close()

    def test_ranged_read_cursor_can_continue_without_start_line(self):
        tools = self.registry(max_chars=5)
        tools.workspace.resolve('/private/scratch').joinpath('a.txt').write_text('ignore\nabcdefghijk\n', encoding='utf-8')
        async def run():
            result = await tools.execute('read', json.dumps({'ref':'/private/scratch/a.txt', 'start_line':2}), ['read'])
            text = result.data['text']
            while result.next_cursor:
                result = await tools.execute('read', json.dumps({'ref':'/private/scratch/a.txt', 'cursor':result.next_cursor}), ['read'])
                self.assertIsNone(result.error)
                text += result.data['text']
            self.assertEqual(text, 'abcdefghijk\n')
        asyncio.run(run())

    def test_regex_scan_yields_to_external_deadline(self):
        tools = self.registry()
        tools.workspace.resolve('/private/scratch').joinpath('a.txt').write_text(('a'*22+'!\n')*100, encoding='utf-8')
        async def run():
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(tools.execute('grep', json.dumps({'root':'/private/scratch','pattern':'(a|aa)+$', 'mode':'regex'}), ['grep']), .01)
            self.assertLess(time.monotonic()-started, .5)
        asyncio.run(run())

    def run_model(self, content, *, collaboration=None, limit=100000):
        tools = self.registry()
        class Model:
            async def complete(self, *args, **kwargs):
                return {'message':{'role':'assistant','content':content}, 'finish_reason':'stop'}
        prepared = {'agent_type':'composer', 'system_prompt':'汇总', 'registered_tools':[], 'input':{'intent_ids':[]}}
        loop = AgentLoop(self.memory, PromptManager(input_budget=20000), Model(), tools, settings=LoopSettings(max_total_tokens_estimate=limit))
        return asyncio.run(loop.run(self.req, 'r', prepared, self.memory.public_snapshot(self.req), collaboration_results=collaboration))

    def test_composer_preserves_evidence_identity_and_failed_coverage(self):
        result = self.run_model('汇总', collaboration=[
            {'evidence':[{'evidence_id':'p1','text':'来源', 'score':.7}], 'execution_status':'completed','unresolved':[]},
            {'evidence':[{'evidence_id':'p1','text':'来源', 'score':.8}], 'execution_status':'failed','unresolved':['费用未核验']},
        ])
        self.assertEqual([e['evidence_id'] for e in result.evidence], ['p1'])
        self.assertEqual(result.execution_status, 'incomplete')
        self.assertIn('费用未核验', result.unresolved)

    def test_final_response_cannot_bypass_token_budget(self):
        result = self.run_model('a'*10000, limit=6000)
        self.assertEqual(result.execution_status, 'incomplete')
        self.assertEqual(result.stop_reason, 'token_limit')

    def test_profile_cannot_cite_sources_after_extraction_cutoff(self):
        self.memory.finish_turn(self.req, 'answer')
        later = self.req.model_copy(update={'turn_id':'later', 'message':'以后详细'})
        self.memory.begin_turn(later)
        self.memory.finish_turn(later, 'answer')
        source = next(r['record_id'] for r in self.memory.query(later, limit=20) if r['turn_id']=='later' and r['role']=='user')
        with self.assertRaises(ValueError):
            self.memory.commit_profile('u', [{'key':'response_style','value':'详细','operation':'add','source_ids':[source]}], base_revision=0, cutoff=1)

    def test_terminal_status_contains_final_tool_counter(self):
        tools = self.registry()
        class Model:
            async def complete(self, *args, **kwargs):
                calls=[{'id':str(i),'type':'function','function':{'name':'unknown','arguments':'{}'}} for i in range(3)]
                return {'message':{'role':'assistant','content':'','tool_calls':calls}, 'finish_reason':'tool_calls'}
        prepared={'agent_type':'general','system_prompt':'客服','registered_tools':['read'],'input':{'intent_ids':[]}}
        loop=AgentLoop(self.memory, PromptManager(input_budget=20000), Model(), tools)
        result=asyncio.run(loop.run(self.req,'r',prepared,self.memory.public_snapshot(self.req)))
        status=self.memory.instant_snapshot(self.req,'r')['status']
        self.assertEqual(status['tool_calls_used'],result.tool_calls_used)
        self.assertEqual(status['phase'],'finished')

    def test_compression_checks_revision_inside_write_transaction(self):
        self.registry()
        for step in range(3):
            self.memory.append(self.req,'r','assistant',{'content':'','tool_calls':[{'id':str(step)}]},step)
            self.memory.append(self.req,'r','tool',{'tool_call_id':str(step),'content':'原文'*100},step)
        original=self.memory.db
        checking=False
        class Proxy:
            def __getattr__(self,name): return getattr(original,name)
            def __enter__(self): original.__enter__(); return self
            def __exit__(self,*args): return original.__exit__(*args)
            def execute(self,sql,args=()):
                if checking and sql.startswith('SELECT') and 'FROM runs' in sql:
                    if not original.in_transaction: raise AssertionError('revision read before write transaction')
                return original.execute(sql,args)
        async def compress(rows):
            nonlocal checking
            checking=True
            return {row['record_id']:'摘要' for row in rows}
        self.memory.db=Proxy()
        self.memory.compressor=compress
        async def run():
            task=self.memory.trigger_compression(self.req,'r',input_tokens=800,input_budget=1000)
            self.assertTrue(await task)
        try: asyncio.run(run())
        finally: self.memory.db=original
