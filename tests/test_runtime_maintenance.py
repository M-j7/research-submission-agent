import asyncio
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.runtime.maintenance'),'memory maintenance missing')

    def test_short_summary_and_profile_extraction_use_public_raw_sources(self):
        from submission_assistant.runtime.maintenance import MemoryMaintenance
        from submission_assistant.runtime.memory import MemoryManager
        from submission_assistant.runtime.contracts import Request
        calls=[]
        class Ops:
            async def short_summary(self,rows): calls.append(('summary',rows)); return '更早条件'
            async def profile_changes(self,rows,profile):
                calls.append(('profile',rows))
                return [{'key':'response_style','value':'简短','operation':'add','source_ids':[rows[0]['record_id']]}]
        async def exercise(memory):
            req=Request(user_id='u',conv_id='c',turn_id='t',message='以后简短')
            for i in range(6):
                req=req.model_copy(update={'turn_id':str(i)})
                memory.begin_turn(req); memory.finish_turn(req,'回答')
            maintenance=MemoryMaintenance(memory,Ops(),profile_every=6)
            await maintenance.update(req)
            await maintenance.update(req)
            public=memory.public_snapshot(req)
            self.assertEqual(public['summary'],'更早条件')
            self.assertEqual(public['profile']['response_style'],'简短')
            self.assertEqual([x[0] for x in calls],['summary','profile'])
            self.assertTrue(all(r['role']=='user' for r in calls[1][1]))
        with tempfile.TemporaryDirectory() as directory:
            memory=MemoryManager(Path(directory)/'m.sqlite3')
            asyncio.run(exercise(memory)); memory.close()

    def test_profile_conflict_is_not_injected_as_an_active_fact(self):
        from submission_assistant.runtime.memory import MemoryManager
        from submission_assistant.runtime.contracts import Request
        with tempfile.TemporaryDirectory() as directory:
            memory=MemoryManager(Path(directory)/'m.sqlite3')
            req=Request(user_id='u',conv_id='c',turn_id='t',message='以后简短')
            memory.begin_turn(req); memory.finish_turn(req,'好')
            source=next(r['record_id'] for r in memory.query(req,limit=20) if r['role']=='user')
            memory.commit_profile('u',[{'key':'response_style','value':'简短','operation':'add','source_ids':[source]}],base_revision=0,cutoff=1)
            memory.commit_profile('u',[{'key':'response_style','value':'详细','operation':'conflict','source_ids':[source]}],base_revision=1,cutoff=1)
            self.assertNotIn('response_style',memory.public_snapshot(req)['profile'])
            memory.close()

    def test_oversize_memory_query_is_paginated_without_losing_record_tail(self):
        from submission_assistant.runtime.memory import MemoryManager
        from submission_assistant.runtime.contracts import Request
        from submission_assistant.runtime.workspace import WorkspaceManager
        from submission_assistant.runtime.tools import ToolRegistry
        import json
        async def exercise(memory,directory):
            req=Request(user_id='u',conv_id='c',turn_id='t',message='a'*350)
            memory.begin_turn(req); memory.create_run(req,'r','general',[])
            registry=ToolRegistry(WorkspaceManager(Path(directory),req,'r'),memory,req,'r',max_chars=80)
            args={'scope':'conversation'}
            result=await registry.execute('query_memory',json.dumps(args),['query_memory'])
            text=result.data['records'][0]['content']
            self.assertTrue(result.partial)
            while result.next_cursor:
                result=await registry.execute('query_memory',json.dumps({**args,'cursor':result.next_cursor}),['query_memory'])
                text+=''.join(r['content'] for r in result.data['records'])
            self.assertEqual(text,'a'*350)
        with tempfile.TemporaryDirectory() as directory:
            memory=MemoryManager(Path(directory)/'m.sqlite3')
            asyncio.run(exercise(memory,directory)); memory.close()
