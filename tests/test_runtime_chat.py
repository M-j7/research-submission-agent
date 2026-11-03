import asyncio
import copy
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def response(content='', calls=None, reason='stop'):
    message={'role':'assistant','content':content}
    if calls: message['tool_calls']=calls
    return {'message':message,'finish_reason':reason,'usage':{'prompt_tokens':10,'completion_tokens':5}}


def call(identifier,name,args):
    return {'id':identifier,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}


class SequenceModel:
    def __init__(self, items):
        self.items=list(items)
        self.inputs=[]

    async def complete(self,messages,tools,*,allow_tools=True):
        self.inputs.append((copy.deepcopy(messages),tools,allow_tools))
        return self.items.pop(0)


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.runtime.loop'),'loop missing')
        self.api=importlib.import_module('submission_assistant.runtime.loop')
        from submission_assistant.runtime.contracts import Request
        from submission_assistant.runtime.memory import MemoryManager
        from submission_assistant.runtime.prompt import PromptManager
        from submission_assistant.runtime.workspace import WorkspaceManager
        from submission_assistant.runtime.tools import ToolRegistry
        self.tmp=tempfile.TemporaryDirectory()
        self.req=Request(user_id='u',conv_id='c',turn_id='t',message='你好')
        self.memory=MemoryManager(Path(self.tmp.name)/'memory.sqlite3')
        self.memory.begin_turn(self.req)
        self.memory.create_run(self.req,'r','general',[])
        self.workspace=WorkspaceManager(Path(self.tmp.name),self.req,'r')
        self.tools=ToolRegistry(self.workspace,self.memory,self.req,'r')
        self.prompt=PromptManager(input_budget=20000)
        self.prepared={'agent_type':'general','system_prompt':'客服','registered_tools':['glob','read'],
                       'input':{'intent_ids':[],'guidance':'接待'}}

    def tearDown(self):
        if hasattr(self,'memory'): self.memory.close()
        if hasattr(self,'tmp'): self.tmp.cleanup()

    def run_loop(self, model, **settings):
        loop=self.api.AgentLoop(self.memory,self.prompt,model,self.tools,settings=self.api.LoopSettings(**settings))
        return asyncio.run(loop.run(self.req,'r',self.prepared,self.memory.public_snapshot(self.req)))

    def test_native_calls_pair_with_results_and_plain_json_is_not_executed(self):
        model=SequenceModel([response(calls=[call('a','glob',{'root':'/private/scratch','pattern':'*'})],reason='tool_calls'),response('{"name":"delete_files"}')])
        result=self.run_loop(model)
        self.assertEqual(result.execution_status,'completed')
        self.assertEqual(result.current_step,2)
        self.assertEqual(result.tool_calls_used,1)
        messages=model.inputs[1][0]
        assistant=next(m for m in messages if m.get('tool_calls'))
        self.assertEqual(assistant['tool_calls'][0]['id'],'a')
        tool=next(m for m in messages if m['role']=='tool')
        self.assertEqual(tool['tool_call_id'],'a')
        self.assertEqual(json.loads(tool['content'])['status'],'empty')
        self.assertEqual(result.answer_text,'{"name":"delete_files"}')

    def test_invalid_tools_are_paired_but_never_executed(self):
        model=SequenceModel([response(calls=[call('a','delete_files',{})],reason='tool_calls'),response('无法执行该动作')])
        result=self.run_loop(model)
        tool=next(m for m in model.inputs[1][0] if m['role']=='tool')
        self.assertEqual(json.loads(tool['content'])['error']['code'],'TOOL_NOT_ALLOWED')
        self.assertEqual(result.current_step,2)

    def test_empty_role_permissions_reject_invented_native_calls_before_execution(self):
        self.prepared['registered_tools'] = []
        async def forbidden(*args, **kwargs):
            self.fail('feedback-only role executed an invented tool call')
        self.tools.execute = forbidden
        model = SequenceModel([response(calls=[call('x', 'glob', {'root': '/private/scratch', 'pattern': '*'})], reason='tool_calls'),
                               response('请说明具体事项。')])
        result = self.run_loop(model)
        self.assertEqual(result.tool_calls_used, 0)
        self.assertFalse(model.inputs[0][2])
        self.assertEqual(model.inputs[0][1], [])
        tool = next(m for m in model.inputs[1][0] if m['role'] == 'tool')
        self.assertEqual(json.loads(tool['content'])['error']['code'], 'BUDGET_EXHAUSTED')

    def test_last_step_disables_tools_without_ninth_call(self):
        model=SequenceModel([response(calls=[call('a','glob',{'root':'/private/scratch','pattern':'*'})],reason='tool_calls'),response('目前未发现文件')])
        result=self.run_loop(model,max_steps=2)
        self.assertFalse(model.inputs[-1][2])
        self.assertEqual(result.current_step,2)
        self.assertEqual(len(model.inputs),2)

    def test_truncation_and_empty_response_are_not_success(self):
        for item,reason in [(response('不完整',reason='length'),'truncated_response'),(response(''),'empty_response')]:
            # Fresh run per case keeps immutable traces separate.
            self.memory.create_run(self.req,reason,'general',[])
            from submission_assistant.runtime.tools import ToolRegistry
            tools=ToolRegistry(self.workspace,self.memory,self.req,reason)
            loop=self.api.AgentLoop(self.memory,self.prompt,SequenceModel([item]),tools)
            result=asyncio.run(loop.run(self.req,reason,self.prepared,self.memory.public_snapshot(self.req)))
            self.assertEqual(result.execution_status,'incomplete')
            self.assertEqual(result.stop_reason,reason)

    def test_two_identical_no_progress_results_prevent_third_execution(self):
        model=SequenceModel([response(calls=[call(str(i),'glob',{'root':'/private/scratch','pattern':'*'})],reason='tool_calls') for i in range(3)])
        result=self.run_loop(model,max_steps=5)
        self.assertEqual(result.stop_reason,'no_progress')
        self.assertEqual(result.tool_calls_used,2)
        self.assertEqual(len(self.memory.instant_snapshot(self.req,'r')['messages']),6)

    def test_batch_tools_keep_ids_when_one_fails(self):
        model=SequenceModel([response(calls=[call('missing','read',{'ref':'/system/skills/missing.md'}),call('good','glob',{'root':'/private/scratch','pattern':'*'})],reason='tool_calls'),response('没有找到资料')])
        self.run_loop(model)
        results={m['tool_call_id']:json.loads(m['content']) for m in model.inputs[1][0] if m['role']=='tool'}
        self.assertEqual(results['missing']['status'],'error')
        self.assertEqual(results['good']['status'],'empty')

    def test_parallel_rag_calls_report_actual_running_and_completed_tools(self):
        async def exercise():
            started = [asyncio.Event(), asyncio.Event()]
            release = [asyncio.Event(), asyncio.Event()]
            class Rag:
                async def search(self, query, *, top_k):
                    index = int(query)
                    started[index].set()
                    await release[index].wait()
                    return {'evidence': []}
            self.tools.rag = Rag()
            self.prepared['registered_tools'] = ['search_knowledge']
            model = SequenceModel([response(calls=[call('a','search_knowledge',{'query':'0'}),
                call('b','search_knowledge',{'query':'1'})], reason='tool_calls'), response('已核验')])
            loop = self.api.AgentLoop(self.memory,self.prompt,model,self.tools)
            task = asyncio.create_task(loop.run(self.req,'r',self.prepared,self.memory.public_snapshot(self.req)))
            try:
                await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 2)
                status = self.memory.instant_snapshot(self.req,'r')['status']
                self.assertEqual(status.get('phase'), 'tool_call')
                self.assertEqual(status.get('tools'), [{'name':'search_knowledge','state':'running'},
                                                       {'name':'search_knowledge','state':'running'}])
                release[0].set()
                async def first_finished():
                    while True:
                        current = self.memory.instant_snapshot(self.req,'r')['status']
                        if current.get('tools',[{}])[0].get('state') == 'completed': return current
                        await asyncio.sleep(.001)
                current = await asyncio.wait_for(first_finished(), 1)
                self.assertEqual(current['tools'][1]['state'], 'running')
                release[1].set()
                result = await task
                self.assertEqual(result.execution_status, 'completed')
                final = self.memory.instant_snapshot(self.req,'r')['status']
                self.assertEqual(final['phase'], 'finished')
                self.assertFalse(any(t['state']=='running' for t in final.get('tools',[])))
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        asyncio.run(exercise())

    def test_tool_timeout_is_recorded_without_exposing_arguments(self):
        class Rag:
            async def search(self, query, *, top_k): raise TimeoutError()
        self.tools.rag = Rag()
        self.prepared['registered_tools'] = ['search_knowledge']
        self.run_loop(SequenceModel([response(calls=[call('a','search_knowledge',{'query':'PRIVATE_QUERY'})],
            reason='tool_calls'), response('检索超时，请重试')]))
        with self.memory.transaction() as db:
            statuses = [json.loads(r['body']) for r in db.execute("SELECT body FROM records WHERE kind='status' ORDER BY seq")]
        self.assertTrue(any(s.get('tools')==[{'name':'search_knowledge','state':'timed_out'}] for s in statuses))
        self.assertNotIn('PRIVATE_QUERY', json.dumps(statuses))

    def test_cancel_seals_run_and_propagates(self):
        async def exercise():
            ready=asyncio.Event()
            class Slow:
                async def complete(self,*a,**k):
                    ready.set(); await asyncio.Event().wait()
            loop=self.api.AgentLoop(self.memory,self.prompt,Slow(),self.tools)
            task=asyncio.create_task(loop.run(self.req,'r',self.prepared,self.memory.public_snapshot(self.req)))
            await ready.wait(); task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            with self.assertRaises(ValueError): self.memory.append(self.req,'r','assistant',{'content':'late'},2)
        asyncio.run(exercise())


class ChatTests(unittest.TestCase):
    def test_multirole_composer_reads_formal_results_not_private_trace_and_count_once(self):
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.runtime.chat'),'chat graph missing')
        from submission_assistant.runtime.chat import ChatService, build_chat_graph
        from submission_assistant.runtime.memory import MemoryManager
        from submission_assistant.runtime.contracts import Request
        from test_independent_jev import judgments_fixture
        class Recognizer:
            async def recognize(self,message,history=None):
                return judgments_fixture(scores={'C02':.99,'C07':.99})
        class Model:
            def __init__(self): self.inputs=[]
            async def complete(self,messages,tools,*,allow_tools=True):
                self.inputs.append(messages)
                is_composer='汇总审核' in messages[0]['content']
                return response('合并最终答复' if is_composer else '角色结论')
        with tempfile.TemporaryDirectory() as directory:
            memory=MemoryManager(Path(directory)/'memory.sqlite3')
            model=Model()
            service=ChatService(Path(directory),memory,Recognizer(),model)
            graph=build_chat_graph(service)
            req=Request(user_id='u',conv_id='c',turn_id='t',message='摘要和费用')
            first=asyncio.run(graph.ainvoke({'request':req}))
            again=asyncio.run(graph.ainvoke({'request':req}))
            self.assertEqual(first['answer'],'合并最终答复')
            self.assertEqual(again['answer'],first['answer'])
            self.assertEqual(len(model.inputs),3)
            self.assertEqual(memory.completed_count('u'),1)
            self.assertEqual(len(memory.public_snapshot(req)['recent']),2)
            self.assertIn('collaboration_results',str(model.inputs[-1]))
            self.assertEqual(len(memory.query(req,record_type='agent_result',limit=20)),3)
            memory.close()

    def test_http_provider_keeps_tools_native_and_errors_do_not_leak(self):
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.runtime.provider'),'provider missing')
        import httpx
        from submission_assistant.runtime.provider import OpenAICompatibleModel, ModelServiceError
        async def exercise():
            def handler(req):
                body=json.loads(req.content)
                self.assertEqual(body['tool_choice'],'none')
                return httpx.Response(401,text='secret provider body')
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                model=OpenAICompatibleModel('https://example.test/v1','secret','model',client=client)
                with self.assertRaises(ModelServiceError) as caught:
                    await model.complete([{'role':'user','content':'x'}],[],allow_tools=False)
                self.assertNotIn('secret',str(caught.exception))
        asyncio.run(exercise())
