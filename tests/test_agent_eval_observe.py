import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import httpx
from evaluation.agent.observe import ObservationSink, PaidCallGate


class AgentObservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_auth_failure_blocks_same_credential_but_keeps_other_provider_available(self):
        sink, gate = ObservationSink('trial'), PaidCallGate()
        def handle(request):
            return httpx.Response(403, json={'error':'fixture'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), event_hooks=sink.http_hooks(gate)) as client:
            await client.post('https://example.test/v1/decisions',headers={'Authorization':'Bearer secret'})
            with self.assertRaises(RuntimeError):
                await client.post('https://example.test/v1/decisions',headers={'Authorization':'Bearer secret'})
            gate.check('Bearer different-secret')
        responses=[e for e in sink.events if e['type']=='http_result']
        self.assertEqual(len(responses),1)
        self.assertEqual(responses[0]['status_code'],403)
        self.assertNotIn('secret',json.dumps(sink.events))

    async def test_model_failure_does_not_report_partial_known_usage_as_complete(self):
        sink=ObservationSink('trial')
        sink.add('model_call',role='general')
        sink.add('model_result',role='general',usage={'input_tokens':3,'output_tokens':2})
        sink.add('model_call',role='general')
        sink.add('model_error',role='general',error='TimeoutError')
        usage=sink.totals()
        self.assertEqual(usage['input_tokens'],3)
        self.assertFalse(usage['complete'])
        self.assertEqual(usage['attempted_calls'],2)

    async def test_quota_body_blocks_but_temporary_rate_limit_does_not(self):
        for code, blocked in [('insufficient_quota',True),('rate_limit_exceeded',False)]:
            sink,gate=ObservationSink('trial'),PaidCallGate()
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(429,json={'error':{'code':code}})),
                                         event_hooks=sink.http_hooks(gate)) as client:
                await client.post('https://example.test/v1/decisions',headers={'Authorization':'Bearer secret'})
            self.assertEqual(gate.blocked,blocked)

    async def test_network_failed_http_attempt_makes_usage_incomplete(self):
        sink,gate=ObservationSink('trial'),PaidCallGate()
        seen=[]
        def handle(req):
            seen.append(1)
            if len(seen)==2:
                raise httpx.ReadTimeout('fixture')
            return httpx.Response(200,json={'usage':{'input_tokens':4,'output_tokens':1}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle),event_hooks=sink.http_hooks(gate)) as client:
            await client.post('https://example.test/v1/decisions')
            with self.assertRaises(httpx.ReadTimeout):
                await client.post('https://example.test/v1/decisions')
        usage=sink.totals('http_result')
        self.assertEqual(usage['attempted_calls'],2)
        self.assertFalse(usage['complete'])

    async def test_auxiliary_models_keep_explicit_concurrent_role_context(self):
        import asyncio
        from evaluation.agent import observe
        self.assertTrue(hasattr(observe,'CURRENT_BRANCH'),'explicit branch context missing')
        sink=ObservationSink('trial')
        class Model:
            async def complete(self,*a,**k):
                await asyncio.sleep(0)
                return {'message':{'role':'assistant','content':'ok'},'usage':None,'finish_reason':'stop'}
        model=observe.ObservedModelRouter(Model(),sink).for_role('rewrite')
        async def one(role,run):
            token=observe.CURRENT_BRANCH.set({'role':role,'run_id':run})
            try:
                await model.complete([],[])
            finally:
                observe.CURRENT_BRANCH.reset(token)
        await asyncio.gather(one('publication_fees','r1'),one('submission_requirements','r2'))
        self.assertEqual({e['run_id'] for e in sink.events if e['type']=='model_result'},{'r1','r2'})


if __name__=='__main__':
    unittest.main()
