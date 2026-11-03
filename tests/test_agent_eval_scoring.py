import asyncio
import copy
import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import httpx


class AgentEvalScoringTests(unittest.IsolatedAsyncioTestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.score'), 'independent scorer missing')
        return importlib.import_module('evaluation.agent.score')

    def fixture(self):
        from evaluation.agent.dataset import load_cases
        c = next(c for c in load_cases() if c['case_id'] == 'unclear-02')
        e = {'case_id': c['case_id'], 'trial_id': 't', 'group': 'business', 'execution_status': 'completed',
             'stop_reason': 'final_answer', 'latency_ms': 10, 'usage': {}, 'events': [], 'runs': [], 'raw_records': [],
             'turns': [{'message': '这个怎么弄', 'answer': '请说明所指', 'route': {'agent_types': ['general']}, 'final_result': {}}]}
        return c, e

    async def test_completed_does_not_mean_business_pass_and_unknown_stays_unknown(self):
        api = self.api()
        c, e = self.fixture()
        r = api.score_case(c, e, {'checks': {s['id']: {'probability': .5} for s in c['semantic_checks']}})
        self.assertEqual(r['business_outcome'], 'unavailable')
        r = api.score_case(c, e, {'checks': {s['id']: {'probability': .8} for s in c['semantic_checks']}})
        self.assertEqual(r['business_outcome'], 'pass')
        r = api.score_case(c, e, {'checks': {s['id']: {'probability': .2} for s in c['semantic_checks']}})
        self.assertEqual(r['business_outcome'], 'fail')
        e['execution_status'] = 'failed'
        r = api.score_case(c, e, {'checks': {}})
        self.assertEqual(r['business_outcome'], 'fail')

    async def test_denominators_keep_unavailable_and_separate_faults(self):
        api = self.api()
        c, e = self.fixture()
        cases = [dict(c, case_id=str(i)) for i in range(3)] + [dict(c, case_id='f', group='fault')]
        es = [dict(e, case_id=c['case_id']) for c in cases]
        scores = [dict(case_id=str(i), group='business', business_outcome=s, rules=[], performance={})
                  for i, s in enumerate(['pass','fail','unavailable'])]
        scores.append(dict(case_id='f', group='fault', business_outcome='pass', rules=[], performance={}))
        summary = api.summarize(cases, es, scores)
        self.assertEqual(summary['business']['verified_ratio'], 1/3)
        self.assertEqual(summary['business']['valid_pass_rate'], .5)
        self.assertEqual(summary['business']['expected'], 3)
        self.assertEqual(summary['fault']['pass'], 1)

    async def test_judge_429_retries_but_403_and_malformed_do_not(self):
        self.api()
        from evaluation.agent.judge import judge_semantics
        from submission_assistant.intent.jev import JevSettings
        c, e = self.fixture()
        settings = JevSettings('https://api.perplexity.ai/v1', 'secret', 'pplx-decider-flash-v1.2', provider='decisions')
        for code, expected in [(429,2), (403,1), (422,1), (200,1)]:
            calls = []
            def handle(req):
                body = json.loads(req.content)
                calls.append(body)
                self.assertNotIn('请说明所指', json.dumps(body['questions'], ensure_ascii=False))
                return httpx.Response(code, json={'invalid': 'fixture'})
            async def no_sleep(_):
                return None
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                judged = await judge_semantics(c, e, settings=settings, client=client, sleep=no_sleep)
            self.assertEqual(judged['attempts'], expected)
            self.assertEqual(len(calls), expected)
            self.assertFalse(judged['checks'])

    async def test_json_array_is_unavailable_not_batch_exception(self):
        self.api()
        from evaluation.agent.judge import judge_semantics
        from submission_assistant.intent.jev import JevSettings
        c,e=self.fixture()
        settings=JevSettings('https://api.perplexity.ai/v1','secret','pplx-decider-flash-v1.2',provider='decisions')
        for payload in ([], 'unexpected'):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=payload))) as client:
                result=await judge_semantics(c,e,settings=settings,client=client)
            self.assertEqual(result['checks'],{})
            self.assertEqual(result['attempts'],1)
            self.assertIn('invalid_judge_response',result['errors'])

    async def test_accepted_whitelist_or_path_violation_is_program_failure(self):
        api=self.api()
        c,e=self.fixture()
        for ref,allowed in [('/system/manuals/../../private.txt',['read']),('/system/manuals/a.md',[]),
                            ('C:/private.txt',['read']),('/system/manuals/config.env',['read'])]:
            e['raw_records']=[{'run':'r','kind':'execution','role':'assistant','body':{'tool_calls':[
                {'id':'x','function':{'name':'read','arguments':json.dumps({'ref':ref})}}]}},
                {'run':'r','kind':'execution','role':'tool','body':{'tool_call_id':'x','content':'{"status":"ok","data":{}}'}}]
            e['events']=[{'type':'tool_call','run_id':'r','name':'read','allowed':allowed}]
            s=api.score_case(c,e,{'checks':{v['id']:{'probability':1} for v in c['semantic_checks']}})
            self.assertEqual(s['business_outcome'],'fail',ref)


if __name__ == '__main__':
    unittest.main()
