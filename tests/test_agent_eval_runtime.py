import asyncio
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


class AgentEvalRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def api(self):
        spec = importlib.util.find_spec('evaluation.agent.runtime')
        self.assertIsNotNone(spec, 'isolated full service collector missing')
        return importlib.import_module('evaluation.agent.runtime')

    async def test_parallel_trials_use_real_graph_and_separate_raw_memory(self):
        api = self.api()
        from evaluation.agent.dataset import load_cases
        from evaluation.agent.faults import FaultRecognizer
        class Model:
            model = 'fixture'
            async def complete(self, messages, tools, *, allow_tools=True):
                return {'message': {'role': 'assistant', 'content': '请说明具体问题。'},
                        'finish_reason': 'stop', 'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}
        c = next(c for c in load_cases() if c['case_id'] == 'unclear-02')
        with tempfile.TemporaryDirectory() as d:
            async def one(t):
                return await api.execute_case(c, trial_id=t, run_dir=Path(d), values={}, live=False,
                                              dependencies={'recognizer': FaultRecognizer(), 'model': Model()})
            left, right = await asyncio.gather(one('left'), one('right'))
            self.assertEqual(left['execution_status'], 'completed')
            self.assertEqual(left['turns'][0]['answer'], '请说明具体问题。')
            self.assertEqual(left['usage']['agent']['input_tokens'], 10)
            self.assertEqual(left['usage']['agent']['output_tokens'], 2)
            self.assertTrue(left['raw_records'])
            self.assertTrue(all(e['trial_id'] == 'left' for e in left['events']))
            self.assertNotEqual(left['identity'], right['identity'])

    async def test_deterministic_faults_exercise_runtime_protections(self):
        api = self.api()
        from evaluation.agent.dataset import load_cases
        with tempfile.TemporaryDirectory() as d:
            results = []
            for c in load_cases():
                if c['fault']:
                    results.append(await api.execute_case(c, trial_id=c['case_id'], run_dir=Path(d),
                                                          values={}, live=False))
            by = {r['case_id']: r for r in results}
            self.assertEqual(by['fault-02']['turns'][0]['final_result']['stop_reason'], 'no_progress')
            self.assertIn('TIMEOUT', json.dumps(by['fault-01']['raw_records']))
            self.assertIn('INVALID_ARGUMENT', json.dumps(by['fault-03']['raw_records']))
            self.assertIn('RESOURCE_UNAVAILABLE', json.dumps(by['fault-04']['raw_records']))
            self.assertIn('RESOURCE_UNAVAILABLE', json.dumps(by['fault-05']['raw_records']))

    async def test_outer_timeout_retains_terminal_and_unknown_usage(self):
        api = self.api()
        from evaluation.agent.dataset import load_cases
        from evaluation.agent.faults import FaultRecognizer
        class Slow:
            model = 'fixture'
            async def complete(self, *a, **k):
                await asyncio.sleep(10)
        with tempfile.TemporaryDirectory() as d:
            r = await api.execute_case(load_cases()[0], trial_id='timed', run_dir=Path(d), values={}, live=False,
                deadline_s=.05, dependencies={'recognizer': FaultRecognizer(), 'model': Slow()})
            self.assertEqual(r['execution_status'], 'cancelled')
            self.assertEqual(r['stop_reason'], 'outer_timeout')
            self.assertIsNone(r['usage']['agent']['input_tokens'])

    async def test_external_cancellation_preserves_interrupted_execution(self):
        api=self.api()
        from evaluation.agent.dataset import load_cases
        from evaluation.agent.faults import FaultRecognizer
        ready=asyncio.Event()
        class Slow:
            async def complete(self,*a,**k):
                ready.set()
                await asyncio.Event().wait()
        with tempfile.TemporaryDirectory() as d:
            task=asyncio.create_task(api.execute_case(load_cases()[0],trial_id='cancelled',run_dir=Path(d),values={},live=False,
                dependencies={'recognizer':FaultRecognizer(),'model':Slow()}))
            await ready.wait()
            task.cancel()
            result=await task
            self.assertEqual(result['execution_status'],'cancelled')
            self.assertEqual(result['stop_reason'],'external_cancelled')
            self.assertTrue(result['raw_records'])


if __name__ == '__main__':
    unittest.main()
