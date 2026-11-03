import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


class AgentEvalRunTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.agent.run'), 'evaluation runner missing')
        return importlib.import_module('evaluation.agent.run')

    def test_offline_fault_batch_roundtrips_and_refuses_overwrite_or_tamper(self):
        api = self.api()
        from evaluation.agent.dataset import load_cases
        case = next(c for c in load_cases() if c['case_id'] == 'fault-02')
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'run'
            summary = api.run_batch([case], output=out, values={}, concurrency=1, live=False)
            self.assertEqual(summary['fault']['pass'], 1)
            manifest, cases, executions, scores = api.load_agent_run(out)
            self.assertTrue(manifest['complete'])
            self.assertEqual(len(executions), 1)
            self.assertEqual(scores[0]['business_outcome'], 'pass')
            original = api.file_hash(out/'scores/v1/scores.jsonl')
            version = api.rescore_missing(out, 'v2', values={})
            self.assertEqual(api.read_rows(version/'scores.jsonl')[0]['business_outcome'], 'pass')
            self.assertEqual(api.file_hash(out/'scores/v1/scores.jsonl'), original)
            with self.assertRaises(FileExistsError):
                api.run_batch([case], output=out, values={}, concurrency=1, live=False)
            with (out/'executions.jsonl').open('a', encoding='utf-8') as f:
                f.write('{}\n')
            with self.assertRaises(ValueError):
                api.load_agent_run(out)

    def test_keyboard_interrupt_finalizes_all_expected_ids_without_claiming_complete(self):
        api=self.api()
        from evaluation.agent.dataset import load_cases
        import evaluation.agent.evalscope_adapter as adapter
        original=adapter.run_evalscope
        def interrupted(home,*,mode,**kwargs):
            if mode=='live':
                raise KeyboardInterrupt()
            return original(home,mode=mode,**kwargs)
        case=next(c for c in load_cases() if c['case_id']=='fault-02')
        with tempfile.TemporaryDirectory() as d, patch.object(adapter,'run_evalscope',side_effect=interrupted):
            home=Path(d)/'run'
            summary=api.run_batch([case],output=home,values={},concurrency=1,live=False)
            m,_,es,_=api.load_agent_run(home)
            self.assertFalse(m['complete'])
            self.assertEqual(len(es),1)
            self.assertEqual(summary['fault']['not_executed'],1)

    def test_failed_final_verification_never_publishes_complete_flag(self):
        api=self.api()
        from evaluation.agent.dataset import load_cases
        case=next(c for c in load_cases() if c['case_id']=='fault-02')
        with tempfile.TemporaryDirectory() as d:
            home=Path(d)/'run'
            with patch.object(api,'load_agent_run',side_effect=ValueError('verification fixture')):
                with self.assertRaises(ValueError):
                    api.run_batch([case],output=home,values={},concurrency=1,live=False)
            self.assertFalse(json.loads((home/'manifest.json').read_text(encoding='utf-8'))['complete'])


if __name__ == '__main__':
    unittest.main()
