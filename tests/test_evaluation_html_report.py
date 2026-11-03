import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class EvaluationHtmlReportTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('mteb'), 'optional retrieval scoring environment')
    def test_saved_rag_metrics_load_without_silent_source_loss(self):
        from evaluation import html_report as api
        from evaluation.rag.dataset import file_hash, write_jsonl
        from evaluation.rag.generation_data import prepare_records
        from evaluation.rag.generate import initialize_run
        from evaluation.rag.ragas_score import METRICS, metric_result
        from evaluation.rag.score import summarize
        from generation_fixture import make_source

        # Exercise real on-disk hash validation and aggregation with synthetic
        # inputs, rather than requiring an unpublished local evaluation archive.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            retrieval = root / 'evaluation/rag/runs'
            retrieval.mkdir(parents=True)
            source = make_source(root, name='fixture', baseline='B3').rename(retrieval / 'fixture')
            summarize(source)
            records = prepare_records([('B3', source, 'B3')])
            run = root / 'evaluation/rag/generation-runs/fixture'
            class Model:
                model = 'fixture'
            manifest = initialize_run(run, records, Model())
            answers = [{**r, 'response': 'synthetic answer', 'generation_status': 'ok'} for r in records]
            write_jsonl(run / 'answers.jsonl', answers)
            manifest.update(complete=True, answers_sha256=file_hash(run / 'answers.jsonl'))
            (run / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            scores = run / 'ragas'
            scores.mkdir()
            rows = [{k: a[k] for k in ('sample_id', 'query_id', 'config_id', 'language', 'input_sha256', 'generation_status')}
                    | {'metrics': {m: metric_result(1.0 if i == 0 else None) for m in METRICS}}
                    for i, a in enumerate(answers)]
            write_jsonl(scores / 'scores.jsonl', rows)
            (scores / 'manifest.json').write_text(json.dumps({
                'complete': True, 'answers_sha256': manifest['answers_sha256'],
                'scores_sha256': file_hash(scores / 'scores.jsonl')}), encoding='utf-8')
            with patch.object(api, 'ROOT', root):
                data = api.load_results()

        selected = [r for r in data['retrieval'] if r['run']=='fixture' and r['language']=='all']
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['metrics']['NDCG@5'], 0.0)
        full = next(r for r in data['generation'] if r['run']=='fixture')
        self.assertEqual(full['count'], 2)
        self.assertEqual(sum(m['valid'] for views in full['summary'].values() for m in views['all'].values()), len(METRICS))
        self.assertEqual(data['warnings'], [])

    def test_static_metrics_escape_text_and_keep_zero_and_missing_distinct(self):
        self.assertIsNotNone(importlib.util.find_spec('evaluation.html_report'), 'static HTML exporter missing')
        api = importlib.import_module('evaluation.html_report')
        self.assertEqual(api.value_cell(0), '0.0000')
        self.assertEqual(api.value_cell(None), '—')
        html = api.render_report({'retrieval':[], 'generation':[], 'agent':[],
                                  'warnings':['</script><script>alert(1)</script>']})
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('RAG 检索', html)
        self.assertIn('RAG 生成', html)
        self.assertIn('Agent', html)

    def test_usage_cells_label_partial_and_unknown_instead_of_python_none(self):
        from evaluation import html_report as api
        fn=getattr(api,'usage_cell',None)
        self.assertIsNotNone(fn,'usage completeness not visible in HTML')
        self.assertEqual(fn({'input_tokens':None,'complete':False},'input_tokens'),'—（未知）')
        self.assertEqual(fn({'input_tokens':3,'complete':False},'input_tokens'),'3（部分）')

    def test_partial_run_prominently_names_blocking_component_and_http_status(self):
        from evaluation.html_report import render_report
        group={'expected':1,'executed':0,'pass':0,'fail':0,'unavailable':1,'not_executed':1,'valid':0,
               'latency_p50_ms':None,'latency_p95_ms':None}
        data={'retrieval':[],'generation':[],'warnings':[], 'agent':[{'run':'partial','manifest':{'complete':False},
            'summary':{'business':group,'fault':group,'limitations':[]},'scores':[],
            'executions':[{'case_id':'x','events':[{'type':'http_result','component':'embedding','status_code':403}]}]}]}
        html=render_report(data)
        self.assertIn('embedding',html)
        self.assertIn('HTTP 403',html)
        self.assertIn('不能解读为业务能力',html)

    def test_context_policy_is_visible_and_distinguished_from_provider_usage(self):
        from evaluation.html_report import render_report
        group={'expected':0,'executed':0,'pass':0,'fail':0,'unavailable':0,'not_executed':0,'valid':0,
               'latency_p50_ms':None,'latency_p95_ms':None}
        data={'retrieval':[],'generation':[],'warnings':[], 'agent':[{'run':'new','manifest':{'complete':True,
            'runtime_config':{'input_budget_tokens':700000,'compression_trigger_tokens':600000,
                              'counting_mode':'local_qwen_tokenizer_estimate_json_with_10pct_margin'}},
            'environment_note':'系统待机影响 <test>',
            'summary':{'business':group,'fault':group,'limitations':[]},'scores':[],'executions':[]}]}
        html=render_report(data)
        self.assertIn('700000',html)
        self.assertIn('600000',html)
        self.assertIn('本地估算',html)
        self.assertIn('系统待机影响 &lt;test&gt;',html)


if __name__ == '__main__':
    unittest.main()
