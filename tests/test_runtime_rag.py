import asyncio
import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class RagTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.runtime.rag'),'Multi-Query wrapper missing')
        self.api=importlib.import_module('submission_assistant.runtime.rag')

    def test_accepted_queries_fuse_unique_child_ids_and_rerank_original_question(self):
        called=[]
        class Backend:
            version='v1'
            async def recall(self,q):
                called.append(('recall',q))
                return {'rankings':[['a','b','a'],['b']], 'candidates':{'a':{'chunk_id':'a'},'b':{'chunk_id':'b'}},'warnings':[]}
            async def rerank(self,q,rows):
                called.append(('rerank',q,[r['chunk_id'] for r in rows])); return rows
            def expand(self,rows,k): return [{'evidence_id':r['chunk_id'],'text':r['chunk_id']} for r in rows[:k]]
        async def rewrite(q,rejected): return ['等价问题','跨界问题','等价问题']
        async def judge(q,candidates): return {c:c=='等价问题' for c in candidates}
        rag=self.api.MultiQueryRAG(Backend(),rewriter=rewrite,judge=judge)
        output=asyncio.run(rag.search('原问题',top_k=5))
        self.assertEqual(output['queries'],['原问题','等价问题'])
        self.assertEqual([e['evidence_id'] for e in output['evidence']],['b','a'])
        self.assertEqual(called[-1],('rerank','原问题',['b','a']))
        self.assertLessEqual(output['rewrite_attempts'],2)

    def test_judge_failure_never_allows_unverified_rewrites(self):
        queries=[]
        class Backend:
            version='v1'
            async def recall(self,q): queries.append(q); return {'rankings':[], 'candidates':{},'warnings':[]}
            async def rerank(self,q,rows): return rows
            def expand(self,rows,k): return []
        async def rewrite(q,rejected): return ['新问题']
        async def judge(q,candidates): raise RuntimeError('secret')
        output=asyncio.run(self.api.MultiQueryRAG(Backend(),rewriter=rewrite,judge=judge).search('原问题'))
        self.assertEqual(queries,['原问题'])
        self.assertTrue(output['warnings'])
        self.assertNotIn('secret',str(output))

    def test_cache_key_includes_index_version_and_outage_falls_back(self):
        from submission_assistant.runtime.cache import VersionedCache
        cache=VersionedCache()
        calls=[]
        class Backend:
            version='v1'
            async def recall(self,q): calls.append(q); return {'rankings':[['a']],'candidates':{'a':{'chunk_id':'a'}},'warnings':[]}
            async def rerank(self,q,rows): return rows
            def expand(self,rows,k): return [{'evidence_id':'a','text':'evidence'}]
        async def exercise():
            backend=Backend(); rag=self.api.MultiQueryRAG(backend,cache=cache)
            await rag.search('问题'); hit=await rag.search('问题')
            self.assertTrue(hit['cache_hit'])
            backend.version='v2'; await rag.search('问题')
            self.assertEqual(len(calls),2)
        asyncio.run(exercise())

    def test_all_recall_failed_is_error_not_empty_policy(self):
        class Backend:
            version='v1'
            async def recall(self,q): raise RuntimeError('secret')
        with self.assertRaises(self.api.RagUnavailable):
            asyncio.run(self.api.MultiQueryRAG(Backend()).search('问题'))
