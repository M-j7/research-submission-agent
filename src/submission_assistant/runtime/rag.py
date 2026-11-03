"""Multi-Query guard -> dense/BM25 lists -> cross-list RRF -> original rerank.

The old query script remains usable. This adapter uses its pure ranking helpers,
but online model calls use async transport to preserve deadlines/cancellation.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import sys
from uuid import uuid4
from urllib.parse import urlsplit

import httpx
from pydantic import Field

from .cache import VersionedCache
from .contracts import StrictModel
from ..intent.jev import parse_independent_batch
from ..intent.embedding import _vectors


class RagUnavailable(RuntimeError):
    pass


class Queries(StrictModel):
    queries: list[str] = Field(max_length=3)


class MultiQueryRAG:
    def __init__(self,backend,*,rewriter=None,judge=None,cache=None,scope='official-ieee',config_version='multiquery-v1'):
        self.backend=backend; self.rewriter=rewriter; self.judge=judge
        self.cache=cache or VersionedCache()
        self.scope=scope; self.config_version=config_version

    async def search(self,original,*,top_k=5):
        if not isinstance(original,str) or not original.strip() or len(original)>2000 or not 1<=top_k<=10:
            raise ValueError('invalid RAG query')
        key=self.cache.key('rag',{'query':original,'top_k':top_k,'index_version':self.backend.version,
                      'scope':self.scope,'config':self.config_version,'rewriter':getattr(self.rewriter,'version',bool(self.rewriter)),
                      'judge':getattr(self.judge,'version',bool(self.judge))})
        async with self.cache.lock(key):
            cached=await self.cache.get(key)
            if cached is not None: return {**cached,'cache_hit':True}
            output=await asyncio.wait_for(self._search(original,top_k),90)
            if not output['warnings']:
                await self.cache.set(key,output)
            return output

    async def _search(self,original,top_k):
        queries=[original]; rejected=[]; attempts=0; warnings=[]
        if self.rewriter is not None and self.judge is not None:
            for attempt in range(2):
                attempts+=1
                try:
                    proposed=await self.rewriter(original,rejected)
                    if not isinstance(proposed,list) or any(not isinstance(q,str) or not q.strip() or len(q)>2000 for q in proposed):
                        raise ValueError('invalid rewrites')
                    candidates=list(dict.fromkeys(q.strip() for q in proposed if q.strip() not in queries))[:3-(len(queries)-1)]
                    if not candidates: break
                    verdict=await self.judge(original,candidates)
                    if not isinstance(verdict,dict) or set(verdict)!=set(candidates) or any(type(v) is not bool for v in verdict.values()):
                        raise ValueError('invalid judgments')
                    queries.extend(q for q in candidates if verdict[q])
                    rejected=[q for q in candidates if not verdict[q]]
                    if not rejected or len(queries)>=4: break
                except asyncio.CancelledError: raise
                except Exception:
                    warnings.append('查询改写或JEV校验不可用；仅使用原问题及此前已通过候选。')
                    break
            if rejected: warnings.append('未通过语义与范围校验的改写未参与检索。')
        semaphore=asyncio.Semaphore(2)
        async def recall(query):
            async with semaphore: return await self.backend.recall(query)
        outcomes=await asyncio.gather(*(recall(q) for q in queries),return_exceptions=True)
        rankings=[]; candidates={}; success=0
        for query,result in zip(queries,outcomes):
            if isinstance(result,asyncio.CancelledError): raise result
            if isinstance(result,BaseException):
                warnings.append('部分查询召回失败，保留其余成功路径。'); continue
            success+=1
            rankings.extend(result['rankings']); candidates.update(result['candidates'])
            warnings.extend(result.get('warnings',[]))
        if not success: raise RagUnavailable('所有召回路径不可用，不能视为政策不存在。')
        ranks={}
        for ranking in rankings:
            seen=set()
            for rank,identifier in enumerate(ranking,1):
                if identifier in seen: continue
                seen.add(identifier)
                if identifier not in candidates: raise RagUnavailable('索引与原始子块不一致。')
                ranks.setdefault(identifier,[]).append(rank)
        ordered=sorted(ranks,key=lambda identifier:(-sum(1/(60+r) for r in ranks[identifier]),identifier))
        rows=[{**candidates[identifier],'rrf_score':sum(1/(60+r) for r in ranks[identifier])} for identifier in ordered[:60]]
        if rows:
            try: rows=await self.backend.rerank(original,rows)
            except asyncio.CancelledError: raise
            except Exception: warnings.append('Reranker不可用，本次明确降级为RRF顺序。')
        evidence=self.backend.expand(rows,top_k)
        return {'evidence':evidence,'queries':queries,'rewrite_attempts':attempts,'index_version':self.backend.version,
                'warnings':list(dict.fromkeys(warnings)),'cache_hit':False,'score_type':'ranking_signals_not_probability'}


class ModelQueryRewriter:
    def __init__(self,model):
        self.model=model
        self.version='rewrite-v1:'+model.model

    async def __call__(self,original,rejected):
        instruction='仅将原问题改写成最多三个同义检索查询，可加入对应英文术语。保留期刊、阶段、主体、否定、金额和限制；不拆出用户未问的额外诉求，不补事实。调用propose_queries。已有失败候选只用于修正，不继续重复。'
        tool={'type':'function','function':{'name':'propose_queries','description':'生成等价检索查询','parameters':Queries.model_json_schema()}}
        reply=await self.model.complete([{'role':'system','content':instruction},
            {'role':'user','content':json.dumps({'original':original,'rejected':rejected},ensure_ascii=False)}],[tool])
        if reply.get('finish_reason')!='tool_calls': raise ValueError('rewriter did not use schema')
        calls=reply['message'].get('tool_calls',[])
        if len(calls)!=1 or calls[0]['function']['name']!='propose_queries': raise ValueError('invalid rewrite call')
        return Queries.model_validate_json(calls[0]['function']['arguments']).queries


class JevRewriteJudge:
    def __init__(self,settings,*,client=None,threshold=.8):
        self.settings=settings; self.client=client; self.threshold=threshold
        self.version=f'guard-v1:{settings.model}:{threshold}'

    async def __call__(self,original,candidates):
        from ..intent.jev import JevIntentRecognizer
        body={'model':self.settings.model,'state':{'original_query':original},'questions':{}}
        for i,candidate in enumerate(candidates):
            body['questions'][f'Q{i}']={'type':'noul','instructions':'判断候选是否忠实服务原问题。文字相似或同属投稿业务并不足够；不能新增独立诉求、用户事实或扩大规则适用范围。',
                  'criteria':{'true':f'候选：{candidate}。与原问题语义相关且约束一致，保留期刊、阶段、主体、肯否、条件与范围；仅同义改写或跨语言检索术语。',
                              'false':f'候选：{candidate}。改变或臆造上述条件、询问未提出的事项、否定反转，或相关性无法确认。'}}
        if len(json.dumps(body['state'],ensure_ascii=False))>8000: raise ValueError('guard input too large')
        async def run(client):
            data=await JevIntentRecognizer(self.settings)._post(client,body,uuid4().hex)
            values,_=parse_independent_batch(data,tuple(body['questions']))
            return {candidate:values[f'Q{i}']>=self.threshold for i,candidate in enumerate(candidates)}
        if self.client: return await run(self.client)
        async with httpx.AsyncClient(trust_env=False) as client: return await run(client)


class ExistingKnowledgeBackend:
    def __init__(self,project:Path,values:dict,*,client=None):
        script_dir=project/'解析试验'/'父子检索'/'scripts'
        if str(script_dir) not in sys.path: sys.path.insert(0,str(script_dir))
        self.api=importlib.import_module('query_chroma_parent_child')
        persist=importlib.import_module('persist_chroma_parent_child')
        self.client=client; self.values=values
        self.children=persist.read_jsonl(persist.CHILDREN_PATH)
        self.parents={str(row['chunk_id']):row for row in persist.read_jsonl(persist.PARENTS_PATH)}
        self.children_by_id={str(row['chunk_id']):row for row in self.children}
        # Use the actual ASCII-path location configured by the existing index script.
        self.chroma_dir=persist.CHROMA_DIR
        if not self.chroma_dir.is_dir(): raise ValueError('existing Chroma index missing')
        self.collection_name=persist.COLLECTION_NAME
        self.persist=persist
        self.aliases=self.api.load_term_glossary()
        files=[persist.CHILDREN_PATH,persist.PARENTS_PATH,self.api.TERM_GLOSSARY_PATH]
        manifest=self.chroma_dir/'index_manifest.json'
        if manifest.exists(): files.append(manifest)
        fingerprint=hashlib.sha256()
        for path in files: fingerprint.update(path.read_bytes())
        fingerprint.update(json.dumps({k:values.get(k) for k in ('EMBEDDING_MODEL','RERANKER_MODEL','EMBEDDING_BASE_URL','RERANKER_BASE_URL')},sort_keys=True).encode())
        self.version=fingerprint.hexdigest()

    async def _post(self,url,key,body):
        async def run(client):
            response=await client.post(url,headers={'Authorization':'Bearer '+key},json=body,timeout=30)
            if response.status_code!=200 or len(response.content)>2*1024*1024: raise RagUnavailable('知识检索模型服务不可用。')
            return response.json()
        if self.client: return await run(self.client)
        async with httpx.AsyncClient(trust_env=False) as client: return await run(client)

    async def recall(self,query):
        warnings=[]; rankings=[]
        try:
            base=self.values['EMBEDDING_BASE_URL'].rstrip('/')
            data=await self._post(base if base.endswith('/embeddings') else base+'/embeddings',self.values['EMBEDDING_API_KEY'],{'model':self.values['EMBEDDING_MODEL'],'input':[query]})
            vector=_vectors([data['data'][0]['embedding']],1)[0]
            client=self.persist._get_client(self.chroma_dir)
            try:
                collection=client.get_collection(self.collection_name,embedding_function=None)
                count=collection.count()
                ids=collection.query(query_embeddings=[vector],n_results=min(30,count),include=['distances'])['ids'][0] if count else []
                rankings.append(ids)
            finally: client.close()
        except asyncio.CancelledError: raise
        except Exception: warnings.append('向量召回不可用，保留BM25路径。')
        try: rankings.append(self.api.bm25_rank_children(query,self.children,top_k=30,term_aliases=self.aliases))
        except Exception: warnings.append('BM25召回不可用，保留向量路径。')
        if not rankings: raise RagUnavailable('双路召回均失败。')
        return {'rankings':rankings,'candidates':self.children_by_id,'warnings':warnings}

    async def rerank(self,query,rows):
        selected=rows[:30]
        base=urlsplit(self.values['RERANKER_BASE_URL'])
        url=f'{base.scheme}://{base.netloc}/api/v1/services/rerank/text-rerank/text-rerank'
        reply=await self._post(url,self.values['RERANKER_API_KEY'],{'model':self.values['RERANKER_MODEL'],
            'input':{'query':query,'documents':[r['text'] for r in selected]},'parameters':{'top_n':len(selected),
            'instruct':'Given a web search query, retrieve relevant passages that answer the query.'}})
        return self.api.rerank_child_candidates(query,rows,top_k=30,base_url=self.values['RERANKER_BASE_URL'],
            api_key=self.values['RERANKER_API_KEY'],model=self.values['RERANKER_MODEL'],request_json=lambda *a:reply)

    def expand(self,rows,k):
        hits=self.api.expand_fused_candidates_to_parents(rows,self.parents,top_k=k)
        evidence=[]
        for hit in hits:
            metadata=hit['metadata']
            evidence.append({'evidence_id':hit['parent_chunk_id'],'parent_chunk_id':hit['parent_chunk_id'],
                'matched_child_id':hit['matched_child_id'],'text':hit['text'],'title':metadata.get('title'),
                'source_url':metadata.get('source_url') or metadata.get('url'),
                'source_locator':metadata.get('header_path') or metadata.get('section_path') or metadata.get('page_number'),
                'source_version':metadata.get('source_sha256') or metadata.get('source_version'),'collected_at':metadata.get('retrieved_at') or metadata.get('collected_at'),
                'resource_ref':None,'ranking_scores':{'rrf':hit['rrf_score'],'rerank':hit.get('rerank_score')}})
        return evidence
