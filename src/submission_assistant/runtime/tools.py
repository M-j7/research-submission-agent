"""Five read-only tools with one Pydantic source of schema and validation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Literal
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator
import regex

from .contracts import StrictModel, ToolResult
from .memory import dumps


class GlobInput(StrictModel):
    root: str = Field(min_length=1,max_length=500)
    pattern: str = Field(min_length=1,max_length=256)
    limit: int = Field(default=50,ge=1,le=100)
    cursor: str | None = None


class GrepInput(StrictModel):
    root: str = Field(min_length=1,max_length=500)
    pattern: str = Field(min_length=1,max_length=256)
    mode: Literal['literal','regex'] = 'literal'
    file_pattern: str = '**/*'
    case_sensitive: bool = False
    context_lines: int = Field(default=2,ge=0,le=5)
    limit: int = Field(default=20,ge=1,le=50)
    cursor: str | None = None


class ReadInput(StrictModel):
    ref: str = Field(min_length=1,max_length=500)
    start_line: int = Field(default=1,ge=1)
    max_lines: int = Field(default=200,ge=1,le=500)
    cursor: str | None = None

    @model_validator(mode='after')
    def cursor_start(self):
        if self.cursor and 'start_line' in self.model_fields_set:
            raise ValueError('cursor cannot include start_line')
        return self


class MemoryInput(StrictModel):
    scope: Literal['instant','conversation','user'] = 'conversation'
    record_type: Literal['dialogue','summary','profile','agent_result','evidence','execution'] | None = None
    keyword: str | None = Field(default=None,max_length=256)
    record_ids: list[str] | None = Field(default=None,max_length=20)
    limit: int = Field(default=5,ge=1,le=20)
    cursor: str | None = None

    @model_validator(mode='after')
    def scope_type(self):
        permitted = {'instant':(None,'execution'),'conversation':(None,'dialogue','summary','agent_result','evidence'),'user':(None,'profile')}
        if self.record_type not in permitted[self.scope]:
            raise ValueError('record type not authorized for scope')
        return self


class KnowledgeInput(StrictModel):
    query: str = Field(min_length=1,max_length=2000)
    intent_hints: list[Literal['scope_fit','format','materials','ethics','platform','review','fees','publication','withdrawal_appeal']] = Field(default_factory=list,max_length=3)
    top_k: int = Field(default=5,ge=1,le=10)
    cursor: str | None = None


DEFINITIONS = {
    'glob':(GlobInput,'按虚拟目录和文件名查找授权文件，不查SQL。'),
    'grep':(GrepInput,'在授权文本中搜索关键词或有界正则，返回位置，不查SQL。'),
    'read':(ReadInput,'按路径和范围读取Skill或授权文件，正文不完整时使用cursor续读。'),
    'query_memory':(MemoryInput,'受控查询当前run/会话/用户的SQL记忆，不接受自由SQL。'),
    'search_knowledge':(KnowledgeInput,'查询官方知识库，返回证据和出处，不办理投稿或财务操作。'),
}


class ToolRegistry:
    def __init__(self, workspace, memory, request, run_id, *, rag=None, max_chars=12000):
        self.workspace,self.memory,self.request,self.run_id = workspace,memory,request,run_id
        self.rag = rag
        self.max_chars = max_chars
        self._cursors = {}

    def schemas(self, allowed):
        return [{'type':'function','function':{'name':name,'description':DEFINITIONS[name][1],
                 'parameters':DEFINITIONS[name][0].model_json_schema()}}
                for name in sorted(set(allowed)) if name in DEFINITIONS and (name!='search_knowledge' or self.rag is not None)]

    def _key(self, name, args):
        excluded={'cursor','start_line'} if name=='read' else {'cursor'}
        return json.dumps([name,args.model_dump(exclude=excluded),self.request.user_id,self.request.conv_id,self.run_id],sort_keys=True)

    def _save_cursor(self, name, args, state):
        token = uuid4().hex
        if len(self._cursors)>=128:
            # Runs have bounded calls; do not accumulate unbounded snapshots.
            self._cursors.pop(next(iter(self._cursors)))
        self._cursors[token]=(self._key(name,args),time.monotonic()+300,state)
        return token

    def _resume(self,name,args):
        entry = self._cursors.get(args.cursor)
        if not entry or entry[0]!=self._key(name,args) or entry[1]<time.monotonic():
            raise LookupError('stale cursor')
        return entry[2]

    def _version(self,path):
        self.workspace.resolve(self.workspace.virtual(path))
        stat = path.stat()
        return f'{stat.st_mtime_ns}:{stat.st_size}'

    def _text(self,path):
        if path.suffix.lower() not in ('.md','.txt','.html','.json','.csv','.yaml','.yml','.log'):
            raise TypeError('unsupported resource')
        if path.stat().st_size>2*1024*1024:
            raise OverflowError('resource too large; request narrower resource')
        return path.read_text(encoding='utf-8-sig')

    def _read(self,args):
        state=self._resume('read',args) if args.cursor else None
        path=self.workspace.resolve(args.ref)
        version=self._version(path)
        text=self._text(path)
        if state:
            if version!=state['version']: raise LookupError('stale cursor')
            offset=state['offset']
        else:
            offset=sum(len(line) for line in text.splitlines(keepends=True)[:args.start_line-1])
        segment=''.join(text[offset:].splitlines(keepends=True)[:args.max_lines])[:self.max_chars]
        end=offset+len(segment)
        more=end<len(text)
        cursor=self._save_cursor('read',args,{'offset':end,'version':version}) if more else None
        return ToolResult(status='partial' if more else 'ok' if segment else 'empty',partial=more,next_cursor=cursor,
            data={'resource_ref':args.ref,'resource_version':version,'text':segment,'encoding':'utf-8',
                  'line_start':text[:offset].count('\n')+1,'line_end':text[:end].count('\n')+1,
                  'character_start':offset,'character_end':end,'complete':offset==0 and not more})

    def _fit_read(self,args,result,limit,count):
        """Clip the serialized view, then bind continuation to the *shown* offset."""
        if count(dumps(result.model_dump()))<=limit:
            return result
        source=result.data
        text=source['text']
        start=source['character_start']
        state={'offset':start,'version':source['resource_version']}
        cursor=self._save_cursor('read',args,state)

        def candidate(size):
            segment=text[:size]
            more=size<len(text) or result.partial
            return result.model_copy(update={
                'status':'partial' if more else result.status,'partial':more,
                'next_cursor':cursor if more else None,
                'data':{**source,'text':segment,'character_end':start+size,
                        'line_end':source['line_start']+segment.count('\n'),
                        'complete':start==0 and not more}})

        # Count the whole JSON string (including escaping, cursor and metadata),
        # not just characters. Never emit a cursor which advances by zero.
        low,high=1,len(text)
        best=None
        while low<=high:
            middle=(low+high)//2
            view=candidate(middle)
            if count(dumps(view.model_dump()))<=limit:
                best=view; low=middle+1
            else:
                high=middle-1
        if best is None:
            self._cursors.pop(cursor,None)
            return ToolResult.failure('PRESENTATION_BUDGET','剩余输入预算不足以展示工具正文。')
        state['offset']=best.data['character_end']
        if not best.next_cursor:
            self._cursors.pop(cursor,None)
        return best

    def present(self,name,arguments,result,*,limit,count,call_id=None):
        """Return a bounded view and its durable source; never externalize read again.

        Non-read output is stored once and shown as a JSON-text excerpt. Its
        next_cursor belongs to read, explicitly named in continuation_tool.
        execute's signature stays unchanged for observation/deadline wrappers.
        """
        if count(dumps(result.model_dump()))<=limit:
            return result,None
        if name=='read' and not isinstance(result.data.get('text'),str):
            return ToolResult.failure('PRESENTATION_BUDGET','剩余输入预算不足以展示工具结果。'),None
        ref=None
        try:
            if name=='read' and isinstance(result.data.get('text'),str):
                args=ReadInput.model_validate(json.loads(arguments) if isinstance(arguments,str) else arguments)
                ref=args.ref
                if self._version(self.workspace.resolve(ref))!=result.data['resource_version']:
                    raise LookupError('stale cursor')
                view=result
            else:
                ref=self.workspace.save_tool_result(call_id or uuid4().hex,result.model_dump())
                args=ReadInput(ref=ref)
                view=self._read(args)
                view.data.update(original_tool=name,original_status=result.status,continuation_tool='read')
                if result.next_cursor:
                    view.data['original_next_cursor']=result.next_cursor
                view.warnings=['正文是完整工具结果的JSON片段；用read(ref=resource_ref,cursor=next_cursor)续读。']
            return self._fit_read(args,view,limit,count),ref
        except LookupError:
            return ToolResult.failure('STALE_CURSOR','原文已更新，请重新读取。'),ref
        except (PermissionError,OSError):
            return ToolResult.failure('RESOURCE_UNAVAILABLE','完整结果暂不可读取。'),ref
        except (TypeError,UnicodeError):
            return ToolResult.failure('UNSUPPORTED_RESOURCE','完整结果没有UTF-8文本表示。'),ref
        except OverflowError:
            return ToolResult.failure('RESULT_TOO_LARGE','完整结果过大，请缩小查询范围。'),ref

    def _pattern(self,pattern):
        if pattern.startswith(('/','\\')) or '..' in pattern.split('/') or '\\' in pattern or ':' in pattern:
            raise ValueError('invalid relative pattern')

    async def _files(self,root,pattern,deadline):
        self._pattern(pattern)
        path = self.workspace.resolve(root)
        if not path.exists():
            raise FileNotFoundError
        if path.is_file():
            return [path]
        # os.walk does not follow reparse directories; inspect before descent.
        import os
        files=[]
        visited=0
        for directory,dirs,names in os.walk(path,followlinks=False):
            await asyncio.sleep(0)
            if time.monotonic()>=deadline: raise TimeoutError
            visited+=1
            if visited>1000:
                raise OverflowError('scan limit exceeded; narrow root')
            base = type(path)(directory)
            safe_dirs=[]
            for name in dirs:
                try:
                    self.workspace.virtual(base/name)
                    safe_dirs.append(name)
                except PermissionError:
                    pass
            dirs[:]=sorted(safe_dirs)
            for name in names:
                await asyncio.sleep(0)
                if time.monotonic()>=deadline: raise TimeoutError
                target=base/name
                relative=target.relative_to(path)
                if not (relative.match(pattern) or (pattern.startswith('**/') and relative.match(pattern[3:]))):
                    continue
                try:
                    self.workspace.virtual(target)
                    files.append(target)
                except PermissionError:
                    continue
                if len(files)>2000:
                    raise OverflowError('scan limit exceeded; narrow file_pattern')
        return sorted(files)

    def _page(self,name,args,items,key,*,versions=None,warnings=None):
        offset=0
        if args.cursor:
            state=self._resume(name,args)
            items,offset,versions=state['items'],state['offset'],state.get('versions')
            if versions and any(self._version(p)!=v for p,v in versions):
                raise LookupError('stale cursor')
        limit = getattr(args,'limit',getattr(args,'top_k',5))
        page=items[offset:offset+limit]
        # Chunk record/evidence bodies without losing their tail. Cursor pins full snapshot.
        if not args.cursor and key in ('records','evidence'):
            field='content' if key=='records' else 'text'
            segmented=[]
            for item in items:
                text=item.get(field,'')
                if isinstance(text,str) and len(text)>self.max_chars:
                    for start in range(0,len(text),self.max_chars):
                        segmented.append({**item,field:text[start:start+self.max_chars],
                                          'character_start':start,'character_end':min(len(text),start+self.max_chars),'complete':False})
                else: segmented.append(item)
            items=segmented
            page=items[offset:offset+limit]
        # One page cannot contain more than the configured body budget.
        if key in ('records','evidence'):
            field='content' if key=='records' else 'text'
            chosen=[]; size=0
            for item in page:
                amount=len(item.get(field,''))
                if chosen and size+amount>self.max_chars: break
                chosen.append(item); size+=amount
            page=chosen
        more=offset+len(page)<len(items)
        cursor=self._save_cursor(name,args,{'items':items,'offset':offset+len(page),'versions':versions}) if more else None
        return ToolResult(status='partial' if more else 'ok' if page else 'empty',partial=more,next_cursor=cursor,data={key:page},warnings=warnings or [])

    async def execute(self,name,arguments,allowed):
        local_deadline=time.monotonic()+5
        if name not in allowed or name not in DEFINITIONS:
            return ToolResult.failure('TOOL_NOT_ALLOWED','该工具未注册或不在本角色白名单。')
        try:
            # Reject duplicate JSON fields instead of last-field-wins execution.
            def pairs(values):
                obj={}
                for key,value in values:
                    if key in obj: raise ValueError('duplicate field')
                    obj[key]=value
                return obj
            parsed=json.loads(arguments,object_pairs_hook=pairs) if isinstance(arguments,str) else arguments
            args=DEFINITIONS[name][0].model_validate(parsed)
        except (ValidationError,ValueError,TypeError):
            return ToolResult.failure('INVALID_ARGUMENT','工具参数不符合Schema；请使用允许字段与类型。')
        try:
            if name=='search_knowledge':
                if self.rag is None:
                    return ToolResult.failure('BACKEND_UNAVAILABLE','知识库未绑定。')
                if args.cursor:
                    return self._page(name,args,[],'evidence')
                payload=await asyncio.wait_for(self.rag.search(args.query,top_k=args.top_k),90)
                for evidence in payload.get('evidence',[]):
                    evidence['resource_ref']=self.workspace.publish_evidence(evidence)
                result=self._page(name,args,payload.get('evidence',[]),'evidence',warnings=payload.get('warnings',[]))
                result.data.update({k:v for k,v in payload.items() if k!='evidence'})
                return result
            # Local operations are bounded by byte/file limits and explicit regex timeout.
            if name=='read':
                return self._read(args)
            if name=='query_memory':
                if args.cursor:
                    return self._page(name,args,[],'records')
                rows=self.memory.query(self.request,scope=args.scope,run_id=self.run_id,record_type=args.record_type,
                                       keyword=args.keyword,record_ids=args.record_ids,limit=1000)
                return self._page(name,args,rows,'records')
            if args.cursor:
                return self._page(name,args,[],'entries' if name=='glob' else 'matches')
            files=await self._files(args.root,args.pattern if name=='glob' else args.file_pattern,local_deadline)
            versions=[]
            for path in files:
                await asyncio.sleep(0)
                if time.monotonic()>=local_deadline: raise TimeoutError
                versions.append((path,self._version(path)))
            if name=='glob':
                items=[{'path':self.workspace.virtual(p),'resource_ref':self.workspace.virtual(p),'type':'file','size_bytes':p.stat().st_size,'resource_version':v} for p,v in versions]
                return self._page(name,args,items,'entries',versions=versions)
            matches=[]
            scanned=0
            warnings=[]
            engine=regex.compile(args.pattern,0 if args.case_sensitive else regex.IGNORECASE) if args.mode=='regex' else None
            for path,version in versions:
                scanned+=path.stat().st_size
                if scanned>10*1024*1024: raise OverflowError
                try: lines=self._text(path).splitlines()
                except (TypeError,UnicodeError):
                    warnings.append('已跳过没有UTF-8文本表示的资源。')
                    continue
                for i,line in enumerate(lines):
                    await asyncio.sleep(0)
                    remaining=local_deadline-time.monotonic()
                    if remaining<=0: raise TimeoutError
                    hit=bool(engine.search(line,timeout=min(.05,remaining))) if engine else (args.pattern in line if args.case_sensitive else args.pattern.casefold() in line.casefold())
                    if hit:
                        start=max(0,i-args.context_lines); end=min(len(lines),i+args.context_lines+1)
                        snippet='\n'.join(lines[start:end])
                        matches.append({'path':self.workspace.virtual(path),'resource_ref':self.workspace.virtual(path),'resource_version':version,
                                        'line_number':i+1,'line_start':start+1,'line_end':end,'text':snippet[:self.max_chars],
                                        'complete':len(snippet)<=self.max_chars})
                        if len(matches)>1000: raise OverflowError
            return self._page(name,args,matches,'matches',versions=versions,warnings=warnings)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return ToolResult.failure('TIMEOUT','执行超过截止时间，请缩小查询范围。',True)
        except LookupError:
            return ToolResult.failure('STALE_CURSOR','续读凭据失效或不匹配，请重新发起查询。')
        except (PermissionError,FileNotFoundError,OSError):
            return ToolResult.failure('RESOURCE_UNAVAILABLE','资源不存在或当前实例无权访问。')
        except (TypeError,UnicodeError):
            return ToolResult.failure('UNSUPPORTED_RESOURCE','没有可读取的UTF-8文本表示。')
        except OverflowError:
            return ToolResult.failure('RESULT_TOO_LARGE','结果或扫描范围过大，请缩小路径或范围。')
        except (ValueError,regex.error):
            return ToolResult.failure('INVALID_ARGUMENT','查询模式或参数无效。')
        except Exception:
            return ToolResult.failure('BACKEND_UNAVAILABLE','后端不可用，未能完成查询。')
