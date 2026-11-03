"""Append-only raw SQLite memory and versioned, replaceable input projections.

No LLM call happens while the SQLite lock is held. The public conversation is
not a second blackboard and never includes another run's private trace.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def now():
    return datetime.now(timezone.utc).isoformat()


class MemoryManager:
    def __init__(self, path: Path, *, compressor=None, compression_timeout_s=30):
        if compression_timeout_s<=0: raise ValueError('compression timeout must be positive')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self._lock = threading.RLock()
        self.compressor = compressor
        self.compression_timeout_s = compression_timeout_s
        self._tasks = {}
        self._attempted = {}
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS turns(
            seq INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, conv TEXT, turn TEXT,
            message TEXT, answer TEXT, state TEXT DEFAULT 'pending', created TEXT,
            UNIQUE(user,conv,turn));
          CREATE TABLE IF NOT EXISTS runs(
            run TEXT PRIMARY KEY, user TEXT, conv TEXT, turn TEXT, role TEXT, intents TEXT,
            state TEXT DEFAULT 'active', status TEXT DEFAULT '{}', projection_rev INTEGER DEFAULT 0);
          CREATE TABLE IF NOT EXISTS records(
            seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, user TEXT, conv TEXT,
            turn TEXT, run TEXT, kind TEXT, role TEXT, step INTEGER, body TEXT, created TEXT);
          CREATE INDEX IF NOT EXISTS scope_records ON records(user,conv,run,kind,seq);
          CREATE TABLE IF NOT EXISTS projections(
            run TEXT, record TEXT, body TEXT, revision INTEGER, PRIMARY KEY(run,record));
          CREATE TABLE IF NOT EXISTS summaries(
            user TEXT, conv TEXT, revision INTEGER, source_ids TEXT, body TEXT,
            PRIMARY KEY(user,conv));
          CREATE TABLE IF NOT EXISTS users(
            user TEXT PRIMARY KEY, profile_rev INTEGER DEFAULT 0, extracted_count INTEGER DEFAULT 0);
          CREATE TABLE IF NOT EXISTS profiles(
            id TEXT PRIMARY KEY, user TEXT, key TEXT, value TEXT, state TEXT,
            operation TEXT, sources TEXT, revision INTEGER, created TEXT);
        ''')
        self.db.commit()

    def close(self):
        if any(not t.done() for t in self._tasks.values()):
            raise RuntimeError('stop memory background tasks before close')
        self.db.close()

    @contextmanager
    def transaction(self):
        """Serialize companion metadata operations with core memory writes."""
        with self._lock, self.db:
            yield self.db

    def _record(self, req, run, kind, role, body, step=0):
        record_id = uuid4().hex
        self.db.execute('INSERT INTO records(id,user,conv,turn,run,kind,role,step,body,created) VALUES(?,?,?,?,?,?,?,?,?,?)',
                        (record_id,req.user_id,req.conv_id,req.turn_id,run,kind,role,step,dumps(body),now()))
        return record_id

    def begin_turn(self, req):
        with self._lock, self.db:
            row = self.db.execute('SELECT * FROM turns WHERE user=? AND conv=? AND turn=?',
                                  (req.user_id,req.conv_id,req.turn_id)).fetchone()
            if row:
                if row['message'] != req.message:
                    raise ValueError('turn ID already belongs to a different message')
                return dict(row)
            self.db.execute('INSERT INTO turns(user,conv,turn,message,created) VALUES(?,?,?,?,?)',
                            (req.user_id,req.conv_id,req.turn_id,req.message,now()))
            self.db.execute('INSERT OR IGNORE INTO users(user) VALUES(?)', (req.user_id,))
            self._record(req, '', 'dialogue', 'user', {'content':req.message})
            return {'state':'pending','answer':None}

    def create_run(self, req, run_id, role, intents):
        with self._lock, self.db:
            self.db.execute('INSERT INTO runs(run,user,conv,turn,role,intents) VALUES(?,?,?,?,?,?)',
                            (run_id,req.user_id,req.conv_id,req.turn_id,role,dumps(intents)))

    def _run(self, req, run_id):
        row = self.db.execute('SELECT * FROM runs WHERE run=? AND user=? AND conv=? AND turn=?',
                              (run_id,req.user_id,req.conv_id,req.turn_id)).fetchone()
        if not row:
            raise PermissionError('resource unavailable')
        return row

    def append(self, req, run_id, role, body, step):
        with self._lock, self.db:
            if self._run(req,run_id)['state'] != 'active':
                raise ValueError('run is sealed')
            if role not in ('assistant','tool'):
                raise ValueError('unsupported execution role')
            return self._record(req,run_id,'execution',role,body,step)

    def set_status(self, req, run_id, status):
        with self._lock, self.db:
            row = self._run(req,run_id)
            if row['state'] != 'active':
                raise ValueError('run is sealed')
            self._record(req,run_id,'status','system',status, status.get('current_step',0))
            self.db.execute('UPDATE runs SET status=? WHERE run=?', (dumps(status),run_id))

    def instant_snapshot(self, req, run_id):
        with self._lock:
            run = self._run(req,run_id)
            rows = self.db.execute('SELECT r.*,p.body AS projected FROM records r LEFT JOIN projections p ON r.run=p.run AND r.id=p.record WHERE r.run=? AND r.kind=? ORDER BY r.seq',
                                   (run_id,'execution')).fetchall()
            messages = []
            for row in rows:
                body = json.loads(row['body'])
                if row['projected'] is not None:
                    body['content'] = row['projected']
                messages.append({'role':row['role'],**{k:v for k,v in body.items() if k in ('content','tool_call_id','tool_calls')}})
            return {'message':req.message,'intent_ids':json.loads(run['intents']), 'messages':messages,
                    'status':json.loads(run['status']), 'projection_revision':run['projection_rev']}

    def public_snapshot(self, req):
        # One short read transaction pins all three read queries, including across processes.
        with self._lock, self.db:
            self.db.execute('BEGIN')
            turns = self.db.execute("SELECT * FROM turns WHERE user=? AND conv=? AND state IN ('completed','incomplete') ORDER BY seq", (req.user_id,req.conv_id)).fetchall()
            ids = [t['turn'] for t in turns]
            rows = self.db.execute("SELECT * FROM records WHERE user=? AND conv=? AND kind='dialogue' ORDER BY seq", (req.user_id,req.conv_id)).fetchall()
            older_ids = set(ids[:-5])
            summary = self.db.execute('SELECT * FROM summaries WHERE user=? AND conv=?',(req.user_id,req.conv_id)).fetchone()
            covered = set(json.loads(summary['source_ids'])) if summary else set()
            def view(row):
                return {'record_id':row['id'],'role':row['role'],'content':json.loads(row['body'])['content'],'turn_id':row['turn']}
            # Concurrent turns can finish in reverse order. Present complete pairs
            # in turn-start order, not interleaved record-arrival order.
            recent = [view(r) for turn_id in ids[-5:] for r in rows if r['turn'] == turn_id]
            older = [view(r) for r in rows if r['turn'] in older_ids and r['id'] not in covered]
            profile_rows = self.db.execute("SELECT * FROM profiles WHERE user=? AND state='active' ORDER BY revision",(req.user_id,)).fetchall()
            user = self.db.execute('SELECT * FROM users WHERE user=?',(req.user_id,)).fetchone()
            return {'recent':recent,'older':older,'summary':summary['body'] if summary else '',
                    'summary_revision':summary['revision'] if summary else 0,
                    'summary_source_ids':list(covered),
                    'older_source_ids':[r['id'] for r in rows if r['turn'] in older_ids],
                    'revision':max((t['seq'] for t in turns),default=0),
                    'profile':{r['key']:json.loads(r['value']) for r in profile_rows},
                    'profile_revision':user['profile_rev'] if user else 0}

    def claim_turn(self, req):
        with self._lock, self.db:
            result=self.db.execute("UPDATE turns SET state='running' WHERE user=? AND conv=? AND turn=? AND state='pending'",(req.user_id,req.conv_id,req.turn_id))
            return result.rowcount==1

    def abort_turn(self,req,state):
        with self._lock,self.db:
            self.db.execute("UPDATE turns SET state=? WHERE user=? AND conv=? AND turn=? AND state IN ('pending','running')",(state,req.user_id,req.conv_id,req.turn_id))

    def finish_turn(self, req, answer, *, state='completed'):
        if not isinstance(answer,str) or not answer.strip():
            raise ValueError('final answer cannot be empty')
        with self._lock, self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT * FROM turns WHERE user=? AND conv=? AND turn=?',(req.user_id,req.conv_id,req.turn_id)).fetchone()
            if not row:
                raise ValueError('turn was not received')
            if row['state'] in ('completed','incomplete'):
                if row['answer'] != answer:
                    raise ValueError('final answer already committed')
                return False
            self._record(req,'','dialogue','assistant',{'content':answer})
            self.db.execute("UPDATE turns SET state=?,answer=? WHERE seq=?",(state,answer,row['seq']))
            return True

    def completed_count(self, user_id):
        with self._lock:
            return self.db.execute("SELECT count(*) FROM turns WHERE user=? AND state='completed'",(user_id,)).fetchone()[0]

    def publish_result(self, req, run_id, result):
        with self._lock, self.db:
            self._run(req,run_id)
            existing = self.db.execute("SELECT body FROM records WHERE run=? AND kind='agent_result'",(run_id,)).fetchone()
            if existing:
                if existing['body'] != dumps(result):
                    raise ValueError('result already published')
                return
            self._record(req,run_id,'agent_result','assistant',result)

    def query(self, req, *, scope='conversation', run_id=None, record_type=None, keyword=None, record_ids=None, limit=5):
        with self._lock:
            if scope=='instant':
                self._run(req,run_id)
                conditions, args = ['user=?','conv=?','run=?','kind=?'], [req.user_id,req.conv_id,run_id,'execution']
            elif scope=='conversation':
                kind = record_type or 'dialogue'
                if kind not in ('dialogue','summary','agent_result','evidence'):
                    raise ValueError('unsupported conversation record type')
                conditions,args = ['user=?','conv=?','kind=?'], [req.user_id,req.conv_id,kind]
            elif scope=='user':
                if record_type not in (None,'profile'):
                    raise ValueError('user scope only exposes active profile')
                rows = self.db.execute("SELECT * FROM profiles WHERE user=? AND state='active' ORDER BY revision DESC",(req.user_id,)).fetchall()
                return [{'record_id':r['id'],'record_type':'profile','content':dumps({r['key']:json.loads(r['value'])}),'source_refs':json.loads(r['sources']),'revision':r['revision'],'state':r['state'],'created_at':r['created']}
                        for r in rows if (not record_ids or r['id'] in record_ids) and (not keyword or keyword in r['value'])][:limit]
            else:
                raise ValueError('unknown scope')
            rows = self.db.execute('SELECT * FROM records WHERE '+' AND '.join(conditions)+' ORDER BY seq DESC', args).fetchall()
            # Literal search on decoded content supports Chinese and treats SQL punctuation as text.
            result = []
            for r in rows:
                body = json.loads(r['body'])
                content = body.get('content') if isinstance(body,dict) else None
                if content is None:
                    content = dumps(body)
                if (record_ids and r['id'] not in record_ids) or (keyword and keyword not in content):
                    continue
                result.append({'record_id':r['id'],'record_type':r['kind'],'turn_id':r['turn'],
                               'role':r['role'],'content':content,'source_refs':[],'revision':r['seq'],
                               'state':'raw','created_at':r['created']})
                if len(result)>=limit:
                    break
            return result

    def commit_short_summary(self, req, snapshot, text):
        if not text.strip() or not snapshot['older_source_ids']:
            return False
        with self._lock, self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT revision FROM summaries WHERE user=? AND conv=?',(req.user_id,req.conv_id)).fetchone()
            revision = row['revision'] if row else 0
            if revision != snapshot['summary_revision']:
                return False
            self.db.execute('INSERT OR REPLACE INTO summaries VALUES(?,?,?,?,?)',
                            (req.user_id,req.conv_id,revision+1,dumps(snapshot['older_source_ids']),text))
            self._record(req,'','summary','system',{'content':text,'source_ids':snapshot['older_source_ids'],'revision':revision+1})
            return True

    def profile_batch(self,user_id):
        with self._lock, self.db:
            self.db.execute('BEGIN')
            user=self.db.execute('SELECT * FROM users WHERE user=?',(user_id,)).fetchone()
            # Final dialogue records are appended in completion order, not arrival order.
            turns=self.db.execute("SELECT t.* FROM turns t JOIN records r ON r.user=t.user AND r.conv=t.conv AND r.turn=t.turn WHERE t.user=? AND t.state='completed' AND r.kind='dialogue' AND r.role='assistant' ORDER BY r.seq",(user_id,)).fetchall()
            cutoff=len(turns)
            extracted=user['extracted_count'] if user else 0
            rows=[]
            for turn in turns[extracted:cutoff]:
                source=self.db.execute("SELECT * FROM records WHERE user=? AND conv=? AND turn=? AND role='user' AND kind='dialogue'",(user_id,turn['conv'],turn['turn'])).fetchone()
                if source:
                    rows.append({'record_id':source['id'],'role':'user','content':json.loads(source['body'])['content'],
                                 'turn_id':source['turn'],'conv_id':source['conv']})
            return {'count':cutoff,'extracted_count':extracted,'profile_revision':user['profile_rev'] if user else 0,'rows':rows}

    def commit_profile(self, user_id, changes, *, base_revision, cutoff):
        with self._lock, self.db:
            self.db.execute('BEGIN IMMEDIATE')
            user = self.db.execute('SELECT * FROM users WHERE user=?',(user_id,)).fetchone()
            if not user or user['profile_rev'] != base_revision:
                return False
            if cutoff < user['extracted_count'] or cutoff > self.completed_count(user_id):
                return False
            covered_turns=self.db.execute("SELECT t.conv,t.turn FROM turns t JOIN records r ON r.user=t.user AND r.conv=t.conv AND r.turn=t.turn WHERE t.user=? AND t.state='completed' AND r.kind='dialogue' AND r.role='assistant' ORDER BY r.seq LIMIT ?",(user_id,cutoff)).fetchall()
            covered={(t['conv'],t['turn']) for t in covered_turns}
            validated = []
            for change in changes:
                key, operation = change.get('key'), change.get('operation')
                if key not in ('response_style','language','citation_preference','research_direction','writing_language','submission_experience','ongoing_requirement') or operation not in ('add','replace','retract','conflict'):
                    raise ValueError('unsupported profile operation')
                sources = change.get('source_ids',[])
                if not sources:
                    raise ValueError('profile requires user sources')
                for source in sources:
                    row = self.db.execute("SELECT r.id,r.conv,r.turn FROM records r JOIN turns t ON r.user=t.user AND r.conv=t.conv AND r.turn=t.turn WHERE r.id=? AND r.user=? AND r.kind='dialogue' AND r.role='user' AND t.state='completed'",(source,user_id)).fetchone()
                    if not row or (row['conv'],row['turn']) not in covered:
                        raise ValueError('invalid profile source')
                validated.append(change)
            revision = base_revision + 1
            for change in validated:
                operation = change['operation']
                if operation in ('replace','retract','conflict'):
                    old_state='disputed' if operation=='conflict' else 'retracted' if operation=='retract' else 'superseded'
                    self.db.execute("UPDATE profiles SET state=? WHERE user=? AND key=? AND state IN ('active','disputed')",(old_state,user_id,change['key']))
                elif operation=='add' and self.db.execute("SELECT 1 FROM profiles WHERE user=? AND key=? AND state='active'",(user_id,change['key'])).fetchone():
                    raise ValueError('existing profile requires replace or conflict')
                self.db.execute('INSERT INTO profiles VALUES(?,?,?,?,?,?,?,?,?)',
                                (uuid4().hex,user_id,change['key'],dumps(change.get('value')), 'disputed' if operation=='conflict' else 'retracted' if operation=='retract' else 'active',operation,dumps(change['source_ids']),revision,now()))
            self.db.execute('UPDATE users SET profile_rev=?,extracted_count=? WHERE user=?',(revision,cutoff,user_id))
            return True

    def trigger_compression(self, req, run_id, *, input_tokens, input_budget, keep_groups=2, trigger_tokens=None, measure=len):
        task = self._tasks.get(run_id)
        if task and not task.done():
            return task
        threshold=input_budget*.7 if trigger_tokens is None else trigger_tokens
        if self.compressor is None or input_tokens < threshold:
            return None
        with self._lock:
            run = self._run(req,run_id)
            if run['state']!='active':
                return None
            rows = self.db.execute("SELECT r.*,p.record AS projected FROM records r LEFT JOIN projections p ON r.run=p.run AND r.id=p.record WHERE r.run=? AND r.kind='execution' ORDER BY r.seq",(run_id,)).fetchall()
            complete = []
            for step in sorted({r['step'] for r in rows}):
                group = [r for r in rows if r['step']==step]
                ids = {c['id'] for r in group if r['role']=='assistant' for c in json.loads(r['body']).get('tool_calls',[])}
                returned = {json.loads(r['body']).get('tool_call_id') for r in group if r['role']=='tool'}
                if ids and ids == returned:
                    complete.append(step)
            eligible = set(complete[:-keep_groups]) if len(complete)>keep_groups else set()
            candidates = [{'record_id':r['id'],'content':json.loads(r['body'])['content']} for r in rows
                          if r['step'] in eligible and r['role']=='tool' and r['projected'] is None]
            selected=[]
            batch_limit=getattr(self.compressor,'max_input_bytes',24000)
            for record in candidates:
                if selected and (len(selected)>=4 or len(json.dumps(selected+[record],ensure_ascii=False).encode('utf-8'))>batch_limit):
                    break
                # A single large record is handled losslessly by ModelMemoryOps.
                selected.append(record)
            fingerprint = tuple(r['record_id'] for r in selected)
            base = run['projection_rev']
            attempt=(base,fingerprint)
            if not selected or self._attempted.get(run_id)==attempt:
                return None
            self._attempted[run_id] = attempt
        def event(reason,**extra):
            with self._lock,self.db:
                self._record(req,run_id,'compression','system',
                             {'reason':reason,'record_ids':list(fingerprint),'base_revision':base,**extra})
        async def compress():
            try:
                event('started')
                candidate = await asyncio.wait_for(self.compressor(selected),self.compression_timeout_s)
                if not isinstance(candidate,dict) or set(candidate)!=set(fingerprint):
                    event('invalid_record_ids')
                    return False
                if any(not isinstance(candidate[r['record_id']],str) or not candidate[r['record_id']].strip() for r in selected):
                    event('invalid_summary')
                    return False
                if sum(measure(candidate[r['record_id']]) for r in selected)>=sum(measure(r['content']) for r in selected):
                    event('not_smaller')
                    return False
                with self._lock, self.db:
                    self.db.execute('BEGIN IMMEDIATE')
                    current = self._run(req,run_id)
                    if current['state']!='active' or current['projection_rev']!=base:
                        self._record(req,run_id,'compression','system',{'reason':'version_conflict_or_sealed','base_revision':base})
                        return False
                    for record_id, text in candidate.items():
                        self.db.execute('INSERT OR REPLACE INTO projections VALUES(?,?,?,?)',(run_id,record_id,text,base+1))
                    self.db.execute('UPDATE runs SET projection_rev=? WHERE run=?',(base+1,run_id))
                    self._record(req,run_id,'compression','system',{'reason':'committed','record_ids':list(fingerprint),'revision':base+1})
                    return True
            except asyncio.CancelledError:
                event('cancelled')
                raise
            except TimeoutError:
                event('timeout')
                return False
            except Exception:
                event('compressor_error')
                return False
        task = asyncio.create_task(compress())
        self._tasks[run_id]=task
        return task

    async def seal_run(self, req, run_id, state):
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
        with self._lock, self.db:
            self._run(req,run_id)
            self.db.execute('UPDATE runs SET state=? WHERE run=?',(state,run_id))
