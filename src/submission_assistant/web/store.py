"""Web metadata shares MemoryManager's connection, lock and transaction boundary."""
from __future__ import annotations
import json
from uuid import uuid4

from ..runtime.memory import now, dumps
from .contracts import ConversationSummary, ConversationView, TurnView, TurnPatch
from .public_views import public_result, public_routing, public_status


class ResourceNotFound(LookupError):
    pass


class TurnConflict(ValueError):
    pass


class ConversationStore:
    def __init__(self, memory, user_id: str):
        self.memory, self.user_id = memory, user_id
        with memory.transaction() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS web_conversations(user TEXT, conv TEXT, title TEXT, updated TEXT,
                PRIMARY KEY(user,conv));
              CREATE TABLE IF NOT EXISTS web_turns(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT, conv TEXT, turn TEXT, message TEXT, state TEXT, metadata TEXT DEFAULT '{}',
                UNIQUE(user,conv,turn));
            ''')

    def _adopt(self, db):
        rows = db.execute('SELECT conv, min(seq) AS first_seq FROM turns WHERE user=? GROUP BY conv', (self.user_id,)).fetchall()
        for row in rows:
            first = db.execute('SELECT message,created FROM turns WHERE seq=?', (row['first_seq'],)).fetchone()
            db.execute('INSERT OR IGNORE INTO web_conversations VALUES(?,?,?,?)',
                       (self.user_id, row['conv'], self._title(first['message']), first['created']))

    @staticmethod
    def _title(message):
        return ' '.join(message.split())[:30] or '新会话'

    def _owned(self, db, conv_id):
        self._adopt(db)
        row = db.execute('SELECT * FROM web_conversations WHERE user=? AND conv=?', (self.user_id, conv_id)).fetchone()
        if row is None:
            raise ResourceNotFound('会话不存在')
        return row

    def create(self) -> ConversationSummary:
        conv, timestamp = uuid4().hex, now()
        with self.memory.transaction() as db:
            db.execute('INSERT INTO web_conversations VALUES(?,?,?,?)', (self.user_id, conv, '新会话', timestamp))
        return ConversationSummary(id=conv, title='新会话', updated_at=timestamp)

    def list(self) -> list[ConversationSummary]:
        with self.memory.transaction() as db:
            self._adopt(db)
            return [ConversationSummary(id=r['conv'], title=r['title'], updated_at=r['updated'])
                    for r in db.execute('SELECT * FROM web_conversations WHERE user=? ORDER BY updated DESC,conv', (self.user_id,))]

    def _turn(self, db, conv_id, turn_id):
        web = db.execute('SELECT * FROM web_turns WHERE user=? AND conv=? AND turn=?', (self.user_id, conv_id, turn_id)).fetchone()
        core = db.execute('SELECT * FROM turns WHERE user=? AND conv=? AND turn=?', (self.user_id, conv_id, turn_id)).fetchone()
        if web is None and core is None:
            raise ResourceNotFound('轮次不存在')
        meta = json.loads(web['metadata']) if web else {}
        state = core['state'] if core and core['state'] in ('completed','incomplete') else (web['state'] if web else core['state'])
        if web is None and state in ('pending', 'running'):
            state = 'interrupted'  # not a task this web process can resume; raw record stays intact
        if state not in ('pending','running','completed','incomplete','failed','cancelled','interrupted'):
            state = 'interrupted'
        return TurnView(turn_id=turn_id, message=web['message'] if web else core['message'], state=state,
                        answer=core['answer'] if core else None, **meta)

    def get(self, conv_id: str) -> ConversationView:
        with self.memory.transaction() as db:
            row = self._owned(db, conv_id)
            core_ids = [r['turn'] for r in db.execute('SELECT turn FROM turns WHERE user=? AND conv=? ORDER BY seq', (self.user_id, conv_id))]
            web_ids = [r['turn'] for r in db.execute('SELECT turn FROM web_turns WHERE user=? AND conv=? ORDER BY seq', (self.user_id, conv_id))]
            ids = list(dict.fromkeys(core_ids + web_ids))
            return ConversationView(id=conv_id, title=row['title'], turns=[self._turn(db, conv_id, t) for t in ids])

    def get_turn(self, conv_id: str, turn_id: str) -> TurnView:
        with self.memory.transaction() as db:
            self._owned(db, conv_id)
            return self._turn(db, conv_id, turn_id)

    def reserve(self, conv_id: str, turn_id: str, message: str) -> tuple[TurnView, bool]:
        with self.memory.transaction() as db:
            self._owned(db, conv_id)
            try:
                old = self._turn(db, conv_id, turn_id)
            except ResourceNotFound:
                old = None
            if old:
                if old.message != message:
                    raise TurnConflict('轮次 ID 已被另一条消息使用')
                return old, False
            first = not db.execute('SELECT 1 FROM web_turns WHERE user=? AND conv=?', (self.user_id, conv_id)).fetchone()
            db.execute('INSERT INTO web_turns(user,conv,turn,message,state) VALUES(?,?,?,?,?)',
                       (self.user_id, conv_id, turn_id, message, 'pending'))
            if first:
                db.execute('UPDATE web_conversations SET title=?,updated=? WHERE user=? AND conv=?',
                           (self._title(message), now(), self.user_id, conv_id))
            return self._turn(db, conv_id, turn_id), True

    def update_public(self, conv_id: str, turn_id: str, patch: TurnPatch) -> TurnView:
        with self.memory.transaction() as db:
            self._owned(db, conv_id)
            row = db.execute('SELECT * FROM web_turns WHERE user=? AND conv=? AND turn=?', (self.user_id, conv_id, turn_id)).fetchone()
            if row is None:
                raise ResourceNotFound('轮次不存在')
            changes = patch.model_dump(exclude_unset=True)
            state = changes.pop('state', None) or row['state']
            meta = json.loads(row['metadata'])
            if 'routing' in changes:
                changes['routing'] = public_routing(changes['routing']) if changes['routing'] else None
            if 'agent_results' in changes:
                changes['agent_results'] = [public_result(r) for r in changes['agent_results'] or []]
            if 'status' in changes:
                changes['status'] = [public_status(s) for s in changes['status'] or []]
            meta.update(changes)
            db.execute('UPDATE web_turns SET state=?,metadata=? WHERE user=? AND conv=? AND turn=?',
                       (state, dumps(meta), self.user_id, conv_id, turn_id))
            db.execute('UPDATE web_conversations SET updated=? WHERE user=? AND conv=?', (now(), self.user_id, conv_id))
            return self._turn(db, conv_id, turn_id)

    def interrupt_pending(self) -> int:
        with self.memory.transaction() as db:
            rows = db.execute("SELECT conv,turn FROM web_turns WHERE user=? AND state IN ('pending','running')", (self.user_id,)).fetchall()
            for row in rows:
                db.execute("UPDATE web_turns SET state='interrupted' WHERE user=? AND conv=? AND turn=?", (self.user_id, row['conv'], row['turn']))
                db.execute("UPDATE turns SET state='interrupted' WHERE user=? AND conv=? AND turn=? AND state IN ('pending','running')", (self.user_id, row['conv'], row['turn']))
            return len(rows)

    def statuses(self, conv_id: str, turn_id: str) -> list[dict]:
        with self.memory.transaction() as db:
            self._owned(db, conv_id)
            return [public_status({**json.loads(r['status']), 'agent_type': r['role']}) for r in
                    db.execute('SELECT role,status FROM runs WHERE user=? AND conv=? AND turn=?', (self.user_id, conv_id, turn_id))]
