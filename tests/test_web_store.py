from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from submission_assistant.runtime.memory import MemoryManager
from submission_assistant.runtime.contracts import Request


class StoreTests(unittest.TestCase):
    def setUp(self):
        from submission_assistant.web.store import ConversationStore
        self.temp = tempfile.TemporaryDirectory()
        self.memory = MemoryManager(Path(self.temp.name) / 'memory.db')
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.memory.close)
        self.store = ConversationStore(self.memory, 'local-user')

    def test_empty_conversation_and_title(self):
        conv = self.store.create()
        self.assertEqual(conv.title, '新会话')
        turn, created = self.store.reserve(conv.id, 't1', '中' * 40)
        self.assertTrue(created)
        self.assertEqual(self.store.get(conv.id).title, '中' * 30)
        self.assertEqual(turn.state, 'pending')

    def test_id_collision(self):
        from submission_assistant.web.store import TurnConflict
        conv = self.store.create()
        self.store.reserve(conv.id, 't1', '问题')
        self.assertFalse(self.store.reserve(conv.id, 't1', '问题')[1])
        with self.assertRaises(TurnConflict): self.store.reserve(conv.id, 't1', '不同问题')

    def test_scoped_conversations_and_legacy_history(self):
        from submission_assistant.web.store import ConversationStore, ResourceNotFound
        req = Request(user_id='local-user', conv_id='old', turn_id='old1', message='旧问题')
        self.memory.begin_turn(req)
        self.memory.finish_turn(req, '原文答案')
        self.assertEqual(self.store.get('old').turns[0].answer, '原文答案')
        other = ConversationStore(self.memory, 'someone-else')
        self.assertEqual(other.list(), [])
        with self.assertRaises(ResourceNotFound): other.get('old')

    def test_public_view_redacts_private_records(self):
        from submission_assistant.web.public_views import public_result, public_status
        raw = {'agent_type': 'general', 'answer_text': '答复', 'system_prompt': 'PRIVATE_TRACE_SENTINEL',
               'evidence': [{'title': '材料', 'source_path': 'C:/secret/private.pdf', 'source_url': 'file:///C:/secret',
                             'text': '参考内容', 'page': 2}], 'execution_status': 'incomplete', 'stop_reason': 'max_steps'}
        result = public_result(raw)
        self.assertNotIn('secret', str(result))
        self.assertNotIn('PRIVATE_TRACE_SENTINEL', str(result))
        self.assertEqual(result['evidence'][0]['page'], 2)
        self.assertEqual(public_status({'current_step': 2, 'messages': 'PRIVATE_TRACE_SENTINEL'}), {'current_step': 2})

    def test_restart_marks_only_local_jobs_interrupted(self):
        conv = self.store.create()
        self.store.reserve(conv.id, 't1', '新问题')
        self.assertEqual(self.store.interrupt_pending(), 1)
        self.assertEqual(self.store.get_turn(conv.id, 't1').state, 'interrupted')

    def test_tool_status_allows_only_known_names_states_and_no_arguments(self):
        from submission_assistant.web.public_views import public_status
        raw = {'phase':'tool_call','tools':[
            {'name':'search_knowledge','state':'running','arguments':'PRIVATE_QUERY','result':'PRIVATE_RESULT'},
            {'name':'read','state':'timed_out'},
            {'name':'PRIVATE_TOOL_NAME','state':'running'},
            {'name':'glob','state':'PRIVATE_STATE'}, None], 'prompt':'PRIVATE_PROMPT'}
        public = public_status(raw)
        self.assertEqual(public, {'phase':'tool_call','tools':[
            {'name':'search_knowledge','state':'running'}, {'name':'read','state':'timed_out'}]})
        self.assertNotIn('PRIVATE_', str(public))

    def test_unmanaged_legacy_running_turn_is_not_resumable(self):
        req = Request(user_id='local-user', conv_id='legacy', turn_id='old', message='旧问题')
        self.memory.begin_turn(req)
        self.memory.claim_turn(req)
        self.assertEqual(self.store.get('legacy').turns[0].state, 'interrupted')
        self.assertEqual(self.memory.begin_turn(req)['state'], 'running')  # raw record not rewritten
