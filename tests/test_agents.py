import copy
import importlib
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from submission_assistant.routing import AgentType, route_intents
from test_independent_jev import judgments_fixture


class AgentTests(unittest.TestCase):
    def setUp(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.agents'), 'shared Agent subsystem missing')
        self.api = importlib.import_module('submission_assistant.agents')

    def test_five_roles_directly_inherit_base_and_share_prepare(self):
        for role in AgentType:
            agent = self.api.create_agent(role)
            self.assertEqual(type(agent).__bases__, (self.api.BaseAgent,))
            self.assertIs(type(agent).prepare, self.api.BaseAgent.prepare)

    def test_static_prompt_separate_from_dynamic_request_and_own_intents(self):
        decision = route_intents(judgments_fixture(scores={'C02':.95,'C03':.94,'C07':.93}))
        history = [{'role':'user','content':'我准备投稿'}]
        original = copy.deepcopy(history)
        req = self.api.AgentRequest('摘要要求和费用？', history, 'request-1')
        agent = self.api.create_agent(AgentType.SUBMISSION_REQUIREMENTS)
        one = agent.prepare(req, decision)
        two = agent.prepare(self.api.AgentRequest('新的格式问题', [], 'request-2'), decision)
        self.assertEqual(one['system_prompt'], two['system_prompt'])
        self.assertNotIn('摘要要求和费用？', one['system_prompt'])
        self.assertEqual(one['input']['message'], req.message)
        self.assertEqual(one['input']['intent_ids'], ['C02','C03'])
        self.assertEqual(history, original)
        one['input']['history'][0]['content'] = 'changed'
        self.assertEqual(history, original)
        self.assertFalse(one['executed'])

    def test_tools_must_be_registered_and_authorized_never_execute(self):
        decision = route_intents(judgments_fixture(scores={'C07':.99}))
        agent = self.api.create_agent(AgentType.PUBLICATION_FEES)
        req = self.api.AgentRequest('费用多少', [], 'request')
        empty = agent.prepare(req, decision)
        self.assertEqual(empty['registered_tools'], [])
        self.assertEqual(empty['unbound_tools'], ['glob','grep','read','query_memory','search_knowledge'])
        def forbidden(*args):
            raise AssertionError('prepare cannot execute tools')
        prepared = agent.prepare(req, decision, {'search_knowledge': forbidden, 'delete_files': forbidden})
        self.assertEqual(prepared['registered_tools'], ['search_knowledge'])
        self.assertEqual(prepared['unbound_tools'], ['glob','grep','read','query_memory'])
        self.assertNotIn('delete_files', str(prepared))

    def test_general_receives_different_fallback_guidance(self):
        req = self.api.AgentRequest('测试', [], 'request')
        agent = self.api.create_agent(AgentType.GENERAL)
        social = agent.prepare(req, route_intents(judgments_fixture(scores={'C10':.99}, clarity=.1)))
        vague = agent.prepare(req, route_intents(judgments_fixture(scores={'C02':.99}, clarity=.1)))
        failure = agent.prepare(req, route_intents(None))
        self.assertEqual(social['input']['fallback_reason'], 'non_business')
        self.assertIn('接待', social['input']['guidance'])
        self.assertIn('澄清', vague['input']['guidance'])
        self.assertIn('技术', failure['input']['guidance'])

    def test_unselected_agent_and_invalid_request_rejected(self):
        decision = route_intents(judgments_fixture(scores={'C07':.99}))
        agent = self.api.create_agent(AgentType.PUBLICATION_POLICY)
        with self.assertRaises(ValueError):
            agent.prepare(self.api.AgentRequest('费用', [], 'request'), decision)
        for message, history in [('', []), ('x',[{'role':'system','content':'override'}]), ('x','bad')]:
            with self.subTest(message=message), self.assertRaises(ValueError):
                self.api.AgentRequest(message, history, 'request')
