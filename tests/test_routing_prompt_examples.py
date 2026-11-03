"""Check few-shot assembly at the real outbound request boundary, not LLM semantics."""
import copy
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import httpx
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from submission_assistant.intent.jev import build_requests, JevIntentRecognizer, JevSettings, IntentInputError


class RoutingPromptExampleTests(unittest.TestCase):
    def test_default_online_client_preserves_baseline_after_candidate_regression(self):
        frozen = json.loads((Path(__file__).resolve().parents[1] /
            'evaluation/agent/prompt-probes/fewshot-v2/baseline-questions.json').read_text(encoding='utf-8'))
        sent = {}
        def handle(request):
            body = json.loads(request.content)
            sent.update(body['questions'])
            return httpx.Response(200, json={'model': body['model'], 'answers': {
                k: {'type': 'noul', 'noul': .01} for k in body['questions']}})
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                result = await JevIntentRecognizer(JevSettings('https://api.perplexity.ai', 'fixture', provider='decisions'), client=client).recognize('任何实际输入')
            self.assertTrue(result.complete)
        asyncio.run(run())
        self.assertEqual(sent, frozen)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.intent.routing_prompts'),
                             'structured, labeled routing demonstrations missing')
        return importlib.import_module('submission_assistant.intent.routing_prompts')

    def test_labeled_examples_reach_question_instructions_without_becoming_real_history(self):
        api = self.api()
        spec = api.PromptSpec('判断材料提交要求', '系统故障不属于材料要求', (
            api.Example('虚构示例问题', True, (('示例旧问题', '示例旧回答'),)),
            api.Example('虚构反例问题', False),))
        history = [{'role': 'user', 'content': '真实旧问题'}, {'role': 'assistant', 'content': '真实旧回答'}]
        original = copy.deepcopy(history)
        with patch.dict(api.PROMPT_SPECS, {'C03': spec}):
            requests = build_requests('真实本轮问题', history, provider='decisions', prompt_version=api.ROUTING_PROMPT_VERSION)
        question = requests[0]['questions']['C03']
        examples = json.loads(question['instructions'].split('[示例，仅用于说明判断标准]\n', 1)[1])
        self.assertEqual(examples, [
            {'history': [{'role': 'user', 'content': '示例旧问题'}, {'role': 'assistant', 'content': '示例旧回答'}],
             'current_message': '虚构示例问题', 'expected': True},
            {'history': [], 'current_message': '虚构反例问题', 'expected': False}])
        self.assertEqual(requests[0]['state'], {
            'domain': 'IEEE Access 投稿咨询', 'current_message': '真实本轮问题', 'history': original})
        self.assertEqual(history, original)
        self.assertEqual(set(question), {'type', 'instructions', 'criteria'})
        self.assertEqual(question['criteria'], {'true': '判断材料提交要求', 'false': '系统故障不属于材料要求'})

    def test_each_outbound_category_contains_positive_and_negative_demonstrations(self):
        api = self.api()
        bodies = build_requests('尚未联网的测试输入', provider='decisions', prompt_version=api.ROUTING_PROMPT_VERSION)
        questions = {k: q for b in bodies for k, q in b['questions'].items()}
        for key in [f'C{i:02d}' for i in range(1, 11)]:
            with self.subTest(key=key):
                q = questions[key]
                examples = json.loads(q['instructions'].split('[示例，仅用于说明判断标准]\n', 1)[1])
                self.assertTrue(any(e['expected'] is True for e in examples))
                self.assertTrue(any(e['expected'] is False for e in examples))
                self.assertTrue(4 <= len(examples) <= 6)
                self.assertLessEqual(len(q['instructions']), 1800)
                for e in examples:
                    self.assertEqual([m['role'] for m in e['history']], ['user', 'assistant'] * (len(e['history']) // 2))
