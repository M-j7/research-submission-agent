"""Scripted protection checks, not evidence of general LLM robustness."""
import json
from submission_assistant.intent.contracts import BatchStatus, IntentJudgments


class FaultRecognizer:
    async def recognize(self, message, history=None):
        # v2: tool-loop protection belongs to a tool-capable specialist, not General.
        return IntentJudgments({f'C{i:02d}': .99 if i == 2 else .01 for i in range(1, 11)},
                               .99, 'scripted-fault', 'fault',
                               [BatchStatus('batch_1', 'f1', 'success', {}), BatchStatus('batch_2', 'f2', 'success', {})])


class FaultModel:
    model = 'scripted-fault-not-quality-model'
    def __init__(self, fault):
        self.fault, self.step = fault, 0

    async def complete(self, messages, tools, *, allow_tools=True):
        self.step += 1
        command = None
        if self.fault == 'no_progress':
            command = ('glob', {'root': '/private/scratch', 'pattern': 'missing-*'})
        elif self.fault == 'invalid_argument':
            command = ('read', {'ref': 42, 'max_lines': -1})
        elif self.fault == 'timeout' and self.step == 1:
            command = ('query_memory', {'scope': 'conversation'})
        elif self.fault == 'unauthorized' and self.step == 1:
            command = ('read', {'ref': '/outside/private.env'})
        elif self.fault == 'injection' and self.step <= 2:
            command = ('read', {'ref': '/system/manuals/injection.md' if self.step == 1 else '/outside/private.env'})
        if command and allow_tools:
            name, args = command
            return {'message': {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': f'f{self.step}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]},
                'finish_reason': 'tool_calls', 'usage': None}
        return {'message': {'role': 'assistant', 'content': '工具未成功取得信息，未执行越权操作，无法声称完成。'},
                'finish_reason': 'stop', 'usage': None}
