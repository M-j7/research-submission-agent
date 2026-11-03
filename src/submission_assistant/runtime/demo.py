"""Controlled offline dependencies; never factual policy answers."""
from submission_assistant.intent import IntentJudgments, BatchStatus

class DemoRecognizer:
    async def recognize(self,message,history=None):
        values={f'C{i:02d}':.01 for i in range(1,11)}
        values.update(C02=.99,C07=.99)
        batches=[BatchStatus(f'batch_{i}',f'demo-{i}','success',{}) for i in (1,2)]
        return IntentJudgments(values,.99,'controlled-demo','demo',batches)


class DemoModel:
    async def complete(self,messages,tools,*,allow_tools=True):
        if '汇总审核' in messages[0]['content']:
            text='离线演示：已收到规范与费用两个角色的正式结果，并合并为一份最终回答。此处不提供真实政策结论。'
        elif not any(m['role']=='tool' for m in messages) and allow_tools:
            return {'message':{'role':'assistant','content':'','tool_calls':[{'id':'demo-read','type':'function','function':{
                'name':'query_memory','arguments':'{"scope":"conversation","limit":2}'}}]},'finish_reason':'tool_calls','usage':None}
        else:
            text='离线角色演示：已完成一次合法的记忆查询，未调用真实模型或知识库。'
        return {'message':{'role':'assistant','content':text},'finish_reason':'stop','usage':None}


