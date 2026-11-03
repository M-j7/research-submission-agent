"""Offline browser QA for real progress plumbing; never used by the live launcher."""
import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from submission_assistant.runtime.demo import DemoRecognizer
from submission_assistant.runtime.factory import create_runtime
from submission_assistant.web.app import create_app


class SlowRecognizer(DemoRecognizer):
    async def recognize(self, message, history=None):
        await asyncio.sleep(1)
        return await super().recognize(message, history)


class PreviewModels:
    def for_role(self, role):
        class Model:
            async def complete(self, messages, tools, *, allow_tools=True):
                await asyncio.sleep(2)
                if role != 'composer' and not any(m['role']=='tool' for m in messages) and allow_tools:
                    return {'message':{'role':'assistant','content':'','tool_calls':[
                        {'id':'qa-search','type':'function','function':{'name':'search_knowledge','arguments':'{"query":"离线进度验收"}'}},
                        {'id':'qa-memory','type':'function','function':{'name':'query_memory','arguments':'{"scope":"conversation","limit":2}'}}]},
                        'finish_reason':'tool_calls'}
                return {'message':{'role':'assistant','content':'离线验收完成：只验证执行进度展示，没有调用模型或真实知识库。'},'finish_reason':'stop'}
        return Model()


class PreviewRag:
    async def search(self, query, *, top_k):
        await asyncio.sleep(12)
        return {'evidence': []}


async def factory(project, *, mode):
    bundle = await create_runtime(project, mode='demo', data_dir=ROOT/'.cache'/'progress-browser-qa')
    bundle.service.recognizer = SlowRecognizer()
    bundle.service.model = PreviewModels()
    bundle.service.rag = PreviewRag()
    return bundle


app = create_app(ROOT, mode='demo', port=8002, runtime_factory=factory)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8002, access_log=False)
