"""Manual browser QA only: synthetic results, never used by the product launcher."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from submission_assistant.runtime.factory import create_runtime
from submission_assistant.web.app import create_app


class PreviewGraph:
    def __init__(self, memory):
        self.memory = memory

    async def astream(self, state, **kwargs):
        req = state['request']
        self.memory.begin_turn(req)
        self.memory.claim_turn(req)
        yield {'receive': {}}
        await asyncio.sleep(1)
        if '失败' in req.message:
            raise RuntimeError('offline simulated error; not a model call')
        result = {'agent_type':'submission_requirements', 'intent_ids':['C02'],
                  'execution_status':'incomplete', 'stop_reason':'max_steps',
                  'unresolved':['这是离线部分完成场景，不是期刊政策。'],
                  'evidence':[{'title':'离线验收资料（非政策）','source_url':'https://example.com',
                               'source_locator':'演示章节','text':'用于检查证据卡片和部分结果展示。','page':2}]}
        yield {'recognize_and_prepare': {'routing_decision': {'agent_types':['submission_requirements'],
                'matched_intents':['C02'],'mode':'single','fallback_reason':None}}}
        yield {'compose': {'final_result': result}}
        self.memory.finish_turn(req, '离线验收部分答复：已验证证据卡片，未给出真实政策结论。', state='incomplete')
        yield {'commit': {}}


async def factory(project, *, mode):
    return await create_runtime(project, mode='demo', data_dir=ROOT / '.cache' / 'workbench-browser-qa')


app = create_app(ROOT, mode='demo', port=8001, runtime_factory=factory)
original_lifespan = app.router.lifespan_context


@asynccontextmanager
async def lifespan(app):
    async with original_lifespan(app):
        app.state.jobs.graph = PreviewGraph(app.state.bundle.service.memory)
        yield


app.router.lifespan_context = lifespan
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8001, access_log=False)
