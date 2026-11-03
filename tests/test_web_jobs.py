import asyncio
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from submission_assistant.runtime.factory import create_runtime
from submission_assistant.web.store import ConversationStore, TurnConflict


class ControlledGraph:
    def __init__(self, memory):
        self.memory = memory
        self.calls = 0
        self.gate = asyncio.Event()
        self.fail = False
        self.result_state = 'completed'

    async def astream(self, state, **kwargs):
        self.calls += 1
        req = state['request']
        self.memory.begin_turn(req)
        self.memory.claim_turn(req)
        yield {'receive': {}}
        await self.gate.wait()
        if self.fail: raise RuntimeError('SECRET_SENTINEL remote raw error')
        self.memory.finish_turn(req, '已保存的答复', state=self.result_state)
        yield {'commit': {'answer': '已保存的答复'}}


class JobsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from submission_assistant.web.jobs import JobController
        self.temp = tempfile.TemporaryDirectory()
        self.bundle = await create_runtime(ROOT, mode='demo', data_dir=Path(self.temp.name))
        self.store = ConversationStore(self.bundle.service.memory, 'local-user')
        self.jobs = JobController(self.bundle, self.store)
        self.jobs.graph = self.graph = ControlledGraph(self.bundle.service.memory)
        self.conv = self.store.create().id

    async def asyncTearDown(self):
        await self.jobs.aclose()
        await self.bundle.aclose()
        self.temp.cleanup()

    async def wait_terminal(self, conv, turn):
        async for event in self.jobs.events(conv, turn):
            if event['turn'].state not in ('pending', 'running'): return event['turn']

    async def test_duplicate_post_runs_once(self):
        await self.jobs.submit(self.conv, 't1', '问题')
        await self.jobs.submit(self.conv, 't1', '问题')
        self.graph.gate.set()
        result = await asyncio.wait_for(self.wait_terminal(self.conv, 't1'), 2)
        self.assertEqual(result.answer, '已保存的答复')
        self.assertEqual(self.graph.calls, 1)
        await self.jobs.submit(self.conv, 't1', '问题')
        self.assertEqual(self.graph.calls, 1)

    async def test_same_id_different_message_conflicts(self):
        await self.jobs.submit(self.conv, 't1', '问题')
        with self.assertRaises(TurnConflict): await self.jobs.submit(self.conv, 't1', '别的问题')

    async def test_parallel_capacity_and_release(self):
        from submission_assistant.web.jobs import ConversationBusy, CapacityBusy
        await self.jobs.submit(self.conv, 't1', '问题')
        with self.assertRaises(ConversationBusy): await self.jobs.submit(self.conv, 't2', '问题')
        c2, c3 = self.store.create().id, self.store.create().id
        await self.jobs.submit(c2, 't1', '问题')
        with self.assertRaises(CapacityBusy): await self.jobs.submit(c3, 't1', '问题')
        self.graph.gate.set()
        await self.wait_terminal(self.conv, 't1')
        await self.wait_terminal(c2, 't1')
        await self.jobs.submit(c3, 't1', '问题')
        self.assertEqual((await self.wait_terminal(c3, 't1')).state, 'completed')

    async def test_disconnect_does_not_cancel_job(self):
        await self.jobs.submit(self.conv, 't1', '问题')
        stream = self.jobs.events(self.conv, 't1')
        await anext(stream)
        await stream.aclose()
        self.graph.gate.set()
        self.assertEqual((await self.wait_terminal(self.conv, 't1')).state, 'completed')

    async def test_finish_requires_persistence(self):
        self.graph.fail = True
        await self.jobs.submit(self.conv, 't1', '问题')
        self.graph.gate.set()
        result = await self.wait_terminal(self.conv, 't1')
        self.assertEqual(result.state, 'failed')
        self.assertIsNone(result.answer)
        self.assertNotIn('SECRET_SENTINEL', result.model_dump_json())

    async def test_restart_does_not_replay(self):
        self.store.reserve(self.conv, 'old', '旧问题')
        self.store.interrupt_pending()
        view = await self.jobs.submit(self.conv, 'old', '旧问题')
        self.assertEqual(view.state, 'interrupted')
        self.assertEqual(self.graph.calls, 0)

    async def test_partial_result_status_is_not_completed(self):
        self.graph.result_state = 'incomplete'
        await self.jobs.submit(self.conv, 't1', '问题')
        self.graph.gate.set()
        result = await self.wait_terminal(self.conv, 't1')
        self.assertEqual(result.state, 'incomplete')
        self.assertEqual(result.status[0]['phase'], 'incomplete')

    async def test_cancel_before_job_start_marks_interrupted(self):
        await self.jobs.submit(self.conv, 't1', '问题')
        await self.jobs.aclose()  # no yield between task creation and cancellation
        self.assertEqual(self.store.get_turn(self.conv, 't1').state, 'interrupted')
        self.assertEqual(len(self.jobs._tasks), 0)

    async def test_shutdown_is_bounded_when_tool_suppresses_cancellation(self):
        started, release = asyncio.Event(), asyncio.Event()
        class ResistantGraph:
            async def astream(self, state, **kwargs):
                started.set()
                yield {'receive': {}}
                try: await asyncio.Event().wait()
                except asyncio.CancelledError: await release.wait()
        self.jobs.graph = ResistantGraph()
        self.jobs.shutdown_timeout_s = .01
        await self.jobs.submit(self.conv, 't1', '问题')
        await started.wait()
        try:
            with self.assertRaises(RuntimeError):
                await asyncio.wait_for(self.jobs.aclose(), .3)
            with self.bundle.service.memory.transaction() as db:
                self.assertEqual(db.execute('SELECT 1').fetchone()[0], 1)
        finally:
            release.set()
            tasks = list(self.jobs._tasks.values())
            if tasks: await asyncio.gather(*tasks, return_exceptions=True)

    async def test_slow_subscriber_is_bounded(self):
        await self.jobs.submit(self.conv, 't1', '问题')
        stream = self.jobs.events(self.conv, 't1')
        await anext(stream)
        self.graph.gate.set()
        await self.wait_terminal(self.conv, 't1')
        self.assertEqual((await anext(stream))['turn'].state, 'completed')
        await stream.aclose()
