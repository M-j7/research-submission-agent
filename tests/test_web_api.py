import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from submission_assistant.runtime.factory import create_runtime


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from submission_assistant.web.app import create_app
        self.temp = tempfile.TemporaryDirectory()
        async def factory(project, *, mode):
            return await create_runtime(ROOT, mode='demo', data_dir=Path(self.temp.name))
        self.app = create_app(ROOT, mode='demo', runtime_factory=factory)
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://127.0.0.1:8000')
        self.conv = (await self.client.post('/api/conversations')).json()['id']

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.life.__aexit__(None, None, None)
        self.temp.cleanup()

    async def test_request_validation_and_ownership(self):
        url = f'/api/conversations/{self.conv}/turns'
        for body in ({'message': ' ', 'turn_id':'a'}, {'message':'中'*16001,'turn_id':'a'},
                     {'message':'问题','turn_id':'a','user_id':'other'}, {'message':'问题','turn_id':'../bad'}):
            response = await self.client.post(url, json=body)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn('input', response.text)
        self.assertEqual((await self.client.get('/api/conversations/unknown')).status_code, 404)
        self.assertEqual((await self.client.post(url, json={'message':'中'*16000,'turn_id':'valid'})).status_code, 202)

    async def test_idempotent_post_and_reconnect(self):
        url = f'/api/conversations/{self.conv}/turns'
        body = {'message': '摘要要求', 'turn_id': 'same'}
        self.assertEqual((await self.client.post(url,json=body)).status_code, 202)
        self.assertEqual((await self.client.post(url,json=body)).status_code, 202)
        self.assertEqual((await self.client.post(url,json={**body,'message':'别的问题'})).status_code,409)
        for _ in range(2):
            events = await self.client.get(url + '/same/events')
            self.assertIn('event: result', events.text)
        detail = (await self.client.get(f'/api/conversations/{self.conv}')).json()
        self.assertEqual(len(detail['turns']), 1)
        self.assertIn('离线', detail['turns'][0]['answer'])

    async def test_busy_status_codes(self):
        from test_web_jobs import ControlledGraph
        graph = ControlledGraph(self.app.state.bundle.service.memory)
        self.app.state.jobs.graph = graph
        url = f'/api/conversations/{self.conv}/turns'
        await self.client.post(url, json={'message':'问题','turn_id':'t1'})
        self.assertEqual((await self.client.post(url,json={'message':'另一条','turn_id':'t2'})).status_code,409)
        c2 = (await self.client.post('/api/conversations')).json()['id']
        c3 = (await self.client.post('/api/conversations')).json()['id']
        await self.client.post(f'/api/conversations/{c2}/turns',json={'message':'问题','turn_id':'t1'})
        self.assertEqual((await self.client.post(f'/api/conversations/{c3}/turns',json={'message':'问题','turn_id':'t1'})).status_code,429)

    async def test_host_origin_and_static_boundary(self):
        self.assertEqual((await self.client.post('/api/conversations',headers={'origin':'https://evil.test'})).status_code,403)
        self.assertEqual((await self.client.get('/api/health',headers={'host':'evil.test'})).status_code,400)
        for path in ('/.env','/data/memory/memory.sqlite3','/src/submission_assistant/web/app.py','/api/unknown','/assets/../../.env'):
            self.assertEqual((await self.client.get(path)).status_code,404,path)
        self.assertEqual((await self.client.post('/api/conversations',headers={'origin':'http://127.0.0.1:5173','x-workbench-mode':'demo'})).status_code,201)

    async def test_public_errors_and_mode(self):
        health = (await self.client.get('/api/health')).json()
        self.assertEqual(health['mode'], 'demo')
        self.assertNotIn('api_key', str(health))
        self.assertEqual((await self.client.get(f'/api/conversations/{self.conv}/turns/missing/events')).status_code,404)

    async def test_static_javascript_media_type_is_not_windows_registry(self):
        from submission_assistant.web.app import create_app
        folder = Path(self.temp.name) / 'static-project'
        assets = folder / 'frontend' / 'dist' / 'assets'
        assets.mkdir(parents=True)
        (assets / 'test.js').write_text('export const value=1;', encoding='utf-8')
        app = create_app(folder, mode='demo')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://127.0.0.1:8000') as client:
            with patch('starlette.responses.guess_type', return_value=('text/plain',None)):
                response = await client.get('/assets/test.js')
            self.assertEqual(response.status_code,200)
            self.assertTrue(response.headers['content-type'].startswith('text/javascript'))

    async def test_unexpected_exception_does_not_escape_to_asgi_logs(self):
        async def broken(): raise RuntimeError('SECRET_SENTINEL remote exception')
        self.app.add_api_route('/api/fault', broken)
        response = await self.client.get('/api/fault')  # default raise_app_exceptions=True
        self.assertEqual(response.status_code,503)
        self.assertNotIn('SECRET_SENTINEL', response.text)

    async def test_stale_mode_rejected_before_create_or_execution(self):
        response = await self.client.post('/api/conversations',headers={'x-workbench-mode':'live'})
        self.assertEqual(response.status_code,409)
        self.assertEqual(response.json()['error'],'service_mode_changed')
        self.assertEqual(len((await self.client.get('/api/conversations')).json()),1)
        url = f'/api/conversations/{self.conv}/turns'
        response = await self.client.post(url,headers={'x-workbench-mode':'live'},json={'message':'问题','turn_id':'stale'})
        self.assertEqual(response.status_code,409)
        self.assertEqual((await self.client.get(f'/api/conversations/{self.conv}')).json()['turns'],[])

    async def test_sse_midstream_errors_are_sanitized(self):
        view, _ = self.app.state.store.reserve(self.conv, 't1', '问题')
        async def broken_events(*args):
            yield {'type':'stage','turn':view}
            raise RuntimeError('SECRET_SENTINEL stream exception')
        self.app.state.jobs.events = broken_events
        response = await self.client.get(f'/api/conversations/{self.conv}/turns/t1/events')
        self.assertEqual(response.status_code,200)
        self.assertIn('service_unavailable', response.text)
        self.assertNotIn('SECRET_SENTINEL', response.text)
