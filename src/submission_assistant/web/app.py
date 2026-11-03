from __future__ import annotations
from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from ..runtime.factory import create_runtime
from .contracts import SendTurn
from .store import ConversationStore, ResourceNotFound, TurnConflict
from .jobs import JobController, ConversationBusy, CapacityBusy


class WebAssets(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        content_type = {'.js':'text/javascript; charset=utf-8', '.css':'text/css; charset=utf-8'}.get(Path(path).suffix)
        if content_type and response.status_code == 200:
            response.headers['content-type'] = content_type
        return response


def create_app(project: Path, *, mode, port: int = 8000, runtime_factory=create_runtime) -> FastAPI:
    project = Path(project)

    @asynccontextmanager
    async def lifespan(app):
        bundle = await runtime_factory(project, mode=mode)
        app.state.bundle = bundle
        jobs = None
        try:
            app.state.store = ConversationStore(bundle.service.memory, 'local-user')
            jobs = app.state.jobs = JobController(bundle, app.state.store)
            yield
        finally:
            if jobs is not None:
                await jobs.aclose()
            await bundle.aclose()

    app = FastAPI(title='投稿咨询助手', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1'])
    allowed_origins = {f'http://127.0.0.1:{port}', 'http://127.0.0.1:5173'}

    @app.middleware('http')
    async def local_boundary(request: Request, call_next):
        expected_mode = request.headers.get('x-workbench-mode')
        if request.url.path.startswith('/api/') and request.url.path != '/api/health':
            if expected_mode is not None and expected_mode != app.state.bundle.mode:
                return JSONResponse({'error': 'service_mode_changed'}, status_code=409)
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            origin = request.headers.get('origin')
            if origin is not None and origin not in allowed_origins:
                return JSONResponse({'error': 'origin_denied'}, status_code=403)
            if origin is not None and expected_mode is None:
                return JSONResponse({'error': 'service_mode_changed'}, status_code=409)
            length = request.headers.get('content-length', '')
            if length.isdigit() and int(length) > 128 * 1024:
                return JSONResponse({'error': 'request_too_large'}, status_code=413)
        try:
            response = await call_next(request)
        except Exception:
            # Catch here, before Starlette's outer ServerErrorMiddleware can rethrow
            # the raw exception to Uvicorn's default traceback logger.
            response = JSONResponse({'error': 'service_unavailable'}, status_code=503)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        # Pydantic default responses echo input, which may include injected credentials.
        return JSONResponse({'error': 'invalid_request'}, status_code=422)

    async def known_error(request, exc):
        if isinstance(exc, ResourceNotFound):
            code, status = 'not_found', 404
        elif isinstance(exc, CapacityBusy):
            code, status = 'service_busy', 429
        else:
            code, status = 'turn_conflict', 409
        return JSONResponse({'error': code}, status_code=status)

    for exception in (ResourceNotFound, TurnConflict, ConversationBusy, CapacityBusy):
        app.add_exception_handler(exception, known_error)

    @app.exception_handler(Exception)
    async def unavailable(request, exc):
        return JSONResponse({'error': 'service_unavailable'}, status_code=503)

    @app.get('/api/health')
    async def health():
        bundle = app.state.bundle
        return {'status': 'ok', 'mode': bundle.mode, 'prompt_version': bundle.prompt_version,
                'dependencies': bundle.dependencies, 'max_active_conversations': 2,
                'streaming': 'stages_and_final_answer'}

    @app.post('/api/conversations', status_code=201)
    async def create_conversation():
        return app.state.store.create()

    @app.get('/api/conversations')
    async def conversations():
        return app.state.store.list()

    @app.get('/api/conversations/{conv_id}')
    async def conversation(conv_id: str):
        return app.state.store.get(conv_id)

    @app.post('/api/conversations/{conv_id}/turns', status_code=202)
    async def submit(conv_id: str, body: SendTurn):
        return await app.state.jobs.submit(conv_id, body.turn_id, body.message)

    @app.get('/api/conversations/{conv_id}/turns/{turn_id}')
    async def turn(conv_id: str, turn_id: str):
        return app.state.jobs.get(conv_id, turn_id)

    @app.get('/api/conversations/{conv_id}/turns/{turn_id}/events')
    async def events(conv_id: str, turn_id: str, mode: str | None = None):
        if mode is not None and mode != app.state.bundle.mode:
            return JSONResponse({'error':'service_mode_changed'}, status_code=409)
        app.state.jobs.get(conv_id, turn_id)  # validate before streaming headers
        async def stream():
            try:
                async for event in app.state.jobs.events(conv_id, turn_id):
                    body = {'turn': event['turn'].model_dump()}
                    yield f"event: {event['type']}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"
            except Exception:
                yield 'event: error\ndata: {"error":"service_unavailable"}\n\n'
        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})

    dist = project / 'frontend' / 'dist'
    if (dist / 'assets').is_dir():
        app.mount('/assets', WebAssets(directory=dist / 'assets'), name='assets')

    @app.get('/')
    async def index():
        if (dist / 'index.html').is_file():
            return FileResponse(dist / 'index.html', media_type='text/html')
        return JSONResponse({'message': '前端尚未构建，请先安装前端依赖并运行 build。'}, status_code=503)

    return app
