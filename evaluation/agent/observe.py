"""Per-trial observation and credential-scoped paid-call stop gate."""
import hashlib
import json
import threading
import time
from contextvars import ContextVar
from uuid import uuid4

CURRENT_BRANCH = ContextVar('evaluation_current_role_branch', default=None)


def component_for(path):
    return ('decisions' if path.endswith('/decisions') else 'embedding' if 'embedding' in path
            else 'reranker' if 'rerank' in path else 'generation')


def billing_error(body):
    if not isinstance(body, dict):
        return False
    error = body.get('error', body)
    if not isinstance(error, dict):
        return False
    codes = {str(error.get(k, '')).lower() for k in ('code','type')}
    return bool(codes & {'insufficient_quota','insufficient_balance','credits_exhausted','credit_balance_exhausted',
                         'billing_hard_limit_reached','balance_not_enough','allocationquota.freetieronly'})


class PaidCallGate:
    def __init__(self):
        self._blocked = set()
        self._lock = threading.Lock()
        self.stopped = threading.Event()

    def check(self, authorization):
        if self.stopped.is_set():
            raise RuntimeError('evaluation_interrupted')
        key = hashlib.sha256(authorization.encode()).hexdigest()
        with self._lock:
            if key in self._blocked:
                raise RuntimeError('credential_paused_after_auth_or_credit_error')

    def block(self, authorization):
        with self._lock:
            self._blocked.add(hashlib.sha256(authorization.encode()).hexdigest())

    @property
    def blocked(self):
        with self._lock:
            return bool(self._blocked)


def usage_values(usage):
    usage = usage if isinstance(usage, dict) else {}
    def known(*keys):
        for k in keys:
            n = usage.get(k)
            if type(n) is int and n >= 0:
                return n
        return None
    return {'input_tokens': known('input_tokens', 'prompt_tokens'),
            'output_tokens': known('output_tokens', 'completion_tokens'),
            'total_tokens': known('total_tokens')}


class ObservationSink:
    def __init__(self, trial_id):
        self.trial_id = trial_id
        self.turn_id = None
        self.events = []
        self.started = time.perf_counter()

    def add(self, kind, **data):
        event = {'trial_id': self.trial_id, 'turn_id': self.turn_id, 'timestamp': time.time(),
                 'offset_ms': round((time.perf_counter() - self.started) * 1000, 3), 'type': kind, **data}
        self.events.append(event)
        return event

    def totals(self, kind='model_result'):
        events = [e for e in self.events if e['type'] == kind]
        fields = ('input_tokens', 'output_tokens')
        totals = {k: sum(e['usage'][k] for e in events if e.get('usage', {}).get(k) is not None)
                  if any(e.get('usage', {}).get(k) is not None for e in events) else None for k in fields}
        totals['known_calls'] = sum(all(e.get('usage', {}).get(k) is not None for k in fields) for e in events)
        totals['calls'] = len(events)
        attempt_kind = 'model_call' if kind=='model_result' else 'http_call'
        totals['attempted_calls'] = sum(e['type']==attempt_kind for e in self.events)
        totals['complete'] = bool(events) and totals['known_calls'] == totals['attempted_calls']
        return totals

    def http_hooks(self, gate):
        async def request(req):
            gate.check(req.headers.get('authorization', ''))
            req.extensions['eval_started'] = time.perf_counter()
            req.extensions['eval_request_id'] = uuid4().hex
            self.add('http_call', component=component_for(req.url.path), call_id=req.extensions['eval_request_id'])
        async def response(resp):
            req = resp.request
            if resp.status_code in (401, 402, 403):
                gate.block(req.headers.get('authorization', ''))
            await resp.aread()
            try:
                body = resp.json()
            except (ValueError, UnicodeError):
                body = {}
            if billing_error(body):
                gate.block(req.headers.get('authorization', ''))
                resp.extensions['evaluation_billing_error'] = True
            component = component_for(req.url.path)
            self.add('http_result', component=component, status_code=resp.status_code,
                     call_id=req.extensions.get('eval_request_id'),
                     latency_ms=round((time.perf_counter()-req.extensions.get('eval_started', time.perf_counter()))*1000, 2),
                     usage=usage_values(body.get('usage') if isinstance(body, dict) else None))
        return {'request': [request], 'response': [response]}


class ObservedModelRouter:
    def __init__(self, router, sink):
        self.router, self.sink = router, sink

    def for_role(self, role):
        target = self.router.for_role(role) if hasattr(self.router, 'for_role') else self.router
        sink = self.sink
        class Model:
            model = getattr(target, 'model', 'unknown')
            max_output_tokens = getattr(target, 'max_output_tokens', 0)
            async def complete(self, messages, tools, *, allow_tools=True):
                identifier = uuid4().hex
                started = time.perf_counter()
                owner = CURRENT_BRANCH.get() or {}
                context = {'role': owner.get('role',role), 'component':role, 'run_id':owner.get('run_id')}
                sink.add('model_call', **context, call_id=identifier, model=self.model)
                try:
                    result = await target.complete(messages, tools, allow_tools=allow_tools)
                    sink.add('model_result', **context, call_id=identifier, model=self.model,
                             usage=usage_values(result.get('usage')), finish_reason=result.get('finish_reason'),
                             latency_ms=round((time.perf_counter()-started)*1000, 2))
                    return result
                except BaseException as exc:
                    sink.add('model_error', **context, call_id=identifier, error=type(exc).__name__,
                             latency_ms=round((time.perf_counter()-started)*1000, 2))
                    raise
        return Model()
