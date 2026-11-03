"""Managed execution is independent of HTTP and subscriptions never execute work."""
from __future__ import annotations
import asyncio

from ..runtime.chat import build_chat_graph
from ..runtime.contracts import Request
from .contracts import TERMINAL, TurnPatch
from .store import ResourceNotFound, TurnConflict


class ConversationBusy(RuntimeError):
    pass


class CapacityBusy(RuntimeError):
    pass


class JobController:
    def __init__(self, bundle, store, *, max_active: int = 2, shutdown_timeout_s: float = 5):
        self.bundle, self.store, self.max_active = bundle, store, max_active
        self.graph = build_chat_graph(bundle.service)
        self._lock = asyncio.Lock()
        self._tasks = {}
        self._subscribers = {}
        self._closed = False
        self.shutdown_timeout_s = shutdown_timeout_s
        self.store.interrupt_pending()

    async def submit(self, conv_id: str, turn_id: str, message: str):
        async with self._lock:
            self.store.get(conv_id)
            try:
                existing = self.store.get_turn(conv_id, turn_id)
            except ResourceNotFound:
                existing = None
            if existing:
                if existing.message != message:
                    raise TurnConflict('同一轮次 ID 对应不同消息')
                return existing
            if self._closed:
                raise CapacityBusy('服务正在关闭')
            if any(c == conv_id for c, _ in self._tasks):
                raise ConversationBusy('该会话正在运行')
            if len(self._tasks) >= self.max_active:
                raise CapacityBusy('当前运行任务已满')
            view, _ = self.store.reserve(conv_id, turn_id, message)
            task = asyncio.create_task(self._run(conv_id, turn_id, message))
            self._tasks[(conv_id, turn_id)] = task
            return view

    def _wake(self, key):
        for signal in self._subscribers.get(key, ()):
            signal.set()

    def get(self, conv_id: str, turn_id: str):
        view = self.store.get_turn(conv_id, turn_id)
        if view.state not in TERMINAL:
            statuses = self.store.statuses(conv_id, turn_id)
            stage = [s for s in view.status if 'agent_type' not in s]
            if statuses:
                view = self.store.update_public(conv_id, turn_id, TurnPatch(status=stage + statuses))
        return view

    async def _run(self, conv_id, turn_id, message):
        key = (conv_id, turn_id)
        req = Request(user_id=self.store.user_id, conv_id=conv_id, turn_id=turn_id, message=message)
        patch = TurnPatch(state='running', status=[{'phase': 'receive'}])
        try:
            self.store.update_public(conv_id, turn_id, patch)
            self._wake(key)
            async for updates in self.graph.astream({'request': req}, stream_mode='updates'):
                for node, output in updates.items():
                    if not isinstance(output, dict):
                        continue
                    fields = {}
                    if 'routing_decision' in output:
                        fields['routing'] = output['routing_decision']
                    if 'agent_results' in output:
                        fields['agent_results'] = output['agent_results']
                    if 'final_result' in output:
                        previous = self.store.get_turn(conv_id, turn_id).agent_results
                        fields['agent_results'] = previous + [output['final_result']]
                    phase = {'receive': 'routing', 'recognize_and_prepare': 'executing',
                             'run_agents': 'composing', 'compose': 'saving', 'commit': 'completed'}.get(node)
                    if node == 'commit':
                        phase = self.store.get_turn(conv_id, turn_id).state
                    if phase:
                        fields['status'] = [{'phase': phase}] + self.store.statuses(conv_id, turn_id)
                    if fields:
                        self.store.update_public(conv_id, turn_id, TurnPatch(**fields))
                    self._wake(key)
            view = self.store.get_turn(conv_id, turn_id)
            if view.answer is None or view.state not in ('completed', 'incomplete'):
                raise RuntimeError('graph did not persist final answer')
            self.store.update_public(conv_id, turn_id, TurnPatch(state=view.state,
                status=[{'phase': view.state}] + self.store.statuses(conv_id, turn_id)))
        except BaseException as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            state = 'interrupted' if cancelled else 'failed'
            # Never log remote exception text or serialize it to a client.
            try:
                self.bundle.service.memory.abort_turn(req, state)
                self.store.update_public(conv_id, turn_id, TurnPatch(state=state,
                    error_code='service_interrupted' if cancelled else 'execution_failed', status=[{'phase': state}]))
            except Exception:
                # A database fault must not be converted into a completed response.
                pass
            if not isinstance(exc, Exception) and not cancelled:
                raise
        finally:
            self._tasks.pop(key, None)
            self._wake(key)

    async def events(self, conv_id: str, turn_id: str):
        key = (conv_id, turn_id)
        self.store.get_turn(conv_id, turn_id)
        signal = asyncio.Event()
        self._subscribers.setdefault(key, set()).add(signal)
        try:
            while True:
                signal.clear()
                view = self.get(conv_id, turn_id)
                event_type = ('error' if view.state in ('failed','cancelled','interrupted') else
                              'result' if view.state in ('completed','incomplete') else 'stage')
                yield {'type': event_type, 'turn': view}
                if view.state in TERMINAL:
                    return
                try:
                    await asyncio.wait_for(signal.wait(), 1)
                except TimeoutError:
                    pass  # bounded snapshot heartbeat also samples actual persisted status
        finally:
            subscribers = self._subscribers.get(key)
            if subscribers is not None:
                subscribers.discard(signal)
                if not subscribers:
                    self._subscribers.pop(key, None)

    async def aclose(self):
        self._closed = True
        owned = list(self._tasks.items())
        for _, task in owned:
            task.cancel()
        if owned:
            _, pending = await asyncio.wait([task for _, task in owned], timeout=self.shutdown_timeout_s)
            if pending:
                # Keep resources open rather than closing SQLite under an uncooperative tool.
                raise RuntimeError('任务未在关闭期限内停止；未关闭仍在使用的存储。')
        for (conv_id, turn_id), task in owned:
            # A task cancelled before its first instruction never runs _run's finally.
            view = self.store.get_turn(conv_id, turn_id)
            if view.state not in TERMINAL:
                req = Request(user_id=self.store.user_id, conv_id=conv_id, turn_id=turn_id, message=view.message)
                self.bundle.service.memory.abort_turn(req, 'interrupted')
                self.store.update_public(conv_id, turn_id, TurnPatch(state='interrupted',
                    error_code='service_interrupted', status=[{'phase':'interrupted'}]))
            self._tasks.pop((conv_id, turn_id), None)
            self._wake((conv_id, turn_id))
