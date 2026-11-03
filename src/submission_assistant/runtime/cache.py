"""Optional Redis cache. Failure/loss cannot change the authoritative result."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import random
import secrets
import time


class VersionedCache:
    _RELEASE_LOCK = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"

    def __init__(self,redis_client=None,*,ttl_s=300,ttl_jitter_ratio=.1,max_entries=256,
                 lock_wait_s=90,lock_lease_s=120,lock_poll_s=.05):
        if ttl_s < 1 or not 0 <= ttl_jitter_ratio <= .5 or max_entries < 1:
            raise ValueError('invalid cache settings')
        if lock_wait_s < 0 or lock_lease_s < 1 or lock_poll_s <= 0:
            raise ValueError('invalid cache lock settings')
        self.redis=redis_client; self.ttl_s=ttl_s; self.ttl_jitter_ratio=ttl_jitter_ratio
        self.max_entries=max_entries; self.lock_wait_s=lock_wait_s
        self.lock_lease_s=lock_lease_s; self.lock_poll_s=lock_poll_s
        self.local={}
        self._locks={}

    def key(self,namespace,parts):
        body=json.dumps(parts,ensure_ascii=False,sort_keys=True,allow_nan=False)
        return 'submission:'+namespace+':'+hashlib.sha256(body.encode()).hexdigest()

    def lock(self,key):
        # Local single-flight remains useful if Redis is absent or unavailable.
        return self._combined_lock(key)

    @asynccontextmanager
    async def _combined_lock(self,key):
        local=self._locks.setdefault(key,asyncio.Lock())
        async with local:
            if self.redis is None:
                yield
                return
            lock_key=f'{key}:lock'; token=secrets.token_urlsafe(24)
            deadline=time.monotonic()+self.lock_wait_s; acquired=False
            while True:
                try:
                    acquired=bool(await asyncio.wait_for(
                        self.redis.set(lock_key,token,nx=True,px=int(self.lock_lease_s*1000)),.5))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Redis is an optimization: lock failure must not block authoritative retrieval.
                    break
                if acquired or time.monotonic()>=deadline:
                    break
                await asyncio.sleep(min(self.lock_poll_s,max(0,deadline-time.monotonic())))
            try:
                yield
            finally:
                if acquired:
                    try:
                        await asyncio.wait_for(
                            self.redis.eval(self._RELEASE_LOCK,1,lock_key,token),.5)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # The lease expires automatically if Redis cannot release the lock.
                        pass

    async def get(self,key):
        if self.redis is not None:
            try:
                raw=await asyncio.wait_for(self.redis.get(key),.5)
                if raw is not None: return json.loads(raw)
            except asyncio.CancelledError: raise
            except Exception: pass
        entry=self.local.get(key)
        if entry and entry[0]>time.monotonic(): return json.loads(entry[1])
        return None

    async def set(self,key,value):
        raw=json.dumps(value,ensure_ascii=False,allow_nan=False)
        if len(raw.encode())>2*1024*1024: return
        ttl=max(1,round(self.ttl_s*random.uniform(1-self.ttl_jitter_ratio,1+self.ttl_jitter_ratio)))
        if len(self.local)>=self.max_entries:
            self.local.pop(next(iter(self.local)))
        self.local[key]=(time.monotonic()+ttl,raw)
        if self.redis is not None:
            try: await asyncio.wait_for(self.redis.set(key,raw,ex=ttl),.5)
            except asyncio.CancelledError: raise
            except Exception: pass


def optional_redis(url):
    if not url: return None
    try:
        from redis.asyncio import Redis
        return Redis.from_url(url,socket_connect_timeout=.5,socket_timeout=.5)
    except ImportError:
        return None
