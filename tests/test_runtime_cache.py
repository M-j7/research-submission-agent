import asyncio
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, *, ex=None, nx=False, px=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        if ex is not None:
            self.expirations[key] = ex
        if px is not None:
            self.expirations[key] = px
        return True

    async def eval(self, script, numkeys, key, token):
        if self.values.get(key) == token:
            del self.values[key]
            self.expirations.pop(key, None)
            return 1
        return 0


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.cache_module = importlib.import_module('submission_assistant.runtime.cache')

    def test_cache_ttl_adds_jitter_to_local_and_redis_expiry(self):
        async def exercise():
            redis = FakeRedis()
            cache = self.cache_module.VersionedCache(redis, ttl_s=100, ttl_jitter_ratio=.1)
            with patch.object(self.cache_module.random, 'uniform', return_value=1.08):
                await cache.set('k', {'answer': 'cached'})
            self.assertEqual(redis.expirations['k'], 108)
            self.assertEqual(await cache.get('k'), {'answer': 'cached'})
        asyncio.run(exercise())

    def test_redis_lock_serializes_same_key_across_cache_instances(self):
        async def exercise():
            redis = FakeRedis()
            first = self.cache_module.VersionedCache(redis, lock_wait_s=.5, lock_lease_s=1)
            second = self.cache_module.VersionedCache(redis, lock_wait_s=.5, lock_lease_s=1)
            first_entered, release_first, second_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()

            async def first_worker():
                async with first.lock('same-key'):
                    first_entered.set()
                    await release_first.wait()

            async def second_worker():
                async with second.lock('same-key'):
                    second_entered.set()

            task1 = asyncio.create_task(first_worker())
            await first_entered.wait()
            task2 = asyncio.create_task(second_worker())
            await asyncio.sleep(.05)
            self.assertFalse(second_entered.is_set())
            release_first.set()
            await asyncio.gather(task1, task2)
            self.assertTrue(second_entered.is_set())
            self.assertNotIn('same-key:lock', redis.values)
        asyncio.run(exercise())

    def test_lock_release_never_deletes_another_owners_lock(self):
        async def exercise():
            redis = FakeRedis()
            cache = self.cache_module.VersionedCache(redis, lock_wait_s=.1, lock_lease_s=1)
            async with cache.lock('same-key'):
                redis.values['same-key:lock'] = 'replacement-owner'
            self.assertEqual(redis.values.get('same-key:lock'), 'replacement-owner')
        asyncio.run(exercise())

    def test_redis_outage_falls_back_to_local_cache_and_lock(self):
        class BrokenRedis:
            async def get(self, key):
                raise OSError('redis offline')

            async def set(self, key, value, **kwargs):
                raise OSError('redis offline')

        async def exercise():
            cache = self.cache_module.VersionedCache(BrokenRedis())
            async with cache.lock('k'):
                await cache.set('k', {'source': 'recomputed'})
                self.assertEqual(await cache.get('k'), {'source': 'recomputed'})
        asyncio.run(exercise())


if __name__ == '__main__':
    unittest.main()
