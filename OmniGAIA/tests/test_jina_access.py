"""Проверки авторизации и ограничения параллельных обращений Reader."""
import asyncio
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jina_access import AnonymousLimiter, headers, required_keys, retry_delay

class ReaderTests(unittest.IsolatedAsyncioTestCase):
    def test_anonymous_never_sends_existing_key(self):
        with patch.dict(os.environ, {'JINA_READER_MODE':'anonymous', 'JINA_API_KEY':'unused'}):
            self.assertNotIn('Authorization', headers('unused'))
            self.assertEqual(required_keys(), ('SERPER_API_KEY',))

    def test_authenticated_preserves_key(self):
        with patch.dict(os.environ, {'JINA_READER_MODE':'authenticated'}):
            self.assertEqual(headers('test')['Authorization'], 'Bearer test')

    async def test_concurrent_requests_are_spaced_and_cooldown_shared(self):
        clock=[0.0]
        async def sleep(seconds):
            clock[0]+=seconds
        limiter=AnonymousLimiter()
        with patch('jina_access.time.monotonic', side_effect=lambda:clock[0]), patch('jina_access.asyncio.sleep', side_effect=sleep):
            await asyncio.gather(*(limiter.acquire() for _ in range(21)))
            self.assertGreaterEqual(clock[0], 60)
            limiter.defer(120)
            before=clock[0]
            await limiter.acquire()
            self.assertGreaterEqual(clock[0]-before,120)
        self.assertEqual(retry_delay('120'),120)
        self.assertEqual(retry_delay('invalid'),60)
