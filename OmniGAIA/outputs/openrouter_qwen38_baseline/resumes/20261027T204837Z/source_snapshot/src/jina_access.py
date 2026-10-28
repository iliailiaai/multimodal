"""Явный выбор режима Reader и общий лимит запросов без ключа."""
import asyncio
import os
import time
from email.utils import parsedate_to_datetime


def anonymous():
    return os.environ.get("JINA_READER_MODE", "authenticated") == "anonymous"


def headers(key=None):
    result = {"X-Return-Format": "markdown"}
    if not anonymous():
        result["Authorization"] = "Bearer " + (key or os.environ["JINA_API_KEY"])
    return result


def required_keys():
    return ("SERPER_API_KEY",) if anonymous() else ("SERPER_API_KEY", "JINA_API_KEY")


def retry_delay(value):
    try:
        return max(60.0, float(value))
    except (TypeError, ValueError):
        try:
            return max(60.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return 60.0


class AnonymousLimiter:
    """Равномерные запросы без начального всплеска; общий cooldown при 429."""
    def __init__(self):
        self.lock = asyncio.Lock()
        self.next_at = 0.0

    async def acquire(self):
        async with self.lock:
            while self.next_at > time.monotonic():
                await asyncio.sleep(self.next_at - time.monotonic())
            self.next_at = time.monotonic() + 3.1

    def defer(self, seconds):
        self.next_at = max(self.next_at, time.monotonic() + seconds)
