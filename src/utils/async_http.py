"""Async HTTP utilities with connection pooling, retry, and rate limiting."""

import asyncio
import json
from typing import Optional, Any

import aiohttp

from utils.logging_config import get_logger

logger = get_logger("async_http")

# Default timeout configuration
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=20, connect=5, sock_read=12)

# Retry configuration
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF_FACTOR = 0.3
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class AsyncRateLimiter:
    """Cooperative rate limiter for asyncio (single-threaded, no lock needed)."""

    def __init__(self, calls_per_second: float = 10.0):
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    async def acquire(self):
        loop = asyncio.get_running_loop()
        now = loop.time()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_call = asyncio.get_running_loop().time()


class AsyncHTTPClient:
    """Shared async HTTP client with connection pooling, retry, and rate limit handling."""

    def __init__(
        self,
        concurrency_limit: int = 30,
        per_host_limit: int = 10,
        timeout: aiohttp.ClientTimeout = None,
        headers: dict = None,
    ):
        self._concurrency_limit = concurrency_limit
        self._per_host_limit = per_host_limit
        self._timeout = timeout or DEFAULT_TIMEOUT
        self._headers = headers or {"User-Agent": "AI-Dataset-Radar/5.0"}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self._concurrency_limit,
                limit_per_host=self._per_host_limit,
            )
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                connector=connector,
                headers=self._headers,
            )
        return self._session

    async def get(
        self,
        url: str,
        headers: dict = None,
        params: dict = None,
        max_retries: int = DEFAULT_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ) -> Optional[bytes]:
        """Async GET with retry and rate limit handling. Returns response body bytes."""
        session = await self._get_session()
        for attempt in range(max_retries):
            try:
                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    elif resp.status == 429:
                        retry_after = resp.headers.get("Retry-After", "60")
                        try:
                            wait = min(int(retry_after), 120)
                        except ValueError:
                            wait = 60
                        logger.warning("Rate limited on %s, waiting %ds", url, wait)
                        await asyncio.sleep(wait)
                    elif resp.status >= 500:
                        wait = backoff_factor * (2 ** attempt)
                        logger.warning(
                            "Server error %d for %s, retry in %.1fs",
                            resp.status, url, wait,
                        )
                        await asyncio.sleep(wait)
                    elif resp.status == 403:
                        logger.warning("Forbidden (403) for %s", url)
                        return None
                    elif resp.status == 404:
                        return None
                    else:
                        logger.warning("HTTP %d for %s", resp.status, url)
                        return None
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait = backoff_factor * (2 ** attempt)
                    logger.warning("Timeout for %s, retry in %.1fs", url, wait)
                    await asyncio.sleep(wait)
                else:
                    logger.error("All retries timed out for %s", url)
                    return None
            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    wait = backoff_factor * (2 ** attempt)
                    logger.warning("Request error for %s: %s, retry in %.1fs", url, e, wait)
                    await asyncio.sleep(wait)
                else:
                    logger.error("All retries failed for %s: %s", url, e)
                    return None
        return None

    async def get_json(
        self,
        url: str,
        headers: dict = None,
        params: dict = None,
        max_retries: int = DEFAULT_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ) -> Optional[Any]:
        """Async GET and parse JSON response."""
        data = await self.get(url, headers=headers, params=params,
                              max_retries=max_retries, backoff_factor=backoff_factor)
        if data is None:
            return None
        try:
            return json.loads(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("JSON parse error for %s: %s", url, e)
            return None

    async def get_text(
        self,
        url: str,
        headers: dict = None,
        params: dict = None,
        max_retries: int = DEFAULT_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        encoding: str = "utf-8",
    ) -> Optional[str]:
        """Async GET and return response as text."""
        data = await self.get(url, headers=headers, params=params,
                              max_retries=max_retries, backoff_factor=backoff_factor)
        if data is None:
            return None
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")

    async def head(
        self,
        url: str,
        headers: dict = None,
        allow_redirects: bool = True,
        timeout: float = 5,
    ) -> Optional[int]:
        """Async HEAD request. Returns status code or None on failure."""
        session = await self._get_session()
        try:
            req_timeout = aiohttp.ClientTimeout(total=timeout)
            async with session.head(
                url, headers=headers, allow_redirects=allow_redirects, timeout=req_timeout
            ) as resp:
                return resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

    async def close(self):
        """Close the underlying session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
