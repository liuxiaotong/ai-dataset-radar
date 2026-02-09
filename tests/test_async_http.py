"""Tests for AsyncHTTPClient and AsyncRateLimiter (src/utils/async_http.py).

Covers rate limiting, all HTTP methods (get, get_json, get_text, head),
retry logic on 5xx and 429, error handling, session lifecycle,
and async context manager support.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

# Add project src path for imports (matches convention used by other test files)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

aioresponses_mod = pytest.importorskip("aioresponses", reason="aioresponses package not installed")
from aioresponses import aioresponses

from utils.async_http import (
    AsyncHTTPClient,
    AsyncRateLimiter,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    RETRY_STATUS_CODES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_URL = "http://example.com/api/data"
TEST_JSON = {"status": "ok", "items": [1, 2, 3]}
TEST_BODY = b"hello world"
TEST_TEXT = "hello world"


# ---------------------------------------------------------------------------
# AsyncRateLimiter tests
# ---------------------------------------------------------------------------


class TestAsyncRateLimiter:
    """Tests for the AsyncRateLimiter class."""

    def test_init_default(self):
        """Default rate limiter allows 10 calls per second."""
        limiter = AsyncRateLimiter()
        assert limiter._min_interval == pytest.approx(0.1)
        assert limiter._last_call == 0.0

    def test_init_custom(self):
        """Custom calls_per_second is correctly converted to min interval."""
        limiter = AsyncRateLimiter(calls_per_second=5.0)
        assert limiter._min_interval == pytest.approx(0.2)

    def test_init_one_per_second(self):
        """1 call per second means 1.0s minimum interval."""
        limiter = AsyncRateLimiter(calls_per_second=1.0)
        assert limiter._min_interval == pytest.approx(1.0)

    async def test_acquire_first_call_no_delay(self):
        """First acquire should not sleep (last_call starts at 0)."""
        limiter = AsyncRateLimiter(calls_per_second=10.0)
        # First call: elapsed from 0 is huge, no sleep needed
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await limiter.acquire()
            # First call should not trigger sleep since elapsed >> min_interval
            mock_sleep.assert_not_called()

    async def test_acquire_enforces_interval(self):
        """Rapid back-to-back acquires should sleep to enforce the minimum interval."""
        limiter = AsyncRateLimiter(calls_per_second=2.0)  # 0.5s interval
        # Do the first acquire to set _last_call
        await limiter.acquire()

        # Now measure the second acquire -- it should sleep
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Simulate that almost no time has passed by setting _last_call to now
            loop = asyncio.get_event_loop()
            limiter._last_call = loop.time()
            await limiter.acquire()
            mock_sleep.assert_called_once()
            # The sleep time should be close to min_interval (0.5s)
            sleep_time = mock_sleep.call_args[0][0]
            assert 0.0 < sleep_time <= 0.5

    async def test_multiple_rapid_acquires_throttled(self):
        """Multiple rapid acquires should each be throttled."""
        limiter = AsyncRateLimiter(calls_per_second=100.0)  # 0.01s interval
        sleep_count = 0
        original_sleep = asyncio.sleep

        async def counting_sleep(duration):
            nonlocal sleep_count
            sleep_count += 1
            # Actually sleep a tiny bit to advance loop time
            await original_sleep(0)

        with patch("asyncio.sleep", side_effect=counting_sleep):
            # First acquire sets the baseline
            await limiter.acquire()
            # Force _last_call to current time to ensure subsequent calls need throttling
            loop = asyncio.get_event_loop()
            limiter._last_call = loop.time()
            # Rapid-fire acquires
            for _ in range(5):
                limiter._last_call = loop.time()  # Reset to "just called"
                await limiter.acquire()

        # Each of the 5 subsequent calls should have triggered a sleep
        assert sleep_count == 5


# ---------------------------------------------------------------------------
# AsyncHTTPClient tests
# ---------------------------------------------------------------------------


class TestAsyncHTTPClientInit:
    """Tests for AsyncHTTPClient initialization."""

    def test_default_init(self):
        """Default parameters are applied correctly."""
        client = AsyncHTTPClient()
        assert client._concurrency_limit == 30
        assert client._per_host_limit == 10
        assert client._timeout is DEFAULT_TIMEOUT
        assert client._headers == {"User-Agent": "AI-Dataset-Radar/6.0"}
        assert client._session is None

    def test_custom_init(self):
        """Custom parameters override defaults."""
        custom_timeout = aiohttp.ClientTimeout(total=60)
        custom_headers = {"Authorization": "Bearer token123"}
        client = AsyncHTTPClient(
            concurrency_limit=50,
            per_host_limit=20,
            timeout=custom_timeout,
            headers=custom_headers,
        )
        assert client._concurrency_limit == 50
        assert client._per_host_limit == 20
        assert client._timeout is custom_timeout
        assert client._headers == custom_headers


class TestAsyncHTTPClientGet:
    """Tests for the get() method."""

    async def test_get_success(self):
        """GET returns bytes on 200 OK."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, body=TEST_BODY, status=200)
                result = await client.get(TEST_URL)
                assert result == TEST_BODY

    async def test_get_with_params(self):
        """GET passes query parameters correctly."""
        url_with_params = TEST_URL + "?key=value"
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(url_with_params, body=TEST_BODY, status=200)
                result = await client.get(TEST_URL, params={"key": "value"})
                assert result == TEST_BODY

    async def test_get_with_custom_headers(self):
        """GET passes custom per-request headers."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, body=TEST_BODY, status=200)
                result = await client.get(TEST_URL, headers={"X-Custom": "test"})
                assert result == TEST_BODY

    async def test_get_404_returns_none(self):
        """GET returns None on 404."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, status=404)
                result = await client.get(TEST_URL)
                assert result is None

    async def test_get_403_returns_none(self):
        """GET returns None on 403 Forbidden."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, status=403)
                result = await client.get(TEST_URL)
                assert result is None

    async def test_get_other_4xx_returns_none(self):
        """GET returns None on other 4xx status codes (e.g., 400, 401)."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, status=400)
                result = await client.get(TEST_URL)
                assert result is None

    async def test_get_5xx_retries_then_none(self):
        """GET retries on 5xx and returns None after all retries exhausted."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                # Register 3 failed responses (default retries = 3)
                for _ in range(DEFAULT_RETRIES):
                    m.get(TEST_URL, status=500)
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await client.get(TEST_URL, max_retries=3, backoff_factor=0.0)
                assert result is None

    async def test_get_5xx_retry_then_success(self):
        """GET retries on 5xx and succeeds on a subsequent attempt."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, status=503)
                m.get(TEST_URL, body=TEST_BODY, status=200)
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await client.get(TEST_URL, max_retries=3, backoff_factor=0.0)
                assert result == TEST_BODY

    async def test_get_5xx_backoff_increases(self):
        """GET uses exponential backoff on 5xx retries."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, status=500)
                m.get(TEST_URL, status=500)
                m.get(TEST_URL, body=TEST_BODY, status=200)
                sleep_times = []

                async def capture_sleep(duration):
                    sleep_times.append(duration)

                with patch("asyncio.sleep", side_effect=capture_sleep):
                    result = await client.get(TEST_URL, max_retries=3, backoff_factor=1.0)
                assert result == TEST_BODY
                # backoff_factor * (2 ** attempt): 1*1=1, 1*2=2
                assert sleep_times == [pytest.approx(1.0), pytest.approx(2.0)]

    async def test_get_429_retry_after_header(self):
        """GET respects Retry-After header on 429."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, status=429, headers={"Retry-After": "5"})
                m.get(TEST_URL, body=TEST_BODY, status=200)

                sleep_times = []

                async def capture_sleep(duration):
                    sleep_times.append(duration)

                with patch("asyncio.sleep", side_effect=capture_sleep):
                    result = await client.get(TEST_URL, max_retries=3)
                assert result == TEST_BODY
                # Should have waited 5 seconds as per Retry-After header
                assert sleep_times[0] == 5

    async def test_get_429_retry_after_capped_at_120(self):
        """GET caps Retry-After at 120 seconds."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, status=429, headers={"Retry-After": "300"})
                m.get(TEST_URL, body=TEST_BODY, status=200)

                sleep_times = []

                async def capture_sleep(duration):
                    sleep_times.append(duration)

                with patch("asyncio.sleep", side_effect=capture_sleep):
                    result = await client.get(TEST_URL, max_retries=3)
                assert result == TEST_BODY
                assert sleep_times[0] == 120

    async def test_get_429_retry_after_invalid_defaults_60(self):
        """GET defaults Retry-After to 60s when header value is not parseable."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, status=429, headers={"Retry-After": "not-a-number"})
                m.get(TEST_URL, body=TEST_BODY, status=200)

                sleep_times = []

                async def capture_sleep(duration):
                    sleep_times.append(duration)

                with patch("asyncio.sleep", side_effect=capture_sleep):
                    result = await client.get(TEST_URL, max_retries=3)
                assert result == TEST_BODY
                assert sleep_times[0] == 60

    async def test_get_429_no_retry_after_defaults_60(self):
        """GET defaults to 60s wait when 429 has no Retry-After header."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, status=429)
                m.get(TEST_URL, body=TEST_BODY, status=200)

                sleep_times = []

                async def capture_sleep(duration):
                    sleep_times.append(duration)

                with patch("asyncio.sleep", side_effect=capture_sleep):
                    result = await client.get(TEST_URL, max_retries=3)
                assert result == TEST_BODY
                assert sleep_times[0] == 60

    async def test_get_network_error_retries(self):
        """GET retries on aiohttp.ClientError and returns None after exhaustion."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                for _ in range(3):
                    m.get(TEST_URL, exception=aiohttp.ClientConnectionError("Connection refused"))
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await client.get(TEST_URL, max_retries=3, backoff_factor=0.0)
                assert result is None

    async def test_get_network_error_then_success(self):
        """GET retries on network error and succeeds on next attempt."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, exception=aiohttp.ClientConnectionError("Connection refused"))
                m.get(TEST_URL, body=TEST_BODY, status=200)
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await client.get(TEST_URL, max_retries=3, backoff_factor=0.0)
                assert result == TEST_BODY

    async def test_get_timeout_retries(self):
        """GET retries on TimeoutError and returns None after exhaustion."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                for _ in range(3):
                    m.get(TEST_URL, exception=asyncio.TimeoutError())
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await client.get(TEST_URL, max_retries=3, backoff_factor=0.0)
                assert result is None

    async def test_get_timeout_then_success(self):
        """GET retries on timeout and succeeds on next attempt."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, exception=asyncio.TimeoutError())
                m.get(TEST_URL, body=TEST_BODY, status=200)
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await client.get(TEST_URL, max_retries=3, backoff_factor=0.0)
                assert result == TEST_BODY


class TestAsyncHTTPClientGetJson:
    """Tests for the get_json() method."""

    async def test_get_json_success(self):
        """get_json returns parsed dict on 200 with valid JSON body."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, payload=TEST_JSON, status=200)
                result = await client.get_json(TEST_URL)
                assert result == TEST_JSON

    async def test_get_json_invalid_json_returns_none(self):
        """get_json returns None when response body is not valid JSON."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, body=b"not json at all {{{", status=200)
                result = await client.get_json(TEST_URL)
                assert result is None

    async def test_get_json_http_error_returns_none(self):
        """get_json returns None when underlying get() fails."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, status=404)
                result = await client.get_json(TEST_URL)
                assert result is None

    async def test_get_json_nested_structure(self):
        """get_json correctly parses nested JSON structures."""
        nested = {"data": {"users": [{"id": 1, "name": "test"}]}, "meta": {"total": 1}}
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, payload=nested, status=200)
                result = await client.get_json(TEST_URL)
                assert result == nested
                assert result["data"]["users"][0]["id"] == 1


class TestAsyncHTTPClientGetText:
    """Tests for the get_text() method."""

    async def test_get_text_success(self):
        """get_text returns decoded string on 200."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, body=TEST_BODY, status=200)
                result = await client.get_text(TEST_URL)
                assert result == TEST_TEXT
                assert isinstance(result, str)

    async def test_get_text_http_error_returns_none(self):
        """get_text returns None when underlying get() fails."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, status=500)
                # Exhaust retries
                m.get(TEST_URL, status=500)
                m.get(TEST_URL, status=500)
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await client.get_text(TEST_URL, max_retries=3, backoff_factor=0.0)
                assert result is None

    async def test_get_text_utf8_encoding(self):
        """get_text correctly decodes UTF-8 content."""
        utf8_body = "Hello, world! Unicode: \u4f60\u597d".encode("utf-8")
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, body=utf8_body, status=200)
                result = await client.get_text(TEST_URL)
                assert result == "Hello, world! Unicode: \u4f60\u597d"

    async def test_get_text_custom_encoding(self):
        """get_text uses the specified encoding parameter."""
        latin1_body = "caf\u00e9".encode("latin-1")
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, body=latin1_body, status=200)
                result = await client.get_text(TEST_URL, encoding="latin-1")
                assert result == "caf\u00e9"

    async def test_get_text_bad_encoding_falls_back(self):
        """get_text falls back to utf-8 with errors=replace on decode failure."""
        # Create bytes invalid in utf-8 but valid in latin-1
        bad_bytes = b"\xff\xfe"
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, body=bad_bytes, status=200)
                result = await client.get_text(TEST_URL, encoding="utf-8")
                # Should not raise, should contain replacement characters
                assert isinstance(result, str)


class TestAsyncHTTPClientHead:
    """Tests for the head() method."""

    async def test_head_success(self):
        """HEAD returns the status code on success."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.head(TEST_URL, status=200)
                result = await client.head(TEST_URL)
                assert result == 200

    async def test_head_redirect(self):
        """HEAD returns status code for redirects."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.head(TEST_URL, status=301)
                result = await client.head(TEST_URL)
                assert result == 301

    async def test_head_404(self):
        """HEAD returns 404 status code (does not return None for non-200)."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.head(TEST_URL, status=404)
                result = await client.head(TEST_URL)
                assert result == 404

    async def test_head_network_error_returns_none(self):
        """HEAD returns None on aiohttp.ClientError."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.head(TEST_URL, exception=aiohttp.ClientConnectionError("fail"))
                result = await client.head(TEST_URL)
                assert result is None

    async def test_head_timeout_returns_none(self):
        """HEAD returns None on asyncio.TimeoutError."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.head(TEST_URL, exception=asyncio.TimeoutError())
                result = await client.head(TEST_URL)
                assert result is None


class TestAsyncHTTPClientSession:
    """Tests for session lifecycle management."""

    async def test_session_lazy_creation(self):
        """Session is None until first request."""
        client = AsyncHTTPClient()
        assert client._session is None
        try:
            with aioresponses() as m:
                m.get(TEST_URL, body=TEST_BODY, status=200)
                await client.get(TEST_URL)
                assert client._session is not None
                assert not client._session.closed
        finally:
            await client.close()

    async def test_session_reused_across_requests(self):
        """The same session is reused for multiple requests."""
        async with AsyncHTTPClient() as client:
            with aioresponses() as m:
                m.get(TEST_URL, body=TEST_BODY, status=200)
                m.get(TEST_URL, body=TEST_BODY, status=200)
                await client.get(TEST_URL)
                session1 = client._session
                await client.get(TEST_URL)
                session2 = client._session
                assert session1 is session2

    async def test_close_sets_session_none(self):
        """close() closes the session and sets it to None."""
        client = AsyncHTTPClient()
        with aioresponses() as m:
            m.get(TEST_URL, body=TEST_BODY, status=200)
            await client.get(TEST_URL)
            assert client._session is not None
            await client.close()
            assert client._session is None

    async def test_close_idempotent(self):
        """Calling close() multiple times does not raise."""
        client = AsyncHTTPClient()
        await client.close()  # No session yet, should not error
        with aioresponses() as m:
            m.get(TEST_URL, body=TEST_BODY, status=200)
            await client.get(TEST_URL)
        await client.close()
        await client.close()  # Already closed, should not error

    async def test_context_manager(self):
        """Async context manager enters and exits cleanly."""
        async with AsyncHTTPClient() as client:
            assert isinstance(client, AsyncHTTPClient)
            with aioresponses() as m:
                m.get(TEST_URL, body=TEST_BODY, status=200)
                result = await client.get(TEST_URL)
                assert result == TEST_BODY
                _ = client._session  # verify session exists before exit
        # After exiting the context manager, session should be closed
        assert client._session is None

    async def test_context_manager_returns_self(self):
        """__aenter__ returns the client instance itself."""
        client = AsyncHTTPClient()
        entered = await client.__aenter__()
        assert entered is client
        await client.__aexit__(None, None, None)


class TestAsyncHTTPClientModuleConstants:
    """Tests for module-level constants."""

    def test_default_timeout(self):
        """DEFAULT_TIMEOUT has expected values."""
        assert DEFAULT_TIMEOUT.total == 20
        assert DEFAULT_TIMEOUT.connect == 5
        assert DEFAULT_TIMEOUT.sock_read == 12

    def test_default_retries(self):
        """DEFAULT_RETRIES is 2."""
        assert DEFAULT_RETRIES == 2

    def test_default_backoff_factor(self):
        """DEFAULT_BACKOFF_FACTOR is 0.3."""
        assert DEFAULT_BACKOFF_FACTOR == 0.3

    def test_retry_status_codes(self):
        """RETRY_STATUS_CODES contains the expected codes."""
        assert RETRY_STATUS_CODES == {429, 500, 502, 503, 504}
