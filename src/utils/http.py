"""HTTP utilities with unified timeout and retry configuration."""

import time
from typing import Optional, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.logging_config import get_logger

logger = get_logger("http")

# Default timeout configuration
DEFAULT_TIMEOUT = 15  # seconds
DEFAULT_CONNECT_TIMEOUT = 5  # seconds
DEFAULT_READ_TIMEOUT = 15  # seconds

# Retry configuration
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5  # 0.5, 1, 2 seconds
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]


def create_session(
    retries: int = DEFAULT_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    status_forcelist: list[int] = None,
) -> requests.Session:
    """Create a requests session with retry configuration.

    Args:
        retries: Number of retries for failed requests.
        backoff_factor: Backoff factor for retries (exponential).
        status_forcelist: HTTP status codes to retry on.

    Returns:
        Configured requests.Session.
    """
    if status_forcelist is None:
        status_forcelist = RETRY_STATUS_CODES

    session = requests.Session()

    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def get_with_retry(
    url: str,
    headers: dict = None,
    params: dict = None,
    timeout: tuple[int, int] = None,
    session: requests.Session = None,
) -> Optional[requests.Response]:
    """Make GET request with retry logic.

    Args:
        url: URL to fetch.
        headers: Request headers.
        params: Query parameters.
        timeout: (connect_timeout, read_timeout) tuple.
        session: Optional session to reuse.

    Returns:
        Response object or None if all retries failed.
    """
    if timeout is None:
        timeout = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)

    if session is None:
        session = create_session()

    try:
        response = session.get(
            url,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            # Rate limited - get retry-after header
            retry_after = e.response.headers.get("Retry-After", "60")
            try:
                wait_time = int(retry_after)
            except ValueError:
                wait_time = 60
            logger.warning("Rate limited, waiting %d seconds", wait_time)
            time.sleep(min(wait_time, 120))  # Cap at 2 minutes
            # One more attempt after waiting
            try:
                response = session.get(url, headers=headers, params=params, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.RequestException:
                pass
        logger.error("HTTP error for %s: %s", url, e)
        return None
    except requests.exceptions.Timeout:
        logger.error("Timeout fetching %s", url)
        return None
    except requests.exceptions.RequestException as e:
        logger.error("Request error for %s: %s", url, e)
        return None


def get_json(
    url: str,
    headers: dict = None,
    params: dict = None,
    timeout: tuple[int, int] = None,
    session: requests.Session = None,
) -> Optional[Any]:
    """Make GET request and parse JSON response.

    Args:
        url: URL to fetch.
        headers: Request headers.
        params: Query parameters.
        timeout: (connect_timeout, read_timeout) tuple.
        session: Optional session to reuse.

    Returns:
        Parsed JSON or None if failed.
    """
    response = get_with_retry(url, headers, params, timeout, session)
    if response is None:
        return None

    try:
        return response.json()
    except ValueError as e:
        logger.error("JSON parse error for %s: %s", url, e)
        return None


# Global session for reuse
_session: Optional[requests.Session] = None


def get_session() -> requests.Session:
    """Get or create global session with retry configuration."""
    global _session
    if _session is None:
        _session = create_session()
    return _session
