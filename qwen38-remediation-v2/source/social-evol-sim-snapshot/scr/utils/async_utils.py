"""
Shared async utilities: retry decorator, timeout wrapper, exponential backoff.
"""

import asyncio
import random
from typing import TypeVar, Callable, Awaitable
from functools import wraps
from scr.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def exponential_backoff_delay(
    attempt: int,
    base: float = 1.0,
    max_delay: float = 30.0,
) -> float:
    """Calculate exponential backoff delay with jitter.

    Returns min(2^attempt * base + random(0,1), max_delay)
    """
    delay = min(2**attempt * base + random.random(), max_delay)
    return delay


async def with_timeout(
    coro: Awaitable[T],
    timeout_seconds: float = 60.0,
    operation_name: str = "operation",
) -> T:
    """Wrap a coroutine with a timeout.

    Args:
        coro: The coroutine to wrap.
        timeout_seconds: Timeout in seconds.
        operation_name: Name for logging.

    Returns:
        The result of the coroutine.

    Raises:
        asyncio.TimeoutError: If the operation exceeds the timeout.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"{operation_name} timed out after {timeout_seconds}s")
        raise
