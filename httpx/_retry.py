"""
An explicitly enabled retry policy for clients.

A `RetryPolicy` bounds a single logical request by a unified attempt budget
and an overall deadline. The same budget is shared across the authentication
flow, redirect handling, and transport-level retries, so none of those layers
restarts its own count from zero.

Note that HTTPX does not attempt to deduplicate requests on the server side.
If a request fails after it may have been processed (for example a read
failure once the request has been sent), the caller is responsible for
deciding whether retrying is safe. Retries are therefore restricted to
`GET`/`HEAD` requests, plus any methods the caller explicitly permits.
"""

from __future__ import annotations

import datetime
import email.utils
import time
import typing

from ._content import ByteStream
from ._exceptions import RequestNotReplayable, RetryBudgetExceeded

if typing.TYPE_CHECKING:  # pragma: no cover
    from ._models import Request, Response

__all__ = ["RetryPolicy"]


def _parse_retry_after(value: str) -> float | None:
    """
    Parse a `Retry-After` header value into a number of seconds,
    or `None` if the value cannot be understood.
    """
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    return max(0.0, (parsed - now).total_seconds())


class RetryPolicy:
    """
    Configuration for retrying requests, enabled explicitly per client or
    per request.

    **Parameters:**

    * **max_attempts** - The total number of times a request may be sent onto
    the wire for one logical request, including authentication re-sends and
    redirect hops. Must be at least 1.
    * **deadline** - *(optional)* The total number of seconds one logical
    request may take across all attempts and backoff sleeps. `None` disables
    the time limit.
    * **backoff** - The base delay in seconds between retries. The delay
    doubles after each retry, up to **max_backoff**.
    * **max_backoff** - The maximum backoff delay in seconds.
    * **retry_methods** - The methods that may be retried. Defaults to
    `("GET", "HEAD")`. Other methods require an explicit idempotency permit,
    either by including them here or by setting `extensions={"idempotent":
    True}` on the request.
    * **retry_status_codes** - Response status codes that trigger a retry.
    * **retry_on_connect_error** - Whether to retry when establishing the
    connection fails. The server cannot have seen these requests.
    * **retry_on_read_error** - Whether to retry when reading the response
    fails after the request was sent. The server may already have processed
    such requests; HTTPX leaves that judgement to the caller and does not
    protect against duplicate writes.
    * **respect_retry_after** - Whether to honour `Retry-After` response
    headers when computing the delay before the next attempt.
    * **clock** - The monotonic clock used to measure the deadline.
    * **sleep** / **async_sleep** - The callables used to wait between
    attempts for sync and async clients.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        deadline: float | None = None,
        backoff: float = 0.5,
        max_backoff: float = 30.0,
        retry_methods: typing.Iterable[str] = ("GET", "HEAD"),
        retry_status_codes: typing.Iterable[int] = (429, 502, 503, 504),
        retry_on_connect_error: bool = True,
        retry_on_read_error: bool = True,
        respect_retry_after: bool = True,
        clock: typing.Callable[[], float] = time.monotonic,
        sleep: typing.Callable[[float], None] = time.sleep,
        async_sleep: typing.Callable[[float], typing.Awaitable[None]] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("'max_attempts' must be at least 1.")
        if deadline is not None and deadline <= 0:
            raise ValueError("'deadline' must be greater than 0.")
        if backoff < 0:
            raise ValueError("'backoff' must not be negative.")
        if max_backoff < 0:
            raise ValueError("'max_backoff' must not be negative.")
        if async_sleep is None:
            import anyio

            async_sleep = anyio.sleep

        self.max_attempts = max_attempts
        self.deadline = deadline
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.retry_methods = frozenset(method.upper() for method in retry_methods)
        self.retry_status_codes = frozenset(retry_status_codes)
        self.retry_on_connect_error = retry_on_connect_error
        self.retry_on_read_error = retry_on_read_error
        self.respect_retry_after = respect_retry_after
        self.clock = clock
        self.sleep = sleep
        self.async_sleep = async_sleep

    def is_method_allowed(self, request: Request) -> bool:
        """
        Return `True` if retrying the given request is permitted, either
        because the method is in `retry_methods`, or because the request
        carries an explicit idempotency permit.
        """
        if request.method in self.retry_methods:
            return True
        return bool(request.extensions.get("idempotent", False))

    def retry_after_delay(self, response: Response) -> float | None:
        """
        Return the server-requested delay from a `Retry-After` header,
        or `None` if there is none or it cannot be parsed.
        """
        if not self.respect_retry_after:
            return None
        value = response.headers.get("retry-after")
        if value is None:
            return None
        return _parse_retry_after(value)


class RetryState:
    """
    Tracks the shared attempt budget and deadline for one logical request.

    A single instance is created per `Client.send()` call and is threaded
    through the authentication, redirect, and transport layers, so that
    every wire-level send draws from the same budget instead of each layer
    restarting its own count from zero.
    """

    def __init__(self, policy: RetryPolicy) -> None:
        self.policy = policy
        self.attempts = 0
        self.retries = 0
        self.started_at = policy.clock()

    @property
    def elapsed(self) -> float:
        return self.policy.clock() - self.started_at

    def before_attempt(self, request: Request) -> None:
        """
        Account for one wire-level send, raising `RetryBudgetExceeded` if
        the unified attempt budget or deadline has been exhausted.
        """
        if self.attempts >= self.policy.max_attempts:
            raise RetryBudgetExceeded(
                "Exceeded the retry policy budget of "
                f"{self.policy.max_attempts} attempts.",
                request=request,
            )
        deadline = self.policy.deadline
        if deadline is not None and self.elapsed >= deadline:
            raise RetryBudgetExceeded(
                f"Exceeded the retry policy deadline of {deadline} seconds.",
                request=request,
            )
        self.attempts += 1

    def retry_delay(
        self, request: Request, response: Response | None = None
    ) -> float | None:
        """
        Return the delay in seconds before the next attempt, or `None` if no
        further retry should be made, either because the budget or deadline
        is exhausted, or because the request method is not permitted.

        Raises `RequestNotReplayable` if the request body is a streaming body
        that cannot be replayed, refusing the retry.
        """
        policy = self.policy
        if self.attempts >= policy.max_attempts:
            return None
        if not policy.is_method_allowed(request):
            return None
        if not isinstance(request.stream, ByteStream):
            raise RequestNotReplayable(
                "Cannot retry a request with a non-replayable streaming body.",
                request=request,
            )

        delay = min(policy.backoff * (2**self.retries), policy.max_backoff)
        if response is not None:
            retry_after = policy.retry_after_delay(response)
            if retry_after is not None:
                delay = max(delay, retry_after)

        if policy.deadline is not None:
            remaining = policy.deadline - self.elapsed
            if delay >= remaining:
                return None

        self.retries += 1
        return delay
