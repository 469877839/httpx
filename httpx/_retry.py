"""
An opt-in retry policy for clients.

A `Retry` instance bounds a single *logical* request. A logical request
includes any redirects that are followed and any authentication flow
requests, so retries, redirects and authentication do not each get their
own fresh attempt budget or deadline.

Note that retrying is only ever safe for idempotent requests. A server may
have already processed a request before a connection, read, or status-code
based failure is observed. The retry policy deliberately leaves the
"the server may have processed this" judgement to the caller, and does not
attempt to deduplicate writes.
"""

from __future__ import annotations

import time
import typing
from email.utils import parsedate_to_datetime

from ._exceptions import (
    ConnectError,
    ConnectTimeout,
    NonReplayableRequest,
    PoolTimeout,
    ReadError,
    ReadTimeout,
    RetryExhausted,
    RetryTimeout,
    TransportError,
)

if typing.TYPE_CHECKING:  # pragma: no cover
    from ._models import Request, Response

__all__ = ["Retry"]


DEFAULT_RETRY_STATUS_CODES = (502, 503, 504)
DEFAULT_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD"})

# Failures where the request was never sent over the wire.
CONNECT_ERRORS = (ConnectError, ConnectTimeout, PoolTimeout)
# Failures where the request was sent, but reading the response failed.
READ_ERRORS = (ReadError, ReadTimeout)


async def _default_async_sleep(seconds: float) -> None:
    import anyio

    await anyio.sleep(seconds)


class Retry:
    """
    A retry policy, limiting the number of attempts and the overall
    duration of a single logical request.

    Retries are *disabled by default*. Pass an instance to a client to
    enable them:

    ```python
    retry = httpx.Retry(max_attempts=3, total=10.0)
    client = httpx.Client(retry=retry)
    ```

    **Parameters:**

    * **max_attempts** - *(optional)* The maximum number of times a request
    may be sent, including the initial attempt, any redirects, and any
    authentication flow requests.
    * **total** - *(optional)* The maximum number of seconds a logical
    request may take, including time spent waiting between retries.
    * **idempotent_methods** - *(optional)* The HTTP methods that may be
    retried. Defaults to `{"GET", "HEAD"}`. Other methods require an
    explicit per-request opt-in, using `client.request(..., idempotent=True)`.
    * **retry_status_codes** - *(optional)* Response status codes that
    trigger a retry. Defaults to `(502, 503, 504)`. Set to `None` to
    disable status-code based retries.
    * **retry_connect_errors** - *(optional)* Whether to retry when
    establishing a connection fails. Defaults to `True`.
    * **retry_read_errors** - *(optional)* Whether to retry when reading
    the response fails after the request has been sent. Defaults to `True`.
    * **backoff_factor** - *(optional)* The base delay, in seconds, used to
    compute the wait between attempts. The delay grows exponentially.
    * **backoff_max** - *(optional)* The maximum delay, in seconds, between
    attempts.
    * **respect_retry_after** - *(optional)* Whether to wait for the
    duration indicated by a `Retry-After` response header. Defaults to
    `True`.
    * **clock** - *(optional)* A monotonic clock, used to measure the
    overall deadline. Defaults to `time.monotonic`.
    * **sleep** - *(optional)* The function used to wait between attempts
    in a synchronous client. Defaults to `time.sleep`.
    * **async_sleep** - *(optional)* The function used to wait between
    attempts in an asynchronous client. Defaults to `anyio.sleep`.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        total: float = 30.0,
        idempotent_methods: typing.Iterable[str] = DEFAULT_IDEMPOTENT_METHODS,
        retry_status_codes: typing.Iterable[int] | None = DEFAULT_RETRY_STATUS_CODES,
        retry_connect_errors: bool = True,
        retry_read_errors: bool = True,
        backoff_factor: float = 0.5,
        backoff_max: float = 60.0,
        respect_retry_after: bool = True,
        clock: typing.Callable[[], float] = time.monotonic,
        sleep: typing.Callable[[float], None] = time.sleep,
        async_sleep: typing.Callable[[float], typing.Awaitable[None]] = (
            _default_async_sleep
        ),
    ) -> None:
        if max_attempts < 1:
            raise ValueError("'max_attempts' must be at least 1.")
        if total <= 0:
            raise ValueError("'total' must be greater than zero.")

        self.max_attempts = max_attempts
        self.total = total
        self.idempotent_methods = frozenset(
            method.upper() for method in idempotent_methods
        )
        self.retry_status_codes = (
            frozenset() if retry_status_codes is None else frozenset(retry_status_codes)
        )
        self.retry_connect_errors = retry_connect_errors
        self.retry_read_errors = retry_read_errors
        self.backoff_factor = backoff_factor
        self.backoff_max = backoff_max
        self.respect_retry_after = respect_retry_after
        self.clock = clock
        self.sleep = sleep
        self.async_sleep = async_sleep

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"<{class_name}(max_attempts={self.max_attempts}, total={self.total})>"

    def new_budget(self) -> RetryBudget:
        """
        Start tracking attempts and elapsed time for one logical request.
        """
        return RetryBudget(self)

    def is_retryable_method(self, request: Request) -> bool:
        """
        Return `True` if the request method may be retried.

        Methods outside of `idempotent_methods` require an explicit
        per-request opt-in, using `client.request(..., idempotent=True)`.
        """
        if request.method in self.idempotent_methods:
            return True
        return bool(request.extensions.get("idempotent", False))

    def is_retryable_exception(self, exc: TransportError) -> bool:
        """
        Return `True` if a transport-level failure may be retried.

        Connection establishment failures and post-send read failures are
        distinguished, and each can be disabled independently. Failures
        while sending the request body, protocol errors, and proxy errors
        are never retried, since the request may have been partially sent.
        """
        if isinstance(exc, CONNECT_ERRORS):
            return self.retry_connect_errors
        if isinstance(exc, READ_ERRORS):
            return self.retry_read_errors
        return False

    def is_retryable_status(self, response: Response) -> bool:
        """
        Return `True` if the response status code may be retried.
        """
        return response.status_code in self.retry_status_codes

    def compute_delay(self, attempt: int, response: Response | None) -> float:
        """
        Return the delay, in seconds, before the next attempt.

        A `Retry-After` header on a retryable response takes precedence
        over the exponential backoff.
        """
        if response is not None and self.respect_retry_after:
            retry_after = self._parse_retry_after(response)
            if retry_after is not None:
                return retry_after
        delay = self.backoff_factor * (2 ** max(attempt - 1, 0))
        return min(delay, self.backoff_max)

    def _parse_retry_after(self, response: Response) -> float | None:
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        now = time.time()
        return max(0.0, retry_at.timestamp() - now)


class RetryBudget:
    """
    Tracks the attempts made and the time elapsed for one logical request.

    A single budget is shared across redirects and authentication flows,
    so that each does not restart the attempt count or the deadline.
    """

    def __init__(self, retry: Retry) -> None:
        self._retry = retry
        self._started = retry.clock()
        self.attempts = 0

    @property
    def elapsed(self) -> float:
        return self._retry.clock() - self._started

    def register_attempt(self, request: Request) -> None:
        """
        Count an attempt, raising if the attempt budget has been spent.
        """
        self.attempts += 1
        if self.attempts > self._retry.max_attempts:
            raise RetryExhausted(
                f"Exceeded maximum of {self._retry.max_attempts} attempts.",
                request=request,
            )

    def has_attempt_remaining(self) -> bool:
        """
        Return `True` if another attempt can still be made.
        """
        return self.attempts < self._retry.max_attempts

    def check_deadline(self, request: Request, delay: float = 0.0) -> None:
        """
        Raise if waiting `delay` seconds would exceed the overall deadline.
        """
        if self.elapsed + delay > self._retry.total:
            raise RetryTimeout(
                f"Exceeded retry deadline of {self._retry.total} seconds.",
                request=request,
            )

    def check_replayable(self, request: Request) -> None:
        """
        Raise if the request body cannot be sent again.
        """
        if not request.stream.is_replayable:
            raise NonReplayableRequest(
                "Cannot retry a request with a non-replayable streaming body.",
                request=request,
            )
