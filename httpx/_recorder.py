"""
An optional, privacy-conscious recorder of the logical steps that make up a
single client request.

`Response.history` only contains the responses whose redirect was followed
successfully.  That is not enough to reconstruct what happened when a request
failed part-way through authentication or redirect handling.  A
`RequestRecorder` instead records every network attempt, why it was made, and
where its outcome became known.

Records are deliberately limited to information that is available without
consuming any request or response stream.  Headers and bodies are omitted by
default, and credential-bearing headers are always redacted when header
capture is enabled.

Usage:

```python
recorder = httpx.RequestRecorder()
client = httpx.Client(recorder=recorder, transport=...)
client.get("https://example.org/")

for attempt in recorder.records[0].attempts:
    print(attempt.trigger, attempt.method, attempt.url, attempt.outcome)
```
"""

from __future__ import annotations

import dataclasses
import enum
import threading
import time
import typing
from dataclasses import field

from ._models import Headers, Request, Response

__all__ = [
    "AttemptOutcome",
    "AttemptTrigger",
    "AuthStep",
    "AuthStepOutcome",
    "RequestAttempt",
    "RequestRecord",
    "RequestRecorder",
]


#: Header names whose values are redacted when header capture is enabled.
SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "www-authenticate",
        "proxy-authenticate",
        "x-api-key",
    }
)

REDACTED = "[redacted]"


class AttemptTrigger(str, enum.Enum):
    """
    The step that caused a network attempt to be made.
    """

    #: The first attempt in a logical request.
    INITIAL = "initial"
    #: A request yielded by an authentication flow after a response.
    AUTH_CONTINUATION = "auth-continuation"
    #: A request built from a redirect response.
    REDIRECT = "redirect"


class AttemptOutcome(str, enum.Enum):
    """
    How far an individual network attempt got before its boundary became known.

    An attempt that reached a response keeps its response fields even when a
    *later* step fails.  Only one final outcome is recorded per attempt.
    """

    #: The attempt has started, but no boundary has been observed yet.
    PENDING = "pending"
    #: The transport returned a response.  This does not imply the body was read.
    RESPONSE = "response"
    #: The transport raised an exception instead of returning a response.
    TRANSPORT_ERROR = "transport-error"
    #: A request or response event hook raised, interrupting the attempt.
    HOOK_ERROR = "hook-error"
    #: The response body was delivered, but reading it later failed.
    READ_ERROR = "read-error"


class AuthStepOutcome(str, enum.Enum):
    """
    The boundary observed for one authentication-flow continuation step.
    """

    #: The flow asked for another request.
    CONTINUED = "continued"
    #: The flow returned, so no further request was made.
    ENDED = "ended"
    #: The flow raised while handling the response.
    EXCEPTION = "exception"


@dataclasses.dataclass
class RequestAttempt:
    """
    One real transport-level attempt to send a request.

    Timing values use `time.perf_counter()` and are therefore monotonic, but
    they are not wall-clock timestamps.  Exact contents of headers and bodies
    are only present when the recorder was configured to capture them.
    """

    #: Position of this attempt in the logical request, starting at 0.
    sequence: int
    #: The step that caused this attempt.
    trigger: AttemptTrigger
    #: Authentication-flow step that yielded this attempt (0-based).
    auth_step: int
    #: Redirect step that produced this attempt (0-based).
    redirect_step: int
    #: The method actually sent to the transport.
    method: str
    #: The target actually sent, with URL credentials removed.
    url: str
    #: Monotonic start time, taken immediately before the transport call.
    started: float
    #: Monotonic end time, or `None` while no boundary has been observed.
    finished: float | None = None
    #: `finished - started`, or `None` while no boundary has been observed.
    elapsed: float | None = None
    #: The furthest boundary known for this attempt.
    outcome: AttemptOutcome = AttemptOutcome.PENDING
    #: Status of the response returned by the transport, if any.
    response_status: int | None = None
    #: HTTP version reported by the transport, if any.
    http_version: str | None = None
    #: The redirect status that caused this attempt, if any.
    redirect_status: int | None = None
    #: For redirect/auth follow-ups, the method used by the preceding attempt.
    previous_method: str | None = None
    #: Whether this attempt differs in method from the preceding attempt.
    method_changed: bool | None = None
    #: Exception class name when the attempt failed.
    exception_type: str | None = None
    #: Exception message when the attempt failed.
    exception_message: str | None = None
    #: Request headers, only when `capture_headers=True`.
    #: Sensitive values are redacted.
    request_headers: list[tuple[str, str]] | None = None
    #: Request body, only when `capture_content=True` and already in memory.
    request_content: bytes | None = None
    #: Response headers, only when `capture_headers=True`.
    #: Sensitive values are redacted.
    response_headers: list[tuple[str, str]] | None = None
    #: Response body, only when `capture_content=True` and already in memory.
    response_content: bytes | None = None

    def as_dict(self) -> dict[str, typing.Any]:
        """
        Return a plain dictionary, with enums converted to their values.
        """
        result: dict[str, typing.Any] = {}
        for name, value in dataclasses.asdict(self).items():
            if isinstance(value, enum.Enum):
                value = value.value
            result[name] = value
        return result


@dataclasses.dataclass
class AuthStep:
    """
    One continuation step of an authentication flow.
    """

    #: Position in the logical request's authentication flow.
    sequence: int
    #: The step's outcome.
    outcome: AuthStepOutcome
    #: Status of the response handed to the flow, if a response was available.
    response_status: int | None = None
    #: Exception class name when `outcome` is `exception`.
    exception_type: str | None = None
    #: Exception message when `outcome` is `exception`.
    exception_message: str | None = None


@dataclasses.dataclass
class RequestRecord:
    """
    The complete logical record for a single `client.send()` call.
    """

    #: The request as it was passed to `client.send()`.
    request: Request
    #: Every network attempt made while handling this logical request.
    attempts: list[RequestAttempt] = field(default_factory=list)
    #: Authentication continuation steps, in order.
    auth_steps: list[AuthStep] = field(default_factory=list)


def _exception_details(exc: BaseException) -> tuple[str, str]:
    return type(exc).__name__, str(exc)


def _redact_url(request: Request) -> str:
    url = request.url
    if url.username or url.password:
        url = url.copy_with(username=None, password=None)
    return str(url)


class RequestRecorder:
    """
    Collect in-memory records of logical requests, including every actual
    network attempt and the step that triggered it.

    The recorder never reads a request or response stream.  Consequently a
    body is only available if it had already been materialised before the
    recorder observed the request or response.

    **Parameters:**

    * **capture_headers** - Include request and response headers, redacting
      credential-bearing values.  Defaults to `False`.
    * **capture_content** - Include request and response bodies that are
      already available in memory.  Defaults to `False`.
    """

    def __init__(
        self,
        *,
        capture_headers: bool = False,
        capture_content: bool = False,
    ) -> None:
        self.capture_headers = capture_headers
        self.capture_content = capture_content
        self._lock = threading.Lock()
        self._sequence = 0
        self._records: list[RequestRecord] = []

    @property
    def records(self) -> list[RequestRecord]:
        """
        The recorded logical requests, oldest first.
        """
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        """
        Discard all previously recorded requests.
        """
        with self._lock:
            self._records.clear()
            self._sequence = 0

    def _start_record(self, request: Request) -> RequestRecord:
        with self._lock:
            record = RequestRecord(request=request)
            self._records.append(record)
            return record

    def _next_sequence(self) -> int:
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
            return sequence

    def _header_snapshot(self, headers: Headers) -> list[tuple[str, str]] | None:
        if not self.capture_headers:
            return None
        return [
            (
                key,
                REDACTED if key.lower() in SENSITIVE_HEADERS else value,
            )
            for key, value in headers.multi_items()
        ]

    def _content_snapshot(self, message: Request | Response) -> bytes | None:
        if not self.capture_content:
            return None
        return getattr(message, "_content", None)

    def _record_attempt(
        self,
        record: RequestRecord,
        request: Request,
        *,
        trigger: AttemptTrigger,
        auth_step: int,
        redirect_step: int,
        response: Response | None = None,
        redirect_status: int | None = None,
    ) -> RequestAttempt:
        attempt = RequestAttempt(
            sequence=self._next_sequence(),
            trigger=trigger,
            auth_step=auth_step,
            redirect_step=redirect_step,
            method=request.method,
            url=_redact_url(request),
            started=time.perf_counter(),
            redirect_status=redirect_status,
            request_headers=self._header_snapshot(request.headers),
            request_content=self._content_snapshot(request),
        )
        if response is not None:
            attempt.previous_method = response.request.method
            attempt.method_changed = request.method != response.request.method
        with self._lock:
            record.attempts.append(attempt)
        return attempt

    def _finish_attempt(
        self,
        attempt: RequestAttempt,
        outcome: AttemptOutcome,
        *,
        response: Response | None = None,
        exc: BaseException | None = None,
    ) -> None:
        attempt.outcome = outcome
        attempt.finished = time.perf_counter()
        attempt.elapsed = attempt.finished - attempt.started
        if response is not None:
            attempt.response_status = response.status_code
            attempt.http_version = response.http_version
            attempt.response_headers = self._header_snapshot(response.headers)
            attempt.response_content = self._content_snapshot(response)
        if exc is not None:
            attempt.exception_type, attempt.exception_message = _exception_details(exc)

    def _record_auth_step(
        self,
        record: RequestRecord,
        *,
        outcome: AuthStepOutcome,
        sequence: int,
        response: Response | None = None,
        exc: BaseException | None = None,
    ) -> None:
        step = AuthStep(
            sequence=sequence,
            outcome=outcome,
            response_status=None if response is None else response.status_code,
        )
        if exc is not None:
            step.exception_type, step.exception_message = _exception_details(exc)
        with self._lock:
            record.auth_steps.append(step)
