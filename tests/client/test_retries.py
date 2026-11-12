from __future__ import annotations

import typing

import pytest

import httpx


class FakeClock:
    """
    A deterministic clock with explicit sleep/async sleep functions, so
    retry timing can be asserted without waiting in real time.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    async def async_sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


Event = typing.Union[httpx.Response, Exception]

FAILURE_CLASSES = {
    "connect_error": httpx.ConnectError,
    "connect_timeout": httpx.ConnectTimeout,
    "pool_timeout": httpx.PoolTimeout,
    "read_error": httpx.ReadError,
    "read_timeout": httpx.ReadTimeout,
    "write_error": httpx.WriteError,
    "protocol_error": httpx.RemoteProtocolError,
}


class SequenceTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """
    Replays a fixed sequence of responses or exceptions.

    Unlike `MockTransport`, this does not read the request body before
    invoking its handler, so non-replayable streams stay non-replayable,
    as they would with a real transport.
    """

    def __init__(self, events: typing.Sequence[Event]) -> None:
        self.events = list(events)
        self.requests: list[httpx.Request] = []

    def _next(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.events) - 1)
        event = self.events[index]
        if isinstance(event, BaseException):
            raise event
        return event

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._next(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._next(request)


class CloseTrackingStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closes = 0

    def __iter__(self) -> typing.Iterator[bytes]:
        yield b""

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        yield b""

    def close(self) -> None:
        self.closes += 1

    async def aclose(self) -> None:
        self.closes += 1


def make_retry(clock: FakeClock, **kwargs: typing.Any) -> httpx.Retry:
    kwargs.setdefault("max_attempts", 4)
    kwargs.setdefault("total", 100.0)
    kwargs.setdefault("backoff_factor", 1.0)
    return httpx.Retry(
        clock=clock,
        sleep=clock.sleep,
        async_sleep=clock.async_sleep,
        **kwargs,
    )


def ok_response(text: str = "ok") -> httpx.Response:
    return httpx.Response(200, text=text)


# ---------------------------------------------------------------------------
# Policy defaults and validation
# ---------------------------------------------------------------------------


def test_retries_are_not_enabled_by_default() -> None:
    assert httpx.Client().retry is None
    assert httpx.AsyncClient().retry is None


def test_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        httpx.Retry(max_attempts=0)
    with pytest.raises(ValueError):
        httpx.Retry(total=0.0)


def test_client_rejects_invalid_retry_argument() -> None:
    with pytest.raises(TypeError):
        httpx.Client(retry="always")  # type: ignore[arg-type]


def test_default_idempotent_methods() -> None:
    retry = httpx.Retry()
    assert retry.is_retryable_method(httpx.Request("GET", "http://example.org"))
    assert retry.is_retryable_method(httpx.Request("HEAD", "http://example.org"))
    assert not retry.is_retryable_method(httpx.Request("POST", "http://example.org"))


# ---------------------------------------------------------------------------
# Transport failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    ["connect_error", "connect_timeout", "pool_timeout", "read_error", "read_timeout"],
)
def test_retries_transport_failures(failure: str) -> None:
    clock = FakeClock()
    transport = SequenceTransport(
        [
            FAILURE_CLASSES[failure]("failure"),
            FAILURE_CLASSES[failure]("failure"),
            ok_response(),
        ]
    )
    retry = make_retry(clock, max_attempts=3, backoff_factor=0.25)

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert len(transport.requests) == 3
    assert clock.sleeps == [0.25, 0.5]


@pytest.mark.parametrize("failure", ["write_error", "protocol_error"])
def test_does_not_retry_failures_that_may_have_sent_the_request(failure: str) -> None:
    clock = FakeClock()
    transport = SequenceTransport([FAILURE_CLASSES[failure]("failure")])
    retry = make_retry(clock)

    with httpx.Client(transport=transport, retry=retry) as client:
        with pytest.raises(FAILURE_CLASSES[failure]):
            client.get("http://example.org/")

    assert len(transport.requests) == 1
    assert clock.sleeps == []


def test_read_errors_can_be_disabled_independently() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ReadError("no"), ok_response()])
    retry = make_retry(clock, retry_read_errors=False)

    with httpx.Client(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.ReadError):
            client.get("http://example.org/")

    assert len(transport.requests) == 1


def test_connect_errors_can_be_disabled_independently() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no"), ok_response()])
    retry = make_retry(clock, retry_connect_errors=False)

    with httpx.Client(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.ConnectError):
            client.get("http://example.org/")

    assert len(transport.requests) == 1


def test_exhausted_attempts_raise_from_final_error() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no")])
    retry = make_retry(clock, max_attempts=2, backoff_factor=0.1)

    with httpx.Client(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.RetryExhausted) as exc_info:
            client.get("http://example.org/")

    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)
    assert exc_info.value.request.url == httpx.URL("http://example.org/")
    assert len(transport.requests) == 2


# ---------------------------------------------------------------------------
# Status codes and Retry-After
# ---------------------------------------------------------------------------


def test_retries_configured_status_codes() -> None:
    clock = FakeClock()
    transport = SequenceTransport(
        [httpx.Response(503), httpx.Response(503), ok_response()]
    )
    retry = make_retry(clock, max_attempts=3, backoff_factor=0.5)

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert len(transport.requests) == 3
    assert clock.sleeps == [0.5, 1.0]


def test_does_not_retry_unlisted_status_codes() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.Response(500)])
    retry = make_retry(clock)

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 500
    assert len(transport.requests) == 1


def test_status_code_retries_can_be_disabled() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.Response(503)])
    retry = make_retry(clock, retry_status_codes=None)

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 503
    assert len(transport.requests) == 1


def test_returns_final_response_when_attempts_exhausted_by_status() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.Response(503)])
    retry = make_retry(clock, max_attempts=2, backoff_factor=0.1)

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 503
    assert len(transport.requests) == 2
    assert clock.sleeps == [0.1]


def test_retry_after_header_takes_precedence_over_backoff() -> None:
    clock = FakeClock()
    transport = SequenceTransport(
        [httpx.Response(503, headers={"Retry-After": "5"}), ok_response()]
    )
    retry = make_retry(clock, max_attempts=2, backoff_factor=0.5)

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert clock.sleeps == [5.0]


def test_retry_after_can_be_ignored() -> None:
    clock = FakeClock()
    transport = SequenceTransport(
        [httpx.Response(503, headers={"Retry-After": "5"}), ok_response()]
    )
    retry = make_retry(
        clock, max_attempts=2, backoff_factor=0.5, respect_retry_after=False
    )

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert clock.sleeps == [0.5]


def test_retry_after_http_date() -> None:
    clock = FakeClock()
    response = httpx.Response(
        503, headers={"Retry-After": "Wed, 21 Oct 2065 07:28:00 GMT"}
    )
    retry = make_retry(clock)
    delay = retry._parse_retry_after(response)
    assert delay is not None and delay > 0

    malformed = httpx.Response(503, headers={"Retry-After": "not-a-date"})
    assert retry._parse_retry_after(malformed) is None


# ---------------------------------------------------------------------------
# Overall deadline
# ---------------------------------------------------------------------------


def test_deadline_stops_retries_before_the_next_attempt() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no")])
    retry = make_retry(clock, max_attempts=10, total=3.0, backoff_factor=1.0)

    with httpx.Client(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.RetryTimeout) as exc_info:
            client.get("http://example.org/")

    assert exc_info.value.request.url == httpx.URL("http://example.org/")
    # Attempts at t=0, t=1 and t=3. The next wait of 4s would take us to
    # t=7, beyond the 3s deadline, so no further attempt is made.
    assert len(transport.requests) == 3
    assert clock.sleeps == [1.0, 2.0]
    assert clock.now == 3.0


def test_retry_after_beyond_deadline_raises_retry_timeout() -> None:
    clock = FakeClock()
    transport = SequenceTransport(
        [httpx.Response(503, headers={"Retry-After": "30"})]
    )
    retry = make_retry(clock, max_attempts=5, total=10.0)

    with httpx.Client(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.RetryTimeout):
            client.get("http://example.org/")

    assert len(transport.requests) == 1
    assert clock.sleeps == []


# ---------------------------------------------------------------------------
# Idempotency and replayability
# ---------------------------------------------------------------------------


def test_post_is_not_retried_by_default() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no")])
    retry = make_retry(clock)

    with httpx.Client(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.ConnectError):
            client.post("http://example.org/", content=b"data")

    assert len(transport.requests) == 1


def test_post_is_retried_with_explicit_idempotent_opt_in() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no"), ok_response()])
    retry = make_retry(clock, max_attempts=2, backoff_factor=0.1)

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.post(
            "http://example.org/", content=b"data", idempotent=True
        )

    assert response.status_code == 200
    assert len(transport.requests) == 2


def test_additional_idempotent_methods_can_be_configured() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no"), ok_response()])
    retry = make_retry(
        clock, max_attempts=2, backoff_factor=0.1, idempotent_methods={"PUT"}
    )

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.put("http://example.org/", content=b"data")

    assert response.status_code == 200
    assert len(transport.requests) == 2


def test_non_replayable_streaming_body_rejects_retry() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no")])
    retry = make_retry(clock)

    def streaming_body() -> typing.Iterator[bytes]:
        yield b"chunk"

    with httpx.Client(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.NonReplayableRequest):
            client.post(
                "http://example.org/", content=streaming_body(), idempotent=True
            )

    assert len(transport.requests) == 1
    assert clock.sleeps == []


def test_non_replayable_streaming_body_rejects_status_code_retry() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.Response(503)])
    retry = make_retry(clock)

    def streaming_body() -> typing.Iterator[bytes]:
        yield b"chunk"

    with httpx.Client(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.NonReplayableRequest):
            client.post(
                "http://example.org/", content=streaming_body(), idempotent=True
            )

    assert len(transport.requests) == 1


# ---------------------------------------------------------------------------
# MockTransport integration
# ---------------------------------------------------------------------------


def test_retries_with_mock_transport() -> None:
    clock = FakeClock()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, text="Hello, world!")

    retry = make_retry(clock, max_attempts=3, backoff_factor=0.5)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=retry) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert response.text == "Hello, world!"
    assert calls == 3
    assert clock.sleeps == [0.5, 1.0]
    assert clock.now == 1.5


@pytest.mark.anyio
async def test_retries_with_mock_transport_async() -> None:
    clock = FakeClock()
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "2"})
        return httpx.Response(200, text="Hello, world!")

    retry = make_retry(clock, max_attempts=3, backoff_factor=0.5)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), retry=retry
    ) as client:
        response = await client.get("http://example.org/")

    assert response.status_code == 200
    assert calls == 2
    assert clock.sleeps == [2.0]
    assert clock.now == 2.0


# ---------------------------------------------------------------------------
# Async transport failures and deadline
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_retries_transport_failures() -> None:
    clock = FakeClock()
    transport = SequenceTransport(
        [httpx.ConnectError("no"), httpx.ReadError("no"), ok_response()]
    )
    retry = make_retry(clock, max_attempts=3, backoff_factor=0.25)

    async with httpx.AsyncClient(transport=transport, retry=retry) as client:
        response = await client.get("http://example.org/")

    assert response.status_code == 200
    assert len(transport.requests) == 3
    assert clock.sleeps == [0.25, 0.5]


@pytest.mark.anyio
async def test_async_exhausted_attempts_raise_from_final_error() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no")])
    retry = make_retry(clock, max_attempts=2, backoff_factor=0.1)

    async with httpx.AsyncClient(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.RetryExhausted) as exc_info:
            await client.get("http://example.org/")

    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)
    assert len(transport.requests) == 2


@pytest.mark.anyio
async def test_async_deadline_stops_retries() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no")])
    retry = make_retry(clock, max_attempts=10, total=3.0, backoff_factor=1.0)

    async with httpx.AsyncClient(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.RetryTimeout):
            await client.get("http://example.org/")

    assert len(transport.requests) == 3
    assert clock.sleeps == [1.0, 2.0]
    assert clock.now == 3.0


@pytest.mark.anyio
async def test_async_returns_final_response_when_attempts_exhausted() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.Response(503)])
    retry = make_retry(clock, max_attempts=2, backoff_factor=0.1)

    async with httpx.AsyncClient(transport=transport, retry=retry) as client:
        response = await client.get("http://example.org/")

    assert response.status_code == 503
    assert len(transport.requests) == 2
    assert clock.sleeps == [0.1]


@pytest.mark.anyio
async def test_async_post_is_retried_with_explicit_idempotent_opt_in() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no"), ok_response()])
    retry = make_retry(clock, max_attempts=2, backoff_factor=0.1)

    async with httpx.AsyncClient(transport=transport, retry=retry) as client:
        response = await client.post(
            "http://example.org/", content=b"data", idempotent=True
        )

    assert response.status_code == 200
    assert len(transport.requests) == 2


@pytest.mark.anyio
async def test_async_non_replayable_streaming_body_rejects_retry() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no")])
    retry = make_retry(clock)

    async def streaming_body() -> typing.AsyncIterator[bytes]:
        yield b"chunk"

    async with httpx.AsyncClient(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.NonReplayableRequest):
            await client.post(
                "http://example.org/", content=streaming_body(), idempotent=True
            )

    assert len(transport.requests) == 1


# ---------------------------------------------------------------------------
# Response streams are closed before retrying
# ---------------------------------------------------------------------------


def test_response_stream_is_closed_between_retries() -> None:
    clock = FakeClock()
    streams = [CloseTrackingStream() for _ in range(3)]
    events: list[Event] = [
        httpx.Response(503, stream=streams[0]),
        httpx.Response(503, stream=streams[1]),
        httpx.Response(200, stream=streams[2]),
    ]
    transport = SequenceTransport(events)
    retry = make_retry(clock, max_attempts=3, backoff_factor=0.1)

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert streams[0].closes == 1
    assert streams[1].closes == 1
    assert len(transport.requests) == 3


@pytest.mark.anyio
async def test_response_stream_is_closed_between_retries_async() -> None:
    clock = FakeClock()
    streams = [CloseTrackingStream() for _ in range(3)]
    events = [
        httpx.Response(503, stream=streams[0]),
        httpx.Response(503, stream=streams[1]),
        httpx.Response(200, stream=streams[2]),
    ]
    transport = SequenceTransport(events)
    retry = make_retry(clock, max_attempts=3, backoff_factor=0.1)

    async with httpx.AsyncClient(transport=transport, retry=retry) as client:
        response = await client.get("http://example.org/")

    assert response.status_code == 200
    assert streams[0].closes == 1
    assert streams[1].closes == 1
    assert len(transport.requests) == 3


def test_streaming_response_body_read_failure_is_not_retried() -> None:
    # Failures that occur while the caller consumes the response stream
    # happen after the inner retry loop has returned, so they surface
    # unchanged.
    clock = FakeClock()

    class FailingStream(httpx.SyncByteStream):
        def __iter__(self) -> typing.Iterator[bytes]:
            yield b"partial"
            raise httpx.ReadError("connection dropped mid-body")

    transport = SequenceTransport([httpx.Response(200, stream=FailingStream())])
    retry = make_retry(clock)

    with httpx.Client(transport=transport, retry=retry) as client:
        with client.stream("GET", "http://example.org/") as response:
            with pytest.raises(httpx.ReadError):
                response.read()

    assert len(transport.requests) == 1
    assert clock.sleeps == []


# ---------------------------------------------------------------------------
# Event hooks and cancellation must not be swallowed
# ---------------------------------------------------------------------------


class RequestHookError(RuntimeError):
    pass


class ResponseHookError(RuntimeError):
    pass


def test_request_hook_exception_is_not_swallowed() -> None:
    clock = FakeClock()

    def failing_hook(request: httpx.Request) -> None:
        raise RequestHookError("hook failed")

    transport = SequenceTransport([httpx.ConnectError("no"), ok_response()])
    retry = make_retry(clock)

    with httpx.Client(
        transport=transport,
        retry=retry,
        event_hooks={"request": [failing_hook]},
    ) as client:
        with pytest.raises(RequestHookError):
            client.get("http://example.org/")

    assert transport.requests == []
    assert clock.sleeps == []


def test_response_hook_exception_is_not_swallowed() -> None:
    clock = FakeClock()

    def failing_hook(response: httpx.Response) -> None:
        raise ResponseHookError("hook failed")

    transport = SequenceTransport([httpx.Response(503), ok_response()])
    retry = make_retry(clock)

    with httpx.Client(
        transport=transport,
        retry=retry,
        event_hooks={"response": [failing_hook]},
    ) as client:
        with pytest.raises(ResponseHookError):
            client.get("http://example.org/")

    # The 503 was consumed by the inner retry loop before any user hook ran.
    assert len(transport.requests) == 2
    assert clock.sleeps


@pytest.mark.anyio
async def test_async_request_hook_exception_is_not_swallowed() -> None:
    clock = FakeClock()

    async def failing_hook(request: httpx.Request) -> None:
        raise RequestHookError("hook failed")

    transport = SequenceTransport([httpx.ConnectError("no"), ok_response()])
    retry = make_retry(clock)

    async with httpx.AsyncClient(
        transport=transport,
        retry=retry,
        event_hooks={"request": [failing_hook]},
    ) as client:
        with pytest.raises(RequestHookError):
            await client.get("http://example.org/")

    assert transport.requests == []


class Cancellation(BaseException):
    """
    Stands in for an async cancellation (e.g. asyncio.CancelledError), which
    inherits from BaseException rather than Exception.
    """


def test_cancellation_is_not_swallowed_by_sync_retries() -> None:
    clock = FakeClock()
    transport = SequenceTransport([Cancellation("cancelled")])
    retry = make_retry(clock)

    with httpx.Client(transport=transport, retry=retry) as client:
        with pytest.raises(Cancellation):
            client.get("http://example.org/")

    assert len(transport.requests) == 1
    assert clock.sleeps == []


@pytest.mark.anyio
async def test_cancellation_is_not_swallowed_by_async_retries() -> None:
    clock = FakeClock()
    transport = SequenceTransport([Cancellation("cancelled")])
    retry = make_retry(clock)

    async with httpx.AsyncClient(transport=transport, retry=retry) as client:
        with pytest.raises(Cancellation):
            await client.get("http://example.org/")

    assert len(transport.requests) == 1
    assert clock.sleeps == []


# ---------------------------------------------------------------------------
# Policy units
# ---------------------------------------------------------------------------


def test_retry_repr() -> None:
    retry = httpx.Retry(max_attempts=5, total=12.0)
    assert repr(retry) == "<Retry(max_attempts=5, total=12.0)>"


def test_backoff_is_capped() -> None:
    retry = httpx.Retry(backoff_factor=100.0, backoff_max=5.0)
    assert retry.compute_delay(1, None) == 5.0
    assert retry.compute_delay(4, None) == 5.0


def test_retry_after_absent_or_malformed() -> None:
    retry = httpx.Retry()
    assert retry._parse_retry_after(httpx.Response(503)) is None
    response = httpx.Response(503, headers={"Retry-After": "soon"})
    assert retry._parse_retry_after(response) is None


def test_client_retry_property() -> None:
    retry = httpx.Retry()
    client = httpx.Client()
    assert client.retry is None
    client.retry = retry
    assert client.retry is retry
    client.retry = None
    assert client.retry is None
    with pytest.raises(TypeError):
        client.retry = "always"  # type: ignore[assignment]


def test_multipart_with_in_memory_files_is_replayable() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no"), ok_response()])
    retry = make_retry(clock, max_attempts=2, backoff_factor=0.1)

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.post(
            "http://example.org/",
            files={"upload": ("hello.txt", b"Hello, world!")},
            idempotent=True,
        )

    assert response.status_code == 200
    assert len(transport.requests) == 2


def test_multipart_with_file_objects_is_not_replayable() -> None:
    import io

    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no")])
    retry = make_retry(clock)

    with httpx.Client(transport=transport, retry=retry) as client:
        with pytest.raises(httpx.NonReplayableRequest):
            client.post(
                "http://example.org/",
                files={"upload": ("hello.txt", io.BytesIO(b"Hello, world!"))},
                idempotent=True,
            )

    assert len(transport.requests) == 1


def test_default_sleep_functions_are_used() -> None:
    # A zero backoff factor means the default `time.sleep` is called with
    # zero seconds, so this does not actually wait.
    transport = SequenceTransport([httpx.ConnectError("no"), ok_response()])
    retry = httpx.Retry(max_attempts=2, backoff_factor=0.0)

    with httpx.Client(transport=transport, retry=retry) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert len(transport.requests) == 2


@pytest.mark.anyio
async def test_default_async_sleep_function_is_used() -> None:
    transport = SequenceTransport([httpx.ConnectError("no"), ok_response()])
    retry = httpx.Retry(max_attempts=2, backoff_factor=0.0)

    async with httpx.AsyncClient(transport=transport, retry=retry) as client:
        response = await client.get("http://example.org/")

    assert response.status_code == 200
    assert len(transport.requests) == 2


# ---------------------------------------------------------------------------
# Redirects and authentication share one logical request budget
# ---------------------------------------------------------------------------


def test_redirects_and_retries_share_the_attempt_budget() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.Response(503)])
    retry = make_retry(clock, max_attempts=3, backoff_factor=0.1)

    with httpx.Client(
        transport=transport, retry=retry, follow_redirects=True
    ) as client:
        response = client.get("http://example.org/start")

    # Three attempts in total, rather than three per redirect hop.
    assert response.status_code == 503
    assert len(transport.requests) == 3
    assert clock.sleeps == [0.1, 0.2]


def test_redirect_then_retryable_failure_shares_budget() -> None:
    clock = FakeClock()
    events: list[Event] = [
        httpx.Response(302, headers={"Location": "http://example.org/next"}),
        httpx.ConnectError("no"),
        httpx.ConnectError("no"),
        ok_response(),
    ]
    transport = SequenceTransport(events)
    retry = make_retry(clock, max_attempts=4, backoff_factor=0.1)

    with httpx.Client(
        transport=transport, retry=retry, follow_redirects=True
    ) as client:
        response = client.get("http://example.org/start")

    assert response.status_code == 200
    assert [str(request.url) for request in transport.requests] == [
        "http://example.org/start",
        "http://example.org/next",
        "http://example.org/next",
        "http://example.org/next",
    ]


class TwoRoundAuth(httpx.Auth):
    def auth_flow(self, request: httpx.Request) -> typing.Generator[
        httpx.Request, httpx.Response, None
    ]:
        yield request
        request.headers["X-Auth-Round"] = "2"
        yield request


def test_authentication_flow_shares_the_attempt_budget() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no")])
    retry = make_retry(clock, max_attempts=4, backoff_factor=0.1)

    with httpx.Client(
        transport=transport, auth=TwoRoundAuth(), retry=retry
    ) as client:
        with pytest.raises(httpx.RetryExhausted):
            client.get("http://example.org/")

    assert len(transport.requests) == 4


@pytest.mark.anyio
async def test_authentication_flow_shares_the_attempt_budget_async() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no")])
    retry = make_retry(clock, max_attempts=4, backoff_factor=0.1)

    async with httpx.AsyncClient(
        transport=transport, auth=TwoRoundAuth(), retry=retry
    ) as client:
        with pytest.raises(httpx.RetryExhausted):
            await client.get("http://example.org/")

    assert len(transport.requests) == 4


def test_redirects_remain_subject_to_max_redirects() -> None:
    clock = FakeClock()
    transport = SequenceTransport(
        [httpx.Response(302, headers={"Location": "http://example.org/loop"})]
    )
    retry = make_retry(clock, max_attempts=10)

    with httpx.Client(
        transport=transport,
        retry=retry,
        follow_redirects=True,
        max_redirects=2,
    ) as client:
        with pytest.raises(httpx.TooManyRedirects):
            client.get("http://example.org/loop")


def test_idempotent_opt_in_survives_request_construction() -> None:
    clock = FakeClock()
    transport = SequenceTransport([httpx.ConnectError("no"), ok_response()])
    retry = make_retry(clock, max_attempts=2, backoff_factor=0.1)

    with httpx.Client(transport=transport, retry=retry) as client:
        request = client.build_request(
            "POST", "http://example.org/", content=b"data", idempotent=True
        )
        response = client.send(request)

    assert response.status_code == 200
    assert request.extensions["idempotent"] is True
    assert len(transport.requests) == 2
