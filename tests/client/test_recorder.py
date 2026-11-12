"""
Tests for the optional logical request recorder.

Each scenario in this module is an in-memory reproduction that compares the
recorded attempts item-by-item with the calls actually made by the transport.
The sync and async variants produce the same recordings.
"""

import typing

import pytest

import httpx


class LoginThenRedirectAuth(httpx.Auth):
    """
    A minimal authentication flow that logs in, then has the server redirect
    the authenticated request.
    """

    def auth_flow(
        self, request: httpx.Request
    ) -> typing.Generator[httpx.Request, httpx.Response, None]:
        response = yield request
        if response.status_code == 401:
            request.headers["Authorization"] = "Basic dXNlcjpwYXNz"
            yield request


class CallLog:
    """
    A transport handler that records the exact calls it receives.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def record(self, request: httpx.Request) -> None:
        self.calls.append(
            (
                request.method,
                str(request.url),
                request.headers.get("Authorization"),
            )
        )


class FailingSyncStream(httpx.SyncByteStream):
    def __iter__(self) -> typing.Iterator[bytes]:
        yield b"partial"
        raise httpx.ReadError("stream broke before completion")

    def close(self) -> None:
        pass


class FailingAsyncStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        yield b"partial"
        raise httpx.ReadError("stream broke before completion")

    async def aclose(self) -> None:
        pass


class UnreadSyncStream(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.iterated = False

    def __iter__(self) -> typing.Iterator[bytes]:
        self.iterated = True
        yield b"never recorded"

    def close(self) -> None:
        pass


def _attempt(attempt: httpx.RequestAttempt) -> dict[str, typing.Any]:
    """
    The recording fields relevant to a transport call log, normalized so that
    sync and async recordings compare directly and timings do not interfere.
    """
    return {
        "trigger": attempt.trigger,
        "auth_step": attempt.auth_step,
        "redirect_step": attempt.redirect_step,
        "method": attempt.method,
        "url": attempt.url,
        "previous_method": attempt.previous_method,
        "method_changed": attempt.method_changed,
        "redirect_status": attempt.redirect_status,
        "response_status": attempt.response_status,
        "outcome": attempt.outcome,
        "exception_type": attempt.exception_type,
        "has_timing": attempt.finished is not None
        and attempt.elapsed is not None
        and attempt.elapsed >= 0,
    }


def _auth_step(step: httpx.AuthStep) -> dict[str, typing.Any]:
    return {
        "sequence": step.sequence,
        "outcome": step.outcome,
        "response_status": step.response_status,
        "exception_type": step.exception_type,
    }


def _assert_matches_transport(record: httpx.RequestRecord, log: CallLog) -> None:
    assert len(record.attempts) == len(log.calls)
    for attempt, (method, url, _auth) in zip(record.attempts, log.calls):
        assert attempt.method == method
        assert attempt.url == url


def test_auth_then_redirect_recording() -> None:
    log = CallLog()

    def app(request: httpx.Request) -> httpx.Response:
        log.record(request)
        if request.url.path == "/login":
            if request.headers.get("Authorization") is None:
                return httpx.Response(401, headers={"www-authenticate": "Basic"})
            return httpx.Response(302, headers={"location": "/home"})
        return httpx.Response(200, content=b"welcome")

    recorder = httpx.RequestRecorder()
    with httpx.Client(
        auth=LoginThenRedirectAuth(),
        follow_redirects=True,
        transport=httpx.MockTransport(app),
        recorder=recorder,
    ) as client:
        response = client.post("http://example.org/login", content=b"credentials")

    assert response.status_code == 200
    assert response.history[-1].status_code == 302

    record = recorder.records[0]
    _assert_matches_transport(record, log)

    assert [_attempt(a) for a in record.attempts] == [
        {
            "trigger": httpx.AttemptTrigger.INITIAL,
            "auth_step": 0,
            "redirect_step": 0,
            "method": "POST",
            "url": "http://example.org/login",
            "previous_method": None,
            "method_changed": None,
            "redirect_status": None,
            "response_status": 401,
            "outcome": httpx.AttemptOutcome.RESPONSE,
            "exception_type": None,
            "has_timing": True,
        },
        {
            "trigger": httpx.AttemptTrigger.AUTH_CONTINUATION,
            "auth_step": 1,
            "redirect_step": 0,
            "method": "POST",
            "url": "http://example.org/login",
            "previous_method": "POST",
            "method_changed": False,
            "redirect_status": None,
            "response_status": 302,
            "outcome": httpx.AttemptOutcome.RESPONSE,
            "exception_type": None,
            "has_timing": True,
        },
        {
            "trigger": httpx.AttemptTrigger.REDIRECT,
            "auth_step": 1,
            "redirect_step": 1,
            "method": "GET",
            "url": "http://example.org/home",
            "previous_method": "POST",
            "method_changed": True,
            "redirect_status": 302,
            "response_status": 200,
            "outcome": httpx.AttemptOutcome.RESPONSE,
            "exception_type": None,
            "has_timing": True,
        },
    ]
    assert [_auth_step(s) for s in record.auth_steps] == [
        {
            "sequence": 0,
            "outcome": httpx.AuthStepOutcome.CONTINUED,
            "response_status": 401,
            "exception_type": None,
        },
        {
            "sequence": 1,
            "outcome": httpx.AuthStepOutcome.ENDED,
            "response_status": 200,
            "exception_type": None,
        },
    ]


def test_second_hop_transport_failure_recording() -> None:
    log = CallLog()

    def app(request: httpx.Request) -> httpx.Response:
        log.record(request)
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "/target"})
        raise httpx.ConnectError("second hop failed", request=request)

    recorder = httpx.RequestRecorder()
    with httpx.Client(
        follow_redirects=True,
        transport=httpx.MockTransport(app),
        recorder=recorder,
    ) as client:
        with pytest.raises(httpx.ConnectError):
            client.get("http://example.org/redirect")

    record = recorder.records[0]
    _assert_matches_transport(record, log)

    assert [_attempt(a) for a in record.attempts] == [
        {
            "trigger": httpx.AttemptTrigger.INITIAL,
            "auth_step": 0,
            "redirect_step": 0,
            "method": "GET",
            "url": "http://example.org/redirect",
            "previous_method": None,
            "method_changed": None,
            "redirect_status": None,
            "response_status": 302,
            "outcome": httpx.AttemptOutcome.RESPONSE,
            "exception_type": None,
            "has_timing": True,
        },
        {
            "trigger": httpx.AttemptTrigger.REDIRECT,
            "auth_step": 0,
            "redirect_step": 1,
            "method": "GET",
            "url": "http://example.org/target",
            "previous_method": "GET",
            "method_changed": False,
            "redirect_status": 302,
            "response_status": None,
            "outcome": httpx.AttemptOutcome.TRANSPORT_ERROR,
            "exception_type": "ConnectError",
            "has_timing": True,
        },
    ]
    # The failed attempt never produced a response, so it cannot have auth steps.
    assert record.auth_steps == []


def test_stream_read_failure_recording() -> None:
    log = CallLog()

    def app(request: httpx.Request) -> httpx.Response:
        log.record(request)
        return httpx.Response(200, stream=FailingSyncStream())

    recorder = httpx.RequestRecorder()
    with httpx.Client(transport=httpx.MockTransport(app), recorder=recorder) as client:
        with pytest.raises(httpx.ReadError):
            client.get("http://example.org/data")

    record = recorder.records[0]
    _assert_matches_transport(record, log)

    assert [_attempt(a) for a in record.attempts] == [
        {
            "trigger": httpx.AttemptTrigger.INITIAL,
            "auth_step": 0,
            "redirect_step": 0,
            "method": "GET",
            "url": "http://example.org/data",
            "previous_method": None,
            "method_changed": None,
            "redirect_status": None,
            "response_status": 200,
            "outcome": httpx.AttemptOutcome.READ_ERROR,
            "exception_type": "ReadError",
            "has_timing": True,
        }
    ]


@pytest.mark.anyio
async def test_async_auth_then_redirect_recording_matches_sync() -> None:
    log = CallLog()

    async def app(request: httpx.Request) -> httpx.Response:
        log.record(request)
        if request.url.path == "/login":
            if request.headers.get("Authorization") is None:
                return httpx.Response(401, headers={"www-authenticate": "Basic"})
            return httpx.Response(302, headers={"location": "/home"})
        return httpx.Response(200, content=b"welcome")

    recorder = httpx.RequestRecorder()
    async with httpx.AsyncClient(
        auth=LoginThenRedirectAuth(),
        follow_redirects=True,
        transport=httpx.MockTransport(app),
        recorder=recorder,
    ) as client:
        response = await client.post("http://example.org/login", content=b"credentials")

    assert response.status_code == 200
    record = recorder.records[0]
    _assert_matches_transport(record, log)
    assert [_attempt(a) for a in record.attempts] == [
        {
            "trigger": httpx.AttemptTrigger.INITIAL,
            "auth_step": 0,
            "redirect_step": 0,
            "method": "POST",
            "url": "http://example.org/login",
            "previous_method": None,
            "method_changed": None,
            "redirect_status": None,
            "response_status": 401,
            "outcome": httpx.AttemptOutcome.RESPONSE,
            "exception_type": None,
            "has_timing": True,
        },
        {
            "trigger": httpx.AttemptTrigger.AUTH_CONTINUATION,
            "auth_step": 1,
            "redirect_step": 0,
            "method": "POST",
            "url": "http://example.org/login",
            "previous_method": "POST",
            "method_changed": False,
            "redirect_status": None,
            "response_status": 302,
            "outcome": httpx.AttemptOutcome.RESPONSE,
            "exception_type": None,
            "has_timing": True,
        },
        {
            "trigger": httpx.AttemptTrigger.REDIRECT,
            "auth_step": 1,
            "redirect_step": 1,
            "method": "GET",
            "url": "http://example.org/home",
            "previous_method": "POST",
            "method_changed": True,
            "redirect_status": 302,
            "response_status": 200,
            "outcome": httpx.AttemptOutcome.RESPONSE,
            "exception_type": None,
            "has_timing": True,
        },
    ]
    assert [_auth_step(s) for s in record.auth_steps] == [
        {
            "sequence": 0,
            "outcome": httpx.AuthStepOutcome.CONTINUED,
            "response_status": 401,
            "exception_type": None,
        },
        {
            "sequence": 1,
            "outcome": httpx.AuthStepOutcome.ENDED,
            "response_status": 200,
            "exception_type": None,
        },
    ]


@pytest.mark.anyio
async def test_async_second_hop_transport_failure_matches_sync() -> None:
    log = CallLog()

    async def app(request: httpx.Request) -> httpx.Response:
        log.record(request)
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "/target"})
        raise httpx.ConnectError("second hop failed", request=request)

    recorder = httpx.RequestRecorder()
    async with httpx.AsyncClient(
        follow_redirects=True,
        transport=httpx.MockTransport(app),
        recorder=recorder,
    ) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("http://example.org/redirect")

    record = recorder.records[0]
    _assert_matches_transport(record, log)
    assert [_attempt(a) for a in record.attempts] == [
        {
            "trigger": httpx.AttemptTrigger.INITIAL,
            "auth_step": 0,
            "redirect_step": 0,
            "method": "GET",
            "url": "http://example.org/redirect",
            "previous_method": None,
            "method_changed": None,
            "redirect_status": None,
            "response_status": 302,
            "outcome": httpx.AttemptOutcome.RESPONSE,
            "exception_type": None,
            "has_timing": True,
        },
        {
            "trigger": httpx.AttemptTrigger.REDIRECT,
            "auth_step": 0,
            "redirect_step": 1,
            "method": "GET",
            "url": "http://example.org/target",
            "previous_method": "GET",
            "method_changed": False,
            "redirect_status": 302,
            "response_status": None,
            "outcome": httpx.AttemptOutcome.TRANSPORT_ERROR,
            "exception_type": "ConnectError",
            "has_timing": True,
        },
    ]
    assert record.auth_steps == []


@pytest.mark.anyio
async def test_async_stream_read_failure_matches_sync() -> None:
    log = CallLog()

    async def app(request: httpx.Request) -> httpx.Response:
        log.record(request)
        return httpx.Response(200, stream=FailingAsyncStream())

    recorder = httpx.RequestRecorder()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(app), recorder=recorder
    ) as client:
        with pytest.raises(httpx.ReadError):
            await client.get("http://example.org/data")

    record = recorder.records[0]
    _assert_matches_transport(record, log)
    assert [_attempt(a) for a in record.attempts] == [
        {
            "trigger": httpx.AttemptTrigger.INITIAL,
            "auth_step": 0,
            "redirect_step": 0,
            "method": "GET",
            "url": "http://example.org/data",
            "previous_method": None,
            "method_changed": None,
            "redirect_status": None,
            "response_status": 200,
            "outcome": httpx.AttemptOutcome.READ_ERROR,
            "exception_type": "ReadError",
            "has_timing": True,
        }
    ]


def test_response_hook_error_marks_hook_boundary_and_keeps_hook_order() -> None:
    calls = []

    def first_hook(response: httpx.Response) -> None:
        calls.append("first")

    def failing_hook(response: httpx.Response) -> None:
        calls.append("failing")
        raise RuntimeError("hook failed")

    def last_hook(response: httpx.Response) -> None:
        calls.append("last")

    recorder = httpx.RequestRecorder()
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        event_hooks={"response": [first_hook, failing_hook, last_hook]},
        recorder=recorder,
    ) as client:
        with pytest.raises(RuntimeError):
            client.get("http://example.org/")

    assert calls == ["first", "failing"]
    attempt = recorder.records[0].attempts[0]
    assert attempt.outcome is httpx.AttemptOutcome.HOOK_ERROR
    assert attempt.response_status == 200
    assert attempt.exception_type == "RuntimeError"
    assert attempt.exception_message == "hook failed"


def test_request_hook_error_does_not_record_unstarted_network_step() -> None:
    def failing_hook(request: httpx.Request) -> None:
        raise RuntimeError("request hook failed")

    recorder = httpx.RequestRecorder()
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        event_hooks={"request": [failing_hook]},
        recorder=recorder,
    ) as client:
        with pytest.raises(RuntimeError):
            client.get("http://example.org/")

    # The hook runs before the transport, so no attempt should be recorded.
    assert recorder.records[0].attempts == []


def test_auth_generator_exit_records_exception_boundary_without_extra_attempt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    class FailingAuth(httpx.Auth):
        def auth_flow(
            self, request: httpx.Request
        ) -> typing.Generator[httpx.Request, httpx.Response, None]:
            yield request
            raise RuntimeError("auth flow failed")

    recorder = httpx.RequestRecorder()
    with httpx.Client(
        auth=FailingAuth(),
        transport=httpx.MockTransport(handler),
        recorder=recorder,
    ) as client:
        with pytest.raises(RuntimeError):
            client.get("http://example.org/")

    record = recorder.records[0]
    assert len(record.attempts) == 1
    assert record.attempts[0].outcome is httpx.AttemptOutcome.RESPONSE
    assert [_auth_step(s) for s in record.auth_steps] == [
        {
            "sequence": 0,
            "outcome": httpx.AuthStepOutcome.EXCEPTION,
            "response_status": 200,
            "exception_type": "RuntimeError",
        }
    ]


def test_auth_flow_failing_before_first_request_records_no_attempt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the transport must never be called")

    class ImmediateFailureAuth(httpx.Auth):
        def sync_auth_flow(
            self, request: httpx.Request
        ) -> typing.Generator[httpx.Request, httpx.Response, None]:
            raise RuntimeError("auth setup failed")
            yield  # pragma: no cover

    recorder = httpx.RequestRecorder()
    with httpx.Client(
        auth=ImmediateFailureAuth(),
        transport=httpx.MockTransport(handler),
        recorder=recorder,
    ) as client:
        with pytest.raises(RuntimeError):
            client.get("http://example.org/")

    record = recorder.records[0]
    assert record.attempts == []
    assert [_auth_step(s) for s in record.auth_steps] == [
        {
            "sequence": 0,
            "outcome": httpx.AuthStepOutcome.EXCEPTION,
            "response_status": None,
            "exception_type": "RuntimeError",
        }
    ]


@pytest.mark.anyio
async def test_async_auth_flow_failing_before_first_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the transport must never be called")

    class ImmediateFailureAuth(httpx.Auth):
        async def async_auth_flow(
            self, request: httpx.Request
        ) -> typing.AsyncGenerator[httpx.Request, httpx.Response]:
            raise RuntimeError("auth setup failed")
            yield  # pragma: no cover

    recorder = httpx.RequestRecorder()
    async with httpx.AsyncClient(
        auth=ImmediateFailureAuth(),
        transport=httpx.MockTransport(handler),
        recorder=recorder,
    ) as client:
        with pytest.raises(RuntimeError):
            await client.get("http://example.org/")

    record = recorder.records[0]
    assert record.attempts == []
    assert [_auth_step(s) for s in record.auth_steps] == [
        {
            "sequence": 0,
            "outcome": httpx.AuthStepOutcome.EXCEPTION,
            "response_status": None,
            "exception_type": "RuntimeError",
        }
    ]


def test_redirected_request_hook_error_records_no_unstarted_attempt() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "/target"})
        return httpx.Response(200)

    def failing_hook(request: httpx.Request) -> None:
        if request.url.path == "/target":
            raise RuntimeError("redirect hook failed")

    recorder = httpx.RequestRecorder()
    with httpx.Client(
        follow_redirects=True,
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [failing_hook]},
        recorder=recorder,
    ) as client:
        with pytest.raises(RuntimeError):
            client.get("http://example.org/redirect")

    assert calls == ["/redirect"]
    assert len(recorder.records[0].attempts) == 1
    assert recorder.records[0].attempts[0].response_status == 302


def test_headers_and_content_are_opt_in_and_sensitive_values_are_redacted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Set-Cookie": "session=secret", "X-Trace": "visible"},
            content=b"response body",
        )

    recorder = httpx.RequestRecorder(capture_headers=True, capture_content=True)
    with httpx.Client(
        auth=("user", "pass"),
        transport=httpx.MockTransport(handler),
        recorder=recorder,
    ) as client:
        client.get(
            "http://user:password@example.org/path",
            headers={"Cookie": "session=secret", "X-Trace": "visible"},
        )

    attempt = recorder.records[0].attempts[0]
    assert attempt.url == "http://example.org/path"
    assert attempt.request_headers is not None
    assert attempt.response_headers is not None
    assert ("authorization", "[redacted]") in attempt.request_headers
    assert ("cookie", "[redacted]") in attempt.request_headers
    assert ("x-trace", "visible") in attempt.request_headers
    assert ("set-cookie", "[redacted]") in attempt.response_headers
    assert ("x-trace", "visible") in attempt.response_headers
    assert attempt.response_content == b"response body"


def test_default_recording_omits_headers_and_bodies() -> None:
    recorder = httpx.RequestRecorder()
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"body")
        ),
        recorder=recorder,
    ) as client:
        client.get("http://example.org/")

    attempt = recorder.records[0].attempts[0]
    assert attempt.request_headers is None
    assert attempt.request_content is None
    assert attempt.response_headers is None
    assert attempt.response_content is None


def test_recorder_does_not_read_unconsumed_response_stream() -> None:
    stream = UnreadSyncStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    recorder = httpx.RequestRecorder(capture_content=True)
    with httpx.Client(
        transport=httpx.MockTransport(handler), recorder=recorder
    ) as client:
        request = client.build_request("GET", "http://example.org/")
        response = client.send(request, stream=True)
        response.close()

    assert stream.iterated is False
    assert recorder.records[0].attempts[0].response_content is None


def test_invalid_recorder_argument_is_rejected() -> None:
    with pytest.raises(TypeError):
        httpx.Client(recorder=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        httpx.AsyncClient(recorder=object())  # type: ignore[arg-type]


def test_multiple_logical_requests_are_recorded_separately() -> None:
    recorder = httpx.RequestRecorder()
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        recorder=recorder,
    ) as client:
        client.get("http://example.org/one")
        client.get("http://example.org/two")

    assert len(recorder.records) == 2
    assert recorder.records[0].attempts[0].url == "http://example.org/one"
    assert recorder.records[1].attempts[0].url == "http://example.org/two"

    recorder.clear()
    assert recorder.records == []


def test_as_dict_converts_enums_to_values() -> None:
    recorder = httpx.RequestRecorder()
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        recorder=recorder,
    ) as client:
        client.get("http://example.org/")

    attempt = recorder.records[0].attempts[0]
    assert attempt.as_dict()["trigger"] == "initial"
    assert attempt.as_dict()["outcome"] == "response"
