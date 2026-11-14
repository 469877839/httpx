import asyncio
import typing

import pytest

import httpx


class FakeClock:
    """
    A deterministic clock. Sleeps advance the clock instantly,
    so tests can assert exactly how much of the deadline was consumed.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay

    async def async_sleep(self, delay: float) -> None:
        self.sleep(delay)


def make_policy(clock: FakeClock, **kwargs: typing.Any) -> httpx.RetryPolicy:
    return httpx.RetryPolicy(
        clock=clock.monotonic,
        sleep=clock.sleep,
        async_sleep=clock.async_sleep,
        **kwargs,
    )


class SequenceHandler:
    """
    A MockTransport-compatible handler playing a scripted sequence of
    responses and exceptions, recording every request actually sent.
    """

    def __init__(self, script: list[typing.Union[httpx.Response, Exception]]):
        self.script = list(script)
        self.requests: list[httpx.Request] = []
        self.responses: list[httpx.Response] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        self.responses.append(item)
        return item


def test_retry_policy_validation():
    with pytest.raises(ValueError):
        httpx.RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        httpx.RetryPolicy(deadline=0)
    with pytest.raises(ValueError):
        httpx.RetryPolicy(backoff=-1)
    with pytest.raises(ValueError):
        httpx.RetryPolicy(max_backoff=-1)


def test_retry_on_connect_error():
    clock = FakeClock()
    handler = SequenceHandler(
        [
            httpx.ConnectError("boom"),
            httpx.ConnectError("boom"),
            httpx.Response(200, content=b"ok"),
        ]
    )
    policy = make_policy(clock, backoff=0.5)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert len(handler.requests) == 3
    assert clock.sleeps == [0.5, 1.0]
    assert clock.now == 1.5


def test_retry_on_connect_error_disabled():
    handler = SequenceHandler([httpx.ConnectError("boom"), httpx.Response(200)])
    policy = make_policy(FakeClock(), retry_on_connect_error=False)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        with pytest.raises(httpx.ConnectError):
            client.get("http://example.org/")

    assert len(handler.requests) == 1


def test_retry_on_read_error():
    handler = SequenceHandler([httpx.ReadError("boom"), httpx.Response(200)])
    policy = make_policy(FakeClock())
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert len(handler.requests) == 2


def test_retry_on_read_error_disabled():
    handler = SequenceHandler([httpx.ReadError("boom"), httpx.Response(200)])
    policy = make_policy(FakeClock(), retry_on_read_error=False)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        with pytest.raises(httpx.ReadError):
            client.get("http://example.org/")

    assert len(handler.requests) == 1


def test_retry_on_status_code_closes_responses():
    clock = FakeClock()
    handler = SequenceHandler(
        [httpx.Response(503), httpx.Response(502), httpx.Response(200)]
    )
    policy = make_policy(clock, backoff=0.5)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert len(handler.requests) == 3
    # The response streams of retried responses must have been closed
    # before the next attempt.
    assert handler.responses[0].is_closed
    assert handler.responses[1].is_closed
    assert clock.sleeps == [0.5, 1.0]


def test_retry_status_codes_are_configurable():
    handler = SequenceHandler([httpx.Response(503), httpx.Response(200)])
    policy = make_policy(FakeClock(), retry_status_codes=[500])
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 503
    assert len(handler.requests) == 1


def test_retry_after_header_is_respected():
    clock = FakeClock()
    handler = SequenceHandler(
        [
            httpx.Response(429, headers={"Retry-After": "5"}),
            httpx.Response(200),
        ]
    )
    policy = make_policy(clock, backoff=0.5)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert clock.sleeps == [5.0]


def test_retry_after_header_can_be_ignored():
    clock = FakeClock()
    handler = SequenceHandler(
        [
            httpx.Response(429, headers={"Retry-After": "5"}),
            httpx.Response(200),
        ]
    )
    policy = make_policy(clock, backoff=0.5, respect_retry_after=False)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert clock.sleeps == [0.5]


def test_max_attempts_returns_last_response():
    clock = FakeClock()
    handler = SequenceHandler([httpx.Response(503) for _ in range(5)])
    policy = make_policy(clock, max_attempts=3, backoff=0.5)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 503
    assert len(handler.requests) == 3
    assert clock.sleeps == [0.5, 1.0]


def test_max_attempts_reraises_last_transport_error():
    clock = FakeClock()
    handler = SequenceHandler([httpx.ConnectError("boom") for _ in range(5)])
    policy = make_policy(clock, max_attempts=3, backoff=0.5)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        with pytest.raises(httpx.ConnectError):
            client.get("http://example.org/")

    assert len(handler.requests) == 3
    assert clock.sleeps == [0.5, 1.0]


def test_deadline_stops_retries_on_status_codes():
    clock = FakeClock()
    handler = SequenceHandler([httpx.Response(503) for _ in range(5)])
    policy = make_policy(clock, max_attempts=10, deadline=1.0, backoff=0.6)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.get("http://example.org/")

    # The first retry sleeps 0.6s. The next backoff of 1.2s would exceed
    # the remaining 0.4s of the deadline, so the last response is returned.
    assert response.status_code == 503
    assert len(handler.requests) == 2
    assert 0.0 < clock.now <= 1.0


def test_deadline_stops_retries_on_transport_errors():
    clock = FakeClock()
    handler = SequenceHandler([httpx.ConnectError("boom") for _ in range(5)])
    policy = make_policy(clock, max_attempts=10, deadline=1.0, backoff=0.6)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        with pytest.raises(httpx.ConnectError):
            client.get("http://example.org/")

    assert len(handler.requests) == 2
    assert 0.0 < clock.now <= 1.0


def test_post_is_not_retried_by_default():
    clock = FakeClock()
    handler = SequenceHandler([httpx.Response(503), httpx.Response(200)])
    policy = make_policy(clock)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.post("http://example.org/", content=b"data")

    assert response.status_code == 503
    assert len(handler.requests) == 1
    assert clock.sleeps == []


def test_post_transport_error_is_not_retried_by_default():
    handler = SequenceHandler([httpx.ConnectError("boom"), httpx.Response(200)])
    policy = make_policy(FakeClock())
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        with pytest.raises(httpx.ConnectError):
            client.post("http://example.org/", content=b"data")

    assert len(handler.requests) == 1


def test_post_retried_with_explicit_idempotency_permit():
    clock = FakeClock()
    handler = SequenceHandler([httpx.Response(503), httpx.Response(200)])
    policy = make_policy(clock)
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.post(
            "http://example.org/",
            content=b"data",
            extensions={"idempotent": True},
        )

    assert response.status_code == 200
    assert len(handler.requests) == 2
    assert handler.requests[0].content == b"data"
    assert handler.requests[1].content == b"data"


def test_retry_methods_are_configurable():
    handler = SequenceHandler([httpx.Response(503), httpx.Response(200)])
    policy = make_policy(FakeClock(), retry_methods=["GET", "POST"])
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.post("http://example.org/", content=b"data")

    assert response.status_code == 200
    assert len(handler.requests) == 2


class ConnectErrorTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        raise httpx.ConnectError("boom", request=request)


class StatusTransport(httpx.BaseTransport):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(self.status_code)


def test_non_replayable_body_refused_on_transport_error():
    transport = ConnectErrorTransport()
    policy = make_policy(FakeClock())
    with httpx.Client(transport=transport, retry=policy) as client:
        with pytest.raises(httpx.RequestNotReplayable):
            client.post(
                "http://example.org/",
                content=iter([b"streaming", b"body"]),
                extensions={"idempotent": True},
            )

    assert transport.calls == 1


def test_non_replayable_body_refused_on_status_code():
    transport = StatusTransport(503)
    policy = make_policy(FakeClock())
    with httpx.Client(transport=transport, retry=policy) as client:
        with pytest.raises(httpx.RequestNotReplayable):
            client.post(
                "http://example.org/",
                content=iter([b"streaming", b"body"]),
                extensions={"idempotent": True},
            )

    assert transport.calls == 1


def test_redirects_share_the_attempt_budget():
    requests = []

    def redirect_loop(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(303, headers={"Location": "/loop"})

    policy = make_policy(FakeClock(), max_attempts=3)
    with httpx.Client(
        transport=httpx.MockTransport(redirect_loop),
        retry=policy,
        follow_redirects=True,
    ) as client:
        with pytest.raises(httpx.RetryBudgetExceeded):
            client.get("http://example.org/")

    # Redirect hops do not restart the attempt count.
    assert len(requests) == 3


class ResendOn401Auth(httpx.Auth):
    """
    A toy auth flow that re-sends the request once with credentials
    after a 401 response.
    """

    def auth_flow(
        self, request: httpx.Request
    ) -> typing.Generator[httpx.Request, httpx.Response, None]:
        response = yield request
        if response.status_code == 401:
            request.headers["Authorization"] = "Bearer token"
            yield request


def test_auth_flow_shares_the_attempt_budget():
    handler = SequenceHandler(
        [httpx.Response(401), httpx.Response(503), httpx.Response(200)]
    )
    policy = make_policy(FakeClock(), max_attempts=2)
    with httpx.Client(
        transport=httpx.MockTransport(handler), retry=policy, auth=ResendOn401Auth()
    ) as client:
        response = client.get("http://example.org/")

    # The 401 authentication re-send consumed the second attempt, so no
    # budget remains to retry the 503, and it is returned to the caller.
    assert response.status_code == 503
    assert len(handler.requests) == 2


def test_auth_flow_then_retry_within_budget():
    handler = SequenceHandler(
        [httpx.Response(401), httpx.Response(503), httpx.Response(200)]
    )
    policy = make_policy(FakeClock(), max_attempts=3)
    with httpx.Client(
        transport=httpx.MockTransport(handler), retry=policy, auth=ResendOn401Auth()
    ) as client:
        response = client.get("http://example.org/")

    assert response.status_code == 200
    assert len(handler.requests) == 3


def test_event_hook_exceptions_are_not_swallowed():
    def raise_on_response(response: httpx.Response) -> None:
        raise RuntimeError("hook failed")

    handler = SequenceHandler([httpx.Response(503), httpx.Response(200)])
    policy = make_policy(FakeClock())
    with httpx.Client(
        transport=httpx.MockTransport(handler),
        retry=policy,
        event_hooks={"response": [raise_on_response]},
    ) as client:
        with pytest.raises(RuntimeError, match="hook failed"):
            client.get("http://example.org/")

    assert len(handler.requests) == 1


def test_cancellation_is_not_swallowed():
    def cancelled(request: httpx.Request) -> httpx.Response:
        raise KeyboardInterrupt

    policy = make_policy(FakeClock())
    with httpx.Client(
        transport=httpx.MockTransport(cancelled), retry=policy
    ) as client:
        with pytest.raises(KeyboardInterrupt):
            client.get("http://example.org/")


def test_per_request_retry_override():
    clock = FakeClock()
    handler = SequenceHandler([httpx.Response(503), httpx.Response(200)])
    # No client-level retry policy; enabled just for this request.
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = client.get(
            "http://example.org/", extensions={"retry": make_policy(clock)}
        )

    assert response.status_code == 200
    assert len(handler.requests) == 2


def test_per_request_retry_disabled():
    handler = SequenceHandler([httpx.Response(503), httpx.Response(200)])
    policy = make_policy(FakeClock())
    with httpx.Client(transport=httpx.MockTransport(handler), retry=policy) as client:
        response = client.get("http://example.org/", extensions={"retry": None})

    assert response.status_code == 503
    assert len(handler.requests) == 1


@pytest.mark.anyio
async def test_async_retry_on_connect_error():
    clock = FakeClock()
    handler = SequenceHandler(
        [
            httpx.ConnectError("boom"),
            httpx.ConnectError("boom"),
            httpx.Response(200),
        ]
    )
    policy = make_policy(clock, backoff=0.5)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), retry=policy
    ) as client:
        response = await client.get("http://example.org/")

    assert response.status_code == 200
    assert len(handler.requests) == 3
    assert clock.sleeps == [0.5, 1.0]
    assert clock.now == 1.5


@pytest.mark.anyio
async def test_async_retry_on_status_code_closes_responses():
    clock = FakeClock()
    handler = SequenceHandler([httpx.Response(503), httpx.Response(200)])
    policy = make_policy(clock)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), retry=policy
    ) as client:
        response = await client.get("http://example.org/")

    assert response.status_code == 200
    assert len(handler.requests) == 2
    assert handler.responses[0].is_closed


@pytest.mark.anyio
async def test_async_deadline_stops_retries():
    clock = FakeClock()
    handler = SequenceHandler([httpx.Response(503) for _ in range(5)])
    policy = make_policy(clock, max_attempts=10, deadline=1.0, backoff=0.6)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), retry=policy
    ) as client:
        response = await client.get("http://example.org/")

    assert response.status_code == 503
    assert len(handler.requests) == 2
    assert 0.0 < clock.now <= 1.0


@pytest.mark.anyio
async def test_async_post_retried_with_idempotency_permit():
    handler = SequenceHandler([httpx.Response(503), httpx.Response(200)])
    policy = make_policy(FakeClock())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), retry=policy
    ) as client:
        response = await client.post(
            "http://example.org/",
            content=b"data",
            extensions={"idempotent": True},
        )

    assert response.status_code == 200
    assert len(handler.requests) == 2


@pytest.mark.anyio
async def test_async_cancellation_is_not_swallowed():
    def cancelled(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    policy = make_policy(FakeClock())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(cancelled), retry=policy
    ) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.get("http://example.org/")
