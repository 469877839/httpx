from __future__ import annotations

import threading
import typing

import pytest

import httpx

#
# Mock transport
#
# `/login` issues a `Set-Cookie: session=...` header whose value depends on
# the host, then 302-redirects cross-origin to `target.org/target`.
# `/target` and `/echo_cookies` simply report the `Cookie` header they see.
#

SEEN_COOKIE_HEADERS: list[tuple[str, str | None]] = []
SEEN_PARTITIONS: list[tuple[str, str | None]] = []


def partitioned_app(request: httpx.Request) -> httpx.Response:
    partition = getattr(request, "cookie_partition", None)
    SEEN_PARTITIONS.append((str(request.url), partition))
    SEEN_COOKIE_HEADERS.append((str(request.url), request.headers.get("cookie")))

    if request.url.path == "/login":
        host = request.url.host
        headers = {
            "set-cookie": f"session=token-for-{host}; Path=/",
            "location": "http://target.org/target",
        }
        return httpx.Response(302, headers=headers)

    if request.url.path == "/target":
        headers = {"set-cookie": "target=token-for-target.org; Path=/"}
        return httpx.Response(
            200, headers=headers, json={"cookie": request.headers.get("cookie")}
        )

    if request.url.path == "/echo_cookies":
        return httpx.Response(200, json={"cookie": request.headers.get("cookie")})

    if request.url.path == "/set_cookie":
        return httpx.Response(200, headers={"set-cookie": "name=value; Path=/"})

    raise NotImplementedError(request.url.path)  # pragma: no cover


@pytest.fixture(autouse=True)
def _reset_records() -> typing.Iterator[None]:
    SEEN_COOKIE_HEADERS.clear()
    SEEN_PARTITIONS.clear()
    yield
    SEEN_COOKIE_HEADERS.clear()
    SEEN_PARTITIONS.clear()



# ---------------------------------------------------------------------------
# Two partitions hitting the same domain and path must not share cookies
# ---------------------------------------------------------------------------


def test_partitions_isolate_cookies_for_same_domain_and_path() -> None:
    client = httpx.Client(transport=httpx.MockTransport(partitioned_app))

    client.cookies.set("session", "alpha", domain="example.org", partition_key="task-a")
    client.cookies.set("session", "beta", domain="example.org", partition_key="task-b")

    response_a = client.get(
        "http://example.org/echo_cookies", cookie_partition="task-a"
    )
    response_b = client.get(
        "http://example.org/echo_cookies", cookie_partition="task-b"
    )

    assert response_a.json() == {"cookie": "session=alpha"}
    assert response_b.json() == {"cookie": "session=beta"}

    # The default, unpartitioned jar stays empty.
    assert "session" not in client.cookies
    assert client.cookies.get("session") is None

    # Set-Cookie responses are attributed to the requesting partition.
    client.get("http://example.org/set_cookie", cookie_partition="task-a")
    assert client.cookies.get("name", partition_key="task-a") == "value"
    assert client.cookies.get("name", partition_key="task-b") is None
    assert client.cookies.get("name") is None


@pytest.mark.anyio
async def test_async_partitions_isolate_cookies_for_same_domain_and_path() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(partitioned_app)
    ) as client:
        client.cookies.set(
            "session", "alpha", domain="example.org", partition_key="task-a"
        )
        client.cookies.set(
            "session", "beta", domain="example.org", partition_key="task-b"
        )

        response_a = await client.get(
            "http://example.org/echo_cookies", cookie_partition="task-a"
        )
        response_b = await client.get(
            "http://example.org/echo_cookies", cookie_partition="task-b"
        )

        assert response_a.json() == {"cookie": "session=alpha"}
        assert response_b.json() == {"cookie": "session=beta"}

        await client.get(
            "http://example.org/set_cookie", cookie_partition="task-a"
        )
        assert client.cookies.get("name", partition_key="task-a") == "value"
        assert client.cookies.get("name", partition_key="task-b") is None


# ---------------------------------------------------------------------------
# The partition key survives a cross-origin redirect
# ---------------------------------------------------------------------------


def test_partition_follows_cross_origin_redirect_sync() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(partitioned_app), follow_redirects=True
    )

    response = client.get("http://example.org/login", cookie_partition="task-a")
    assert response.url == "http://target.org/target"
    # The example.org cookie must not leak cross-origin onto target.org.
    assert response.json() == {"cookie": None}

    # Every request on the way carried the partition key.
    assert all(partition == "task-a" for _, partition in SEEN_PARTITIONS)

    # Both the redirecting origin's cookie and the target origin's cookie
    # are attributed to the partition, scoped to their own domains.
    assert (
        client.cookies.get("session", domain="example.org", partition_key="task-a")
        == "token-for-example.org"
    )
    assert (
        client.cookies.get("target", domain="target.org", partition_key="task-a")
        == "token-for-target.org"
    )
    assert client.cookies.get("session") is None
    assert client.cookies.get("target") is None

    # Same-domain requests reuse the partition cookie; the other
    # partition and the default jar stay empty.
    response = client.get(
        "http://example.org/echo_cookies", cookie_partition="task-a"
    )
    assert response.json() == {"cookie": "session=token-for-example.org"}
    response = client.get("http://target.org/target", cookie_partition="task-a")
    assert response.json() == {"cookie": "target=token-for-target.org"}

    response = client.get(
        "http://example.org/echo_cookies", cookie_partition="task-b"
    )
    assert response.json() == {"cookie": None}
    assert client.cookies.partition_keys() == ["task-a"]


@pytest.mark.anyio
async def test_partition_follows_cross_origin_redirect_async() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(partitioned_app), follow_redirects=True
    ) as client:
        response = await client.get(
            "http://example.org/login", cookie_partition="task-a"
        )
        assert response.url == "http://target.org/target"
        # The example.org cookie must not leak cross-origin onto target.org.
        assert response.json() == {"cookie": None}

        assert all(partition == "task-a" for _, partition in SEEN_PARTITIONS)
        assert (
            client.cookies.get("session", domain="example.org", partition_key="task-a")
            == "token-for-example.org"
        )
        assert (
            client.cookies.get("target", domain="target.org", partition_key="task-a")
            == "token-for-target.org"
        )
        assert client.cookies.get("session") is None
        assert client.cookies.get("target") is None

        response = await client.get(
            "http://example.org/echo_cookies", cookie_partition="task-a"
        )
        assert response.json() == {"cookie": "session=token-for-example.org"}
        response = await client.get(
            "http://example.org/echo_cookies", cookie_partition="task-b"
        )
        assert response.json() == {"cookie": None}


# ---------------------------------------------------------------------------
# Import / export / clear of a named partition
# ---------------------------------------------------------------------------


def test_partition_import_export_and_clear() -> None:
    client = httpx.Client(transport=httpx.MockTransport(partitioned_app))
    client.cookies.set("session", "alpha", domain="example.org", partition_key="task-a")
    client.cookies.set("other", "keep", domain="example.org", partition_key="task-b")

    exported = client.cookies.export_partition("task-a")
    assert exported.get("session") == "alpha"
    assert "other" not in exported

    fresh = httpx.Client(transport=httpx.MockTransport(partitioned_app))
    fresh.cookies.import_partition("restored", exported)
    assert fresh.cookies.get("session", partition_key="restored") == "alpha"
    assert "restored" in fresh.cookies.partition_keys()

    fresh.cookies.clear_partition("restored")
    assert "restored" not in fresh.cookies.partition_keys()
    assert fresh.cookies.get("session", partition_key="restored") is None

    # Clearing one partition leaves the other untouched.
    client.cookies.clear_partition("task-a")
    assert "task-a" not in client.cookies.partition_keys()
    assert client.cookies.get("other", partition_key="task-b") == "keep"


# ---------------------------------------------------------------------------
# No partition key -> existing global jar behaviour
# ---------------------------------------------------------------------------


def test_unpartitioned_requests_keep_global_jar_behaviour() -> None:
    client = httpx.Client(transport=httpx.MockTransport(partitioned_app))

    client.get("http://example.org/set_cookie")
    assert client.cookies["name"] == "value"

    response = client.get("http://example.org/echo_cookies")
    assert response.json() == {"cookie": "name=value"}

    assert all(partition is None for _, partition in SEEN_PARTITIONS)


# ---------------------------------------------------------------------------
# Domain / path / same-name matching rules still apply inside a partition
# ---------------------------------------------------------------------------


def test_partition_respects_domain_path_and_same_name_rules() -> None:
    cookies = httpx.Cookies()

    cookies.set("dup", "root", domain="example.org", path="/", partition_key="p")
    cookies.set("dup", "sub", domain="sub.example.org", path="/", partition_key="p")
    cookies.set("scoped", "deep", domain="example.org", path="/deep", partition_key="p")

    partition = cookies.partition("p")
    assert partition.get("dup", domain="example.org") == "root"
    assert partition.get("dup", domain="sub.example.org") == "sub"
    assert partition.get("scoped", path="/deep") == "deep"
    assert partition.get("scoped", path="/") is None

    with pytest.raises(httpx.CookieConflict):
        partition.get("dup")

    partition.delete("dup", domain="sub.example.org", path="/")
    assert partition.get("dup", domain="sub.example.org") is None
    assert partition.get("dup", domain="example.org") == "root"


# ---------------------------------------------------------------------------
# Partition key propagates through auth-generated requests
# ---------------------------------------------------------------------------


class _RecordingAuth(httpx.Auth):
    def __init__(self) -> None:
        self.seen_partitions: list[str | None] = []

    def auth_flow(
        self, request: httpx.Request
    ) -> typing.Generator[httpx.Request, httpx.Response, None]:
        self.seen_partitions.append(request.cookie_partition)
        # Emit a brand new request object, as DigestAuth does on a 401.
        retried = httpx.Request(request.method, request.url)
        retried.headers["Authorization"] = "Bearer token"
        yield retried


def test_partition_propagates_through_auth_flow() -> None:
    auth = _RecordingAuth()
    client = httpx.Client(transport=httpx.MockTransport(partitioned_app), auth=auth)

    client.get("http://example.org/echo_cookies", cookie_partition="task-a")
    assert auth.seen_partitions == ["task-a"]
    assert all(partition == "task-a" for _, partition in SEEN_PARTITIONS)


# ---------------------------------------------------------------------------
# Concurrent partitioned requests never swap client.cookies
# ---------------------------------------------------------------------------


def test_concurrent_partitioned_requests_are_isolated() -> None:
    client = httpx.Client(transport=httpx.MockTransport(partitioned_app))

    for partition, value in (("task-a", "alpha"), ("task-b", "beta")):
        client.cookies.set(
            "session", value, domain="example.org", partition_key=partition
        )

    results: dict[str, str | None] = {}
    errors: list[BaseException] = []

    def worker(partition: str) -> None:
        try:
            for _ in range(20):
                response = client.get(
                    "http://example.org/echo_cookies", cookie_partition=partition
                )
                expected = {"task-a": "session=alpha", "task-b": "session=beta"}
                assert response.json() == {"cookie": expected[partition]}
            results[partition] = "ok"
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("task-a",)),
        threading.Thread(target=worker, args=("task-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert results == {"task-a": "ok", "task-b": "ok"}

    # Each observed request carried its own partition key throughout.
    assert {partition for _, partition in SEEN_PARTITIONS} == {"task-a", "task-b"}
    for url, cookie_header in SEEN_COOKIE_HEADERS:
        assert cookie_header in ("session=alpha", "session=beta")
