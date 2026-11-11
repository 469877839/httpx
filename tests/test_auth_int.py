"""
Tests for HTTP Digest authentication with `qop=auth-int`.

The mock server recomputes every digest from scratch, using only the
credentials and the RFC 2617/7616 formulas, so the client's `auth-int`
implementation is verified against an independent calculation.
"""

from __future__ import annotations

import gzip
import hashlib
import typing
from urllib.request import parse_keqv_list

import anyio
import pytest

import httpx
from httpx._auth import MAX_IN_MEMORY_BODY_SIZE, _SpooledByteStream
from httpx._types import AsyncByteStream, SyncByteStream

USERNAME = "Mufasa"
PASSWORD = "Circle of Life"
REALM = "http-auth@example.org"
NONCE = "7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v"
NEW_NONCE = "a4b0e4f8a7a5d5ecb5d8d4a2c4d0e2f8"
QOP = b"auth-int"


def digest(algorithm: str, data: bytes) -> bytes:
    hash_name = algorithm.lower().replace("-sess", "")
    return hashlib.new(hash_name, data).hexdigest().encode()


class _ChunkedStream(SyncByteStream, AsyncByteStream):
    """A raw response stream that yields the entity in fixed-size chunks."""

    def __init__(self, data: bytes, chunk_size: int) -> None:
        self._data = data
        self._chunk_size = chunk_size

    def __iter__(self) -> typing.Iterator[bytes]:
        for i in range(0, len(self._data), self._chunk_size):
            yield self._data[i : i + self._chunk_size]

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        for chunk in self:
            yield chunk

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


class DigestAuthIntApp:
    """
    A mock server that validates `auth-int` request digests and optionally
    returns a matching (or deliberately broken) `rspauth`.
    """

    def __init__(
        self,
        *,
        body: bytes = b"response body",
        algorithm: str = "SHA-256",
        qop: str = "auth-int",
        stale_after_attempt: int | None = None,
        stale_every_attempt: bool = False,
        bad_rspauth: bool = False,
        compress_response: bool = False,
        chunk_size: int | None = None,
        pre_read: bool = False,
        auth_info: str | None = "default",
    ) -> None:
        self.body = body
        self.algorithm = algorithm
        self.qop = qop
        self.stale_after_attempt = stale_after_attempt
        self.stale_every_attempt = stale_every_attempt
        self.bad_rspauth = bad_rspauth
        self.compress_response = compress_response
        self.chunk_size = chunk_size
        self.pre_read = pre_read
        # "default" -> a correct rspauth header; None -> no header at all;
        # anything else -> the header is returned verbatim.
        self.auth_info = auth_info
        self.calls = 0
        self.request_entities: list[bytes] = []
        self.received_authorization: list[str] = []

    def _challenge(self, nonce: str, *, stale: bool = False) -> httpx.Response:
        stale_field = ", stale=true" if stale else ""
        return httpx.Response(
            401,
            headers={
                "www-authenticate": (
                    f'Digest realm="{REALM}", qop="{self.qop}", '
                    f'nonce="{nonce}", algorithm={self.algorithm}{stale_field}'
                )
            },
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if "authorization" not in request.headers:
            return self._challenge(NONCE)

        if self.stale_every_attempt:
            return self._challenge(
                NEW_NONCE if self.calls % 2 == 0 else NONCE, stale=True
            )
        if (
            self.stale_after_attempt is not None
            and self.calls == self.stale_after_attempt
        ):
            return self._challenge(NEW_NONCE, stale=True)

        fields = parse_keqv_list(
            request.headers["authorization"].split(", ")
        )
        self.received_authorization.append(request.headers["authorization"])
        self.request_entities.append(request.content)

        nonce = fields["nonce"]
        algorithm = fields["algorithm"]
        qop = fields.get("qop", "")
        entity = request.content

        HA1 = digest(
            algorithm,
            b":".join(
                (USERNAME.encode(), REALM.encode(), PASSWORD.encode())
            ),
        )
        if algorithm.lower().endswith("-sess"):
            HA1 = digest(
                algorithm,
                b":".join(
                    (HA1, nonce.encode(), fields["cnonce"].encode())
                ),
            )

        # A2 = method ":" digest-uri-value [ ":" H(entity-body) ]
        HA2 = digest(
            algorithm,
            b":".join(
                (
                    request.method.encode(),
                    request.url.raw_path,
                    digest(algorithm, entity),
                )
            )
            if qop == "auth-int"
            else b":".join((request.method.encode(), request.url.raw_path)),
        )

        expected = digest(
            algorithm,
            b":".join(
                (
                    HA1,
                    nonce.encode(),
                    fields["nc"].encode(),
                    fields["cnonce"].encode(),
                    qop.encode(),
                    HA2,
                )
            ),
        )
        assert fields["response"] == expected.decode(), (
            "request digest mismatch",
            fields["response"],
            expected,
        )

        # rspauth: same formula, but with an empty method in A2 and the
        # response entity. When the response is content-encoded, the entity
        # is the encoded bytes, not the decoded content.
        encoded_body = gzip.compress(self.body) if self.compress_response else self.body
        rspauth = digest(
            algorithm,
            b":".join(
                (
                    HA1,
                    nonce.encode(),
                    fields["nc"].encode(),
                    fields["cnonce"].encode(),
                    qop.encode(),
                    digest(
                        algorithm,
                        b":".join(
                            (b"", request.url.raw_path, digest(algorithm, encoded_body))
                        ),
                    ),
                )
            ),
        )
        if self.bad_rspauth:
            rspauth = b"0" * len(rspauth)

        headers: dict[str, str] = {}
        if self.compress_response:
            headers["content-encoding"] = "gzip"
        if self.auth_info == "default":
            headers["authentication-info"] = f'rspauth="{rspauth.decode()}"'
        elif self.auth_info is not None:
            headers["authentication-info"] = self.auth_info

        if self.pre_read:
            # Mirror a response whose entity has already been buffered.
            return httpx.Response(200, content=encoded_body, headers=headers)

        # A real transport hands back an unread raw stream, so keep the raw
        # (still encoded) entity available to the client-side hash wrapper.
        if self.chunk_size is None:
            stream: SyncByteStream | AsyncByteStream = httpx.ByteStream(encoded_body)
        else:
            stream = _ChunkedStream(encoded_body, self.chunk_size)
        return httpx.Response(200, stream=stream, headers=headers)


def auth() -> httpx.DigestAuth:
    return httpx.DigestAuth(username=USERNAME, password=PASSWORD)


# Run the same assertions against the sync and async clients.
@pytest.mark.parametrize("algorithm", ["MD5", "MD5-SESS", "SHA-256", "SHA-256-SESS"])
def test_auth_int_binary_body(algorithm: str) -> None:
    # Include bytes that are invalid UTF-8 and look like transfer framing.
    body = b"\x00\x01\xff\xfe\r\n8\r\nbinary entity bytes\r\n0\r\n\r\n"
    app = DigestAuthIntApp(body=b"ok", algorithm=algorithm)

    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.post(
            "https://example.org/upload",
            content=body,
            auth=auth(),
            headers={"content-type": "application/octet-stream"},
        )

    assert response.status_code == 200
    assert response.history[-1].status_code == 401
    assert app.request_entities == [body]
    assert "qop=auth-int" in app.received_authorization[-1]
    assert response.digest_auth_verified == "verified"


@pytest.mark.anyio
@pytest.mark.parametrize("algorithm", ["MD5", "SHA-256", "SHA-512"])
async def test_auth_int_binary_body_async(algorithm: str) -> None:
    body = b"\x00\x01\xff\xfe\r\n8\r\nbinary entity bytes\r\n0\r\n\r\n"
    app = DigestAuthIntApp(body=b"ok", algorithm=algorithm)

    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.post(
            "https://example.org/upload",
            content=body,
            auth=auth(),
            headers={"content-type": "application/octet-stream"},
        )

    assert response.status_code == 200
    assert app.request_entities == [body]
    assert response.digest_auth_verified == "verified"


def test_auth_int_digest_is_over_entity_bytes_not_framing() -> None:
    """
    The digest must cover the entity bytes, not the transfer framing. A body
    that itself looks like chunked framing must be hashed verbatim.
    """
    body = b"5\r\nhello\r\n0\r\n\r\n"
    app = DigestAuthIntApp(body=b"ok")
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        client.post("https://example.org/", content=body, auth=auth())
    assert app.request_entities == [body]


def test_auth_int_compressed_response_hashes_raw_entity() -> None:
    """
    Content decoding must happen after the digest: `rspauth` covers the
    compressed entity, while the caller still sees the decoded content.
    """
    app = DigestAuthIntApp(body=b"x" * 1000, compress_response=True)
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.get("https://example.org/", auth=auth())

    assert response.content == b"x" * 1000
    assert response.text == "x" * 1000
    assert response.digest_auth_verified == "verified"


@pytest.mark.anyio
async def test_auth_int_compressed_response_hashes_raw_entity_async() -> None:
    app = DigestAuthIntApp(body=b"x" * 1000, compress_response=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.get("https://example.org/", auth=auth())

    assert response.content == b"x" * 1000
    assert response.digest_auth_verified == "verified"


def test_auth_int_stale_rechallenge() -> None:
    """
    A 'stale=true' 401 must be answered with a fresh nonce for the same
    entity, and the nonce count restarts at 1.
    """
    body = b"entity used across the stale challenge"
    app = DigestAuthIntApp(body=b"ok", stale_after_attempt=2)

    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.post("https://example.org/", content=body, auth=auth())

    assert response.status_code == 200
    assert app.calls == 3
    # The stale 401 carried no Authorization; only the two authenticated
    # requests reached the digest validation path, replaying the same entity.
    assert app.request_entities == [body]
    assert app.received_authorization[-1]  # the re-authenticated request
    # The re-challenge restarts the count.
    last_auth = parse_keqv_list(app.received_authorization[-1].split(", "))
    assert last_auth["nc"] == "00000001"
    assert last_auth["nonce"] == NEW_NONCE
    assert response.digest_auth_verified == "verified"


@pytest.mark.anyio
async def test_auth_int_stale_rechallenge_async() -> None:
    body = b"entity used across the stale challenge"
    app = DigestAuthIntApp(body=b"ok", stale_after_attempt=2)

    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.post("https://example.org/", content=body, auth=auth())

    assert response.status_code == 200
    assert app.calls == 3
    assert app.request_entities == [body]
    last_auth = parse_keqv_list(app.received_authorization[-1].split(", "))
    assert last_auth["nc"] == "00000001"
    assert last_auth["nonce"] == NEW_NONCE


def test_auth_int_bad_rspauth_raises() -> None:
    app = DigestAuthIntApp(body=b"ok", bad_rspauth=True)
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        with pytest.raises(httpx.DigestAuthError):
            client.get("https://example.org/", auth=auth())


@pytest.mark.anyio
async def test_auth_int_bad_rspauth_raises_async() -> None:
    app = DigestAuthIntApp(body=b"ok", bad_rspauth=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        with pytest.raises(httpx.DigestAuthError):
            await client.get("https://example.org/", auth=auth())


def test_auth_int_rspauth_verified_after_full_stream() -> None:
    app = DigestAuthIntApp(body=b"streamed response entity", chunk_size=3)
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        with client.stream("GET", "https://example.org/", auth=auth()) as response:
            content = response.read()

    assert content == b"streamed response entity"
    assert response.digest_auth_verified == "verified"


@pytest.mark.anyio
async def test_auth_int_rspauth_verified_after_full_async_stream() -> None:
    app = DigestAuthIntApp(body=b"streamed response entity", chunk_size=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        async with client.stream(
            "GET", "https://example.org/", auth=auth()
        ) as response:
            content = await response.aread()

    assert content == b"streamed response entity"
    assert response.digest_auth_verified == "verified"


def test_auth_int_early_close_is_unverified() -> None:
    """
    Closing the stream before the whole entity was consumed must never be
    reported as a successful verification.
    """
    app = DigestAuthIntApp(body=b"a" * 500, compress_response=True)
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        with client.stream("GET", "https://example.org/", auth=auth()) as response:
            chunks = response.iter_bytes()
            next(chunks)
            # Stop consuming early; the context manager closes the response.

    assert response.digest_auth_verified == "unverified"


@pytest.mark.anyio
async def test_auth_int_early_close_is_unverified_async() -> None:
    app = DigestAuthIntApp(body=b"a" * 500, compress_response=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        async with client.stream(
            "GET", "https://example.org/", auth=auth()
        ) as response:
            # Close the response without consuming the entity at all.
            pass

    assert response.digest_auth_verified == "unverified"


def test_auth_int_uses_only_available_algorithm() -> None:
    """A server that only offers 'auth-int' must be honoured."""
    app = DigestAuthIntApp(qop="auth-int")
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.post("https://example.org/", content=b"body", auth=auth())
    assert response.status_code == 200


def test_auth_int_is_not_chosen_when_auth_is_offered() -> None:
    """'auth' keeps being preferred whenever the server offers both."""
    app = DigestAuthIntApp(qop="auth, auth-int")
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.post("https://example.org/", content=b"body", auth=auth())
    assert response.status_code == 200


def test_auth_int_streaming_request_body_is_spooled() -> None:
    """
    A one-shot generator body must be buffered for the digest and replayed
    on the authenticated request instead of being consumed and lost.
    """
    body = b"chunked request body" * 32

    def stream() -> typing.Iterator[bytes]:
        yield body[:100]
        yield body[100:]

    app = DigestAuthIntApp(body=b"ok")
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.post("https://example.org/", content=stream(), auth=auth())

    assert response.status_code == 200
    assert app.request_entities == [body]
    assert response.digest_auth_verified == "verified"


@pytest.mark.anyio
async def test_auth_int_streaming_request_body_is_spooled_async() -> None:
    body = b"chunked request body" * 32

    async def stream() -> typing.AsyncIterator[bytes]:
        yield body[:100]
        yield body[100:]

    app = DigestAuthIntApp(body=b"ok")
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.post(
            "https://example.org/", content=stream(), auth=auth()
        )

    assert response.status_code == 200
    assert app.request_entities == [body]
    assert response.digest_auth_verified == "verified"


def test_spool_large_body_uses_temporary_file() -> None:
    """Large streaming bodies spill to a temporary file, not unbounded RAM."""
    half = MAX_IN_MEMORY_BODY_SIZE // 2
    chunks = [b"x" * half for _ in range(3)]

    spooled = _SpooledByteStream.from_sync_iterable(iter(chunks))
    try:
        assert spooled._file is not None
        assert spooled._buffer is None
        assert spooled.entity_bytes == b"x" * (half * 3)
    finally:
        spooled.close()
    assert spooled._file is None


def test_spool_small_body_stays_in_memory() -> None:
    spooled = _SpooledByteStream.from_sync_iterable(iter([b"ab", b"cd"]))
    assert spooled._file is None
    assert spooled.entity_bytes == b"abcd"
    spooled.close()


def test_spool_async_large_body_uses_temporary_file() -> None:
    half = MAX_IN_MEMORY_BODY_SIZE // 2

    async def chunks() -> typing.AsyncIterator[bytes]:
        for _ in range(3):
            yield b"y" * half

    async def main() -> None:
        spooled = await _SpooledByteStream.from_async_iterable(chunks())
        try:
            assert spooled._file is not None
            assert spooled.entity_bytes == b"y" * (half * 3)
        finally:
            await spooled.aclose()
        assert spooled._file is None

    anyio.run(main)


def test_auth_int_large_streaming_body() -> None:
    """An oversized streaming body is spooled to disk but still replayed."""
    body = bytes(range(256)) * 1 + b"z" * (MAX_IN_MEMORY_BODY_SIZE + 10)

    def stream() -> typing.Iterator[bytes]:
        for i in range(0, len(body), 65536):
            yield body[i : i + 65536]

    app = DigestAuthIntApp(body=b"ok")
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.post("https://example.org/", content=stream(), auth=auth())

    assert response.status_code == 200
    assert app.request_entities == [body]
    assert response.digest_auth_verified == "verified"


def test_auth_int_raw_request_stream_error_is_raised_unchanged() -> None:
    class BrokenStream(SyncByteStream):
        def __iter__(self) -> typing.Iterator[bytes]:
            raise RuntimeError("original stream error")

    digest_auth = auth()
    digest_auth._last_challenge = httpx._auth._DigestAuthChallenge(
        realm=REALM.encode(),
        nonce=NONCE.encode(),
        algorithm="SHA-256",
        opaque=None,
        qop=QOP,
        stale=False,
    )
    request = httpx.Request(
        "POST", "https://example.org/", stream=BrokenStream()
    )
    flow = digest_auth.sync_auth_flow(request)
    with pytest.raises(RuntimeError, match="original stream error"):
        next(flow)


def test_auth_int_concurrent_requests_reserve_distinct_nonce_counts() -> None:
    """
    A shared Auth instance must serialise nonce reservations so each
    in-flight request owns a distinct (nc, cnonce) pair.
    """
    import threading

    digest_auth = auth()
    digest_auth._last_challenge = httpx._auth._DigestAuthChallenge(
        realm=REALM.encode(),
        nonce=NONCE.encode(),
        algorithm="SHA-256",
        opaque=None,
        qop=QOP,
        stale=False,
    )

    seen: list[str] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        flow = digest_auth.sync_auth_flow(
            httpx.Request("GET", "https://example.org/", content=b"")
        )
        seen.append(next(flow).headers["authorization"])
        flow.close()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ncs = [parse_keqv_list(h.split(", "))["nc"] for h in seen]
    cnonces = [parse_keqv_list(h.split(", "))["cnonce"] for h in seen]
    assert len(set(ncs)) == 8
    assert len(set(cnonces)) == 8
    assert sorted(int(nc, 16) for nc in ncs) == list(range(1, 9))


def test_auth_int_repeated_stale_is_bounded() -> None:
    """A server that answers every authenticated request with stale=true
    cannot make the client loop forever: after one stale retry the 401 is
    returned to the caller."""
    app = DigestAuthIntApp(body=b"ok", stale_every_attempt=True)
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.post("https://example.org/", content=b"body", auth=auth())

    # unauthenticated request + first authenticated attempt + one stale retry
    assert app.calls == 3
    assert response.status_code == 401


@pytest.mark.anyio
async def test_auth_int_repeated_stale_is_bounded_async() -> None:
    app = DigestAuthIntApp(body=b"ok", stale_every_attempt=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.post(
            "https://example.org/", content=b"body", auth=auth()
        )

    assert app.calls == 3
    assert response.status_code == 401


def test_auth_int_bad_rspauth_on_streaming_response_raises() -> None:
    app = DigestAuthIntApp(body=b"streamed entity", chunk_size=4, bad_rspauth=True)
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        with pytest.raises(httpx.DigestAuthError):
            client.get("https://example.org/", auth=auth())


@pytest.mark.anyio
async def test_auth_int_bad_rspauth_on_streaming_response_raises_async() -> None:
    app = DigestAuthIntApp(body=b"streamed entity", chunk_size=4, bad_rspauth=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        with pytest.raises(httpx.DigestAuthError):
            await client.get("https://example.org/", auth=auth())


def test_auth_int_preread_response_is_verified() -> None:
    """A buffered response without content encoding can be verified directly."""
    app = DigestAuthIntApp(body=b"buffered entity", pre_read=True)
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.get("https://example.org/", auth=auth())

    assert response.digest_auth_verified == "verified"
    assert response.content == b"buffered entity"


def test_auth_int_preread_encoded_response_cannot_falsely_verify() -> None:
    """Buffered content-encoded responses hold decoded bytes only, so the raw
    entity cannot be re-hashed; the result must be 'unverified', never 'failed'
    and never a false 'verified'."""
    app = DigestAuthIntApp(
        body=b"z" * 200, compress_response=True, pre_read=True
    )
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.get("https://example.org/", auth=auth())

    assert response.digest_auth_verified == "unverified"
    assert response.content == b"z" * 200


def test_auth_int_no_authentication_info_header_is_not_verified() -> None:
    app = DigestAuthIntApp(auth_info=None)
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.get("https://example.org/", auth=auth())

    assert not hasattr(response, "digest_auth_verified")


def test_auth_int_malformed_authentication_info_is_ignored() -> None:
    """A header without an rspauth value leaves the response untouched."""
    app = DigestAuthIntApp(auth_info='nextnonce="abc", qop=auth-int')
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.get("https://example.org/", auth=auth())

    assert not hasattr(response, "digest_auth_verified")


def test_auth_int_qop_auth_path_remains_unchanged() -> None:
    """qop=auth must keep working and never add verification state."""
    app = DigestAuthIntApp(qop="auth, auth-int")
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.post("https://example.org/", content=b"body", auth=auth())

    assert response.status_code == 200
    assert not hasattr(response, "digest_auth_verified")
    sent = parse_keqv_list(app.received_authorization[-1].split(", "))
    assert sent["qop"] == "auth"


def test_auth_int_empty_body_hashes_the_empty_entity() -> None:
    app = DigestAuthIntApp(body=b"ok")
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.post("https://example.org/", content=b"", auth=auth())

    assert response.status_code == 200
    assert app.request_entities == [b""]
    assert response.digest_auth_verified == "verified"


def test_spooled_stream_replays_on_sync_and_async() -> None:
    spooled = _SpooledByteStream(buffer=b"replay me")
    assert b"".join(spooled) == b"replay me"
    assert b"".join([chunk for chunk in spooled]) == b"replay me"

    async def main() -> None:
        chunks = [chunk async for chunk in spooled]
        assert b"".join(chunks) == b"replay me"

    anyio.run(main)
    spooled.close()


def test_auth_int_response_verified_attribute_constants() -> None:
    assert httpx._auth.DIGEST_AUTH_VERIFIED == "verified"
    assert httpx._auth.DIGEST_AUTH_FAILED == "failed"
    assert httpx._auth.DIGEST_AUTH_UNVERIFIED == "unverified"


def test_auth_int_is_a_protocol_error_for_unknown_qop() -> None:
    app = DigestAuthIntApp(qop="something-else")
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        with pytest.raises(httpx.ProtocolError):
            client.get("https://example.org/", auth=auth())


@pytest.mark.anyio
async def test_auth_int_async_concurrent_nonce_counts() -> None:
    digest_auth = auth()
    digest_auth._last_challenge = httpx._auth._DigestAuthChallenge(
        realm=REALM.encode(),
        nonce=NONCE.encode(),
        algorithm="SHA-256",
        opaque=None,
        qop=QOP,
        stale=False,
    )

    seen: list[str] = []

    async def worker(index: int) -> None:
        flow = digest_auth.async_auth_flow(
            httpx.Request("POST", "https://example.org/", content=b"body")
        )
        request = await flow.__anext__()
        seen.append(request.headers["authorization"])
        await flow.aclose()

    async with anyio.create_task_group() as task_group:
        for index in range(6):
            task_group.start_soon(worker, index)

    ncs = [parse_keqv_list(h.split(", "))["nc"] for h in seen]
    cnonces = [parse_keqv_list(h.split(", "))["cnonce"] for h in seen]
    assert len(set(ncs)) == 6
    assert len(set(cnonces)) == 6
    assert sorted(int(nc, 16) for nc in ncs) == list(range(1, 7))


def test_auth_int_challenge_reuse_sends_nonce_count_sequence() -> None:
    """After the first negotiated request, later requests are preemptive and
    continue the nonce count without a fresh challenge."""
    app = DigestAuthIntApp(body=b"ok")
    digest_auth = auth()
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        first = client.post("https://example.org/", content=b"one", auth=digest_auth)
        second = client.post("https://example.org/", content=b"two", auth=digest_auth)

    assert first.status_code == second.status_code == 200
    first_nc = parse_keqv_list(app.received_authorization[0].split(", "))["nc"]
    second_nc = parse_keqv_list(app.received_authorization[1].split(", "))["nc"]
    assert int(second_nc, 16) == int(first_nc, 16) + 1


def test_auth_int_authentication_info_without_rspauth_is_ignored() -> None:
    """Bare tokens and empty values must not crash the header parser and must
    leave the response unverified rather than falsely passing."""
    app = DigestAuthIntApp(auth_info='bareflag, rspauth=, nextnonce="abc"')
    with httpx.Client(transport=httpx.MockTransport(app)) as client:
        response = client.get("https://example.org/", auth=auth())

    assert not hasattr(response, "digest_auth_verified")


def test_auth_int_malformed_digest_challenge_raises_protocol_error() -> None:
    headers = {"www-authenticate": 'Digest realm="httpx@example.org", qop="auth-int"'}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, headers=headers)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.ProtocolError):
            client.get("https://example.org/", auth=auth())


@pytest.mark.anyio
async def test_auth_int_reuses_spooled_body_on_async_rechallenge() -> None:
    """
    The second authenticated attempt must hash the already spooled entity
    rather than trying to consume the one-shot stream again.
    """
    body = b"one shot entity"
    app = DigestAuthIntApp(body=b"ok", stale_after_attempt=2)

    async def stream() -> typing.AsyncIterator[bytes]:
        yield body

    digest_auth = auth()
    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as client:
        response = await client.post(
            "https://example.org/", content=stream(), auth=digest_auth
        )

    assert response.status_code == 200
    assert app.request_entities == [body]


def test_auth_int_qop_none_is_still_supported() -> None:
    """A server that omits qop entirely (RFC 2069) still works, and an
    'Authentication-Info' rspauth is not checked for non auth-int qop."""
    headers = {
        "www-authenticate": (
            f'Digest realm="{REALM}", nonce="{NONCE}", algorithm=SHA-256'
        )
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "authorization" not in request.headers:
            return httpx.Response(401, headers=headers)
        return httpx.Response(
            200,
            content=b"ok",
            headers={"authentication-info": 'rspauth="whatever"'},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = client.get("https://example.org/", auth=auth())

    assert response.status_code == 200
    assert not hasattr(response, "digest_auth_verified")
    authorization = response.request.headers["authorization"]
    assert "qop=" not in authorization
    assert "nc=" not in authorization


def test_spooled_stream_update_hash_is_incremental() -> None:
    data = b"abc" * 100
    spooled = _SpooledByteStream(buffer=data)
    hexdigest = spooled.update_hash(hashlib.sha256()).hexdigest()
    assert hexdigest == hashlib.sha256(data).hexdigest()


def test_spooled_file_update_hash_does_not_materialize() -> None:
    data = b"q" * (MAX_IN_MEMORY_BODY_SIZE + 100)
    spooled = _SpooledByteStream.from_sync_iterable(iter([data]))
    try:
        assert spooled._file is not None
        assert spooled.update_hash(hashlib.sha256()).hexdigest() == hashlib.sha256(
            data
        ).hexdigest()
    finally:
        spooled.close()


def test_spool_error_closes_temporary_file() -> None:
    class FailingStream(SyncByteStream):
        def __iter__(self) -> typing.Iterator[bytes]:
            yield b"x" * (MAX_IN_MEMORY_BODY_SIZE + 1)
            raise RuntimeError("stream failed mid body")

    with pytest.raises(RuntimeError, match="stream failed mid body"):
        _SpooledByteStream.from_sync_iterable(FailingStream())


@pytest.mark.anyio
async def test_async_spool_error_closes_temporary_file() -> None:
    class FailingAsyncStream(AsyncByteStream):
        async def __aiter__(self) -> typing.AsyncIterator[bytes]:
            yield b"x" * (MAX_IN_MEMORY_BODY_SIZE + 1)
            raise RuntimeError("async stream failed mid body")

    with pytest.raises(RuntimeError, match="async stream failed mid body"):
        await _SpooledByteStream.from_async_iterable(FailingAsyncStream())


def test_auth_int_generic_auth_flow_delegates_to_sync_flow() -> None:
    """Direct `.auth_flow()` callers keep working for ordinary digest auth."""
    digest_auth = auth()
    request = httpx.Request("POST", "https://example.org/", content=b"body")
    flow = digest_auth.auth_flow(request)
    sent = next(flow)
    assert "authorization" not in sent.headers

    app = DigestAuthIntApp(qop="auth, auth-int")
    challenge = app(request)
    challenge.request = request
    authenticated = flow.send(challenge)
    assert authenticated.headers["authorization"].startswith("Digest ")
    fields = parse_keqv_list(authenticated.headers["authorization"].split(", "))
    assert fields["qop"] == "auth"
