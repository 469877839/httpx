from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
import time
import typing
from base64 import b64encode
from urllib.request import parse_http_list

import anyio

from ._exceptions import DigestAuthError, ProtocolError
from ._models import Cookies, Request, Response
from ._types import AsyncByteStream, SyncByteStream
from ._utils import to_bytes, to_str, unquote

if typing.TYPE_CHECKING:  # pragma: no cover
    from hashlib import _Hash


__all__ = ["Auth", "BasicAuth", "DigestAuth", "FunctionAuth", "NetRCAuth"]


# Maximum amount of request body data that is buffered in memory when an
# 'auth-int' digest has to be computed for a streaming request. Larger
# bodies are spooled to a temporary file instead of being collected wholesale.
MAX_IN_MEMORY_BODY_SIZE = 1024 * 1024

# Tri-state values exposed as `response.digest_auth_verified` once a response
# carrying an 'Authentication-Info' header has been processed.
DIGEST_AUTH_VERIFIED = "verified"
DIGEST_AUTH_FAILED = "failed"
DIGEST_AUTH_UNVERIFIED = "unverified"


class _SpooledByteStream(SyncByteStream, AsyncByteStream):
    """
    A replayable request body stream.

    Data below the in-memory limit is kept as a single bytes value. Larger
    bodies are written to a temporary file, so streaming requests are never
    collected into memory without bound. Both sync and async iteration replay
    the same buffered entity, and the underlying temporary resource is closed
    by `close()` / `aclose()`.
    """

    def __init__(
        self, buffer: bytes | None = None, *, file: typing.BinaryIO | None = None
    ) -> None:
        self._buffer = buffer
        self._file = file

    @classmethod
    def from_sync_iterable(cls, stream: typing.Iterable[bytes]) -> "_SpooledByteStream":
        buffer = bytearray()
        file: typing.BinaryIO | None = None
        try:
            for chunk in stream:
                file = cls._spool_chunk(buffer, file, chunk)
        except BaseException:
            if file is not None:
                file.close()
            raise
        return cls._finalize(buffer, file)

    @classmethod
    async def from_async_iterable(
        cls, stream: typing.AsyncIterable[bytes]
    ) -> "_SpooledByteStream":
        buffer = bytearray()
        file: typing.BinaryIO | None = None
        try:
            async for chunk in stream:
                file = cls._spool_chunk(buffer, file, chunk)
        except BaseException:
            if file is not None:
                file.close()
            raise
        return cls._finalize(buffer, file)

    @staticmethod
    def _spool_chunk(
        buffer: bytearray, file: typing.BinaryIO | None, chunk: bytes
    ) -> typing.BinaryIO | None:
        if file is not None:
            file.write(chunk)
        elif len(buffer) + len(chunk) <= MAX_IN_MEMORY_BODY_SIZE:
            buffer.extend(chunk)
        else:
            file = tempfile.SpooledTemporaryFile(max_size=MAX_IN_MEMORY_BODY_SIZE)
            file.write(bytes(buffer))
            file.write(chunk)
            buffer.clear()
        return file

    @classmethod
    def _finalize(
        cls, buffer: bytearray, file: typing.BinaryIO | None
    ) -> "_SpooledByteStream":
        if file is not None:
            file.seek(0)
            return cls(file=file)
        return cls(buffer=bytes(buffer))

    @property
    def entity_bytes(self) -> bytes:
        """
        The full spooled entity. For spooled-file bodies this reads the
        temporary file back into memory. Prefer `update_hash()` for computing
        an entity digest, which streams through the file instead.
        """
        if self._file is not None:
            self._file.seek(0)
            return self._file.read()
        return self._buffer or b""

    def update_hash(self, hash_state: "_Hash") -> "_Hash":
        """
        Feed the whole spooled entity into a hash state without collecting it
        into memory a second time, returning the updated state.
        """
        for chunk in self:
            hash_state.update(chunk)
        return hash_state

    def __iter__(self) -> typing.Iterator[bytes]:
        if self._file is not None:
            self._file.seek(0)
            while True:
                chunk = self._file.read(65_536)
                if not chunk:
                    break
                yield chunk
        else:
            yield self._buffer or b""

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        for chunk in self:
            yield chunk

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    async def aclose(self) -> None:
        self.close()


class _HashingByteStream(SyncByteStream):
    """
    Wraps a sync response stream, feeding every raw entity byte into a hash
    while still allowing the response to be read or decoded normally.

    When the whole entity has been consumed the `on_complete` callback runs.
    An early `close()` (for example the caller stopping iteration) instead
    runs `on_incomplete`, so a partial entity can never be reported as
    verified.
    """

    def __init__(
        self,
        stream: SyncByteStream,
        hash_state: "_Hash",
        on_complete: typing.Callable[[str], None],
        on_incomplete: typing.Callable[[], None],
    ) -> None:
        self._stream = stream
        self._hash_state = hash_state
        self._on_complete = on_complete
        self._on_incomplete = on_incomplete
        self._is_complete = False

    def __iter__(self) -> typing.Iterator[bytes]:
        for chunk in self._stream:
            self._hash_state.update(chunk)
            yield chunk
        self._is_complete = True
        self._on_complete(self._hash_state.hexdigest())

    def close(self) -> None:
        if not self._is_complete:
            self._on_incomplete()
        self._stream.close()


class _AsyncHashingByteStream(AsyncByteStream):
    """
    Wraps an async response stream, feeding every raw entity byte into a hash
    while still allowing the response to be read or decoded normally.

    When the whole entity has been consumed the `on_complete` callback runs.
    An early `aclose()` (for example the caller stopping iteration) instead
    runs `on_incomplete`, so a partial entity can never be reported as
    verified.
    """

    def __init__(
        self,
        stream: AsyncByteStream,
        hash_state: "_Hash",
        on_complete: typing.Callable[[str], None],
        on_incomplete: typing.Callable[[], None],
    ) -> None:
        self._stream = stream
        self._hash_state = hash_state
        self._on_complete = on_complete
        self._on_incomplete = on_incomplete
        self._is_complete = False

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        async for chunk in self._stream:
            self._hash_state.update(chunk)
            yield chunk
        self._is_complete = True
        self._on_complete(self._hash_state.hexdigest())

    async def aclose(self) -> None:
        if not self._is_complete:
            self._on_incomplete()
        await self._stream.aclose()


class Auth:
    """
    Base class for all authentication schemes.

    To implement a custom authentication scheme, subclass `Auth` and override
    the `.auth_flow()` method.

    If the authentication scheme does I/O such as disk access or network calls, or uses
    synchronization primitives such as locks, you should override `.sync_auth_flow()`
    and/or `.async_auth_flow()` instead of `.auth_flow()` to provide specialized
    implementations that will be used by `Client` and `AsyncClient` respectively.
    """

    requires_request_body = False
    requires_response_body = False

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        """
        Execute the authentication flow.

        To dispatch a request, `yield` it:

        ```
        yield request
        ```

        The client will `.send()` the response back into the flow generator. You can
        access it like so:

        ```
        response = yield request
        ```

        A `return` (or reaching the end of the generator) will result in the
        client returning the last response obtained from the server.

        You can dispatch as many requests as is necessary.
        """
        yield request

    def sync_auth_flow(
        self, request: Request
    ) -> typing.Generator[Request, Response, None]:
        """
        Execute the authentication flow synchronously.

        By default, this defers to `.auth_flow()`. You should override this method
        when the authentication scheme does I/O and/or uses concurrency primitives.
        """
        if self.requires_request_body:
            request.read()

        flow = self.auth_flow(request)
        request = next(flow)

        while True:
            response = yield request
            if self.requires_response_body:
                response.read()

            try:
                request = flow.send(response)
            except StopIteration:
                break

    async def async_auth_flow(
        self, request: Request
    ) -> typing.AsyncGenerator[Request, Response]:
        """
        Execute the authentication flow asynchronously.

        By default, this defers to `.auth_flow()`. You should override this method
        when the authentication scheme does I/O and/or uses concurrency primitives.
        """
        if self.requires_request_body:
            await request.aread()

        flow = self.auth_flow(request)
        request = next(flow)

        while True:
            response = yield request
            if self.requires_response_body:
                await response.aread()

            try:
                request = flow.send(response)
            except StopIteration:
                break


class FunctionAuth(Auth):
    """
    Allows the 'auth' argument to be passed as a simple callable function,
    that takes the request, and returns a new, modified request.
    """

    def __init__(self, func: typing.Callable[[Request], Request]) -> None:
        self._func = func

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        yield self._func(request)


class BasicAuth(Auth):
    """
    Allows the 'auth' argument to be passed as a (username, password) pair,
    and uses HTTP Basic authentication.
    """

    def __init__(self, username: str | bytes, password: str | bytes) -> None:
        self._auth_header = self._build_auth_header(username, password)

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        request.headers["Authorization"] = self._auth_header
        yield request

    def _build_auth_header(self, username: str | bytes, password: str | bytes) -> str:
        userpass = b":".join((to_bytes(username), to_bytes(password)))
        token = b64encode(userpass).decode()
        return f"Basic {token}"


class NetRCAuth(Auth):
    """
    Use a 'netrc' file to lookup basic auth credentials based on the url host.
    """

    def __init__(self, file: str | None = None) -> None:
        # Lazily import 'netrc'.
        # There's no need for us to load this module unless 'NetRCAuth' is being used.
        import netrc

        self._netrc_info = netrc.netrc(file)

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        auth_info = self._netrc_info.authenticators(request.url.host)
        if auth_info is None or not auth_info[2]:
            # The netrc file did not have authentication credentials for this host.
            yield request
        else:
            # Build a basic auth header with credentials from the netrc file.
            request.headers["Authorization"] = self._build_auth_header(
                username=auth_info[0], password=auth_info[2]
            )
            yield request

    def _build_auth_header(self, username: str | bytes, password: str | bytes) -> str:
        userpass = b":".join((to_bytes(username), to_bytes(password)))
        token = b64encode(userpass).decode()
        return f"Basic {token}"


class DigestAuth(Auth):
    _ALGORITHM_TO_HASH_FUNCTION: dict[str, typing.Callable[[bytes], _Hash]] = {
        "MD5": hashlib.md5,
        "MD5-SESS": hashlib.md5,
        "SHA": hashlib.sha1,
        "SHA-SESS": hashlib.sha1,
        "SHA-256": hashlib.sha256,
        "SHA-256-SESS": hashlib.sha256,
        "SHA-512": hashlib.sha512,
        "SHA-512-SESS": hashlib.sha512,
    }

    # At most one 'stale=true' re-challenge is followed within a single flow,
    # so a misbehaving server cannot make the client authenticate forever.
    _MAX_STALE_REQUESTS = 1

    def __init__(self, username: str | bytes, password: str | bytes) -> None:
        self._username = to_bytes(username)
        self._password = to_bytes(password)
        self._last_challenge: _DigestAuthChallenge | None = None
        # The nonce count is owned by the Auth instance, so a sequence of
        # requests against the same challenge shares a single count. Each
        # (count, cnonce) pair is reserved under a lock, so concurrent use of
        # the same instance cannot interleave another request's sequence.
        self._nonce_count = 1
        self._sync_lock = threading.Lock()
        self._async_lock = anyio.Lock()

    def auth_flow(self, request: Request) -> typing.Generator[Request, Response, None]:
        # Keep the generic entry point working for callers that invoke
        # `.auth_flow()` directly. It can only drive a synchronous body, which
        # covers the ordinary `qop=None`/`qop=auth` cases unchanged.
        yield from self.sync_auth_flow(request)

    async def async_auth_flow(
        self, request: Request
    ) -> typing.AsyncGenerator[Request, Response]:
        sent_auth: tuple[_DigestAuthChallenge, int, bytes] | None = None
        async with self._async_lock:
            challenge = self._last_challenge
            if challenge is not None:
                sent_auth = (challenge, *self._next_nonce_pair())
        if sent_auth is not None:
            challenge, nonce_count, cnonce = sent_auth
            body_digest = await self._spool_request_body_async(request, challenge)
            request.headers["Authorization"] = self._build_auth_header(
                request, challenge, nonce_count, cnonce, body_digest=body_digest
            )

        response = yield request
        retries = 0
        stale_requests = 0

        while response.status_code == 401 and "www-authenticate" in response.headers:
            auth_header = self._find_digest_header(response)
            if auth_header is None:
                # A 401 that does not use Digest cannot be handled here.
                return

            challenge = self._parse_challenge(request, response, auth_header)
            if not self._can_retry(
                challenge, retries=retries, stale_requests=stale_requests
            ):
                # Ordinary re-challenges only get a single retry; stale
                # re-challenges get one extra, bounded by the limit above.
                return
            stale_requests += int(challenge.stale)

            async with self._async_lock:
                self._replace_challenge(challenge)
                nonce_count, cnonce = self._next_nonce_pair()
            body_digest = await self._spool_request_body_async(request, challenge)
            request.headers["Authorization"] = self._build_auth_header(
                request, challenge, nonce_count, cnonce, body_digest=body_digest
            )
            sent_auth = (challenge, nonce_count, cnonce)
            if response.cookies:
                Cookies(response.cookies).set_cookie_header(request=request)
            retries += 1
            response = yield request

        if sent_auth is not None and self._is_auth_int(sent_auth[0], request):
            self._wrap_response_for_rspauth(
                request, response, *sent_auth, async_mode=True
            )

    def sync_auth_flow(
        self, request: Request
    ) -> typing.Generator[Request, Response, None]:
        sent_auth: tuple[_DigestAuthChallenge, int, bytes] | None = None
        with self._sync_lock:
            challenge = self._last_challenge
            if challenge is not None:
                sent_auth = (challenge, *self._next_nonce_pair())
        if sent_auth is not None:
            challenge, nonce_count, cnonce = sent_auth
            body_digest = self._spool_request_body_sync(request, challenge)
            request.headers["Authorization"] = self._build_auth_header(
                request, challenge, nonce_count, cnonce, body_digest=body_digest
            )

        response = yield request
        retries = 0
        stale_requests = 0

        while response.status_code == 401 and "www-authenticate" in response.headers:
            auth_header = self._find_digest_header(response)
            if auth_header is None:
                # A 401 that does not use Digest cannot be handled here.
                return

            challenge = self._parse_challenge(request, response, auth_header)
            if not self._can_retry(
                challenge, retries=retries, stale_requests=stale_requests
            ):
                # Ordinary re-challenges only get a single retry; stale
                # re-challenges get one extra, bounded by the limit above.
                return
            stale_requests += int(challenge.stale)

            with self._sync_lock:
                self._replace_challenge(challenge)
                nonce_count, cnonce = self._next_nonce_pair()
            body_digest = self._spool_request_body_sync(request, challenge)
            request.headers["Authorization"] = self._build_auth_header(
                request, challenge, nonce_count, cnonce, body_digest=body_digest
            )
            sent_auth = (challenge, nonce_count, cnonce)
            if response.cookies:
                Cookies(response.cookies).set_cookie_header(request=request)
            retries += 1
            response = yield request

        if sent_auth is not None and self._is_auth_int(sent_auth[0], request):
            self._wrap_response_for_rspauth(
                request, response, *sent_auth, async_mode=False
            )

    # ----- challenge handling -------------------------------------------------

    def _find_digest_header(self, response: Response) -> str | None:
        for auth_header in response.headers.get_list("www-authenticate"):
            if auth_header.lower().startswith("digest "):
                return auth_header
        return None

    def _parse_challenge(
        self, request: Request, response: Response, auth_header: str
    ) -> _DigestAuthChallenge:
        """
        Returns a challenge from a Digest WWW-Authenticate header.
        These take the form of:
        `Digest realm="realm@host.com",qop="auth,auth-int",nonce="abc",opaque="xyz"`
        """
        scheme, _, fields = auth_header.partition(" ")

        # This method should only ever have been called with a Digest auth header.
        assert scheme.lower() == "digest"

        header_dict: dict[str, str] = {}
        for field in parse_http_list(fields):
            key, value = field.strip().split("=", 1)
            header_dict[key] = unquote(value)

        try:
            realm = header_dict["realm"].encode()
            nonce = header_dict["nonce"].encode()
            algorithm = header_dict.get("algorithm", "MD5")
            opaque = header_dict["opaque"].encode() if "opaque" in header_dict else None
            qop = header_dict["qop"].encode() if "qop" in header_dict else None
            stale = header_dict.get("stale", "").lower() == "true"
            return _DigestAuthChallenge(
                realm=realm,
                nonce=nonce,
                algorithm=algorithm,
                opaque=opaque,
                qop=qop,
                stale=stale,
            )
        except KeyError as exc:
            message = "Malformed Digest WWW-Authenticate header"
            raise ProtocolError(message, request=request) from exc

    def _replace_challenge(self, challenge: _DigestAuthChallenge) -> None:
        # A new (possibly stale) challenge always restarts the nonce count.
        self._last_challenge = challenge
        self._nonce_count = 1

    def _can_retry(
        self,
        challenge: _DigestAuthChallenge,
        *,
        retries: int,
        stale_requests: int,
    ) -> bool:
        if retries == 0:
            # The first challenge is always answered.
            return True
        if not challenge.stale:
            # The credentials were already tried once: surface the 401.
            return False
        # 'stale=true' allows one more attempt with the new nonce.
        return stale_requests < self._MAX_STALE_REQUESTS

    def _is_auth_int(
        self, challenge: _DigestAuthChallenge, request: Request
    ) -> bool:
        return self._resolve_qop(challenge.qop, request=request) == b"auth-int"

    def _resolve_qop(self, qop: bytes | None, request: Request) -> bytes | None:
        if qop is None:
            return None
        qops = re.split(b", ?", qop)
        # 'auth' is preferred whenever the server offers it, preserving the
        # behaviour used before 'auth-int' was supported.
        if b"auth" in qops:
            return b"auth"
        if b"auth-int" in qops:
            return b"auth-int"

        message = f'Unexpected qop value "{qop!r}" in digest auth'
        raise ProtocolError(message, request=request)

    # ----- nonce state --------------------------------------------------------

    def _next_nonce_pair(self) -> tuple[int, bytes]:
        # Must be called while holding the sync/async lock.
        nonce_count = self._nonce_count
        cnonce = self._get_client_nonce(nonce_count, self._last_nonce)
        self._nonce_count += 1
        return nonce_count, cnonce

    @property
    def _last_nonce(self) -> bytes:
        assert self._last_challenge is not None
        return self._last_challenge.nonce

    def _get_client_nonce(self, nonce_count: int, nonce: bytes) -> bytes:
        s = str(nonce_count).encode()
        s += nonce
        s += time.ctime().encode()
        s += os.urandom(8)

        return hashlib.sha1(s).hexdigest()[:16].encode()

    # ----- request body spooling ----------------------------------------------

    def _spool_request_body_sync(
        self, request: Request, challenge: _DigestAuthChallenge
    ) -> bytes | None:
        """
        Return the H(entity-body) digest for an 'auth-int' challenge, or None
        for qop=None/qop=auth. The digest covers the raw entity bytes:
        transfer framing (chunking) has already been removed by the stream,
        while content decoding (gzip etc.) never applies to a request body.
        """
        if not self._is_auth_int(challenge, request):
            return None

        if hasattr(request, "_content"):
            # Bytes bodies are already fully buffered, so hash them directly.
            return self._hash_factory(challenge)(request.content).hexdigest().encode()
        if isinstance(request.stream, _SpooledByteStream):
            # A re-challenge reuses the already spooled, replayable entity.
            spooled = request.stream
        else:
            spooled = _SpooledByteStream.from_sync_iterable(request.stream)
            request.stream = spooled
        return spooled.update_hash(self._hash_factory(challenge)()).hexdigest().encode()

    async def _spool_request_body_async(
        self, request: Request, challenge: _DigestAuthChallenge
    ) -> bytes | None:
        if not self._is_auth_int(challenge, request):
            return None

        if hasattr(request, "_content"):
            # Bytes bodies are already fully buffered, so hash them directly.
            return self._hash_factory(challenge)(request.content).hexdigest().encode()
        if isinstance(request.stream, _SpooledByteStream):
            # A re-challenge reuses the already spooled, replayable entity.
            spooled = request.stream
        else:
            spooled = await _SpooledByteStream.from_async_iterable(request.stream)
            request.stream = spooled
        return spooled.update_hash(self._hash_factory(challenge)()).hexdigest().encode()

    def _hash_factory(
        self, challenge: _DigestAuthChallenge
    ) -> typing.Callable[[bytes], _Hash]:
        return self._ALGORITHM_TO_HASH_FUNCTION[challenge.algorithm.upper()]

    # ----- Authorization header -----------------------------------------------

    def _build_auth_header(
        self,
        request: Request,
        challenge: _DigestAuthChallenge,
        nonce_count: int,
        cnonce: bytes,
        body_digest: bytes | None = None,
    ) -> str:
        hash_func = self._hash_factory(challenge)

        def digest(data: bytes) -> bytes:
            return hash_func(data).hexdigest().encode()

        A1 = b":".join((self._username, challenge.realm, self._password))

        path = request.url.raw_path
        qop = self._resolve_qop(challenge.qop, request=request)
        if qop == b"auth-int":
            # H(entity-body): the entity after transfer framing is removed and
            # before any content decoding. The body has been spooled so the
            # same bytes are replayed when the request is sent.
            assert body_digest is not None
            A2 = b":".join((request.method.encode(), path, body_digest))
        else:
            A2 = b":".join((request.method.encode(), path))
        HA2 = digest(A2)

        nc_value = b"%08x" % nonce_count

        HA1 = digest(A1)
        if challenge.algorithm.lower().endswith("-sess"):
            HA1 = digest(b":".join((HA1, challenge.nonce, cnonce)))

        if qop is None:
            # Following RFC 2069
            digest_data = [HA1, challenge.nonce, HA2]
        else:
            # Following RFC 2617/7616
            digest_data = [HA1, challenge.nonce, nc_value, cnonce, qop, HA2]

        format_args = {
            "username": self._username,
            "realm": challenge.realm,
            "nonce": challenge.nonce,
            "uri": path,
            "response": digest(b":".join(digest_data)),
            "algorithm": challenge.algorithm.encode(),
        }
        if challenge.opaque:
            format_args["opaque"] = challenge.opaque
        if qop:
            format_args["qop"] = qop
            format_args["nc"] = nc_value
            format_args["cnonce"] = cnonce

        return "Digest " + self._get_header_value(format_args)

    def _get_header_value(self, header_fields: dict[str, bytes]) -> str:
        NON_QUOTED_FIELDS = ("algorithm", "qop", "nc")
        QUOTED_TEMPLATE = '{}="{}"'
        NON_QUOTED_TEMPLATE = "{}={}"

        header_value = ""
        for i, (field, value) in enumerate(header_fields.items()):
            if i > 0:
                header_value += ", "
            template = (
                QUOTED_TEMPLATE
                if field not in NON_QUOTED_FIELDS
                else NON_QUOTED_TEMPLATE
            )
            header_value += template.format(field, to_str(value))

        return header_value

    # ----- rspauth verification -----------------------------------------------

    def _wrap_response_for_rspauth(
        self,
        request: Request,
        response: Response,
        challenge: _DigestAuthChallenge,
        nonce_count: int,
        cnonce: bytes,
        *,
        async_mode: bool,
    ) -> None:
        """
        Arrange for the 'rspauth' value of an 'Authentication-Info' header to
        be checked once the response entity has been fully consumed.

        The digest covers the raw entity (transfer framing removed, content
        encoding still applied), so the stream itself is hashed rather than
        the decoded `response.content`.
        """
        auth_info = self._parse_authentication_info(response)
        if auth_info is None or "rspauth" not in auth_info:
            # Nothing to verify: leave the response completely untouched.
            return

        expected = auth_info["rspauth"].encode()
        hash_func = self._hash_factory(challenge)

        def digest(data: bytes) -> bytes:
            return hash_func(data).hexdigest().encode()

        def verify(entity_hexdigest: str) -> None:
            # Reconstruct the full 'rspauth' from the parameters that were
            # actually used on the authenticated request. The response uses
            # an empty method in A2, while the entity hash is the raw digest
            # measured while the stream was consumed.
            A1 = b":".join((self._username, challenge.realm, self._password))
            HA1 = digest(A1)
            if challenge.algorithm.lower().endswith("-sess"):
                HA1 = digest(b":".join((HA1, challenge.nonce, cnonce)))
            A2 = b":".join(
                (b"", request.url.raw_path, entity_hexdigest.encode())
            )
            HA2 = digest(A2)
            nc_value = b"%08x" % nonce_count
            rspauth = digest(
                b":".join(
                    (HA1, challenge.nonce, nc_value, cnonce, b"auth-int", HA2)
                )
            )
            if rspauth == expected:
                response.digest_auth_verified = DIGEST_AUTH_VERIFIED
            else:
                response.digest_auth_verified = DIGEST_AUTH_FAILED
                message = (
                    "Digest 'rspauth' does not match the response entity "
                    "for the requested resource."
                )
                raise DigestAuthError(message, request=request)

        def mark_unverified() -> None:
            # The stream was closed before the whole entity was consumed.
            # A partial hash must never be reported as a success.
            response.digest_auth_verified = DIGEST_AUTH_UNVERIFIED

        if hasattr(response, "_content"):
            # The entity was already fully read. `_content` holds the decoded
            # content, which only equals the raw entity when no content
            # encoding is applied. If an encoding is present the raw bytes are
            # no longer recoverable, so report it as unverified rather than
            # risk verifying the wrong bytes.
            if response.headers.get("content-encoding") is None:
                entity_digest = self._hash_factory(challenge)(
                    response.content
                ).hexdigest()
                verify(entity_digest)
            else:
                response.digest_auth_verified = DIGEST_AUTH_UNVERIFIED
            return

        hash_state = self._hash_factory(challenge)()
        if async_mode:
            assert isinstance(response.stream, AsyncByteStream)
            response.stream = _AsyncHashingByteStream(
                response.stream, hash_state, verify, mark_unverified
            )
        else:
            assert isinstance(response.stream, SyncByteStream)
            response.stream = _HashingByteStream(
                response.stream, hash_state, verify, mark_unverified
            )

    def _parse_authentication_info(self, response: Response) -> dict[str, str] | None:
        header = response.headers.get("authentication-info")
        if not header:
            return None
        result: dict[str, str] = {}
        for field in parse_http_list(header):
            if "=" not in field:
                continue
            key, value = field.strip().split("=", 1)
            if value:
                result[key.lower()] = unquote(value)
        return result


class _DigestAuthChallenge(typing.NamedTuple):
    realm: bytes
    nonce: bytes
    algorithm: str
    opaque: bytes | None
    qop: bytes | None
    stale: bool = False
