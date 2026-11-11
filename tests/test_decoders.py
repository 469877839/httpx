from __future__ import annotations

import gzip
import io
import typing
import zlib

import chardet
import pytest
import zstandard as zstd

import httpx


def test_deflate():
    """
    Deflate encoding may use either 'zlib' or 'deflate' in the wild.

    https://stackoverflow.com/questions/1838699/how-can-i-decompress-a-gzip-stream-with-zlib#answer-22311297
    """
    body = b"test 123"
    compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    compressed_body = compressor.compress(body) + compressor.flush()

    headers = [(b"Content-Encoding", b"deflate")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_zlib():
    """
    Deflate encoding may use either 'zlib' or 'deflate' in the wild.

    https://stackoverflow.com/questions/1838699/how-can-i-decompress-a-gzip-stream-with-zlib#answer-22311297
    """
    body = b"test 123"
    compressed_body = zlib.compress(body)

    headers = [(b"Content-Encoding", b"deflate")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_gzip():
    body = b"test 123"
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    compressed_body = compressor.compress(body) + compressor.flush()

    headers = [(b"Content-Encoding", b"gzip")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_brotli():
    body = b"test 123"
    compressed_body = b"\x8b\x03\x80test 123\x03"

    headers = [(b"Content-Encoding", b"br")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_zstd():
    body = b"test 123"
    compressed_body = zstd.compress(body)

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_zstd_decoding_error():
    compressed_body = "this_is_not_zstd_compressed_data"

    headers = [(b"Content-Encoding", b"zstd")]
    with pytest.raises(httpx.DecodingError):
        httpx.Response(
            200,
            headers=headers,
            content=compressed_body,
        )


def test_zstd_empty():
    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(200, headers=headers, content=b"")
    assert response.content == b""


def test_zstd_truncated():
    body = b"test 123"
    compressed_body = zstd.compress(body)

    headers = [(b"Content-Encoding", b"zstd")]
    with pytest.raises(httpx.DecodingError):
        httpx.Response(
            200,
            headers=headers,
            content=compressed_body[1:3],
        )


def test_zstd_multiframe():
    # test inspired by urllib3 test suite
    data = (
        # Zstandard frame
        zstd.compress(b"foo")
        # skippable frame (must be ignored)
        + bytes.fromhex(
            "50 2A 4D 18"  # Magic_Number (little-endian)
            "07 00 00 00"  # Frame_Size (little-endian)
            "00 00 00 00 00 00 00"  # User_Data
        )
        # Zstandard frame
        + zstd.compress(b"bar")
    )
    compressed_body = io.BytesIO(data)

    headers = [(b"Content-Encoding", b"zstd")]
    response = httpx.Response(200, headers=headers, content=compressed_body)
    response.read()
    assert response.content == b"foobar"


def test_multi():
    body = b"test 123"

    deflate_compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    compressed_body = deflate_compressor.compress(body) + deflate_compressor.flush()

    gzip_compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    compressed_body = (
        gzip_compressor.compress(compressed_body) + gzip_compressor.flush()
    )

    headers = [(b"Content-Encoding", b"deflate, gzip")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


def test_multi_with_identity():
    body = b"test 123"
    compressed_body = b"\x8b\x03\x80test 123\x03"

    headers = [(b"Content-Encoding", b"br, identity")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body

    headers = [(b"Content-Encoding", b"identity, br")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compressed_body,
    )
    assert response.content == body


@pytest.mark.anyio
async def test_streaming():
    body = b"test 123"
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)

    async def compress(body: bytes) -> typing.AsyncIterator[bytes]:
        yield compressor.compress(body)
        yield compressor.flush()

    headers = [(b"Content-Encoding", b"gzip")]
    response = httpx.Response(
        200,
        headers=headers,
        content=compress(body),
    )
    assert not hasattr(response, "body")
    assert await response.aread() == body


@pytest.mark.parametrize("header_value", (b"deflate", b"gzip", b"br", b"identity"))
def test_empty_content(header_value):
    headers = [(b"Content-Encoding", header_value)]
    response = httpx.Response(
        200,
        headers=headers,
        content=b"",
    )
    assert response.content == b""


@pytest.mark.parametrize("header_value", (b"deflate", b"gzip", b"br", b"identity"))
def test_decoders_empty_cases(header_value):
    headers = [(b"Content-Encoding", header_value)]
    response = httpx.Response(content=b"", status_code=200, headers=headers)
    assert response.read() == b""


@pytest.mark.parametrize("header_value", (b"deflate", b"gzip", b"br"))
def test_decoding_errors(header_value):
    headers = [(b"Content-Encoding", header_value)]
    compressed_body = b"invalid"
    with pytest.raises(httpx.DecodingError):
        request = httpx.Request("GET", "https://example.org")
        httpx.Response(200, headers=headers, content=compressed_body, request=request)

    with pytest.raises(httpx.DecodingError):
        httpx.Response(200, headers=headers, content=compressed_body)


@pytest.mark.parametrize(
    ["data", "encoding"],
    [
        ((b"Hello,", b" world!"), "ascii"),
        ((b"\xe3\x83", b"\x88\xe3\x83\xa9", b"\xe3", b"\x83\x99\xe3\x83\xab"), "utf-8"),
        ((b"Euro character: \x88! abcdefghijklmnopqrstuvwxyz", b""), "cp1252"),
        ((b"Accented: \xd6sterreich abcdefghijklmnopqrstuvwxyz", b""), "iso-8859-1"),
    ],
)
@pytest.mark.anyio
async def test_text_decoder_with_autodetect(data, encoding):
    async def iterator() -> typing.AsyncIterator[bytes]:
        nonlocal data
        for chunk in data:
            yield chunk

    def autodetect(content):
        return chardet.detect(content).get("encoding")

    # Accessing `.text` on a read response.
    response = httpx.Response(200, content=iterator(), default_encoding=autodetect)
    await response.aread()
    assert response.text == (b"".join(data)).decode(encoding)

    # Streaming `.aiter_text` iteratively.
    # Note that if we streamed the text *without* having read it first, then
    # we won't get a `charset_normalizer` guess, and will instead always rely
    # on utf-8 if no charset is specified.
    text = "".join([part async for part in response.aiter_text()])
    assert text == (b"".join(data)).decode(encoding)


@pytest.mark.anyio
async def test_text_decoder_known_encoding():
    async def iterator() -> typing.AsyncIterator[bytes]:
        yield b"\x83g"
        yield b"\x83"
        yield b"\x89\x83x\x83\x8b"

    response = httpx.Response(
        200,
        headers=[(b"Content-Type", b"text/html; charset=shift-jis")],
        content=iterator(),
    )

    await response.aread()
    assert "".join(response.text) == "トラベル"


def test_text_decoder_empty_cases():
    response = httpx.Response(200, content=b"")
    assert response.text == ""

    response = httpx.Response(200, content=[b""])
    response.read()
    assert response.text == ""


@pytest.mark.parametrize(
    ["data", "expected"],
    [((b"Hello,", b" world!"), ["Hello,", " world!"])],
)
def test_streaming_text_decoder(
    data: typing.Iterable[bytes], expected: list[str]
) -> None:
    response = httpx.Response(200, content=iter(data))
    assert list(response.iter_text()) == expected


def test_line_decoder_nl():
    response = httpx.Response(200, content=[b""])
    assert list(response.iter_lines()) == []

    response = httpx.Response(200, content=[b"", b"a\n\nb\nc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    # Issue #1033
    response = httpx.Response(
        200, content=[b"", b"12345\n", b"foo ", b"bar ", b"baz\n"]
    )
    assert list(response.iter_lines()) == ["12345", "foo bar baz"]


def test_line_decoder_cr():
    response = httpx.Response(200, content=[b"", b"a\r\rb\rc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    response = httpx.Response(200, content=[b"", b"a\r\rb\rc\r"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    # Issue #1033
    response = httpx.Response(
        200, content=[b"", b"12345\r", b"foo ", b"bar ", b"baz\r"]
    )
    assert list(response.iter_lines()) == ["12345", "foo bar baz"]


def test_line_decoder_crnl():
    response = httpx.Response(200, content=[b"", b"a\r\n\r\nb\r\nc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    response = httpx.Response(200, content=[b"", b"a\r\n\r\nb\r\nc\r\n"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    response = httpx.Response(200, content=[b"", b"a\r", b"\n\r\nb\r\nc"])
    assert list(response.iter_lines()) == ["a", "", "b", "c"]

    # Issue #1033
    response = httpx.Response(200, content=[b"", b"12345\r\n", b"foo bar baz\r\n"])
    assert list(response.iter_lines()) == ["12345", "foo bar baz"]


def test_invalid_content_encoding_header():
    headers = [(b"Content-Encoding", b"invalid-header")]
    body = b"test 123"

    response = httpx.Response(
        200,
        headers=headers,
        content=body,
    )
    assert response.content == body


# A short payload exercising multibyte characters, CRLF split points and a
# final segment with no trailing newline.
CONSISTENCY_BODY = "aé中\r\nx\ry\nz end".encode("utf-8")


def _gzip(data: bytes) -> bytes:
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    return compressor.compress(data) + compressor.flush()


def _raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


def _partitions(data: bytes):
    """All contiguous 1- and 2-cut partitions, plus empty-chunk interleaves."""
    n = len(data)
    for i in range(1, n):
        yield [data[:i], data[i:]]
        for j in range(i + 1, n):
            yield [data[:i], data[i:j], data[j:]]
    for i in range(0, n + 1):
        yield [data[:i], b"", data[i:]]
    yield [b"", data]
    yield [data, b""]


CONSISTENCY_CASES = [
    ("identity", b"identity", CONSISTENCY_BODY, lambda b: b),
    ("gzip", b"gzip", _gzip(CONSISTENCY_BODY), gzip.decompress),
    (
        "deflate_zlib",
        b"deflate",
        zlib.compress(CONSISTENCY_BODY, 9),
        zlib.decompress,
    ),
    (
        "deflate_raw",
        b"deflate",
        _raw_deflate(CONSISTENCY_BODY),
        lambda b: zlib.decompress(b, -zlib.MAX_WBITS),
    ),
]


@pytest.mark.parametrize("case", CONSISTENCY_CASES, ids=lambda case: case[0])
def test_chunking_consistency_sync(case):
    name, encoding, wire, reference_decode = case
    reference_bytes = reference_decode(wire)
    reference_text = reference_bytes.decode("utf-8", errors="replace")
    reference_lines = reference_text.splitlines()

    for chunks in _partitions(wire):
        # The same response body must decode identically regardless of how the
        # underlying raw stream chooses to chunk it. Each read interface gets a
        # fresh response, since a streaming response can only be consumed once.
        def fresh_response():
            return httpx.Response(
                200,
                headers=[(b"Content-Encoding", encoding)],
                content=iter(list(chunks)),
            )

        assert b"".join(fresh_response().iter_raw()) == wire
        assert b"".join(fresh_response().iter_bytes()) == reference_bytes
        assert "".join(fresh_response().iter_text()) == reference_text
        assert list(fresh_response().iter_lines()) == reference_lines

        response = fresh_response()
        assert response.read() == reference_bytes
        assert response.text == reference_text


class _AsyncChunkIterator:
    """An async iterator class, avoiding garbage-collected async generators."""

    def __init__(self, chunks: typing.Iterable[bytes]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> "_AsyncChunkIterator":
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class _AsyncChunks(httpx.AsyncByteStream):
    def __init__(self, chunks: typing.Iterable[bytes]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _AsyncChunkIterator:
        return _AsyncChunkIterator(self._chunks)


@pytest.mark.anyio
@pytest.mark.parametrize("case", CONSISTENCY_CASES, ids=lambda case: case[0])
async def test_chunking_consistency_async(case):
    name, encoding, wire, reference_decode = case
    reference_bytes = reference_decode(wire)
    reference_text = reference_bytes.decode("utf-8", errors="replace")
    reference_lines = reference_text.splitlines()

    for chunks in _partitions(wire):
        def fresh_response():
            return httpx.Response(
                200,
                headers=[(b"Content-Encoding", encoding)],
                stream=_AsyncChunks(chunks),
            )

        assert b"".join([part async for part in fresh_response().aiter_raw()]) == wire
        assert b"".join(
            [part async for part in fresh_response().aiter_bytes()]
        ) == reference_bytes
        assert "".join(
            [part async for part in fresh_response().aiter_text()]
        ) == reference_text
        assert [part async for part in fresh_response().aiter_lines()] == (
            reference_lines
        )

        response = fresh_response()
        assert await response.aread() == reference_bytes
        assert response.text == reference_text


@pytest.mark.parametrize(
    ["encoding", "compressed"],
    [
        (b"gzip", _gzip(CONSISTENCY_BODY)),
        (b"deflate", zlib.compress(CONSISTENCY_BODY, 9)),
        (b"deflate", _raw_deflate(CONSISTENCY_BODY)),
    ],
)
@pytest.mark.parametrize("cut", [1, 3, 7])
def test_truncated_compressed_stream_raises(encoding, compressed, cut):
    # The stdlib rejects truncated streams; httpx must do the same at flush,
    # regardless of how the remaining bytes are partitioned.
    truncated = compressed[:-cut]
    for chunks in _partitions(truncated):
        response = httpx.Response(
            200,
            headers=[(b"Content-Encoding", encoding)],
            content=iter(chunks),
        )
        with pytest.raises(httpx.DecodingError):
            response.read()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ["encoding", "compressed"],
    [
        (b"gzip", _gzip(CONSISTENCY_BODY)),
        (b"deflate", zlib.compress(CONSISTENCY_BODY, 9)),
    ],
)
async def test_truncated_compressed_stream_raises_async(encoding, compressed):
    truncated = compressed[:-3]
    response = httpx.Response(
        200,
        headers=[(b"Content-Encoding", encoding)],
        stream=_AsyncChunks([truncated[i - 1 : i] for i in range(1, len(truncated))]),
    )
    with pytest.raises(httpx.DecodingError):
        await response.aread()


def test_deflate_fallback_when_header_error_is_delayed():
    # Raw deflate streams can have a first byte that passes the zlib header
    # check; the failure only surfaces when a later chunk is fed in. The
    # fallback must still kick in regardless of chunk boundaries.
    body = CONSISTENCY_BODY
    compressed = _raw_deflate(body)
    first = zlib.decompressobj()
    try:
        first.decompress(compressed[:1])
        delayed = True
    except zlib.error:
        delayed = False

    for chunks in _partitions(compressed):
        response = httpx.Response(
            200,
            headers=[(b"Content-Encoding", b"deflate")],
            content=iter(list(chunks)),
        )
        assert response.read() == body

    assert delayed, "test fixture no longer exercises the delayed-error path"


def test_empty_stream_with_compression_headers():
    # An empty body is not a truncated stream and must remain acceptable.
    for encoding in (b"gzip", b"deflate"):
        response = httpx.Response(
            200, headers=[(b"Content-Encoding", encoding)], content=b""
        )
        assert response.read() == b""


def test_illegal_utf8_bytes_are_replaced_consistently():
    body = b"a\xff\xfe b"
    reference = body.decode("utf-8", errors="replace")
    for chunks in _partitions(body):
        response = httpx.Response(200, content=iter(list(chunks)))
        assert "".join(response.iter_text()) == reference

        response = httpx.Response(200, content=iter(list(chunks)))
        assert list(response.iter_lines()) == reference.splitlines()
