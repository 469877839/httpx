# Request recorder

`Response.history` only contains the responses whose redirects were followed
successfully. When a request fails in the middle of authentication, a redirect
chain, or while reading a response body, it does not tell you how far the
request actually got.

A `RequestRecorder` records the *logical* request made by a client as a
sequence of *real* network attempts, including the step that triggered each
one. It is fully in-memory and never reads a request or response stream.

```python
import httpx

recorder = httpx.RequestRecorder()
client = httpx.Client(recorder=recorder, transport=...)
client.get("https://example.org/")

record = recorder.records[0]
for attempt in record.attempts:
    print(attempt.trigger, attempt.method, attempt.url, attempt.outcome)
```

The same `RequestRecorder` instance can be shared between a `Client` and an
`AsyncClient`; sync and async requests produce records with identical field
meanings.

## What is recorded

Each `RequestRecord` corresponds to one `client.send()` call and contains:

* `attempts` - every real attempt to reach the transport, in order.
* `auth_steps` - the continuation steps taken by the authentication flow.

A `RequestAttempt` captures:

* `trigger` - why the attempt was made: `initial`, `auth-continuation`, or
  `redirect`.
* `auth_step` and `redirect_step` - the position inside the authentication
  flow and its redirect chain.
* `method` and `url` - the request actually sent, with URL credentials
  removed.
* `started`, `finished` and `elapsed` - monotonic `perf_counter()` timing.
* `response_status` and `http_version` - when a response was produced.
* `previous_method` and `method_changed` - method changes caused by a
  redirect (or by an authentication continuation).
* `redirect_status` - the status code of the redirect that caused the
  attempt.
* `outcome` - the furthest known boundary of the attempt.
* `exception_type` and `exception_message` - when it failed.

`outcome` is one of:

* `response` - the transport returned a response; the body is not read yet.
* `transport-error` - the transport raised instead of returning a response.
* `hook-error` - an event hook raised after a response was available.
* `read-error` - the response was delivered, but reading the body failed.
* `pending` - the attempt started, but no boundary was observed yet.

`auth_steps` use `AuthStepOutcome`:

* `continued` - the authentication flow asked for another request.
* `ended` - the flow returned without requesting anything else.
* `exception` - the flow raised while handling the response.

A failed step only records what actually happened. If a request event hook
raises before the transport is called, no attempt exists for that call. If a
transport call fails, no later attempt or authentication step is recorded.

## Privacy: headers and bodies

Headers and bodies are sensitive and are not captured by default.

```python
recorder = httpx.RequestRecorder(
    capture_headers=True,
    capture_content=True,
)
```

Even with `capture_headers=True`, credential-bearing headers
(`Authorization`, `Cookie`, `Set-Cookie`, `WWW-Authenticate`, and similar)
have their values replaced by `[redacted]`. Credentials embedded in URLs are
always removed from recorded targets.

With `capture_content=True`, a body is only included if it was already
materialised in memory before the recorder observed it. The recorder never
iterates a stream to obtain content, so a streaming request or response that
has not been read remains `None` in the record.

## Hook ordering

Recording wraps the existing send pipeline, it does not replace it. Request
and response event hooks keep exactly the same order and semantics as without
a recorder; a recorder simply observes the boundaries that already exist.
