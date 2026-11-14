# Retries

HTTPX does not retry requests by default. An explicit `RetryPolicy` enables
bounded retries, limiting a single logical request by a unified attempt
budget and an overall deadline.

```python
policy = httpx.RetryPolicy(max_attempts=3, deadline=10.0)

with httpx.Client(retry=policy) as client:
    response = client.get("https://www.example.com/")
```

The policy can also be enabled or overridden per request, using the
`"retry"` request extension:

```python
# Retry just this request...
response = client.get(url, extensions={"retry": httpx.RetryPolicy()})

# ...or disable retries for this request on a client that has a policy.
response = client.get(url, extensions={"retry": None})
```

## A unified budget

A single logical request may involve several wire-level sends: authentication
re-sends, redirect hops, and retries. The `max_attempts` budget and the
`deadline` are shared across all of them, so none of those layers restarts
its own count from zero. If following redirects or completing an
authentication flow would exceed the budget, `httpx.RetryBudgetExceeded`
is raised.

When the budget is exhausted while retrying, the last failure stands: the
final response is returned for retryable status codes, and the final
exception is re-raised for transport errors.

## What is retried

* **Connection failures** - `ConnectError` and `ConnectTimeout`, controlled
by `retry_on_connect_error`. The server cannot have seen these requests.
* **Read failures** - `ReadError` and `ReadTimeout`, controlled by
`retry_on_read_error`. These fail *after* the request was sent, so the server
may already have processed it.
* **Status codes** - Responses with a status in `retry_status_codes`,
defaulting to `(429, 502, 503, 504)`. `Retry-After` response headers are
honoured unless `respect_retry_after=False` is used. The response stream is
always closed before the next attempt.

Retries are only permitted for replayable `GET`/`HEAD` requests by default.
Other methods require an explicit idempotency permit, either by listing them
in `retry_methods`, or per request:

```python
client.post(url, content=b"...", extensions={"idempotent": True})
```

Requests with a streaming body that cannot be replayed are never resent;
attempting to retry one raises `httpx.RequestNotReplayable`.

!!! warning
    HTTPX does not deduplicate requests on the server side. A request that
    failed after being sent may already have been processed, and retrying it
    can result in duplicate writes. Deciding whether a retry is safe is left
    to the caller.

## Backoff and deadlines

The delay between attempts starts at `backoff` seconds and doubles after
each retry, capped by `max_backoff`. A `Retry-After` header may increase
the delay further. If sleeping would push the logical request past its
`deadline`, no further attempt is made.

For testing, the `clock`, `sleep`, and `async_sleep` callables can be
replaced with fakes, allowing deterministic control over the deadline and
the observed delays.
