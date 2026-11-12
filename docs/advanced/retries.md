Retries are **disabled by default**. You can enable them by configuring an
explicit `httpx.Retry` policy on a client.

```python
import httpx

retry = httpx.Retry(max_attempts=3, total=10.0)
client = httpx.Client(retry=retry)
```

A retry policy bounds a single *logical* request. Redirects that are followed,
and requests issued as part of an authentication flow, all draw from the same
attempt count and the same deadline. They do not each restart their own budget.

**Parameters**

* `max_attempts` - The maximum number of times a request may be sent, including
  the initial attempt, redirects and authentication flow requests.
* `total` - The maximum number of seconds a logical request may take,
  including time spent waiting between retries.
* `idempotent_methods` - The methods that may be retried. Defaults to
  `{"GET", "HEAD"}`.
* `retry_status_codes` - The response status codes that trigger a retry.
  Defaults to `(502, 503, 504)`. Set to `None` to disable status-code retries.
* `retry_connect_errors` - Whether failures that occur while establishing a
  connection may be retried. Defaults to `True`.
* `retry_read_errors` - Whether failures that occur while reading the response
  after the request was sent may be retried. Defaults to `True`.
* `backoff_factor`, `backoff_max` - Configure the exponential wait between
  attempts. Defaults to `0.5` seconds, capped at `60` seconds.
* `respect_retry_after` - Whether to wait for the duration indicated by a
  `Retry-After` response header. Defaults to `True`.
* `clock`, `sleep`, `async_sleep` - Hooks for measuring the deadline and
  waiting between attempts. These are useful for testing.

**Idempotency**

Only `GET` and `HEAD` are retried by default. A request with any other method
is retried only if you explicitly mark it as idempotent, either per request:

```python
client.request("DELETE", url, idempotent=True)
```

Or by configuring `idempotent_methods` on the policy.

Requests with a streaming body, such as content given as a generator or a
file object, cannot be resent from the start. If such a request needs to be
retried, `httpx.NonReplayableRequest` is raised rather than sending a partial
body.

**What is retried**

Failed connection attempts, post-send read failures, and the configured status
codes are retried. Failures that may have partially sent the request, such as
write errors or protocol errors, are not retried. Response streams are closed
before a retry is scheduled.

When the attempt budget is spent, a request that keeps returning a retryable
status code returns that final response. A request that keeps failing at the
transport level raises `httpx.RetryExhausted`, with the final transport error
as the exception cause. If waiting for the next attempt would exceed the total
deadline, `httpx.RetryTimeout` is raised instead.

**A note on duplicate writes**

An HTTP client cannot know whether a server processed a request before the
connection or response failed. Retrying is therefore only safe for idempotent
requests, and this policy does not attempt to eliminate duplicate writes. For
requests that are not naturally idempotent, use an idempotency key or another
form of server-side deduplication, and decide for yourself whether a retry is
safe.

Exceptions raised by event hooks, and cancellation of an async request, always
propagate unchanged. They are never treated as retryable failures.
