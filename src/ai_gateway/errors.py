"""Error hierarchy.

Two families:
- `ProviderError`: something went wrong talking to an upstream provider. Carries a `reason`
  (used by fallback policies) and `retryable` (used by the retry loop).
- `GatewayError` subclasses raised to the client: auth, rate limits, bad requests, no route.
"""

from __future__ import annotations

import email.utils
import time
from typing import Any

# Reasons a provider call can fail. Fallback policies (`fallback_on`) match on these strings.
TIMEOUT = "timeout"
HTTP_5XX = "http_5xx"
HTTP_429 = "http_429"
CONNECTION = "connection"
CIRCUIT_OPEN = "circuit_open"
CONTEXT_LENGTH = "context_length"
PROVIDER_AUTH = "provider_auth"
INVALID_REQUEST = "invalid_request"
REFUSAL = "refusal"
STREAM_INTERRUPTED = "stream_interrupted"
MALFORMED_RESPONSE = "malformed_response"


class GatewayError(Exception):
    status_code: int = 500
    error_type: str = "gateway_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def headers(self) -> dict[str, str]:
        return {}

    def to_openai(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.error_type, "code": None}}


class ProviderError(GatewayError):
    status_code = 502
    error_type = "provider_error"

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        reason: str,
        retryable: bool,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.reason = reason
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after


class AuthenticationError(GatewayError):
    status_code = 401
    error_type = "invalid_api_key"


class PermissionDeniedError(GatewayError):
    status_code = 403
    error_type = "permission_denied"


class InvalidRequestError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"


class ModelNotFoundError(GatewayError):
    status_code = 404
    error_type = "model_not_found"


class RateLimitedError(GatewayError):
    status_code = 429
    error_type = "rate_limit_exceeded"

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after

    def headers(self) -> dict[str, str]:
        if self.retry_after is None:
            return {}
        return {"Retry-After": str(max(1, round(self.retry_after)))}


class BudgetExceededError(RateLimitedError):
    error_type = "budget_exceeded"


class AllTargetsFailedError(GatewayError):
    status_code = 502
    error_type = "all_targets_failed"

    def __init__(self, alias: str, errors: list[ProviderError]) -> None:
        summary = "; ".join(f"{e.provider}[{e.reason}]" for e in errors) or "no eligible targets"
        super().__init__(f"all targets failed for '{alias}': {summary}")
        self.errors = errors
        # If every failure was a client-side problem, surface it as such instead of a 502.
        if errors and all(e.reason == INVALID_REQUEST for e in errors):
            self.status_code = 400
            self.error_type = "invalid_request_error"


def parse_retry_after(value: str | None) -> float | None:
    """`Retry-After` is either delta-seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    parsed = email.utils.parsedate_to_datetime(value)
    if parsed is None:  # pragma: no cover - defensive, parsedate raises on garbage in 3.12+
        return None
    return max(0.0, parsed.timestamp() - time.time())


_CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "prompt is too long",
)


def classify_http_error(
    provider: str, status: int, body: str, headers: dict[str, str] | None = None
) -> ProviderError:
    """Map an upstream HTTP status to a ProviderError with retry/fallback semantics."""
    headers = headers or {}
    snippet = body[:300]
    lowered = body.lower()
    if status == 429:
        return ProviderError(
            provider,
            f"rate limited: {snippet}",
            reason=HTTP_429,
            retryable=True,
            status=status,
            retry_after=parse_retry_after(headers.get("retry-after")),
        )
    if status in (408, 504):
        return ProviderError(
            provider, f"upstream timeout ({status})", reason=TIMEOUT, retryable=True, status=status
        )
    if status >= 500:
        return ProviderError(
            provider,
            f"upstream {status}: {snippet}",
            reason=HTTP_5XX,
            retryable=True,
            status=status,
        )
    if status in (401, 403):
        # The *gateway's* provider credential is wrong. Retrying won't help; another provider might.
        return ProviderError(
            provider,
            f"provider auth failed ({status})",
            reason=PROVIDER_AUTH,
            retryable=False,
            status=status,
        )
    if status in (400, 413, 422) and any(m in lowered for m in _CONTEXT_MARKERS):
        return ProviderError(
            provider,
            f"context too long: {snippet}",
            reason=CONTEXT_LENGTH,
            retryable=False,
            status=status,
        )
    return ProviderError(
        provider,
        f"request rejected ({status}): {snippet}",
        reason=INVALID_REQUEST,
        retryable=False,
        status=status,
    )
