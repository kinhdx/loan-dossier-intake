"""Model access: one structured-output call, with token usage, cost and latency recorded for every call.

The pipeline depends on the `ModelClient` protocol, not on the SDK, so tests run with a scripted client
and no network access.
"""
from __future__ import annotations

import json
import time
from typing import Protocol

from loan_intake.schemas import Usage

# USD per million tokens: input, output, cache read. Cache writes are billed at 1.25x input.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
# Models that accept the server-side fallback option for declined requests.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


class ModelError(Exception):
    pass


class ModelClient(Protocol):
    def complete(self, *, model: str, system: str, content: list[dict], schema: dict, step: str,
                 effort: str = "low") -> tuple[dict, Usage]: ...


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    price_in, price_out, price_read = PRICES.get(model, PRICES["claude-opus-5-5"])
    return (input_tokens * price_in + output_tokens * price_out + cache_read * price_read
            + cache_write * price_in * 1.25) / 1_000_000


class CachingClient:
    """Stores each response under a hash of the request in `folder`, and serves it again for an identical
    request. Re-running an evaluation after a rule change then costs nothing: only requests that changed
    (prompt, schema, document, model) reach the API."""

    def __init__(self, inner: ModelClient, folder):
        self.inner, self.folder = inner, folder

    def complete(self, *, model: str, system: str, content: list[dict], schema: dict, step: str,
                 effort: str = "low") -> tuple[dict, Usage]:
        import hashlib
        from dataclasses import asdict

        payload = json.dumps([model, system, content, schema, effort], sort_keys=True, ensure_ascii=False)
        path = self.folder / model / f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return saved["output"], Usage(**saved["usage"])
        output, usage = self.inner.complete(model=model, system=system, content=content, schema=schema, step=step,
                                            effort=effort)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"output": output, "usage": asdict(usage)}, ensure_ascii=False), encoding="utf-8")
        return output, usage


class AnthropicClient:
    def __init__(self, max_tokens: int = 4000):
        import anthropic

        self._client = anthropic.Anthropic(timeout=120.0, max_retries=3)
        self._max_tokens = max_tokens

    def complete(self, *, model: str, system: str, content: list[dict], schema: dict, step: str,
                 effort: str = "low") -> tuple[dict, Usage]:
        request = dict(
            model=model,
            max_tokens=self._max_tokens,
            system=system,
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": content}],
        )
        if model in FALLBACK_MODELS:
            request.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        started = time.monotonic()
        response = self._client.beta.messages.create(**request)
        u = response.usage
        read = getattr(u, "cache_read_input_tokens", 0) or 0
        write = getattr(u, "cache_creation_input_tokens", 0) or 0
        usage = Usage(model=response.model, step=step, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                      cache_read_tokens=read, cache_write_tokens=write,
                      cost_usd=cost_usd(response.model, u.input_tokens, u.output_tokens, read, write),
                      seconds=round(time.monotonic() - started, 3))
        if response.stop_reason == "refusal":
            raise ModelError("the model declined the request")
        if response.stop_reason == "max_tokens":
            raise ModelError("the answer was cut off at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise ModelError("no structured answer was returned")
        return json.loads(text), usage
