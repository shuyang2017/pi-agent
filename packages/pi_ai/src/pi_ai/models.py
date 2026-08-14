"""Model helpers (Python port of packages/ai/src/models.ts cost calculation)."""

from __future__ import annotations

from typing import Any, Dict

from .types import AssistantMessage, Model, Usage


def calculate_cost(model: Model, usage: Usage) -> None:
    """Fill ``usage.cost`` from ``model.cost`` rates (in $/million tokens).

    Mirrors upstream: components are priced individually and ``total`` is their
    sum. Tiered pricing (``model.cost.tiers``) is applied when present.
    """
    rates = model.cost
    total_input = usage.input
    applied = rates

    if rates.tiers:
        # Highest matching input threshold applies to the full request.
        for tier in sorted(rates.tiers, key=lambda t: t.inputTokensAbove, reverse=True):
            if total_input > tier.inputTokensAbove:
                applied = tier
                break

    def price(tokens: int, rate: float) -> float:
        return (tokens / 1_000_000.0) * rate

    usage.cost = type(usage.cost)(
        input=price(usage.input, applied.input),
        output=price(usage.output, applied.output),
        cacheRead=price(usage.cacheRead, applied.cacheRead),
        cacheWrite=price(usage.cacheWrite, applied.cacheWrite),
    )
    usage.cost.total = (
        usage.cost.input + usage.cost.output + usage.cost.cacheRead + usage.cost.cacheWrite
    )


def default_model(
    model_id: str = "claude-sonnet-4-6",
    provider: str = "anthropic",
    api: str = "anthropic-messages",
    base_url: str = "https://api.anthropic.com/v1",
    max_tokens: int = 8_192,
    context_window: int = 200_000,
) -> Model:
    """Construct a minimal Anthropic model for tests / local runs."""
    return Model(
        id=model_id,
        name=model_id,
        api=api,
        provider=provider,
        baseUrl=base_url,
        reasoning=True,
        cost=type(model_cost_zero())(
            input=3.0, output=15.0, cacheRead=0.30, cacheWrite=3.75
        ),
        contextWindow=context_window,
        maxTokens=max_tokens,
    )


def model_cost_zero() -> Any:
    from .types import ModelCost

    return ModelCost()


__all__ = ["calculate_cost", "default_model"]
