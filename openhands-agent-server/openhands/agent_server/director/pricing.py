"""What a model's use costs where the model did not say.

A model reached through a gateway or under a name of its own often reports
tokens and no cost. LiteLLM's public price list knows most models by their
plain names, so the use is priced as the model its name ends in.
"""

from litellm import cost_per_token

from openhands.sdk.llm.utils.metrics import TokenUsage


def _names(model: str) -> list[str]:
    """``model`` and what is left of it without its leading provider
    names: ``litellm_proxy/anthropic/claude-x``, ``anthropic/claude-x``,
    ``claude-x``."""
    parts = model.split("/")
    return ["/".join(parts[start:]) for start in range(len(parts))]


def priced(model: str, tokens: TokenUsage) -> float | None:
    """What ``tokens`` of ``model`` cost in US dollars by the public price
    list, with what was read from and written to the cache at its own price.

    Returns:
        None when the list has no price for the model under any of its names.
    """
    for name in _names(model):
        try:
            prompt, completion = cost_per_token(
                model=name,
                prompt_tokens=tokens.prompt_tokens,
                completion_tokens=tokens.completion_tokens,
                cache_read_input_tokens=tokens.cache_read_tokens,
                cache_creation_input_tokens=tokens.cache_write_tokens,
            )
        except Exception:
            # The list does not know the model under this name.
            continue
        if prompt + completion > 0:
            return prompt + completion
    return None
