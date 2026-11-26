import asyncio
from types import SimpleNamespace

import pytest

from brag.models import ClaudeModels, TokenUsage, _convert_openai_messages


class FakeMessages:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            stop_details=None,
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=900
            ),
        )


@pytest.fixture
def models(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    m = ClaudeModels(settings)
    m.client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    return m


def test_system_prompt_is_cached(models):
    asyncio.run(models.complete("chunk text", system_prompt="long extraction instructions"))
    call = models.client.beta.messages.calls[0]
    assert call["system"] == [
        {"type": "text", "text": "long extraction instructions", "cache_control": {"type": "ephemeral"}}
    ]
    assert call["fallbacks"] == "default"


def test_no_system_block_without_system_prompt(models):
    asyncio.run(models.complete("hello"))
    assert "system" not in models.client.beta.messages.calls[0]


def test_complete_json_sets_output_format(models):
    asyncio.run(models.complete_json("sys", "prompt", {"type": "object"}))
    config = models.client.beta.messages.calls[0]["output_config"]
    assert config["format"] == {"type": "json_schema", "schema": {"type": "object"}}
    assert config["effort"] == models.settings.llm_effort


def test_usage_is_accumulated(models):
    asyncio.run(models.complete("a", system_prompt="s"))
    asyncio.run(models.complete("b", system_prompt="s"))
    assert models.usage.requests == 2 and models.usage.cache_read_tokens == 1800
    assert "1,820 input tokens (99% read from cache)" in models.usage.summary()  # 1800 / 1820


def test_usage_summary_without_requests():
    assert "0%" in TokenUsage().summary()


def test_openai_messages_are_converted():
    system, messages = _convert_openai_messages([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
        ]},
    ])
    assert system == "be brief"
    assert messages[0]["content"][1]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}


def test_explicit_fallback_model_on_claude_api(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    settings.llm_fallback_model = "claude-opus-4-8"
    assert ClaudeModels(settings)._fallback_kwargs() == {
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": [{"model": "claude-opus-4-8"}],
    }


@pytest.mark.parametrize("provider,cls_name", [
    ("bedrock", "AsyncAnthropicBedrockMantle"),
    ("vertex", "AsyncAnthropicVertex"),
    ("foundry", "AsyncAnthropicFoundry"),
])
def test_cloud_providers_use_client_side_fallback(settings, monkeypatch, provider, cls_name):
    import anthropic

    from brag import models as models_module

    created = {}

    class FakeClient:
        def __init__(self, middleware):
            created["middleware"] = middleware

    monkeypatch.setattr(anthropic, cls_name, FakeClient)
    settings.llm_provider, settings.llm_fallback_model = provider, "claude-opus-4-8"
    m = models_module.ClaudeModels(settings)

    assert isinstance(m.client, FakeClient)
    assert [type(x).__name__ for x in created["middleware"]] == ["BetaRefusalFallbackMiddleware"]
    assert m._fallback_kwargs() == {}  # no server-side fallbacks off the Claude API


def test_rerank_func(settings, monkeypatch):
    from brag import models as models_module

    assert models_module.build_rerank_func(settings) is None

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [len(doc) for _, doc in pairs]

    monkeypatch.setattr(models_module, "_load_cross_encoder", lambda name: FakeCrossEncoder())
    settings.rerank_model = "some/cross-encoder"
    rerank = models_module.build_rerank_func(settings)
    ranked = asyncio.run(rerank(query="q", documents=["aa", "a", "aaa"], top_n=2))
    assert ranked == [{"index": 2, "relevance_score": 3.0}, {"index": 0, "relevance_score": 2.0}]
    assert asyncio.run(rerank(query="q", documents=[])) == []
