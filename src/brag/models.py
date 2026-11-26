"""Model functions in the shapes LightRAG / RAG-Anything call them with.

* LLM + vision: Claude via the Anthropic SDK, on the Claude API, Amazon Bedrock,
  Google Vertex AI or Microsoft Foundry
* Embeddings: a local sentence-transformers model (Anthropic has no embeddings API)
* Reranking (optional): a local sentence-transformers cross-encoder
"""

import asyncio
import base64
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Awaitable, Callable

import anthropic
import numpy as np
from lightrag.utils import EmbeddingFunc

from .config import Settings

logger = logging.getLogger(__name__)

_MAGIC = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF8": "image/gif",
    b"RIFF": "image/webp",
}


def _media_type(b64: str) -> str:
    head = base64.b64decode(b64[:16] + "=" * (-len(b64[:16]) % 4))
    return next((mt for magic, mt in _MAGIC.items() if head.startswith(magic)), "image/png")


def _image_block(b64: str, media_type: str | None = None) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type or _media_type(b64), "data": b64},
    }


def _convert_openai_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """RAG-Anything builds OpenAI-style VLM messages; translate them to the Messages API."""
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for msg in messages:
        role, content = msg.get("role"), msg.get("content")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else "")
            continue
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        blocks = []
        for part in content or []:
            if part.get("type") == "text":
                blocks.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("data:"):
                    header, data = url.split(",", 1)
                    blocks.append(_image_block(data, header[5:].split(";")[0]))
                else:
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        converted.append({"role": role, "content": blocks})
    return ("\n\n".join(p for p in system_parts if p) or None), converted


@dataclass
class TokenUsage:
    """Running totals across every Claude call, to show what an ingest cost."""

    requests: int = 0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.requests += 1
        self.input_tokens += usage.input_tokens or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", None) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", None) or 0
        self.output_tokens += usage.output_tokens or 0

    def summary(self) -> str:
        total_in = self.input_tokens + self.cache_write_tokens + self.cache_read_tokens
        cached = f"{self.cache_read_tokens / total_in:.0%}" if total_in else "0%"
        return (
            f"{self.requests} Claude requests: {total_in:,} input tokens ({cached} read from cache), "
            f"{self.output_tokens:,} output tokens"
        )


def _system_blocks(system: str) -> list[dict[str, Any]]:
    # LightRAG's entity-extraction system prompt (instructions, entity types,
    # examples) is identical for every chunk; only the user turn changes. Caching
    # it makes every chunk after the first read it at a fraction of the price.
    # Prompts under the model's minimum cacheable length are simply not cached.
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


FALLBACK_BETA = "server-side-fallback-2026-07-01"


def make_client(settings: Settings) -> Any:
    """The async client for the configured platform.

    Server-side refusal fallbacks exist only on the Claude API; on Bedrock,
    Vertex and Foundry the SDK's client-side middleware retries a declined
    request on the fallback model instead.
    """
    if settings.llm_provider == "anthropic":
        return anthropic.AsyncAnthropic()
    middleware = []
    if settings.llm_fallback_model:
        middleware.append(anthropic.BetaRefusalFallbackMiddleware([{"model": settings.llm_fallback_model}]))
    cls = {
        "bedrock": anthropic.AsyncAnthropicBedrockMantle,  # AWS_REGION + standard AWS credentials
        "vertex": anthropic.AsyncAnthropicVertex,  # CLOUD_ML_REGION, ANTHROPIC_VERTEX_PROJECT_ID + ADC
        "foundry": anthropic.AsyncAnthropicFoundry,  # ANTHROPIC_FOUNDRY_RESOURCE + ANTHROPIC_FOUNDRY_API_KEY
    }[settings.llm_provider]
    return cls(middleware=middleware)


class ClaudeModels:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = make_client(settings)
        self.usage = TokenUsage()

    def _fallback_kwargs(self) -> dict[str, Any]:
        if self.settings.llm_provider != "anthropic":
            return {}  # handled by the client middleware
        fallbacks: Any = (
            [{"model": self.settings.llm_fallback_model}] if self.settings.llm_fallback_model else "default"
        )
        return {"betas": [FALLBACK_BETA], "fallbacks": fallbacks}

    async def _create(
        self, messages: list[dict[str, Any]], system: str | None, output_format: dict[str, Any] | None = None
    ) -> str:
        kwargs: dict[str, Any] = {}
        if system:
            kwargs["system"] = _system_blocks(system)
        output_config: dict[str, Any] = {"effort": self.settings.llm_effort}
        if output_format:
            output_config["format"] = output_format
        response = await self.client.beta.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            output_config=output_config,
            messages=messages,
            # Re-run on a fallback model if a safety classifier declines the request.
            **self._fallback_kwargs(),
            **kwargs,
        )
        self.usage.add(response.usage)
        if response.stop_reason == "refusal":
            logger.warning("Claude declined a request (%s)", response.stop_details)
        elif response.stop_reason == "max_tokens":
            logger.warning("Claude response truncated at max_tokens=%d", self.settings.llm_max_tokens)
        return "".join(b.text for b in response.content if b.type == "text")

    async def complete_json(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        """One request whose reply is JSON matching ``schema`` (structured outputs)."""
        return await self._create(
            [{"role": "user", "content": prompt}], system, {"type": "json_schema", "schema": schema}
        )

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **_: Any,  # LightRAG passes hashing_kv, keyword_extraction, etc.
    ) -> str:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in history_messages or []
            if m.get("role") in ("user", "assistant")
        ]
        messages.append({"role": "user", "content": prompt})
        return await self._create(messages, system_prompt)

    async def vision(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        image_data: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        if messages:
            system, converted = _convert_openai_messages(messages)
            return await self._create(converted, system or system_prompt)
        if image_data:
            content = [_image_block(image_data), {"type": "text", "text": prompt}]
            return await self._create([{"role": "user", "content": content}], system_prompt)
        return await self.complete(prompt, system_prompt, history_messages, **kwargs)


@lru_cache(maxsize=2)
def _load_sentence_transformer(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def build_embedding_func(settings: Settings) -> EmbeddingFunc:
    model = _load_sentence_transformer(settings.embedding_model)

    async def embed(texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(model.encode, texts, normalize_embeddings=True, convert_to_numpy=True)

    return EmbeddingFunc(
        embedding_dim=model.get_sentence_embedding_dimension(),
        max_token_size=model.max_seq_length,
        func=embed,
        model_name=settings.embedding_model,
    )


@lru_cache(maxsize=2)
def _load_cross_encoder(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


def build_rerank_func(settings: Settings) -> Callable[..., Awaitable[list[dict[str, Any]]]] | None:
    """LightRAG rerank function backed by a local cross-encoder, or None when not configured."""
    if not settings.rerank_model:
        return None
    model = _load_cross_encoder(settings.rerank_model)

    async def rerank(query: str, documents: list[str], top_n: int | None = None, **_: Any) -> list[dict[str, Any]]:
        if not documents:
            return []
        scores = await asyncio.to_thread(model.predict, [(query, d) for d in documents])
        ranked = sorted(range(len(documents)), key=lambda i: float(scores[i]), reverse=True)
        return [{"index": i, "relevance_score": float(scores[i])} for i in ranked[:top_n]]

    return rerank
