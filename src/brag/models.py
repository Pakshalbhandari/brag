"""Model functions in the shapes LightRAG / RAG-Anything call them with.

* LLM + vision: Claude via the Anthropic SDK
* Embeddings: a local sentence-transformers model (Anthropic has no embeddings API)
"""

import asyncio
import base64
import logging
from functools import lru_cache
from typing import Any

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


class ClaudeModels:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = anthropic.AsyncAnthropic()

    async def _create(self, messages: list[dict[str, Any]], system: str | None) -> str:
        kwargs: dict[str, Any] = {}
        if system:
            kwargs["system"] = system
        response = await self.client.beta.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            output_config={"effort": self.settings.llm_effort},
            # Re-run on a fallback model if a safety classifier declines the request.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
            **kwargs,
        )
        if response.stop_reason == "refusal":
            logger.warning("Claude declined a request (%s)", response.stop_details)
        elif response.stop_reason == "max_tokens":
            logger.warning("Claude response truncated at max_tokens=%d", self.settings.llm_max_tokens)
        return "".join(b.text for b in response.content if b.type == "text")

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
