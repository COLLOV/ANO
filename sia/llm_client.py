from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from .config import Settings


logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Settings):
        base_url = (
            settings.openai_base_url
            if settings.llm_mode == "api"
            else settings.vllm_base_url
        )
        api_key = (
            settings.openai_api_key
            if settings.llm_mode == "api"
            else (settings.openai_api_key or "sk-local")
        )

        self.model = (
            settings.model_api if settings.llm_mode == "api" else settings.model_local
        )

        self.client = OpenAI(base_url=base_url, api_key=api_key)

        logger.info(
            "LLM client initialized",
            extra={
                "mode": settings.llm_mode,
                "base_url": base_url,
                "model": self.model,
            },
        )

    def chat_json(
        self,
        system: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        json_schema: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request and expect a strict JSON object.

        Fails loudly if the response cannot be parsed as JSON.
        """
        response_format: Dict[str, Any]
        if json_schema is not None:
            response_format = {"type": "json_schema", "json_schema": json_schema}
        else:
            response_format = {"type": "json_object"}

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, *messages],
            response_format=response_format,
        )

        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("LLM returned invalid JSON", extra={"content": content[:500]})
            raise

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data
