from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from core.agents.prompt_manager import PromptManager


class ContextManager:
    """Builds structured context blocks for tutoring."""

    def __init__(self, inference_service):
        self._inference = inference_service
        self._extract_response = inference_service.extract_response_payload

    async def build_structured_context(self, raw_text: str) -> str:
        if not raw_text.strip():
            return ""
        prompt = PromptManager.env_summary_prompt(raw_text)
        response = await asyncio.to_thread(
            self._inference.chat_completions,
            [{'role': 'user', 'content': prompt}],
        )
        extracted, _ = self._extract_response(response)
        return extracted or ""

    async def build_global_chunk_summary(
        self,
        raw_text: str,
        page_start: int,
        page_end: int,
    ) -> str:
        if not raw_text.strip():
            return ""
        prompt = PromptManager.global_summary_prompt(raw_text, page_start, page_end)
        response = await asyncio.to_thread(
            self._inference.chat_completions,
            [{'role': 'user', 'content': prompt}],
        )
        extracted, _ = self._extract_response(response)
        return extracted or ""
