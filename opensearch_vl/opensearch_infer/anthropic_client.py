"""Minimal client for the public Anthropic Messages API.

Used to run a Claude model as the *teacher* that generates agent trajectories
(and, later, as an LLM judge). The repository's original Claude runner talks to
an internal HMAC gateway that is not available outside the original authors'
infrastructure; this client talks to ``https://api.anthropic.com`` directly.

Configuration (environment variables only; never put keys in code or in git):

    ANTHROPIC_API_KEY     required
    ANTHROPIC_BASE_URL    optional, default https://api.anthropic.com

Features
* retries with exponential backoff on rate limits and transient server errors
  (honours the ``retry-after`` header when present);
* prompt caching: the system prompt and the end of the conversation are marked
  as cache breakpoints, so the long, repeated prefix (system prompt, tool
  definitions, earlier turns) is billed at the cheaper cache-read price on
  every turn after the first.
"""

from __future__ import annotations

import copy
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class AnthropicClient:
    """Thin wrapper around ``POST /v1/messages``."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 5,
        timeout: float = 180.0,
        use_cache: bool = True,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it in your shell before "
                "running (do not write it into any file that is committed)."
            )
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self.use_cache = use_cache

    @classmethod
    def from_env(cls) -> "AnthropicClient":
        return cls(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
            base_url=os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL).strip()
            or DEFAULT_BASE_URL,
        )

    # ------------------------------------------------------------------
    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a copy of ``messages`` with a cache breakpoint on the last block."""
        prepared = copy.deepcopy(messages)
        if self.use_cache and prepared:
            content = prepared[-1].get("content")
            if isinstance(content, list) and content:
                content[-1]["cache_control"] = {"type": "ephemeral"}
        return prepared

    def build_body(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        system_instruction: Optional[str],
        max_tokens: int,
        temperature: Optional[float],
        stop_sequences: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": int(max_tokens),
            "messages": self._prepare_messages(messages),
        }
        if system_instruction:
            if self.use_cache:
                body["system"] = [
                    {
                        "type": "text",
                        "text": system_instruction,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                body["system"] = system_instruction
        if temperature is not None:
            body["temperature"] = float(temperature)
        if stop_sequences:
            body["stop_sequences"] = list(stop_sequences)
        return body

    # ------------------------------------------------------------------
    def call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> requests.Response:
        """POST one request, retrying transient failures. Returns the raw response."""
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        body = self.build_body(
            model, messages, system_instruction, max_tokens, temperature, stop_sequences
        )
        url = f"{self.base_url}/v1/messages"

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url, headers=headers, json=body, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Anthropic request failed (%s), attempt %d", type(exc).__name__, attempt + 1)
                if attempt >= self.max_retries:
                    break
                time.sleep(self._delay(attempt, None))
                continue

            if response.status_code in RETRY_STATUS and attempt < self.max_retries:
                logger.warning(
                    "Anthropic returned HTTP %d, retrying (attempt %d)",
                    response.status_code,
                    attempt + 1,
                )
                time.sleep(self._delay(attempt, response.headers.get("retry-after")))
                continue
            return response

        raise RuntimeError(f"Anthropic request failed after retries: {last_exc}")

    @staticmethod
    def _delay(attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return min(2.0 ** attempt, 30.0) + random.uniform(0, 0.5)
