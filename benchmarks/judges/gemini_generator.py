"""Gemini API generator. Used as a controlled fast generator across all 4 systems
so the only variable in the comparison is the memory system itself.

Reads GEMINI_API_KEY from environment (or .env via python-dotenv if installed).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import requests

ANSWER_PROMPT = """You are a helpful assistant with access to a memory of past conversation. Answer the QUESTION using ONLY the provided CONTEXT. If the answer is not contained in the context, reply: "I don't know."

Keep the answer concise (one sentence when possible).

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def _load_env_file(path: str = ".env") -> None:
    """Lightweight .env loader — populates os.environ if a .env file exists."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env_file()


class GeminiGenerator:
    """Synchronous client for Gemini's REST endpoint with a small retry loop.

    Concurrency: instances are thread-safe so the runner can drive parallel
    requests via a thread pool.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        rate_limit_qps: float = 9.0,  # 1000 RPM ~ 16 qps; stay below
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set (env or .env)")
        self.timeout = timeout
        self.max_retries = max_retries
        self._url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._min_interval = 1.0 / rate_limit_qps if rate_limit_qps > 0 else 0.0

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def generate(self, context: str, question: str) -> str:
        if not context or not context.strip():
            context = "(no relevant memory found)"
        prompt = ANSWER_PROMPT.format(context=context, question=question)

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 256,
                "topP": 0.95,
            },
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

        for attempt in range(self.max_retries):
            try:
                self._throttle()
                resp = requests.post(self._url, headers=headers, json=body, timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                cands = data.get("candidates", [])
                if not cands:
                    return ""
                parts = cands[0].get("content", {}).get("parts", [])
                if not parts:
                    return ""
                return parts[0].get("text", "").strip()
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    return f"[gemini error: {e}]"
                time.sleep(2 ** attempt)
        return ""
