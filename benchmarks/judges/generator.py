"""Ollama-backed answer generator. Same model used across all systems for fair comparison."""
from __future__ import annotations

import re

import requests

ANSWER_PROMPT = """You are a helpful assistant with access to a memory of past conversation. Answer the QUESTION using ONLY the provided CONTEXT. If the answer is not contained in the context, reply: "I don't know."

Keep the answer concise (one sentence when possible).

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


class OllamaGenerator:
    def __init__(
        self,
        model: str = "gemma4:latest",
        host: str = "http://localhost:11434",
        timeout: int = 600,
        num_predict: int = 256,
        disable_thinking: bool = True,
    ):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.num_predict = num_predict
        self.disable_thinking = disable_thinking

    def generate(self, context: str, question: str) -> str:
        if not context or not context.strip():
            context = "(no relevant memory found)"
        prompt = ANSWER_PROMPT.format(context=context, question=question)
        if self.disable_thinking and "qwen3" in self.model.lower():
            prompt += "\n/no_think"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": self.num_predict},
        }
        if self.disable_thinking:
            payload["think"] = False
        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return _strip_think(resp.json().get("response", "")).strip()
        except Exception as e:
            return f"[generation error: {e}]"
