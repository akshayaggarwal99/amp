"""Ollama-backed binary judge for benchmark scoring.

Returns 1 (CORRECT) or 0 (WRONG) per (question, ground_truth, prediction).

Handles models with thinking modes (Qwen3, DeepSeek R1, Gemma 4) by stripping
<think>...</think> blocks before parsing the verdict.
"""
from __future__ import annotations

import re
from typing import Literal

import requests

JUDGE_PROMPT = """You are an impartial judge. Decide whether the PREDICTION correctly answers the QUESTION given the GROUND TRUTH.

Be generous: paraphrases, partial answers that capture the essential fact, and equivalent phrasings count as CORRECT. Mark WRONG only if the prediction is empty, hallucinated, contradicts the ground truth, or misses the key fact entirely.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
PREDICTION: {prediction}

After any reasoning, finish with exactly one word on its own line: CORRECT or WRONG."""


def _parse_verdict(text: str) -> int:
    """Return 1 for CORRECT, 0 for WRONG. Strips <think> blocks first."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip().upper()
    matches = re.findall(r"\b(CORRECT|WRONG)\b", text)
    if not matches:
        return 0
    # Use the LAST verdict mentioned — handles "is it CORRECT? ... WRONG"
    return 1 if matches[-1] == "CORRECT" else 0


class LocalJudge:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: int = 600,
        num_predict: int = 2048,
        disable_thinking: bool = True,
    ):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.num_predict = num_predict
        self.disable_thinking = disable_thinking

    def _build_prompt(self, question: str, ground_truth: str, prediction: str) -> str:
        prompt = JUDGE_PROMPT.format(
            question=question, ground_truth=ground_truth, prediction=prediction
        )
        # Qwen3 supports /no_think to skip CoT
        if self.disable_thinking and "qwen3" in self.model.lower():
            prompt += "\n/no_think"
        return prompt

    def score(self, question: str, ground_truth: str, prediction: str) -> Literal[0, 1]:
        if not prediction or not prediction.strip():
            return 0
        # Deterministic guard: an "I don't know" / refusal counts as WRONG.
        # LLM judges have a strong leniency bias toward marking these as
        # CORRECT, which inflates accuracy for systems whose retrieval often
        # returns empty/irrelevant context. We override before asking the judge.
        low = prediction.strip().lower()
        if low.startswith((
            "i don", "i cannot", "i can not", "i'm not able",
            "i am not able", "i do not know", "i could not",
            "i couldn't", "no information", "the context does not",
            "the provided context", "based on the provided",
        )) and any(s in low for s in ("don't know", "do not know", "cannot determine",
                                      "not enough information", "not provided", "no information",
                                      "unable to", "context does not", "isn't enough",
                                      "is not enough", "context doesn't")):
            return 0
        # Tighter rule: if it literally starts with "I don't know" we don't
        # need a second clause.
        if low.startswith(("i don't know", "i don't have", "i do not know")):
            return 0
        payload = {
            "model": self.model,
            "prompt": self._build_prompt(question, ground_truth, prediction),
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": self.num_predict},
        }
        # Ollama supports a top-level `think` flag to disable model thinking on
        # capability-tagged models (Gemma 4, Qwen 3, DeepSeek R1, etc.).
        if self.disable_thinking:
            payload["think"] = False
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return _parse_verdict(resp.json().get("response", ""))
