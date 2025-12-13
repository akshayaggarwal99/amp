"""
AMP Local LLM Integration using Ollama.

This module provides a lightweight abstraction for local LLM operations
without requiring cloud API dependencies.
"""
import requests
import json
import re
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str  # person, place, thing, concept
    description: Optional[str] = None


class LocalLLM:
    """
    Ollama-based local LLM for entity extraction and temporal parsing.
    
    Usage:
        llm = LocalLLM()
        entities = llm.extract_entities("Alice met Bob at the coffee shop.")
        date = llm.extract_date("We went hiking last Tuesday.")
    """
    
    def __init__(
        self, 
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        self._api_url = f"{base_url}/api/generate"
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion for a prompt."""
        try:
            resp = requests.post(self._api_url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens}
            }, timeout=60)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        
        Returns list of Entity objects with name, type, and optional description.
        """
        prompt = f"""Extract all named entities (people, places, things) from this text.
Return ONLY a JSON array like: [{{"name": "Alice", "type": "person"}}, {{"name": "Paris", "type": "place"}}]
If no entities found, return: []

Text: {text}

JSON:"""
        
        result = self.complete(prompt)
        
        # Parse JSON from response
        try:
            # Find JSON array in response
            match = re.search(r'\[.*?\]', result, re.DOTALL)
            if match:
                entities_data = json.loads(match.group())
                return [Entity(name=e.get("name", ""), type=e.get("type", "thing")) 
                        for e in entities_data if e.get("name")]
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return []
    
    def extract_date(self, text: str, reference_date: Optional[datetime] = None) -> Optional[str]:
        """
        Extract and normalize date from text.
        
        Converts relative dates ("last Tuesday") to ISO format if possible.
        Returns None if no date found.
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        ref_str = reference_date.strftime("%Y-%m-%d")
        
        prompt = f"""Given today is {ref_str}, extract the date mentioned in this text.
Convert relative dates (like "last week", "yesterday", "next month") to ISO format (YYYY-MM-DD).
If no date is mentioned or you cannot determine the date, respond with: NONE

Text: {text}

Date (YYYY-MM-DD or NONE):"""
        
        result = self.complete(prompt, max_tokens=50)
        
        # Parse date from response
        match = re.search(r'\d{4}-\d{2}-\d{2}', result)
        if match:
            return match.group()
        
        return None
    
    def summarize(self, texts: List[str], max_length: int = 100) -> str:
        """
        Summarize multiple text items into a single coherent summary.
        
        Useful for session summarization during consolidation.
        """
        combined = "\n".join([f"- {t}" for t in texts])
        
        prompt = f"""Summarize these notes into a single, concise paragraph (max {max_length} words):

{combined}

Summary:"""
        
        return self.complete(prompt, max_tokens=max_length * 2)
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except:
            return False


# Singleton instance for convenience
_default_llm: Optional[LocalLLM] = None

def get_llm() -> LocalLLM:
    """Get or create the default LocalLLM instance."""
    global _default_llm
    if _default_llm is None:
        _default_llm = LocalLLM()
    return _default_llm
