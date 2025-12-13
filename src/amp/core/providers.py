"""
AMP Pluggable Providers - Abstract interfaces for community adoption.

This module defines protocols that allow users to swap LLM, Embedder,
and Storage backends without modifying core AMP logic.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Protocol


@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str  # person, place, thing, organization, concept
    description: Optional[str] = None


class LLMProvider(Protocol):
    """
    Abstract interface for LLM providers.
    
    Implementations:
        - OllamaProvider (default, local)
        - OpenAIProvider
        - AnthropicProvider
        - LiteLLMProvider (universal)
    """
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion for a prompt."""
        ...
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text."""
        ...
    
    def extract_date(self, text: str) -> Optional[str]:
        """Extract and normalize date from text to ISO format."""
        ...
    
    def is_available(self) -> bool:
        """Check if the provider is available."""
        ...


class EmbedderProvider(Protocol):
    """
    Abstract interface for embedding providers.
    
    Implementations:
        - FastEmbedProvider (default, local)
        - OpenAIEmbedProvider
        - SentenceTransformersProvider
    """
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        ...
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...


class StorageProvider(Protocol):
    """
    Abstract interface for storage providers.
    
    Implementations:
        - SQLiteProvider (default)
        - PostgresProvider
        - DuckDBProvider
    """
    
    def add_memory(self, memory: Dict[str, Any]) -> str:
        """Add a memory to storage. Returns ID."""
        ...
    
    def search(self, query_embedding: List[float], limit: int) -> List[Dict]:
        """Search memories by embedding similarity."""
        ...
    
    def get_entities(self) -> List[Dict]:
        """Get all entities."""
        ...
    
    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity."""
        ...


# --- Default Implementations ---

class OllamaProvider:
    """
    Default local LLM provider using Ollama.
    
    No API keys required. Runs entirely on local hardware.
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
        import requests
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
            return f"Error: {e}"
    
    def extract_entities(self, text: str) -> List[Entity]:
        import json, re
        prompt = f"""Extract entities (people, places, things) from this text.
Return JSON array: [{{"name": "Alice", "type": "person"}}]
If none, return: []

Text: {text}
JSON:"""
        result = self.complete(prompt)
        try:
            match = re.search(r'\[.*?\]', result, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return [Entity(name=e.get("name", ""), type=e.get("type", "thing"))
                        for e in data if e.get("name")]
        except:
            pass
        return []
    
    def extract_date(self, text: str) -> Optional[str]:
        import re
        from datetime import datetime
        ref = datetime.now().strftime("%Y-%m-%d")
        prompt = f"""Today is {ref}. Extract the date from this text as YYYY-MM-DD.
If no date, respond: NONE

Text: {text}
Date:"""
        result = self.complete(prompt, max_tokens=50)
        match = re.search(r'\d{4}-\d{2}-\d{2}', result)
        return match.group() if match else None
    
    def is_available(self) -> bool:
        import requests
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except:
            return False


class FastEmbedProvider:
    """
    Default local embedding provider using fastembed.
    
    No API keys required. Runs entirely on local hardware.
    """
    
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model)
        self._dimension = 384  # bge-small default
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [list(e) for e in self._model.embed(texts)]
    
    @property
    def dimension(self) -> int:
        return self._dimension


# --- Provider Registry ---

_PROVIDERS = {
    "llm": {
        "ollama": OllamaProvider,
        # Add more: "openai": OpenAIProvider, "anthropic": AnthropicProvider
    },
    "embedder": {
        "fastembed": FastEmbedProvider,
        # Add more: "openai": OpenAIEmbedProvider
    }
}

def get_provider(provider_type: str, name: str, config: Dict[str, Any] = None):
    """
    Get a provider instance by type and name.
    
    Args:
        provider_type: "llm" or "embedder"
        name: Provider name (e.g., "ollama", "openai")
        config: Provider-specific configuration
    
    Returns:
        Provider instance
    """
    if provider_type not in _PROVIDERS:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    providers = _PROVIDERS[provider_type]
    if name not in providers:
        raise ValueError(f"Unknown {provider_type} provider: {name}. Available: {list(providers.keys())}")
    
    return providers[name](**(config or {}))
