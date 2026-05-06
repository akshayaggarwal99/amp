"""Mem0 System Adapter for Benchmarking.

Configurable LLM provider so the runner can switch between local Ollama
(default) and cloud Gemini for ingest-speed comparable to other systems.
"""
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List


class Mem0Adapter:
    """Adapter for Mem0 memory system."""

    name = "Mem0"

    def __init__(
        self,
        llm_provider: str = "gemini",
        llm_model: str = "gemini-2.5-flash",
        embed_model: str = "nomic-embed-text",
        embed_dims: int = 768,
    ):
        self.memory = None
        self.user_id = "benchmark_user"
        self.ingest_time = 0.0
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.embed_model_name = embed_model
        self.embed_dims = embed_dims
        self._db_path = Path("mem0_benchmark_brain")
    
    def reset(self):
        """Reset the memory store."""
        # Clean up previous brain
        if self._db_path.exists():
            shutil.rmtree(self._db_path)
        
        try:
            from mem0 import Memory
            
            # Build provider config — Gemini for cloud ingest speed, Ollama
            # for fully-local. Embedder stays local in both cases.
            if self.llm_provider == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError("GEMINI_API_KEY not set; cannot use gemini provider")
                llm_cfg = {
                    "provider": "gemini",
                    "config": {
                        "model": self.llm_model,
                        "temperature": 0,
                        "api_key": api_key,
                    },
                }
            else:
                llm_cfg = {
                    "provider": "ollama",
                    "config": {
                        "model": self.llm_model,
                        "temperature": 0,
                    },
                }

            config = {
                "llm": llm_cfg,
                "embedder": {
                    "provider": "ollama",
                    "config": {"model": self.embed_model_name},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": "mem0_benchmark_qs",
                        "embedding_model_dims": self.embed_dims,
                    },
                },
                "history_db_path": ":memory:",
            }
            
            self.memory = Memory.from_config(config)
            # Clear existing data
            try:
                self.memory.delete_all(user_id=self.user_id)
            except:
                pass
        except Exception as e:
            print(f"Mem0 Init Error: {e}")
            self.memory = None
    
    def ingest(self, sessions: List[List[Dict]]):
        """Ingest conversation sessions."""
        if not self.memory:
            print("Mem0 not initialized")
            return
        
        start = time.time()
        
        for session in sessions:
            messages = []
            for turn in session:
                role = "user" if len(messages) % 2 == 0 else "assistant"
                messages.append({"role": role, "content": turn["text"]})
            
            try:
                self.memory.add(messages, user_id=self.user_id)
            except Exception as e:
                print(f"Mem0 Ingest Error: {e}")
        
        self.ingest_time = time.time() - start
    
    def search(self, query: str, limit: int = 10) -> str:
        """Search and return context string.

        Mem0 v1.0+ returns a dict ``{'results': [{'memory': ..., ...}, ...]}``,
        though older versions returned a flat list of dicts or strings. We
        normalise all three shapes so the adapter is robust to version drift.
        """
        if not self.memory:
            return ""

        try:
            results = self.memory.search(query, user_id=self.user_id, limit=limit)
            if isinstance(results, dict):
                results = results.get("results", [])

            context_parts = []
            for r in results:
                if isinstance(r, dict):
                    text = r.get("memory") or r.get("text") or str(r)
                    context_parts.append(f"- {text}")
                else:
                    context_parts.append(f"- {str(r)}")
            return "\n".join(context_parts)
        except Exception as e:
            print(f"Mem0 Search Error: {e}")
            return ""
    
    def cleanup(self):
        """Cleanup resources."""
        if self._db_path.exists():
            shutil.rmtree(self._db_path)
        self.memory = None
