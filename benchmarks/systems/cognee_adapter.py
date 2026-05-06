"""
Cognee System Adapter for Benchmarking.
Uses LOCAL Ollama via configuration.
"""
import asyncio
import os
import shutil
from pathlib import Path
from typing import List, Dict
import time

class CogneeAdapter:
    name = "Cognee"

    def __init__(self):
        self.ingest_time = 0.0
        # Configure Cognee for Ollama
        # Cognee usually expects env vars or config.
        # Assuming defaults or simple setup. 
        # Real setup might require more intricate config overrides.
        os.environ["LLM_API_KEY"] = "ollama" 
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["LLM_MODEL"] = "gemma3:4b"
        os.environ["OLLAMA_MODEL"] = "gemma3:4b"
        os.environ["LLM_ENDPOINT"] = "http://localhost:11434/v1"
        os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
        os.environ["OPENAI_API_KEY"] = "ollama"
        # Configure Embeddings
        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        os.environ["EMBEDDING_MODEL"] = "nomic-embed-text" 
        os.environ["EMBEDDING_ENDPOINT"] = "http://localhost:11434/api/embeddings"
        os.environ["EMBEDDING_DIMENSIONS"] = "768"
        os.environ["HUGGINGFACE_TOKENIZER"] = "bert-base-uncased" # Dummy to pass validation

    def reset(self):
        pass # Cognee persistence is tricky to clear programmatic ally without CLI

    def ingest(self, sessions: List[List[Dict]]):
        try:
            import cognee
            start = time.time()
            
            async def _ingest():
                # Prune old data first
                try:
                    await cognee.prune.prune_data()
                    await cognee.prune.prune_system(metadata=True)
                except: pass

                for session in sessions:
                    for turn in session:
                        await cognee.add(turn["text"])
                
                await cognee.cognify()
            
            asyncio.run(_ingest())
            self.ingest_time = time.time() - start
        except Exception as e:
            print(f"Cognee Ingest Error: {e}")

    def search(self, query: str, limit: int = 10) -> str:
        try:
            import cognee
            async def _search():
                return await cognee.search(query)
            
            results = asyncio.run(_search())
            return "\n".join([str(r) for r in results[:limit]])
        except Exception as e:
            print(f"Cognee Search Error: {e}")
            return ""

    def cleanup(self):
        pass
