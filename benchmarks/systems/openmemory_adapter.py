"""
OpenMemory System Adapter for Benchmarking.
Uses LOCAL Ollama.
"""
import shutil
from pathlib import Path
from typing import List, Dict
import time

class OpenMemoryAdapter:
    name = "OpenMemory"

    def __init__(self):
        self.om = None
        self._db_path = "openmemory_benchmark.db"
        self.ingest_time = 0.0

    def reset(self):
        # Cleanup
        if Path(self._db_path).exists():
            Path(self._db_path).unlink()
        if Path(f"{self._db_path}-wal").exists():
            Path(f"{self._db_path}-wal").unlink()
        if Path(f"{self._db_path}-shm").exists():
            Path(f"{self._db_path}-shm").unlink()

        try:
            from openmemory import OpenMemory
            # Configure for local Ollama
            self.om = OpenMemory(
                mode="local",
                path=self._db_path,
                tier="deep", 
                embeddings={
                   "provider": "ollama",
                   "config": {"model": "nomic-embed-text"} 
                }
            )
        except Exception as e:
            print(f"OpenMemory Init Error: {e}")
            self.om = None

    def ingest(self, sessions: List[List[Dict]]):
        if not self.om: return
        start = time.time()
        for session in sessions:
            for turn in session:
                try:
                    self.om.add(turn["text"], userId="benchmark_user")
                except Exception as e:
                    print(f"OM Add Error: {e}")
        self.ingest_time = time.time() - start

    def search(self, query: str, limit: int = 10) -> str:
        if not self.om: return ""
        try:
            results = self.om.query(query, filters={"user_id": "benchmark_user"})
            # Results might be list of dicts
            return "\n".join([f"- {r.get('content', str(r))}" for r in results[:limit]])
        except Exception as e:
            print(f"OM Search Error: {e}")
            return ""

    def cleanup(self):
        if Path(self._db_path).exists():
            try:
                Path(self._db_path).unlink()
            except: pass
