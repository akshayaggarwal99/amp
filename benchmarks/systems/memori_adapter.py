"""
Memori System Adapter for Benchmarking.
Uses LOCAL Ollama via OpenAI-compatible client.
WARNING: Memori likely relies on Cloud Augmentation. This might fail or be rate-limited locally.
"""
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict

class MemoriAdapter:
    name = "Memori"

    def __init__(self):
        self.mem = None
        self.client = None
        self.ingest_time = 0.0
        self._db_path = "memori_benchmark.db"

    def reset(self):
        if Path(self._db_path).exists():
            try: Path(self._db_path).unlink()
            except: pass
            
        try:
            from memori import Memori
            from openai import OpenAI
            import sqlite3
            
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            
            # Use factory for sqlite connection
            self.mem = Memori(conn=lambda: sqlite3.connect(self._db_path))
            self.mem.llm.register(self.client)
            self.mem.attribution(entity_id="benchmark", process_id="bench_proc")
            self.mem.config.storage.build() # Ensure DB tables exist
            
        except Exception as e:
            print(f"Memori Init Error: {e}")
            self.mem = None

    def ingest(self, sessions: List[List[Dict]]):
        if not self.mem: return
        start = time.time()
        
        try:
            for session in sessions:
                self.mem.new_session()
                # Create a conversation history
                messages = []
                for turn in session:
                    # We accept user/system roles if provided, else alternate
                    role = turn.get("role", "user")
                    content = turn["text"]
                    messages.append({"role": role, "content": content})
                
                # Send to LLM (Simulated)
                # We need to trigger Memori. Memori wraps the client.
                # We make a call. We don't care about response, just side effects.
                # We use a cheap "complete" call.
                try:
                    self.client.chat.completions.create(
                        model="gemma3:4b", # Must match local model
                        messages=messages,
                        max_tokens=1 
                    )
                except Exception as e:
                    print(f"Memori Chat Error: {e}")

            # Wait for background augmentation
            if hasattr(self.mem, "augmentation") and hasattr(self.mem.augmentation, "wait"):
                self.mem.augmentation.wait()
                
        except Exception as e:
            print(f"Memori Ingest Error: {e}")
            
        self.ingest_time = time.time() - start

    def search(self, query: str, limit: int = 10) -> str:
        if not self.mem: return ""
        try:
            # Memori recall
            res = self.mem.recall(query, limit=limit)
            # Res is likely a list of objects or dicts
            return str(res)
        except Exception as e:
            print(f"Memori Search Error: {e}")
            return ""

    def cleanup(self):
        if Path(self._db_path).exists():
            try: Path(self._db_path).unlink()
            except: pass
