import sqlite_utils
import json
import numpy as np
from fastembed import TextEmbedding
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid

# --- Models ---
class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 1.0
    embedding: Optional[List[float]] = None

class Storage:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to ~/.amp/brain.db
            home = Path.home()
            amp_dir = home / ".amp"
            amp_dir.mkdir(exist_ok=True)
            db_path = str(amp_dir / "brain.db")
        
        import sqlite3
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.db = sqlite_utils.Database(self.connection)
        self._initialize_schema()
        self.embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def _initialize_schema(self):
        # 1. Long Term Memory (memories)
        if "memories" not in self.db.table_names():
            self.db["memories"].create({
                "id": str,
                "content": str,
                "created_at": str,  # ISO format
                "last_accessed": str,
                "score": float,
                "type": str, # episodic, semantic
                "metadata": str, # JSON string
                "embedding": bytes, # Numpy bytes
            }, pk="id")
            # Enable Full Text Search
            self.db["memories"].enable_fts(["content"], create_triggers=True)
        else:
            # Migration: Ensure embedding column exists
            try:
                self.db["memories"].add_column("embedding", bytes)
            except:
                pass # Column likely exists

        # 2. Short Term Memory (working_memory)
        if "working_memory" not in self.db.table_names():
            self.db["working_memory"].create({
                "id": str,
                "content": str,
                "created_at": str,
                "active": int, # 1=True, 0=False
                "metadata": str,
            }, pk="id")

        # 3. Entities (Knowledge Graph nodes)
        if "entities" not in self.db.table_names():
            self.db["entities"].create({
                "name": str,
                "description": str,
                "relations": str, # JSON
            }, pk="name")
            self.db["entities"].enable_fts(["name", "description"], create_triggers=True)

    def add_to_stm(self, content: str, metadata: Dict[str, Any] = {}) -> str:
        """Adds a thought to Short Term Memory (STM)."""
        item = MemoryItem(content=content, metadata=metadata)
        self.db["working_memory"].insert({
            "id": item.id,
            "content": item.content,
            "created_at": item.created_at.isoformat(),
            "active": 1,
            "metadata":  str(item.metadata) # Simple stringify for now
        })
        return item.id

    def get_stm(self) -> List[Dict]:
        """Returns active Working Memory."""
        return list(self.db["working_memory"].rows_where("active = 1", order_by="created_at desc"))

    def consolidate(self, use_llm: bool = False):
        """
        Moves STM items to LTM.
        
        Args:
            use_llm: If True, extract entities using local LLM (requires Ollama).
        """
        stm_items = self.get_stm()
        if not stm_items:
            return 0

        # Lazy import LLM only if needed
        llm = None
        if use_llm:
            try:
                from amp.core.llm import get_llm
                llm = get_llm()
            except Exception:
                pass

        # Batch embed
        contents = [item["content"] for item in stm_items]
        embeddings = list(self.embed_model.embed(contents))

        for i, item in enumerate(stm_items):
            # Add to LTM
            vec_bytes = np.array(embeddings[i], dtype=np.float32).tobytes()
            
            memory_id = item["id"]
            
            self.db["memories"].insert({
                "id": memory_id,
                "content": item["content"],
                "created_at": item["created_at"],
                "last_accessed": datetime.now(timezone.utc).isoformat(),
                "score": 1.0,
                "type": "episodic",
                "metadata": item["metadata"],
                "embedding": vec_bytes
            })
            
            # Extract entities if LLM is available
            if llm:
                entities = llm.extract_entities(item["content"])
                for entity in entities:
                    # Upsert entity
                    existing = list(self.db["entities"].rows_where("name = ?", [entity.name]))
                    if existing:
                        # Entity exists, could update relations here
                        pass
                    else:
                        self.db["entities"].insert({
                            "name": entity.name,
                            "description": entity.type,
                            "relations": "{}"
                        })
                    
                    # Link memory to entity (via metadata for now)
                    # Future: create memory_entities join table
            
            # Mark inactive in STM (or delete)
            self.db["working_memory"].delete(item["id"])
        
        return len(stm_items)

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Searches LTM using Hybrid Search (Vector + FTS fallback)."""
        # Vector Search
        query_vec = list(self.embed_model.embed([query]))[0]
        
        # 1. Fetch all embeddings (Naive Scan for MVP - okay for <10k items)
        # TODO: Use sqlite-vec or faiss for scale
        all_memories = list(self.db["memories"].rows)
        if not all_memories:
            return []

        scores = []
        for mem in all_memories:
            if not mem["embedding"]:
                continue
            vec = np.frombuffer(mem["embedding"], dtype=np.float32)
            score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            scores.append((score, mem))
        
        # Sort by score
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:limit]
        
        return [
            {**m, "score": float(s)} for s, m in top_k
        ]

    def forget(self, memory_ids: List[str]):
        """Hard delete items."""
        for mid in memory_ids:
            try:
                self.db["memories"].delete(mid)
            except:
                pass 
