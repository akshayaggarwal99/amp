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
    DEFAULT_STM_THRESHOLD = 50
    DEFAULT_RECENCY_WEIGHT = 0.1
    DEFAULT_RECENCY_HALF_LIFE_DAYS = 7.0

    def __init__(
        self,
        db_path: Optional[str] = None,
        stm_threshold: int = DEFAULT_STM_THRESHOLD,
        recency_weight: float = DEFAULT_RECENCY_WEIGHT,
        recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
    ):
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

        self.stm_threshold = stm_threshold
        self.recency_weight = recency_weight
        self.recency_half_life_days = recency_half_life_days

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

    def add_to_stm(self, content: str, metadata: Dict[str, Any] = {}, auto_consolidate: bool = True) -> str:
        """Adds a thought to Short Term Memory (STM).

        If auto_consolidate is True and the active STM size reaches
        ``stm_threshold``, consolidate is invoked automatically.
        """
        item = MemoryItem(content=content, metadata=metadata)
        self.db["working_memory"].insert({
            "id": item.id,
            "content": item.content,
            "created_at": item.created_at.isoformat(),
            "active": 1,
            "metadata":  str(item.metadata) # Simple stringify for now
        })

        if auto_consolidate and len(self.get_stm()) >= self.stm_threshold:
            self.consolidate()

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

    def search(
        self,
        query: str,
        limit: int = 5,
        use_recency: bool = True,
        neighbor_window: int = 0,
    ) -> List[Dict]:
        """Searches LTM using cosine similarity, optionally with a recency boost
        and neighbor padding.

        score(m, q) = alpha * cos(v_q, v_m) + beta * exp(-dt / tau)
        where alpha=1.0, beta=self.recency_weight, tau=self.recency_half_life_days.

        If ``neighbor_window > 0``, each top-k hit is padded with N items
        immediately before and after it in creation order, so the returned
        context preserves local conversational structure. Padded neighbors
        are inserted in chronological order around their anchor.
        """
        # Vector Search
        query_vec = list(self.embed_model.embed([query]))[0]

        # 1. Fetch all embeddings (Naive Scan for MVP - okay for <10k items)
        all_memories = list(self.db["memories"].rows)
        if not all_memories:
            return []

        now = datetime.now(timezone.utc)
        beta = self.recency_weight if use_recency else 0.0
        tau = max(self.recency_half_life_days, 1e-6)

        # Stable order by created_at (then id) so neighbor lookup is deterministic.
        all_memories.sort(key=lambda m: (m.get("created_at", ""), m.get("id", "")))
        idx_by_id = {m["id"]: i for i, m in enumerate(all_memories)}

        scores = []
        for mem in all_memories:
            if not mem["embedding"]:
                continue
            vec = np.frombuffer(mem["embedding"], dtype=np.float32)
            denom = (np.linalg.norm(query_vec) * np.linalg.norm(vec)) or 1.0
            cos_sim = float(np.dot(query_vec, vec) / denom)

            recency = 0.0
            if beta > 0.0:
                ts_str = mem.get("last_accessed") or mem.get("created_at")
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    dt_days = max((now - ts).total_seconds() / 86400.0, 0.0)
                    recency = float(np.exp(-dt_days / tau))
                except Exception:
                    recency = 0.0

            score = cos_sim + beta * recency
            scores.append((score, cos_sim, recency, mem))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_anchors = scores[:limit]

        if neighbor_window <= 0:
            return [
                {**m, "score": float(s), "cos_sim": float(c), "recency": float(r)}
                for s, c, r, m in top_anchors
            ]

        # Expand each anchor with surrounding neighbors, preserving order and
        # de-duplicating by id. Anchors keep their score; neighbors get the
        # anchor's score so downstream consumers can still rank.
        seen: dict[str, dict] = {}
        anchor_ids: list[str] = []
        for s, c, r, mem in top_anchors:
            anchor_idx = idx_by_id.get(mem["id"], -1)
            if anchor_idx < 0:
                continue
            lo = max(0, anchor_idx - neighbor_window)
            hi = min(len(all_memories), anchor_idx + neighbor_window + 1)
            for i in range(lo, hi):
                neighbor = all_memories[i]
                nid = neighbor["id"]
                if nid in seen:
                    continue
                seen[nid] = {
                    **neighbor,
                    "score": float(s),
                    "cos_sim": float(c) if i == anchor_idx else 0.0,
                    "recency": float(r) if i == anchor_idx else 0.0,
                }
                if i == anchor_idx:
                    anchor_ids.append(nid)

        # Return anchors first by score, with their neighborhoods grouped.
        result: list[dict] = []
        emitted: set[str] = set()
        for aid in anchor_ids:
            if aid not in seen:
                continue
            anchor_idx = idx_by_id[aid]
            lo = max(0, anchor_idx - neighbor_window)
            hi = min(len(all_memories), anchor_idx + neighbor_window + 1)
            for i in range(lo, hi):
                nid = all_memories[i]["id"]
                if nid in emitted:
                    continue
                if nid in seen:
                    result.append(seen[nid])
                    emitted.add(nid)
        return result

    def forget(self, memory_ids: List[str]):
        """Hard delete items."""
        for mid in memory_ids:
            try:
                self.db["memories"].delete(mid)
            except:
                pass 
