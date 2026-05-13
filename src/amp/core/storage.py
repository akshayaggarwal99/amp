import sqlite_utils
import json
import hashlib
import numpy as np
from collections import OrderedDict
from fastembed import TextEmbedding
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
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


# --- Helpers · module-level so they're cheap to call from anywhere. ---

def _content_hash(text: str) -> str:
    """Stable hash for dedup. SHA-256 is overkill but cheap and unambiguous."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def _to_epoch(ts: Optional[str]) -> float:
    """Parse an ISO timestamp string into a unix epoch float; 0.0 on failure."""
    if not ts:
        return 0.0
    try:
        d = datetime.fromisoformat(ts)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()
    except Exception:
        return 0.0


def _safe_parse_metadata(raw: Any) -> Dict[str, Any]:
    """Read metadata that may be JSON (new rows) or `str(dict)` (legacy rows
    from before the metadata-storage bug fix). Best-effort, never raises.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        # Try JSON first · what new rows look like.
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except Exception:
            pass
        # Try ast.literal_eval for legacy `str(dict)` rows.
        try:
            import ast
            v = ast.literal_eval(raw)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def _stringify_metadata(value: Any) -> str:
    """Inverse of _safe_parse_metadata · write JSON. Handles legacy `str(dict)`
    payloads that may still be sitting in working_memory rows."""
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, str):
        # Already serialised by add_to_stm · pass through if it parses.
        parsed = _safe_parse_metadata(value)
        return json.dumps(parsed)
    return json.dumps({})


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

        # ── Vector cache · daemon-mode win #1 ──────────────────────────────
        # Search() previously did O(n) cosine in a Python loop, reading every
        # row from sqlite each time. Daemon mode means we can keep the
        # embedding matrix in RAM: at 384 dims float32, 10k memories = 15 MB.
        # Lazy-load on first search; rebuild incrementally on add/forget.
        self._vec_matrix: Optional[np.ndarray] = None       # shape (N, D)
        self._vec_ids: Optional[List[str]] = None           # ordered, parallel to matrix rows
        self._vec_meta: Optional[List[Dict[str, Any]]] = None  # parallel · created_at etc · avoids re-reading rows
        self._vec_dirty = True

        # ── Embedding LRU · win #2 ─────────────────────────────────────────
        # Content hash → vector. Cheap and effective when fixtures / batch
        # imports repeat the same line. Bounded at 4096 entries.
        self._embed_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._embed_cache_cap = 4096

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
            # JSON · `str(dict)` was producing single-quoted python reprs that
            # don't survive json.loads on read. Fix is backwards-compatible:
            # readers can `_safe_parse_metadata` which tries both.
            "metadata": json.dumps(item.metadata or {}),
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

        Wins over the previous version:
          - Dedup by content-hash · re-saying the same thing doesn't bloat LTM.
          - Embeddings stored L2-normalised so `search()` matmul gives a true
            cosine without a per-row norm divide.
          - Vector cache invalidated at the end so next search rebuilds.
          - LRU embedding cache hit-path avoids re-embed for repeat content.
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

        # Dedup against LTM by content hash · don't insert duplicate rows.
        existing_hashes = self._loaded_content_hashes()
        to_embed: List[Tuple[int, str, str]] = []  # (index, content_hash, content)
        for i, item in enumerate(stm_items):
            h = _content_hash(item["content"])
            if h in existing_hashes:
                # Already in LTM · drop STM row without re-embedding.
                self.db["working_memory"].delete(item["id"])
                continue
            to_embed.append((i, h, item["content"]))

        if not to_embed:
            self._vec_dirty = True
            return 0

        # Batch embed only what's new · then L2-normalise so search is cheap.
        new_contents = [c for (_, _, c) in to_embed]
        new_vecs = np.asarray(list(self.embed_model.embed(new_contents)), dtype=np.float32)
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_vecs = new_vecs / norms

        now_iso = datetime.now(timezone.utc).isoformat()
        for (i, h, _content), vec in zip(to_embed, new_vecs):
            item = stm_items[i]
            vec_bytes = vec.astype(np.float32, copy=False).tobytes()
            self.db["memories"].insert({
                "id": item["id"],
                "content": item["content"],
                "created_at": item["created_at"],
                "last_accessed": now_iso,
                "score": 1.0,
                "type": "episodic",
                # Store metadata as JSON · keep parsable on read.
                "metadata": _stringify_metadata(item.get("metadata")),
                "embedding": vec_bytes,
            })
            existing_hashes.add(h)

            # Optional · LLM-extracted entities.
            if llm:
                entities = llm.extract_entities(item["content"])
                for entity in entities:
                    has = list(self.db["entities"].rows_where("name = ?", [entity.name]))
                    if not has:
                        self.db["entities"].insert({
                            "name": entity.name,
                            "description": entity.type,
                            "relations": "{}",
                        })

            self.db["working_memory"].delete(item["id"])

        # Invalidate vector cache · next search rebuilds with the new rows.
        self._vec_dirty = True
        return len(to_embed)

    def _loaded_content_hashes(self) -> set:
        """Snapshot of content hashes already in LTM · cheap O(N) one-shot."""
        out = set()
        for row in self.db.query("SELECT content FROM memories"):
            c = row.get("content")
            if c:
                out.add(_content_hash(c))
        return out

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

        Implementation · vectorised cosine over a cached (N, D) matrix. The
        matrix is loaded lazily on first search and kept in RAM for the
        process lifetime; rebuild is triggered by add / forget. This makes
        repeated daemon queries sub-100 ms even at 10 k+ rows.
        """
        # Lazily build the in-RAM matrix · cached across queries.
        matrix, ids, meta = self._ensure_vec_cache()
        if matrix is None or matrix.size == 0:
            return []

        # Query embedding · with content-hash cache (saves a re-embed for the
        # common case where the eval harness or heartbeat fires the same
        # recall query repeatedly in a tight loop).
        q_vec = self._embed_cached(query)
        q_norm = float(np.linalg.norm(q_vec)) or 1.0

        # Single matmul · (N, D) @ (D,) → (N,)  · vectorised, no Python loop.
        # Embeddings are pre-normalised at consolidate time (see consolidate)
        # so cos_sim = (matrix @ q_vec) / q_norm. We keep the per-row norm
        # safety divisor too in case older rows pre-date normalization.
        row_norms = np.linalg.norm(matrix, axis=1)
        row_norms[row_norms == 0] = 1.0
        cos_sims = (matrix @ q_vec) / (row_norms * q_norm)

        # Optional recency boost · also vectorised.
        beta = self.recency_weight if use_recency else 0.0
        if beta > 0.0:
            now_ts = datetime.now(timezone.utc).timestamp()
            tau_sec = max(self.recency_half_life_days, 1e-6) * 86400.0
            ages = np.maximum(now_ts - np.asarray([m["ts"] for m in meta], dtype=np.float64), 0.0)
            recencies = np.exp(-ages / tau_sec)
        else:
            recencies = np.zeros(len(ids), dtype=np.float32)

        final_scores = cos_sims + beta * recencies

        # Top-k via argpartition · O(N) instead of full sort O(N log N).
        k = max(min(limit, len(ids)), 0)
        if k == 0:
            return []
        if k >= len(ids):
            order = np.argsort(-final_scores)
        else:
            part = np.argpartition(-final_scores, k - 1)[:k]
            order = part[np.argsort(-final_scores[part])]

        top_anchors: List[Tuple[float, float, float, int]] = [
            (float(final_scores[i]), float(cos_sims[i]), float(recencies[i]), int(i))
            for i in order
        ]

        if neighbor_window <= 0:
            return [self._row_from_index(idx, s, c, r) for s, c, r, idx in top_anchors]

        # Neighbor padding · groups each top hit with its ±N chronological
        # neighbors so the caller can see local conversational structure.
        seen: dict[str, dict] = {}
        anchor_ids: list[str] = []
        for s, c, r, anchor_idx in top_anchors:
            lo = max(0, anchor_idx - neighbor_window)
            hi = min(len(ids), anchor_idx + neighbor_window + 1)
            for i in range(lo, hi):
                nid = ids[i]
                if nid in seen:
                    continue
                seen[nid] = self._row_from_index(
                    i, s, c if i == anchor_idx else 0.0, r if i == anchor_idx else 0.0
                )
                if i == anchor_idx:
                    anchor_ids.append(nid)

        result: list[dict] = []
        emitted: set[str] = set()
        for aid in anchor_ids:
            if aid not in seen:
                continue
            anchor_idx = ids.index(aid)
            lo = max(0, anchor_idx - neighbor_window)
            hi = min(len(ids), anchor_idx + neighbor_window + 1)
            for i in range(lo, hi):
                nid = ids[i]
                if nid in emitted:
                    continue
                if nid in seen:
                    result.append(seen[nid])
                    emitted.add(nid)
        return result

    def forget(self, memory_ids: List[str]):
        """Hard delete items · invalidates the vector cache so next search
        rebuilds from sqlite."""
        for mid in memory_ids:
            try:
                self.db["memories"].delete(mid)
            except Exception:
                pass
        self._vec_dirty = True

    # ── Vector cache internals ──────────────────────────────────────────

    def _ensure_vec_cache(self) -> Tuple[Optional[np.ndarray], List[str], List[Dict[str, Any]]]:
        """Load the (N, D) embedding matrix from sqlite once, keep in RAM.

        Returns (matrix, ids_in_order, parallel_meta) · meta carries only the
        cheap fields (ts, content, type, metadata json) so search() doesn't
        re-read full rows. Heavy embedding BLOBs are loaded once here.
        """
        if not self._vec_dirty and self._vec_matrix is not None:
            return self._vec_matrix, self._vec_ids or [], self._vec_meta or []

        # Order by (created_at, id) for deterministic neighbor lookups.
        rows = list(self.db.query(
            "SELECT id, content, created_at, last_accessed, score, type, metadata, embedding "
            "FROM memories ORDER BY created_at ASC, id ASC"
        ))
        if not rows:
            self._vec_matrix = np.zeros((0, 0), dtype=np.float32)
            self._vec_ids = []
            self._vec_meta = []
            self._vec_dirty = False
            return self._vec_matrix, self._vec_ids, self._vec_meta

        vectors: List[np.ndarray] = []
        ids: List[str] = []
        meta: List[Dict[str, Any]] = []
        for r in rows:
            blob = r.get("embedding")
            if not blob:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            vectors.append(vec)
            ids.append(r["id"])
            ts = _to_epoch(r.get("last_accessed") or r.get("created_at"))
            meta.append({
                "id": r["id"],
                "content": r.get("content"),
                "created_at": r.get("created_at"),
                "last_accessed": r.get("last_accessed"),
                "score": r.get("score"),
                "type": r.get("type"),
                "metadata": _safe_parse_metadata(r.get("metadata")),
                "ts": ts,
            })

        self._vec_matrix = np.stack(vectors).astype(np.float32, copy=False)
        self._vec_ids = ids
        self._vec_meta = meta
        self._vec_dirty = False
        return self._vec_matrix, ids, meta

    def _row_from_index(self, idx: int, score: float, cos_sim: float, recency: float) -> Dict[str, Any]:
        """Reconstruct a search result row from cached meta · avoids hitting sqlite."""
        m = (self._vec_meta or [{}])[idx]
        return {
            **m,
            "score": float(score),
            "cos_sim": float(cos_sim),
            "recency": float(recency),
        }

    def _embed_cached(self, text: str) -> np.ndarray:
        """Hash-keyed embedding LRU · saves the FastEmbed call when the same
        query is fired repeatedly (eval harness, heartbeat retries)."""
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._embed_cache.get(key)
        if cached is not None:
            self._embed_cache.move_to_end(key)
            return cached
        vec = np.asarray(list(self.embed_model.embed([text]))[0], dtype=np.float32)
        self._embed_cache[key] = vec
        if len(self._embed_cache) > self._embed_cache_cap:
            self._embed_cache.popitem(last=False)
        return vec
