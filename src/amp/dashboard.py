
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from amp.server import get_storage, initialize
import os
import json
import numpy as np
from pathlib import Path

app = FastAPI(title="AMP Dashboard")

# Ensure storage is initialized (lazy load)
initialize()

class MemoryItem(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class Query(BaseModel):
    query: str
    limit: int = 50

def _clean(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove binary fields that break JSON serialization."""
    cleaned = []
    for m in memories:
        m_copy = m.copy()
        if "embedding" in m_copy:
            del m_copy["embedding"]
        cleaned.append(m_copy)
    return cleaned

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/memories/stm")
def get_stm():
    return get_storage().get_stm()

@app.get("/api/memories/ltm")
def get_ltm(limit: int = 100):
    rows = get_storage().db["memories"].rows_where(order_by="created_at desc", limit=limit)
    return _clean(list(rows))

@app.get("/api/graph")
def get_graph():
    try:
        rows = list(get_storage().db["memories"].rows)
        # Handle cases with few memories
        if not rows: return {"nodes": [], "edges": []}

        valid_items = []
        vectors = []
        for r in rows:
            if r.get("embedding"):
                try:
                    vec = np.frombuffer(r["embedding"], dtype=np.float32)
                    if vec.shape[0] >= 384:
                        vectors.append(vec)
                        valid_items.append(r)
                except: pass

        nodes = []
        edges = []
        N = len(vectors)

        # --- V5.0: PCA Projection (The Galaxy Map) ---
        pca_coords = np.zeros((N, 2))
        try:
            if N > 2:
                # Center the data
                X = np.array(vectors)
                X_centered = X - np.mean(X, axis=0)
                # SVD for PCA
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                # Project to first 2 components
                pca_coords = (X_centered @ Vt.T[:, :2])
                
                # Normalize to -1..1 range for easier frontend rendering
                max_val = np.abs(pca_coords).max()
                if max_val > 0:
                    pca_coords = pca_coords / (max_val * 1.2) # 1.2 padding
        except Exception as e:
            print(f"PCA Error: {e}")

        for i, item in enumerate(valid_items):
            nodes.append({
                "id": item["id"],
                "label": item["content"], 
                "type": item["type"],
                "created_at": item.get("created_at", ""),
                "x": float(pca_coords[i, 0]) if i < N else 0.0, # Galaxy X (Normalized)
                "y": float(pca_coords[i, 1]) if i < N else 0.0  # Galaxy Y (Normalized)
            })

        if N > 1:
            X = np.array(vectors)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_norm = X / norms
            sim_matrix = X_norm @ X_norm.T
            
            # --- V4.3: Edge Logic ---
            GLOBAL_THRESHOLD = 0.65 if N > 20 else 0.55
            connected_mask = np.zeros(N, dtype=bool)

            for i in range(N):
                for j in range(i + 1, N):
                    score = float(sim_matrix[i, j])
                    if score > GLOBAL_THRESHOLD:
                        edges.append({
                            "source": nodes[i]["id"],
                            "target": nodes[j]["id"],
                            "weight": score 
                        })
                        connected_mask[i] = True
                        connected_mask[j] = True
            
            # Rescue Orphans (k-NN)
            for i in range(N):
                if not connected_mask[i]:
                    sims = sim_matrix[i].copy()
                    sims[i] = -1.0 # Ignore self
                    best_match_idx = np.argmax(sims)
                    best_score = float(sims[best_match_idx])
                    
                    if best_score > 0.1: 
                        edges.append({
                            "source": nodes[i]["id"],
                            "target": nodes[best_match_idx]["id"],
                            "weight": best_score, 
                            "is_rescue": True 
                        })

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        print(f"Graph Error: {e}")
        return {"nodes": [], "edges": []}


@app.post("/api/memories/add")
def add_memory(item: MemoryItem):
    idx = get_storage().add_to_stm(item.content, item.metadata)
    return {"id": idx, "status": "added to STM"}

@app.post("/api/memories/consolidate")
def consolidate():
    count = get_storage().consolidate()
    return {"count": count, "status": "consolidated"}

@app.post("/api/memories/search")
def search(q: Query):
    results = get_storage().search(q.query, q.limit)
    return _clean(results)

@app.post("/api/memories/forget")
def forget(ids: List[str]):
    get_storage().forget(ids)
    return {"status": "forgotten", "count": len(ids)}

# --- FRONTEND (SERVED FROM STATIC FILE) ---

@app.get("/", response_class=HTMLResponse)
def index():
    # Read from src/amp/static/dashboard.html
    static_path = Path(__file__).parent / "static" / "dashboard.html"
    if not static_path.exists():
        return HTMLResponse("<h1>Error: Static file not found</h1><p>Expected at: " + str(static_path) + "</p>", status_code=500)
    return HTMLResponse(static_path.read_text(encoding="utf-8"))

def start_server(port: int = 8000):
    import uvicorn
    # Use standard uvicorn run
    uvicorn.run(app, host="0.0.0.0", port=port)
