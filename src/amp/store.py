import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Protocol, Any
from datetime import datetime
from abc import ABC, abstractmethod

from .models import Fact, Episode, WorkingMemoryItem

class MemoryBackend(ABC):
    @abstractmethod
    def add_fact(self, fact: Fact): ...
    
    @abstractmethod
    def get_fact(self, key: str) -> Optional[Fact]: ...
    
    @abstractmethod
    def search_facts(self, query: str) -> List[Fact]: ...

    @abstractmethod
    def delete_fact(self, key: str): ...
    
    @abstractmethod
    def add_episode(self, episode: Episode): ...
    
    @abstractmethod
    def search_episodes(self, query: str, limit: int = 10) -> List[Episode]: ...

    @abstractmethod
    def delete_episode(self, episode_id: str): ...

class SQLiteBackend(MemoryBackend):
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    category TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    source TEXT,
                    json_data TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    timestamp TEXT,
                    json_data TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(timestamp DESC)")

    def add_fact(self, fact: Fact):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO facts (key, value, category, created_at, updated_at, source, json_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    fact.key, 
                    fact.value, 
                    fact.category, 
                    fact.created_at.isoformat(), 
                    fact.updated_at.isoformat(), 
                    fact.source,
                    fact.model_dump_json()
                )
            )

    def get_fact(self, key: str) -> Optional[Fact]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT json_data FROM facts WHERE key = ?", (key,))
            row = cursor.fetchone()
            return Fact.model_validate_json(row[0]) if row else None

    def search_facts(self, query: str) -> List[Fact]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT json_data FROM facts WHERE key LIKE ? OR value LIKE ?", 
                (f"%{query}%", f"%{query}%")
            )
            return [Fact.model_validate_json(row[0]) for row in cursor.fetchall()]

    def delete_fact(self, key: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM facts WHERE key = ?", (key,))

    def add_episode(self, episode: Episode):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO episodes (id, content, timestamp, json_data) VALUES (?, ?, ?, ?)",
                (
                    episode.id,
                    episode.content,
                    episode.timestamp.isoformat(),
                    episode.model_dump_json()
                )
            )

    def search_episodes(self, query: str, limit: int = 10) -> List[Episode]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT json_data FROM episodes WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", limit)
            )
            return [Episode.model_validate_json(row[0]) for row in cursor.fetchall()]

    def delete_episode(self, episode_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))

class WorkingMemory:
    """
    Short-Term Memory (STM).
    Volatile, limited capacity buffer.
    """
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[WorkingMemoryItem] = []

    def add(self, item: WorkingMemoryItem):
        # Insert at start (fresh)
        self.items.insert(0, item)
        # Prune if over capacity
        if len(self.items) > self.capacity:
            self.items.pop()
    
    def get_all(self) -> List[WorkingMemoryItem]:
        return self.items

    def clear(self):
        self.items = []

class MemoryStore:
    """
    Cognitive Architecture Entry Point.
    Manages STM (Working Memory) and LTM (SQLite Backend).
    """
    
    def __init__(self, root_dir: Optional[Path] = None, backend: Optional[MemoryBackend] = None):
        if root_dir:
            self.root = root_dir
        else:
            self.root = Path.home() / ".amp"
            
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Initialize LTM
        if backend:
            self.ltm = backend
        else:
            self.ltm = SQLiteBackend(self.root / "amp.db")
            
        # Initialize STM
        self.stm = WorkingMemory()

    # --- STM Operations ---
    
    def hold_thought(self, content: str, ttl: int = 5):
        """Add a thought to Working Memory."""
        item = WorkingMemoryItem(content=content, ttl=ttl)
        self.stm.add(item)
    
    def get_working_memory(self) -> List[WorkingMemoryItem]:
        return self.stm.get_all()

    def clear_stm(self):
        self.stm.clear()

    def consolidate(self):
        """
        Move items from STM to LTM (Episodes).
        In a real brain, this happens during sleep/rest.
        Here, we convert thoughts to Episodes.
        """
        items = self.stm.get_all()
        for item in items:
            # Convert STM item to LTM Episode
            episode = Episode(
                content=item.content,
                timestamp=item.timestamp,
                tags=["consolidated"]
            )
            self.ltm.add_episode(episode)
        
        # Clear STM after consolidation? 
        # Scientific debate: Does consolidation wipe STM? Usually yes for those specific items.
        self.stm.clear()

    # --- LTM Operations ---

    def add_fact(self, fact: Fact):
        self.ltm.add_fact(fact)

    def get_fact(self, key: str) -> Optional[Fact]:
        return self.ltm.get_fact(key)

    def search_facts(self, query: str) -> List[Fact]:
        return self.ltm.search_facts(query)
    
    def delete_fact(self, key: str):
        self.ltm.delete_fact(key)

    def add_episode(self, episode: Episode):
        self.ltm.add_episode(episode)

    def search_episodes(self, query: str, limit: int = 10) -> List[Episode]:
        return self.ltm.search_episodes(query, limit)

    def delete_episode(self, episode_id: str):
        self.ltm.delete_episode(episode_id)

    # --- Cognitive Search ---
    
    def recall(self, query: str) -> dict:
        """
        Search EVERYTHING (STM + LTM).
        """
        results = {
            "stm": [],
            "facts": [],
            "episodes": []
        }
        
        # Search STM
        for item in self.stm.get_all():
            if query.lower() in item.content.lower():
                results["stm"].append(item)
        
        # Search LTM
        results["facts"] = self.ltm.search_facts(query)
        results["episodes"] = self.ltm.search_episodes(query)
        
        return results
