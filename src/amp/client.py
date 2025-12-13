from pathlib import Path
from typing import List, Dict, Optional, Any
from amp.core.storage import Storage

class Amp:
    """
    Client for the Agent Memory Protocol (AMP).
    Allows direct interaction with the memory brain (STM/LTM) from Python.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to ~/.amp/brain.db
            home = Path.home()
            amp_dir = home / ".amp"
            amp_dir.mkdir(exist_ok=True)
            db_path = str(amp_dir / "brain.db")
        
        self.storage = Storage(db_path=db_path)

    def remember(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Adds a thought or observation to Short Term Memory (STM).
        Does NOT immediately persist to LTM (use consolidate() for that).
        
        Args:
            content: The text to remember.
            metadata: Optional dictionary of tags/metadata.
            
        Returns:
            The memory ID in STM.
        """
        return self.storage.add_to_stm(content=content, metadata=metadata or {})

    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Searches Long Term Memory (LTM) for relevant items.
        Uses Hybrid Search (Vector + FTS) if enabled.
        
        Args:
            query: The search query.
            limit: Max number of results.
            
        Returns:
            List of memory items with scores.
        """
        return self.storage.search(query=query, limit=limit)
    
    def consolidate(self, use_llm: bool = False) -> int:
        """
        Triggers the consolidation process:
        1. Reads active STM items.
        2. Generates embeddings.
        3. Optionally extracts entities using local LLM.
        4. Moves them to LTM.
        5. Clears STM.
        
        Args:
            use_llm: If True, extract entities using Ollama (requires local Ollama).
        
        Returns:
            Number of items consolidated.
        """
        return self.storage.consolidate(use_llm=use_llm)
    
    def get_entities(self) -> list:
        """
        Returns all extracted entities from the knowledge graph.
        
        Returns:
            List of entity dicts with name, description, relations.
        """
        return list(self.storage.db["entities"].rows)
    
    def forget(self, memory_ids: List[str]):
        """
        Hard deletes memories from LTM.
        
        Args:
            memory_ids: List of memory IDs to delete.
        """
        self.storage.forget(memory_ids)
