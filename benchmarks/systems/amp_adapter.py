"""AMP System Adapter for Benchmarking."""
from amp.client import Amp
from pathlib import Path
import time

class AmpAdapter:
    """Adapter for AMP memory system.

    ``neighbor_window > 0`` enables the AMP-Padded variant: each top-k anchor
    is padded with N items on each side in creation order, so the retrieved
    context preserves local turn structure.
    """

    name = "AMP"

    def __init__(self, db_path: str = "benchmark_amp.db", neighbor_window: int = 0):
        self.db_path = db_path
        self.brain = None
        self.ingest_time = 0.0
        self.neighbor_window = neighbor_window
        if neighbor_window > 0:
            self.name = f"AMP-Padded{neighbor_window}"

    def reset(self):
        """Reset the memory store."""
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()
        self.brain = Amp(db_path=self.db_path)

    def ingest(self, sessions: list[list[dict]]):
        """Ingest conversation sessions. Each session is a list of {speaker, text}."""
        start = time.time()
        for session in sessions:
            for turn in session:
                msg = f"{turn['speaker']}: {turn['text']}"
                self.brain.remember(msg, {"source": "benchmark"})
            self.brain.consolidate()
        self.ingest_time = time.time() - start

    def search(self, query: str, limit: int = 10) -> str:
        """Search and return context string."""
        results = self.brain.recall(
            query, limit=limit, neighbor_window=self.neighbor_window
        )
        return "\n".join([f"- {r['content']}" for r in results])
    
    def cleanup(self):
        """Cleanup resources."""
        if self.brain and hasattr(self.brain, "storage") and hasattr(self.brain.storage, "db"):
            try:
                self.brain.storage.db.close()
            except:
                pass
        self.brain = None
            
        if Path(self.db_path).exists():
            try:
                Path(self.db_path).unlink()
            except Exception as e:
                print(f"Warning: Failed to delete {self.db_path}: {e}")
