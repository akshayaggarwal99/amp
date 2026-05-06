"""Full Context Baseline System Adapter."""
import time

class FullContextBaseline:
    """Stuffs entire conversation into prompt (no retrieval)."""
    
    name = "Full_Context"
    
    def __init__(self, max_tokens: int = 50000):
        self.max_tokens = max_tokens  # Approximate limit
        self.full_context = ""
        self.ingest_time = 0.0
    
    def reset(self):
        """Reset the memory store."""
        self.full_context = ""
    
    def ingest(self, sessions: list[list[dict]]):
        """
        Ingest conversation sessions.
        """
        start = time.time()
        for session in sessions:
            for turn in session:
                self.full_context += f"{turn['speaker']}: {turn['text']}\n"
        
        # Truncate if too long (rough token estimate: 4 chars per token)
        max_chars = self.max_tokens * 4
        if len(self.full_context) > max_chars:
            self.full_context = self.full_context[-max_chars:]
        
        self.ingest_time = time.time() - start
    
    def search(self, query: str, limit: int = 10) -> str:
        """Return entire context (no search needed)."""
        return self.full_context
    
    def cleanup(self):
        """Cleanup resources."""
        self.full_context = ""
