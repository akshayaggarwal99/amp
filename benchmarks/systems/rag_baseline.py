"""RAG Baseline System Adapter."""
from fastembed import TextEmbedding
import numpy as np
import time

class RagBaseline:
    """Simple RAG baseline using fastembed + numpy."""
    
    name = "RAG_Baseline"
    
    def __init__(self, chunk_size: int = 500):
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = []
        self.ingest_time = 0.0
    
    def reset(self):
        """Reset the memory store."""
        self.chunks = []
        self.embeddings = []
    
    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]
    
    def ingest(self, sessions: list[list[dict]]):
        """
        Ingest conversation sessions.
        """
        start = time.time()
        # Flatten sessions to one big text, then chunk
        full_text = ""
        for session in sessions:
            for turn in session:
                full_text += f"{turn['speaker']}: {turn['text']}\n"
        
        self.chunks = self._chunk_text(full_text)
        
        # Embed chunks
        self.embeddings = list(self.embed_model.embed(self.chunks))
        self.embeddings = [np.array(e, dtype=np.float32) for e in self.embeddings]
        
        self.ingest_time = time.time() - start
    
    def search(self, query: str, limit: int = 10) -> str:
        """Search and return context string."""
        if not self.embeddings:
            return ""
        
        query_vec = list(self.embed_model.embed([query]))[0]
        query_vec = np.array(query_vec, dtype=np.float32)
        
        scores = []
        for i, emb in enumerate(self.embeddings):
            score = np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb))
            scores.append((score, i))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:limit]
        
        return "\n".join([f"- {self.chunks[i]}" for _, i in top_k])
    
    def cleanup(self):
        """Cleanup resources."""
        self.chunks = []
        self.embeddings = []
