"""
Mem0 Cloud API Adapter for Benchmarking.
"""
import time
import os
from typing import List, Dict
from datetime import datetime

class Mem0ApiAdapter:
    """
    Adapter for Mem0 Cloud API.
    """
    
    name = "Mem0 (API)"
    
    def __init__(self):
        self.client = None
        self.user_id = "mem0_benchmark_user_v1"
        self.ingest_time = 0.0
        
        self.api_key = os.environ.get("MEM0_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "MEM0_API_KEY not set. Set it in your environment or in .env "
                "before instantiating Mem0ApiAdapter."
            )
    
    def reset(self):
        """Reset the memory store."""
        try:
            from mem0 import MemoryClient
            self.client = MemoryClient(api_key=self.api_key)
            
            # Clean up existing memories for this user
            try:
                # Note: Mem0 API doesn't have a 'delete_all' for a user efficiently exposed 
                # in the simple client sometimes, but let's try to list and delete or just use a new user_id.
                # For a clean benchmark, appending a timestamp to user_id is safer/faster 
                # than deleting 100s of memories one by one if delete_all isn't available.
                # However, let's try to see if there's a delete all or we just rotate user ID.
                
                # Rotating user ID for fresh start is the most reliable way to "reset" 
                # without hammering the API to delete individual items.
                timestamp = int(time.time())
                self.user_id = f"mem0_bench_{timestamp}"
                print(f"  [Mem0 API] New session for user: {self.user_id}")
                
            except Exception as e:
                print(f"Mem0 Reset Error: {e}")
                
        except ImportError:
            print("Mem0 SDK not installed. Run: pip install mem0ai")
            self.client = None
    
    def ingest(self, sessions: List[List[Dict]]):
        """Ingest conversation sessions."""
        if not self.client:
            print("Mem0 client not initialized")
            return
        
        start = time.time()
        
        # Process all sessions
        for session in sessions:
            messages = []
            for turn in session:
                role = "user" if len(messages) % 2 == 0 else "assistant"
                messages.append({"role": role, "content": turn["text"]})
            
            try:
                # Add all messages from the session
                self.client.add(messages, user_id=self.user_id, version="v2")
                # Small sleep to avoid rate limits if necessary, though benchmarks usually run sequentially
                time.sleep(0.5) 
            except Exception as e:
                print(f"Mem0 Ingest Error: {e}")
        
        self.ingest_time = time.time() - start
    
    def search(self, query: str, limit: int = 10) -> str:
        """Search and return context string."""
        if not self.client:
            return ""
        
        try:
            # Construct filters as per notebook
            filters = {
                "OR": [
                    {"user_id": self.user_id}
                ]
            }
            
            # Search
            results = self.client.search(query, filters=filters, limit=limit, version="v2")
             # The result format from API might be a dict with 'results' key or list
            
            if isinstance(results, dict) and "results" in results:
                results = results["results"]
            
            context_parts = []
            for r in results:
                # memory content is usually in 'memory' key
                content = r.get('memory') if isinstance(r, dict) else str(r)
                context_parts.append(f"- {content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"Mem0 Search Error: {e}")
            return ""
    
    def cleanup(self):
        """Cleanup resources."""
        # Nothing local to clean up
        self.client = None
