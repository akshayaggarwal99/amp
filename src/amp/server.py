from mcp.server.fastmcp import FastMCP
from amp.core.storage import Storage
from typing import List

# Initialize FastMCP Server
mcp = FastMCP("AMP Memory")

# Initialize Storage lazily
_storage = None
_logging_enabled = False

def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage

def initialize(verbose: bool = False):
    """Explicitly initialize storage (useful for CLI logs)."""
    global _logging_enabled
    _logging_enabled = verbose
    get_storage()

def log(activity: str):
    import sys
    import datetime
    from pathlib import Path
    
    # 1. Stderr (if verbose)
    if _logging_enabled:
        sys.stderr.write(f"[AMP] {activity}\n")
        sys.stderr.flush()
    
    # 2. File Log (Always, for observability)
    try:
        # Prmiary: Home Dir
        log_path = Path.home() / ".amp" / "amp.log"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Fallback: Current Dir (for debugging)
        cwd_log = Path.cwd() / "amp_debug.log"
        
        msg = f"[{timestamp}] {activity}\n"
        
        with open(log_path, "a") as f:
            f.write(msg)
            
        with open(cwd_log, "a") as f:
            f.write(msg)
            
    except Exception as e:
        # Aggressive error reporting
        sys.stderr.write(f"[AMP LOG ERROR] {e}\n")
        sys.stderr.flush()

@mcp.tool()
def add_memory(content: str, metadata: str = "{}") -> str:
    """
    Add a thought or observation to Short Term Memory (STM).
    It will be held there until consolidated.
    """
    # Simple JSON parsing for metadata if needed, for now assuming dict
    import json
    try:
        meta = json.loads(metadata)
    except:
        meta = {"raw": metadata}
    
    log(f"TOOL: add_memory | Content: {content[:30]}...")
    mid = get_storage().add_to_stm(content, meta)
    return f"Memory added to STM with ID: {mid}"

@mcp.tool()
def search_memory(query: str, limit: int = 5) -> str:
    """
    Search for memories in Long Term Memory (LTM).
    """
    log(f"TOOL: search_memory | Query: {query}")
    results = get_storage().search(query, limit)
    if not results:
        return "No matching memories found."
    
    formatted = []
    for r in results:
        formatted.append(f"[{r['type']}] {r['content']} (Score: {r['score']})")
    
    return "\n".join(formatted)

@mcp.tool()
def consolidate_memories() -> str:
    """
    Reflect on Short Term Memory and move items to Long Term Memory.
    Call this when a task or session is complete.
    """
    log("TOOL: consolidate_memories | Processing STM...")
    count = get_storage().consolidate()
    log(f"TOOL: consolidate_memories | Moved {count} items to LTM.")
    return f"Consolidated {count} memories from STM to LTM."

@mcp.tool()
def forget_memory(memory_ids: List[str]) -> str:
    """
    Permanently delete memories by ID.
    """
    log(f"TOOL: forget_memory | IDs: {memory_ids}")
    get_storage().forget(memory_ids)
    return f"Forgot {len(memory_ids)} memories."

@mcp.resource("amp://memory/working")
def get_working_memory() -> str:
    """
    Get the current state of Working Memory (STM).
    """
    items = get_storage().get_stm()
    import json
    return json.dumps(items, indent=2)

@mcp.resource("amp://memory/recent")
def get_recent_memories() -> str:
    """
    Get the most recently formed Long Term Memories.
    """
    # SQLite-utils doesn't expose raw SQL easily via 'search', 
    # so we use underlying query execution if needed, 
    # but for now let's reuse search or just query db directly.
    rows = get_storage().db["memories"].rows_where(order_by="created_at desc", limit=10)
    import json
    return json.dumps(list(rows), indent=2)

def create_server():
    return mcp

def serve():
    """Entrypoint for running the server."""
    mcp.run()
