# System Adapters for Benchmarking
from .amp_adapter import AmpAdapter
from .rag_baseline import RagBaseline
from .full_context import FullContextBaseline

# Optional: Mem0 (requires mem0ai package)

try:
    from .mem0_adapter import Mem0Adapter
except ImportError:
    Mem0Adapter = None

try:
    from .openmemory_adapter import OpenMemoryAdapter
except ImportError:
    OpenMemoryAdapter = None

try:
    from .cognee_adapter import CogneeAdapter
except ImportError:
    CogneeAdapter = None

try:
    from .memori_adapter import MemoriAdapter
except ImportError:
    MemoriAdapter = None

try:
    from .mem0_api_adapter import Mem0ApiAdapter
except ImportError:
    Mem0ApiAdapter = None

__all__ = ["AmpAdapter", "RagBaseline", "FullContextBaseline", "Mem0Adapter", "Mem0ApiAdapter"]
