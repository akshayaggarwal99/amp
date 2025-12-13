from amp import Amp
import time
from pathlib import Path

# Setup
db_path = "demo_brain.db"
if Path(db_path).exists():
    Path(db_path).unlink()

print("--- AMP CLIENT DEMO ---")

# 1. Initialize
brain = Amp(db_path=db_path)
print(f"[x] Initialized Brain at {db_path}")

# 2. Add thoughts (STM)
print("\n[Thinking...]")
t1 = brain.remember("The project deadline is next Friday.")
t2 = brain.remember("The user prefers using 'fastembed' over 'sentence-transformers'.")
print(f" -> Added thought {t1}")
print(f" -> Added thought {t2}")

# 3. Consolidate (STM -> LTM)
print("\n[Consolidating...]")
count = brain.consolidate()
print(f" -> Consolidated {count} memories to LTM.")

# 4. Recall (LTM)
print("\n[Recalling...]")
query = "which embedding library does the user like?"
results = brain.recall(query)

if results:
    top = results[0]
    print(f"Q: '{query}'")
    print(f"A: {top['content']} (Score: {top['score']:.2f})")
else:
    print("No results found.")

# Cleanup
Path(db_path).unlink()
