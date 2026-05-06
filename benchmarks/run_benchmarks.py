import time
import uuid
import random
import json
from amp.core.storage import Storage
from pathlib import Path

def run_benchmarks(num_items=100):
    print(f"--- AMP BENCHMARK SUITE (N={num_items}) ---")
    
    # Setup clean brain
    db_path = "bench_brain.db"
    if Path(db_path).exists():
        Path(db_path).unlink()
        
    store = Storage(db_path=db_path)
    
    # 1. Write Latency (STM)
    print(f"\n[1] Testing Write Latency (STM)...")
    dataset = []
    for i in range(num_items):
        key = str(uuid.uuid4())[:8]
        val = f"The secret code for {key} is {random.randint(1000, 9999)}"
        dataset.append((key, val))
        
    start_time = time.time()
    for key, val in dataset:
        store.add_to_stm(val, {"type": "benchmark", "key": key})
    end_time = time.time()
    
    write_time = end_time - start_time
    write_mps = num_items / write_time
    print(f"-> Time: {write_time:.4f}s")
    print(f"-> Throughput: {write_mps:.2f} memories/sec")
    
    # 2. Consolidation Speed
    print(f"\n[2] Testing Consolidation (STM -> LTM)...")
    start_time = time.time()
    count = store.consolidate()
    end_time = time.time()
    
    consol_time = end_time - start_time
    print(f"-> Consolidated {count} items.")
    print(f"-> Time: {consol_time:.4f}s")
    
    # 3. Read Latency & Accuracy
    print(f"\n[3] Testing Read Latency & Accuracy (LTM)...")
    hits = 0
    start_time = time.time()
    for key, expected_val in dataset:
        # Search using the unique key to simulate specific recall
        results = store.search(key, limit=1)
        if results and results[0]["content"] == expected_val:
            hits += 1
    end_time = time.time()
    
    read_time = end_time - start_time
    read_qps = num_items / read_time
    accuracy = (hits / num_items) * 100
    
    print(f"-> Time: {read_time:.4f}s")
    print(f"-> Throughput: {read_qps:.2f} queries/sec")
    print(f"-> Accuracy: {accuracy:.2f}% (Exact Match)")
    
    # Cleanup
    Path(db_path).unlink()
    
    return {
        "n": num_items,
        "write_mps": write_mps,
        "consol_time": consol_time,
        "read_qps": read_qps,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    run_benchmarks(100)
    print("\n--- Running Load Test (N=1000) ---")
    run_benchmarks(1000)
