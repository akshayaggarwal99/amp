import json
import time
from amp.core.storage import Storage
from pathlib import Path

def run_quality_eval():
    print("--- AMP QUALITY EVALUATION (Synthetic LoCoMo) ---")
    
    # 1. Setup
    db_path = "eval_brain.db"
    if Path(db_path).exists():
        Path(db_path).unlink()
    
    storage = Storage(db_path=db_path)
    
    # 2. Load Data
    with open("benchmarks/synthetic_dataset.json", "r") as f:
        data = json.load(f)
        
    conversation = data[0]["conversation"]
    qa_pairs = data[0]["qa_pairs"]
    
    # 3. Ingest Sessions
    print("\n[Phase 1] Ingesting Conversation...")
    sessions = [key for key in conversation.keys() if key.startswith("session_")]
    
    for session_key in sessions:
        print(f" -> Processing {session_key}...")
        for turn in conversation[session_key]:
            msg = f"{turn['speaker']}: {turn['text']}"
            # Add to STM
            storage.add_to_stm(msg, {"source": "chat", "session": session_key})
        
        # Consolidate after each session ( simulating "sleep" or "end of chat")
        storage.consolidate()
        
    print(f" -> Ingested {len(sessions)} sessions.")
    
    # 4. Evaluate Recall
    print("\n[Phase 2] Evaluating Recall...")
    correct = 0
    total = len(qa_pairs)
    
    results_detail = []
    
    for qa in qa_pairs:
        q = qa["question"]
        truth = qa["answer"]
        
        # Search AMP
        results = storage.search(q, limit=3)
        
        # Analyze top result
        top_context = " ".join([r["content"] for r in results])
        
        # Simple heuristic check: Does the retrieved context contain the keywords of the answer?
        # In a real eval, we'd use an LLM Judge. Here, we use simple substring for speed.
        is_correct = truth.lower() in top_context.lower()
        
        if is_correct:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        print(f"[{status}] Q: {q}")
        print(f"       Target: {truth}")
        print(f"       Retrieved: {top_context[:100]}...")
        
        results_detail.append({
            "question": q,
            "status": status,
            "context": top_context
        })
        
    score = (correct / total) * 100
    print(f"\nSimple Recall Score: {score:.1f}%")
    
    # Cleanup
    Path(db_path).unlink()

if __name__ == "__main__":
    run_quality_eval()
