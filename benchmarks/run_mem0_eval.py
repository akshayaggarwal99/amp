import json
import os
import shutil
import google.generativeai as genai
from mem0 import Memory
from pathlib import Path
from tqdm import tqdm

# Configure Gemini for Eval
API_KEY = __import__("os").environ.get("GEMINI_API_KEY") or (_ for _ in ()).throw(RuntimeError("GEMINI_API_KEY not set"))
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

def run_mem0_eval():
    print("--- MEM0 LoCoMo EVALUATION (Baseline) ---")
    
    # Clean up previous brain to avoid dimension mismatch (1536 vs 768)
    if Path("mem0_brain").exists():
        shutil.rmtree("mem0_brain")
    
    # 1. Setup Mem0 (Local Vector Store)
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": "mem0_brain", # Local path
                "embedding_model_dims": 768
            }
        },
        "llm": {
             "provider": "gemini",
             "config": {
                 "api_key": API_KEY,
                 "model": "gemini-2.5-flash-lite"
             }
        },
        "embedder": {
             "provider": "gemini",
             "config": {
                 "api_key": API_KEY,
                 "model": "models/text-embedding-004" 
             }
        }
    }
    
    os.environ["GEMINI_API_KEY"] = API_KEY
    
    try:
        # Pass config explicitly
        m = Memory.from_config(config)
    except Exception as e:
        print(f"Mem0 Init Error: {e}")
        return

    # 2. Load Data (Same subset)
    dataset_path = "temp_research/MemOS/evaluation/data/locomo/locomo10.json"
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    LIMIT_EXAMPLES = 2
    data_subset = data[:LIMIT_EXAMPLES]
    print(f"Processing {len(data_subset)} examples with Mem0...")

    total_correct = 0
    total_questions = 0
    
    for i, example in enumerate(data_subset):
        # Clear memory for each user/session to be fair? 
        # Mem0 uses 'user_id'. We'll use a unique one per example.
        user_id = f"locomo_ex_{i}"
        m.delete_all(user_id=user_id)
        
        conversation = example["conversation"]
        qa_pairs = example["qa"]
        
        # 3. Ingest
        sessions = [key for key in conversation.keys() if key.startswith("session_") and not key.endswith("date_time")]
        sessions.sort(key=lambda x: int(x.split("_")[1]))

        print(f"Ingesting {len(sessions)} sessions...")
        for session_key in tqdm(sessions, desc="Sessions"):
            messages = []
            for turn in conversation[session_key]:
                # Mem0 expects list of messages
                messages.append({"role": "user" if turn['speaker'] == conversation['speaker_a'] else "assistant", "content": turn['text']})
            
            # Batch add? Mem0 'add' takes text or messages.
            m.add(messages, user_id=user_id)
        
        # 4. Evaluate
        print(f"Evaluating {len(qa_pairs)} questions...")
        for qa in qa_pairs:
            if "answer" not in qa: continue
            q = qa["question"]
            truth = str(qa["answer"])
            
            # Mem0 Search / Ask
            # We can use m.search(q) for retrieval comparison
            # Or m.ask(q) if it exists (some versions have it)
            
            results = m.search(q, user_id=user_id, limit=5)
            # Context
            top_context = "\n".join([r['memory'] for r in results])
            
            # Use OUR generation to be fair (only comparing retrieval quality)
            # OR trust Mem0's embeddings.
            
            # Generate Answer (using same generation function as AMP benchmark to isolate retrieval quality)
            prompt = f"Context: {top_context}\nQuestion: {q}\nAnswer:"
            try:
                resp = model.generate_content(prompt)
                predicted = resp.text.strip()
            except:
                predicted = "Error"
            
            # Check correctness (Substring)
            # LLM Judge is better but expensive/slow. Let's use substring for now as quick check.
            is_correct = truth.lower() in predicted.lower()
            
            if is_correct:
                total_correct += 1
            total_questions += 1
            
    print(f"\nMem0 Recall Score: {(total_correct/total_questions)*100:.1f}%")

if __name__ == "__main__":
    run_mem0_eval()
