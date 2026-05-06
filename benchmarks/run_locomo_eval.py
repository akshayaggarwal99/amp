import json
import os
import time
from pathlib import Path

import google.generativeai as genai
from amp.client import Amp
from tqdm import tqdm

# Configure Gemini — read from env (set GEMINI_API_KEY) or .env file at repo root
def _load_env(path: str = ".env") -> None:
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_env()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set (env or .env)")
genai.configure(api_key=API_KEY)
# User requested exact model "gemini-2.5-flash-lite"
model = genai.GenerativeModel('gemini-2.5-flash-lite')

def generate_answer(context: str, question: str) -> str:
    """Generates an answer using the LLM based on context."""
    if not context:
        return "I don't know."
    
    prompt = f"""
    You are a helpful assistant with a perfect memory.
    Answer the question based ONLY on the provided context. 
    If the answer is not in the context, say "I don't know".
    
    Context:
    {context}
    
    Question: {question}
    Answer:
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"GenAI Error: {e}")
        return "Error"

def check_correctness(predicted: str, truth: str, question: str) -> bool:
    """Uses LLM to judge if the predicted answer matches the truth."""
    # Fast fallback: exact substring match
    if truth.lower() in predicted.lower():
        return True
        
    # LLM Judge
    prompt = f"""
    Double check if the predicted answer conveys the same meaning as the ground truth.
    Question: {question}
    Ground Truth: {truth}
    Predicted: {predicted}
    
    Return "YES" if it is correct, "NO" otherwise. output only YES or NO.
    """
    try:
        response = model.generate_content(prompt)
        return "YES" in response.text.upper()
    except:
        return False

def run_locomo_eval():
    print("--- AMP LoCoMo EVALUATION (RAG + LLM Judge) ---")
    
    # 1. Setup
    db_path = "locomo_brain_rag.db"
    if Path(db_path).exists():
        Path(db_path).unlink()
    
    brain = Amp(db_path=db_path)
    
    # 2. Load Data
    dataset_path = "temp_research/MemOS/evaluation/data/locomo/locomo10.json"
    print(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Limit to first N examples for speed
    LIMIT_EXAMPLES = 2
    data_subset = data[:LIMIT_EXAMPLES]
    print(f"Processing {len(data_subset)} examples using Gemini 1.5 Flash...")

    total_correct = 0
    total_questions = 0
    
    for i, example in enumerate(data_subset):
        print(f"\n--- Example {i+1}/{len(data_subset)} ---")
        conversation = example["conversation"]
        qa_pairs = example["qa"]
        
        # 3. Ingest Sessions
        sessions = [key for key in conversation.keys() if key.startswith("session_") and not key.endswith("date_time")]
        sessions.sort(key=lambda x: int(x.split("_")[1]))

        print(f"Ingesting {len(sessions)} sessions...")
        for session_key in tqdm(sessions, desc="Sessions"):
            for turn in conversation[session_key]:
                msg = f"{turn['speaker']}: {turn['text']}"
                brain.remember(msg, {"source": "locomo", "session": session_key})
            brain.consolidate()
        
        # 4. Evaluate Recall + Generation
        print(f"Evaluating {len(qa_pairs)} questions...")
        for qa in qa_pairs:
            if "answer" not in qa: 
                continue
                
            q = qa["question"]
            truth = str(qa["answer"])
            
            # A. Retrieval
            results = brain.recall(q, limit=10) # Increase limit for RAG
            top_context = "\n".join([f"- {r['content']}" for r in results])
            
            # B. Generation (The "Proper Agent" part)
            predicted = generate_answer(top_context, q)
            
            # C. Grading
            is_correct = check_correctness(predicted, truth, q)
            
            if is_correct:
                total_correct += 1
                status = "PASS"
            else:
                status = "FAIL"
                
            total_questions += 1
            if status == "FAIL":
                 # Debug failures
                 print(f"[{status}] Q: {q}")
                 print(f"       Truth: {truth}")
                 print(f"       Pred:  {predicted}")
            
    # Cleanup
    Path(db_path).unlink()

    # Final Score
    if total_questions > 0:
        score = (total_correct / total_questions) * 100
        print(f"\nRAG Recall Score: {score:.1f}% ({total_correct}/{total_questions})")
    else:
        print("\nNo questions evaluated.")

if __name__ == "__main__":
    run_locomo_eval()
