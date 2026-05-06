#!/usr/bin/env python3
"""
Unified Benchmark Harness for Memory System Comparison.

Usage:
    python benchmarks/run_benchmark.py --system amp --examples 2
    python benchmarks/run_benchmark.py --system amp --examples 2 --llm ollama
    python benchmarks/run_benchmark.py --system all --examples 1 --llm ollama
"""
import argparse
import json
import time
import requests
from pathlib import Path
from datetime import datetime

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.metrics import calculate_bleu, calculate_f1, llm_judge
from benchmarks.systems import AmpAdapter, RagBaseline, FullContextBaseline, Mem0Adapter, Mem0ApiAdapter

# Configuration
GEMINI_API_KEY = __import__("os").environ.get("GEMINI_API_KEY") or (_ for _ in ()).throw(RuntimeError("GEMINI_API_KEY not set"))
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"

DATASET_PATH = "temp_research/MemOS/evaluation/data/locomo/locomo10.json"
RESULTS_DIR = Path("benchmarks/results")

SYSTEM_MAP = {
    "amp": AmpAdapter,
    "rag_baseline": RagBaseline,
    "full_context": FullContextBaseline,
}

if Mem0Adapter is not None:
    SYSTEM_MAP["mem0"] = Mem0Adapter

if Mem0ApiAdapter is not None:
    SYSTEM_MAP["mem0_api"] = Mem0ApiAdapter

# LLM abstraction
class LLMProvider:
    def __init__(self, provider: str = "ollama"):
        self.provider = provider
        if provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    def complete(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._ollama_complete(prompt)
        else:
            return self._gemini_complete(prompt)
    
    def _ollama_complete(self, prompt: str) -> str:
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }, timeout=60)
            return resp.json().get("response", "Error")
        except Exception as e:
            return f"Ollama Error: {e}"
    
    def _gemini_complete(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Gemini Error: {e}"

# Global LLM instance (set in main)
llm_provider = None

def llm_judge_local(question: str, gold: str, predicted: str) -> int:
    """Use local LLM to judge if prediction is correct. Returns 1 or 0."""
    global llm_provider
    prompt = f"""Judge if the predicted answer is correct compared to the ground truth.
Be GENEROUS. If the prediction touches on the same topic, mark CORRECT.
For time questions, relative references like "last Tuesday" matching a specific date = CORRECT.

Question: {question}
Ground Truth: {gold}
Predicted: {predicted}

Respond with only one word: CORRECT or WRONG."""
    
    result = llm_provider.complete(prompt)
    return 1 if "CORRECT" in result.upper() else 0


def generate_answer(context: str, question: str) -> str:
    """Generate answer using LLM given retrieved context."""
    global llm_provider
    if not context.strip():
        return "I don't know."
    
    prompt = f"""You are a helpful assistant with access to conversation history.
Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I don't know".
Be concise.

Context:
{context[:10000]}

Question: {question}
Answer:"""
    return llm_provider.complete(prompt)


def extract_sessions(conversation: dict) -> list[list[dict]]:
    """Extract session data from LoCoMo conversation."""
    sessions = []
    session_keys = [k for k in conversation.keys() 
                   if k.startswith("session_") and not k.endswith("date_time")]
    session_keys.sort(key=lambda x: int(x.split("_")[1]))
    
    for key in session_keys:
        sessions.append(conversation[key])
    
    return sessions


def run_benchmark(system_name: str, num_examples: int) -> dict:
    """Run benchmark for a single system."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {system_name.upper()}")
    print(f"{'='*60}")
    
    # Initialize system
    SystemClass = SYSTEM_MAP[system_name]
    system = SystemClass()
    
    # Load dataset
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)
    
    data_subset = data[:num_examples]
    
    # Results
    all_results = []
    totals = {"bleu": 0, "f1": 0, "llm": 0, "count": 0}
    
    for i, example in enumerate(data_subset):
        print(f"\n--- Example {i+1}/{len(data_subset)} ---")
        
        # Reset system
        system.reset()
        
        # Extract conversation
        conversation = example["conversation"]
        sessions = extract_sessions(conversation)
        qa_pairs = example["qa"]
        
        # Ingest
        print(f"Ingesting {len(sessions)} sessions...")
        system.ingest(sessions)
        print(f"Ingestion Time: {system.ingest_time:.2f}s")
        
        # Evaluate QA
        valid_qa = [qa for qa in qa_pairs if "answer" in qa and str(qa.get("category", "")) != "5"]
        print(f"Evaluating {len(valid_qa)} questions...")
        
        for qa in valid_qa:
            question = qa["question"]
            gold = str(qa["answer"])
            category = str(qa.get("category", "unknown"))
            
            # Retrieve
            context = system.search(question, limit=10)
            
            # Generate
            predicted = generate_answer(context, question)
            
            # Calculate metrics
            bleu = calculate_bleu(predicted, gold)
            f1 = calculate_f1(predicted, gold)
            llm_score = llm_judge_local(question, gold, predicted)
            
            all_results.append({
                "example": i,
                "question": question,
                "gold": gold,
                "predicted": predicted,
                "category": category,
                "bleu": bleu,
                "f1": f1,
                "llm_score": llm_score,
            })
            
            totals["bleu"] += bleu
            totals["f1"] += f1
            totals["llm"] += llm_score
            totals["count"] += 1
            
            # Progress indicator
            if totals["count"] % 10 == 0:
                print(f"  Processed {totals['count']} questions...")
        
        # Cleanup
        system.cleanup()
    
    # Calculate averages
    n = totals["count"]
    summary = {
        "system": system_name,
        "num_examples": num_examples,
        "total_questions": n,
        "avg_bleu": totals["bleu"] / n if n > 0 else 0,
        "avg_f1": totals["f1"] / n if n > 0 else 0,
        "avg_llm_score": totals["llm"] / n if n > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }
    
    print(f"\n--- RESULTS: {system_name.upper()} ---")
    print(f"Questions: {n}")
    print(f"Avg BLEU: {summary['avg_bleu']:.4f}")
    print(f"Avg F1: {summary['avg_f1']:.4f}")
    print(f"Avg LLM Score: {summary['avg_llm_score']:.4f} ({summary['avg_llm_score']*100:.1f}%)")
    
    return {
        "summary": summary,
        "details": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Memory System Benchmark")
    available_systems = list(SYSTEM_MAP.keys()) + ["all"]
    parser.add_argument("--system", type=str, default="amp",
                       choices=available_systems,
                       help="System to benchmark")
    parser.add_argument("--examples", type=int, default=2,
                       help="Number of LoCoMo examples to use")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    parser.add_argument("--llm", type=str, default="ollama",
                       choices=["ollama", "gemini"],
                       help="LLM provider to use")
    
    args = parser.parse_args()
    
    # Initialize LLM
    global llm_provider
    llm_provider = LLMProvider(args.llm)
    print(f"Using LLM: {args.llm.upper()}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    systems_to_run = list(SYSTEM_MAP.keys()) if args.system == "all" else [args.system]
    
    all_summaries = []
    
    for system_name in systems_to_run:
        results = run_benchmark(system_name, args.examples)
        all_summaries.append(results["summary"])
        
        # Save individual results
        output_path = RESULTS_DIR / f"{system_name}_{args.examples}ex.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    # Print comparison table
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'System':<20} {'BLEU':>10} {'F1':>10} {'LLM %':>10}")
        print("-" * 52)
        for s in all_summaries:
            print(f"{s['system']:<20} {s['avg_bleu']:>10.4f} {s['avg_f1']:>10.4f} {s['avg_llm_score']*100:>9.1f}%")


if __name__ == "__main__":
    main()
