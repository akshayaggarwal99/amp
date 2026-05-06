#!/usr/bin/env python3
"""
Scientific Benchmark Harness for AMP Open Source Release.

- Same LLM (Ollama) for ALL systems
- Same Embedder (FastEmbed) for ALL systems
- Category breakdown (1-4)
- Statistical metrics (mean ± std)
- Latency tracking

Usage:
    python benchmarks/run_scientific_benchmark.py --examples 10
"""
import argparse
import json
import time
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.metrics import calculate_bleu, calculate_f1
from benchmarks.systems import AmpAdapter, RagBaseline, FullContextBaseline

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"
DATASET_PATH = "temp_research/MemOS/evaluation/data/locomo/locomo10.json"
RESULTS_DIR = Path("benchmarks/results")

# ALL systems use same local providers
SYSTEM_MAP = {
    "amp": AmpAdapter,
    "rag_baseline": RagBaseline,
    "full_context": FullContextBaseline,
}

# Try to add competitors if available
try:
    from benchmarks.systems import Mem0Adapter, OpenMemoryAdapter, CogneeAdapter, MemoriAdapter
    if Mem0Adapter: SYSTEM_MAP["mem0"] = Mem0Adapter
    if OpenMemoryAdapter: SYSTEM_MAP["openmemory"] = OpenMemoryAdapter
    if CogneeAdapter: SYSTEM_MAP["cognee"] = CogneeAdapter
    if MemoriAdapter: SYSTEM_MAP["memori"] = MemoriAdapter
except ImportError:
    pass


def ollama_complete(prompt: str, max_tokens: int = 500) -> str:
    """Use local Ollama for generation."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens}
        }, timeout=120)
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"


def generate_answer(context: str, question: str) -> str:
    """Generate answer using Ollama."""
    if not context.strip():
        return "I don't know."
    
    prompt = f"""You are a helpful assistant answering based on context.
Answer concisely. If unsure, say "I don't know."

Context:
{context[:8000]}

Question: {question}
Answer:"""
    return ollama_complete(prompt)


def llm_judge(question: str, gold: str, predicted: str) -> int:
    """Use Ollama as judge. Returns 1 (correct) or 0 (wrong)."""
    prompt = f"""Judge if the prediction is correct vs ground truth.
Be generous: semantically equivalent = CORRECT.
Temporal matches (e.g. "last week" = specific date) = CORRECT.

Question: {question}
Ground Truth: {gold}
Prediction: {predicted}

Respond with ONE word: CORRECT or WRONG."""
    
    result = ollama_complete(prompt, max_tokens=10)
    return 1 if "CORRECT" in result.upper() else 0


def extract_sessions(conversation: dict) -> list:
    """Extract sessions from LoCoMo conversation."""
    sessions = []
    keys = [k for k in conversation.keys() 
            if k.startswith("session_") and not k.endswith("date_time")]
    keys.sort(key=lambda x: int(x.split("_")[1]))
    for key in keys:
        sessions.append(conversation[key])
    return sessions


def run_benchmark(system_name: str, num_examples: int) -> dict:
    """Run benchmark for a single system."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {system_name.upper()}")
    print(f"{'='*60}")
    
    SystemClass = SYSTEM_MAP[system_name]
    system = SystemClass()
    
    with open(DATASET_PATH) as f:
        data = json.load(f)
    
    data_subset = data[:num_examples]
    
    # Results by category
    results = []
    category_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    latencies = {"ingest": [], "search": [], "generate": [], "judge": []}
    
    for i, example in enumerate(data_subset):
        print(f"\n--- Example {i+1}/{len(data_subset)} ---")
        
        system.reset()
        
        conversation = example["conversation"]
        sessions = extract_sessions(conversation)
        qa_pairs = example["qa"]
        
        # Ingest with timing
        print(f"Ingesting {len(sessions)} sessions...")
        t0 = time.time()
        system.ingest(sessions)
        latencies["ingest"].append(time.time() - t0)
        print(f"Ingestion: {latencies['ingest'][-1]:.2f}s")
        
        # Skip category 5 (unanswerable)
        valid_qa = [qa for qa in qa_pairs 
                    if "answer" in qa and str(qa.get("category", "")) != "5"]
        print(f"Evaluating {len(valid_qa)} questions...")
        
        for j, qa in enumerate(valid_qa):
            question = qa["question"]
            gold = str(qa["answer"])
            category = str(qa.get("category", "unknown"))
            
            # Search with timing
            t0 = time.time()
            context = system.search(question, limit=10)
            latencies["search"].append(time.time() - t0)
            
            # Generate with timing
            t0 = time.time()
            predicted = generate_answer(context, question)
            latencies["generate"].append(time.time() - t0)
            
            # Calculate metrics
            bleu = calculate_bleu(predicted, gold)
            f1 = calculate_f1(predicted, gold)
            
            t0 = time.time()
            llm_score = llm_judge(question, gold, predicted)
            latencies["judge"].append(time.time() - t0)
            
            results.append({
                "example": i,
                "category": category,
                "question": question[:50] + "...",
                "gold": gold[:50],
                "predicted": predicted[:50],
                "bleu": bleu,
                "f1": f1,
                "llm_score": llm_score,
            })
            
            category_scores[category]["correct"] += llm_score
            category_scores[category]["total"] += 1
            
            if (j + 1) % 20 == 0:
                print(f"  Processed {j+1}/{len(valid_qa)} questions...")
        
        system.cleanup()
    
    # Calculate statistics
    n = len(results)
    bleu_scores = [r["bleu"] for r in results]
    f1_scores = [r["f1"] for r in results]
    llm_scores = [r["llm_score"] for r in results]
    
    summary = {
        "system": system_name,
        "num_examples": num_examples,
        "total_questions": n,
        "metrics": {
            "bleu": {
                "mean": statistics.mean(bleu_scores) if bleu_scores else 0,
                "std": statistics.stdev(bleu_scores) if len(bleu_scores) > 1 else 0,
            },
            "f1": {
                "mean": statistics.mean(f1_scores) if f1_scores else 0,
                "std": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0,
            },
            "llm_score": {
                "mean": statistics.mean(llm_scores) if llm_scores else 0,
                "std": statistics.stdev(llm_scores) if len(llm_scores) > 1 else 0,
            },
        },
        "category_breakdown": {
            cat: data["correct"] / data["total"] if data["total"] > 0 else 0
            for cat, data in category_scores.items()
        },
        "latency_ms": {
            "ingest": statistics.mean(latencies["ingest"]) * 1000 if latencies["ingest"] else 0,
            "search": statistics.mean(latencies["search"]) * 1000 if latencies["search"] else 0,
            "generate": statistics.mean(latencies["generate"]) * 1000 if latencies["generate"] else 0,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {system_name.upper()}")
    print(f"{'='*60}")
    print(f"Questions: {n}")
    print(f"BLEU: {summary['metrics']['bleu']['mean']:.4f} ± {summary['metrics']['bleu']['std']:.4f}")
    print(f"F1:   {summary['metrics']['f1']['mean']:.4f} ± {summary['metrics']['f1']['std']:.4f}")
    print(f"LLM Score: {summary['metrics']['llm_score']['mean']*100:.1f}% ± {summary['metrics']['llm_score']['std']*100:.1f}%")
    print(f"\nPer-Category Accuracy:")
    for cat, score in sorted(summary['category_breakdown'].items()):
        print(f"  Cat {cat}: {score*100:.1f}%")
    print(f"\nLatency:")
    print(f"  Ingest: {summary['latency_ms']['ingest']:.0f}ms")
    print(f"  Search: {summary['latency_ms']['search']:.1f}ms")
    print(f"  Generate: {summary['latency_ms']['generate']:.0f}ms")
    
    return {"summary": summary, "details": results}


def main():
    parser = argparse.ArgumentParser(description="Scientific Memory System Benchmark")
    available_systems = list(SYSTEM_MAP.keys()) + ["all"]
    parser.add_argument("--system", type=str, default="all", help="System to benchmark (or 'all', or comma-separated list)")
    parser.add_argument("--examples", type=int, default=10,
                       help="Number of LoCoMo examples (max 10)")
    
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.system == "all":
        systems_to_run = sorted(SYSTEM_MAP.keys())
    else:
        systems_to_run = args.system.split(",")
        systems_to_run = [s.strip() for s in systems_to_run if s.strip() in SYSTEM_MAP]
    all_summaries = []
    
    for sys_name in systems_to_run:
        result = run_benchmark(sys_name, args.examples)
        all_summaries.append(result["summary"])
        
        # Save results
        output_path = RESULTS_DIR / f"{sys_name}_scientific_{args.examples}ex.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {output_path}")
    
    # Final comparison table
    if len(all_summaries) > 1:
        print(f"\n{'='*70}")
        print("SCIENTIFIC COMPARISON (Same LLM, Same Embedder)")
        print(f"{'='*70}")
        print(f"{'System':<15} {'BLEU':>12} {'F1':>12} {'LLM %':>10} {'Search ms':>12}")
        print("-" * 61)
        for s in all_summaries:
            m = s['metrics']
            print(f"{s['system']:<15} "
                  f"{m['bleu']['mean']:.4f}±{m['bleu']['std']:.2f} "
                  f"{m['f1']['mean']:.4f}±{m['f1']['std']:.2f} "
                  f"{m['llm_score']['mean']*100:>6.1f}% "
                  f"{s['latency_ms']['search']:>10.1f}")


if __name__ == "__main__":
    main()
