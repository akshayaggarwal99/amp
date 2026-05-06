"""Full LoCoMo evaluation.

Generator: Gemini 2.5 Flash via API (fast, controlled, identical across all 4
memory systems so the only variable is the memory system itself).
Judges: two LOCAL Ollama models (qwen3:8b + deepseek-r1:8b) from non-Google
families, dispatched in parallel per question. Cohen's kappa reported.

Outputs:
    benchmarks/results/locomo_local/results.csv

Each row: system, conv_id, qa_idx, category, question, ground_truth,
prediction, ingest_seconds, query_seconds + one column per judge model.

Resumable: skips conversation/system pairs already in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from judges.local_judge import LocalJudge
from judges.gemini_generator import GeminiGenerator
from judges.generator import OllamaGenerator

CATEGORY_MAP = {
    1: "multi_hop",
    2: "temporal_reasoning",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def load_locomo(path: Path) -> list:
    return json.loads(path.read_text())


def extract_sessions(conversation: dict) -> list[list[dict]]:
    session_keys = sorted(
        [k for k in conversation if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda k: int(k.split("_")[1]),
    )
    sessions = []
    for k in session_keys:
        turns = []
        for turn in conversation[k]:
            turns.append({"speaker": turn.get("speaker", "user"), "text": turn.get("text", "")})
        sessions.append(turns)
    return sessions


def stratified_sample(qa_pairs: list, max_total: int, seed: int = 42) -> list:
    """Sample up to max_total questions, stratified by category. Stable across runs."""
    if max_total <= 0 or len(qa_pairs) <= max_total:
        return list(qa_pairs)
    rng = random.Random(seed)
    by_cat: dict[int, list] = {}
    for qa in qa_pairs:
        by_cat.setdefault(qa.get("category", 0), []).append(qa)

    n_cats = len(by_cat)
    per_cat_target = max(1, max_total // n_cats)
    sample: list = []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        sample.extend(items[: per_cat_target])

    if len(sample) < max_total:
        # Fill remainder from leftovers
        leftover = [qa for cat, items in by_cat.items() for qa in items[per_cat_target:]]
        rng.shuffle(leftover)
        sample.extend(leftover[: max_total - len(sample)])
    elif len(sample) > max_total:
        rng.shuffle(sample)
        sample = sample[:max_total]
    return sample


def build_systems(mem0_provider: str, mem0_llm: str, amp_padded_window: int = 0):
    systems = []
    from systems.amp_adapter import AmpAdapter

    systems.append(("AMP", AmpAdapter(db_path=str(ROOT / "benchmark_amp_run.db"))))

    if amp_padded_window > 0:
        systems.append((
            f"AMP-Padded{amp_padded_window}",
            AmpAdapter(
                db_path=str(ROOT / f"benchmark_amp_padded{amp_padded_window}.db"),
                neighbor_window=amp_padded_window,
            ),
        ))

    from systems.rag_baseline import RagBaseline

    systems.append(("RAG", RagBaseline(chunk_size=200)))

    from systems.full_context import FullContextBaseline

    systems.append(("FullContext", FullContextBaseline(max_tokens=50_000)))

    try:
        from systems.mem0_adapter import Mem0Adapter

        systems.append(("Mem0", Mem0Adapter(llm_provider=mem0_provider, llm_model=mem0_llm)))
    except Exception as e:
        print(f"[warn] Mem0Adapter unavailable: {e}", file=sys.stderr)

    return systems


def already_done(csv_path: Path, system_name: str, conv_id: str, expected_count: int) -> bool:
    if not csv_path.exists():
        return False
    found = 0
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["system"] == system_name and row["conversation_id"] == conv_id:
                found += 1
    return found >= expected_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=Path,
        default=ROOT / "temp_research/MemOS/evaluation/data/locomo/locomo10.json",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks/results/locomo_local")
    p.add_argument("--gen-backend", choices=["gemini", "ollama"], default="gemini")
    p.add_argument("--gen-model", default="gemini-2.5-flash")
    p.add_argument("--mem0-provider", default="gemini",
                   choices=["gemini", "ollama"],
                   help="LLM provider for Mem0's internal extraction step")
    p.add_argument("--mem0-llm", default="gemini-2.5-flash",
                   help="LLM model name for Mem0's internal extraction step")
    p.add_argument("--judges", nargs="+", default=["qwen3:8b", "deepseek-r1:8b"])
    p.add_argument("--n-conversations", type=int, default=10)
    p.add_argument("--max-questions-per-conv", type=int, default=50,
                   help="Stratified by category; 0 disables sampling")
    p.add_argument("--amp-padded-window", type=int, default=0,
                   help="If >0, also evaluate AMP-PaddedN with neighbor padding")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke run: 1 conversation, max 5 questions per system")
    args = p.parse_args()

    if args.smoke:
        args.n_conversations = 1
        args.max_questions_per_conv = 5

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "results.csv"

    data = load_locomo(args.data)[: args.n_conversations]
    print(f"Loaded {len(data)} LoCoMo conversations from {args.data}", flush=True)

    if args.gen_backend == "gemini":
        generator = GeminiGenerator(model=args.gen_model)
    else:
        generator = OllamaGenerator(model=args.gen_model)

    judges = [(m, LocalJudge(model=m)) for m in args.judges]
    judge_cols = [f"judge_{m.replace(':', '_').replace('/', '_')}" for m, _ in judges]
    judge_pool = ThreadPoolExecutor(max_workers=len(judges))

    fieldnames = [
        "system", "conversation_id", "qa_idx", "category", "question",
        "ground_truth", "prediction", "ingest_seconds", "query_seconds",
    ] + judge_cols

    write_header = not csv_path.exists()
    out = csv_path.open("a", newline="")
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        out.flush()

    total_rows = 0
    t_total_start = time.time()

    for conv_idx, conv in enumerate(data):
        conv_id = conv.get("sample_id", f"conv_{conv_idx}")
        sessions = extract_sessions(conv["conversation"])
        all_qa = [qa for qa in conv["qa"] if "answer" in qa]
        qa_pairs = stratified_sample(all_qa, args.max_questions_per_conv)

        n_turns = sum(len(s) for s in sessions)
        print(f"\n=== conv {conv_idx+1}/{len(data)} ({conv_id}): "
              f"{len(sessions)} sessions, {n_turns} turns, "
              f"{len(qa_pairs)}/{len(all_qa)} qs sampled ===", flush=True)

        systems = build_systems(args.mem0_provider, args.mem0_llm, args.amp_padded_window)
        for sys_name, adapter in systems:
            if already_done(csv_path, sys_name, conv_id, len(qa_pairs)):
                print(f"  [{sys_name}] already in CSV — skipping", flush=True)
                continue

            t0 = time.time()
            try:
                adapter.reset()
            except Exception as e:
                print(f"  [{sys_name}] reset failed: {e}", file=sys.stderr, flush=True)
                continue

            try:
                adapter.ingest(sessions)
            except Exception as e:
                print(f"  [{sys_name}] ingest failed: {e}", file=sys.stderr, flush=True)
                continue
            ingest_t = time.time() - t0
            print(f"  [{sys_name}] ingest: {ingest_t:.1f}s", flush=True)

            for qa_idx, qa in enumerate(qa_pairs):
                question = qa["question"]
                gt = str(qa["answer"])
                category = CATEGORY_MAP.get(qa.get("category"), "unknown")

                tq = time.time()
                try:
                    context = adapter.search(question, limit=10)
                except Exception as e:
                    context = ""
                    print(f"  [{sys_name}] search err q{qa_idx}: {e}",
                          file=sys.stderr, flush=True)

                try:
                    prediction = generator.generate(context, question)
                except Exception as e:
                    prediction = f"[gen error: {e}]"

                # Parallel judges
                future_to_col = {
                    judge_pool.submit(j.score, question, gt, prediction): col
                    for (m, j), col in zip(judges, judge_cols)
                }
                judge_scores: dict[str, int | str] = {}
                for fut, col in future_to_col.items():
                    try:
                        judge_scores[col] = fut.result()
                    except Exception as e:
                        judge_scores[col] = ""
                        print(f"  [{sys_name}] judge {col} err q{qa_idx}: {e}",
                              file=sys.stderr, flush=True)

                query_t = time.time() - tq

                row = {
                    "system": sys_name,
                    "conversation_id": conv_id,
                    "qa_idx": qa_idx,
                    "category": category,
                    "question": question,
                    "ground_truth": gt,
                    "prediction": prediction,
                    "ingest_seconds": round(ingest_t, 3),
                    "query_seconds": round(query_t, 3),
                    **judge_scores,
                }

                writer.writerow(row)
                out.flush()
                total_rows += 1
                if (qa_idx + 1) % 10 == 0:
                    elapsed = time.time() - t_total_start
                    rate = total_rows / max(elapsed, 1.0)
                    print(f"    q{qa_idx+1}/{len(qa_pairs)} | "
                          f"verdicts={[row[c] for c in judge_cols]} | "
                          f"{rate:.2f} rows/s overall", flush=True)

            try:
                adapter.cleanup()
            except Exception:
                pass

    out.close()
    judge_pool.shutdown(wait=True)
    total_elapsed = time.time() - t_total_start
    print(f"\nWrote {total_rows} rows to {csv_path} in {total_elapsed/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
