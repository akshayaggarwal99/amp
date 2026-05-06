"""Ablation: AMP variants compared to AMP-Full.

Variants:
- AMP-NoRecency: alpha*cos only (recency_weight=0)
- AMP-Full: alpha*cos + beta*recency (default settings)

Both consolidate STM->LTM at threshold tau_stm=50.

Output: benchmarks/results/ablation/results.csv (same schema as main runner).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from judges.local_judge import LocalJudge
from judges.gemini_generator import GeminiGenerator
from amp.client import Amp
from amp.core.storage import Storage

# Mirror the structure of run_locomo_local for consistent CSV schema
from run_locomo_local import (  # type: ignore
    extract_sessions, stratified_sample, load_locomo, CATEGORY_MAP, already_done,
)


class AmpVariant:
    def __init__(self, name: str, db_path: str, recency_weight: float):
        self.name = name
        self.db_path = db_path
        self.recency_weight = recency_weight
        self.brain: Amp | None = None
        self.ingest_time = 0.0

    def reset(self) -> None:
        from pathlib import Path as _P
        if _P(self.db_path).exists():
            _P(self.db_path).unlink()
        self.brain = Amp.__new__(Amp)
        self.brain.storage = Storage(
            db_path=self.db_path,
            recency_weight=self.recency_weight,
        )

    def ingest(self, sessions: list[list[dict]]) -> None:
        t0 = time.time()
        for session in sessions:
            for turn in session:
                msg = f"{turn['speaker']}: {turn['text']}"
                self.brain.remember(msg, {"source": "ablation"})
            self.brain.consolidate()
        self.ingest_time = time.time() - t0

    def search(self, query: str, limit: int = 10) -> str:
        results = self.brain.recall(query, limit=limit)
        return "\n".join([f"- {r['content']}" for r in results])

    def cleanup(self) -> None:
        try:
            self.brain.storage.connection.close()
        except Exception:
            pass
        self.brain = None
        from pathlib import Path as _P
        if _P(self.db_path).exists():
            _P(self.db_path).unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=ROOT / "temp_research/MemOS/evaluation/data/locomo/locomo10.json")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "benchmarks/results/ablation")
    p.add_argument("--gen-model", default="gemini-2.5-flash")
    p.add_argument("--judges", nargs="+", default=["qwen3:8b", "deepseek-r1:8b"])
    p.add_argument("--n-conversations", type=int, default=10)
    p.add_argument("--max-questions-per-conv", type=int, default=50)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.n_conversations = 1
        args.max_questions_per_conv = 5

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "results.csv"

    data = load_locomo(args.data)[: args.n_conversations]
    print(f"Loaded {len(data)} conversations", flush=True)

    generator = GeminiGenerator(model=args.gen_model)
    judges = [(m, LocalJudge(model=m)) for m in args.judges]
    judge_cols = [f"judge_{m.replace(':', '_').replace('/', '_')}" for m, _ in judges]
    judge_pool = ThreadPoolExecutor(max_workers=len(judges))

    variants = [
        AmpVariant("AMP-NoRecency", str(ROOT / "ablation_norecency.db"), recency_weight=0.0),
        AmpVariant("AMP-Full", str(ROOT / "ablation_full.db"), recency_weight=0.1),
    ]

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

    t_total = time.time()
    total_rows = 0

    for conv_idx, conv in enumerate(data):
        conv_id = conv.get("sample_id", f"conv_{conv_idx}")
        sessions = extract_sessions(conv["conversation"])
        all_qa = [qa for qa in conv["qa"] if "answer" in qa]
        qa_pairs = stratified_sample(all_qa, args.max_questions_per_conv)

        print(f"\n=== conv {conv_idx+1}/{len(data)} ({conv_id}): "
              f"{len(qa_pairs)} qs ===", flush=True)

        for variant in variants:
            if already_done(csv_path, variant.name, conv_id, len(qa_pairs)):
                print(f"  [{variant.name}] skipping (in CSV)", flush=True)
                continue
            variant.reset()
            variant.ingest(sessions)
            print(f"  [{variant.name}] ingest: {variant.ingest_time:.1f}s", flush=True)

            for qa_idx, qa in enumerate(qa_pairs):
                question = qa["question"]
                gt = str(qa["answer"])
                category = CATEGORY_MAP.get(qa.get("category"), "unknown")

                tq = time.time()
                context = variant.search(question, limit=10)
                prediction = generator.generate(context, question)
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
                        print(f"    judge {col} err: {e}", file=sys.stderr, flush=True)
                qt = time.time() - tq

                row = {
                    "system": variant.name,
                    "conversation_id": conv_id,
                    "qa_idx": qa_idx,
                    "category": category,
                    "question": question,
                    "ground_truth": gt,
                    "prediction": prediction,
                    "ingest_seconds": round(variant.ingest_time, 3),
                    "query_seconds": round(qt, 3),
                    **judge_scores,
                }
                writer.writerow(row)
                out.flush()
                total_rows += 1

            variant.cleanup()

    out.close()
    judge_pool.shutdown(wait=True)
    print(f"\nWrote {total_rows} rows in {(time.time()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
