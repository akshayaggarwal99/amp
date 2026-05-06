"""Reads raw results CSV, emits paper-ready summaries + LaTeX tables."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from analysis.bootstrap_ci import bootstrap_mean_ci
from judges.agreement import kappa, raw_agreement


_IDK_PREFIXES = (
    "i don't know", "i don't have", "i do not know",
    "i cannot determine", "i can't determine",
    "i'm not able", "i am not able",
    "i couldn't", "i could not",
)


def _is_idk(prediction: str) -> bool:
    """Heuristic: prediction is a refusal / 'I don't know' response."""
    if not prediction:
        return True
    low = prediction.strip().lower()
    if low.startswith(_IDK_PREFIXES):
        return True
    if low.startswith("i don") and any(
        s in low for s in ("'t know", " not know", "'t have")
    ):
        return True
    return False


def load_rows(csv_path: Path, idk_strict: bool = True) -> list[dict]:
    """Load + dedup on (system, conversation_id, qa_idx) keeping the last.

    If ``idk_strict`` (default True), retroactively override every judge
    column to 0 when the prediction is an ``I don't know`` / refusal. This
    counteracts a strong leniency bias observed in our local LLM judges
    (both Qwen3 and DeepSeek-R1 mark IDK predictions CORRECT $\\sim 80$--$100\\%$
    of the time, which inflates accuracy for systems whose retrieval
    returns empty/irrelevant context).
    """
    with csv_path.open() as f:
        all_rows = list(csv.DictReader(f))
    keyed: dict[tuple, dict] = {}
    for r in all_rows:
        key = (r["system"], r["conversation_id"], r["qa_idx"])
        keyed[key] = r  # later wins
    rows = list(keyed.values())
    if idk_strict:
        for r in rows:
            if _is_idk(r.get("prediction", "")):
                for c in r:
                    if c.startswith("judge_"):
                        r[c] = "0"
    return rows


def judge_cols(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    return [c for c in rows[0].keys() if c.startswith("judge_")]


def mean_judge(row: dict, jcols: list[str]) -> float:
    vals = [int(row[c]) for c in jcols if row.get(c) not in ("", None)]
    return sum(vals) / len(vals) if vals else 0.0


def summarize(csv_path: Path, out_dir: Path) -> None:
    rows = load_rows(csv_path)
    if not rows:
        print(f"No rows in {csv_path}")
        return
    jcols = judge_cols(rows)

    # Main table: per-system accuracy w/ CI
    by_sys: dict[str, list[float]] = {}
    for r in rows:
        by_sys.setdefault(r["system"], []).append(mean_judge(r, jcols))

    main_rows = []
    for system, scores in sorted(by_sys.items(), key=lambda kv: -sum(kv[1]) / max(len(kv[1]), 1)):
        m, lo, hi = bootstrap_mean_ci(scores)
        main_rows.append({
            "system": system, "n": len(scores),
            "accuracy": m, "ci_low": lo, "ci_high": hi,
        })
    with (out_dir / "main_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "n", "accuracy", "ci_low", "ci_high"])
        w.writeheader()
        w.writerows(main_rows)

    # Per-category
    by_sys_cat: dict[tuple, list[float]] = {}
    for r in rows:
        by_sys_cat.setdefault((r["system"], r["category"]), []).append(mean_judge(r, jcols))

    cat_rows = []
    for (system, cat), scores in by_sys_cat.items():
        m, lo, hi = bootstrap_mean_ci(scores)
        cat_rows.append({
            "system": system, "category": cat, "n": len(scores),
            "accuracy": m, "ci_low": lo, "ci_high": hi,
        })
    with (out_dir / "per_category.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "category", "n", "accuracy", "ci_low", "ci_high"])
        w.writeheader()
        w.writerows(cat_rows)

    # Latency table
    lat_rows = []
    for system, scores in by_sys.items():
        sub = [r for r in rows if r["system"] == system]
        ingest = sorted(float(r["ingest_seconds"]) for r in sub)
        query = sorted(float(r["query_seconds"]) for r in sub)
        lat_rows.append({
            "system": system,
            "ingest_median": ingest[len(ingest) // 2] if ingest else 0.0,
            "query_median": query[len(query) // 2] if query else 0.0,
            "query_p95": query[int(len(query) * 0.95)] if query else 0.0,
        })
    with (out_dir / "latency.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "ingest_median", "query_median", "query_p95"])
        w.writeheader()
        w.writerows(lat_rows)

    # Inter-judge agreement
    if len(jcols) >= 2:
        a = [int(r[jcols[0]]) for r in rows if r[jcols[0]] not in ("", None)
             and r[jcols[1]] not in ("", None)]
        b = [int(r[jcols[1]]) for r in rows if r[jcols[0]] not in ("", None)
             and r[jcols[1]] not in ("", None)]
        with (out_dir / "agreement.txt").open("w") as f:
            f.write(f"judge_a: {jcols[0]}\n")
            f.write(f"judge_b: {jcols[1]}\n")
            f.write(f"n: {len(a)}\n")
            f.write(f"cohen_kappa: {kappa(a, b):.4f}\n")
            f.write(f"raw_agreement: {raw_agreement(a, b):.4f}\n")

    # LaTeX main table
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"System & N & Accuracy & 95\% CI \\",
        r"\midrule",
    ]
    for r in main_rows:
        lines.append(
            f"{r['system']} & {r['n']} & {r['accuracy']*100:.1f}\\% & "
            f"({r['ci_low']*100:.1f}, {r['ci_high']*100:.1f}) \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (out_dir / "main_results.tex").write_text("\n".join(lines))

    print("Wrote:")
    for name in ("main_summary.csv", "per_category.csv", "latency.csv",
                 "agreement.txt", "main_results.tex"):
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summarize(args.results, args.out_dir)
