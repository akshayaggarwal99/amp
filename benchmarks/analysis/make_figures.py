"""Generate paper PDF figures from summary CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "font.family": "serif", "pdf.fonttype": 42})

CATEGORY_ORDER = ["single_hop", "multi_hop", "temporal_reasoning", "open_domain", "adversarial"]


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fig_main(summary_csv: Path, out_pdf: Path) -> None:
    rows = sorted(read_csv(summary_csv), key=lambda r: float(r["accuracy"]))
    systems = [r["system"] for r in rows]
    accs = [float(r["accuracy"]) * 100 for r in rows]
    los = [float(r["accuracy"]) * 100 - float(r["ci_low"]) * 100 for r in rows]
    his = [float(r["ci_high"]) * 100 - float(r["accuracy"]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.barh(systems, accs, xerr=[los, his], capsize=4, color="#4C72B0")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def fig_per_category(per_cat_csv: Path, out_pdf: Path) -> None:
    rows = read_csv(per_cat_csv)
    systems = sorted({r["system"] for r in rows})
    cats_present = [c for c in CATEGORY_ORDER if c in {r["category"] for r in rows}]
    if not cats_present:
        cats_present = sorted({r["category"] for r in rows})

    grid = {(r["system"], r["category"]): float(r["accuracy"]) * 100 for r in rows}

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    n_sys = len(systems)
    width = 0.8 / max(n_sys, 1)
    for i, sys_name in enumerate(systems):
        ys = [grid.get((sys_name, c), 0.0) for c in cats_present]
        xs = [j + (i - (n_sys - 1) / 2) * width for j in range(len(cats_present))]
        ax.bar(xs, ys, width=width, label=sys_name)

    ax.set_xticks(range(len(cats_present)))
    ax.set_xticklabels([c.replace("_", " ") for c in cats_present], rotation=20)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def fig_latency(latency_csv: Path, summary_csv: Path, out_pdf: Path) -> None:
    lat = read_csv(latency_csv)
    summ = {r["system"]: r for r in read_csv(summary_csv)}

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for r in lat:
        sys_name = r["system"]
        s = summ.get(sys_name)
        if not s:
            continue
        x = float(r["query_median"])
        y = float(s["accuracy"]) * 100
        yerr_lo = y - float(s["ci_low"]) * 100
        yerr_hi = float(s["ci_high"]) * 100 - y
        ax.errorbar(
            x, y,
            yerr=[[yerr_lo], [yerr_hi]],
            marker="o", capsize=4, label=sys_name,
        )
    ax.set_xlabel("Median query latency (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--summary-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig_main(args.summary_dir / "main_summary.csv", args.out_dir / "results_main.pdf")
    fig_per_category(args.summary_dir / "per_category.csv", args.out_dir / "per_category.pdf")
    fig_latency(args.summary_dir / "latency.csv",
                args.summary_dir / "main_summary.csv",
                args.out_dir / "latency_accuracy.pdf")
    print(f"Figures written to {args.out_dir}")
