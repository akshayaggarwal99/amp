"""Recompute every number in the paper from the released per-question CSV.

Usage:  python benchmarks/analysis/recompute_kappa.py [path/to/results.csv]

Uses the benchmark's own refusal heuristic, kappa, raw-agreement and
bootstrap implementations (benchmarks/analysis, benchmarks/judges). The
three extra filters in Table 3 are defined here, on the prediction text
alone; none reads the gold answer or a judge column. Raking uses iterative
proportional fitting on the 2x2 judge table, which preserves the odds ratio.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

from analysis.bootstrap_ci import bootstrap_mean_ci          # noqa: E402
from analysis.summarize import _is_idk, load_rows             # noqa: E402
from judges.agreement import kappa, raw_agreement             # noqa: E402

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "benchmarks/results/locomo_local/results.csv"
SYSTEMS = ["FullContext", "AMP-Padded2", "RAG", "AMP"]
A, B = "judge_qwen3_8b", "judge_deepseek-r1_8b"

# ------------------------------------------------------------------ filters
_DEIXIS = (
    "yesterday", "today", "tomorrow", "last week", "last month", "last year",
    "this week", "this month", "this year", "next week", "next month",
    "days ago", "weeks ago", "months ago", "years ago", "recently",
    "last night", "last tuesday", "last friday", "a few days",
)


def f_refusal(p: str) -> bool:
    """The paper's rule: nine refusal prefixes (benchmarks/analysis/summarize.py)."""
    return _is_idk(p)


def f_truncated(p: str) -> bool:
    """Format validator: a complete answer ends in terminal punctuation."""
    p = p.strip()
    return bool(p) and p[-1] not in ".!?"


def f_short(p: str) -> bool:
    """Minimum-length guard: fewer than 30 characters."""
    return len(p.strip()) < 30


def f_deixis(p: str) -> bool:
    """Self-containment check: answer contains an unresolved relative time expression."""
    low = p.lower()
    return any(d in low for d in _DEIXIS)


FILTERS = [("refusal", f_refusal), ("truncation", f_truncated),
           ("min length < 30", f_short), ("unresolved deixis", f_deixis)]


# ------------------------------------------------------------------ helpers
def vec(rows, col):
    return [int(r[col]) for r in rows]


def apply(rows, fn):
    out = [dict(r) for r in rows]
    for r in out:
        if fn(r["prediction"]):
            r[A] = r[B] = "0"
    return out


def p_e(a, b):
    n = len(a)
    pa, pb = sum(a) / n, sum(b) / n
    return pa * pb + (1 - pa) * (1 - pb)


def table(rows):
    c = Counter((int(r[A]), int(r[B])) for r in rows)
    return {k: c[k] for k in [(0, 0), (0, 1), (1, 0), (1, 1)]}


def kappa_from_table(t):
    n = sum(t.values())
    po = (t[(0, 0)] + t[(1, 1)]) / n
    pa = (t[(1, 0)] + t[(1, 1)]) / n
    pb = (t[(0, 1)] + t[(1, 1)]) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


def rake(t, target, iters=500):
    """IPF to equal row and column marginals P(1)=target; odds ratio preserved."""
    n = sum(t.values())
    t = {k: float(v) for k, v in t.items()}
    for _ in range(iters):
        r1 = t[(1, 0)] + t[(1, 1)]; r0 = n - r1
        for k in t:
            t[k] *= (target * n / r1) if k[0] == 1 else ((1 - target) * n / r0)
        c1 = t[(0, 1)] + t[(1, 1)]; c0 = n - c1
        for k in t:
            t[k] *= (target * n / c1) if k[1] == 1 else ((1 - target) * n / c0)
    return t


def pct(x):
    return f"{100 * x:.1f}"


# ------------------------------------------------------------------ main
rows = [r for r in load_rows(CSV, idk_strict=False) if r["system"] in SYSTEMS]
assert len(rows) == 600, len(rows)
T = [r for r in rows if f_refusal(r["prediction"])]
U = [r for r in rows if not f_refusal(r["prediction"])]
post = apply(rows, f_refusal)
a, b = vec(rows, A), vec(rows, B)
a2, b2 = vec(post, A), vec(post, B)
au, bu = vec(U, A), vec(U, B)

print(f"sample: {len(rows)} rows, |T|={len(T)}, |U|={len(U)}")
print("categories:", dict(Counter(r["category"] for r in rows if r["system"] == "AMP")))
print("refusals per system:", {s: sum(1 for r in T if r["system"] == s) for s in SYSTEMS})
print("refusal categories:", dict(Counter(r["category"] for r in T)))

print("\n[Table 1] judge leniency on refusals (% scored CORRECT)")
for s in SYSTEMS + ["All"]:
    ts = [r for r in T if s == "All" or r["system"] == s]
    print(f"  {s:12s} n={len(ts):3d}  qwen {pct(sum(vec(ts, A)) / len(ts))}  "
          f"deepseek {pct(sum(vec(ts, B)) / len(ts))}")

print("\n[Table 2] agreement")
print(f"  all  pre : raw {pct(raw_agreement(a, b))}  kappa {kappa(a, b):.4f}")
print(f"  all  post: raw {pct(raw_agreement(a2, b2))}  kappa {kappa(a2, b2):.4f}")
at, bt = vec(T, A), vec(T, B)
print(f"  T    pre : raw {pct(raw_agreement(at, bt))}  kappa {kappa(at, bt):.4f}")
print(f"  U        : raw {pct(raw_agreement(au, bu))}  kappa {kappa(au, bu):.4f} (pre = post)")
print(f"  2x2 pre  {table(rows)}\n  2x2 post {table(post)}\n  2x2 U    {table(U)}")

print("\n[marginals, % CORRECT]  qwen / deepseek")
for name, rs in [("all pre", rows), ("U", U), ("all post", post)]:
    print(f"  {name:8s} {pct(sum(vec(rs, A)) / len(rs))} / {pct(sum(vec(rs, B)) / len(rs))}")

print("\n[constant-rater control, pre-rule, N=600]")
const = [1] * len(rows)
print(f"  deepseek actual : raw {100 * raw_agreement(a, b):.2f}  kappa {kappa(a, b):.4f}")
print(f"  always CORRECT  : raw {100 * raw_agreement(a, const):.2f}  kappa {kappa(a, const):.4f}")

print("\n[Table 3] four filters, channel decomposition")
po_pre, pe_pre = raw_agreement(a, b), p_e(a, b)
for name, fn in FILTERS:
    o = apply(rows, fn)
    x, y = vec(o, A), vec(o, B)
    n_t = sum(1 for r in rows if fn(r["prediction"]))
    overlap = sum(1 for r in rows if fn(r["prediction"]) and f_refusal(r["prediction"]))
    forced = (raw_agreement(x, y) - pe_pre) / (1 - pe_pre)
    marg = (po_pre - p_e(x, y)) / (1 - p_e(x, y))
    print(f"  {name:18s} |T|={n_t:3d} overlap={overlap:3d}  kappa post {kappa(x, y):.3f}  "
          f"forced-only {forced:.3f}  marginal-only {marg:.3f}")

print("\n[Table 4] raking to common marginal, odds ratio preserved")
t0 = table(rows)
OR = t0[(0, 0)] * t0[(1, 1)] / (t0[(0, 1)] * t0[(1, 0)])
print(f"  observed odds ratio {OR:.2f}")
tT = table(T)
for tg in [0.50, 0.70, 0.90, 0.988]:
    t = rake(t0, tg)
    # Apply the rule on the raked table: the refusal rows keep their observed
    # per-cell composition, scaled by the same cell multipliers raking applied,
    # and are moved into (0,0).
    mult = {k: t[k] / t0[k] for k in t0}
    tTr = {k: tT[k] * mult[k] for k in tT}
    post_t = {k: t[k] - tTr[k] for k in t}
    post_t[(0, 0)] += sum(tTr.values())
    kpre, kpost = kappa_from_table(t), kappa_from_table(post_t)
    print(f"  target {tg:.3f}: kappa pre {kpre:.3f}  post {kpost:.3f}  gain {kpost - kpre:+.3f}")

print("\n[Appendix] accuracy, mean of judges, 95% bootstrap CI, seed 42")
for s in SYSTEMS:
    for lab, rs in [("pre", rows), ("post", post)]:
        sc = [(int(r[A]) + int(r[B])) / 2 for r in rs if r["system"] == s]
        m, lo, hi = bootstrap_mean_ci(sc)
        print(f"  {s:12s} {lab:4s} {pct(m)} ({pct(lo)}, {pct(hi)})")

print("\n[Appendix] paired bootstrap on per-question differences")
def keyed(rs, s):
    return {(r["conversation_id"], r["qa_idx"]): (int(r[A]) + int(r[B])) / 2
            for r in rs if r["system"] == s}
for lab, rs, pairs in [("pre", rows, [("FullContext", "RAG"), ("RAG", "AMP-Padded2"), ("AMP-Padded2", "AMP")]),
                       ("post", post, [("FullContext", "AMP-Padded2"), ("AMP-Padded2", "RAG"), ("RAG", "AMP")])]:
    for s1, s2 in pairs:
        k1, k2 = keyed(rs, s1), keyed(rs, s2)
        d = [k1[k] - k2[k] for k in k1]
        m, lo, hi = bootstrap_mean_ci(d)
        print(f"  {lab:4s} {s1} - {s2}: {100*m:+.1f} ({100*lo:+.1f}, {100*hi:+.1f})")

print("\n[Appendix] per-judge accuracy alone")
for s in SYSTEMS:
    pr = [r for r in rows if r["system"] == s]; po = [r for r in post if r["system"] == s]
    print(f"  {s:12s} pre {pct(sum(vec(pr, A))/150)} / {pct(sum(vec(pr, B))/150)}   "
          f"post {pct(sum(vec(po, A))/150)} / {pct(sum(vec(po, B))/150)}")

print("\n[Appendix] per-system disagreements")
for s in SYSTEMS:
    ts = [r for r in T if r["system"] == s]; us = [r for r in U if r["system"] == s]
    print(f"  {s:12s} T {len(ts):3d} disagree {sum(r[A] != r[B] for r in ts):2d}   "
          f"U {len(us):3d} disagree {sum(r[A] != r[B] for r in us):2d}")

print("\n[Appendix] closed form under a representative-sample override")
n = len(rows)
po = raw_agreement(a, b); pA = sum(a) / n; pB = sum(b) / n
def kappa_closed(f):
    po2 = (1 - f) * po + f; pA2 = (1 - f) * pA; pB2 = (1 - f) * pB
    pe2 = pA2 * pB2 + (1 - pA2) * (1 - pB2)
    return (po2 - pe2) / (1 - pe2)
print(f"  p_o {po:.3f} p_A {pA:.3f} p_B {pB:.3f}")
for f in [0.05, 0.10, len(T) / n, 0.30]:
    print(f"  f={f:.3f}: predicted kappa post {kappa_closed(f):.3f}")
