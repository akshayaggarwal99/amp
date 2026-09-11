# ⚠️ Superseded — do not cite this file

**The previous contents of this file have been withdrawn.** They reported a December 2025
benchmark run whose numbers are **not** the numbers in the paper and must not be quoted as
AMP results.

That run is unsound and is retracted for three reasons:

1.  **A single generous judge.** The judge prompt read, verbatim, *"Be GENEROUS. If the
    prediction touches on the same topic, mark CORRECT."*
2.  **Refusals were scored as correct.** The harness emits the literal string `I don't know.`
    when it retrieves no context, and the generous judge marked those CORRECT. Every accuracy
    in that run was inflated by exactly the artifact the paper was later written to eliminate.
3.  **A different, smaller setup.** 1 conversation instead of 3, and one cloud model used for
    both generation and judging — the self-judging bias the paper's dual-judge protocol exists
    to avoid.

The withdrawn text also attributed a Mem0 failure to "API breaking changes" on Mem0's side.
That attribution was wrong and has been removed. The prior contents remain in this
repository's git history for anyone auditing how the numbers changed.

---

## The current results

`paper/` is the only authoritative source. Headline table, reproduced from
[`paper/tables/main_results.tex`](../paper/tables/main_results.tex), which is generated
directly from [`results/locomo_local/main_summary.csv`](results/locomo_local/main_summary.csv):

LoCoMo, 3 conversations, **N = 150** stratified questions per system, seed 42. Accuracy is
the mean of two independent local judges (`qwen3:8b`, `deepseek-r1:8b`) under the
**strict-IDK rule** — any prediction opening with a refusal phrase is scored WRONG before the
judges are consulted. Intervals are 95% bootstrap CIs, 10,000 resamples.

| System | Accuracy | 95% CI |
| :--- | :---: | :---: |
| Full Context | **82.0%** | (76.3, 87.3) |
| **AMP-Padded2** (recommended default, `w = 2`) | **72.0%** | (65.3, 78.3) |
| Naive RAG (200-word chunks, top-10) | 66.3% | (59.0, 73.3) |
| AMP anchor-only (ablation, `w = 0`) | 60.7% | (53.3, 68.0) |

Paired bootstrap on the per-question differences:

| Comparison | Δ | 95% CI | Reading |
| :--- | :---: | :---: | :--- |
| AMP-Padded2 − Full Context | −10.0 pp | (−15.0, −5.0) | Full Context is **better**; the gap is real |
| AMP-Padded2 − RAG | +5.7 pp | (−0.3, +11.7) | **Not separated** by this sample |
| AMP-Padded2 − AMP anchor-only | +11.3 pp | (+6.3, +16.7) | Neighbor padding helps |

AMP's advantage over Full Context is prompt cost, not accuracy: measured **8.9–16.4×** fewer
input tokens per query.

## Raw data and reproduction

*   Per-question verdicts, both judges, all systems: [`results/locomo_local/results.csv`](results/locomo_local/results.csv)
*   Per-system summary: [`results/locomo_local/main_summary.csv`](results/locomo_local/main_summary.csv)
*   Per-category breakdown: [`results/locomo_local/per_category.csv`](results/locomo_local/per_category.csv)
*   Ingest and query latency: [`results/locomo_local/latency.csv`](results/locomo_local/latency.csv)
*   Step-by-step re-run instructions: [`../paper/REPRODUCE.md`](../paper/REPRODUCE.md)

Full method, per-category analysis, latency table, ablations and limitations: [`../paper/`](../paper/).
