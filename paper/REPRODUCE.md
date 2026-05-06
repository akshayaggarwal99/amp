# Reproducing AMP Paper Results

## Hardware
- MacBook Pro M1 Pro, 16 GB unified memory (results reported in the paper)
- macOS 14+
- ~50 GB free disk for models + results

## Setup

```bash
git clone https://github.com/akshayaggarwal99/amp
cd amp
uv pip install -e .
uv pip install matplotlib

# Pull local models
ollama pull gemma4:latest          # 9.6 GB — generator (or use Gemini API)
ollama pull qwen3:8b               # 5.2 GB — judge A
ollama pull deepseek-r1:8b         # 5.2 GB — judge B
ollama pull nomic-embed-text       # 274 MB — Mem0's embedder

# Start Ollama (if not already running)
ollama serve &
```

## Configuration

Set the Gemini API key (used as the controlled generator across all four systems):

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

Or use a fully-local generator:

```bash
python benchmarks/run_locomo_local.py --gen-backend ollama --gen-model gemma4:latest [...]
```

## Dataset

LoCoMo dataset is at `temp_research/MemOS/evaluation/data/locomo/locomo10.json` (vendored from the MemOS evaluation suite). Source: Maharana et al., 2024.

## Run Main Eval

```bash
python benchmarks/run_locomo_local.py \
  --n-conversations 3 \
  --max-questions-per-conv 50 \
  --gen-backend gemini \
  --gen-model gemini-2.5-flash \
  --judges qwen3:8b deepseek-r1:8b \
  --mem0-provider gemini \
  --mem0-llm gemini-2.5-flash \
  --out-dir benchmarks/results/locomo_local
```

Resumable: kill and rerun, completed (system, conversation) pairs are skipped via the existing CSV.

## Run Ablation

```bash
python benchmarks/run_ablation.py \
  --n-conversations 3 \
  --max-questions-per-conv 50 \
  --out-dir benchmarks/results/ablation
```

## Analyze + Generate Figures

```bash
python -m benchmarks.analysis.summarize \
  --results benchmarks/results/locomo_local/results.csv \
  --out-dir benchmarks/results/locomo_local/

python -m benchmarks.analysis.summarize \
  --results benchmarks/results/ablation/results.csv \
  --out-dir benchmarks/results/ablation/

python -m benchmarks.analysis.make_figures \
  --summary-dir benchmarks/results/locomo_local/ \
  --out-dir paper/figures/
```

Copy the LaTeX table fragments into the paper:

```bash
cp benchmarks/results/locomo_local/main_results.tex paper/tables/
cp benchmarks/results/ablation/main_results.tex   paper/tables/ablation.tex
# latency table is generated under benchmarks/results/locomo_local/
# (custom shaping if needed)
```

## Build Paper

```bash
cd paper && make
open main.pdf
```

Requires `pdflatex` (TeX Live or MacTeX) or `tectonic` (`brew install tectonic`).

## Pinned Versions

- Python: 3.10+ (tested 3.12)
- mem0ai: 1.0.1
- fastembed: 0.3+
- Embedding model: BAAI/bge-small-en-v1.5
- Generator: gemini-2.5-flash (REST API)
- Judges: qwen3:8b, deepseek-r1:8b (Ollama)

## Expected Outputs

- `benchmarks/results/locomo_local/results.csv` — raw per-question rows
- `benchmarks/results/locomo_local/main_summary.csv` — accuracy + 95% CI per system
- `benchmarks/results/locomo_local/per_category.csv` — per-category breakdown
- `benchmarks/results/locomo_local/latency.csv` — ingest + query latency
- `benchmarks/results/locomo_local/agreement.txt` — Cohen's kappa
- `benchmarks/results/locomo_local/main_results.tex` — LaTeX table
- `paper/figures/results_main.pdf`, `per_category.pdf`, `latency_accuracy.pdf`
- `paper/main.pdf` — final paper

## Common Issues

- **Ollama not responding**: `ollama serve` must be running; restart if requests time out.
- **Gemini 429 / quota errors**: the `GeminiGenerator` retries with exponential backoff; sustained 429 means free-tier limits hit. Wait or upgrade.
- **OOM on Full-Context eval**: reduce `--max-questions-per-conv` or set `FullContextBaseline(max_tokens=...)` lower.
- **Mem0 API shape changes**: the adapter handles both `list[dict]` and `list[str]` returns; if Mem0 ships a new shape, update `Mem0Adapter.search`.
