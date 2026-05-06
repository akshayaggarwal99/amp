# Paper Outline — AMP

**Working title:** AMP: A Cognitive-Architecture Memory Protocol for Long-Horizon LLM Agents

**Target length:** 8–10 pages main + appendix
**Target venue:** arXiv (cs.CL primary, cs.AI cross-list); workshop submissions to follow

## §1 Introduction (~700 words)
- Motivation: long-horizon agents need memory; full-context fails (cost, lost-in-middle); RAG flattens narrative
- Thesis: extraction-first systems compress too aggressively; we propose context-first
- Contributions:
  1. AMP architecture — STM/LTM/Graph
  2. MCP-native protocol surface for memory
  3. Empirical results on LoCoMo beating Mem0 / RAG / Full-Context with dual-judge eval
  4. Open-source release with full reproducibility
- Architecture figure on page 1 (Figure 1)

## §2 Related Work (DONE)
- Long-context LMs (lost-in-middle)
- RAG (Lewis 2020) baseline
- Extraction-first: Mem0, MemoryBank
- OS-style: MemGPT/Letta
- Graph-based: Zep, A-MEM
- Side-network: LongMem
- AMP positioning

## §3 Method (Day 2)
- 3.1 Three-Layer Architecture
  - STM: per-session FIFO, verbatim turns
  - LTM: consolidated insights, fastembed bge-small-en-v1.5, SQLite BLOB
  - Graph: entity co-occurrence, edge weights
- 3.2 Consolidation Algorithm (Algorithm 1, pseudocode)
  - Trigger condition (STM size ≥ τ)
  - LLM-driven summarization into insights
  - Entity extraction
  - Embedding + storage
  - STM clear
- 3.3 Retrieval Scoring
  - score(m, q) = α·cos(v_q, v_m) + β·recency(m) + γ·entity_overlap(m, q)
  - Final context: top-k LTM + full STM + 1-hop graph neighborhood
- 3.4 Protocol Surface
  - MCP tools: write_memory, query_memory, consolidate, list_entities
  - Local-first deployment

## §4 Experiments (Day 5)
- 4.1 Dataset: LoCoMo (10 conversations, ~150 Q each, sessions up to 35)
- 4.2 Baselines: Mem0 (pinned version), naive RAG (chunk + cosine), Full-Context
- 4.3 Metrics:
  - Accuracy: dual local LLM judge (llama3.1:8b + qwen2.5:7b)
  - Cohen's κ for inter-judge agreement
  - 95% bootstrap CI (10k resamples)
- 4.4 Implementation: M3 Max, Ollama llama3.1:8b for generation, bge-small-en-v1.5 embeddings
- 4.5 Main Results: Table 1 + Figure 2
- 4.6 Ablation: STM-only / STM+LTM / Full (Table 2 + Figure 3)
- 4.7 Per-Category Analysis (Figure 4)
- 4.8 Latency-Accuracy Frontier (Figure 5)
- 4.9 Judge Agreement: κ value, raw agreement %

## §5 Discussion + Limitations
- Why context-first wins on multi-hop (preserves chains)
- When extraction-first suffices (single-hop simple recall)
- Local-first reproducibility argument
- Mem0 reproduction notes (API stability, pinned version)
- Limitations:
  - Single dataset (LoCoMo); generalization unproven
  - Both judges are open-weight; could share systematic biases
  - No MemGPT/Letta direct comparison (integration cost)
  - Sole-author preprint, no peer review

## §6 Conclusion (~200 words)
- Summary of contribution
- Future work: cloud sync, multi-agent shared memory, coding-agent benchmarks

## Appendix A Reproducibility (Day 6)
- Hardware spec, exact commands, model versions, expected outputs
- See REPRODUCE.md

## Appendix B Hyperparameters
- Embedding dim, consolidation threshold τ, retrieval weights α/β/γ, recency half-life, top-k, bootstrap N

## Appendix C Failure Analysis
- 20 sampled AMP failures, dominant failure modes

## Figures
1. Architecture diagram (page 1, hero)
2. Main results bar chart with CIs
3. Ablation bar chart
4. Per-category accuracy
5. Latency-accuracy frontier

## Tables
1. Main results (system × accuracy with CI)
2. Ablation (variant × accuracy with CI)
3. Latency (system × ingest p50, query p50, query p95)
4. Judge agreement (κ + raw agreement)
5. Related work comparison matrix (axes: storage, organization, deployment)

## Citation Strategy (Post-Submission)
- arXiv first, immediate distribution
- HuggingFace Daily Papers, X, HN, r/LocalLLaMA, r/MachineLearning
- Cold emails to Mem0 / MemGPT / LoCoMo authors
- Workshop tracks: NeurIPS FM4Agents, ICLR Tiny Papers, EMNLP demos
