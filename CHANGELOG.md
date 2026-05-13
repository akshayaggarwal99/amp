# Changelog

## 0.2.0 — 2026-05-12

### Added
- **Auto-consolidation**: `Storage.add_to_stm()` now triggers `consolidate()` when active STM size reaches `stm_threshold` (default 50). Disable with `auto_consolidate=False`.
- **Recency-weighted retrieval**: `Storage.search()` scores candidates as `alpha * cos_sim + beta * exp(-dt / tau_recency)`. Defaults: `beta=0.1`, `tau_recency=7` days. Pass `use_recency=False` to disable.
- **Neighbor padding**: `Storage.search()` and `Amp.recall()` accept `neighbor_window=N`; each top-k anchor is returned with the N items immediately preceding and following it in creation order (deduplicated). Useful for preserving local conversational structure that pure turn-level retrieval loses.
- `Storage` constructor now accepts `stm_threshold`, `recency_weight`, `recency_half_life_days` for explicit configuration.

### Changed
- `Storage.search()` returns extra fields per result: `cos_sim` and `recency` (the raw components of the scoring function), in addition to the combined `score`.

### Compatibility
- All new parameters default to behavior compatible with 0.1.0 except auto-consolidation (which defaults on). Pass `auto_consolidate=False` to `add_to_stm()` if you rely on STM growing past 50 items without consolidation.

### Documentation
- New paper "AMP: A Cognitive-Architecture Memory Protocol for Long-Horizon LLM Agents" added under `paper/`. Empirical evaluation on a LoCoMo subset against Mem0, naive RAG, and full-context baselines. Includes a dual-judge protocol with a strict-IDK rule that surfaces an under-discussed LLM-judge leniency bias on refusal predictions.

## 0.1.0 — 2025-12-13
- Initial release.
