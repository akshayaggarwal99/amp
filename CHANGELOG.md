# Changelog

## 0.3.0 — 2026-05-13

### Added
- **`amp daemon` subcommand**: persistent multi-client HTTP MCP server. Replaces the per-process `stdio` model so multiple MCP clients (Claude Desktop, Cursor, IDE plugins) can share one warm AMP brain. Commands: `amp daemon start [--port N] [--host H]`, `amp daemon stop`, `amp daemon status`, `amp daemon restart`.
- `src/amp/daemon.py` module — daemon lifecycle (start/stop/status), port selection, PID tracking.
- **Vector cache** in `Storage`: embedding matrix held in RAM (~15 MB per 10k memories at 384-dim float32), lazy-loaded on first search and rebuilt incrementally on add/forget. Replaces the previous O(n) per-search read-every-row-from-SQLite loop. Pays off most in daemon mode where the cache survives between queries.
- **Embedding LRU cache** (4096 entries, keyed by content SHA-256). Skips re-embedding on repeated content (fixtures, batch imports).
- Helpers: `_content_hash`, `_to_epoch`, `_safe_parse_metadata`, `_stringify_metadata` — module-level utilities for dedup, timestamp parsing, and metadata round-tripping.

### Changed
- CLI gains a `daemon` Typer sub-app alongside existing `serve`, `dashboard`, `setup`, `reset`.
- **STM metadata storage now JSON, not `str(dict)`**. Previous releases wrote Python `repr` strings which did not survive `json.loads` on read. New rows are JSON; readers use `_safe_parse_metadata` which still handles legacy `str(dict)` rows transparently.

### Fixed
- Metadata round-trip bug: STM rows written by 0.1.0/0.2.0 used `str(item.metadata)` which produced single-quoted Python repr strings instead of valid JSON. Reads now tolerate both shapes; new writes are correct.

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
