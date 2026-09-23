# Plan: Correlated-noise decoder research and approval

Depth: tree 3   Mode: orchestrated

## Contract

- `solve.py` remains unchanged during independent research and until explicit user approval.
- Track A owns `experiments/tracks/A/`; Track B owns `experiments/tracks/B/`; Track C owns `experiments/tracks/C/`.
- Benchmark infrastructure and scoring are read-only.
- Candidate comparisons record paired batches, seeds, parameter points, accuracy, timing, memory, and failure modes.
- The parent presents a trade-off table and waits for approval before merging a hybrid into `solve.py`.

## Tree

- Track A: copula/MAP L=3 ........ experiments/tracks/A/
- Track B: graph/hyperedge L=5/7  experiments/tracks/B/
- Track C: cluster residual filter experiments/tracks/C/
- Parent integration: cross_track_tradeoffs.md
- Approval checkpoint: user review required before solve.py changes

## Status log

- 2026-09-23 protected `solve.py`; launched three independent tracks in parallel.
- 2026-09-23 tracks A, B, and C completed; no candidate merged.
- 2026-09-23 parent trade-off memo written; awaiting explicit user approval.
