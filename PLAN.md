# Plan: Correlated-noise decoder research and approval

Depth: tree 3   Mode: orchestrated

## Contract

- User approved the proposed L=3 QMC-MAP + L=5/7 data-only MWPM hybrid on 2026-09-23. Only this design may be deployed now.
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
- 2026-09-23 parent trade-off memo written; user approved the proposed hybrid.
- 2026-09-23 user requested two agents, authorizing one to implement the proposed hybrid and another to research new architecture. Stable prior work is committed as d1eef3d; no uncommitted work was present.
- 2026-09-23 hybrid agent completed full paired five-seed 1M validation; hybrid deployed and committed as c03dff2. New Track D architecture evaluated and committed as research-only 391cdf3.

## Approved implementation ownership

- Hybrid agent: `solve.py`, `tests/test_submission.py`, `tests/test_hybrid_validation.py`, `scripts/validate_hybrid.py`, and `experiments/hybrid/` only. Preserve data-only MWPM as an explicit ablation. Run the complete five-seed, 24-point, 1M-shot suite; retain paired error counts, per-point timing, source hashes, and memory provenance.
- Architecture agent: `experiments/tracks/D/` only. Read prior tracks; prototype and test a genuinely different method for L=5/7, with separate training and held-out seeds. Do not deploy it or alter frozen tracks.
- Parent: documentation, plots, gates, final review, and commits. Do not attribute the new architecture's experimental gains to the deployed hybrid. No subagent may delete coordination files or commit changes.
