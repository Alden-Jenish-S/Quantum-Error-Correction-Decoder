# Acceptance gates: correlated-noise decoder overhaul

- [x] G1: Track A produces covariance/MAP artifacts and held-out L=3 results.
  EVIDENCE: experiments/tracks/A/analysis.md and results_official_100k_5seeds.json.
- [x] G2: Track B evaluates graph/hyperedge candidates or documents incompatibility/failure.
  EVIDENCE: experiments/tracks/B/analysis.md and results_official_100k.summary.json.
- [x] G3: Track C evaluates vectorized residual candidates under the time budget.
  EVIDENCE: experiments/tracks/C/analysis.md and results_runtime_1m.summary.json.
- [x] G4: Parent trade-off table compares candidates and resource costs.
  EVIDENCE: experiments/tracks/cross_track_tradeoffs.md.
- [x] G5: User approval is recorded before solve.py changes.
  EVIDENCE: User: "Spawn two subagents, proceed one with the proposed hybrid, and the other to further experiment and find new architecture." Approval applies to Track A L=3 plus unchanged L=5/7 MWPM, not future research variants.
- [x] G6: Approved hybrid preserves interface, self-contained submission, no-I/O decode, size, and time constraints.
  CHECK: test -s solve.py && test $(wc -c < solve.py) -lt 200000
  EXPECT: exit 0
  EVIDENCE: `solve.py` is 5,506 bytes; `uv run pytest` 200 passed; isolated submission/I/O/timing tests pass.
- [x] G7: Full five-seed validation runs after approval and covers all 24 points.
  EVIDENCE: `experiments/hybrid/official_1m_5seeds_serial.summary.json`: 120 rows, 120M shots/decoder, 0 timeouts, 358,992 hybrid errors, max 0.3113 s.
- [x] G8: Final plots, README, REPORT.md, and EXPERIMENT_LOG.md reflect the approved decoder.
  EVIDENCE: `README.md`, `REPORT.md`, `EXPERIMENT_LOG.md`, `plots/hybrid_*.png`, and `scripts/make_hybrid_plots.py`.
- [x] G9: Stable approved changes are committed and repository state verified.
  EVIDENCE: commits `c03dff2`, `391cdf3`, `1e9478a`, and `82ac0ff`; `git status` clean and `origin/main` at `82ac0ff`.
- [x] G10: Second agent delivers independently evaluated new architecture without automatically deploying it.
  EVIDENCE: `experiments/tracks/D/analysis.md`: 34 tests/audit pass; candidate remains research-only.
