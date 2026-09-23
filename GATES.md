# Acceptance gates: correlated-noise decoder overhaul

- [x] G1: Track A produces covariance/MAP artifacts and held-out L=3 results.
  EVIDENCE: experiments/tracks/A/analysis.md and results_official_100k_5seeds.json.
- [x] G2: Track B evaluates graph/hyperedge candidates or documents incompatibility/failure.
  EVIDENCE: experiments/tracks/B/analysis.md and results_official_100k.summary.json.
- [x] G3: Track C evaluates vectorized residual candidates under the time budget.
  EVIDENCE: experiments/tracks/C/analysis.md and results_runtime_1m.summary.json.
- [x] G4: Parent trade-off table compares candidates and resource costs.
  EVIDENCE: experiments/tracks/cross_track_tradeoffs.md.
- [ ] G5: User approval is recorded before solve.py changes.
  EVIDENCE: pending
- [ ] G6: Approved hybrid preserves interface, self-contained submission, no-I/O decode, size, and time constraints.
  CHECK: test -s solve.py && test $(wc -c < solve.py) -lt 200000
  EXPECT: exit 0
  EVIDENCE: pending approval
- [ ] G7: Full five-seed validation runs after approval and covers all 24 points.
  EVIDENCE: pending approval
- [ ] G8: Final plots, README, REPORT.md, and EXPERIMENT_LOG.md reflect the approved decoder.
  EVIDENCE: pending approval
- [ ] G9: Approved changes are committed and pushed.
  EVIDENCE: pending approval
