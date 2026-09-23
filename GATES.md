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
- [ ] G6: Approved hybrid preserves interface, self-contained submission, no-I/O decode, size, and time constraints.
  CHECK: test -s solve.py && test $(wc -c < solve.py) -lt 200000
  EXPECT: exit 0
  EVIDENCE: pending implementation and verification
- [ ] G7: Full five-seed validation runs after approval and covers all 24 points.
  EVIDENCE: pending approved full-scale validation
- [ ] G8: Final plots, README, REPORT.md, and EXPERIMENT_LOG.md reflect the approved decoder.
  EVIDENCE: pending updated results
- [ ] G9: Stable approved changes are committed and repository state verified.
  EVIDENCE: pending verified milestone commits; no new push requested this turn
- [ ] G10: Second agent delivers independently evaluated new architecture without automatically deploying it.
  EVIDENCE: pending research
