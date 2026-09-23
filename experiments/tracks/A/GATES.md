# Gates: track A, L=3 copula Bayes decoding

Scope: Only `experiments/tracks/A/**`; numerical model, candidate, paired evaluation, and scientific report.

- [x] G1: Derive latent and binary covariance, validate actual Stim geometry and jitter.
  EVIDENCE: README.md lines 10-41 and covariance_validation.json record the 9 coordinates, C_ij, 1e-12 jitter, tail-correct binary covariance, quadrature, and seeded empirical checks.
- [x] G2: Enumerate all 512 masks through SurfaceCodeExperiment, preserve 8-detector order and observable 0; construct all eight official L=3 tables with numerical convergence diagnostics.
  EVIDENCE: test_track_a.py passed mask/linearity checks; tables_qmc.json has 8 points, 512 mask probabilities each, 256-entry tables, and no coarse/refined table changes.
- [x] G3: Standalone build_decoder interface with L=5/7 data-only fallback and no online file reads, plus targeted tests.
  CHECK: PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider experiments/tracks/A/test_track_a.py
  EXPECT: 9 passed
  EVIDENCE: 9 passed in 11.45s; full repository suite 142 passed in 19.86s.
- [x] G4: Full-grid paired pilot and five-seed held-out evaluation if promising, official shared RNG sequence, errors, timing, memory and uncertainty recorded.
  EVIDENCE: results_official_100k_5seeds.json/.jsonl record 5 seeds x 24 points x 100,000 shots, shared paired batches, per-point errors/timing/RSS/Wilson intervals, and aggregate L3/full-grid results.
- [x] G5: README and analysis contain exact commands, provenance, numerical tradeoffs, limitations and L=3/full-grid conclusions.
  EVIDENCE: README.md and analysis.md document commands, covariance tail convention, QMC approximation/convergence, no leakage, results, timing, fallback scope, and conclusion.

- [x] G6: Ownership incident is disclosed and no fabricated restoration is presented as complete.
  EVIDENCE: INCIDENT.md records the accidental deletion of pre-existing untracked root coordination files and recovery checks; all research deliverables are under A.

## Prospective analysis plan

Freeze model-based tables before evaluation. Construct probabilities independently of benchmark RNG and truth, with independent randomized integration seeds. Primary comparison: paired L=3 candidate versus current final on the same shots; full-grid hybrid and supplied weighted MWPM comparisons secondary. Report error/accuracy and pointwise Wilson 95% intervals; paired differences from discordant outcomes with normal-approximation intervals and exact McNemar tests (Holm correction for eight per-point comparisons). Report aggregate uncertainty stratified by seed/point. Pilot: 10,000 shots/point with nonofficial seed 8675309. If candidate differs from final and model expected risk is lower, evaluate 100,000 shots/point at all five official seeds; this is held out from integration and pilot. If useful precision is not obtained, report uncertainty rather than tune on held-out samples. Official 2.5-second build+decode timer excludes offline integration and sampling. All full-grid RNG draws must occur in official point order, once per point without chunking.
