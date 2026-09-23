# Gates: C vectorized residual / cluster classifier

Scope: only experiments/tracks/C/**; standalone candidates, paired measurements, reproducible analysis.

- [x] G1: Geometry/order and vectorized features verified for L=3,5,7 against independent reference checks.
  CHECK: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/tracks/C/test_track_c.py
  EXPECT: TRACK C TESTS PASS
  EVIDENCE: `TRACK C TESTS PASS`; independent coordinate/order, edge-count, component-oracle, contract, and decoder-loop checks pass for L=3,5,7.

- [x] G2: Both conservative rule and small fixed-score candidates satisfy API and avoid per-shot Python loops and decoding-time training/I/O.
  EVIDENCE: `test_track_c.py` AST check finds no for/while in `ResidualDecoder.decode`; candidate constants are embedded, with no decode-time filesystem, truth, or fitting path.

- [x] G3: Paired 24-point small-shot sweep compares baseline, final, rule and classifier on identical batches.
  EVIDENCE: `results_controlled_5k.jsonl` contains 96 rows (24 points × 4 decoders), one shared batch hash per point, seed 9029, 5,000 shots/point.

- [x] G4: Held-out accuracy, paired uncertainty and L/p/xi/weight/cluster diagnostics recorded; official five-seed validation if promising.
  EVIDENCE: `analysis.md` reports held-out paired rescue/harm counts, McNemar p-values, approximate CI, L/p/xi table, and weight/cluster strata. The classifier had 0/0 paired changes and the rule regressed, so the predeclared “promising” condition was not met and official validation was not run.

- [x] G5: Million-shot runtime build/decode/total and process RSS measured with output contract checks; 2.5-second requirement assessed with margin.
  EVIDENCE: `results_runtime_1m.summary.json`: 96/96 rows contract-valid, 0 timeouts; max rule 1.976s and max classifier 2.058s, both below 2.5s, with process RSS/high-water fields recorded.

- [x] G6: README, analysis, candidate, runner, result files and feature diagnostics complete with exact commands and keep/reject decision.
  CHECK: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/tracks/C/run_track_c.py --validate-deliverables
  EXPECT: TRACK C DELIVERABLES PASS
  EVIDENCE: `TRACK C DELIVERABLES PASS`; required source, diagnostics, fit receipt, controlled/runtime JSONL and summaries, README, and analysis are present.

## Analysis plan (before measurements)

Compare paired logical-error indicators on identical generated arrays, preserving official point order and shared seeded RNG. Primary endpoint is pooled error-rate change versus current final, with paired normal-approximation 95% CI using discordant pairs; report exact binomial discordance test where informative. Report seed-level dispersion and subgroup results descriptively, with no unadjusted subgroup significance claims. Training, if needed, uses seeds 8011 and 8017 only; controlled held-out sweep uses seed 9029, and official validation uses [42,137,256,1729,31415]. No tuning on validation. Timing excludes sampling, scoring and feature diagnostics; includes fresh build plus decode. Track candidate and solve source hashes.
