# Gates: Track B graph and hyperedge reweighting

Scope: Audit PyMatching and deliver reproducible, data-only correlation-aware candidates and results for L=5/L=7.

- [x] G1: Track-B implementation contains only track-local candidate and runner code, with import/compile checks passing.
  CHECK: uv run python -m compileall -q experiments/tracks/B && uv run python experiments/tracks/B/run_track_b.py --self-test
  EXPECT: /SELF_TEST_OK/
  EVIDENCE: SELF_TEST_OK: exact xi=0 equivalence, graph/fault IDs, single errors, contract, marginals, runtime audit

- [x] G2: PyMatching 2.3.1 capabilities are audited from the installed runtime and documented, including fault IDs, observables, boundary edges, and hyperedge limitations.
  CHECK: uv run python experiments/tracks/B/run_track_b.py --audit-json
  EXPECT: /"pymatching_version": "2\.3\.1"/
  EVIDENCE: "correlation_support": "two-pass heuristic for graphlike DEM decompositions; enable on load AND decode" | }

- [x] G3: Controlled seed-42 comparisons cover L=5/L=7, xi=0/2/5/10, and p=.005/.01 for the requested candidate families, with timing/RSS/graph diagnostics recorded.
  CHECK: uv run python experiments/tracks/B/run_track_b.py --quick-results
  EXPECT: /"shots": 5000/
  EVIDENCE: {"rows": 160, "shots": 5000, "xi0_exact": true}

- [x] G4: Full track deliverables exist and contain formulas, exact commands, numerical conclusions, and a regression analysis that explicitly checks xi=0.
  CHECK: uv run python experiments/tracks/B/run_track_b.py --validate-deliverables
  EXPECT: /DELIVERABLES_OK/
  EVIDENCE: DELIVERABLES_OK

- [x] G5: Promising variants are validated on official seeds, or the limitation is explicitly recorded with a candid reason.
  CHECK: uv run python experiments/tracks/B/run_track_b.py --official-results
  EXPECT: /OFFICIAL_RESULTS_OK/
  EVIDENCE: OFFICIAL_RESULTS_OK: 160 rows
