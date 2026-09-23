# Track C — Vectorized residual / cluster classifier

## Scope and decision

This track is self-contained under `experiments/tracks/C/**`. It prototypes a
post-processing filter over the current data-only MWPM logical prediction. The
base prediction is constructed with the same deterministic data-only X-error
DEM as the current `solve.py` submission. No parent decoder, benchmark source,
test, report, or root submission was edited.

**Decision: reject both post-processing policies for approval.** The rule is
measurably harmful on the held-out 24-point sweep. The frozen score is nearly
inert on the held-out data and has a small negative aggregate change at 1M
shots/point. The feature extractor and API are retained as a fast research
instrument; the filter should not replace current final MWPM.

## Geometry and detector order

Stim is the authority for detector order. For the official one-round circuits,
detectors are ordered as two equal spatial layers:

* L=3: 8 detectors, 4 spatial positions per layer;
* L=5: 24 detectors, 12 spatial positions per layer;
* L=7: 48 detectors, 24 spatial positions per layer.

The first layer is the detector-coordinate order returned by Stim. The second
layer has the same `(x,y)` coordinates and is matched by coordinates rather
than by assuming an integer offset. In the benchmark's data-only injection,
the first layer carries the active detector graph and the second layer is
inactive; both layers are still represented and validated.

Within the first layer, the coordinate order is column-major in `x = 0, 2,
..., 2L`. For even column index `k`, the y coordinates are `4, 8, ...`; for
odd `k`, they are `2, 6, ...`. Track C does not hard-code those coordinates for
decoding: `build_geometry` reads Stim coordinates and precomputes the indices.
The data-only DEM supplies spatial adjacency edges, with 3, 15, and 35 edges
for L=3,5,7 respectively. `feature_diagnostics.json` records the full order,
indices, adjacency edges, boundary flags, row IDs, and column IDs.

## Feature extractor

`candidate_c.py:extract_features` accepts a flattened syndrome batch and a
base MWPM prediction. All shot operations are NumPy array operations. The only
Python loops in the connected-component implementation are over fixed graph
edges, fixed propagation rounds, or fixed roots; there is no Python loop over
shots in `decode`.

The full feature set includes:

* total syndrome weight, per-layer weights, temporal disagreement, and active
  spatial weight;
* spatial adjacency-pair count;
* exact connected-component count, largest and second component, and component
  density; trivial 0/1/2-active-node rows use direct vectorized formulas, and
  complex rows use fixed-depth label propagation;
* boundary count/fraction, minimum normalized boundary distance, near-boundary
  count, four-edge counts, and edge balance;
* occupied row/column counts, row/column maxima, and row/column spans;
* cluster class: 0 empty, 1 singleton, 2 separated singletons, 3 one
  connected cluster, or 4 mixed connected components;
* residual features conditioned on the current logical prediction: prediction
  bit, signed weight, signed adjacency, signed boundary fraction, signed
  component mass, and signed adjacency density.

“Residual” here means syndrome morphology signed/conditioned by the logical
MWPM decision. The decoder API exposes only a logical observable prediction,
not a physical correction chain, so no hidden truth-dependent physical
residual is constructed.

The candidate path uses `fast=True`: it retains the exact connected-component
calculation and model columns but omits diagnostic-only spans and edge-balance
columns. The full extractor remains available for feature audits.

## Candidate policies

### `rule`

The conservative analytic rule makes no change for `xi < 5`. For `xi >= 5`,
it flips only when all of the following hold:

```text
base logical prediction = 1
total syndrome weight >= 4
largest connected component >= 3
adjacency pairs >= 2
boundary fraction <= 0.5
```

This was intentionally high-specificity rather than tuned to rescue every
cluster. It still regressed on the held-out sweep, especially L=7 and
weights/classes 5/3 and 5/4.

### `classifier`

The score is a 13-column standardized ridge least-squares risk score over the
prediction, morphology, `xi`, `p`, and `L`. Coefficients, means, scales, and
threshold are frozen as constants in `candidate_c.py`. They were fit only on
seeds 8011 and 8017, 5,000 shots per point, using 240,000 training rows. The
threshold minimized training logical errors, breaking ties toward fewer flips.
The candidate has an explicit `xi < 5` no-change gate. There is no fitting,
file access, truth access, or network access during decode.

`classifier_fit.json` is the fitting receipt. Re-running the fit command is an
offline audit; it is not part of the parent-harness API.

## Exact commands

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/tracks/C/test_track_c.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/tracks/C/run_track_c.py --self-test
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/tracks/C/run_track_c.py --feature-diagnostics
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/tracks/C/run_track_c.py \
  --fit-classifier --shots 5000 --seeds 8011 8017
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/tracks/C/run_track_c.py \
  --shots 5000 --seeds 9029 \
  --variants baseline final rule classifier \
  --output results_controlled_5k.jsonl
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/tracks/C/run_track_c.py \
  --shots 1000000 --seeds 9029 \
  --variants baseline final rule classifier \
  --output results_runtime_1m.jsonl
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/tracks/C/run_track_c.py \
  --validate-deliverables
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  experiments/tracks/C/candidate_c.py \
  experiments/tracks/C/run_track_c.py \
  experiments/tracks/C/test_track_c.py
```

If a future candidate passes the held-out gate, the official five-seed command
is:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/tracks/C/run_track_c.py \
  --shots 100000 --seeds 42 137 256 1729 31415 \
  --variants baseline final rule classifier \
  --output results_official_100k.jsonl
```

That official command was deliberately not run here: the predeclared held-out
gate was not met, so official validation would not justify promotion.

## Files

* `candidate_c.py` — geometry, full/fast vectorized features, MWPM wrapper,
  rule and frozen-score candidates;
* `run_track_c.py` — fitting, paired shared-batch evaluation, runtime/RSS
  measurements, feature strata, diagnostics, and artifact validation;
* `test_track_c.py` — independent geometry, connected-component, contract, and
  no-per-shot-loop checks;
* `feature_diagnostics.json` — detector order and precomputed geometry receipt;
* `classifier_fit.json` — offline training receipt;
* `results_controlled_5k.jsonl` and `.summary.json` — held-out seed 9029,
  24-point, 5,000-shot paired sweep;
* `results_runtime_1m.jsonl` and `.summary.json` — 1,000,000-shot/point
  runtime and RSS sweep on the same 24-point grid;
* `results_controlled_3k.jsonl` and `.summary.json` — earlier pilot audit;
* `analysis.md` — numerical results, uncertainty, strata, and decision.
