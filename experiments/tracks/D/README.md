# Track D: direct syndrome posterior research candidate

This is a research-only candidate. It is not approved for deployment and does
not modify `solve.py`. All Track-D code, constants, tests, receipts, and reports
are in this directory.

## Mechanism

Track D estimates `P(logical | complete active syndrome, L, p, xi)` directly
from the benchmark Gaussian-copula sampler and Stim's exact logical label.
It does not reweight a matching graph, fit syndrome morphology features, or
enumerate the 49-bit physical mask space.

* **L5:** a 4096-entry logical decision table.
* **L7:** the IID MWPM decoder is the backoff; only supported syndrome IDs that
  are confidently changed are stored as sorted `uint32` IDs.
* **L3:** variants use the unchanged Track-A table/reference path; D makes no
  new L3 claim.
* **Shrinkage:** cells require at least 20 training observations and the 1%
  lower quantile of `Beta(wrong+1, right+9)` must exceed 0.5. The raw variant
  uses an unregularized majority table. `supported_nosym` is the symmetry
  ablation.

The exhaustive single-error audit found that only identity and 180-degree
rotation are valid square-coordinate transformations for this detector order;
the other six of eight geometric isometries were rejected. A valid action must
preserve the active syndrome columns and have a verified GF(2) logical action.
Symmetry pooling uses one canonical observation per independently sampled mask,
not augmented pseudo-replicates.

## Reproduce

Run from the repository root with the locked environment:

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest -q -p no:cacheprovider experiments/tracks/D/test_track_d.py
.venv/bin/python experiments/tracks/D/audit_track_d.py
.venv/bin/python experiments/tracks/D/reproduce_training.py
.venv/bin/python experiments/tracks/D/run_track_d.py evaluate --mode test --label test_replay
```

The last command reruns the frozen test design with a new receipt filename;
error counts should match, while timings can differ. The training reproduction
command regenerates counts in memory and verifies them without replacing the
retained model. Full training/design history (for a fresh output directory with
no `freeze.json` or prior result files) was:

```bash
.venv/bin/python experiments/tracks/D/run_track_d.py train --shots 250000
.venv/bin/python experiments/tracks/D/run_track_d.py evaluate \
  --mode pilot --label pilot_250k
.venv/bin/python experiments/tracks/D/archive_stage.py
.venv/bin/python experiments/tracks/D/run_track_d.py train --shots 1000000
.venv/bin/python experiments/tracks/D/run_track_d.py evaluate --mode pilot --label pilot_1mtrain
.venv/bin/python -m pytest -q -p no:cacheprovider experiments/tracks/D/test_track_d.py
.venv/bin/python experiments/tracks/D/run_track_d.py freeze
.venv/bin/python experiments/tracks/D/run_track_d.py evaluate --mode test
.venv/bin/python experiments/tracks/D/run_track_d.py evaluate --mode stress
.venv/bin/python experiments/tracks/D/audit_track_d.py
```

The first 250k stage is preserved under `stage_250k/`. Do not rerun `train`
after `freeze.json` exists: the runner intentionally refuses retraining after
the design freeze. The test runner advances one `default_rng(seed)` through all
24 points and makes one official sampler call at every point, including L3
when a result is not being retained. Training, pilot, official-test and stress
seeds are disjoint.

## Files and provenance

* `candidate_d.py`: importable `build_decoder(point, variant=...)` API.
* `reference_iid.py`: explicit immutable IID data-only MWPM reference.
* `geometry.py`, `geometry_audit.json`: active-check packing and symmetry audit.
* `run_track_d.py`: training, freeze, paired evaluation, and runtime stress.
* `frozen_tables.py`: generated embedded constants; decode performs no file or
  network I/O, training, randomness, or truth access.
* `training.json`, `training_counts.npz`: frozen training provenance/counts.
* `freeze.json`: pre-test design freeze and source hashes.
* `results_*.jsonl` and `results_*.summary.json`: pilot, held-out, and stress
  records with paired counts, timing, memory, support, and hashes.
* `numeric_tables.md`: pointwise held-out results and posterior support cells.
* `audit.json`, `artifact_manifest.json`: receipt and artifact checks.
* `reproduce_training.py`, `reproducibility.json`: non-overwriting seeded
  training replay and exact count/table comparison.
* `PROTOCOL.md`, `SCALING.md`, `analysis.md`: design and interpretation.

Memory receipts report whole-process RSS/high-water memory, including sampled
syndrome arrays and allocator retention. Decoder table/payload bytes are also
reported separately; they are not presented as process memory.
