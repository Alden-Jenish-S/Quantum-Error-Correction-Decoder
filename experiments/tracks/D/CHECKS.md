# Executed checks

All commands ran from the repository root with:

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

## Tests

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider experiments/tracks/D/test_track_d.py
```

Result: **34 passed**. Run on the initial 250k tables, on the final 1M tables
before freezing, and once after freeze plus the training replay. Existing tests
were neither modified nor used for Track D.

## Reproducibility

```bash
.venv/bin/python experiments/tracks/D/reproduce_training.py
```

Result: **PASS**; 12,000,000 training samples replayed, all 48 count/key arrays
and all embedded table strings exactly equal. Frozen sources/constants were
not overwritten. Runtime 14.91s, whole-process peak RSS 173,654,016 bytes.
Full machine-readable receipt: `reproducibility.json`.

## Artifact audit

```bash
.venv/bin/python experiments/tracks/D/audit_track_d.py
```

Result: **PASS** for both 24-row exploratory pilots, 72-row held-out test, and
2-row stress evaluation. Checks frozen SHA-256 provenance, seeds, sizes,
paired-count identities, summary regeneration, support and serialization.

Frozen design SHA-256:
`33fb9f3072ef522fea354c0e00ee235a6279aca71d8be9e0755afcdb8b74f56c`.

The parent may review `README.md` for complete historical training/evaluation
commands. No git commit, push, merge or candidate deployment was performed.
