# Reproducibility guide

## Environment

The project targets Python 3.10+. The committed `pyproject.toml` and `uv.lock`
the repository root; do not copy a virtual environment between machines.
the repository root; do not copy a virtual environment between machines.

```bash
uv sync --dev
uv run pytest
```

Without `uv`:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

## Standard runs

```bash
uv run python run.py --grid tiny --shots 1000
uv run python run.py --shots 100000
uv run python run.py --validate --shots 50000
```

The first command is a quick smoke test. The second uses the full 24-point grid
with one seed. The third uses the five validation seeds defined by the challenge.

## Research runs

`scripts/run_experiment.py` writes one JSON object per decoder/seed/parameter
point to JSONL and a sibling summary file. Each row records decoder name, seed,
`L`, `p`, `xi`, shots, errors, error rate, timed decoder duration, timeout state,
Python version, and a platform-specific max-RSS diagnostic field.

```bash
uv run python scripts/run_experiment.py \
  --decoder baseline --shots 50000 --validate \
  --output experiments/baseline_50k_validate.jsonl
uv run python scripts/run_experiment.py \
  --decoder final --shots 50000 --validate \
  --output experiments/final_50k_validate.jsonl

uv run python scripts/validate_hybrid.py \
  --label official_1m_5seeds_serial --compatibility

uv run python scripts/make_hybrid_plots.py \
  --input experiments/hybrid/official_1m_5seeds_serial.jsonl \
  --outdir plots
```

`baseline` is the supplied circuit-level MWPM implementation. `final` loads the
root `solve.py` submission through `build_decoder`. The research harness does
not change the official evaluator or score calculation.

## Result interpretation

Use pooled counts as the primary aggregate:

```text
round(sum(errors) * 1e6 / sum(shots))
```

The standard `--validate` CLI additionally reports the mean of five separately
rounded seed scores; those two summaries can differ by one error/M. Runtime
rows measure construction plus decode only, matching the evaluator; end-to-end
CLI wall time also includes syndrome generation. RSS values are diagnostic
high-water-mark deltas and are not a portable peak-memory guarantee.

The hybrid receipt compares the supplied circuit MWPM, the retained pre-hybrid
data-only MWPM, and the deployed hybrid on identical syndrome batches. It
stores paired rescue/harm counts, per-point regressions, source hashes, and the
separate unchanged `run.py --validate` compatibility report.

## Artifact provenance

Results under `experiments/` are reviewed, named outputs. Temporary logs, raw
cache files, virtual environments, and interpreter caches are ignored by the
root `.gitignore`. Plots are regenerated from JSONL using
`scripts/make_plots.py`.
