# Track B — Graph and hyperedge reweighting

## Scope and conclusion

This track investigates correlation-aware decoding for the benchmark's exact
data-only X noise: each data qubit has marginal error probability `p`, and for
`xi > 0` the Bernoulli field is sampled by thresholding a Gaussian field with
latent correlation

```text
rho_ij = exp(-||x_i - x_j|| / xi).
```

No changes were made outside `experiments/tracks/B/**`. The default exported
candidate is `build_decoder(point, variant="iid")`; the parent harness can use
the other variants by passing their name explicitly.

**Approval conclusion:** approve `iid` only as a conservative track-B
candidate/reference. Do not approve any correlation-aware or burst candidate
from this track. The controlled candidates either exactly matched iid or were
significantly worse on official-seed validation. The native PyMatching graph
cannot represent the benchmark's general Gaussian-copula hyperedge structure.

## Files

* `candidate_b.py` — candidate implementations and the parent-harness API.
* `run_track_b.py` — runtime audit, self-tests, deterministic shared-batch
  experiments, diagnostics, and deliverable checks.
* `graph_diagnostics.json` — graph counts, boundary counts, fault-ID widths,
  weights, and candidate model details for L=5/L=7 and the requested grid.
* `results_seed42_5k.jsonl` and `.summary.json` — initial 5,000-shot sweep.
* `results_seed42_10k.jsonl` and `.summary.json` — 10,000-shot seed-42 sweep.
* `results_official_100k.jsonl` and `.summary.json` — five official seeds,
  100,000 shots per point, for iid and the only shortlisted negative-control
  variant (`pair_shortcut`).
* `analysis.md` — formulas, PyMatching audit, statistics, and conclusions.

## Candidate inventory

| Variant | Mechanism | Exact at `xi=0`? | Status |
|---|---|---:|---|
| `iid` | data-only X DEM with the exact marginal `p`; weighted MWPM | yes | conservative reference; approved |
| `noop` | alias of `iid` | yes | no-op control |
| `parity` | replace parallel graphlike-column weights using an exact bivariate copula tail | yes | technically valid; no observed change |
| `parity_reg` | same, with half-strength log-weight update | yes | technically valid regularized control; no observed change |
| `pair_uncorr` | marginal-preserving local pair-source DEM, decoded as ordinary MWPM | yes | technically valid surrogate; no improvement |
| `pair_corr` | same pair-source surrogate with PyMatching two-pass correlations | yes | valid only for the decomposed surrogate; worse |
| `pair_corr90` | same, with 90% of the source budget allocated to pairs | yes | valid surrogate sensitivity; worse |
| `cluster_corr` | four-site connected-burst/tree surrogate, not an exact 4-variate Gaussian probability | yes | explicitly approximate; worse |
| `pair_shortcut` | pair-source approximation collapsed to a graphlike parity edge when its total parity has <=2 detectors | yes | technically valid shortcut; officially worse |
| `pair_shortcut_corr` | shortcut plus two-pass flag | yes | technically valid sensitivity; no benefit over shortcut |

The word “hyperedge” is not used to claim native support: `cluster_corr` is a
surrogate, and `pair_shortcut*` only collapse a source when its *resulting
parity* is graphlike. No undecomposed multi-detector source is passed to
PyMatching as if it were supported.

## Exact commands

Run from the repository root. `uv run` is required in this checkout because
the shell `python` command is not installed.

```bash
uv run python experiments/tracks/B/run_track_b.py --audit-json
uv run python experiments/tracks/B/run_track_b.py --self-test
uv run python experiments/tracks/B/run_track_b.py --diagnostics
uv run python experiments/tracks/B/run_track_b.py \
  --shots 5000 --seeds 42 --output results_seed42_5k.jsonl
uv run python experiments/tracks/B/run_track_b.py \
  --shots 10000 --seeds 42 --output results_seed42_10k.jsonl
uv run python experiments/tracks/B/run_track_b.py \
  --shots 100000 --seeds 42 137 256 1729 31415 \
  --variants iid pair_shortcut --output results_official_100k.jsonl
uv run python experiments/tracks/B/run_track_b.py --quick-results
uv run python experiments/tracks/B/run_track_b.py --official-results
uv run python experiments/tracks/B/run_track_b.py --validate-deliverables
uv run python -m compileall -q experiments/tracks/B
```

The runner samples each full official grid in the same sequential RNG order,
but decodes only L=5 and L=7. For every shared batch, `iid` is decoded first;
each other prediction is compared shot-by-shot with the same truth and the
same iid prediction. Build plus decode wall time is measured inside the 2.5 s
per-point budget. Sampling, import, diagnostics, and scoring are outside that
timer. No candidate performs I/O during `decode`.
