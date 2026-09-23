# Quantum Error-Correction Decoder Investigation

## Executive conclusion

The deployed candidate combines a frozen, offline Gaussian-copula MAP table for `L=3` with the data-only iid MWPM decoder for `L=5` and `L=7`. This is a narrow, physically justified extension of the previous calibration correction rather than brute-force search or seed memorization. It preserves the decoder interface, does not access files or the network at decode time, and stays under the 2.5-second per-point limit.

On the official 24-point grid with five seeds and 1,000,000 shots per point, the supplied baseline produced **364,283 errors / 120,000,000 = 3,036 errors/M**. The pre-hybrid data-only decoder produced **360,682 errors = 3,006/M**. The deployed hybrid produced **358,992 errors = 2,992/M**, saving 1,690 errors against the pre-hybrid decoder and 5,291 against the supplied baseline. These are pooled paired results; the hybrid rescued 3,971 and harmed 2,281 shots versus pre-hybrid. The gain is modest and should not be described as a complete correlated-noise decoder.

## Problem and executable benchmark

The executable implementation is authoritative; `docs/qec_quickstart_historical.md` contains stale descriptions of multi-round tensors, datasets, and scoring. The benchmark uses rotated surface-code `memory_z` circuits with one round and Pauli X data errors. For `L ∈ {3,5,7}`, the decoder receives a flattened `(shots, num_detectors)` binary syndrome array and returns one binary logical prediction per shot. Detector counts are 8, 24, and 48 respectively; data-qubit counts are 9, 25, and 49.

The 24 challenge points are `L ∈ {3,5,7}`, `p ∈ {0.005,0.01}`, and `xi ∈ {0,2,5,10}`. Errors are sampled by a Gaussian copula in the physical data-qubit coordinates, with latent correlation `exp(-distance/xi)` for positive `xi`; `xi=0` is independent Bernoulli noise. Stim converts each data-error mask to detector flips and the first observable flip. The score is pooled errors per million. Decoder construction plus batch decode must complete within 2.5 seconds per point; crashes and timeouts score all shots wrong.

## Baseline and observed behavior

The supplied `solve.py` constructs `MWPMDecoder(point, weighted=True)`. Its matching graph is made from a fresh Stim circuit with `after_clifford_depolarization=p`, and it ignores `xi`. This is not the same distribution as the evaluator: the evaluator injects a data-only correlated X mask after the first tick, while the baseline graph represents circuit-level depolarizing noise. This mismatch is a stronger and more actionable limitation than the generic statement that MWPM ignores correlations.

The 10,000-shot seed-42 baseline sweep gave 3,146 errors/M. Logical errors increased strongly with correlation length, especially for L=3 and L=5, while larger distance suppressed errors at low `p`. The retained artifacts support the error-rate trends; a separate syndrome-weight diagnostic was exploratory and is not used as quantitative evidence here. The deployed table changes only four L=3 parameter points (`xi=5,10` at both p values) in the five-seed run; L=5/L=7 predictions remain identical to pre-hybrid MWPM.

## Hypotheses and prioritization

1. **Data-only iid graph (high priority):** theoretically exact for `xi=0`, low implementation cost, negligible decode cost, and directly compatible with the interface.
2. **Uniform or correlation-adjusted matching weights (medium):** cheap but risks regression at `xi=0`; test as an ablation.
3. **Syndrome feature residual correction (medium):** use syndrome weight, spatial extent, and boundary proximity around MWPM; potentially useful but requires careful calibration and risks overfitting.
4. **Finite-syndrome MAP lookup for L=3 (low):** theoretically attractive for a small code, but rare syndromes and training distribution make it fragile without a permitted offline training artifact.

## Controlled experiments and failed approaches

The `data_iid` candidate and supplied baseline were evaluated on identical seeded generated batches. At 10k shots, seed 42 scores were 3,146 errors/M for the baseline and 3,083 errors/M for data-only MWPM. A uniform-weight ablation also scored 3,083 errors/M in that single 24-point seed-42 run; no broader uniform-weight claim is made. An L=3 syndrome MAP table evaluated on independent 50k-shot batches produced 3,028 errors/M in a hybrid, compared with approximately 3,004/M for the same-seed final decoder, and was rejected. The lookup artifact does not encode all training provenance, so it is retained as a failed directional experiment rather than a definitive model comparison.

The data-only graph is therefore the only retained modification. It is intentionally conservative: it corrects an identified baseline-model mismatch but does not claim to solve the full Gaussian-copula inference problem.

## Final design

`solve.py` constructs the same noiseless surface-code prefix and suffix as the benchmark. For L=5/L=7 it inserts one `X_ERROR` instruction per data qubit with probability `p`, computes a decomposed detector error model, and builds a PyMatching graph once in `build_decoder`. For exact official L=3 `(p, xi)` pairs it uses a 256-entry table generated offline from the reviewed Track-A QMC-MAP artifact; detector IDs are packed little-endian and inactive detector inputs are rejected. Off-grid L=3 points conservatively fall back to data-only MWPM without rounding or interpolation. `decode` performs either a vectorized table lookup or one vectorized `decode_batch` call and returns `uint8`. There is no file access, network access, training, random sampling, or parameter search during decoding.

## Quantitative validation

| Evaluation | Supplied baseline | Pre-hybrid data-only MWPM | Deployed hybrid | Relative change vs pre-hybrid |
|---|---:|---:|---:|---:|
| 24 points, seed 42, 10k/point | 3,146/M | 3,083/M | not run in this hybrid receipt | — |
| 24 points, five seeds, 100k/point Track-A holdout | 3,021/M | 2,990/M | 2,978/M | -0.40% |
| 24 points, five seeds, 1M/point paired validation | 3,036/M | 3,006/M | **2,992/M** | **-0.47%** |

The full-resolution paired result is the primary deployment comparison. The hybrid had 20 improved point-seed rows, 100 tied rows, and no regressions against pre-hybrid MWPM; against the supplied circuit MWPM, 92 rows improved, 1 regressed, and 27 tied. The absolute gain is small relative to sampling noise in many individual points. Pointwise JSONL files retain every result, seed, parameter, error count, elapsed time, paired rescue/harm count, and timeout flag.

## Runtime and memory

The deployed standard validation reported a mean of 2,992 errors/M and zero timeouts across 120 point-seed rows. At 1M shots, maximum build-plus-decode times were 0.027 s (L=3), 0.202 s (L=5), and 0.311 s (L=7) for the hybrid. The paired suite took 131.4 s wall-clock; the separate unchanged standard compatibility subprocess took 87.8 s. The benchmark timer excludes syndrome generation and measures construction plus decode, as specified by the repository; research JSONL rows record that timed phase separately. Whole-process high-water RSS reached approximately 1.69 GB during the sequential paired run and includes sampling and allocator retention, not isolated decoder memory. The 256-byte runtime L=3 table is small; memory is dominated by full-batch syndrome arrays and decoder conversions.

The post-reorganization verifier (`scripts/verify_submission.py`) independently exercised the unmodified evaluator on all 24 points with 100,000 shots per point: 2,400,000 simulations, 0 timeouts, 0 crashes, all output-contract checks passed, and a maximum observed timed decoder phase of approximately 0.071 seconds on the recorded Apple arm64 host. The current submission is 3,409 bytes; its SHA-256 is recorded in `experiments/final_50k_validate.summary.json`. These are local measurements and should be rechecked on the target submission host.

## Ablations and limitations

The supplied graph, data-only graph, uniform graph, Track-A L=3 table, and Track-C residual filters were compared. Track A's L=3 table produced the only approved paired gain. Track B graph/hyperedge surrogates and Track C filters regressed or were inert. Track D is retained as a research-only direct syndrome-posterior architecture: it saved 72 errors at L=5 and 8 at L=7 on its frozen held-out design, but L=7 uncertainty and rare-syndrome support were insufficient for deployment. Remaining limitations are important: L=5/L=7 deployed decoding still assumes independent data errors and therefore does not explicitly model `xi`; the table uses approximate QMC priors; and the 0.47% aggregate reduction versus pre-hybrid is not evidence of optimality or a large decoding breakthrough. The stale quickstart documentation and evaluator infrastructure were not modified.

## Artifacts

- Final allowed decoder: `solve.py`
- Experiment harness and candidates: `scripts/`
- Machine-readable results: `experiments/*.jsonl` and summaries
- Hybrid validation receipt: `experiments/hybrid/official_1m_5seeds_serial.summary.json`
- Diagnostic figures: `plots/hybrid_errors_by_point.png`, `plots/hybrid_improvement_heatmap.png`, `plots/hybrid_runtime_by_distance.png`, `plots/hybrid_seed_variability.png`
- Research-only new architecture: `experiments/tracks/D/`
- Decision trace: `EXPERIMENT_LOG.md`
- Environment specification: `pyproject.toml` and `uv.lock`; local virtual environments are ignored by git

All plots are regenerated from saved JSONL files; no external datasets or proprietary services are required.
