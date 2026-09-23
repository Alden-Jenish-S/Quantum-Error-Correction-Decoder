# Quantum Error-Correction Decoder Investigation

## Executive conclusion

The final candidate replaces the supplied circuit-level MWPM calibration with a MWPM graph generated from the benchmark's actual independent component: data-qubit-only X errors at marginal rate `p`. This is a narrow, physically justified correction to the baseline model rather than brute-force search or seed memorization. It preserves the decoder interface, does not access files or the network at decode time, and stays under the 2.5-second per-point limit.

On the official 24-point grid with seed 42 and 1,000,000 shots per point, the supplied baseline produced **72,595 errors / 24,000,000 = 3,025 errors/M**. The final decoder produced **71,927 errors / 24,000,000 = 2,997 errors/M**, a **0.93% relative reduction**. In five-seed validation at 50,000 shots per point, the baseline scored **3,047 errors/M** and the final decoder **3,016 errors/M** (18,284 versus 18,098 errors), a **1.02% relative reduction**. The gain is modest and should not be described as a general correlated-noise decoder.

## Problem and executable benchmark

The executable implementation is authoritative; `docs/qec_quickstart_historical.md` contains stale descriptions of multi-round tensors, datasets, and scoring. The benchmark uses rotated surface-code `memory_z` circuits with one round and Pauli X data errors. For `L ∈ {3,5,7}`, the decoder receives a flattened `(shots, num_detectors)` binary syndrome array and returns one binary logical prediction per shot. Detector counts are 8, 24, and 48 respectively; data-qubit counts are 9, 25, and 49.

The 24 challenge points are `L ∈ {3,5,7}`, `p ∈ {0.005,0.01}`, and `xi ∈ {0,2,5,10}`. Errors are sampled by a Gaussian copula in the physical data-qubit coordinates, with latent correlation `exp(-distance/xi)` for positive `xi`; `xi=0` is independent Bernoulli noise. Stim converts each data-error mask to detector flips and the first observable flip. The score is pooled errors per million. Decoder construction plus batch decode must complete within 2.5 seconds per point; crashes and timeouts score all shots wrong.

## Baseline and observed behavior

The supplied `solve.py` constructs `MWPMDecoder(point, weighted=True)`. Its matching graph is made from a fresh Stim circuit with `after_clifford_depolarization=p`, and it ignores `xi`. This is not the same distribution as the evaluator: the evaluator injects a data-only correlated X mask after the first tick, while the baseline graph represents circuit-level depolarizing noise. This mismatch is a stronger and more actionable limitation than the generic statement that MWPM ignores correlations.

The 10,000-shot seed-42 baseline sweep gave 3,146 errors/M. Logical errors increased strongly with correlation length, especially for L=3 and L=5, while larger distance suppressed errors at low `p`. The retained artifacts support the error-rate trends; a separate syndrome-weight diagnostic was exploratory and is not used as quantitative evidence here.

## Hypotheses and prioritization

1. **Data-only iid graph (high priority):** theoretically exact for `xi=0`, low implementation cost, negligible decode cost, and directly compatible with the interface.
2. **Uniform or correlation-adjusted matching weights (medium):** cheap but risks regression at `xi=0`; test as an ablation.
3. **Syndrome feature residual correction (medium):** use syndrome weight, spatial extent, and boundary proximity around MWPM; potentially useful but requires careful calibration and risks overfitting.
4. **Finite-syndrome MAP lookup for L=3 (low):** theoretically attractive for a small code, but rare syndromes and training distribution make it fragile without a permitted offline training artifact.

## Controlled experiments and failed approaches

The `data_iid` candidate and supplied baseline were evaluated on identical seeded generated batches. At 10k shots, seed 42 scores were 3,146 errors/M for the baseline and 3,083 errors/M for data-only MWPM. A uniform-weight ablation also scored 3,083 errors/M in that single 24-point seed-42 run; no broader uniform-weight claim is made. An L=3 syndrome MAP table evaluated on independent 50k-shot batches produced 3,028 errors/M in a hybrid, compared with approximately 3,004/M for the same-seed final decoder, and was rejected. The lookup artifact does not encode all training provenance, so it is retained as a failed directional experiment rather than a definitive model comparison.

The data-only graph is therefore the only retained modification. It is intentionally conservative: it corrects an identified baseline-model mismatch but does not claim to solve the full Gaussian-copula inference problem.

## Final design

`solve.py` constructs the same noiseless surface-code prefix and suffix as the benchmark, inserts one `X_ERROR` instruction per data qubit with probability `p`, computes a decomposed detector error model, and builds a PyMatching graph once in `build_decoder`. `decode` performs one vectorized `decode_batch` call and returns observable column zero as `uint8`. There is no file access, network access, training, random sampling, or parameter search during decoding.

## Quantitative validation

| Evaluation | Supplied baseline | Final data-only MWPM | Relative change |
|---|---:|---:|---:|
| 24 points, seed 42, 10k/point | 3,146/M | 3,083/M | -2.00% |
| 24 points, five seeds, 50k/point | 3,047/M | 3,016/M | -1.02% |
| 24 points, seed 42, 1M/point | 3,025/M | 2,997/M | -0.93% |

The full-resolution result is the strongest single comparison. The multi-seed result supports generalization beyond seed 42, though the absolute gain is small relative to sampling noise in many individual points. Pointwise JSONL files retain every result, seed, parameter, error count, elapsed time, and timeout flag.

## Runtime and memory

The final standard benchmark completed in 11.0 seconds wall-clock for all 24 points and had zero timeouts on the recorded machine. The official timer excludes syndrome generation and measures construction plus decode, as specified by the repository; research JSONL rows record that timed phase separately. Precomputation is O(L²) graph construction per point and decoding is batch MWPM. Research runs record process max-RSS deltas where the platform exposes them; these are host- and phase-specific diagnostics rather than portable peak-memory bounds. The final decoder is small and memory is dominated by the benchmark's supplied full-batch syndrome arrays and any dtype-conversion copy.

The post-reorganization verifier (`scripts/verify_submission.py`) independently exercised the unmodified evaluator on all 24 points with 100,000 shots per point: 2,400,000 simulations, 0 timeouts, 0 crashes, all output-contract checks passed, and a maximum observed timed decoder phase of approximately 0.071 seconds on the recorded Apple arm64 host. The current submission is 3,409 bytes; its SHA-256 is recorded in `experiments/final_50k_validate.summary.json`. These are local measurements and should be rechecked on the target submission host.

## Ablations and limitations

The supplied graph, data-only graph, uniform graph, and L=3 lookup hybrid were compared. Uniform weights did not improve over the data-only graph in the controlled 10k experiment. The lookup experiment failed to establish a robust gain. Remaining limitations are important: the final graph still assumes independent data errors and therefore does not explicitly model `xi`; improvements are largest in some L=5/L=7 correlated points but not universal; and the 1% aggregate gain is not evidence of a large decoding breakthrough. The stale quickstart documentation and evaluator's permissive output validation were not modified, because benchmark infrastructure and rules were preserved.

## Artifacts

- Final allowed decoder: `solve.py`
- Experiment harness and candidates: `scripts/`
- Machine-readable results: `experiments/*.jsonl` and summaries
- Diagnostic figures: `plots/error_comparison.png`, `plots/improved_regimes.png`, `plots/runtime.png`, `plots/variability.png`
- Decision trace: `EXPERIMENT_LOG.md`
- Environment specification: `pyproject.toml` and `uv.lock`; local virtual environments are ignored by git

All plots are regenerated from saved JSONL files; no external datasets or proprietary services are required.
