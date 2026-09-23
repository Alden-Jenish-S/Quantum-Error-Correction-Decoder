# Benchmark contract

This document describes the executable implementation in this repository. It
supersedes the historical quickstart text retained in the upstream challenge,
which describes a different multi-round interface.

## Simulation

For each parameter point, the benchmark creates a rotated surface-code
`memory_z` circuit with one round and no built-in stochastic noise. It samples a
Boolean mask over the data qubits and injects X errors after the first circuit
tick. The mask has marginal error probability `p` per data qubit.

For `xi <= 0`, errors are independent Bernoulli samples. For `xi > 0`, the
sampler constructs a Gaussian-copula latent correlation matrix

```text
correlation(i, j) = exp(-distance(i, j) / xi)
```

where distance is Euclidean distance between Stim's two-dimensional data-qubit
coordinates. Thresholding the correlated Gaussian samples at the Bernoulli
quantile preserves the requested marginal probability while inducing spatial
correlation.

## Decoder interface

The evaluator calls `build_decoder(point)` once for every parameter point. The
returned object must implement:

```python
decode(syndrome_array: numpy.ndarray) -> numpy.ndarray
```

The input is a binary array shaped `(shots, num_detectors)`. The output must be a
binary array shaped `(shots,)`; each value predicts whether the first logical
observable flipped. The official detector counts are 8, 24, and 48 for
`L = 3, 5, 7`.

## Official parameter space

The challenge grid is the Cartesian product of:

```text
L   = 3, 5, 7
p   = 0.005, 0.01
xi  = 0, 2, 5, 10
```

There are 24 points. The evaluator uses one shared seeded random generator and
generates fresh samples on each run.

## Scoring and constraints

The score is:

```text
round(total logical errors * 1,000,000 / total simulations)
```

The 2.5-second limit covers `build_decoder` plus `decode` for each point. A
crash or timeout scores that point as all wrong. Syndrome generation is outside
that timer in the supplied evaluator. The challenge rules prohibit filesystem
and network access during decoding and cap `solve.py` at 200 KB.

## Final candidate

`solve.py` uses a frozen 256-entry offline MAP table for the eight exact official
L=3 `(p, xi)` combinations. The table was generated from the Track-A
Gaussian-copula model using approximate QMC rectangle probabilities; it is not
an exact orthant-probability certificate. For L=5/L=7, and for off-grid points,
the decoder constructs a matching graph from the same data-only iid X-error
process used by the benchmark when `xi=0`. It precomputes the graph during
`build_decoder` and performs one batch PyMatching decode. There is no runtime
file dependency or online training.
