# Track A analysis and conclusion

## Frozen design and provenance

The candidate was frozen after an offline build. The pilot (`results_pilot_10k.*`)
was run with 10,000 shots per official point and seed `8675309`. The held-out
run (`results_official_100k_5seeds.*`) used 100,000 shots per point and the five
repository validation seeds `[42, 137, 256, 1729, 31415]`. For each seed, one
`numpy.random.default_rng(seed)` was advanced through the 24 points in
`challenge_grid()` order, with exactly one `sample_correlated` call per point.
The same syndrome/truth batch was passed to Track A, `solve.py`'s current final
decoder, and the supplied weighted MWPM baseline. Therefore the comparisons are
paired; they are not separate-sample error-rate comparisons. There was no
benchmark-shot training or table selection.

The integration itself used seed `902100` for the coarse run and `902101` for
the refined run. `tables_qmc.json` records the code hashes, dependency versions,
mask probabilities, QMC 3-SE diagnostics, table margins, and coarse/refined
convergence. On this machine the dependencies were NumPy 2.2.6, SciPy 1.15.3,
Stim 1.15.0, and PyMatching 2.3.1.

## Model checks

The implementation has nine data qubits and eight detectors at L=3. The 512
mask enumeration is sent through `SurfaceCodeExperiment.sample_from_mask`.
The observed detector IDs are 0 through 15; the other 240 of the 256 table
indices are retained as zero-prior entries. GF(2) linearity checks verify that
the enumerated mask-to-syndrome and mask-to-observable maps agree with the
single-mask columns produced by Stim. The logical value is observable 0, in
the same order used by the benchmark.

For `xi=0`, all mask probabilities are exact iid products and the probability
sum is 1 to floating-point precision. For `xi>0`, the probability is a
9-dimensional Gaussian rectangle probability. It is computed by SciPy's
randomized-lattice QMC `_qmvn` conditional transformation with eight randomized
shifts per integration call. This is explicitly a numerical QMC approximation,
not an exact MVN orthant calculation. The final recorded build used 4,096
coarse and 32,768 refined QMC points per mask. Refined wall times were
5.08--5.69 s for correlated points and about 0.02 s for iid points, per point,
offline. Peak whole-process RSS was about 124 MB. QMC raw probability sums were
within `4.5e-6` of one at correlated points in the final build;
they were deliberately not silently renormalized. The maximum coarse/refined
per-mask changes were `6.1e-5` to `1.3e-4`; no MAP table entry changed between
coarse and refined tables. The refined table differs from data-only MWPM only
at syndrome ID 9 for `xi=5` and `xi=10` (both p values). QMC uncertainty is
therefore small relative to most posterior margins, but it is not zero, and the
MAP claim is not an exact-MAP claim.

The covariance audit is in `covariance_validation.json`. It records the actual
coordinates and the benchmark's `1e-12 I` Cholesky jitter. In the intended
unit-diagonal model, with `t = Phi^-1(1-p)`,

```text
Cov(E_i,E_j) = Phi_2(t,t;rho) - (1-p)^2
             = Phi_2(-t,-t;rho) - p^2,
rho = C_ij / sqrt(C_ii C_jj).
```

Here `Phi_2` in the first expression is the conventional lower-tail CDF; the
second expression uses the symmetry-equivalent lower-tail CDF at `(-t,-t)`.
Equivalently, `Phi_2(t,t;rho)-p^2` is valid when `Phi_2` denotes the joint
upper-tail probability. The covariance is binary, not the latent Gaussian
correlation `rho`. The audit evaluates this identity by Plackett one-dimensional
quadrature and compares it with seeded empirical joint frequencies. Because the
production sampler adds tiny diagonal jitter before Cholesky, its realized
marginal differs from `p` only at approximately the 1e-12 relative
implementation level; the audit reports this effective marginal.

## Pilot: 10,000 shots per point, one seed

| stratum | decoder | errors / shots | error rate | score-like errors/M |
|---|---:|---:|---:|---:|
| L=3 | Track A | 380 / 80,000 | 0.004750 | 4,750 |
| L=3 | current final | 390 / 80,000 | 0.004875 | 4,875 |
| full grid | Track A | 726 / 240,000 | 0.003025 | 3,025 |
| full grid | current final | 736 / 240,000 | 0.003067 | 3,067 |
| full grid | supplied baseline | 745 / 240,000 | 0.003104 | 3,104 |

The L=3 Track A versus final paired result was 13 wins versus 3 losses, saving
10 errors. This is only 16 discordant shots, so the pilot is directional rather
than decisive. L=3 point errors were:

| p | xi=0 | xi=2 | xi=5 | xi=10 |
|---:|---:|---:|---:|---:|
| 0.005 | 4 / 4 / 4 | 18 / 18 / 18 | 43 / 44 / 44 | 45 / 48 / 48 |
| 0.010 | 13 / 13 / 13 | 73 / 73 / 73 | 72 / 75 / 75 | 112 / 115 / 115 |

Cells are `Track A / final / supplied baseline`.

## Held-out official run: five seeds, 100,000 shots per point

### L=3-specific result

| decoder | errors / 4,000,000 | error rate | accuracy | 95% stratified normal CI for error rate |
|---|---:|---:|---:|---:|
| Track A | 18,138 | 0.0045345 | 0.9954655 | [0.0044687, 0.0046003] |
| current final | 18,287 | 0.0045718 | 0.9954283 | [0.0045057, 0.0046378] |
| supplied baseline | 18,287 | 0.0045718 | 0.9954283 | [0.0045057, 0.0046378] |

Track A saved 149 errors against the current final on paired samples, a rate
difference of `3.725e-5` (approximate paired 95% CI
`[2.484e-5, 4.966e-5]`). There were 395 candidate wins and 246 losses;
the exact two-sided McNemar p-value was `4.32e-9`. This is a validation-set
summary over eight L=3 points, not evidence that every point improves:

| p | xi=0 | xi=2 | xi=5 | xi=10 |
|---:|---:|---:|---:|---:|
| 0.005 | 233 / 233 / 233 | 956 / 956 / 956 | 2,018 / 2,023 / 2,023 | 2,279 / 2,308 / 2,308 |
| 0.010 | 857 / 857 / 857 | 2,595 / 2,595 / 2,595 | 4,351 / 4,396 / 4,396 | 4,849 / 4,919 / 4,919 |

Again cells are `Track A / final / supplied baseline`. The L=3-specific gain
is concentrated at `xi=5` and `xi=10`; xi=0 and xi=2 are tied in these paired
draws. The eight-point Holm-adjusted McNemar family result is retained in the
JSON receipts; the conclusion here is driven primarily by the paired effect
and its uncertainty, not by an unpaired subtraction.

### Full-grid aggregate and fallback interpretation

The L=5/7 candidate path is intentionally a data-only MWPM fallback, so it is
not a claimed copula improvement:

| decoder | errors / 12,000,000 | error rate | accuracy | score-like errors/M |
|---|---:|---:|---:|---:|
| Track A hybrid | 35,730 | 0.0029775 | 0.9970225 | 2,978 |
| current final | 35,879 | 0.0029899 | 0.9970101 | 2,990 |
| supplied baseline | 36,251 | 0.0030209 | 0.9969791 | 3,021 |

The hybrid saved 149 errors versus final and 521 versus supplied MWPM. Since
all 149 versus-final savings come from L=3, the full-grid improvement should
not be presented as an independent L=5/7 result. Against the supplied baseline,
the L=5/7 fallback happened to save 372 errors in these held-out draws, but
that is a baseline implementation comparison, not a new Track A model claim.

## Timing, memory and deployment tradeoffs

The frozen L=3 table decoder had a maximum build+decode receipt of
`0.00101 s` across held-out points and mean batch decode time
`0.000838 s`; there were no 2.5-second timeouts. The current final had maximum
`0.0257 s` and mean decode `0.00178 s` in the same run. The L=3 table itself is
256 bytes as a NumPy uint8 array; the QMC construction is intentionally offline
and excluded from the official timer. Whole-process high-water RSS reached
approximately 124 MB during integration. All receipt-level build/decode times,
memory readings, table byte counts and timeouts are in the JSONL rows.

The main tradeoff is accuracy versus offline construction: the correlated
tables take roughly a second each at the recorded QMC budget and carry sampling
error, while the resulting decode is a direct array lookup. Increasing QMC
budget improves probability estimates but does not make the method mathematically
exact. The no-files-at-decode property comes from embedding the generated table
in `frozen_tables.py`.

## Artifact audit

`audit_track_a.py` re-derived the retained artifact claims without rerunning
the experiments. It verified all eight 512-mask integrations, 256-entry
tables, 120 held-out JSONL rows, frozen-table/source hashes, and unchanged
coarse/refined predictions. The recorded L=3 saved-error counts by official
seed were `{42: 37, 137: 16, 256: 44, 1729: 32, 31415: 20}`. Its output is
`results_audit.json`; this is an audit of receipts, not an additional test set.

## Conclusion

For this implementation and held-out evaluation, the rigorous conclusion is
that the L=3 copula MAP table is a small but measurable improvement over the
current data-only final on strongly correlated points (`xi=5,10`), with ties at
`xi=0,2` in the five-seed run. It is not an exact MAP solver: correlated mask
priors are randomized QMC rectangle estimates, and QMC normalization residuals
are recorded rather than hidden. The evidence supports retaining this as an
offline L=3 research candidate, while the L=5/7 path should be described only
as a fallback for hybrid aggregate testing. The candidate does not replace or
edit `solve.py`.
