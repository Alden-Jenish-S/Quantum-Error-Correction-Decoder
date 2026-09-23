# Track B analysis

## 1. PyMatching 2.3.1 audit

The installed runtime is PyMatching **2.3.1**, with Stim 1.15.0, NumPy 2.2.6,
SciPy 1.15.3, and Python 3.10.20. The audit is reproduced in
`graph_diagnostics.json` and by `--audit-json`.

### What the DEM loader accepts

`Matching.from_detector_error_model(dem, enable_correlations=False)` creates a
node for each detector and turns each graphlike DEM mechanism into an edge.
Graphlike means one or two detection events. A DEM error touching more than two
detectors is ignored unless Stim supplies a decomposition into graphlike
components. The direct audit used

```text
error(0.01) D0 D1 D2 D3 L0
```

and got **zero** native matching edges. Therefore a general Gaussian-copula
multi-qubit event cannot be inserted into native PyMatching as a true
hyperedge. Pretending that each such event is independently split into edges
changes its probability model and is not acceptable.

### Correlated matching

PyMatching 2.3.1 has `enable_correlations=True` when loading a DEM and when
calling `decode`/`decode_batch`. This is a two-pass heuristic for correlated
graphlike decompositions. It can exploit a decomposable hyperedge such as

```text
error(q) D0 D1 L0 ^ D2 D3
```

but it does not turn PyMatching into a general hypergraph decoder. The audit
found two graph edges for this decomposable four-detector source. The matching
graph still contains only graph edges; the correlation metadata affects the
second pass.

### Fault IDs, observables, and boundaries

`fault_ids` are attached to edges and are returned as an observable prediction
column by `decode_batch`. This track preserves the data-only DEM's one logical
fault-ID column. In particular, it does not replace observable targets with
physical-mask IDs. The audit confirms that a boundary edge is a virtual edge
from a detector to `None` and can carry a fault ID; its observable prediction
is retained. Parallel edges are merged. The merged edge keeps only the first
edge's fault IDs, and its probability is the independent-mechanism merge

```text
p_merge = p_1 + p_2 - 2 p_1 p_2.
```

Consequently, arbitrary parallel mechanisms with different logical labels
cannot be represented losslessly by a single ordinary matching edge. Candidate
construction checks the physical-column labels before relying on this merge.

## 2. Candidate formulas

### Baseline: data-only iid MWPM

For each data qubit, `candidate_b.py` reconstructs the challenge injection
order and appends `X_ERROR(q, p)` before the suffix. It then uses
`detector_error_model(decompose_errors=True)` and
`Matching.from_detector_error_model`. The graph has exact detector connectivity,
boundary edges, fault IDs, and observable handling for the iid data-only model.

### Bivariate Gaussian-copula parity reweighting

For two data coordinates separated by `d`, let `t = Phi^{-1}(1-p)` and
`rho = exp(-d/xi)`. The joint error probability is computed before decoding as

```text
J(p,rho) = p^2 + integral_0^rho
           exp(-t^2/(1+r)) / (2*pi*sqrt(1-r^2)) dr.
```

This is Plackett's identity for the equal-threshold bivariate normal tail.
For a pair of physical errors that produces the same graphlike detector column,
the parity-flip probability is

```text
q_parity = P(X_i != X_j) = 2(p - J).
```

The replacement edge weight is `log((1-q_parity)/q_parity)`. `parity_reg`
uses the midpoint between this value and the iid edge weight. At `xi=0`,
`rho=0`, `J=p^2`, and the intended finite-limit weight reduces to the iid
parity weight. The implementation deliberately returns the original graph for
`xi<=0`, so xi=0 is exactly bit-for-bit unchanged, including tie behavior.

This is a legitimate marginal/second-order reweighting, but it cannot encode
the full conditional distribution of all other qubits. It also changes only
parallel graphlike mechanisms, so it has no route to represent a general burst.

### Pair-source marginal-preserving surrogate

The `pair_*` variants use only nearest-neighbor data-qubit pairs (`d=2`) and
construct a latent XOR-source model. If `r_ij` is the source rate, its
second-order moment contribution obeys

```text
J_ij - p^2 = ((1-2p)^2 / 4) * (exp(2 r_ij) - 1),
r_ij = 1/2 log(1 + 4(J_ij-p^2)/(1-2p)^2).
```

The rate budget is chosen from `-log(1-2p)` and scaled to keep all singleton
probabilities nonnegative. Singleton sources fill the remaining marginal
budget exactly; diagnostics show `physical_marginal_max_abs_error=0.0` for the
tested points. `pair_uncorr` presents the resulting graphlike DEM to ordinary
MWPM. `pair_corr` and `pair_corr90` ask PyMatching's two-pass correlated
matching to exploit the explicit `^` pair decomposition.

This is not the benchmark sampler. It matches a local pair moment and the
single-qubit marginals, but not the Gaussian copula's full higher-order law.

### Cluster surrogate

`cluster_corr` uses four-site 2x2 plaquette sources. Its source probability is
the connected tree approximation

```text
q_4 = max(0, p * (J_nn/p)^3 - p^4),
```

then a marginal budget correction is applied. This is explicitly a burst/cluster
surrogate, not an exact four-variate Gaussian probability and not native
hyperedge support. Its DEM is decomposed for the PyMatching correlated pass.

### Graphlike pair shortcut

`pair_shortcut` uses the same pair source but computes its detector and logical
parity. If the resulting detector parity contains at most two detectors, it is
encoded as one graphlike edge. This is technically valid for that parity, but
it is still only a pair-source approximation to the actual Gaussian-copula
data. It is not a general auxiliary-node hypergraph embedding.

## 3. Shared-batch results

The 5k and 10k seed-42 files each cover 16 requested points total: L in
`{5,7}`, both `p` values, and all four `xi` values. Each candidate sees the same
syndrome/truth batch at each point. The main observations are:

* `iid`, `noop`, `parity`, and `parity_reg` are exactly equal on every tested
  batch, including every xi=0 stratum.
* `pair_corr` and `pair_corr90` are substantially worse already at xi=2 and
  remain worse at xi=5 and xi=10.
* `cluster_corr` is equal to iid through xi=5 in these batches, then is worse
  at xi=10.
* `pair_shortcut` and `pair_shortcut_corr` are close to iid but do not improve
  it; the 10k run had positive deltas at L=5 and L=7 for xi=2/5/10.

The exact per-point counts, paired rescued/harmed counts, elapsed times, and
process RSS are in the JSON summaries. A paired exact binomial test on rescued
versus harmed shots is descriptive here; the seed-42 selection sweep is
exploratory and is not treated as an unbiased model-selection procedure.

## 4. Official validation

Only `iid` and `pair_shortcut` were taken forward. This is intentionally
conservative: the other correlation variants were already clearly negative on
the shared seed-42 sweep. The official run uses seeds
`42, 137, 256, 1729, 31415`, 100,000 shots per point, and therefore 500,000
shots per L/p/xi stratum after aggregating five seeds. Pooling both p values
gives 1,000,000 shots per L/xi stratum. These are official seeds and sampling
order, but a reduced shot budget relative to the official 1M shots per point.

### Aggregate error counts and rate deltas

Here `delta` is `pair_shortcut errors - iid errors`; positive is worse. These
are computed from `results_official_100k.jsonl`.

| L | p | xi | iid errors / 500k | pair shortcut / 500k | delta | rescued / harmed |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | .005 | 0 | 19 | 19 | 0 | 0 / 0 |
| 5 | .005 | 2 | 248 | 250 | +2 | 0 / 2 |
| 5 | .005 | 5 | 1104 | 1231 | +127 | 82 / 209 |
| 5 | .005 | 10 | 1789 | 1866 | +77 | 120 / 197 |
| 5 | .01 | 0 | 144 | 144 | 0 | 0 / 0 |
| 5 | .01 | 2 | 974 | 985 | +11 | 1 / 12 |
| 5 | .01 | 5 | 2762 | 3006 | +244 | 191 / 435 |
| 5 | .01 | 10 | 3837 | 3964 | +127 | 277 / 404 |
| 7 | .005 | 0 | 4 | 4 | 0 | 0 / 0 |
| 7 | .005 | 2 | 46 | 51 | +5 | 2 / 7 |
| 7 | .005 | 5 | 554 | 661 | +107 | 72 / 179 |
| 7 | .005 | 10 | 1147 | 1233 | +86 | 174 / 260 |
| 7 | .01 | 0 | 20 | 20 | 0 | 0 / 0 |
| 7 | .01 | 2 | 345 | 363 | +18 | 13 / 31 |
| 7 | .01 | 5 | 1711 | 1918 | +207 | 266 / 473 |
| 7 | .01 | 10 | 2888 | 3113 | +225 | 380 / 605 |

The JSON summary pools both p values within each `(variant,L,xi)` group:
L5 xi2/5/10 deltas are +13/+371/+204 and L7 deltas are +23/+314/+311 per
million shots. The table above breaks the same rows back out by p.

At xi=0, all 20 point/seed batches are exactly unchanged. There is
no regression sacrifice at the independent-noise point. At xi>0, however,
the shortcut's harmful flips dominate rescued flips in every stratum. Holm-
adjusted paired p-values for the nonzero-xi comparisons are reported in the
summary; the L5/L7 xi=2 adjusted values are approximately .00195 and .00219.
The multiplicity family comprises six nonzero-xi L/xi comparisons. Individual
p-specific low-count strata have much less power; no significance claim is
made for every p-specific stratum. These results reject this particular
surrogate; they do not rule out all graph-based approaches.

## 5. Runtime and memory

All variants stayed well below the official 2.5 s build+decode timer. Typical
seed-42 maxima were about 0.05–0.06 s for L=5 and 0.09–0.13 s for L=7 at 5k
or 10k shots. The official shortcut maximum was 0.077 s for L=5 and 0.224 s
for L=7; iid maxima were 0.060 s and 0.130 s. There were zero timeouts and no
decode exceptions. The occasional larger L=7 shortcut value is still far
inside the limit but demonstrates an avoidable runtime cost.

The runner records both current RSS and process high-water RSS. High-water
RSS includes sampling and retained allocator state; it is not isolated decoder
allocation. The official summary reaches roughly 270.6 MiB current process
RSS. The graph sizes are
small: data-only iid has 21 edges/6 boundary edges for L=5 and 43 edges/8
boundary edges for L=7, with one observable fault-ID column. Reweighted and
surrogate graphs retain the same detector/fault-ID interface; graph-specific
edge lists and weight ranges are in `graph_diagnostics.json`.

## 6. Final decision

Technically valid:

1. data-only iid MWPM;
2. copula parity reweighting and its regularized control;
3. the explicitly labeled pair-source surrogate and its two-pass variant;
4. graphlike pair shortcut variants when the resulting parity has at most two
   detectors;
5. the four-site cluster as a clearly labeled approximation.

Technically unsupported: a claim that native PyMatching is doing exact general
hyperedge matching for this Gaussian-copula noise. The audit rules that out.

Numerically, none of the correlation-aware candidates improves iid on the
requested L=5/L=7 grid. The exact parity reweights are inert for this graph,
while pair and cluster approximations make wrong graph assumptions. The
recommended parent-harness choice is therefore the unchanged data-only iid
candidate. No candidate from this track merits approval as a replacement.
