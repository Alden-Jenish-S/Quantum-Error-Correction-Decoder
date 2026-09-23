# Cross-track ablation review (approval checkpoint)

This is the parent integration memo for the three protected research tracks.
No candidate has been merged into `solve.py`. All comparisons below come from
track-owned receipts and use paired syndrome batches within each experiment.

## Candidate summary

| Candidate | Scope | Accuracy result | Timed build+decode | Memory / implementation cost | Recommendation |
|---|---|---:|---:|---|---|
| Current final data-only MWPM | All L, all xi | Reference | L=3/5/7 approximately 0.026/0.12/0.173 s max in Track A full-grid run | Small graph; no extra table | Retain fallback |
| Track A copula MAP table | L=3; current final fallback for L=5/7 | L=3: 4,534/M vs 4,572/M current final on 4M paired shots; 149 fewer errors | L=3 max 0.00101 s; hybrid full-grid max 0.197 s | 256-byte table; 5–6 s offline QMC build per correlated point; QMC approximation | **Promising; approval candidate** |
| Track B parity reweight | L=5/7 | No prediction changes versus iid on seed-42 sweeps | Well below 2.5 s | Low extra cost, but inert | Reject for deployment |
| Track B pair/cluster surrogate | L=5/7 | Worse: pair shortcut +13/+371/+204 errors/M at L=5/7 xi=2/5/10; cluster variants also regress | L=5 max ~0.077 s; L=7 max ~0.224 s | Extra graph construction and incorrect higher-order approximation | Reject |
| Track C rule filter | All L | 3,475/M vs 3,192/M current final on held-out 120k shots; 34 net extra errors | 1.976 s max at 1M shots/point; margin 0.524 s | Large vectorized temporaries; ~1.1 GiB current RSS in run | Reject |
| Track C frozen classifier | All L | 3,192/M tied current final on held-out 120k; 3,034/M vs 3,027/M at 24M runtime run | 2.058 s max at 1M shots/point; margin 0.442 s | More features/temporary memory; classifier effectively inert | Reject |

## Track A details

The Track A L=3 table enumerates all 512 physical X masks, computes the
Gaussian-copula mask priors with randomized QMC rectangle integration, and
chooses the posterior-MAP logical label for each of 256 syndrome IDs. The
covariance audit distinguishes latent covariance from thresholded binary
covariance. The retained result is numerically estimated, not an exact MVN
orthant calculation.

On the five official seeds at 100,000 shots per point, L=3 only:

```text
Track A:       18,138 / 4,000,000 = 4,534 errors/M
Current final: 18,287 / 4,000,000 = 4,572 errors/M
Saved errors:  149 (395 wins, 246 losses)
```

The gains were concentrated at `xi=5` and `xi=10`; `xi=0` and `xi=2` tied in
that held-out run. The full hybrid (Track A at L=3, current final at L=5/7)
scored 35,730/12,000,000 = 2,978/M versus 35,879/12,000,000 = 2,990/M for
current final in the same Track A experiment. This full-grid number must not be
interpreted as a new L=5/7 improvement.

## Track B and C conclusions

Track B confirmed that PyMatching 2.3.1 accepts graphlike decompositions and a
limited correlated two-pass mode, but it does not natively represent arbitrary
Gaussian-copula multi-detector hyperedges. Valid reweighting variants were
inert; pair and cluster source surrogates made the wrong higher-order model and
regressed. No B candidate merits approval.

Track C's spatial features and classifier are valuable diagnostic infrastructure,
but the frozen classifier made zero held-out flips and was slightly worse at
the 1M-shot stress run. The rule-based filter was decisively harmful on held-out
data. No C candidate merits approval.

## Proposed hybrid for user approval

**Proposed deployment:** Track A's frozen L=3 QMC-MAP table plus current
data-only MWPM unchanged at L=5 and L=7. This is the only candidate with a
held-out paired gain and has a large measured runtime margin. It increases
deployment artifact complexity (embedded table and provenance) and retains QMC
approximation uncertainty, but has only a 256-byte runtime table and no file or
network access in `decode()`.

**Not proposed:** Track B graph/hyperedge variants or Track C filters.

The parent will wait for explicit user approval before copying any Track A table
or hybrid logic into `solve.py`, running the final five-seed validation after the
change, regenerating public plots, or updating the final report as a deployed
decoder claim.
