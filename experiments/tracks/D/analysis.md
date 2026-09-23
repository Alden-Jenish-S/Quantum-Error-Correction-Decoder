# Track D analysis and conclusion

## Design and references

Tracks A, B and C were read before implementation. A's useful result is the
small L3 copula table; its L5/L7 path is data-only MWPM. B found that pair,
cluster and correlated-matching surrogates worsen L5/L7 or make unsupported
graph assumptions. C's residual rule and ridge morphology score regress or are
inert. Track D therefore tests a different architectural question: whether the
complete syndrome, rather than a graph surrogate or hand-built morphology,
contains enough conditional logical information to justify a direct Bayes
decision.

The immutable IID reference is `reference_iid.IIDDecoder`; it reconstructs the
Track-A data injection order independently of `solve.py`. The `a_reference`
variant is the frozen Track-A candidate for an explicit inherited-L3 reference.
All comparisons use the same syndrome/truth batch within each point.

Training used independent benchmark-model samples only. The 250k training used
SeedSequence entropy `[2718281, L, int(p*1e6), int(xi)]`; after the predeclared
support diagnostic, the permitted 1M/point scale-up used the identical design
and was frozen before any held-out test sample. The exploratory pilot seed was
32452843. The held-out test used untouched seeds `[104729, 130363, 155921]`,
100,000 shots per point, and official 24-point RNG order. TEST therefore has
300,000 shots per point and 2.4M shots per distance. The stress seed was
49979687 and is supplementary only.

## Held-out result

The primary conservative supported-vs-IID result is:

| distance | supported errors | IID errors | saved | rescues / harms | saved errors per million | stratified paired 95% interval | exact McNemar p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L5 | 6,593 | 6,665 | +72 | 157 / 85 | +30.0 | [+17.30, +42.70] | 4.30e-6 | 8.60e-6 |
| L7 | 3,958 | 3,966 | +8 | 19 / 11 | +3.33 | [-1.14, +7.81] | 0.2005 | 0.2005 |

The primary L5 effect is small but positive on this frozen held-out set. L7 is
directionally positive with too few changed shots to establish a reliable
pointwise benefit. No pointwise non-inferiority claim is made.

| variant | L5 errors / 2.4M | L7 errors / 2.4M | L5 rescue/harm | L7 rescue/harm |
|---|---:|---:|---:|---:|
| immutable IID / A fallback | 6,665 | 3,966 | 0/0 | 0/0 |
| raw | 6,559 | 4,120 | 976/870 | 306/460 |
| supported | 6,593 | 3,958 | 157/85 | 19/11 |
| supported without symmetry | 6,614 | 3,965 | 110/59 | 5/4 |

Raw L5 saves 106 with approximate 95% paired interval [+9.08, +79.25] errors/M,
but raw L7 adds 154 with interval [+41.56, +86.77] added errors/M. These are
ablations rather than an invitation to select a new per-distance policy after
seeing TEST. No decoder was changed after freeze.

The complete pointwise table is in `numeric_tables.md`. Important regressions
and ties are retained there rather than hidden:

* `supported` has no negative held-out pointwise delta on the 16 L5/L7 grid
  points after pooling the three seeds. It ties all xi=0 and xi=2 cells,
  improves all four L5 xi=5/10 cells,
  and improves only L7 p=.01, xi=10 by 8 errors; the other L7 cells tie.
  There is one supported seed-by-point regression: seed 130363 at L5 p=.005,
  xi=10 adds one error (5 rescues, 6 harms in 100k shots). The pooled cell saves 11.
* `raw` is not acceptable as a full-grid candidate: it regresses at both L5
  xi=2 cells and every L7 nonzero-xi cell. The worst point is L7 p=.01, xi=5:
  58 added errors per 300k, followed by 55 added at L7 p=.01, xi=10.
* `supported_nosym` is weaker than the supported design on L5 and nearly inert
  at L7, showing that the validated 180-degree pooling supplies useful support
  rather than merely adding table complexity.
* `a_reference` and D's L3 variants reproduce the inherited A L3 behavior; D
  makes no claim of a new L3 architecture.

None of the 16 supported pointwise comparisons passes the prespecified Holm
family at .05. L5 p=.01,xi=10 comes closest at 0.0500098; rounding this to .05
must not be interpreted as passing. The strong distance-level L5 aggregate
does not establish each individual cell's improvement.

For all 24 points (7.2M shots), IID has 21,819 errors, frozen A 21,717,
supported 21,637, raw 21,765 and no-sym 21,665. Supported saves 182 versus IID,
of which 102 are inherited A L3 savings and 80 are new L5/L7 savings. Against
frozen A, the full-grid gain is 11.11 errors/M [6.62, 15.60], rescues/harms 176/96.
This attribution is essential: full-grid benefit is not all new architecture.

The p-values are paired McNemar tests. The L5/L7 primary family
uses Holm correction over two distance-level comparisons; the pointwise report
also retains a 16-point Holm family for transparency. CIs are fixed-stratum
paired normal intervals. A zero discordance is reported with a one-sided 95%
upper bound in JSON and is not treated as proof of exact equality.

## Support, storage and runtime

At 1M training shots per nonzero-xi point, L5 has two supported overrides at
each of p=.005 xi=5/10 and p=.01 xi=5, and four at p=.01 xi=10. L7 has four overrides
only at p=.01, xi=10 and none elsewhere. Test support was high for the
observed shot mass, but the rare nonzero syndrome tail was sparsely covered:
for example L7 p=.01 xi=10 had 4,558 unseen pooled syndrome shots among
300,000 total, and only 30 test shots changed under the supported policy.
That is 11.24% of nonzero-syndrome test shots unseen after pooling, versus
13.47% without pooling. At L5 p=.01 xi=10, only 113 of 25,107 nonzero-syndrome
shots are unseen after pooling (0.45%). High all-shot support coverage is
dominated by syndrome zero; it should not obscure the L7 tail. This is a
plausible reason for the conservative L7 behavior and its wide uncertainty.

The supported uncompressed L7 override payload is 16 bytes across all six
nonzero-xi L7 points (the no-sym ablation is 4 bytes). The full generated
`frozen_tables.py` is 58,173 bytes because it also contains all L5 table
payloads, the much larger raw L7 ablation (62,684 uncompressed bytes), and
Python source syntax. Supported tables alone have 24,592 uncompressed bytes
or 1,340 encoded string characters across L5/L7; Python container and matching
graph overhead are additional. The L5 logical table is 4,096 bytes
per point before source encoding. The complete candidate remains well below
the requested 200KB L7 representation when counted as its L7 decision payload.

The official-order 100k test had no timeouts; maximum build+decode was about 0.228s
for supported L5 and 1.292s for supported L7. The representative 1M/point
hard-point stress had no timeouts, with maximum build+decode 0.0305s at L5 and
0.4843s at L7. The stress decoder results were supported saved 123 L5 errors
and 32 L7 errors. Stress paired rescues/harms are 261/138 and 58/26, respectively;
L7's supplementary improvement is 32 errors/M [14.04, 49.96]. Only p=.01, xi=10
at L5/L7 was decoded, though all 24 points were sampled to preserve RNG order.
These two selected points do not establish full-grid runtime or primary L7
accuracy. The stress result is not part of primary model selection. Whole-process
high-water RSS was approximately 297MB on the test run and 1.74GB during the
sequential 1M stress run; these include million-shot arrays and allocator
retention, not incremental decoder allocation. Current RSS is unavailable
(`psutil` was not installed); JSON uses null, and only OS high-water RSS is
interpreted. In MiB, training/test/stress peaks were 169.9/283.5/1655.2.
The final 12M training samples took 21.27s in 25k batches; the initial 3M took 11.34s.
Imports occur before timing except the first frozen table import, which occurs
in the first posterior decoder build and is included there. Build/decode
timings are local observations and can be affected by concurrent workspace work.

## Statistical limitations and failed assumptions

The posterior is an empirical, finite-training approximation; it is not an
exact Gaussian-orthant integration or an exact Bayes decoder. The fixed
Beta(1,9) residual prior supplies ten IID-favoring pseudo-observations; n>=20
and posterior lower bound >.5 are fixed pre-test rules. The 99% per-cell
posterior criterion is not a global frequentist guarantee over thousands of
searched cells. Symmetry does not multiply the effective sample size: each
independent mask draw enters one canonical cell exactly once. Independent
draws remain independent under deterministic canonicalization, even if they
share a physical mask or syndrome.

Rare labels make empirical majority decisions unstable. At 1M training,
raw L7 changes 15,671 syndrome IDs across its six nonzero-xi points, often on
one observation; supported changes just 4. This large coverage/variance tradeoff
is the central failure of the raw architecture. The 1M scale-up reduced pilot
raw L7 harm from 18 to 8 errors but did not reverse its held-out failure. Pilot
reuse was exploratory and disclosed; the three TEST seeds were untouched
until the single final freeze. There was no missing-shot filtering, training
on test truth, online updating or external data.

## Verification

`test_track_d.py` has 34 passing tests. It checks GF(2) mask/syndrome/logical
linearity for L3/L5/L7, exhaustive single-error symmetry action, valid and
rejected isometries, exact IID equivalence, all 4096 L5 lookups, L7 storage,
one-observation-per-sample pooling, posterior shrinkage, deterministic outputs,
zero-shot shape, frozen-hash protection, official RNG ordering including L3,
and a static no-I/O/no-training decode check. `audit_track_d.py` rechecks all
72 held-out rows, 24 pilot rows at each training stage, both stress rows,
summary recomputation, paired-count identities, frozen hashes, support counts,
serialization and pointwise regression lists. Its receipt is `audit.json`.
`reproduce_training.py` regenerated all 12M training labels in memory and checked
the exact counts and embedded strings against retained artifacts. Source,
seed and dependency provenance are in `freeze.json` and `artifact_manifest.json`.
The scope incident involving two accidentally created, immediately removed
root files is disclosed in `INCIDENT.md`; existing root files were not touched
by this track.

## Recommendation

Do not deploy or merge this candidate automatically. Keep it as a reviewed
research artifact. The conservative direct posterior is promising for L5 on
the frozen held-out design, with no observed supported pointwise regression.
The primary L7 effect remains uncertain despite positive supplementary
hard-point evidence. Reject the raw full-grid variant. Submit the conservative
design for user review, with stronger evidence at L5 than L7. Any future
approval should be a deliberate user review of the
pointwise receipts, storage accounting and the L7 uncertainty, not an
aggregate full-grid claim.
