# Track D design (written before training/test sampling)

Hypothesis: conditioning directly on the complete active syndrome can improve
logical classification over data-only MWPM at L5/L7 under the actual copula law.
No graph surrogate or morphology classifier is used.

Training: initial 250,000 independent masks per nonzero-xi L5/L7 point; per-point
NumPy SeedSequence `[2718281, L, int(p*1e6), int(xi)]`. Batch 25,000. One mask is
one observation. Train on the exact benchmark sampler and Stim labels. Allowed
scaling to 1M/point only if pilot diagnostics show supported changes with useful
held-out gain. No official validation seeds used for training/design.

Symmetry: test all eight square isometries on exhaustive single-qubit probes;
retain only permutations preserving active check support and whose observable
action is a GF(2) linear function of syndrome. Canonicalize each training sample
once; do not multiply sample size by augmentation. Compare no-symmetry ablation.

Decisions: raw posterior majority of canonicalized labels with IID tie/backoff;
supported posterior flips only when sample size >=20 and the 1% lower quantile
of Beta(wrong+1, right+9) exceeds .5. Ten prior observations favor IID. This
per-cell posterior criterion is shrinkage, not simultaneous frequentist coverage.
L5 materializes 4096 decisions; L7 stores only sorted overridden syndrome IDs.
All xi=0 points use exact immutable IID. L3 non-IID variants use unchanged A.

Exploratory pilot: seed 32452843, 50k/point, all 24 points official RNG order.
Freeze source/table hashes and design before any TEST evaluation. TEST seeds
104729,130363,155921, 100k/point, all24 points, one default_rng per seed with one
official sample_correlated call at every point including L3. No test-based tuning.
Evaluate IID, frozen A, raw, supported, supported_nosym on identical batches.

Report paired rescue/harm, saved-error rates with fixed-stratum paired normal
95% intervals, exact two-sided McNemar/binomial tests, and Holm correction over
16 L5/L7 pointwise supported-vs-IID comparisons. Zero discordance does not prove
equivalence. Independent sample units are shots, not pooled symmetry copies.
Primary comparisons: supported vs IID separately at L5 and L7 (Holm family 2).
Fullgrid aggregate is secondary and must expose inherited L3 gains.

If promising, one runtime stress uses 1M draws/point in full official RNG order
but decodes only hard L5/L7 p=.01 xi=10, seed 49979687. This is supplementary
runtime/accuracy evidence, not part of primary held-out test or design selection.
Memory: whole-process RSS/high-water includes sampling; table bytes separately.
No deployment, root-file edits, commits, or approval-ledger changes authorized.
