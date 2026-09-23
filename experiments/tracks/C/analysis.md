# Track C analysis

## 1. Protocol and provenance

The benchmark semantics were preserved: one `default_rng(seed)` was used per
seed, all 24 official points were sampled in `challenge_grid()` order, and the
same syndrome/truth batch was passed to supplied baseline, current final,
rule, and classifier decoders. Timing starts at fresh decoder construction and
ends at `decode` return. Sampling, scoring, feature-strata accounting, and JSON
writing are outside the timer. The output contract was checked as exact shape
`(shots,)`, dtype `uint8`, and binary values.

The classifier training seeds were fixed in advance as 8011 and 8017. The
controlled held-out seed was 9029. No seed-42 observations were used to fit or
select the score. Official validation seeds are `[42, 137, 256, 1729, 31415]`;
they were not run because neither candidate passed the held-out promotion
gate.

Environment in the result receipts: Python 3.10.20, NumPy 2.2.6, SciPy
1.15.3, Stim 1.15.0, and PyMatching 2.3.1.

## 2. Geometry and implementation checks

`test_track_c.py` compares vectorized connected-component summaries with a
plain graph traversal on independently generated syndrome arrays for all three
distances. It also checks the Stim-derived coordinate order, expected detector
counts, expected data-qubit counts, expected spatial edge counts, output
contract, empty batches, and the absence of `for`/`while` nodes in the decoder
method itself. The result was:

```text
TRACK C TESTS PASS
```

The full feature extractor uses exact component summaries. Its million-shot
candidate path reduces retained diagnostics to two columns after deciding and
uses direct vectorized formulas for empty, singleton, and pair rows; this is a
memory optimization, not truth-dependent branching.

## 3. Offline score receipt

The score was fit with ridge least squares on 240,000 rows: two seeds × 24
points × 5,000 shots. The training base had 737 errors; the frozen threshold
made 4 flips and 735 training errors. These are training diagnostics only and
are not used inside `decode`. The fitted constants are embedded in
`candidate_c.py`; `classifier_fit.json` records means, scales, coefficients,
threshold, seeds, and row counts.

## 4. Held-out controlled sweep

The held-out run is `results_controlled_5k.jsonl` with seed 9029, 5,000 shots
per point, 24 points, and therefore 120,000 paired shots per decoder.

| decoder | errors / 120,000 | score / million | max build+decode (s) |
|---|---:|---:|---:|
| supplied baseline | 387 | 3,225 | 0.0705 |
| current final | 383 | 3,192 | 0.1082 |
| rule | 417 | 3,475 | 0.2981 |
| classifier | 383 | 3,192 | 0.2195 |

The supplied baseline and current final differ by 4 errors in this batch. The
classifier had zero paired changes versus final: rescued=0, harmed=0,
McNemar exact p=1.0. Zero discordances are not proof of equality; a simple
95% upper bound on an unseen discordance rate is about `3/120000`, or 25 per
million. The classifier therefore supplied no evidence of improvement.

The rule had 31 rescued and 65 harmed shots, for `+34` errors versus final,
or `+283` errors per million. The approximate paired 95% interval for its rate
difference is `[+123, +443]` per million; the exact paired binomial p-value is
0.000675. This is a clear held-out regression.

### Rule delta by L, p, and xi

Entries are `rule errors - final errors`; positive is worse. Each cell contains
5,000 paired shots.

| L | p | xi=0 | xi=2 | xi=5 | xi=10 |
|---:|---:|---:|---:|---:|---:|
| 3 | .005 | 0 | 0 | 0 | 0 |
| 3 | .010 | 0 | 0 | 0 | 0 |
| 5 | .005 | 0 | 0 | 0 | +2 |
| 5 | .010 | 0 | 0 | +1 | +1 |
| 7 | .005 | 0 | 0 | +5 | +2 |
| 7 | .010 | 0 | 0 | +12 | +11 |

The classifier's corresponding delta is zero in every L/p/xi cell in this
held-out batch. The rule's regressions are concentrated at L=7 and xi=5/10,
with the largest p=.01 effects.

### Syndrome-weight and cluster strata

The runner records weight bins 0 through 8, with 8 meaning 8 or more, and
cluster classes 0 through 4. The classifier made no held-out flips, so all of
its strata are unchanged. The rule's nonzero paired strata were:

| weight | class | shots | rescued | harmed | delta |
|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 600 | 3 | 2 | -1 |
| 5 | 3 | 37 | 2 | 16 | +14 |
| 5 | 4 | 171 | 6 | 27 | +21 |
| 6 | 4 | 122 | 7 | 6 | -1 |
| 7 | 3 | 10 | 1 | 4 | +3 |
| 7 | 4 | 31 | 6 | 6 | 0 |
| 8+ | 3 | 13 | 1 | 0 | -1 |
| 8+ | 4 | 42 | 5 | 4 | -1 |

The harmful signal is therefore not a generic syndrome-weight effect; it is
mostly the connected/mixed cluster classes at weight 5, especially class 4.
These strata are descriptive and were not used to retune the frozen policy.

## 5. One-million-shot runtime and memory

`results_runtime_1m.jsonl` uses seed 9029 and 1,000,000 shots at each of the
24 points. All 96 rows had valid output contracts and zero timeouts.

| decoder | errors / 24,000,000 | score / million | max build+decode (s) | max build (s) | max decode (s) |
|---|---:|---:|---:|---:|---:|
| supplied baseline | 73,442 | 3,060 | 0.364 | 0.083 | 0.282 |
| current final | 72,657 | 3,027 | 0.244 | 0.111 | 0.132 |
| rule | 80,892 | 3,370 | 1.976 | 0.233 | 1.772 |
| classifier | 72,805 | 3,034 | 2.058 | 0.207 | 1.850 |

The classifier is 148 errors worse than final over 24M shots; the rule is 8,235
worse. The classifier has a 0.442s maximum margin to the 2.5s limit, and the
rule has a 0.524s margin. Maximum elapsed times by distance were:

| decoder | L=3 | L=5 | L=7 |
|---|---:|---:|---:|
| supplied baseline | 0.055s | 0.115s | 0.364s |
| current final | 0.060s | 0.121s | 0.244s |
| rule | 0.214s | 0.805s | 1.976s |
| classifier | 0.279s | 0.717s | 2.058s |

The largest observed current-RSS values for rule/classifier rows were about
1,095 MiB and 938 MiB respectively; the process-wide high-water RSS reached about 1,601 MiB during the
full sequential run. These are whole-process diagnostics including sampled
syndrome arrays, allocator reuse, and previous decoder work, not isolated
candidate allocations or a portable memory guarantee. The O(shots×nodes)
component path avoids the earlier shots×nodes×nodes temporary.

## 6. Decision

* **Keep as research code:** the Stim-derived geometry, exact vectorized full
  feature extractor, diagnostic receipts, and parent-harness-compatible
  `build_decoder(point, variant=...)` API.
* **Reject for approval:** `rule`, because its held-out paired harm dominates
  rescue (65 versus 31) and it loses substantially at 1M shots.
* **Reject for approval:** `classifier`, because it made no held-out changes
  and is slightly worse at 1M shots while consuming substantially more decode
  time than current final.
* **Official five-seed validation:** intentionally not run. The predeclared
  condition was “promising” held-out performance; neither candidate met it.

The current data-only MWPM prediction remains the appropriate parent-harness
choice for this track. No Track-C code should be copied into `solve.py` based
on these measurements.
