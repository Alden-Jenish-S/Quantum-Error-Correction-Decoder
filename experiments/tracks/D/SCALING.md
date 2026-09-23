# Exploratory scale-up decision, before TEST seeds

Initial independent 250k/point training took 11.34 seconds and 165.6 MB peak
whole-process RSS. It found two supported L5 xi10 p=.01 syndrome overrides,
both with 86 canonical observations, and no L7 supported override. Separate
50k/point pilot seed 32452843 yielded L5 rescue/harm 11/7 (saved4/400k;
95% paired interval includes zero). Raw L5 tied overall and raw L7 lost18.

The positive but imprecise L5 signal and low training cost justify the single
pre-permitted scale-up to 1M/point to assess whether posterior support is the
limitation. No threshold, prior, symmetry rule or variant is changed. L7 raw
harm is explicitly retained. This is exploratory budget selection, not proof
of improvement. Original stage artifacts are preserved in `stage_250k/`.

After scale-up, rerun the same exploratory pilot for learning-curve diagnostics,
then freeze regardless of its outcome and evaluate all variants on the three
previously untouched TEST seeds. No additional training budget is planned.
