# Submission constraint compliance

The challenge submission is the root-level `solve.py`. The research package,
benchmark runner, and experiment scripts are not imported by the submission.

## Static checks

`tests/test_submission.py` verifies:

- `solve.py` is smaller than 200,000 bytes;
- imports are limited to `numpy`, `pymatching`, `stim`, and `typing`;
- the generated data-only circuit exactly matches the benchmark construction;
- all 24 official parameter points can construct a decoder;
- outputs have shape `(shots,)`, dtype `numpy.uint8`, and binary values;
- input batches remain unchanged, including read-only arrays;
- empty batches are handled;
- the file loads in a fresh interpreter without `qec_benchmark` available.

## I/O audit

The isolated audit installs Python audit hooks and guards filesystem, socket,
subprocess, memory-map, and relevant `os` operations while `decode()` runs. It
also includes positive controls to confirm that the guards fire. This is a
Python-level audit, not a kernel syscall sandbox; native-library behavior is
therefore not proved beyond the observed implementation and test scope.

## Runtime audit

The official evaluator measures decoder construction plus batch decoding and
marks a point all-wrong if that phase exceeds 2.5 seconds. The final candidate
constructs one small matching graph and calls vectorized `decode_batch`; the
recorded full benchmark had zero timeouts. Re-run the standard benchmark on the
target machine because performance is hardware and dependency-version
dependent.

## Reviewer checklist

```bash
uv run pytest
wc -c solve.py
uv run python run.py --grid tiny --shots 1000
uv run python run.py --shots 1000000
```

No benchmark or scoring source files need to be changed to submit `solve.py`.
