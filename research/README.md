# research/

Experiments that decide what goes into the day-of model. The results are in `RESULTS.md`; the method is in the
repository README.

## Protocol

- Development folds are 5-fold x 3 seeds (42, 43, 44) over the 10,000 in-sample sessions. Every fit that uses labels or
  bars after 49 happens inside the fold.
- The CV score is the mean over seeds of each seed's pooled out-of-fold Sharpe.
- A change is kept when its paired gain over the current best is at least 2.5 bootstrap standard errors (2,000 session
  resamples) and positive for every seed (1.5 until the 25 Nov review). It is then confirmed on fresh targets, the
  held-out sessions' later windows: `uv run python research/confirm_later.py <candidate> <base>`.
- Every script takes `--data <folder>` for new competition files (a new draw of the generator). Results then go to
  `oof_<folder>/` and `trials_<folder>.csv`, never mixed with the known dataset's.
- Hyperparameters come only from the grids declared in the plan, never from open-ended search.
- Every evaluation appends a row to `trials.csv`, which is committed. Out-of-fold predictions go to `oof/`, which is
  not committed.
- The 1,000 datathon sessions stay out of everything here until the final check (`r6_confirm.py`).

## Run a rung

```bash
uv run python research/r0_reference.py
```

Each script prints one line per idea: `idea | CV | worst | Δ vs best (z) | KEEP/drop`.
