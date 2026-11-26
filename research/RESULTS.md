# Research results (known Algo Arena dataset)

**Current best: `r3_bundle_aug_clip2`** (CV 3.568, worst fold 3.263; confirmation 3.562), frozen as `primary`

Every number uses the development folds (5-fold x seeds 42/43/44, pooled out-of-fold Sharpe per seed, mean over
seeds), unless it says otherwise. The format is `idea | CV mean | worst fold | Δ vs best (z) | kept?`. The full log is
`trials.csv`.

## Reference (R0)

| Idea | CV mean | Seeds | Worst fold | Note |
|---|---|---|---|---|
| r0a_baseline | 3.400 | 3.409 / 3.412 / 3.381 | 2.963 | quickstart drift + main industry ÷ vol |
| r0b_sector | **3.537** | see trials.csv | 3.225 | Edinburgh-style: joint template fit (kernel 0.5/1×9/0.5, WLS), 7 decayed residual sums, E[r]/E[r²]; Δ vs r0a +0.137 (z 3.0) **kept** |
| r0b_sector_volunits | 3.539 | see trials.csv | 3.265 | same, push ∝ vol in the fit; Δ +0.002 (z 0.1), dropped |
| r0c_winner | 3.471 | see trials.csv | 3.194 | 7 Nov winner-style: lag 1–9 rates (+ OOS halves), counts, refined drift, peers, bar-risk sizing; +0.071 vs r0a (z 1.7) |

**Reference = `r0b_sector` (3.537)**, the best of R0a/b/c. Every later rung is compared with the current best, starting from this.

## Headroom (`headroom.py`, optimistic: rates fitted on all in-sample sessions)

| What the predictor knows | ÷ vol^0 | ÷ vol^1 | ÷ vol^2 |
|---|---|---|---|
| owed push, all headlines | 2.708 | 2.722 | 2.637 |
| owed push, main industry | 3.105 | **3.151** | 3.083 |
| plus headlines in bars 50–58 (unknowable) | 4.127 | **4.184** | 4.079 |

The ceiling of a headline-drift approach is about 4.2. The model's extra features (unexplained move, counts) are worth
about +0.25 over the owed push alone.

## Ladder

(one line per rung, appended as rungs run)

### R1: generator fit (`r1_generator.py`). Nothing beat `r0b_sector`.

| Step (vs current best) | CV | Δ (z) | Verdict |
|---|---|---|---|
| kernel flat lags 1–10 | 3.533 | −0.004 (−0.6) | drop |
| kernel measured from data, label-free: 0.55, 0.99, 1.01, 0.95, 1.02, 0.97, 0.98, 1.00, 0.94, 1.02, 0.57 | 3.536 | −0.001 (−1.0) | drop: it **confirms** 0.5 / 1×9 / 0.5 |
| coarse event types (`type`) | 3.529 | −0.008 (−0.5) | drop |
| ridge λ 50 / 800 | 3.535 / 3.542 | −0.002 / +0.005 | drop |
| without OOS first halves in the rate fit | 3.507 | −0.030 (−2.0) | drop: the first halves matter |
| no residual momentum / short half-lives | 3.474 / 3.527 | −0.064 / −0.011 | drop: momentum matters |
| + news count, live count, conflict, repeat push, peer push | 3.528–3.541 | −0.009…+0.004 | drop |
| calibration α 10 / 200 | 3.539 / 3.531 | +0.002 / −0.007 | drop |

The linear structural model is saturated at about 3.54. Gains have to come from sizing, from better calibration of
risk, or from diverse learners in a blend.

### R2: sizing (`r2_sizing.py`, `r2b_second_moment.py`), with two pre-registered R1 add-ons. Nothing beat `r0b_sector`.

| Step (vs current best) | CV | Δ (z) | Verdict |
|---|---|---|---|
| R1 add-on: outsider push (off-industry headlines as their own feature) | 3.524 | −0.013 (−1.7) | drop |
| R1 add-on: surprise (realised vs expected move of recent headlines) | 3.533 | −0.004 (−0.7) | drop |
| risk model for E[r²] (bar ranges, counts, owed²) | 3.538 | +0.000 (0.1) | drop: variance in vol units is flat |
| vol sizing m·vol^−p, p = 0.5 / 1 / 1.25 / 1.5 | 3.441 / 3.406 / 3.377 / 3.340 | −0.10 … −0.20 (z ≤ −2.9) | drop: the m² term is essential |
| clip at 2 / 3 s.d. | 3.552 / 3.537 | +0.014 (0.9) / −0.001 | drop: noise |
| σ² = residual variance | 3.539 | +0.001 (1.2) | drop |
| m² weight 0.5 / 2 | 3.508 / 3.542 | −0.029 (−2.7) / +0.004 (0.3) | drop: weight 1 (theory) is right |

Diagnostics behind this (25 Nov):
- **Rates:** perfect (leaky) rates would score 3.574, so better rates are worth at most +0.04.
- **Vol:** a fixed per-session parameter. Future 10-bar variance correlates 0.87 with the past 50 bars, recent vol adds
  nothing, and the variance ratio between the two halves has median 1.00.
- **Push shape and size:** the same shape for every common template. There is no detectable per-headline variation in
  push size: early and late residuals correlate at 0.034, against −0.030 for a control, n = 2,804.
- **Industry ties:** in only 1.2% of sessions.

### R3: time-shift augmentation (`r3_augment.py`)

| Step (vs current best) | CV | Δ (z) | Verdict |
|---|---|---|---|
| calibration on every 4th window (offsets 0, 4, …, 40) | 3.554 | +0.016 (1.3) | just short of the bar |
| calibration on all 41 windows | 3.554 | +0.016 (1.2) | same; 3x slower |
| **bundle: every-4th windows + clip at 2 s.d.** (pre-registered) | **3.568** | **+0.030 (1.5)** | **kept, new best** |

### R4: diverse learners as standalone signals (`r4_learners.py`)

| Learner | CV | Δ vs best (z) | Note |
|---|---|---|---|
| owed-push TF-IDF + ridge, α 1000 / 3000 / 10000 | 3.187 / 3.192 / 3.082 | −0.38 … −0.49 | no augmentation |
| same with augmentation (every 4th window), α 1000 / **3000** / 10000 | 3.381 / **3.405** / 3.401 | −0.16 … −0.19 | augmentation adds about +0.2 |
| template × age MLP, 32 / **64** hidden (21 windows) | 2.918 / **2.933** | −0.65 / −0.63 | weak on its own |

Best of each goes to R5: `r4_tfidf_a3000_aug` and `r4_mlp_h64`.

### R5: blend (`r5_blend.py`). Nothing beat the generator model alone.

| Weights (generator, TF-IDF, MLP) | CV | worst | Δ vs best (z) |
|---|---|---|---|
| 1, 1, 0 | 3.532 | 3.233 | −0.036 (−1.5) |
| 1, 0, 1 | 3.403 | 3.033 | −0.164 (−3.5) |
| 1, 1, 1 | 3.445 | 3.152 | −0.123 (−3.0) |
| 2, 1, 1 | 3.501 | 3.207 | −0.067 (−2.2) |
| 3, 1, 1 | 3.527 | 3.232 | −0.041 (−1.7) |

The diverse learners estimate the same signal with more noise. The structural model already holds it, so blending only
dilutes it.

### R6: confirmation, check set, stress tests (`r6_confirm.py`)

| Candidate | dev CV | confirmation CV (seeds 1042–1044) | check set, 1,000 datathon sessions (±0.51) |
|---|---|---|---|
| r3_bundle_aug_clip2 | **3.568** | **3.562** | 4.032 |
| r3_aug_every4 | 3.554 | 3.549 | 4.088 |
| r2_clip2 | 3.552 | 3.552 | 4.033 |
| r0b_sector | 3.537 | 3.538 | 4.094 |
| r0c_winner | 3.471 | 3.465 | 4.076 |

- **Confirmation:** the CVs reproduce within 0.006 on fresh seeds, and the ranking is unchanged.
- **Check set:** every model scores well, with no sign of overfitting. The differences are inside its noise.
- **Stress:** dropping 10% of test headlines costs 0.23 for both the primary and R0b (3.569 → 3.336, 3.538 → 3.311).
  Half-split stability: the primary wins both ways (3.342 / 3.739 vs 3.285 / 3.658).
- **New-draw rehearsal:** on synthetic data with 5 industries and vol-scaled pushes, the runner exits 0, prints the
  generator facts (push duration 10, 5 industries) and gives a quick CV of 5.20 (first written here as 4.78, which
  did not reproduce; the 25 Nov review measured 5.202 three times, also on the commit that wrote 4.78).

## Frozen upload candidates (`src/arena/pipeline.py`, 25 Nov)

| # | Label | What | dev CV |
|---|---|---|---|
| 01 | primary | generator fit + augmented calibration + 2-s.d. clip | **3.568** |
| 02 | clip2 | primary without augmentation | 3.552 |
| 03 | aug | primary without clipping | 3.554 |
| 04 | sector | Edinburgh-style sector model (R0b) | 3.537 |
| 05 | winner | 7 Nov winner-style model (R0c), a different family | 3.471 |
| 06 | blend_tfidf | primary + owed-push TF-IDF, equal weights | 3.532 |
| 07 | volinv | primary + 1/vol calibration feature (added after the review, below) | 3.583 |
| 08 | baseline | R0a, our own quickstart model: the compliance fallback | 3.400 |

Each comes as `_hrtonly` and `_withextra`. The full runner takes 19 s on the practice data (primary 1.3 s).

For reference: the best public scores on this data are LB 3.582 (Edinburgh, CV 3.53–3.55) and LB ≈ 3.6 (7 Nov winner,
CV 3.50). Our primary's CV is 3.568.

## 25 Nov evening: review fixes and the fresh-target gate

Four independent code reviews of the research code and the day-of runner, plus follow-up experiments.

**Runner, checked on the practice data and on rehearsal draws:**
- The 12 original upload files are byte-identical before and after the fixes. The fixes change no number on SAME DATA.
- Horizon read off the files. On a holdout of the in-sample sessions with halfway 59 / exit 69 and default flags, the
  runner prints "exit at close of bar 69". The primary then scores 3.487 on the hidden 3,000 sessions (sector 3.527,
  baseline 3.386). The old runner with default flags scored 0.97 there.
- Industries come from the HRT sessions only, and extra sessions are mapped onto them. Before the fix, extra sessions
  from another draw broke the groups into 80/32/16/16/16 and changed the `_hrtonly` files. Now the groups are the same
  with or without them (test).
- On a draw with no clear industries, the eigengap estimate (gap 0.48 on the known data, 0.01 there) falls back to 5 and
  warns. The NEW DATA quick CV with and without the filter on the 59/69 holdout: 3.609 vs 3.251.
- A re-saved copy of the known files (other compression) is now recognised as SAME DATA ("same content").

**Second review of the fixes** (claims re-checked independently):
- All 12 original files are byte-identical to the pre-fix version's.
- Horizon checks hold, and `_hrtonly` files do not depend on extra sessions, also with colliding ids.
- No leakage was introduced.
- Fixed after it:
  - a corrupt extra folder only costs the `_withextra` files (before: no file at all);
  - stale upload files from other data or code move to `submissions/old/`;
  - with no clear industries the filter is off. On its no-industry draw the primary's quick CV goes from 3.775 (5
    arbitrary groups) to 6.189; on its 5-industry draw the filter stays on (5.348 vs 4.670 without).

**Gate.** About 30 trials were selected on the same dev targets, so the keep bar is now 2.5 s.e. (was 1.5).
Every new rung is also checked on fresh targets with `arena.confirm.later_window_check` (`research/confirm_later.py`).
That check scores fold models on the held-out sessions' later windows (offsets 10–40), which nothing was selected on.

| Idea vs base | dev Δ (z) | later windows Δ (z) | verdict |
|---|---|---|---|
| augmentation (r3_aug_every4 vs r0b_sector) | +0.017 | +0.016 (2.4) | real (review measurement) |
| 2-s.d. clip | +0.015 | +0.004 | not confirmed, harmless (review measurement) |
| 1/vol feature (volinv vs primary) | +0.0155 (2.25) | +0.0024 (0.64), seeds +0.0028 / +0.0020 | not confirmed: variant `07`, not the primary |

The 1/vol effect is specific to bars 49 → 59. There, the lowest-vol fifth of sessions drifts 6.7 bp less than the rest
(z −3.5). In later windows the pattern is flat, so the later-window check cannot confirm it by construction. In the
1,000 datathon sessions the sign is the same (−10.8 bp, z −1.8). It is probably real but small: +0.015 is invisible on
the leaderboard (s.e. ≈ 0.16).

**Late augmentation windows are fine (`x_aug_early.py`).** The review warned that windows at offset ≥ 30 have less
news. Augmenting with offsets 0–28 only: dev 3.565, Δ −0.003 (z −1.2) vs the primary; later windows Δ −0.0011
(z −0.76). The primary keeps offsets 0–40.

**Closed, with measurements (review):** up/down target (−0.04 to −0.08), tanh squash (+0.000 at best), predicting
future headlines, kurtosis sizing, OHLC/intrabar features and other vol estimators all gain nothing. The noise is
Gaussian, and simultaneous pushes add linearly.

**Attribution.** The primary family re-implements the approach published by the Edinburgh team (24 Nov) with their
settings. `08_baseline` (R0a, 3.400) is our own model, kept as a fallback in case reusing public work was not allowed.

## R7: what is left after the ladder (25 Nov evening, `r7_*.py`)

First the headroom probes (label-free statistics or in-sample description), then only the ideas with room left.

| Probe | Result | Verdict |
|---|---|---|
| A market path shared by all sessions? | Per-bar cross-sectional means show a steady positive drift (+0.15 to +0.25 vol units per 10-bar window, in every window), not a random common path. In-sample and out-of-sample first halves correlate 0.16; session betas do not persist (0.01) | nothing to forecast |
| The 1/vol effect | Strongest exactly in the target window (low-vol fifth −6.7 bp in-sample, −10.8 bp in the datathon sessions), fading and flipping sign in later windows | structural to bars 49→59: the later-window check cannot confirm it by construction. Keep `07_volinv` as a variant |
| Main-industry misassignment | 0.48% of sessions (the ambiguous 1.2%) | no headroom |
| Industry-specific drift | mean y/vol by industry, \|t\| ≤ 1.3 for all five | none |
| One kernel for every template? | 254 frequent templates: lags 1–9 flat (1.02 / 1.00), lag 10 0.46, lags 11–13 0.03; spread at noise level | one kernel fits all |
| News intensity | density falls from 0.20 to 0.105 headlines per bar over the session (0.187 in bars 50–59); counts are Poisson-like; past main-industry counts do not predict the target window's (corr −0.015) | no sizing signal |

Pre-declared variants, each one change to the primary (`r7_target_window.py`, `r7_gbm_calibrator.py`):

| Variant | dev CV | Δ vs primary (z) |
|---|---|---|
| sizing variance from the target window only | 3.568 | −0.000 (−0.3) |
| ridge weight 3 on the target window's rows | 3.568 | +0.000 (0.1) |
| m² weight 0.75 / 1.5 | 3.567 / 3.566 | −0.001 (−0.4) / −0.002 (−0.4) |
| calibration alpha 10 / 200 | 3.567 / 3.566 | −0.001 (−0.3) / −0.001 (−0.6) |
| half-lives 50 and 100 added | 3.568 | +0.000 (0.4) |
| gradient-boosted calibrator on the structural features (+ vol, 1/vol, count) | 3.519 | −0.049 (−1.7) |

**Conclusion.** The primary sits on a flat optimum: no single change moves it by more than 0.002 except the GBM, which
loses. The measured ceilings agree. Perfect push rates give 3.574, the kernel, industries and templates are captured,
and the rest of the signal is in headlines that arrive after bar 49, which cannot be predicted. What is left is at most
about +0.01–0.02 (e.g. `07_volinv`), far below the leaderboard's noise (1 s.e. ≈ 0.16). More trials on the same dev
targets would mostly add false discoveries.
