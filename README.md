# HRT Algo Arena: news-driven return prediction on synthetic markets

Our team's research code and competition pipeline for the **VIS × Hudson River Trading Algo Arena** (ETH Zurich,
26 Nov 2026), a 60-minute team sprint scored on a live leaderboard.

- **Model:** a structural model of the data generator. Each headline template pushes the price for a fixed number of
  bars. We fit the push rates jointly, forecast the push still owed after the trading bar, and size positions with a
  Sharpe-optimal rule.
- **Validation:** a strict harness: grouped, repeated cross-validation, paired bootstrap tests, a pre-registered keep
  gate and a confirmation on fresh targets.
- **Result:** cross-validated Sharpe **3.568** (best public model on this data: 3.53–3.55). Final leaderboard Sharpe
  **3.661**, 11th of 33 teams at our last check (1st: 3.756).

---

## Contents

1. [The task](#1-the-task)
2. [Results](#2-results)
3. [Quick start](#3-quick-start)
4. [What the data generator does](#4-what-the-data-generator-does)
5. [The model](#5-the-model)
6. [Validation](#6-validation)
7. [Research log, R0 to R7](#7-research-log-r0-to-r7)
8. [How much signal is left](#8-how-much-signal-is-left)
9. [The day-of pipeline](#9-the-day-of-pipeline)
10. [Fallback for real headlines](#10-fallback-for-real-headlines)
11. [Competition day](#11-competition-day)
12. [Lessons](#12-lessons)
13. [Repository layout](#13-repository-layout)
14. [Credits](#14-credits)

---

## 1. The task

- **Sessions:** each session is one synthetic stock with 100 bars of OHLC prices, starting at 1.0, plus fictional
  headlines attached to bars.
- **In-sample:** 10,000 sessions with all 100 bars and their target returns.
- **Out-of-sample:** 10,000 sessions with bars 0–49 and their headlines only. The upload is scored on these.
- **Trade:** take a position `pred` at the close of bar 49 and hold it to the close of bar 59.
- **Target:** the market-adjusted return, i.e. the simple return close(49) → close(59) minus its mean across sessions.
- **Score:** each session counts as one day, with `pnl = pred · ret` and
  `Sharpe = √252 · mean(pnl) / std(pnl)` (sample std).
- **Uploads:** 10, as a CSV `headline_id,pred`. The leaderboard ranks each team's best upload.

Three properties shape everything below:
- **Scale-free:** multiplying all positions by a constant changes nothing.
- **Constant positions earn nothing,** because the target is demeaned.
- **Noise:** one score on 10,000 sessions has a standard error of about √(252/n) ≈ 0.16, far larger than most model
  improvements.

## 2. Results

| Model | Dev CV (5-fold × 3 seeds) | Worst fold | Leaderboard |
|---|---|---|---|
| R0a, template drift + main industry ÷ vol (our baseline) | 3.400 | 2.963 | – |
| R0c, re-implementation of the 7 Nov winner's write-up | 3.471 | 3.194 | 3.641 |
| R0b, structural sector model (reference) | 3.537 | 3.225 | – |
| **Primary:** R0b + time-shift augmentation + 2-s.d. clip | **3.568** | **3.263** | 3.558 |
| Primary + 1/vol calibration feature | 3.583 | 3.291 | 3.585 |
| Blend of the two families, weights from leaderboard scores | – | – | **3.661** |

Leaderboard scores are single uploads. The `winner` and `1/vol` uploads were the versions also fitted on the 1,000
extra labelled sessions; organisers allowed outside data.

- **Confirmation:** the primary scores 3.562 on unused seeds (1042–1044).
- **Check set:** on 1,000 labelled sessions from the same generator draw, never used in development, it scores
  4.03 ± 0.51.

## 3. Quick start

```bash
uv sync                                         # Python 3.12, exact versions from uv.lock
scripts/fetch_practice_data.sh                  # public copies of earlier editions' data -> data/practice/
uv run python scripts/arena_run.py --practice   # fingerprint, fit the 8 candidates, write 16 upload files (~20 s)
uv run pytest                                   # 190 tests, coverage gate 80% (currently 98%)
uv run python research/r3_augment.py            # re-run any rung of the research ladder
```

- **Data ownership:** the data belongs to HRT and is not in this repository. `scripts/fetch_practice_data.sh` copies
  it from earlier participants' public repositories.
- **Offline use:** `scripts/make_synthetic_data.py` builds a synthetic stand-in with `arena.synth`, which the tests
  also use.

## 4. What the data generator does

We established these facts from the in-sample data and re-checked them label-free on the visible halves:

| Fact | Evidence |
|---|---|
| **302 headline templates** once the company name and numbers are masked | entity detection + number masking (`arena.text`) |
| **Each template pushes the price at a constant rate** for 10 bars: half a push on the headline's bar, a full push for 9 bars, half on the 11th | the measured kernel, label-free: 0.55, 0.99, 1.01, 0.95, 1.02, 0.97, 0.98, 1.00, 0.94, 1.02, 0.57 |
| **The push scales with the stock's volatility,** and its size does not vary per headline instance | early and late residuals of one push correlate 0.034 (control: −0.030, n = 2,804) |
| **One kernel fits all templates** | 254 frequent templates: lags 1–9 flat (1.02 / 1.00), lag 10 0.46, lags 11–13 0.03 |
| **80 companies form 5 industries of 16.** Each session is built around one industry, and other industries' headlines barely move the price | co-occurrence clustering; keeping only main-industry headlines lifts the owed-push signal from 2.72 to 3.15 |
| **Volatility is a fixed per-session parameter** | future 10-bar variance correlates 0.87 with the past 50 bars; recent volatility adds nothing |
| **Noise is Gaussian,** and simultaneous pushes add linearly | residual diagnostics |
| **News density falls over the session,** and counts are Poisson-like | 0.20 → 0.105 headlines per bar; past counts do not predict the target window's (corr −0.015) |
| **Numbers inside headlines and company identity carry no signal** | no per-company push multiplier: 1.14–1.20× at most, at noise level |

## 5. The model

Code: `src/arena/generator.py` (`GeneratorModel`), `universe.py`, `structure.py`, `text.py`, `augment.py`, `sizing.py`.

### 5.1 Preprocessing (label-free)

1. **Company detection:** take the first two tokens of a headline, and extend while that prefix is always followed by
   the same next word (up to four tokens). This is done jointly over all headlines, so names are consistent across
   sessions.
2. **Templates:** remove the company and mask the numbers (`$X`, `N%`, `N`), giving 302 templates. A coarser event
   type (50 types) also masks the slot phrases (sector, region, partner, …).
3. **Industries:** build a company × company *lift* matrix from which companies appear in the same session (visible
   bars of in-sample and out-of-sample sessions). Spectral clustering then gives 5 groups of 16.
4. **Main industry:** each session's dominant group among its visible companies. A headline is *main* if its company
   belongs to that group (79% of headlines).

### 5.2 The structural fit

The model for the per-bar, market-adjusted log returns of session *i* (the cross-sectional mean of each bar removed):

```
r[i,t] = vol_i · Σ_h  rate[template(h)] · kernel[t − bar(h)]  +  noise,      kernel = (0.5, 1 × 9, 0.5)
```

- **Push rates:** one rate per template, fitted jointly by ridge least squares (λ = 200). The design rows are every
  (session, bar) pair of the labelled sessions (all 100 bars). The visible bars of the unlabelled sessions (test fold
  and out-of-sample first halves) are added as well: this is label-free and worth +0.03 Sharpe.
- **Volatility:** Garman–Klass over bars 0–49, with a floor.

### 5.3 Features at the decision bar

- **Owed push:** the push that headlines seen by bar 49 still owe in bars 50–59, in units of session volatility
  (`Σ_h rate · kernel weight falling in 50..59`).
- **Residual momentum:** the vol-normalised returns of bars 1–49 *minus the explained pushes*, summed with
  exponential decay at half-lives 1, 2, 3, 5, 10, 25 and 1000 bars. Without these, the score drops by 0.06.

### 5.4 Calibration and time-shift augmentation

- **Ridge calibration** (α = 50) of `ret / vol` on the standardised features. The forecast `m` is the prediction minus
  the intercept.
- **Time-shift augmentation:** an in-sample session has 100 bars, so the same problem (see 50 bars, predict the next
  10) can be posed at decision bars 49, 53, …, 89. Windows at offsets 0, 4, …, 40 multiply the calibration data by 11.
  Targets are demeaned per offset, and a session's windows stay in its own fold. It is confirmed on fresh targets
  (+0.016, z 2.4).

### 5.5 Sizing

For independent bets, maximising pooled Sharpe gives a position proportional to `E[r] / E[r²]`. With a forecast `m`
in vol units:

```
pred = m / (vol · (σ² + m²)),     σ² = Var(ret / vol)
```

- **The m² term is essential:** plain volatility sizing (`m · vol^−p`) loses 0.10–0.20.
- **Weight on m²:** 1 is optimal, as theory predicts (0.5 loses 0.03).
- **Clip:** positions are capped at 2 s.d.

### 5.6 Other candidates

| Candidate | Difference |
|---|---|
| `volinv` | adds 1/vol as a calibration feature. In bars 49→59 the lowest-vol fifth drifts 6.7 bp less than the rest (z −3.5); the same sign shows in the independent sessions (−10.8 bp) |
| `winner` (R0c) | rates from mean per-bar returns at lags 1–9; news counts, peer-company pushes and a refined drift; a risk model of `ret²` from bar statistics, weighted least squares with weights 1/v and sizing `∝ v^−0.5` |
| `blend_tfidf` | primary + a TF-IDF ridge learner on the owed headlines (with augmentation), equal weights |
| `baseline` (R0a) | per-template drift still owed, main-industry headlines only, ÷ vol |

## 6. Validation

Code: `src/arena/validation.py`, `confirm.py`, `cv.py`, `trials.py`; protocol helpers in `research/common.py`.

- **Folds:** 5-fold × seeds 42/43/44, split by session. The score of a seed is the Sharpe of its pooled out-of-fold
  predictions; CV is the mean over seeds. Fold-level spreads are never used, because the 15 folds overlap.
- **No peeking:** a model sees a session only through `truncate(s, 49)`. Everything fitted on labels or on bars after 49
  (rates, scalers, calibrations, augmented windows) is fitted inside the fold. Label-free statistics may use every
  visible bar, including the out-of-sample first halves: templates, industries, and rates measured inside bars 0–49.
  Every model has a test that its predictions do not change when bars after 49 are removed.
- **Paired comparison:** a candidate is compared with the current best by a paired session bootstrap (2,000 resamples)
  of the Sharpe difference.
- **Keep gate:** keep a change only if the gain is at least **2.5 bootstrap s.e.** and positive for every seed. The bar
  was 1.5 until about 30 trials had been run on the same targets.
- **Fresh-target confirmation** (`later_window_check`): fold models are refit and scored on the held-out sessions'
  *later* windows (offsets 10–40, targets in bars 59–99). No idea was ever selected on those targets.
- **Discipline:** hyperparameters come only from short grids declared in advance. Every evaluation is appended to
  `research/trials.csv` with its git commit, and the out-of-fold predictions are saved for paired tests.
- **Untouched sets:** the 1,000 extra labelled sessions are scored once, on the final shortlist. Confirmation seeds
  1042–1044 are used only after freezing.

## 7. Research log, R0 to R7

Full tables are in `research/RESULTS.md`; every run is in `research/trials.csv`.

| Rung | Question | Answer |
|---|---|---|
| R0 | References | Baseline 3.400; 7 Nov winner re-implementation 3.471; structural sector model **3.537** (+0.137, z 3.0) |
| R1 | Generator fit: kernel shape, coarse types, ridge λ, unlabelled halves, momentum, extra features | saturated at about 3.54. The measured kernel confirms 0.5/1×9/0.5; the unlabelled halves (+0.03) and momentum (+0.06) matter |
| R2 | Sizing: risk model for E[r²], vol powers, clipping, m² weight, residual variance | E[r]/E[r²] with weight 1 is optimal. A risk model adds nothing, because variance in vol units is flat |
| R3 | Time-shift augmentation of the calibration | **3.568 with a 2-s.d. clip: kept** |
| R4 | Diverse learners: owed-push TF-IDF (3.405 with augmentation), template × age MLP (2.933) | weaker estimates of the same signal |
| R5 | Blends with R4 | all worse than the primary alone (best 3.532) |
| R6 | Confirmation seeds, check set, stress tests | 3.562 on fresh seeds; 4.03 ± 0.51 on the check set; dropping 10% of headlines costs 0.23 |
| R7 | Headroom probes, then pre-declared variants (target-window variance and weight, m² weight, α, half-lives, a gradient-boosted calibrator) | all within ±0.002, except the GBM (−0.049). The primary sits on a flat optimum |

**Closed with measurements:**
- an up/down classification target and tanh squashing;
- kurtosis-based sizing, OHLC and intrabar features, alternative volatility estimators;
- per-company or per-session push multipliers;
- industry-specific drift (|t| ≤ 1.3);
- a shared market path across sessions (only a steady drift exists);
- predicting future headlines.

## 8. How much signal is left

From `research/headroom.py`; rates fitted on all sessions, so these bounds are optimistic:

| What the predictor knows | Sharpe |
|---|---|
| Owed push, all headlines | 2.72 |
| Owed push, main industry only | 3.15 |
| The same, with perfect push rates | 3.57 (with momentum and sizing) |
| Plus the headlines of bars 50–58, which cannot be known | 4.18 |

- **Our primary (3.568) is at the ceiling** of what the visible data allows.
- **The rest of the signal is unobservable:** it sits in headlines that arrive after the decision bar, and those
  cannot be predicted from past news.

## 9. The day-of pipeline

`uv run python scripts/arena_run.py` (`src/arena/runner.py`) turns the downloaded files into validated upload files in
about 20 s, written defensively for a 60-minute event:

- **Data location:** it reads `data/raw/` (subfolders are searched), `--data` or `$ARENA_DATA`, and prints the folder.
  Practice data needs an explicit `--practice`.
- **Fingerprint:** SHA-256 of each file, plus a canonical content hash (sorted rows, normalised dtypes), so re-saved
  copies are still recognised. It reports **SAME DATA** (every stored CV applies) or **NEW DATA** (refit everything),
  and whether HRT's scorer file changed.
- **Horizon detection:** the halfway bar is the last out-of-sample bar. The exit bar is the one whose
  close-to-close return reproduces the given in-sample returns (correlation ≥ 0.9999), and the demeaning is detected
  too. A contradicting flag stops the run before anything is written.
  - On a rehearsal draw with a 59 → 69 horizon, default flags still give 3.49 on hidden sessions; a fixed horizon would
    have scored 0.97.
- **Industries on a new draw:** the industry count comes from the eigengap of the normalised lift matrix (5, gap 0.48
  on the known data).
  - With no clear structure (gap < 0.2), the industry filter is switched off: 6.19 vs 3.78 on a draw without industries.
  - The quick CV of the other setting is printed next to it.
- **Extra labelled sessions:** they are mapped onto the industries learnt from the competition sessions, so files that
  do not use them stay identical bit for bit. Colliding session ids are renumbered.
- **Failure isolation:**
  - every file is written before any cross-validation runs;
  - a failing candidate or CV is reported while the rest continues;
  - non-finite predictions are refused;
  - upload files left from runs on other data or code are moved aside.
- **Tests:** 190 tests with 98% coverage, and CI on every push.

## 10. Fallback for real headlines

In case the event switched to the EPFL format (about 137k real Bloomberg headlines), `notebooks/01_text_models.ipynb`
has a ready pipeline:
- walk-forward folds by date, with a gap, and a per-day scorer;
- TF-IDF + Ridge, tuned with Optuna on the mean − std of fold Sharpes;
- cached `gte-modernbert-base` headline embeddings + Ridge;
- volatility sizing with a 3-s.d. clip.

The embedding model was chosen in a bake-off. Each model's vectors went into a logistic-regression probe on 2,388
held-out finance headlines (macro-F1):

| Model | Macro-F1 | 137k headlines on a laptop GPU |
|---|---|---|
| `Alibaba-NLP/gte-modernbert-base` | **0.807** | 6–8 min |
| `BatuhanECB/FinModernBERT-embed-large-v1` | 0.803 | 18 min |
| `BAAI/bge-large-en-v1.5` | 0.776 | 14 min |
| `BAAI/bge-small-en-v1.5` | 0.760 | 1.2 min |
| TF-IDF, word 1–2-grams | 0.758 | seconds |
| `ProsusAI/finbert` (mean-pooled) | 0.757 | 4.5 min |

`scripts/build_news_rehearsal.py` builds a public stand-in dataset to rehearse this path (see
`docs/REAL_DATA_REHEARSAL.md`).

## 11. Competition day

- **The data:** the competition files turned out byte-identical to the dataset we prepared on, so the runner wrote all
  16 candidate files in 20 s. The out-of-sample set is also the private test of the ETH Datathon 2026 (first halves
  identical).
- **First upload:** the primary scored **3.558** against a CV of 3.568. The validation was calibrated.
- **Best upload counts:** the leaderboard ranks each team's best upload and shows the full-set score. So after the
  first file, the goal becomes maximising the best of the remaining draws.
- **Correlated candidates:** our candidates' positions correlate 0.95–0.998, so near-copies can only move the score
  by ±0.01–0.03. The most different family (`winner`, lowest CV) scored highest on this set: **3.641**.
- **Leaderboard-informed blend:** two files with known scores `s` and position correlation `C` give a predicted blend
  Sharpe of `√(sᵀC⁻¹s)` at weights `∝ C⁻¹s`. The predicted optimum was 67/33 at 3.657–3.673; it scored **3.661**.
- **A teammate's re-sizing:** a teammate's sizing adjustment, using auxiliary information in the public datathon
  files (cleared with the organisers), added +0.005.

| Upload | Score |
|---|---|
| primary (HRT data only / + extra sessions) | 3.558 / 3.559 |
| primary + 1/vol | 3.585 |
| winner-style family | 3.641 |
| same, re-sized by a teammate | 3.646 |
| **blend 0.67 / 0.33 of the two families** | **3.661** |

## 12. Lessons

1. **With "best upload counts", diversity beats CV rank.** Spend the first uploads on the most *different* model
   families, measured by the correlation of their positions, not on the top three by CV.
2. **Plan leaderboard-blend slots from the start.** Every distinct scored file improves the next blend; we started
   blending at upload 6.
3. **Keep weaker but different models alive in research.** On one 10,000-session test set, a family with 0.1 less CV
   scored 0.06 *more*.
4. **The validation harness paid off:** CV predicted the leaderboard within 0.01, and the blend landed inside its
   predicted range.
5. **Test the event tooling the day before:** browser access, login links and the data download.

## 13. Repository layout

```
src/arena/          library
  data.py            immutable Sessions arrays, loaders, truncate/subset (no-peek views)
  text.py            company detection, templates, coarse types, keyword scores
  structure.py       co-occurrence lift, spectral industries, eigengap estimate, main-industry flags
  universe.py        one label-free preprocessing pass shared by all models
  generator.py       structural model: joint push-rate fit, features, calibration, sizing
  augment.py         time-shift windows
  references.py      R0a baseline and the R0c re-implementation
  learners.py        TF-IDF and template x age learners      blend.py   fixed-weight blends
  validation.py      evaluate / compare (paired bootstrap)   confirm.py fresh-target check
  cv.py              k-fold, walk-forward, walk-forward by time      metrics.py  HRT-exact Sharpe
  fingerprint.py     byte + content hashes                   horizon.py  halfway/exit detection
  pipeline.py        the frozen candidates                   runner.py   day-of command
  signals.py         signal exchange and blending            submission.py  writing, validation, ledger
research/           one script per rung (r0 ... r7), RESULTS.md, trials.csv
scripts/            arena_run.py, data fetchers, environment check, rehearsal-data builder
notebooks/          00 quickstart, 01 real-headline fallback, 03 timed practice drill
tests/              190 tests on synthetic data
```

## 14. Credits

- **The challenge and its data:** Hudson River Trading and VIS (ETH Zurich). The data is not redistributed here.
- **Prior public work:**
  - R0b follows the approach published by the Edinburgh team for the 24 Nov edition
    ([wryzxec/hrt-algo-arena](https://github.com/wryzxec/hrt-algo-arena)): joint template fit, residual momentum and
    E[r]/E[r²] sizing.
  - R0c re-implements the write-up of the 7 Nov winning entry
    ([suspiciously-haunted/algo-arena](https://github.com/suspiciously-haunted/algo-arena)).
  - Both were re-implemented on our own interfaces and validated with our own harness.
- **Team:** AMK TRADING.

MIT licensed (see `LICENSE`). The license covers the code only, not HRT's data.
