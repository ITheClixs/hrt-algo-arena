# Real-headline rehearsal dataset (EPFL-format fallback)

A stand-in for the EPFL edition (about 137k real Bloomberg headlines, 2016 holdout), built only from public data.
Use it to dry-run the real-headline fallback (see the README): time-based folds, the per-unit scorer, the upload cell.
It is never committed: `data/` is gitignored, and the text belongs to Bloomberg (research use only).

## Build (about 20 s once the downloads exist)

```bash
# downloads (about 1.1 GB): Bloomberg corpus + FNSPID daily prices
uv run --with huggingface_hub python -c "from huggingface_hub import hf_hub_download as d; \
  d('danidanou/Bloomberg_Financial_News','bloomberg_financial_data.parquet.gzip',repo_type='dataset',local_dir='data/rehearsal/raw'); \
  d('Zihan1004/FNSPID','Stock_price/full_history.zip',repo_type='dataset',local_dir='data/rehearsal/raw')"
unzip -q data/rehearsal/raw/Stock_price/full_history.zip 'full_history/*' -d data/rehearsal/raw
uv run python scripts/build_news_rehearsal.py --raw data/rehearsal/raw --out data/rehearsal/news
```


## Files

| File | Rows | Columns |
|---|---|---|
| `news_train.parquet` | 29,823 (entries up to 2012-12-31, 571 trading days) | `id, time, entry_date, ticker, headline, vol20, ret1, ret5, logdv20, price, ret` |
| `news_test.parquet` | 13,846 (2013-01-02 to 2013-11-26, 227 days) | same, without `ret` |
| `news_test_answers.parquet` | 13,846 | `id, ret`: hidden answers for a mock leaderboard |

- **Ticker:** the first `(TICKER)` tag in the article's first 600 characters (Bloomberg style "Marriott International Inc. (MAR)"), kept only if FNSPID has US prices for it.
- **Timing (no look-ahead):** published before 16:00 → traded at that day's close; otherwise at the next trading day's close.
  Bloomberg timestamps are assumed to be US Eastern time.
- **Target `ret`:** next-day return adjclose(entry+1)/adjclose(entry) − 1, minus SPY's over the same day. Sd 0.0195, about 55 rows per day.
- **Past-only features at the entry close:** `vol20` (20-day return sd), `ret1`, `ret5`, `logdv20` (log 20-day mean dollar volume), `price`.

## Known limitations

- Smaller than EPFL (43.7k vs 137k rows). Coverage is 2011–2013 only, because the 2010 articles carry no ticker tags.
- **Ticker collisions:** some tags are non-US exchange codes that match a different US ticker
  (for example "(SU)" for Schneider Electric maps to Suncor). Those rows have unrelated returns. That is extra label noise,
  so treat absolute Sharpe levels as pessimistic.
- One-day horizon and SPY adjustment are our choices. EPFL's exact target and scoring unit are unknown, so map columns on the day.
