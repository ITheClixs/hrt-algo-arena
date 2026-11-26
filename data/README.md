# data/

Gitignored. HRT owns the data: never commit it.
- `raw/`: competition data on the day (`scripts/fetch_gdrive.sh <url>`)
- `practice/`: earlier editions (`scripts/fetch_practice_data.sh`)
- `synthetic/`: offline fake data (`uv run python scripts/make_synthetic_data.py`)
