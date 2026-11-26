#!/usr/bin/env bash
# Practice data from earlier HRT challenges that participants published on GitHub.
#   data/practice/arcade   -> Algo Arena (arcade) format: ins_*/oos_* parquet + HRT's
#                             sessions.py and algoarena.ipynb starter notebook
#   data/practice/datathon -> ETH Datathon 2026 format: {bars,headlines}_{seen,unseen}_*.parquet
# The data belongs to HRT: use it for practice only and never commit it (data/ is gitignored).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fetch() {  # repo  subdir-in-repo  dest  file-glob...
  local repo="$1" sub="$2" dest="$3"
  shift 3
  echo "==> $repo -> $dest"
  git clone -q --depth 1 "https://github.com/$repo.git" "$TMP/${repo//\//_}"
  mkdir -p "$dest"
  for pattern in "$@"; do
    cp "$TMP/${repo//\//_}/$sub"/$pattern "$dest"/
  done
}

fetch suspiciously-haunted/algo-arena . data/practice/arcade "*.parquet" sessions.py algoarena.ipynb
fetch wstaempfli/datathon-2026 data data/practice/datathon "*.parquet"

uv run python -c "
from arena.data import list_data_files
for d in ('data/practice/arcade', 'data/practice/datathon'):
    print(d); print(list_data_files(d).to_string(index=False)); print()
"
