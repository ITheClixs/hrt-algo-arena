#!/usr/bin/env bash
# Download the competition data HRT shares through Google Drive (what their Colab notebook
# does with `gdown --folder`).
#   scripts/fetch_gdrive.sh "https://drive.google.com/drive/folders/<id>"  [dest=data/raw]
# If Drive says "Too many users have viewed or downloaded this file recently", ONE person
# downloads it in the browser and AirDrops / USB-shares the folder to everyone else.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

URL="${1:?usage: scripts/fetch_gdrive.sh <google-drive-folder-or-file-url> [dest]}"
DEST="${2:-data/raw}"
mkdir -p "$DEST"

if [[ "$URL" == *"/folders/"* ]]; then
  uv run gdown --folder "$URL" --output "$DEST"
else
  uv run gdown --fuzzy "$URL" --output "$DEST/"
fi

echo
uv run python -c "from arena.data import list_data_files; print(list_data_files('$DEST').to_string(index=False))"
