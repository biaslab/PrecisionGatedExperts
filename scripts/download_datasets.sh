#!/usr/bin/env bash
set -euo pipefail

# Simple dataset downloader (no Python). Checks for files and downloads if missing.
# Usage: scripts/download_data.sh [OUTPUT_DIR]
# Default OUTPUT_DIR = data

OUT_DIR=${1:-data}
mkdir -p "$OUT_DIR"

download_drive_file_with_curl() {
  local id="$1"
  local dest="$2"
  local cookie base confirm url
  cookie=$(mktemp)
  base="https://drive.google.com/uc?export=download&id=${id}"
  # Get confirmation token for large files
  confirm=$(curl -fs -sc "$cookie" "$base" | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p') || true
  url="$base"
  if [ -n "${confirm:-}" ]; then
    url="$base&confirm=$confirm"
  fi
  echo "Fetching $dest with curl..."
  curl -f -L -b "$cookie" -o "$OUT_DIR/$dest" "$url"
  rm -f "$cookie"
}

download_drive_file_with_wget() {
  local id="$1"
  local dest="$2"
  local cookie base confirm url
  cookie=$(mktemp)
  base="https://drive.google.com/uc?export=download&id=${id}"
  # Get confirmation token for large files
  confirm=$(wget --quiet --save-cookies "$cookie" --keep-session-cookies --no-check-certificate "$base" -O - | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' | head -n1) || true
  url="$base"
  if [ -n "${confirm:-}" ]; then
    url="$base&confirm=$confirm"
  fi
  echo "Fetching $dest with wget..."
  wget --quiet --load-cookies "$cookie" --no-check-certificate -O "$OUT_DIR/$dest" "$url"
  rm -f "$cookie"
}

download_drive_file() {
  local id="$1"
  local dest="$2"
  echo "Downloading $dest ..."
  if command -v wget >/dev/null 2>&1; then
    download_drive_file_with_wget "$id" "$dest"
  else
    download_drive_file_with_curl "$id" "$dest"
  fi
  # Basic sanity check (avoid saving HTML error pages as CSV)
  if head -c 512 "$OUT_DIR/$dest" 2>/dev/null | grep -qi "<html"; then
    echo "Warning: $dest appears to be an HTML page (permission or quota issue)." >&2
  fi
}

# Google Drive file IDs provided by user
ELECTRICITY_ID="1KXzH38LCeAqNrdleZg7Qt4XUaMcGu0hL"
ETTH1_ID="1_nntLW6FcPHVPBPLD3I5buuJ504aRFjv"
ETTH2_ID="1mVqwb5NOQYJlPXUc8C5J8suSJ9SG_zGp"
EXCHANGE_ID="1IQNjOyhQO8DYvSYVuTumTAQkDhgGcVhI"
TRAFFIC_ID="1MaIJswpE5gFzRQHZHrbZJQ8yvmJlJCCA"

# Download each dataset only if missing
if [ -f "$OUT_DIR/electricity.csv" ]; then
  echo "electricity.csv already present; skipping."
else
  download_drive_file "$ELECTRICITY_ID" "electricity.csv"
fi

if [ -f "$OUT_DIR/ETTh1.csv" ]; then
  echo "ETTh1.csv already present; skipping."
else
  download_drive_file "$ETTH1_ID" "ETTh1.csv"
fi

if [ -f "$OUT_DIR/ETTh2.csv" ]; then
  echo "ETTh2.csv already present; skipping."
else
  download_drive_file "$ETTH2_ID" "ETTh2.csv"
fi

if [ -f "$OUT_DIR/exchange_rate.csv" ]; then
  echo "exchange_rate.csv already present; skipping."
else
  download_drive_file "$EXCHANGE_ID" "exchange_rate.csv"
fi

if [ -f "$OUT_DIR/traffic.csv" ]; then
  echo "traffic.csv already present; skipping."
else
  download_drive_file "$TRAFFIC_ID" "traffic.csv"
fi

echo "Done. Files are in: $OUT_DIR"
