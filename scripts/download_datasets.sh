#!/usr/bin/env bash
set -euo pipefail

# Download datasets from hardcoded Google Drive links into data/
# Usage:
#   scripts/download_data.sh [OUTPUT_DIR]
# Default OUTPUT_DIR = data

OUT_DIR=${1:-data}
export PATH="$HOME/.local/bin:$PATH"
FOLDER_ID="1x3lrzu0qMUXMAJPxg6gIWV_4h6sGFKn7"
# Optional: set these to direct file IDs or share URLs
# You can export them before running, or edit below.
: "${ELECTRICITY_ID:=}"
: "${ELECTRICITY_URL:=}"
: "${TRAFFIC_ID:=}"
: "${TRAFFIC_URL:=https://drive.google.com/file/d/1MaIJswpE5gFzRQHZHrbZJQ8yvmJlJCCA/view?usp=drive_link}"

mkdir -p "$OUT_DIR"

ensure_gdown() {
  if command -v gdown >/dev/null 2>&1; then
    return 0
  fi
  echo "gdown not found; attempting install via pip..." >&2
  if command -v python3 >/dev/null 2>&1; then
    python3 -m pip install --user --quiet gdown || true
  elif command -v pip3 >/dev/null 2>&1; then
    pip3 install --user --quiet gdown || true
  elif command -v pip >/dev/null 2>&1; then
    pip install --user --quiet gdown || true
  fi
  hash -r 2>/dev/null || true
}

HAS_GDOWN=0
if command -v gdown >/dev/null 2>&1; then
  HAS_GDOWN=1
else
  ensure_gdown
  if command -v gdown >/dev/null 2>&1; then
    HAS_GDOWN=1
  fi
fi

extract_id_and_type() {
  # Echoes: "file <id>" or "folder <id>" if recognized, else returns non-zero
  local url="$1"

  # Folder patterns
  if [[ "$url" =~ drive\.google\.com/drive/folders/([^/?#]+) ]]; then
    echo "folder ${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$url" =~ drive\.google\.com/folderview\?id=([^&#]+) ]]; then
    echo "folder ${BASH_REMATCH[1]}"
    return 0
  fi

  # File patterns
  if [[ "$url" =~ drive\.google\.com/file/d/([^/?#]+) ]]; then
    echo "file ${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$url" =~ drive\.google\.com/open\?id=([^&#]+) ]]; then
    echo "file ${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "$url" =~ drive\.google\.com/uc\?id=([^&#]+) ]]; then
    echo "file ${BASH_REMATCH[1]}"
    return 0
  fi

  return 1
}

download_file_with_curl() {
  local id="$1"
  local outname="${2:-}"
  local cookie
  cookie=$(mktemp)
  local base="https://drive.google.com/uc?export=download&id=${id}"

  # First request to get confirmation token for large files
  local confirm
  confirm=$(curl -fs -sc "$cookie" "$base" | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p') || true
  local url="$base"
  if [ -n "${confirm:-}" ]; then
    url="$base&confirm=$confirm"
  fi

  (
    cd "$OUT_DIR"
    if [ -n "$outname" ]; then
      curl -f -L -b "$cookie" -o "$outname" "$url"
    else
      # -J uses server-provided filename, -O writes to a file, -L follows redirects
      curl -f -L -b "$cookie" -OJ "$url"
    fi
  )

  rm -f "$cookie"
}

download_file_with_wget() {
  local id="$1"
  local outname="${2:-}"
  local cookie
  cookie=$(mktemp)
  local base="https://drive.google.com/uc?export=download&id=${id}"

  # First request to get confirmation token for large files
  local confirm
  confirm=$(wget --quiet --save-cookies "$cookie" --keep-session-cookies --no-check-certificate "$base" -O - | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' | head -n1) || true
  local url="$base"
  if [ -n "${confirm:-}" ]; then
    url="$base&confirm=$confirm"
  fi

  (
    cd "$OUT_DIR"
    if [ -n "$outname" ]; then
      wget --quiet --load-cookies "$cookie" --no-check-certificate -O "$outname" "$url"
    else
      wget --quiet --load-cookies "$cookie" --content-disposition --no-check-certificate "$url"
    fi
  )

  rm -f "$cookie"
}

download_file() {
  local id="$1"
  local outname="${2:-}"
  if command -v wget >/dev/null 2>&1; then
    download_file_with_wget "$id" "$outname"
  else
    download_file_with_curl "$id" "$outname"
  fi
  # If a target name was provided and looks like an HTML page, warn
  if [ -n "$outname" ] && [ -f "$OUT_DIR/$outname" ]; then
    if head -c 512 "$OUT_DIR/$outname" 2>/dev/null | grep -qi "<html"; then
      echo "Warning: '$outname' appears to be HTML (maybe a Drive auth or error page)." >&2
    fi
  fi
}

download_folder_without_gdown() {
  local folder_id="$1"
  echo "gdown not available; attempting folder scrape for $folder_id"
  local tmp_page
  tmp_page=$(mktemp)
  # Fetch the public folder page HTML
  if command -v wget >/dev/null 2>&1; then
    wget --quiet "https://drive.google.com/embeddedfolderview?id=${folder_id}#list" -O "$tmp_page" || true
    if [ ! -s "$tmp_page" ]; then
      wget --quiet "https://drive.google.com/drive/folders/${folder_id}" -O "$tmp_page" || true
    fi
  else
    curl -fsSL "https://drive.google.com/embeddedfolderview?id=${folder_id}#list" -o "$tmp_page" || true
    if [ ! -s "$tmp_page" ]; then
      curl -fsSL "https://drive.google.com/drive/folders/${folder_id}" -o "$tmp_page" || true
    fi
  fi
  if [ ! -s "$tmp_page" ]; then
    echo "Failed to access Google Drive folder: $folder_id" >&2
    rm -f "$tmp_page"
    return 1
  fi

  # Extract file IDs using multiple heuristics (Drive uses dynamic HTML)
  local tmp_ids
  tmp_ids=$(mktemp)
  {
    # Embedded folder view links
    grep -Eo 'uc\?id=[A-Za-z0-9_-]{20,}' "$tmp_page" | sed -E 's/.*uc\?id=([A-Za-z0-9_-]{20,}).*/\1/' || true
    # Direct /file/d/<id> style
    grep -Eo '/file/d/[A-Za-z0-9_-]+' "$tmp_page" | sed -E 's#.*/file/d/([A-Za-z0-9_-]+).*#\1#' || true
    # JSON-like id fields
    grep -Eo '"id":"[A-Za-z0-9_-]{20,}"' "$tmp_page" | sed -E 's/.*"id":"([A-Za-z0-9_-]{20,})".*/\1/' || true
    # data-id attributes
    grep -Eo 'data-id="[A-Za-z0-9_-]{20,}"' "$tmp_page" | sed -E 's/.*data-id="([A-Za-z0-9_-]{20,})".*/\1/' || true
    # Escaped JSON in HTML
    grep -Eo '\\x22id\\x22:\\x22[A-Za-z0-9_-]{20,}\\x22' "$tmp_page" | sed -E 's/.*\\x22id\\x22:\\x22([A-Za-z0-9_-]{20,})\\x22.*/\1/' || true
  } | sort -u > "$tmp_ids"
  rm -f "$tmp_page"

  local count
  count=$(wc -l < "$tmp_ids" | tr -d '[:space:]')
  if [ -z "$count" ] || [ "$count" -eq 0 ]; then
    echo "No files found in the folder or scraping failed." >&2
    rm -f "$tmp_ids"
    return 1
  fi

  echo "Found $count file(s) in folder. Downloading..."
  while IFS= read -r fid; do
    [ -n "$fid" ] || continue
    download_file "$fid" || true
  done < "$tmp_ids"
  rm -f "$tmp_ids"
}

resolve_file_in_folder_by_name() {
  # Try to resolve a specific file ID by filename from a public folder page
  local folder_id="$1"
  local filename="$2"
  local content
  # Fetch folder page HTML (may include JSON with names and ids)
  if command -v wget >/dev/null 2>&1; then
    content=$(wget --quiet -O - "https://drive.google.com/embeddedfolderview?id=${folder_id}#list" || true)
    if [ -z "${content:-}" ]; then
      content=$(wget --quiet -O - "https://drive.google.com/drive/folders/${folder_id}" || true)
    fi
  else
    content=$(curl -fsSL "https://drive.google.com/embeddedfolderview?id=${folder_id}#list" || true)
    if [ -z "${content:-}" ]; then
      content=$(curl -fsSL "https://drive.google.com/drive/folders/${folder_id}" || true)
    fi
  fi
  [ -n "${content:-}" ] || return 1

  # Flatten to a single line to allow simple bounded matching
  # Prefer embedded view where the download link and filename are near
  local flat
  flat=$(printf "%s" "$content" | tr '\n' ' ')

  local id
  # Try: ... uc?id=<ID> ... >traffic.csv<
  id=$(printf "%s" "$flat" | grep -Eo 'uc\?id=[A-Za-z0-9_-]{20,}.{0,200}>'"$filename"'<' | head -n1 | sed -E 's/.*uc\?id=([A-Za-z0-9_-]{20,}).*/\1/')
  if [ -z "${id:-}" ]; then
    id=$(printf "%s" "$flat" | grep -Eo '"id":"[A-Za-z0-9_-]{20,}".{0,250}"name":"'"$filename"'"' | head -n1 | sed -E 's/.*"id":"([A-Za-z0-9_-]{20,})".*/\1/')
  fi
  if [ -z "${id:-}" ]; then
    id=$(printf "%s" "$flat" | grep -Eo '"name":"'"$filename"'".{0,250}"id":"[A-Za-z0-9_-]{20,}"' | head -n1 | sed -E 's/.*"id":"([A-Za-z0-9_-]{20,})".*/\1/')
  fi

  if [ -n "${id:-}" ]; then
    echo "Resolved $filename -> $id"
    download_file "$id" "$filename"
    return $?
  fi
  return 1
}

ensure_from_folder() {
  # Ensure a file exists; if missing, try to fetch it from the public folder
  local filename="$1"
  if [ -f "$OUT_DIR/$filename" ]; then
    echo "$filename already present; skipping."
    return 0
  fi

  echo "$filename missing; attempting to download from folder..."
  if [ "$HAS_GDOWN" -eq 1 ]; then
    # Download the whole folder to ensure file presence
    gdown --folder --id "$FOLDER_ID" -O "$OUT_DIR" || true
  else
    # Targeted fetch by resolving the file ID from the folder page
    resolve_file_in_folder_by_name "$FOLDER_ID" "$filename" || true
  fi

  if [ -f "$OUT_DIR/$filename" ]; then
    echo "Downloaded $filename"
    return 0
  else
    echo "Failed to download $filename from the folder." >&2
    return 1
  fi
}

download_url() {
  local url="$1"

  # Determine type and ID
  local type id out
  out=$(extract_id_and_type "$url") || true
  if [ -z "${out:-}" ]; then
    echo "Skipping unsupported URL: $url" >&2
    return 0
  fi
  type=${out%% *}
  id=${out#* }

  if [ "$type" = "folder" ]; then
    if [ "$HAS_GDOWN" -eq 1 ]; then
      gdown --folder --id "$id" -O "$OUT_DIR"
    else
      download_folder_without_gdown "$id" || echo "Skipped folder due to missing gdown and scrape failure: $url" >&2
    fi
  else
    if [ "$HAS_GDOWN" -eq 1 ]; then
      (
        cd "$OUT_DIR"
        gdown --id "$id"
      )
    else
      download_file "$id"
    fi
  fi
}

# Ensure electricity.csv
if [ -f "$OUT_DIR/electricity.csv" ]; then
  echo "electricity.csv already present; skipping."
else
  if [ -n "$ELECTRICITY_URL" ]; then
    echo "Downloading electricity from URL..."
    download_url "$ELECTRICITY_URL" || true
  elif [ -n "$ELECTRICITY_ID" ]; then
    echo "Downloading electricity by ID..."
    download_file "$ELECTRICITY_ID" "electricity.csv" || true
  fi
  ensure_from_folder "electricity.csv" || true
fi

# Ensure traffic.csv
if [ -f "$OUT_DIR/traffic.csv" ]; then
  echo "traffic.csv already present; skipping."
else
  if [ -n "$TRAFFIC_URL" ]; then
    echo "Downloading traffic from URL..."
    out=$(extract_id_and_type "$TRAFFIC_URL") || out=""
    if [ -n "$out" ]; then
      t=${out%% *}; i=${out#* }
      if [ "$t" = "file" ]; then
        download_file "$i" "traffic.csv" || true
      else
        download_url "$TRAFFIC_URL" || true
      fi
    else
      download_url "$TRAFFIC_URL" || true
    fi
  elif [ -n "$TRAFFIC_ID" ]; then
    echo "Downloading traffic by ID..."
    download_file "$TRAFFIC_ID" "traffic.csv" || true
  fi
  ensure_from_folder "traffic.csv" || true
fi

echo "Done. Files are in: $OUT_DIR"
