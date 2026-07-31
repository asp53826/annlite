#!/usr/bin/env bash
# SIFT small: 10k base vectors, 100 queries, and exact ground truth.
# The standard tiny ANN benchmark, about 5MB.
set -euo pipefail

readonly DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
readonly DATASET_DIR="$DATA_DIR/siftsmall"
readonly ARCHIVE_URL="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
readonly ARCHIVE_SHA256="b8f1e59b20319ac44279d5251706909dd3a5b8ca5ce2a11ddb1e73902252770e"
readonly REQUIRED_FILES=(
  siftsmall_base.fvecs
  siftsmall_groundtruth.ivecs
  siftsmall_learn.fvecs
  siftsmall_query.fvecs
)

dataset_complete() {
  local root="$1"
  local file
  for file in "${REQUIRED_FILES[@]}"; do
    [[ -s "$root/$file" ]] || return 1
  done
}

mkdir -p "$DATA_DIR"
if dataset_complete "$DATASET_DIR"; then
  echo "siftsmall already present and complete"
  exit 0
fi
if [[ -e "$DATASET_DIR" ]]; then
  echo "refusing to overwrite incomplete dataset at $DATASET_DIR" >&2
  exit 1
fi

download_dir="$(mktemp -d "$DATA_DIR/.siftsmall-download.XXXXXX")"
cleanup() {
  rm -rf "$download_dir"
}
trap cleanup EXIT

archive="$download_dir/siftsmall.tar.gz"
curl --fail --location --retry 3 --retry-all-errors \
  --output "$archive" "$ARCHIVE_URL"
printf '%s  %s\n' "$ARCHIVE_SHA256" "$archive" | shasum -a 256 --check
tar -xzf "$archive" -C "$download_dir"

if ! dataset_complete "$download_dir/siftsmall"; then
  echo "downloaded archive is missing required SIFT files" >&2
  exit 1
fi

mv "$download_dir/siftsmall" "$DATASET_DIR"
echo "installed verified siftsmall dataset"
