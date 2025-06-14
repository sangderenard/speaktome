#!/usr/bin/env bash
# Build a small wheelhouse based on requirements.txt
set -euo pipefail
WHEELHOUSE_DIR=${1:-"wheelhouse"}
REQ_FILE=${2:-"requirements.txt"}
PYTAG="cp312"
LINUX_PLATFORM="manylinux_2_28_x86_64"
WIN_PLATFORM="win_amd64"
mkdir -p "$WHEELHOUSE_DIR/wheels/linux/$PYTAG" "$WHEELHOUSE_DIR/wheels/windows/$PYTAG"
while IFS= read -r pkg; do
  [[ -z "$pkg" || "$pkg" == \#* ]] && continue
  echo "Downloading $pkg"
  pip download "$pkg" \
    --python-version 312 \
    --only-binary=:all: \
    --platform "$WIN_PLATFORM" \
    --dest "$WHEELHOUSE_DIR/wheels/windows/$PYTAG"
  pip download "$pkg" \
    --python-version 312 \
    --only-binary=:all: \
    --platform "$LINUX_PLATFORM" \
    --dest "$WHEELHOUSE_DIR/wheels/linux/$PYTAG"
done < "$REQ_FILE"
find "$WHEELHOUSE_DIR" -name '*.whl' -print0 | sort -z | xargs -0 sha256sum > "$WHEELHOUSE_DIR/SHA256SUMS.txt"
echo "Wheelhouse written to $WHEELHOUSE_DIR"
