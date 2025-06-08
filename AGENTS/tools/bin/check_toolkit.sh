#!/usr/bin/env bash
# Verify presence of toolkit binaries
DIR="$(dirname "$0")"
missing=0
for bin in nano busybox bat fzf rg; do
  if [ ! -x "$DIR/$bin" ]; then
    echo "Missing $bin" >&2
    missing=1
  fi
done
if [ $missing -eq 0 ]; then
  echo "Toolkit complete"
fi
