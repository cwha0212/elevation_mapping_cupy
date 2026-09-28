#!/bin/bash
# Fetch the SAM-TP model assets from this repo's GitHub Release into ~/samtp
# and verify them. The ONNX (130 MB) is above GitHub's per-file limit, so it
# and the Orin engine live as release assets, not in git.
#
#   bash samtp/fetch_assets.sh            # onnx + engine
#   bash samtp/fetch_assets.sh onnx       # onnx only (then build_engine.sh)
#
# Uses gh if logged in, otherwise curl against the public release URL.
set -euo pipefail
REPO=${SAMTP_REPO:-cwha0212/elevation_mapping_cupy}
TAG=${SAMTP_TAG:-samtp-assets-v1}
DEST=${SAMTP_DIR:-$HOME/samtp}
HERE=$(cd "$(dirname "$0")" && pwd)
WANT=${1:-all}
mkdir -p "$DEST"
files="samtp_512.onnx samtp_512_fp16.engine"
[ "$WANT" = "onnx" ] && files="samtp_512.onnx"
[ "$WANT" = "engine" ] && files="samtp_512_fp16.engine"
for f in $files; do
  if [ -f "$DEST/$f" ] && (cd "$DEST" && grep " $f\$" "$HERE/SHA256SUMS" | sha256sum -c --quiet 2>/dev/null); then
    echo "have $f (checksum ok)"; continue
  fi
  echo "fetching $f from $REPO@$TAG"
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$f" --dir "$DEST" --clobber
  else
    curl -fL --progress-bar -o "$DEST/$f" "https://github.com/$REPO/releases/download/$TAG/$f"
  fi
done
cd "$DEST" && for f in $files; do grep " $f\$" "$HERE/SHA256SUMS" | sha256sum -c; done
echo "assets in $DEST"
