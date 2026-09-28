#!/bin/bash
# Build the TensorRT engine for THIS machine from the ONNX. Engines are tied
# to the GPU and the TensorRT version; the one shipped in the release was
# built on the haechi Orin (AGX Orin 32 GB, L4T R36.5.2). Rebuild when the
# board or JetPack changes, or when the ONNX is re-exported.
#
#   bash samtp/build_engine.sh [onnx] [engine]
set -euo pipefail
ONNX=${1:-$HOME/samtp/samtp_512.onnx}
ENGINE=${2:-$HOME/samtp/samtp_512_fp16.engine}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
[ -f "$ONNX" ] || { echo "missing $ONNX (run samtp/fetch_assets.sh onnx)"; exit 1; }
[ -x "$TRTEXEC" ] || { echo "trtexec not found at $TRTEXEC"; exit 1; }
"$TRTEXEC" --onnx="$ONNX" --fp16 --saveEngine="$ENGINE" --memPoolSize=workspace:2048
ls -la "$ENGINE"
