# SAM-TP assets

`samtp_node.py` runs SAM-TP (GeNIE's traversability model: SAM2-tiny with the
prompt encoder replaced by learned embeddings) as a TensorRT engine and
publishes the negated raw logit as the `untrav` channel.

| file | size | where |
|---|---|---|
| `samtp_512.onnx` | 130 MB | GitHub Release `samtp-assets-v1` (above the 100 MB per-file limit) |
| `samtp_512_fp16.engine` | 68 MB | same release; built on the haechi Orin (L4T R36.5.2). Only valid there |
| `SHA256SUMS` | | in git |
| `fetch_assets.sh` | | downloads and verifies into `~/samtp` |
| `build_engine.sh` | | rebuilds the engine from the ONNX with trtexec |
| `export_samtp_onnx.py` | | re-exports the ONNX from the GENIE-SAMTP checkpoint (`checkpoint_2_512.pt`, 403 MB, not distributed here) |

On the board:

    bash samtp/fetch_assets.sh          # onnx + the Orin engine, checksummed
    # on a different GPU or JetPack:
    bash samtp/fetch_assets.sh onnx && bash samtp/build_engine.sh

The launch files look for `~/samtp/samtp_512_fp16.engine` (`samtp_engine:=` overrides).

Re-export (only when the model changes): `export_samtp_onnx.py` expects the
GENIE-SAMTP source tree and checkpoint (`REPO` at the top of the file) inside
the training container; see that repo's REPRODUCE.md.
