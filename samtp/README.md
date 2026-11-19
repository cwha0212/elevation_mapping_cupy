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

On the board:

    bash samtp/fetch_assets.sh          # onnx + the Orin engine, checksummed
    # on a different GPU or JetPack:
    bash samtp/fetch_assets.sh onnx && bash samtp/build_engine.sh

The launch files look for `~/samtp/samtp_512_fp16.engine` (`samtp_engine:=` overrides).

Re-exporting the ONNX (only when the model changes) belongs to the GENIE-SAMTP
training repo (checkpoint `checkpoint_2_512.pt`, see its REPRODUCE.md). The
export script that used to live here was removed in the 2026-11 cleanup; git
history still has it.
