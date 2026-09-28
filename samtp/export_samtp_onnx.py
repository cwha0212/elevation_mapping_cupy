"""Export SAM-TP (image -> low-res mask logits) to ONNX for TensorRT.

The CustomPromptEncoder ignores all prompts and returns learned embeddings,
so the whole pipeline collapses to a single image-only graph.
"""
import os
import sys

import torch

REPO = "/ws/GENIE-SAMTP"
sys.path.insert(0, REPO)
os.chdir(REPO)

from sam2.build_sam import build_sam2

CKPT_1024 = ("sam2_logs/configs/sam2.1_training_tiny/"
             "sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt")


class SamTPExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        backbone_out = self.model.forward_image(x)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        s = self.model.image_size
        sizes = [(s // 4, s // 4), (s // 8, s // 8), (s // 16, s // 16)]
        feats = [
            f.permute(1, 2, 0).reshape(1, -1, *sz)
            for f, sz in zip(vision_feats[::-1], sizes[::-1])
        ][::-1]
        sparse, dense = self.model.sam_prompt_encoder(None, None, None)
        low_res_masks, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=feats[-1],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=feats[:-1],
        )
        return low_res_masks  # (1, 1, s//4, s//4)


for tag, cfg, ckpt in [
    ("1024", "sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml", CKPT_1024),
    ("256", "sam2/configs/sam2.1_inference_tiny/sam2.1_custom2_256.yaml", "checkpoint_2_256.pt"),
]:
    model = build_sam2(cfg, ckpt, device="cpu")
    model.eval()
    wrapper = SamTPExport(model).eval()
    s = model.image_size
    x = torch.randn(1, 3, s, s)
    with torch.no_grad():
        ref = wrapper(x)
        torch.onnx.export(
            wrapper, x, f"/ws/samtp_{tag}.onnx",
            opset_version=17, input_names=["image"], output_names=["mask_logits"],
            dynamo=False,
        )
    print(tag, "exported, output shape", tuple(ref.shape), flush=True)
