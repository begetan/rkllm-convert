#!/usr/bin/env python3
"""
export_vision_onnx.py — Export Qwen3-VL vision encoder to ONNX

Exports the full vision pipeline as a single ONNX model that accepts a raw
[1, 3, H, W] float32 RGB image (NCHW, range [0,255]), matching the format
passed by RunImgEnc() in the RK35llm C++ demo (after BGR->RGB conversion).

rknn-toolkit2 applies normalisation at the NPU boundary via mean/std config:
  normalised = (pixel - 127.5) / 127.5  =>  range [-1, 1]

The wrapper then:
  1. Reshapes the normalised image into patch tokens for patch_embed (Conv3d)
  2. Constructs grid_thw
  3. Runs the vision encoder forward pass
  4. Returns 3 deepstack features + pooler_output = 4 tensors total,
     matching the reference Rockchip model output layout

Output tensors (4x): [num_merged_patches, embed_dim]
  where num_merged_patches = (H/16 * W/16) / spatial_merge_size^2
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
from PIL import Image
import numpy as np

VALID_RES = {448: 196, 640: 400, 896: 784}


class Qwen3VLVisionWrapper(nn.Module):
    """
    Wraps the Qwen3-VL vision encoder to accept a normalised
    [1, 3, H, W] float32 image and return 4 output tensors matching
    the reference Rockchip RKNN model layout.

    Input has already been normalised by rknn mean/std config to [-1, 1].
    Patch extraction replicates Qwen2VLImageProcessorFast exactly:
      view(B, grid_t, temporal, C, gh//ms, ms, P, gw//ms, ms, P)
      .permute(0,1,4,7,5,8,3,2,6,9)
      .reshape(B, grid_t*gh*gw, C*temporal*P*P)
    """
    def __init__(self, vision_model, res: int, patch_size: int,
                 temporal: int, merge_size: int):
        super().__init__()
        self.vision_model = vision_model
        self.res        = res
        self.patch_size = patch_size
        self.temporal   = temporal
        self.merge_size = merge_size
        self.gh         = res // patch_size   # grid_h (e.g. 56 for 896)
        self.gw         = res // patch_size   # grid_w

    def forward(self, pixel: torch.Tensor) -> tuple:
        # pixel: [1, 3, H, W] float32, normalised to [-1,1] by rknn mean/std
        B, C, H, W = pixel.shape

        # Temporal duplication: images have 1 frame, need temporal_patch_size=2
        # patches: [B, 1, C, H, W] -> [B, temporal, C, H, W]
        x = pixel.unsqueeze(1).expand(-1, self.temporal, -1, -1, -1)

        # Replicate Qwen2VLImageProcessorFast patch extraction exactly:
        # view into 10 dims then permute(0,1,4,7,5,8,3,2,6,9) then flatten
        ms = self.merge_size
        P  = self.patch_size
        gh = self.gh
        gw = self.gw
        grid_t = 1  # 1 temporal frame after duplication / temporal

        x = x.reshape(
            B,
            grid_t,       # grid_t = 1
            self.temporal, # temporal_patch_size = 2
            C,
            gh // ms,     # merge groups along H
            ms,           # merge_size
            P,            # patch_h
            gw // ms,     # merge groups along W
            ms,           # merge_size
            P,            # patch_w
        )
        # (batch, grid_t, grid_h//ms, grid_w//ms, ms, ms, C, temporal, P, P)
        x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        x = x.reshape(B, grid_t * gh * gw, C * self.temporal * P * P)
        # Remove batch dim: [N, 1536]
        x = x.squeeze(0)

        grid_thw = torch.tensor([[grid_t, gh, gw]], dtype=torch.long)

        out = self.vision_model(x, grid_thw)

        # Output order must match reference Rockchip model exactly:
        # [pooler_output, deepstack_0, deepstack_1, deepstack_2]
        return (out.pooler_output,) + tuple(out.deepstack_features)


def get_local_model_path(model_id: str) -> str:
    token = os.environ.get("HF_TOKEN")
    return snapshot_download(repo_id=model_id, token=token)


def load_vision_only(model_path: str) -> tuple:
    """
    Load only the vision encoder weights without loading the full LLM.
    Instantiates only Qwen3VLVisionModel (not the full 32B model),
    then loads matching weights from the sharded checkpoint.
    """
    import json
    import gc
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = AutoConfig.from_pretrained(model_path)

    # Instantiate ONLY the vision encoder on CPU — ~1-2 GB, not 64 GB
    # attn_implementation="eager" is required for ONNX tracing —
    # SDPA (scaled_dot_product_attention) cannot be exported with opset<18
    print(f"[vision_onnx] Instantiating Qwen3VLVisionModel on CPU ...")
    vision_config = config.vision_config
    vision_config._attn_implementation = "eager"
    vision_model = Qwen3VLVisionModel(vision_config)
    vision_model = vision_model.to(torch.float32)

    prefix = "model.visual."

    # Find which safetensor shards contain visual weights
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.exists():
        shard_files = [Path(model_path) / "model.safetensors"]
    else:
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_files = sorted({
            Path(model_path) / shard
            for key, shard in weight_map.items()
            if key.startswith(prefix)
        })

    print(f"[vision_onnx] Loading visual weights from {len(shard_files)} shard(s) ...")

    state_dict = {}
    for shard in shard_files:
        shard_weights = load_file(str(shard), device="cpu")
        for key, tensor in shard_weights.items():
            if key.startswith(prefix):
                local_key = key[len(prefix):]
                state_dict[local_key] = tensor.to(torch.float32)
        del shard_weights
        gc.collect()

    print(f"[vision_onnx] Loaded {len(state_dict)} visual tensors")
    missing, unexpected = vision_model.load_state_dict(state_dict, strict=True)

    vision_model.eval()
    gc.collect()

    return vision_model, config


def export_vision_encoder(model_path: str, output_path: str, res: int):
    print(f"[vision_onnx] Loading model from {model_path} ...")

    vision_model, config = load_vision_only(model_path)
    cfg         = config.vision_config
    patch_size  = cfg.patch_size           # 16
    temporal    = cfg.temporal_patch_size  # 2
    merge_size  = cfg.spatial_merge_size   # 2
    gh          = res // patch_size
    gw          = res // patch_size

    print(f"[vision_onnx] patch_size={patch_size}  temporal={temporal}  "
          f"merge_size={merge_size}  grid={gh}x{gw}={gh*gw}  res={res}x{res}")

    # Validate against AutoProcessor to catch any patch format mismatch
    processor = AutoProcessor.from_pretrained(model_path)
    dummy_img = Image.fromarray(np.full((res, res, 3), 128, dtype=np.uint8))
    inputs = processor(
        images=dummy_img,
        text="<|vision_start|><|image_pad|><|vision_end|>",
        return_tensors="pt",
        min_pixels=res * res,
        max_pixels=res * res,
    )
    expected_pv_shape = list(inputs["pixel_values"].shape)
    print(f"[vision_onnx] AutoProcessor pixel_values shape: {expected_pv_shape}")
    expected_patches = expected_pv_shape[0]
    assert expected_patches == gh * gw, \
        f"Patch count mismatch: wrapper={gh*gw}, processor={expected_patches}"

    wrapper = Qwen3VLVisionWrapper(vision_model, res, patch_size, temporal, merge_size)
    wrapper.eval()

    # Dummy input: [1, 3, H, W] float32 normalised to [-1, 1]
    # (rknn applies mean/std before feeding ONNX, so ONNX sees normalised values)
    dummy_pixel = torch.zeros(1, 3, res, res, dtype=torch.float32)

    print(f"[vision_onnx] Tracing with input [1, 3, {res}, {res}] float32 ...")
    with torch.no_grad():
        test_out = wrapper(dummy_pixel)
        print(f"[vision_onnx] Output tensors: {len(test_out)}")
        for i, o in enumerate(test_out):
            print(f"  [{i}] shape={list(o.shape)}")

        output_names = ["pooler_output"] + [f"deepstack_{i}" for i in range(len(test_out) - 1)]

        torch.onnx.export(
            wrapper,
            dummy_pixel,
            output_path,
            input_names=["pixel"],
            output_names=output_names,
            dynamic_axes=None,
            opset_version=18,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"[vision_onnx] Saved -> {output_path}  ({size_mb:.0f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, help="HF model ID or local path")
    p.add_argument("--output", required=True, help="Output .onnx path")
    p.add_argument("--res",    type=int, default=896, choices=list(VALID_RES.keys()))
    args = p.parse_args()

    if os.path.isdir(args.model):
        model_path = args.model
    else:
        print(f"[vision_onnx] Resolving {args.model} from HF cache ...")
        model_path = get_local_model_path(args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_vision_encoder(model_path, args.output, args.res)


if __name__ == "__main__":
    main()