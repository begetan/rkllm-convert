#!/usr/bin/env python3
"""
export_rkllm.py — Convert Qwen3-VL LLM half -> .rkllm (W8A8, RK3588)

Uses rkllm-toolkit 1.2.3. The vision encoder is stripped automatically
by the toolkit — this is expected and correct.

Conversion time estimates (32-core x86 PC):
    4B  ->  ~30-60 min
    8B  ->  ~2-4 h
    32B ->  ~16-24 h  (requires >=80 GB RAM)
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from rkllm.api import RKLLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True, help="HF model ID or local path")
    p.add_argument("--calib",       required=True, help="data_quant.json calibration file")
    p.add_argument("--output",      required=True, help="Output .rkllm path")
    p.add_argument("--max_context", type=int, default=4096,
                   help="Max context length (multiple of 32, max 16384). Default 4096.")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")

    if os.path.isdir(args.model):
        model_path = args.model
    else:
        print(f"[rkllm] Resolving {args.model} from HF cache ...")
        model_path = snapshot_download(repo_id=args.model, token=token)

    print(f"[rkllm] Model path:   {model_path}")
    print(f"[rkllm] Calib file:   {args.calib}")
    print(f"[rkllm] Output:       {args.output}")
    print(f"[rkllm] Max context:  {args.max_context}")
    print(f"[rkllm] Target:       rk3588  Quant: w8a8  Cores: 3")

    llm = RKLLM()

    print(f"\n[rkllm] Loading model ...")
    ret = llm.load_huggingface(model=model_path, device="cpu")
    assert ret == 0, f"load_huggingface failed (ret={ret})"

    print(f"[rkllm] Building and quantising W8A8 (slow — may take several hours) ...")
    ret = llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype="w8a8",
        quantized_algorithm="normal",
        target_platform="rk3588",
        num_npu_core=3,
        dataset=args.calib,
        max_context=args.max_context,
    )
    assert ret == 0, f"build failed (ret={ret})"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"[rkllm] Exporting -> {args.output} ...")
    ret = llm.export_rkllm(args.output)
    assert ret == 0, f"export_rkllm failed (ret={ret})"

    size_gb = os.path.getsize(args.output) / 1024**3
    print(f"[rkllm] Done. Output: {args.output}  ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()