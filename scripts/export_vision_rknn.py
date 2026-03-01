#!/usr/bin/env python3
"""
export_vision_rknn.py — Convert vision encoder ONNX -> .rknn (fp16)

Uses rknn-toolkit2. The vision encoder stays fp16 (no quantisation);
W8A8 quantisation is applied to the LLM half only (export_rkllm.py).
"""

import argparse
import os
from pathlib import Path
from rknn.api import RKNN

# Must match VALID_RES in export_vision_onnx.py
VALID_RES = [448, 640, 896]


def convert(onnx_path: str, output_path: str, res: int):
    print(f"[vision_rknn] Input:  {onnx_path}")
    print(f"[vision_rknn] Output: {output_path}")
    print(f"[vision_rknn] Res:    {res}x{res}")

    rknn = RKNN(verbose=False)

    # Input is [1, 3, H, W] uint8 RGB NCHW.
    # rknn applies normalisation at NPU boundary: (x/255 - 0.5) / 0.5
    # which equals mean=127.5, std=127.5 on [0,255] scale.
    rknn.config(
        target_platform="rk3588",
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[127.5, 127.5, 127.5]],
    )

    ret = rknn.load_onnx(
        model=onnx_path,
        input_size_list=[[1, 3, res, res]],
    )
    assert ret == 0, f"load_onnx failed (ret={ret})"

    # No quantisation — vision encoder runs fp16 on NPU
    ret = rknn.build(do_quantization=False)
    assert ret == 0, f"build failed (ret={ret})"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(output_path)
    assert ret == 0, f"export_rknn failed (ret={ret})"

    rknn.release()

    # rknn-toolkit2 leaves intermediate weight files in the output directory.
    # Patterns cover all model sizes:
    # - 2B/4B dump files without prefix (blocks.*, patch_embed*, etc.)
    # - 8B+ dump files with vision_model.* prefix
    import glob
    import shutil
    out_dir = Path(output_path).parent
    for pattern in [
        "vision_model.*", "_vision_model_*",   # 8B+
        "blocks.*", "deepstack_merger*", "merger.*",
        "patch_embed*", "pos_embed*",
        "_Constant*", "Constant_*",             # 2B/4B
    ]:
        for f in glob.glob(str(out_dir / pattern)):
            try:
                if os.path.isfile(f):
                    os.remove(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)
            except OSError:
                pass

    size_mb = os.path.getsize(output_path) // 1024**2
    print(f"[vision_rknn] Done -> {output_path}  ({size_mb} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx",   required=True, help="Input .onnx path")
    p.add_argument("--output", required=True, help="Output .rknn path")
    p.add_argument("--res",    type=int, default=448, choices=VALID_RES)
    args = p.parse_args()

    convert(args.onnx, args.output, args.res)


if __name__ == "__main__":
    main()