#!/usr/bin/env python3
"""
convert.py — Qwen3-VL → RK3588 conversion orchestrator

Usage (via docker compose):
  docker compose run --rm convert Qwen/Qwen3-VL-4B-Instruct [448|640|896]

Arguments:
  model_id   — HuggingFace model ID, e.g. Qwen/Qwen3-VL-8B-Instruct
  vision_res — Vision encoder input resolution (default: 448)
               Qwen3-VL uses patch_size=16, merge_size=2:
               448 -> 196 tokens  (~1s encode on board)  fastest
               640 -> 400 tokens  (~3s encode on board)  balanced
               896 -> 784 tokens  (~8s encode on board)  best detail/OCR
"""

import sys
import os
import subprocess
import psutil
from pathlib import Path
from huggingface_hub import snapshot_download, repo_info


# Always use the conda env Python — 'python3' on Ubuntu resolves to the
# system Python (/usr/bin/python3) which lacks rknn-toolkit2, rkllm, etc.
PYTHON      = "/opt/conda/envs/rkllm/bin/python"
OUTPUT_DIR  = Path("/output")
CACHE_DIR   = Path("/cache/huggingface/hub")
SCRIPTS_DIR = Path("/scripts")

# Qwen3-VL resolutions: must be multiples of patch_size*merge_size = 32
# 448 -> 196 tokens, 640 -> 400 tokens, 896 -> 784 tokens
VALID_RESOLUTIONS = [448, 640, 896]


def run(cmd: list, check=True, **kwargs):
    print(f"\n[convert] $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def check_memory(model_id: str):
    """Fetch model size from HF and compare against available system RAM."""
    token = os.environ.get("HF_TOKEN")
    try:
        info = repo_info(model_id, token=token, files_metadata=True)
        model_bytes = sum(
            f.size for f in info.siblings
            if f.size and f.rfilename.endswith((".safetensors", ".bin", ".pt"))
        )
    except Exception as e:
        print(f"[convert] Could not fetch model size from HF: {e}")
        return

    available_bytes = psutil.virtual_memory().available
    total_bytes     = psutil.virtual_memory().total

    model_gb     = model_bytes     / 1024**3
    available_gb = available_bytes / 1024**3
    total_gb     = total_bytes     / 1024**3

    # Conversion needs the BF16 weights fully in RAM; add ~20% headroom
    required_bytes = model_bytes * 1.2

    print(f"[convert] Model size (BF16 weights): {model_gb:.1f} GB")
    print(f"[convert] System RAM: {available_gb:.1f} GB available / {total_gb:.1f} GB total")

    if required_bytes > available_bytes:
        required_gb = required_bytes / 1024**3
        print(f"\n[convert] WARNING: Not enough system memory.")
        print(f"[convert]   Required:  {required_gb:.1f} GB  (model + 20% headroom)")
        print(f"[convert]   Available: {available_gb:.1f} GB")
        print(f"[convert]   Proceeding — expect OOM during quantisation.\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model_id    = sys.argv[1]
    vision_res  = int(sys.argv[2]) if len(sys.argv) > 2 else 896
    max_context = int(sys.argv[3]) if len(sys.argv) > 3 else 4096

    if vision_res not in VALID_RESOLUTIONS:
        print(f"[convert] ERROR: vision_res must be one of {VALID_RESOLUTIONS}")
        sys.exit(1)

    short    = model_short_name(model_id)
    out_dir  = OUTPUT_DIR / short
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_out  = out_dir / f"{short}_vision_{vision_res}.onnx"
    rknn_out  = out_dir / f"{short}_vision_{vision_res}_rk3588.rknn"
    calib_out = out_dir / "data_quant.json"
    llm_out   = out_dir / f"{short}_w8a8_rk3588.rkllm"

    print(f"\n[convert] ══════════════════════════════════════════════")
    print(f"[convert] Model:      {model_id}")
    print(f"[convert] Vision res: {vision_res}x{vision_res}")
    print(f"[convert] Output dir: {out_dir}")
    print(f"[convert] ══════════════════════════════════════════════\n")

    check_memory(model_id)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    model_cache = CACHE_DIR / f"models--{model_id.replace('/', '--')}"
    if model_cache.exists():
        print(f"[convert] Step 1/5: Cache found — skipping download.")
    else:
        print(f"[convert] Step 1/5: Downloading {model_id} ...")
        run([PYTHON, str(SCRIPTS_DIR / "download_model.py"), "--model", model_id])

    # ── Step 2: Vision encoder → ONNX ────────────────────────────────────────
    if onnx_out.exists() and onnx_out.stat().st_size > 10 * 1024**2:  # >10 MB = real export
        print(f"[convert] Step 2/5: {onnx_out.name} found — skipping.")
    else:
        print(f"[convert] Step 2/5: Exporting vision encoder -> ONNX ...")
        run([
            PYTHON, str(SCRIPTS_DIR / "export_vision_onnx.py"),
            "--model", model_id, "--output", str(onnx_out), "--res", str(vision_res),
        ])

    # ── Step 3: ONNX → .rknn ─────────────────────────────────────────────────
    if rknn_out.exists():
        print(f"[convert] Step 3/5: {rknn_out.name} found — skipping.")
    else:
        print(f"[convert] Step 3/5: Converting ONNX -> .rknn ...")
        run([
            PYTHON, str(SCRIPTS_DIR / "export_vision_rknn.py"),
            "--onnx", str(onnx_out), "--output", str(rknn_out), "--res", str(vision_res),
        ])

    # ── Step 4: Calibration data ──────────────────────────────────────────────
    if calib_out.exists():
        print(f"[convert] Step 4/5: {calib_out.name} found — skipping.")
    else:
        print(f"[convert] Step 4/5: Generating calibration data ...")
        run([
            PYTHON, str(SCRIPTS_DIR / "generate_calib.py"),
            "--model", model_id, "--output", str(calib_out),
        ])

    # ── Step 5: LLM → .rkllm ─────────────────────────────────────────────────
    if llm_out.exists():
        print(f"[convert] Step 5/5: {llm_out.name} found — skipping.")
    else:
        print(f"[convert] Step 5/5: Converting LLM -> .rkllm (slow — may take several hours) ...")
        run([
            PYTHON, str(SCRIPTS_DIR / "export_rkllm.py"),
            "--model", model_id, "--calib", str(calib_out), "--output", str(llm_out),
            "--max_context", str(max_context),
        ])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[convert] Done.")
    print(f"[convert]   ONNX:   {onnx_out}  ({onnx_out.stat().st_size / 1024**2:.0f} MB)")
    print(f"[convert]   RKNN:   {rknn_out}  ({rknn_out.stat().st_size / 1024**2:.0f} MB)")
    print(f"[convert]   RKLLM:  {llm_out}  ({llm_out.stat().st_size / 1024**3:.1f} GB)")
    print(f"\n[convert] Copy to board:")
    print(f"  scp {str(rknn_out).lstrip('/')} radxa@rock5:/opt/models/rkllm/{short}/{rknn_out.name}")
    print(f"  scp {str(llm_out).lstrip('/')} radxa@rock5:/opt/models/rkllm/{short}/{llm_out.name}")


if __name__ == "__main__":
    main()