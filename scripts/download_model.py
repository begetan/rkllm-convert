#!/usr/bin/env python3
"""
download_model.py — Download HF model to persistent cache
HF_HOME is set to /cache/huggingface so models survive container restarts.
"""

import argparse
import os
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[download] WARNING: HF_TOKEN not set. Downloads may be rate-limited.")

    print(f"[download] Downloading {args.model} …")
    print(f"[download] Cache: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")

    local_path = snapshot_download(
        repo_id=args.model,
        token=token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*",
                         "rust_model*", "*.ot"],
    )

    print(f"[download] ✓ Downloaded to: {local_path}")


if __name__ == "__main__":
    main()