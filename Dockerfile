# syntax=docker/dockerfile:1.6
# ══════════════════════════════════════════════════════════════════════════════
# rkllm-convert — Qwen3-VL → RK3588 conversion image
# rkllm-toolkit 1.2.3  +  rknn-toolkit2 2.3.2  +  transformers>=4.57
#
# Radxa install guide: https://docs.radxa.com/en/rock5/rock5itx/app-development/rkllm_install
# Build:   docker compose build --no-cache
# Run:     docker compose run --rm convert <model_id> [448|640|896]
# ══════════════════════════════════════════════════════════════════════════════
FROM ubuntu:24.04

ARG RKLLM_TAG=release-v1.2.3
ARG MINIFORGE_VERSION=25.11.0-0
ARG DEBIAN_FRONTEND=noninteractive

ENV \
    PATH=/opt/conda/envs/rkllm/bin:/opt/conda/bin:$PATH \
    BUILD_CUDA_EXT=0 \
    HF_HOME=/cache/huggingface \
    HF_DATASETS_CACHE=/cache/huggingface/datasets

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs wget \
        libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Miniforge ────────────────────────────────────────────────────────────────
RUN curl -fsSL \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
        -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy

# ── conda env ────────────────────────────────────────────────────────────────
RUN /opt/conda/bin/conda create -y -n rkllm python=3.12 \
    && /opt/conda/bin/conda clean -afy

# ── rknn-llm repo ────────────────────────────────────────────────────────────
RUN git clone --depth 1 --branch ${RKLLM_TAG} \
        https://github.com/airockchip/rknn-llm.git /opt/rknn-llm

# ── Python packages ───────────────────────────────────────────────────────────
RUN /opt/conda/bin/conda run -n rkllm pip install --no-cache-dir \
        /opt/rknn-llm/rkllm-toolkit/packages/rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl

# rkllm-toolkit pins transformers==4.55.2 (Qwen2-VL era); Qwen3-VL requires >=4.57
RUN /opt/conda/bin/conda run -n rkllm pip install --no-cache-dir "transformers>=4.57" accelerate

RUN /opt/conda/bin/conda run -n rkllm pip install --no-cache-dir rknn-toolkit2==2.3.2

RUN /opt/conda/bin/conda run -n rkllm pip install --no-cache-dir \
        "huggingface_hub[cli]" "onnx==1.16.2" psutil pillow numpy

# setuptools must be last: pip dependency resolution drops it from the
# Python 3.12 conda env. rknn-toolkit2 requires pkg_resources at runtime.
RUN /opt/conda/bin/conda run -n rkllm pip install --no-cache-dir \
        --force-reinstall setuptools==75.8.0

# rkllm-toolkit pulls torch 2.4.0+cu121 and torchvision 0.21.0+cu124 which
# are mismatched (different CUDA builds). Reinstall matching CPU versions:
#   torch 2.4.0  <->  torchvision 0.19.0
RUN /opt/conda/bin/conda run -n rkllm pip install --no-cache-dir \
        torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cpu

# ── torchvision patch ─────────────────────────────────────────────────────────
# torchvision==0.21.0+cpu: operator torchvision::nms missing on CPU builds
# (pytorch/vision#8985). Patch wraps the broken registration in try/except.
COPY patch_torchvision.py /tmp/patch_torchvision.py
RUN /opt/conda/bin/conda run -n rkllm python /tmp/patch_torchvision.py && \
    /opt/conda/bin/conda run -n rkllm python -c "import torchvision; print('torchvision OK')"

# ── Directories ───────────────────────────────────────────────────────────────
RUN mkdir -p /cache/huggingface /output /scripts /workspace

WORKDIR /workspace

# ── Smoke test ────────────────────────────────────────────────────────────────
# RKNN() is instantiated (not just imported) — pkg_resources is only checked
# inside RKNNBase.__init__, so import alone would not catch a missing setuptools.
RUN /opt/conda/bin/conda run -n rkllm python -c "from rkllm.api import RKLLM; print('rkllm-toolkit OK')" && \
    /opt/conda/bin/conda run -n rkllm python -c "import pkg_resources; print('pkg_resources OK')" && \
    /opt/conda/bin/conda run -n rkllm python -c "from rknn.api import RKNN; r=RKNN(); r.release(); print('rknn-toolkit2 OK')" && \
    /opt/conda/bin/conda run -n rkllm python -c "from transformers import Qwen3VLForConditionalGeneration; print('Qwen3VL transformers OK')"

CMD ["python"]
