# rkllm-convert

Convert Qwen3-VL models to `.rkllm` / `.rknn` format for the RK3588 NPU.

## Requirements

- Docker and Docker Compose
- HuggingFace account and access token
- Enough disk space for model weights (see table below)

| Model | Source weights | Converted output | Conversion RAM | RK3588 board RAM | 32 GB board |
|---|---|---|---|---|---|
| Qwen3-VL-2B | ~5 GB | ~3 GB | ~8 GB | ~3 GB | ✅ |
| Qwen3-VL-4B | ~9 GB | ~5.5 GB | ~16 GB | ~5 GB | ✅ |
| Qwen3-VL-8B | ~18 GB | ~9.5 GB | ~34 GB | ~10 GB | ✅ tight |
| Qwen3-VL-30B-A3B | ~12 GB | ~6 GB | ~24 GB | ~6 GB | ✅ (broken, see TODO) |
| Qwen3-VL-32B | ~66 GB | ~33 GB | ~128 GB | ~33 GB | ❌ |

> **Conversion RAM** is peak Python process memory during W8A8 quantisation (float32 load + calibration overhead). Measured: 8B = 33.5 GB.
> **32B conversion** requires ~128 GB RAM+swap on the conversion PC.
> **32B on board** requires ~33 GB — exceeds the 32 GB RK3588 board limit. Not recommended.

## Setup

```bash
# 1. Copy and fill in your HuggingFace token
cp .env.example .env
nano .env

# 2. Create cache and output directories
mkdir -p cache/huggingface output

# 3. Build the Docker image (once)
docker compose build
```

## Convert a model

```bash
docker compose run --rm convert <model_id> [vision_resolution] [max_context]
```

`vision_resolution` is optional, default `896`. Options: `448`, `640`, `896`.

| `<model_id>` | Output folder |
|---|---|
| `Qwen/Qwen3-VL-2B-Instruct` | `qwen3-vl-2b-instruct` |
| `Qwen/Qwen3-VL-4B-Instruct` | `qwen3-vl-4b-instruct` |
| `Qwen/Qwen3-VL-8B-Instruct` | `qwen3-vl-8b-instruct` |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | `qwen3-vl-30b-a3b-instruct` |
| `Qwen/Qwen3-VL-32B-Instruct` | `qwen3-vl-32b-instruct` |

```bash
docker compose run --rm convert Qwen/Qwen3-VL-2B-Instruct 896
docker compose run --rm convert Qwen/Qwen3-VL-4B-Instruct 896
docker compose run --rm convert Qwen/Qwen3-VL-8B-Instruct 896
```

Each step is cached — if the run is interrupted, re-running the same command
will skip completed steps and resume from where it stopped.

## Output files

Results are written to `output/<model-name>/`:

```
output/qwen3-vl-2b-instruct/
├── qwen3-vl-2b-instruct_vision_896.onnx          # intermediate, reused on re-runs
├── qwen3-vl-2b-instruct_vision_896_rk3588.rknn   # copy to board
├── data_quant.json                                # calibration data
└── qwen3-vl-2b-instruct_w8a8_rk3588.rkllm        # copy to board
```

## Run on board

Build or obtain the `VLM_NPU` demo application from [Qwen3-VL-2B-NPU](https://github.com/Qengineering/Qwen3-VL-2B-NPU), then run:

```bash
time ./VLM_NPU \
  /opt/images/IMG_8220.jpg \
  /opt/models/rkllm/qwen3-vl-4b-instruct/qwen3-vl-4b-instruct_vision_896_rk3588.rknn \
  /opt/models/rkllm/qwen3-vl-4b-instruct/qwen3-vl-4b-instruct_w8a8_rk3588.rkllm \
  2048 4096 \
  "Describe this image in detail."
```

Arguments: `<image> <vision.rknn> <llm.rkllm> <embed_size> <max_context> <prompt>`

| Model | embed_size |
|---|---|
| 2B | 2048 |
| 4B | 2560 |
| 8B | 3584 |

## Copy to board

Replace `rock5` with your board hostname or IP. Create the destination directory first if it doesn't exist.

```bash
# 2B
scp output/qwen3-vl-2b-instruct/qwen3-vl-2b-instruct_vision_896_rk3588.rknn \
    radxa@rock5:/opt/models/rkllm/qwen3-vl-2b-instruct/qwen3-vl-2b-instruct_vision_896_rk3588.rknn
scp output/qwen3-vl-2b-instruct/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm \
    radxa@rock5:/opt/models/rkllm/qwen3-vl-2b-instruct/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm

# 4B
scp output/qwen3-vl-4b-instruct/qwen3-vl-4b-instruct_vision_896_rk3588.rknn \
    radxa@rock5:/opt/models/rkllm/qwen3-vl-4b-instruct/qwen3-vl-4b-instruct_vision_896_rk3588.rknn
scp output/qwen3-vl-4b-instruct/qwen3-vl-4b-instruct_w8a8_rk3588.rkllm \
    radxa@rock5:/opt/models/rkllm/qwen3-vl-4b-instruct/qwen3-vl-4b-instruct_w8a8_rk3588.rkllm

# 8B
scp output/qwen3-vl-8b-instruct/qwen3-vl-8b-instruct_vision_896_rk3588.rknn \
    radxa@rock5:/opt/models/rkllm/qwen3-vl-8b-instruct/qwen3-vl-8b-instruct_vision_896_rk3588.rknn
scp output/qwen3-vl-8b-instruct/qwen3-vl-8b-instruct_w8a8_rk3588.rkllm \
    radxa@rock5:/opt/models/rkllm/qwen3-vl-8b-instruct/qwen3-vl-8b-instruct_w8a8_rk3588.rkllm
```

## Context size

The default max context is 4096 tokens. Pass it as a third positional argument:

```bash
# Usage: convert <model_id> <vision_res> [max_context]
# max_context must be a multiple of 32, maximum 16384

docker compose run --rm convert Qwen/Qwen3-VL-4B-Instruct 448 8192
```

Larger contexts increase board RAM usage at inference time:

| Context | Approx. extra RAM |
|---|---|
| 4096 (default) | baseline |
| 8192 | +~1 GB |
| 16384 | +~3 GB |

## Debug shell

```bash
docker compose run --rm shell
```

## TODO

### Qwen3-VL-30B-A3B support

The 30B-A3B is a Mixture-of-Experts model — only 3B parameters are active per forward pass, so it fits easily on a 32 GB board and converts with modest RAM. However W8A8 quantisation currently produces garbage mixed-language output.

Root cause: the MoE router's gating weights are sensitive to activation quantisation error, causing incorrect expert selection at inference time.

Fixes to investigate:
- Switch to `w4a16` (weight-only quantisation) — avoids quantising activations entirely, which is where MoE routing is most sensitive
- Expand calibration data — current 20 samples are too sparse for the router to calibrate correctly; need more diverse, longer reasoning samples that force different experts to activate