#!/usr/bin/env python3
"""
generate_calib.py - Generate quantisation calibration data (data_quant.json)

Produces a JSON file consumed by rkllm-toolkit's build() dataset parameter.
Format: [{"input": "<prompt>", "target": "<completion>"}, ...]

- 20-64 samples is sufficient; more does not meaningfully improve accuracy.
- Samples should reflect your actual use case and target language(s).
- Text-only prompts are fine even for a VLM — the vision encoder is calibrated
  separately and stays fp16; only the LLM half uses this data.
"""

import argparse
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download


CALIBRATION_PROMPTS = [
    # Vision-language
    {"input": "Describe this image in detail.",
     "target": "The image shows a busy urban street scene."},
    {"input": "What text is visible in this photo?",
     "target": "The sign reads 'Open 24 Hours'."},
    {"input": "What is the main subject of this image?",
     "target": "The main subject is a golden retriever sitting in a park."},
    {"input": "List all the objects you can see.",
     "target": "Chair, table, laptop, coffee mug, window, curtain."},
    {"input": "What time of day is shown?",
     "target": "The image appears to be taken at sunset based on the orange light."},
    {"input": "Is there any text in the image?",
     "target": "Yes, there is text on the whiteboard that reads 'Meeting at 3pm'."},
    {"input": "Describe the background of this scene.",
     "target": "The background shows a mountainous landscape with snow-capped peaks."},
    {"input": "Count the people in this image.",
     "target": "There are four people visible in the image."},

    # OCR
    {"input": "Read and transcribe all text in this document image.",
     "target": "Invoice #4521, Date: 2024-03-15, Total: $1,234.56"},
    {"input": "What does the road sign say?",
     "target": "The road sign reads 'Speed Limit 50 km/h'."},

    # Reasoning
    {"input": "Explain why the sky is blue.",
     "target": "The sky appears blue due to Rayleigh scattering of sunlight."},
    {"input": "What is the capital of France?",
     "target": "The capital of France is Paris."},
    {"input": "Summarise the key points of this chart.",
     "target": "The chart shows a steady increase in revenue from Q1 to Q4."},

    # Chinese — CJK token embeddings occupy different activation ranges
    {"input": "请描述这张图片的内容。",
     "target": "这张图片展示了一个繁忙的城市街道场景，有很多行人和车辆。"},
    {"input": "图中有哪些文字？",
     "target": "图中可以看到'欢迎光临'的字样。"},
    {"input": "这张照片是在哪里拍摄的？",
     "target": "照片看起来是在公园里拍摄的，背景有很多树木和草地。"},

    # Spatial / layout
    {"input": "Where is the red car in relation to the building?",
     "target": "The red car is parked in front of the building on the left side."},
    {"input": "Describe the layout of this room.",
     "target": "The room has a sofa on the left, a TV on the right wall, and a coffee table in the centre."},

    # Technical
    {"input": "What type of chart is this?",
     "target": "This is a bar chart showing monthly sales data."},
    {"input": "What is shown in the diagram?",
     "target": "The diagram illustrates the water cycle including evaporation, condensation, and precipitation."},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, help="HF model ID or local path (used to verify cache only)")
    p.add_argument("--output", required=True, help="Output data_quant.json path")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")

    # Resolve model path (ensures cache is populated for subsequent steps)
    if not os.path.isdir(args.model):
        snapshot_download(repo_id=args.model, token=token)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(CALIBRATION_PROMPTS, f, ensure_ascii=False, indent=2)

    print(f"[calib] Wrote {len(CALIBRATION_PROMPTS)} calibration records -> {args.output}")


if __name__ == "__main__":
    main()