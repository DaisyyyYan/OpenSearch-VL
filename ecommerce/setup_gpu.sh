#!/usr/bin/env bash
# Environment for the rented GPU machine (Linux, A100). UNTESTED draft: run it step by step
# the first time and fix whatever the machine's image already provides.
#
# Expected layout after you upload things:
#   /workspace/OpenSearch-VL      (git clone of your fork, branch ecommerce)
#   /workspace/sft                (output folder of ecommerce/to_sft.py: dataset_info.json, json files, images/)
#   /workspace/data               (questions/, world/, clip_index/ for evaluation; same folders as on your PC)
set -euo pipefail
cd /workspace

# 1. code
[ -d OpenSearch-VL ] || git clone -b ecommerce https://github.com/DaisyyyYan/OpenSearch-VL.git

# 2. training framework
[ -d LLaMA-Factory ] || git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
python -m pip install -e "./LLaMA-Factory[torch,metrics]"

# 3. what the agent loop and the offline tools need (OCR, CLIP, BM25 knowledge base)
python -m pip install rapidocr_onnxruntime pillow numpy
python -m pip install -r OpenSearch-VL/requirements.txt || echo "no/failed requirements.txt: install missing packages by hand"

# 4. sanity checks
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available(), "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
ls /workspace/sft | head
echo "next: llamafactory-cli train OpenSearch-VL/ecommerce/qwen3vl_2b_lora_sft.yaml  (try max_steps=20 first)"
echo "after training, merge: llamafactory-cli export <merge yaml>  -> /workspace/outputs/merged"
