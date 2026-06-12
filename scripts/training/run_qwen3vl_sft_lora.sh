#!/usr/bin/env bash
# Stage-1 VLM SFT for ReCogDrive on Qwen3-VL-2B-Instruct (LoRA), via ms-swift.
#
# WHY this script exists:
#   internvl_chat/internvl/train/internvl_chat_finetune.py hardcodes
#   InternVLChatModel/InternVLChatConfig and CANNOT load Qwen3-VL
#   (model_type=qwen3_vl). To fine-tune the Qwen3 backbone we use ms-swift,
#   which has first-class Qwen3-VL support.
#
# OUTPUT: a LoRA adapter under ${OUTPUT_DIR}. Merge it back into the base model
#   with scripts/training/merge_qwen3vl_lora.sh, then point the ReCogDrive
#   pipeline's QWEN3_VL_PATH / agent.vlm_path at the MERGED directory before
#   running run_caching_recogdrive_qwen3_hidden_state.sh.
#
# PREREQ: pip install "ms-swift" (not installed in the recogdrive env yet).
#   ms-swift pulls a transformers that supports Qwen3-VL; the active env already
#   has transformers==4.57.1 which is sufficient.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- Base model (ORIGINAL, un-finetuned Qwen3-VL-2B-Instruct) ---
QWEN3_VL_PATH="${QWEN3_VL_PATH:-/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct}"

# --- Training data (ShareGPT-style jsonl produced by run_generate_dataset.py) ---
# Placeholder paths. Replace with the actual Mogo/NAVSIM trajectory+QA jsonl.
# Each line: {"id":..,"image":["/abs/path.jpg"],"conversations":[
#   {"from":"system","value":..},{"from":"human","value":"<image>\n.."},
#   {"from":"gpt","value":".."}]}
# ms-swift reads this sharegpt schema directly (it maps from->role,
# value->content, and "image"->images). If your loader needs explicit columns,
# convert with the helper noted in CLAUDE.md "Stage 1 (Qwen3-VL SFT)".
DATASET="${DATASET:-/path/to/recogdrive_qwen3_sft.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/work_dirs/qwen3vl_2b_sft_lora}"
mkdir -p "${OUTPUT_DIR}"

# --- Distributed / hardware ---
GPUS="${GPUS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${GPUS}"
export MASTER_PORT="${MASTER_PORT:-34229}"

# --- LoRA + optimization hyperparameters ---
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACC="${GRADIENT_ACC:-16}"
LR="${LR:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

# freeze_vit=true keeps the vision tower frozen (cheaper, common for 2B LoRA).
# Set FREEZE_VIT=false to also adapt the ViT via LoRA target modules.
FREEZE_VIT="${FREEZE_VIT:-true}"

swift sft \
  --model "${QWEN3_VL_PATH}" \
  --model_type qwen3_vl \
  --dataset "${DATASET}" \
  --train_type lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --freeze_vit "${FREEZE_VIT}" \
  --torch_dtype bfloat16 \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACC}" \
  --learning_rate "${LR}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --max_length "${MAX_LENGTH}" \
  --gradient_checkpointing true \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 5 \
  --logging_steps 5 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard \
  "$@" 2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
