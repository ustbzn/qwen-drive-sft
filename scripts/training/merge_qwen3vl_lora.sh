#!/usr/bin/env bash
# Merge a Qwen3-VL LoRA adapter (from run_qwen3vl_sft_lora.sh) back into the
# base Qwen3-VL-2B-Instruct, producing a standalone weight directory that the
# ReCogDrive pipeline can load directly via QWEN3_VL_PATH / agent.vlm_path.
#
# WHY merge: ReCogDriveBackbone loads the VLM with
#   Qwen3VLForConditionalGeneration.from_pretrained(vlm_path) and has NO LoRA
#   loading path. It expects a full model directory, not an adapter. Merging is
#   the simplest way to feed an SFT'd Qwen3 into hidden-state caching.
set -euo pipefail

# Path to the LoRA checkpoint directory written by `swift sft`
# (e.g. work_dirs/qwen3vl_2b_sft_lora/checkpoint-xxx).
ADAPTER_DIR="${ADAPTER_DIR:?Set ADAPTER_DIR to the swift sft LoRA checkpoint dir}"

# Where to write the merged full model. Point QWEN3_VL_PATH here afterwards.
MERGED_DIR="${MERGED_DIR:?Set MERGED_DIR to the output merged-weights dir}"

swift export \
  --adapters "${ADAPTER_DIR}" \
  --merge_lora true \
  --output_dir "${MERGED_DIR}" \
  --torch_dtype bfloat16 \
  "$@"

echo ""
echo "Merged weights written to: ${MERGED_DIR}"
echo "Next: export QWEN3_VL_PATH=${MERGED_DIR} and re-run"
echo "      scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh"
