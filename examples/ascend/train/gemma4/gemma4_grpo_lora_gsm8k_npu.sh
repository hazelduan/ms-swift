#!/usr/bin/env bash
set -euo pipefail

# Verified smoke and steady-state environment:
#   Python 3.11, torch 2.9.0+cpu, torch_npu 2.9.0, transformers 5.9.0,
#   huggingface_hub 1.21.0, hf-xet 1.5.1, click 8.4.2.
# Hardware: 8 * Ascend 910-class NPUs.
#
# Gemma4 uses a multimodal processor path. In transformers-engine GRPO rollout,
# the shared fast tokenizer may fail with "RuntimeError: Already borrowed" when
# batch encoding runs concurrently. Keep SWIFT_SERIAL_BATCH_ENCODE=1 for Gemma4
# GRPO unless you have verified concurrent processor encoding in your setup.

MODEL_PATH=${MODEL_PATH:-gemma-4-26B-A4B-it}
DATASET=${DATASET:-modelscope/gsm8k}
OUTPUT_DIR=${OUTPUT_DIR:-output/gemma4_grpo_lora_gsm8k_npu}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-'Solve briefly. End with exactly: #### <number>.'}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
export HCCL_OP_BASE_FFTS_MODE_ENABLE=${HCCL_OP_BASE_FFTS_MODE_ENABLE:-TRUE}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_DISABLED=${WANDB_DISABLED:-true}
export SWIFT_SERIAL_BATCH_ENCODE=${SWIFT_SERIAL_BATCH_ENCODE:-1}

NPROC_PER_NODE=${NPROC_PER_NODE:-8} \
swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --model_type gemma4 \
    --external_plugins examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py \
    --reward_funcs gsm8k_accuracy gsm8k_format \
    --tuner_type lora \
    --lora_rank 4 \
    --lora_alpha 8 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --dataset "${DATASET}" \
    --columns '{"answer": "solution"}' \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_steps "${MAX_STEPS:-200}" \
    --max_length 512 \
    --max_completion_length "${MAX_COMPLETION_LENGTH:-512}" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --save_strategy no \
    --eval_strategy no \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 1 \
    --deepspeed zero3 \
    --num_generations "${NUM_GENERATIONS:-4}" \
    --temperature "${TEMPERATURE:-0.6}" \
    --system "${SYSTEM_PROMPT}" \
    --log_completions true \
    --report_to none \
    --beta 0.0
