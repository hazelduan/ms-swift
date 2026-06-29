#!/usr/bin/env bash
set -euo pipefail

# Verified smoke environment:
#   Python 3.11, torch 2.9.0+cpu, torch_npu 2.9.0, transformers 5.9.0,
#   huggingface_hub 1.21.0, hf-xet 1.5.1, click 8.4.2.
# Hardware: 8 * Ascend 910-class NPUs.

MODEL_PATH=${MODEL_PATH:-gemma-4-26B-A4B-it}
DATASET=${DATASET:-AI-ModelScope/alpaca-gpt4-data-zh#2000}
OUTPUT_DIR=${OUTPUT_DIR:-output/gemma4_sft_lora_smoke_npu}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HCCL_OP_BASE_FFTS_MODE_ENABLE=${HCCL_OP_BASE_FFTS_MODE_ENABLE:-TRUE}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_DISABLED=${WANDB_DISABLED:-true}

NPROC_PER_NODE=${NPROC_PER_NODE:-8} \
swift sft \
    --model "${MODEL_PATH}" \
    --model_type gemma4 \
    --dataset "${DATASET}" \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --tuner_type lora \
    --lora_rank 4 \
    --lora_alpha 8 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --max_steps "${MAX_STEPS:-1}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --save_strategy no \
    --eval_strategy no \
    --max_length 256 \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 1 \
    --deepspeed zero3 \
    --report_to none
