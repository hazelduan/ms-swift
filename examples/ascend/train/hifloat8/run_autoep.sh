#!/usr/bin/env bash
# Qwen3.5-35B-A3B text-only AutoEP validation. Source CANN before invoking.
set -euo pipefail
precision=${1:?usage: run_autoep.sh bf16|hifloat8 [additional Swift arguments]}
shift
case "$precision" in bf16|hifloat8) ;; *) exit 2;; esac
: "${HIF8_MODEL:?set the existing model directory}"
: "${HIF8_DATASET:?set the training JSONL}"
: "${HIF8_OUTPUT_DIR:?set a fresh output directory}"
: "${ASCEND_RT_VISIBLE_DEVICES:?select the allocated physical cards}"
: "${RANK_TABLE_FILE:?set the task-owned CANN v2 rank table}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../../../.." && pwd)
trainable_regex='^model\.language_model\.layers\.(?:0|39)\.mlp\.(?:experts\.(?:gate_up_proj|down_proj)|shared_expert\.(?:gate_proj|up_proj|down_proj)\.weight)$'
python -m torch.distributed.run --nproc_per_node=4 --master_port="${MASTER_PORT:-29654}" \
    "$repo_dir/swift/cli/sft.py" \
    --model "$HIF8_MODEL" --use_hf true --check_model false \
    --dataset "$HIF8_DATASET" --split_dataset_ratio 0 \
    --tuner_type full --freeze_parameters_regex '.*' --trainable_parameters_regex "$trainable_regex" \
    --torch_dtype bfloat16 --bf16 true --fp16 false --attn_impl eager --experts_impl eager \
    --deepspeed "$script_dir/zero2_autoep_$precision.json" --enable_npu_model_patch false \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --gradient_checkpointing false \
    --max_length 256 --max_steps 20 --learning_rate 1e-5 --optim adamw_torch \
    --lr_scheduler_type cosine --warmup_ratio 0 --weight_decay 0 --max_grad_norm 1 \
    --seed 42 --data_seed 42 --dataset_shuffle true --lazy_tokenize false \
    --load_from_cache_file false --dataset_num_proc 1 --dataloader_num_workers 0 --dataloader_persistent_workers false \
    --logging_steps 1 --logging_first_step true --save_strategy steps --save_steps 10 \
    --report_to none --disable_tqdm true --include_num_input_tokens_seen true \
    --add_version false --output_dir "$HIF8_OUTPUT_DIR" "$@"
