#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$HERE/../../../.." && pwd -P)"
: "${HIF8_MODEL:?set HIF8_MODEL to the existing Qwen3-0.6B directory}"
: "${HIF8_SOURCE_DATA:?set HIF8_SOURCE_DATA to the existing 500-row JSONL file}"
: "${HIF8_OUTPUT_ROOT:?set HIF8_OUTPUT_ROOT to a task-owned experiment directory}"
: "${HIF8_CONDA_ENV:?set HIF8_CONDA_ENV to the isolated Python environment}"
: "${HIF8_CANN_ENV:?set HIF8_CANN_ENV to the selected CANN set_env.sh}"
: "${HIF8_DEEPSPEED_REPO:?set HIF8_DEEPSPEED_REPO to the editable DeepSpeed checkout}"
: "${HIF8_TORCH_NPU_REPO:?set HIF8_TORCH_NPU_REPO to the editable torch-npu checkout}"
HIF8_DEVICE="${HIF8_DEVICE:-4}"
HIF8_CPUSET="${HIF8_CPUSET:-0-63,128-191}"
HIF8_EXPECTED_CANN_VERSION="${HIF8_EXPECTED_CANN_VERSION:-9.1.0}"
HIF8_MASTER_PORT="${HIF8_MASTER_PORT:-29531}"
export HIF8_MODEL HIF8_SOURCE_DATA HIF8_OUTPUT_ROOT HIF8_CONDA_ENV HIF8_CANN_ENV
export HIF8_DEEPSPEED_REPO HIF8_TORCH_NPU_REPO HIF8_DEVICE HIF8_CPUSET
export HIF8_EXPECTED_CANN_VERSION
export HIF8_MASTER_PORT

die() {
    echo "ERROR: $*" >&2
    exit 1
}

activate_clean_environment() {
    local name torch_lib
    while IFS='=' read -r name _; do
        case "$name" in
            ASCEND*|ACL*|HCCL*|LD_LIBRARY_PATH|PYTHONHOME|PYTHONPATH|VIRTUAL_ENV|CUDA_VISIBLE_DEVICES|ASCEND_RT_VISIBLE_DEVICES|RANK|WORLD_SIZE|LOCAL_RANK|LOCAL_WORLD_SIZE|GROUP_RANK|ROLE_RANK|NPROC_PER_NODE|NNODES|NODE_RANK|MASTER_ADDR|MASTER_PORT)
                unset "$name"
                ;;
        esac
    done < <(env)
    # shellcheck disable=SC1090
    set +u
    source "$HIF8_CANN_ENV"
    set -u
    # A prefix cloned with `conda create --clone` need not contain bin/activate.
    # Selecting its executables directly is deterministic and does not mutate any
    # base environment.
    export CONDA_PREFIX="$HIF8_CONDA_ENV"
    export CONDA_DEFAULT_ENV="$HIF8_CONDA_ENV"
    export CONDA_SHLVL=1
    export PATH="$HIF8_CONDA_ENV/bin:$PATH"
    hash -r
    torch_lib="$(python - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / 'lib')
PY
)"
    export LD_LIBRARY_PATH="$torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ASCEND_RT_VISIBLE_DEVICES="$HIF8_DEVICE"
    export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
    export MASTER_ADDR=localhost MASTER_PORT="$HIF8_MASTER_PORT"
    export PYTHONHASHSEED=42
    export TOKENIZERS_PARALLELISM=false
    export TMPDIR="$HIF8_OUTPUT_ROOT/tmp"
    export HF_HOME="$HIF8_OUTPUT_ROOT/cache/huggingface"
    export TORCH_EXTENSIONS_DIR="$HIF8_OUTPUT_ROOT/cache/torch_extensions"
    export PIP_CACHE_DIR="$HIF8_OUTPUT_ROOT/cache/pip"
}

device_check() {
    local label="$1"
    local snapshot="$HIF8_OUTPUT_ROOT/device_${label}.log"
    npu-smi info >"$snapshot" 2>&1
    grep -Fq "No running processes found in NPU $HIF8_DEVICE" "$snapshot" || \
        die "NPU $HIF8_DEVICE is occupied or ambiguous; see $snapshot"
    awk -F '|' -v id="$HIF8_DEVICE" '
        function trim(value) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", value); return value }
        trim($2) == id && trim($4) == "OK" { ok = 1 }
        END { exit(ok ? 0 : 1) }
    ' "$snapshot" || \
        die "NPU $HIF8_DEVICE is not unambiguously healthy; see $snapshot"
}

preflight() {
    [[ "$HIF8_DEVICE" == 4 ]] || die "phase-1 is restricted to physical NPU 4"
    [[ "$EUID" == 0 ]] || die "A5 npu-smi/DCMI health checks require running this harness as root"
    [[ -x "$HIF8_CONDA_ENV/bin/python" ]] || die "isolated environment is unavailable"
    [[ "$(command -v python)" == "$HIF8_CONDA_ENV/bin/python" ]] || \
        die "python does not resolve from the isolated environment"
    [[ "$(command -v swift)" == "$HIF8_CONDA_ENV/bin/swift" ]] || \
        die "swift does not resolve from the isolated environment"
    [[ -f "$HIF8_CANN_ENV" ]] || die "CANN set_env.sh is unavailable"
    command -v npu-smi >/dev/null || die "npu-smi is unavailable"
    command -v taskset >/dev/null || die "taskset is unavailable"
    [[ -d "$HIF8_MODEL" ]] || die "model directory does not exist"
    [[ -f "$HIF8_SOURCE_DATA" ]] || die "dataset does not exist"
    mkdir -p "$HIF8_OUTPUT_ROOT/data" "$HIF8_OUTPUT_ROOT/runs" "$TMPDIR" "$HF_HOME"
    mkdir -p "$TORCH_EXTENSIONS_DIR" "$PIP_CACHE_DIR"
    device_check preflight
    python - "$HIF8_OUTPUT_ROOT/environment_and_revisions.json" <<'PY'
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

import deepspeed
import swift
import torch
import torch_npu
import transformers

expected = {
    'deepspeed': Path(os.environ['HIF8_DEEPSPEED_REPO']).resolve(),
    'torch_npu': Path(os.environ['HIF8_TORCH_NPU_REPO']).resolve(),
    'swift': Path.cwd().resolve(),
}
modules = {'deepspeed': deepspeed, 'torch_npu': torch_npu, 'swift': swift}
paths = {name: Path(inspect.getfile(module)).resolve() for name, module in modules.items()}
for name in expected:
    if not paths[name].is_relative_to(expected[name]):
        raise SystemExit(f'{name} is not editable from {expected[name]}: {paths[name]}')
if sys.version_info[:3] != (3, 11, 15):
    raise SystemExit(f'expected Python 3.11.15, got {sys.version.split()[0]}')
if torch.__version__ != '2.9.0+cpu' or torch_npu.__version__ != '2.9.0.post4':
    raise SystemExit(f'expected torch 2.9.0+cpu / torch_npu 2.9.0.post4, got {torch.__version__} / {torch_npu.__version__}')
if Version(transformers.__version__) != Version('5.16.1'):
    raise SystemExit(f'expected Transformers 5.16.1, got {transformers.__version__}')
expected_cann = os.environ['HIF8_EXPECTED_CANN_VERSION']
cann_root_name = Path(os.environ['HIF8_CANN_ENV']).resolve().parent.name
if expected_cann not in cann_root_name:
    raise SystemExit(f'expected CANN {expected_cann}, got environment root {cann_root_name}')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


model_root = Path(os.environ['HIF8_MODEL'])
weight_files = sorted(path for path in model_root.rglob('*.safetensors') if path.is_file())
index_files = sorted(path for path in model_root.rglob('*.safetensors.index.json') if path.is_file())
metadata_files = [model_root / name for name in ('config.json', 'tokenizer.json') if (model_root / name).is_file()]
if not weight_files:
    raise SystemExit(f'no safetensors model weights found below {model_root}')
if any(re.search(r'-\d{5}-of-\d{5}\.safetensors$', path.name) for path in weight_files) and not index_files:
    raise SystemExit('sharded safetensors weights require a *.safetensors.index.json manifest')
for index_path in index_files:
    weight_map = json.loads(index_path.read_text(encoding='utf-8')).get('weight_map')
    if not isinstance(weight_map, dict):
        raise SystemExit(f'invalid safetensors index: {index_path}')
    missing = sorted({name for name in weight_map.values() if not (index_path.parent / name).is_file()})
    if missing:
        raise SystemExit(f'safetensors index {index_path} references missing files: {missing}')
model_files = sorted(set(weight_files + index_files + metadata_files))
model_manifest = {
    str(path.relative_to(model_root)): {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    for path in model_files
}

payload = {
    'python': sys.version,
    'torch': torch.__version__,
    'torch_npu': torch_npu.__version__,
    'transformers': transformers.__version__,
    'deepspeed': deepspeed.__version__,
    'swift': getattr(swift, '__version__', 'unknown'),
    'module_paths': {name: str(path) for name, path in paths.items()},
    'git_revisions': {
        name: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        for name, path in expected.items()
    },
    'git_dirty': {
        name: bool(subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip())
        for name, path in expected.items()
    },
    'visible_devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
    'distributed_environment': {
        name: os.environ.get(name)
        for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT')
    },
    'cpu_affinity': os.environ['HIF8_CPUSET'],
    'cann_environment': os.environ['HIF8_CANN_ENV'],
    'model_manifest': model_manifest,
    'source_dataset_sha256': sha256(os.environ['HIF8_SOURCE_DATA']),
}
dirty = [name for name, value in payload['git_dirty'].items() if value]
if dirty:
    raise SystemExit(f'experiment repositories must be clean: {dirty}')
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
    python "$HERE/prepare_alpaca_split.py" "$HIF8_SOURCE_DATA" "$HIF8_OUTPUT_ROOT/data" \
        --seed 42 --train-size 400 --eval-size 100
    local extension
    extension="$(python - <<'PY'
import importlib.util
spec = importlib.util.find_spec('torch_npu._C')
print(spec.origin if spec else '')
PY
)"
    [[ -n "$extension" ]] || die "torch_npu native extension is unavailable"
    ldd "$extension" >"$HIF8_OUTPUT_ROOT/torch_npu_extension_ldd.log"
    ! grep -Fq 'not found' "$HIF8_OUTPUT_ROOT/torch_npu_extension_ldd.log" || \
        die "torch_npu has unresolved shared libraries; see torch_npu_extension_ldd.log"
    grep -Eq '/cann-(8|9\.0)|ascend-toolkit/latest' "$HIF8_OUTPUT_ROOT/torch_npu_extension_ldd.log" && \
        die "torch_npu resolved a stale CANN library; see torch_npu_extension_ldd.log"
    return 0
}

launch() {
    local name="$1" yaml="$2" deepspeed_config="$3" max_steps="$4"
    local eval_strategy="$5" save_strategy="$6"
    shift 6
    local run_dir="$HIF8_OUTPUT_ROOT/runs/$name"
    local validate_lora_gradients=0
    local audit_op_contract=0
    [[ "$name" == accuracy/lora_* || "$name" == resume/lora_* ]] && validate_lora_gradients=1
    [[ "$name" == accuracy/* || "$name" == resume/* ]] && audit_op_contract=1
    [[ ! -e "$run_dir" ]] || die "refusing to overwrite $run_dir"
    mkdir -p "$run_dir"
    device_check "${name//\//_}"
    local command=(
        swift sft "$yaml"
        --model "$HIF8_MODEL"
        --dataset "$HIF8_OUTPUT_ROOT/data/train.jsonl"
        --val_dataset "$HIF8_OUTPUT_ROOT/data/eval.jsonl"
        --deepspeed "$deepspeed_config"
        --ddp_backend gloo
        --output_dir "$run_dir/output"
        --max_steps "$max_steps"
        --eval_strategy "$eval_strategy"
        --eval_steps 25
        --save_strategy "$save_strategy"
        --save_steps 50
        --logging_steps 1
        --seed 42
        --data_seed 42
        "$@"
    )
    printf '%q ' env "HIF8_VALIDATE_LORA_GRADS=$validate_lora_gradients" \
        "HIF8_AUDIT_OP_CONTRACT=$audit_op_contract" \
        taskset -c "$HIF8_CPUSET" "${command[@]}" >"$run_dir/command.sh"
    printf '\n' >>"$run_dir/command.sh"
    cp "$deepspeed_config" "$run_dir/deepspeed_config.json"
    set +e
    env "HIF8_VALIDATE_LORA_GRADS=$validate_lora_gradients" \
        "HIF8_AUDIT_OP_CONTRACT=$audit_op_contract" \
        taskset -c "$HIF8_CPUSET" "${command[@]}" 2>&1 | tee "$run_dir/stdout.log"
    local status="${PIPESTATUS[0]}"
    set -e
    printf '%s\n' "$status" >"$run_dir/exitcode"
    [[ "$status" == 0 ]] || return "$status"
}

primitive() {
    local rows
    mkdir -p "$HIF8_OUTPUT_ROOT/primitives"
    for rows in 1 127 128 2560; do
        device_check "primitive_m${rows}"
        set +e
        taskset -c "$HIF8_CPUSET" python "$HERE/validate_hifloat8_linear.py" \
            --rows "$rows" --output "$HIF8_OUTPUT_ROOT/primitives/m${rows}.json" \
            >"$HIF8_OUTPUT_ROOT/primitives/m${rows}.log" 2>&1
        local status="$?"
        set -e
        printf '%s\n' "$status" >"$HIF8_OUTPUT_ROOT/primitives/m${rows}.exitcode"
        [[ "$status" == 0 ]] || return "$status"
    done
}

accuracy() {
    launch accuracy/full_bf16 "$HERE/qwen3_0_6b_full.yaml" "$HERE/zero2_bf16.json" 100 steps steps
    launch accuracy/full_hifloat8 "$HERE/qwen3_0_6b_full.yaml" "$HERE/zero2_hifloat8.json" 100 steps steps
    launch accuracy/lora_bf16 "$HERE/qwen3_0_6b_lora.yaml" "$HERE/zero2_bf16.json" 100 steps steps
    launch accuracy/lora_hifloat8 "$HERE/qwen3_0_6b_lora.yaml" "$HERE/zero2_hifloat8_lora.json" 100 steps steps
}

resume() {
    local tuner precision yaml config checkpoint
    for tuner in full lora; do
        for precision in bf16 hifloat8; do
            yaml="$HERE/qwen3_0_6b_${tuner}.yaml"
            config="$HERE/zero2_bf16.json"
            [[ "$precision" == hifloat8 && "$tuner" == full ]] && config="$HERE/zero2_hifloat8.json"
            [[ "$precision" == hifloat8 && "$tuner" == lora ]] && config="$HERE/zero2_hifloat8_lora.json"
            checkpoint="$HIF8_OUTPUT_ROOT/runs/accuracy/${tuner}_${precision}/output/checkpoint-50"
            [[ -d "$checkpoint" ]] || die "missing checkpoint $checkpoint"
            launch "resume/${tuner}_${precision}" "$yaml" "$config" 100 steps steps \
                --resume_from_checkpoint "$checkpoint"
        done
    done
}

performance() {
    local repeat
    for repeat in 1 2 3; do
        launch "perf/full_bf16_r${repeat}" "$HERE/qwen3_0_6b_full.yaml" "$HERE/zero2_bf16.json" 50 no no
        launch "perf/full_hifloat8_r${repeat}" "$HERE/qwen3_0_6b_full.yaml" "$HERE/zero2_hifloat8.json" 50 no no
        launch "perf/lora_bf16_r${repeat}" "$HERE/qwen3_0_6b_lora.yaml" "$HERE/zero2_bf16.json" 50 no no
        launch "perf/lora_hifloat8_r${repeat}" "$HERE/qwen3_0_6b_lora.yaml" "$HERE/zero2_hifloat8_lora.json" 50 no no
    done
}

profile() {
    local args=(--external_plugins "$HERE/npu_profile.py" --callbacks hifloat8_profiler)
    launch profile/full_hifloat8 "$HERE/qwen3_0_6b_full.yaml" "$HERE/zero2_hifloat8.json" 4 no no "${args[@]}"
    launch profile/lora_hifloat8 "$HERE/qwen3_0_6b_lora.yaml" "$HERE/zero2_hifloat8_lora.json" 4 no no "${args[@]}"
}

report() {
    python "$HERE/summarize_phase1.py" "$HIF8_OUTPUT_ROOT" \
        --output-dir "$HIF8_OUTPUT_ROOT/results"
}

main() {
    activate_clean_environment
    cd "$REPO_ROOT"
    preflight
    case "${1:-all}" in
        preflight) ;;
        primitive) primitive ;;
        accuracy) accuracy ;;
        resume) resume ;;
        perf) performance ;;
        profile) profile ;;
        report) report ;;
        all) primitive; accuracy; resume; performance; profile; report ;;
        *) die "usage: $0 [preflight|primitive|accuracy|resume|perf|profile|report|all]" ;;
    esac
}

main "$@"
