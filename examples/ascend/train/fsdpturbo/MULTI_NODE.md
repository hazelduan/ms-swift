# 多机 A3：DeepSeek V4 Flash 与 Qwen3.5-122B-A10B

本指南使用独立 `swift fsdpturbo sft` 后端。Swift 负责模型注册、模板和数据；FSDPTurbo 负责 FSDP、EP/EFSDP、优化器、梯度裁剪及 DCP checkpoint。无需 Megatron 或 mcore-bridge。

## 代码

```bash
git clone --branch fsdpturbo_backend https://github.com/hazelduan/ms-swift.git
git clone --branch swift-fsdpturbo-compat https://gitcode.com/hazeldxq/FSDPTurbo.git
git -C ms-swift rev-parse HEAD
git -C FSDPTurbo rev-parse HEAD
```

所有节点使用同一提交。已有仓库可 `git fetch` 后切换相应分支，使用 `git merge --ff-only origin/<branch>` 更新。先确认 `git status --porcelain` 为空。具体验证提交及实验范围见 [README](README.md)。

本次代码验证点：Swift `0400d8c2d5c5c68cea4c6e166dcf5892781d47c5`，Turbo `ee117ade23ab9b479df1b40b86780d8a8a64a251`。后续仅文档提交不改变这套代码。upstream 基线分别为 `0bd7b0aa1d3f1f0fdd8642ae19578420f578d9e1` 和 `0a4b3bc3965fac3fb7423115da2ab0627dfe0076`。

维护已有 fork 的命令如下。训练节点只需拉取 fork 分支；合并 upstream、解决冲突和回归验证应在开发 workspace 完成，不要在运行中的训练目录直接更新。

```bash
# Swift 开发 workspace；origin 是 hazelduan/ms-swift，upstream 是 modelscope/ms-swift。
git -C ms-swift fetch upstream
git -C ms-swift fetch origin
git -C ms-swift switch fsdpturbo_backend
git -C ms-swift merge upstream/main
git -C ms-swift rev-list --count HEAD..upstream/main  # 本次为 0，不 behind
# 完成回归后发布：
git -C ms-swift push origin fsdpturbo_backend

# Turbo 开发 workspace；fork 是 hazeldxq/FSDPTurbo，upstream 是 Ascend/FSDPTurbo。
git -C FSDPTurbo fetch upstream
git -C FSDPTurbo fetch fork
git -C FSDPTurbo switch swift-fsdpturbo-compat
git -C FSDPTurbo merge upstream/main
git -C FSDPTurbo rev-list --count HEAD..upstream/main  # 本次为 0
git -C FSDPTurbo push fork swift-fsdpturbo-compat
```

GitCode 分支已从 `codex/swift-fsdpturbo-compat` 更名为 `swift-fsdpturbo-compat`，旧名称不再用于部署。若 GitCode HTTPS push 没有凭据，可使用已配置的 SSH key：`git push git@gitcode.com:hazeldxq/FSDPTurbo.git swift-fsdpturbo-compat:swift-fsdpturbo-compat`。

## 环境

已经实机验证的组合：Python 3.11.15、CANN 9.0.0、driver 26.0.rc1、torch 2.9.0+cpu、torch_npu 2.9.0、Transformers 5.9.0、Accelerate 1.13.0、datasets 4.8.4、safetensors 0.7.0、Triton 3.2.0、triton-ascend 3.2.1。FLA 0.5.2 可以保留；Qwen 的原生 Torch GDN fallback 已验证，FLA 快路径与长上下文性能需要另外验证。

在每台节点相同的 conda 环境内安装。已有可用 Ascend 环境可直接复用；不要因安装后端而替换 torch、torch_npu 或 CANN。

```bash
conda create -n swift-fsdpturbo python=3.11 pip -y
conda activate swift-fsdpturbo
# 复用已有环境时改为 conda activate <已有环境路径>，跳过上面的 create。
source /usr/local/Ascend/cann/set_env.sh

# 新环境安装匹配的 PyTorch CPU 与 Ascend 扩展；x86_64 节点示例。
python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-npu==2.9.0
python -m pip install transformers==5.9.0 accelerate==1.13.0 datasets==4.8.4 safetensors==0.7.0
python -m pip install trl==0.29.1 peft==0.18.1 numpy==1.26.4

# Ascend Triton 请使用集群已有的匹配 wheel；当前实验是以下版本。
python -m pip install triton==3.2.0 triton-ascend==3.2.1
python -m pip install -e ./ms-swift
# 当前非量化训练不需要 backend 声明的较新量化工具链。
python -m pip install --no-deps -e ./FSDPTurbo
```

安装后检查实际导入路径，避免旧 `PYTHONPATH` overlay 覆盖 editable checkout：

```bash
python - <<'PY'
import torch, torch_npu, transformers, datasets, swift, fsdp_turbo
for module in (torch, torch_npu, transformers, datasets, swift, fsdp_turbo):
    print(module.__name__, getattr(module, '__version__', None), module.__file__)
from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.distributed.parallel_state import reset_parallel_state
print('backend imports OK')
PY
```

本次 A3 `115.190.166.102` 使用的实际路径：

| 项目 | 路径 / 版本 |
| --- | --- |
| Swift workspace | `/home/dxq/.codex_work/ms-swift-fsdpturbo-release`，`fsdpturbo_backend`，`4.6.0.dev0` |
| Turbo workspace | `/home/dxq/fsdpturbo`，`swift-fsdpturbo-compat`，包声明 `0.1.0`，以 Git SHA 为准 |
| conda / Python | `/home/dxq/envs/dxq_swift_ascend_py311`，Python `3.11.15` |
| CANN | `/usr/local/Ascend/cann -> /usr/local/Ascend/cann-9.0.0`，`9.0.0` / `V100R001C10SPC001B250` |
| torch / torch_npu | conda 下 `lib/python3.11/site-packages/torch` / `torch_npu`，`2.9.0+cpu` / `2.9.0` |
| Transformers / Accelerate | 同一 conda 的 `site-packages/transformers` / `accelerate`，`5.9.0` / `1.13.0` |
| datasets overlay | `/home/dxq/experiments/swift_fsdpturbo_cann91_20260923/dependencies/datasets`，`4.8.4` |

该环境保留了历史依赖 overlay；复现本次 source checkout 时使用以下导入顺序，不能让 overlay 内的旧 Turbo 副本排在 workspace 前面。新建并按上文安装的环境不需要这个历史 overlay。

```bash
conda activate /home/dxq/envs/dxq_swift_ascend_py311
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH=/home/dxq/.codex_work/ms-swift-fsdpturbo-release:/home/dxq/fsdpturbo:/home/dxq/experiments/swift_fsdpturbo_cann91_20260923/dependencies
```

## 模型与数据

准备完整 checkpoint、tokenizer/processor 和 `config.json`。所有节点能访问同一模型路径及 JSONL 数据路径；checkpoint 输出目录必须是共享可写目录。可以使用共享文件系统，也可以预先同步输入文件。

- Qwen：`Qwen/Qwen3.5-122B-A10B` BF16 checkpoint。
- DeepSeek：`deepseek-ai/DeepSeek-V4-Flash`；其 FP8/FP4 checkpoint 会在 global rank 0 由 HF 转换为浮点训练参数。不要设置 Swift `quant_method`。FP32 稳定性参数会保留。
- 数据采用 Swift messages 格式，例如 `{"messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"答案"}]}`，每行一个样本。

启动时只有 global rank 0 完整加载/转换 CPU checkpoint。各 rank 在 meta 上建模并完成 FSDP/TP/EP 分片，再通过 PyTorch DCP 逐张量广播到本地 shard。rank 0 仍需能容纳一份浮点源模型；其他 rank 不会先完整 CPU 加载。初始化完成后该源模型释放。

DeepSeek 使用实例级 RMSNorm/mHC/Sinkhorn 补丁，保留原生 attention 的 causal、sliding、CSA/HCA mask。只输出 top-k 整数索引的 indexer 在 SFT 中保持 pretrained 权重；没有伪造其梯度。

## 节点数与切分

以下是完整 BF16 full-SFT 的起跑配置，假定每节点为 16 个逻辑 die、每 die 64 GiB。它们是部署建议，完整模型跨物理节点的实验由集群执行；已完成的实验见 README。

| 模型 | 节点 | 总进程 | FSDP | TP | EP | EFSDP | 每 rank batch | 全局 batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-122B-A10B | 2 | 32 | 32 | 1 | 16 | 2 | 1 | 32 |
| DeepSeek-V4-Flash | 4 | 64 | 64 | 1 | 32 | 2 | 1 | 64 |

若每台只有 8 个逻辑 die，则对应使用 4 台和 8 台，保持总进程及切分不变。256 experts 可以整除上述 EP。Qwen 的现有 TP plan 只覆盖 full-attention q/k/v/o，首轮使用 TP1；DeepSeek 没有有效纯 TP plan，TP>1 会明确拒绝。FSDP/TP 与 EP/EFSDP 是重叠 mesh，不能把四个尺寸相乘当作 world size。

## 多机启动

每节点先 `npu-smi info` 确认空闲设备。配置节点间 TCP rendezvous、NPU 网络及 HCCL 所需端口；`MASTER_ADDR` 使用所有节点可达的主机地址。多网卡时按集群网络配置选择通信接口，先完成普通 torch_npu HCCL all-reduce 检查。检查设备可见数为 16。

在每台节点分别执行同一模型命令；只改变 `NODE_RANK`。下面的路径改为你的共享路径：

```bash
conda activate swift-fsdpturbo
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NPROC_PER_NODE=16
export MASTER_ADDR=<rank0节点IP>
export MASTER_PORT=29501
export NODE_RANK=<本节点编号，从0开始>
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200
```

rank 0 的完整 checkpoint 转换可能较久，因此扩大初始化等待；PyTorch 的进程组 timeout 也应大于这些 HCCL timeout。`HCCL_CONNECT_TIMEOUT` 的有效范围可查 [CANN 9.0 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0077.html)。

Qwen 两节点，`NODE_RANK=0,1`：

```bash
export NNODES=2
swift fsdpturbo sft \
  --model /shared/models/Qwen3.5-122B-A10B \
  --dataset /shared/data/train.jsonl --output_dir /shared/output/qwen122 \
  --tuner_type full --torch_dtype bfloat16 --attn_impl eager \
  --fsdp_size 32 --tp_size 1 --ep_size 16 --efsdp_size 2 \
  --fsdp_implementation native --ep_dispatcher eager \
  --forward_prefetch 0 --backward_prefetch 0 \
  --max_steps 1000 --max_length 1024 --learning_rate 1e-5 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --gradient_checkpointing true --vit_gradient_checkpointing true \
  --router_aux_loss_coef 0.001 \
  --save_strategy steps --save_steps 100 --eval_strategy no \
  --split_dataset_ratio 0 --report_to none --logging_steps 10 \
  --dataloader_num_workers 0 --enable_npu_model_patch false --add_version false
```

DeepSeek 四节点，`NODE_RANK=0,1,2,3`：

```bash
export NNODES=4
swift fsdpturbo sft \
  --model /shared/models/DeepSeek-V4-Flash \
  --dataset /shared/data/train.jsonl --output_dir /shared/output/deepseek-v4-flash \
  --tuner_type full --torch_dtype bfloat16 --attn_impl eager \
  --fsdp_size 64 --tp_size 1 --ep_size 32 --efsdp_size 2 \
  --fsdp_implementation native --ep_dispatcher eager \
  --forward_prefetch 0 --backward_prefetch 0 \
  --max_steps 1000 --max_length 1024 --learning_rate 1e-5 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --gradient_checkpointing true --vit_gradient_checkpointing true \
  --save_strategy steps --save_steps 100 --eval_strategy no \
  --split_dataset_ratio 0 --report_to none --logging_steps 10 \
  --dataloader_num_workers 0 --enable_npu_model_patch false --add_version false
```

CLI 会启动一次 torchrun；外层不再包另一层 torchrun。第一轮可把 `max_steps` 改为 3、`max_length` 改为 128、`save_steps` 改为 3，验证加载、拓扑、前后向、更新和保存后再扩大序列。长度及 batch 改动属于新的验证配置。

## 保存与恢复

保存格式是 PyTorch DCP，目录为 `output_dir/checkpoint-<step>`，包含分布式模型、优化器、scheduler、每 rank RNG 和数据游标。恢复使用相同的初始模型路径、数据、world size、切分、batch、seed 及 LR schedule，并在上述命令末尾追加：

```bash
--resume_from_checkpoint /shared/output/<run>/checkpoint-100
```

`max_steps` 是累计目标步数。游标使用 rank-local sampler 长度，跨 epoch 恢复会跳过已消费样本。修改并行布局会拒绝恢复，避免把旧游标套到新数据切片。不要使用 `resume_only_model` 或 `ignore_data_skip`。普通 HF 推理命令不能直接读取 DCP；导出 HF checkpoint 是独立转换步骤。

恢复时只读取 checkpoint 的 dtype metadata 来建立 meta 模型，直接分配本地 shard 并恢复 DCP；不会重新完整加载初始 HF 权重。

可加 `--offload_params true`；FSDP 分片参数、梯度和状态放在 CPU，严格 FP32 小参数及 routing/rotary buffers 保持在 NPU。CPU Gloo 和 NPU HCCL 双后端会在 HF 参数初始化前建立。

当前入口仍限定 full causal-LM SFT、GA1、map-style 数据、native FSDP 起跑配置；LoRA、PP、CP、packing/padding-free、multimodal batch 和改变拓扑恢复不在本指南的验证范围。
