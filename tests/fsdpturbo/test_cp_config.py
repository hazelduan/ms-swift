import importlib.util
from types import SimpleNamespace

import pytest
import torch

from swift.fsdpturbo.model_specs import get_model_spec


pytestmark = pytest.mark.skipif(importlib.util.find_spec('fsdp_turbo') is None,
                                reason='requires the optional FSDPTurbo package')


def make_args(cp_size):
    train = SimpleNamespace(
        bf16=True, fp16=False, torch_dtype=torch.bfloat16, weight_decay=0.1,
        adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8, learning_rate=1e-5,
        warmup_ratio=0, lr_scheduler_type='cosine', max_grad_norm=1, seed=42,
        max_steps=20, num_train_epochs=3, gradient_accumulation_steps=1,
        per_device_train_batch_size=1, dataloader_num_workers=0,
        dataloader_pin_memory=False, train_dataloader_shuffle=False,
        logging_steps=1, save_only_model=False, save_strategy='no', save_steps=20,
    )
    return SimpleNamespace(
        training_args=train, model='/local/model', torch_dtype=torch.bfloat16,
        dataset=['/local/data.jsonl'], max_length=256, output_dir='/local/output',
        resume_from_checkpoint=None, resume_only_model=False,
        fsdp_size=8, tp_size=1, cp_size=cp_size, ep_size=8, efsdp_size=1,
        forward_prefetch=0, backward_prefetch=0, offload_params=False,
        pin_memory=True, fsdp_implementation='native', ep_dispatcher='eager',
        gradient_checkpointing=True,
    )


def test_cp1_does_not_enable_model_or_loss_patches():
    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    config = build_fsdpturbo_config(make_args(1), get_model_spec('qwen3_5_moe'))
    assert config.distributed.ulysses_parallel_size == 1
    assert config.module_patches == []
    assert config.distributed.cp_plan.ulysses_function_patches == []
    assert config.distributed.cp_plan.loss_function_patches == []


def test_qwen_cp2_uses_backend_gdn_and_postfusion_interfaces():
    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    config = build_fsdpturbo_config(make_args(2), get_model_spec('qwen3_5_moe'))
    assert config.distributed.ulysses_parallel_size == 2
    assert {item['type'] for item in config.distributed.cp_plan.ulysses_function_patches} == {
        'full_attention', 'gated_delta_net'}
    assert config.distributed.cp_plan.loss_function_patches[0]['type'] == 'causal_lm_loss'
    assert config.module_patches[0]['replacement'].endswith('qwen3_5_moe_model_forward')


def test_unimplemented_deepseek_cp_is_not_silently_enabled():
    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    with pytest.raises(ValueError, match='CP is not implemented'):
        build_fsdpturbo_config(make_args(2), get_model_spec('deepseek_v4'))
