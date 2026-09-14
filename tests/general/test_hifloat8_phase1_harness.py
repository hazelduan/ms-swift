import importlib.util
import json
from pathlib import Path
import sys


HARNESS = (
    Path(__file__).resolve().parents[2]
    / 'examples'
    / 'ascend'
    / 'train'
    / 'hifloat8'
)


def _load_script(name):
    path = HARNESS / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'_hifloat8_{name}', path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HARNESS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_module_census_prefers_structured_artifact(tmp_path):
    summarize = _load_script('summarize_phase1')
    output = tmp_path / 'accuracy' / 'full_hifloat8' / 'output'
    output.mkdir(parents=True)
    names = [
        f'model.layers.{layer}.mlp.{projection}'
        for layer in range(28)
        for projection in ('gate_proj', 'up_proj', 'down_proj')
    ]
    modules = []
    for name in names:
        down = name.endswith('down_proj')
        in_features, out_features = ((3072, 1024) if down else (1024, 3072))
        modules.append({
            'name': name,
            'in_features': in_features,
            'out_features': out_features,
            'weight_numel': in_features * out_features,
            'weight_dtype': 'torch.bfloat16',
            'weight_requires_grad': True,
        })
    (output / 'hifloat8_module_census.json').write_text(
        json.dumps({
            'count': 84,
            'matrix_elements': 264241152,
            'names': names,
            'modules': modules,
        }),
        encoding='utf-8',
    )

    result = summarize.module_census(tmp_path, 'full')

    assert result['pass']
    assert result['suffix_counts'] == {
        '.mlp.gate_proj': 28,
        '.mlp.up_proj': 28,
        '.mlp.down_proj': 28,
    }
    assert result['forbidden'] == []
    assert result['module_details_pass']


def test_checkpoint_comparison_flattens_loss_scaler_state():
    compare = _load_script('compare_checkpoints')

    class LossScaler:
        def __init__(self):
            self.cur_scale = 1.0
            self.dynamic = False

    LossScaler.__module__ = 'deepspeed.runtime.fp16.loss_scaler'
    left, right = LossScaler(), LossScaler()

    result = compare.compare_state(
        {'optimizer_state_dict': {'loss_scaler': left}},
        {'optimizer_state_dict': {'loss_scaler': right}},
        1e-4,
    )

    assert result['pass']
    assert result['nonfloating_or_shape_mismatches'] == []


def test_profile_keeps_timer_and_profiler_callbacks():
    script = (HARNESS / 'run_phase1.sh').read_text(encoding='utf-8')
    profile_body = script.split('profile() {', 1)[1].split('\n}', 1)[0]

    assert '"$HERE/step_timer.py" "$HERE/npu_profile.py"' in profile_body
    assert 'hifloat8_step_timer hifloat8_profiler' in profile_body
