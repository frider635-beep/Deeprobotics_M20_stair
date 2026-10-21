"""Pure-Torch provenance checks before warm-starting a complete v1 actor."""
from pathlib import Path
from ...utils.module_layout import REPO_ROOT
import hashlib
import torch

BASELINE_SHA256 = 'db341fd21914d7b5846ea110992332d66b41a224b38ae4d032d5d665f30f2512'


def validate_checkpoint(path, baseline_path=None):
    path = Path(path).expanduser().resolve()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    manifest = saved.get('m20_training_manifest') or {}
    if manifest.get('m20_profile') not in ('ascent_entry_v1', 'ascent_entry_v1_stage2'):
        raise ValueError('Stage2 initialization requires a complete v1 ascent checkpoint')
    if manifest.get('frozen_full_baseline_sha256') != BASELINE_SHA256:
        raise ValueError('Stage2 source baseline provenance differs')
    baseline_path = Path(baseline_path) if baseline_path else REPO_ROOT/'pre_train/teacher_baseline_.pt'
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError('Original motion baseline SHA256 differs')
    original = torch.load(baseline_path, map_location='cpu', weights_only=True)['actor_state_dict']
    state = saved['actor_state_dict']
    for name, value in original.items():
        if name not in state or not torch.equal(value, state[name]):
            raise ValueError('Frozen original policy changed: '+name)
    expected = set(original) | {'entry_limits'} | {f'entry_mlp.{i}.{kind}' for i in (0, 2, 4) for kind in ('weight', 'bias')}
    if set(state) != expected or not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Incomplete or nonfinite v1 actor')
    limits = torch.tensor([.12, .24, .24]*4+[.12]*4)
    if not torch.equal(state['entry_limits'], limits):
        raise ValueError('Stage2 keeps the original v1 correction limits')
    shapes = {'entry_mlp.0.weight': (128, 247), 'entry_mlp.2.weight': (64, 128),
              'entry_mlp.4.weight': (16, 64), 'entry_mlp.0.bias': (128,),
              'entry_mlp.2.bias': (64,), 'entry_mlp.4.bias': (16,)}
    if any(tuple(state[name].shape) != shape for name, shape in shapes.items()):
        raise ValueError('Stage2 keeps the original v1 actor architecture')
    return {'checkpoint': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_profile': manifest['m20_profile'], 'frozen_full_baseline_exact': True}
