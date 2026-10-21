"""CPU-only validation of complete, finite stage3 warm starts."""
from pathlib import Path
from ...utils.module_layout import REPO_ROOT
import hashlib
import torch


def frozen_source_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith('sequence_mlp.') or name == 'sequence_limits':
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update((name+str(tuple(value.shape))+str(value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_source(path, preserve=None):
    from .ascent_landing_init import frozen_source_digest as stage2_digest
    from .ascent_entry_stage2_init import BASELINE_SHA256
    path = Path(path).expanduser().resolve()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    manifest = saved.get('m20_training_manifest') or {}
    if manifest.get('m20_profile') != 'ascent_entry_v1_stage3':
        raise ValueError('Stage4 requires a complete stage3 checkpoint')
    if manifest.get('frozen_full_baseline_sha256') != BASELINE_SHA256:
        raise ValueError('Stage4 source baseline provenance differs')
    state = saved['actor_state_dict']
    baseline_path = REPO_ROOT/'pre_train/teacher_baseline_.pt'
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError('Original baseline SHA256 differs')
    baseline = torch.load(baseline_path, map_location='cpu', weights_only=True)['actor_state_dict']
    for name, value in baseline.items():
        if name not in state or not torch.equal(value, state[name]):
            raise ValueError('Frozen baseline changed: '+name)
    if stage2_digest(state) != manifest.get('frozen_stage2_state_sha256'):
        raise ValueError('Stage3 source frozen stage2 state differs')
    expected = set(baseline) | {'entry_limits', 'landing_limits'}
    for branch in ('entry_mlp', 'landing_mlp'):
        for layer, shape in [(0, (128, 247)), (2, (64, 128)), (4, (16, 64))]:
            for kind, size in [('weight', shape), ('bias', (shape[0],))]:
                name = f'{branch}.{layer}.{kind}'; expected.add(name)
                if name not in state or tuple(state[name].shape) != size:
                    raise ValueError('Invalid source actor shape: '+name)
    if set(state) != expected or not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Incomplete or nonfinite stage3 source')
    if not torch.equal(state['entry_limits'], torch.tensor([.12, .24, .24]*4+[.12]*4)):
        raise ValueError('Unexpected entry limits')
    if not torch.equal(state['landing_limits'], torch.tensor([.16, 1.92, 1.92]*2+[.12, .48, .48]*2+[.12]*4)):
        raise ValueError('Unexpected landing limits')
    if preserve and Path(preserve).expanduser().resolve() != path:
        raise ValueError('Stage4 preserves its complete initialized source')
    return {'checkpoint': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_profile': manifest['m20_profile'], 'frozen_source_state_sha256': frozen_source_digest(state)}
