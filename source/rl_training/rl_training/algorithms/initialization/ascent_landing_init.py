"""Pure-Torch source checks for the separately versioned landing profile."""
from pathlib import Path
import hashlib
import torch
from .ascent_entry_stage2_init import validate_checkpoint


def frozen_source_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith('landing_mlp.') or name == 'landing_limits':
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update((name+str(tuple(value.shape))+str(value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_source(path, preserve=None):
    provenance = validate_checkpoint(path)
    saved = torch.load(Path(path).expanduser(), map_location='cpu', weights_only=True)
    if saved['m20_training_manifest'].get('m20_profile') != 'ascent_entry_v1_stage2':
        raise ValueError('Stage3 must initialize from a complete stage2 checkpoint')
    if preserve and Path(preserve).expanduser().resolve() != Path(path).expanduser().resolve():
        raise ValueError('Stage3 preserves its complete initialized stage2 source')
    return provenance
