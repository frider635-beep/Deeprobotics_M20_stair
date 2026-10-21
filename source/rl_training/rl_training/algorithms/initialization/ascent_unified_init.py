"""Native Stage7 checkpoint validation; old branches migrate offline, never drop."""
from pathlib import Path
import hashlib
import torch
from ...models.ascent_unified_actor import PROFILE, VERSION, UNIFIED_LIMITS, baseline_digest
from .ascent_entry_stage2_init import BASELINE_SHA256
from .ascent_adaptive_init import load_initialized_critic, attach_curriculum_checkpoint
from ...utils.module_layout import REPO_ROOT


def validate_source(path, preserve=None):
    path = Path(path).expanduser().resolve()
    saved = torch.load(path,map_location='cpu',weights_only=True)
    manifest = saved.get('m20_training_manifest') or {}
    if manifest.get('m20_profile') != PROFILE or manifest.get('refinement_version') != VERSION:
        raise ValueError('Stage7 requires a migrated unified checkpoint; run merge_ascent_branches.py first')
    state = saved['actor_state_dict']
    if any(k.startswith(('entry_mlp.','landing_mlp.','sequence_mlp.')) for k in state):
        raise ValueError('Stage7 checkpoint still contains the old ascent branches')
    if baseline_digest(state) != manifest.get('frozen_motion_state_sha256'):
        raise ValueError('Stage7 frozen motion baseline changed')
    original = REPO_ROOT/'pre_train/teacher_baseline_.pt'
    if hashlib.sha256(original.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError('Stage7 original baseline file changed')
    base = torch.load(original,map_location='cpu',weights_only=True)['actor_state_dict']
    if any(k not in state or not torch.equal(v,state[k]) for k,v in base.items()):
        raise ValueError('Stage7 original motion baseline is not preserved')
    expected = set(base) | {'unified_limits'}
    for layer,shape in ((0,(256,247)),(2,(128,256)),(4,(16,128))):
        for kind,size in (('weight',shape),('bias',(shape[0],))):
            name=f'unified_mlp.{layer}.{kind}';expected.add(name)
            if name not in state or tuple(state[name].shape)!=size:raise ValueError('Invalid unified shape: '+name)
    if set(state)!=expected or not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Stage7 incomplete or nonfinite actor')
    if not torch.equal(state['unified_limits'], torch.tensor(UNIFIED_LIMITS)):
        raise ValueError('Stage7 correction envelope changed')
    if not all(torch.isfinite(v).all() for v in saved['critic_state_dict'].values()):
        raise ValueError('Stage7 nonfinite critic')
    if preserve and Path(preserve).expanduser().resolve()!=path:
        raise ValueError('Stage7 preserves its own consolidated initialization')
    return dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                frozen_motion_state_sha256=manifest['frozen_motion_state_sha256'])
