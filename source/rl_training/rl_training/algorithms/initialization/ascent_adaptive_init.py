"""Validate stage4 warm start without mutating baseline or widening authority."""
from pathlib import Path
from ...utils.module_layout import REPO_ROOT
import hashlib
import torch
from .ascent_sequence_init import frozen_source_digest
from .ascent_landing_init import frozen_source_digest as stage2_digest
from .ascent_entry_stage2_init import BASELINE_SHA256
from ...models.ascent_sequence_actor import SEQUENCE_LIMITS


def validate_source(path, preserve=None):
    path = Path(path).expanduser().resolve()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    manifest = saved.get('m20_training_manifest') or {}
    profile = manifest.get('m20_profile')
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if file_hash == BASELINE_SHA256:
        if preserve and Path(preserve).expanduser().resolve() != path:
            raise ValueError('Stage5 preserves its initialized baseline')
        return dict(checkpoint=str(path), sha256=file_hash, source='original_baseline')
    if profile == 'ascent_entry_v1_stage3':
        from .ascent_sequence_init import validate_source as validate_stage3
        return validate_stage3(path, preserve)
    if profile not in ('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
        raise ValueError('Stage5 initializes from original baseline, complete stage3/4/5; no cross-profile resume')
    state = saved['actor_state_dict']
    if manifest.get('frozen_full_baseline_sha256') != BASELINE_SHA256:
        raise ValueError('Stage5 baseline provenance mismatch')
    if frozen_source_digest(state) != manifest.get('frozen_stage3_state_sha256'):
        raise ValueError('Stage5 frozen stage3 source changed')
    inherited = {k: v for k, v in state.items() if not k.startswith('sequence_mlp.') and k != 'sequence_limits'}
    if stage2_digest(inherited) != manifest.get('frozen_stage2_state_sha256'):
        raise ValueError('Stage5 frozen stage2 source changed')
    baseline_path = REPO_ROOT/'pre_train/teacher_baseline_.pt'
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError('Stage5 original baseline file changed')
    baseline = torch.load(baseline_path, map_location='cpu', weights_only=True)['actor_state_dict']
    if any(k not in state or not torch.equal(v, state[k]) for k, v in baseline.items()):
        raise ValueError('Stage5 inherited baseline parameter changed')
    expected = set(baseline) | {'entry_limits', 'landing_limits', 'sequence_limits'}
    for branch in ('entry_mlp', 'landing_mlp', 'sequence_mlp'):
        for layer, shape in [(0, (128, 247)), (2, (64, 128)), (4, (16, 64))]:
            for kind, size in [('weight', shape), ('bias', (shape[0],))]:
                name = f'{branch}.{layer}.{kind}'; expected.add(name)
                if name not in state or tuple(state[name].shape) != size:
                    raise ValueError('Stage5 source shape mismatch: '+name)
    if set(state) != expected or not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Stage5 incomplete or nonfinite actor source')
    if not torch.equal(state['sequence_limits'], torch.tensor(SEQUENCE_LIMITS)):
        raise ValueError('Stage5 source correction limits changed')
    if not all(torch.isfinite(v).all() for v in saved['critic_state_dict'].values()):
        raise ValueError('Stage5 nonfinite critic source')
    if preserve and Path(preserve).expanduser().resolve() != path:
        raise ValueError('Stage5 uses its complete initialized stage4 policy as reference')
    return dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                frozen_stage3_state_sha256=manifest['frozen_stage3_state_sha256'])


def load_initialized_actor(actor, saved):
    profile = (saved.get('m20_training_manifest') or {}).get('m20_profile')
    state = saved['actor_state_dict']
    if profile in ('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
        actor.load_state_dict(state, strict=True)
    elif profile == 'ascent_entry_v1_stage3':
        actor.load_stage3_checkpoint(state)
    else:
        # validate_source has already required the exact archived baseline.
        result = actor.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not (k.startswith(('entry_mlp.', 'landing_mlp.', 'sequence_mlp.')) or
                k in ('entry_limits', 'landing_limits', 'sequence_limits')) for k in result.missing_keys):
            raise ValueError('Unexpected baseline initialization keys')


def load_initialized_critic(critic, state):
    """Keep the old value function initially; zero-initialize 12 new inputs."""
    expected = critic.state_dict()
    adapted = dict(state)
    old = adapted['mlp.0.weight']; new = expected['mlp.0.weight']
    if old.shape != new.shape:
        if old.shape[0] != new.shape[0] or old.shape[1]+12 != new.shape[1]:
            raise ValueError('Unexpected stage5 critic initialization dimensions')
        expanded = torch.zeros_like(new)
        expanded[:, :old.shape[1]] = old.to(expanded.device)
        adapted['mlp.0.weight'] = expanded
    critic.load_state_dict(adapted, strict=True)


def attach_curriculum_checkpoint(algorithm, env):
    """Keep skill/terrain progress in normal PPO checkpoints, including resume."""
    from types import MethodType
    from ...tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_adaptive_mdp import curriculum_state
    original_save, original_load = algorithm.save, algorithm.load
    def save(self):
        state = original_save()
        state['ascent_adaptive_curriculum'] = curriculum_state(env).state_dict()
        state['ascent_adaptive_terrain_levels'] = env.scene.terrain.terrain_levels.detach().cpu().clone()
        return state
    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = original_load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get('iteration', False):
            if 'ascent_adaptive_curriculum' not in loaded_dict:
                raise ValueError('Stage5 resume requires curriculum state; use actor initialization for stage4')
            curriculum_state(env).load_state_dict(loaded_dict['ascent_adaptive_curriculum'])
            levels = loaded_dict['ascent_adaptive_terrain_levels']
            terrain = env.scene.terrain
            if levels.shape != terrain.terrain_levels.shape or levels.dtype != torch.long:
                raise ValueError('Stage5 full resume requires the same num_envs; use fresh actor initialization to change it')
            if (levels < 0).any() or (levels >= len(env.cfg.ascent_adaptive_height_grid)).any():
                raise ValueError('Stage5 terrain row out of range')
            terrain.terrain_levels.copy_(levels.to(env.device))
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            # Existing physical episodes were initialized on row zero. Reset
            # them on the restored rows before collecting any new rollout.
            env.reset()
        return result
    algorithm.save = MethodType(save, algorithm)
    algorithm.load = MethodType(load, algorithm)
