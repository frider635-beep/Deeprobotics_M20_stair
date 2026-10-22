"""Stage7 -> shared path actor migration, provenance and curriculum persistence."""
from pathlib import Path
from types import MethodType
import hashlib
import torch
from ...models.stair_path_actor import PROFILE, VERSION, DESCENT_LIMITS
from ...models.ascent_unified_actor import baseline_digest, UNIFIED_LIMITS
from .ascent_unified_init import validate_source as validate_stage7
from .ascent_adaptive_init import attach_curriculum_checkpoint as attach_ascent


def frozen_digest(state):
    return baseline_digest(state)


def validate_source(path, preserve=None):
    path = Path(path).expanduser().resolve()
    if preserve and Path(preserve).expanduser().resolve() != path:
        raise ValueError('Stage8 preserves its complete initialized policy')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    manifest = saved.get('m20_training_manifest') or {}
    if manifest.get('m20_profile') == 'ascent_entry_v1_stage7':
        return validate_stage7(path, preserve)
    if manifest.get('m20_profile') != PROFILE or manifest.get('refinement_version') not in (VERSION, VERSION+'_reward_v2'):
        raise ValueError('Stage8 requires a complete Stage7 or Stage8 checkpoint')
    state = saved['actor_state_dict']
    if frozen_digest(state) != manifest.get('frozen_motion_state_sha256'):
        raise ValueError('Stage8 frozen motion source changed')
    source = manifest.get('path_migration') or {}
    if not source.get('source_sha256') or source.get('source_profile') != 'ascent_entry_v1_stage7':
        raise ValueError('Stage8 migration provenance is missing')
    # Reuse the complete Stage7 baseline/envelope audit with identical
    # state keys, shapes and parameter counts; no adaptation is permitted.
    expected = {'unified_limits'}
    for prefix, shapes in {
        'mlp': [(512,244),(256,512),(128,256),(16,128)],
        'residual_mlp': [(128,247),(64,128),(16,64)],
        'unified_mlp': [(256,247),(128,256),(16,128)],
    }.items():
        for index, shape in enumerate(shapes):
            for kind, size in (('weight',shape),('bias',(shape[0],))):
                key = f'{prefix}.{2*index}.{kind}'; expected.add(key)
                if key not in state or tuple(state[key].shape) != size:
                    raise ValueError('Stage8 actor shape mismatch: '+key)
    expected.add('distribution.log_std_param')
    if set(state) != expected or not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Stage8 actor keys/nonfinite state mismatch')
    if not torch.equal(state['unified_limits'], torch.tensor(UNIFIED_LIMITS)):
        raise ValueError('Stage8 ascent envelope changed')
    from ...utils.module_layout import REPO_ROOT
    baseline = torch.load(REPO_ROOT/'pre_train/teacher_baseline_.pt', map_location='cpu', weights_only=True)['actor_state_dict']
    if any(key not in state or not torch.equal(value, state[key]) for key, value in baseline.items()):
        raise ValueError('Stage8 original motion baseline changed')
    if tuple(saved['critic_state_dict']['mlp.0.weight'].shape) != (512, 300):
        raise ValueError('Fixed Stage8 requires the original 300D critic')
    if not all(torch.isfinite(value).all() for value in saved['critic_state_dict'].values()):
        raise ValueError('Stage8 critic contains nonfinite parameters')
    return dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                frozen_motion_state_sha256=manifest['frozen_motion_state_sha256'])


def load_actor(actor, saved):
    if (saved.get('m20_training_manifest') or {}).get('m20_profile') == 'ascent_entry_v1_stage7':
        actor.load_stage7_checkpoint(saved['actor_state_dict'])
    else:
        actor.load_state_dict(saved['actor_state_dict'], strict=True)


def load_critic(critic, state):
    critic.load_state_dict(state, strict=True)


def seed_curriculum(env, saved):
    """Retain learned prefix/row progress, discard outcome windows after changes.

    Also supports a smaller smoke/evaluation run by sampling within each source
    terrain column, rather than assigning ascent skill to the flat/descent ids.
    """
    from ...tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_adaptive_mdp import curriculum_state
    old = saved.get('ascent_adaptive_curriculum'); levels = saved.get('ascent_adaptive_terrain_levels')
    if old is None or levels is None:
        return dict(restored=False, reason='source_has_no_environment_curriculum')
    state = curriculum_state(env); terrain = env.scene.terrain
    old_count = len(levels)
    source_columns = (saved.get('m20_training_manifest') or {}).get('curriculum_terrain_columns', 10)
    old_types = torch.floor(torch.arange(old_count)/(old_count/source_columns)).long()
    select = torch.arange(env.num_envs, device=env.device)
    select = select[terrain.terrain_types < 8]
    for column in range(8):
        ids = select[terrain.terrain_types[select] == column]
        source_ids = torch.where(old_types == column)[0]
        if not len(ids) or not len(source_ids): continue
        mapping = source_ids[torch.linspace(0, len(source_ids)-1, len(ids)).round().long()]
        skill = old['skill'][mapping].to(env.device)
        state.skill[ids] = torch.maximum(skill, state.minimum_skill[ids])
        terrain.terrain_levels[ids] = levels[mapping].to(env.device)
        if column == 2 and source_columns < 13:
            # This column previously contained 30 cm stairs. New 20 cm
            # treads start at 10 cm rise and a real four-tread contact task.
            terrain.terrain_levels[ids] = 0
            state.skill[ids] = torch.maximum(state.minimum_skill[ids], torch.full_like(state.skill[ids], 2))
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.reset()
    return dict(restored=True, source_environments=old_count, destination_environments=env.num_envs,
                restored_fields=['ascent_prefix_skill', 'ascent_terrain_level'], outcome_windows='reset',
                new_20cm_ascent_restart=(dict(row=0, height_m=.10, prefix=4) if source_columns < 13 else None))


def attach_curriculum_checkpoint(algorithm, env):
    from ...tasks.manager_based.locomotion.velocity.mdp.path.stair_path_mdp import descent_course
    attach_ascent(algorithm, env)
    old_save, old_load = algorithm.save, algorithm.load
    def save(self):
        state = old_save(); state['stair_path_descent_curriculum'] = descent_course(env).state_dict()
        return state
    def load(self, loaded, load_cfg=None, strict=True):
        if load_cfg is None or load_cfg.get('iteration', False):
            columns = (loaded.get('m20_training_manifest') or {}).get('curriculum_terrain_columns', 10)
            if columns != env.scene.terrain.terrain_origins.shape[1]:
                raise ValueError('Full Stage8 resume requires the same terrain columns; use actor initialization to change the course')
            if 'stair_path_descent_curriculum' not in loaded:
                raise ValueError('Full Stage8 resume requires its descent curriculum')
            descent_course(env).load_state_dict(loaded['stair_path_descent_curriculum'])
        return old_load(loaded, load_cfg, strict)
    algorithm.save = MethodType(save, algorithm)
    algorithm.load = MethodType(load, algorithm)
