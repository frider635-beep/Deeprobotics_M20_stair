"""Stage7 -> shared path actor migration, provenance and curriculum persistence."""
from pathlib import Path
from types import MethodType
import hashlib
import torch
from ...models.stair_path_actor import VERSION, DESCENT_LIMITS
from ...models.ascent_unified_actor import baseline_digest, UNIFIED_LIMITS
from .ascent_unified_init import validate_source as validate_stage7
from .ascent_adaptive_init import attach_curriculum_checkpoint as attach_ascent

PATH_PROFILES = ('ascent_entry_v1_stage8_fixed', 'ascent_entry_v1_stage9')
PATH_VERSIONS = (VERSION, VERSION+'_reward_v2', VERSION+'_reward_v3_course_width',
                 'm20_stair_path_fixed247_20261023_v1_stage9_fixed_width_curriculum',
                 'm20_stair_path_fixed247_20261023_v2_staged_sampling')


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
    if manifest.get('m20_profile') not in PATH_PROFILES or manifest.get('refinement_version') not in PATH_VERSIONS:
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
    source_manifest = saved.get('m20_training_manifest') or {}
    source_columns = source_manifest.get('curriculum_terrain_columns', 10)
    old_types = torch.floor(torch.arange(old_count)/(old_count/source_columns)).long()
    select = torch.arange(env.num_envs, device=env.device)
    target_ascent_count = int(getattr(env.cfg, 'ascent_adaptive_ascent_columns', 8))
    select = select[terrain.terrain_types < target_ascent_count]
    # Stage8 retained the Stage7 ascent task columns. Stage9 changes both
    # width and tread-depth coverage, so only its wide reference task carries
    # compatible progress into the new course.
    terrain.terrain_levels[select] = 0
    source_profile = source_manifest.get('m20_profile')
    target_profile = getattr(env.cfg, 'm20_refinement_profile', None)
    if target_profile == 'ascent_entry_v1_stage8_fixed' and source_profile in (
            'ascent_entry_v1_stage7', 'ascent_entry_v1_stage8_fixed'):
        compatible_columns = tuple(range(min(8, source_columns)))
        for column in compatible_columns:
            target_ids = select[terrain.terrain_types[select] == column]
            source_ids = torch.where(old_types == column)[0]
            if not len(target_ids) or not len(source_ids):
                continue
            mapping = source_ids[torch.linspace(0, len(source_ids)-1, len(target_ids)).round().long()]
            state.skill[target_ids] = torch.maximum(old['skill'][mapping].to(env.device), state.minimum_skill[target_ids])
            terrain.terrain_levels[target_ids] = levels[mapping].to(env.device)
            if column == 2 and source_columns < 13:
                terrain.terrain_levels[target_ids] = 0
                state.skill[target_ids] = torch.maximum(
                    state.minimum_skill[target_ids], torch.full_like(state.skill[target_ids], 2))
    elif target_profile == 'ascent_entry_v1_stage9':
        source_reference_columns = ((8,) if source_profile == 'ascent_entry_v1_stage9'
                                    else (4, 6))
        target_reference_columns = tuple(getattr(env.cfg, 'ascent_adaptive_reference_columns', (8,)))
        if len(target_reference_columns) == 1:
            target_ids = select[terrain.terrain_types[select] == target_reference_columns[0]]
            source_mask = torch.zeros_like(old_types, dtype=torch.bool)
            for column in source_reference_columns:
                source_mask |= old_types == column
            source_ids = torch.where(source_mask)[0]
            if len(target_ids) and len(source_ids):
                mapping = source_ids[torch.linspace(0, len(source_ids)-1, len(target_ids)).round().long()]
                state.skill[target_ids] = torch.maximum(old['skill'][mapping].to(env.device), state.minimum_skill[target_ids])
                terrain.terrain_levels[target_ids] = levels[mapping].to(env.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.reset()
    return dict(restored=True, source_environments=old_count, destination_environments=env.num_envs,
                restored_fields=['reference_prefix_skill', 'reference_terrain_level'],
                new_width_depth_courses='reset_to_row0_and_minimum_prefix', outcome_windows='reset')


def attach_curriculum_checkpoint(algorithm, env):
    from ...tasks.manager_based.locomotion.velocity.mdp.path.stair_path_mdp import descent_course
    attach_ascent(algorithm, env)
    old_save, old_load = algorithm.save, algorithm.load
    def save(self):
        state = old_save(); state['stair_path_descent_curriculum'] = descent_course(env).state_dict()
        if getattr(env.cfg, 'm20_refinement_profile', None) == 'ascent_entry_v1_stage9':
            state['stair_path_terrain_types'] = env.scene.terrain.terrain_types.detach().cpu().clone()
            state['stair_path_sampling_stage'] = getattr(
                env, '_stair_path_sampling_stage', torch.full((env.num_envs,), -1, dtype=torch.long)).detach().cpu().clone()
        return state
    def load(self, loaded, load_cfg=None, strict=True):
        if load_cfg is None or load_cfg.get('iteration', False):
            if getattr(env.cfg, 'm20_refinement_profile', None) == 'ascent_entry_v1_stage9':
                env._stair_path_sampling_iteration_offset = int(loaded.get('iter', 0))
            columns = (loaded.get('m20_training_manifest') or {}).get('curriculum_terrain_columns', 10)
            if columns != env.scene.terrain.terrain_origins.shape[1]:
                raise ValueError('Full Stage8 resume requires the same terrain columns; use actor initialization to change the course')
            if 'stair_path_descent_curriculum' not in loaded:
                raise ValueError('Full Stage8 resume requires its descent curriculum')
            descent_course(env).load_state_dict(loaded['stair_path_descent_curriculum'])
            if getattr(env.cfg, 'm20_refinement_profile', None) == 'ascent_entry_v1_stage9':
                saved_types = loaded.get('stair_path_terrain_types')
                saved_stage = loaded.get('stair_path_sampling_stage')
                if saved_types is None or saved_types.shape != env.scene.terrain.terrain_types.shape or saved_types.dtype != torch.long:
                    raise ValueError('Full Stage9 resume requires its terrain sampling assignment')
                if saved_stage is None or saved_stage.shape != (env.num_envs,) or saved_stage.dtype != torch.long:
                    raise ValueError('Full Stage9 resume requires its terrain sampling phase')
                columns = env.scene.terrain.terrain_origins.shape[1]
                if (saved_types < 0).any() or (saved_types >= columns).any() or (saved_stage < -1).any() or (saved_stage > 2).any():
                    raise ValueError('Invalid Stage9 terrain sampling state')
                terrain = env.scene.terrain
                terrain.terrain_types.copy_(saved_types.to(env.device))
                terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
                env._stair_path_sampling_stage = saved_stage.to(env.device).clone()
        return old_load(loaded, load_cfg, strict)
    algorithm.save = MethodType(save, algorithm)
    algorithm.load = MethodType(load, algorithm)
