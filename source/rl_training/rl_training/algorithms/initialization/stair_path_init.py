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
                 'm20_stair_path_fixed247_20261023_v2_staged_sampling',
                 'm20_stair_path_fixed247_20261023_v3_sampling_40k',
                 'm20_stair_path_fixed247_20261026_v4_readiness_sampling',
                 'm20_stair_path_fixed247_20261026_v5_progressive_course',
                 'm20_stair_path_fixed247_20261027_v6_mastered_prefix_floor',
                 'm20_stair_path_fixed247_20261027_v7_ascent_pairing',
                 'm20_stair_path_fixed247_20261027_v8_soft_pairing',
                 'm20_stair_path_fixed247_20261027_v9_course_evidence')


def validate_ascent_pairing_resume(prior,target,weights,migrate=False,source='v8'):
    """Explicit v8 -> v9; all old weights/path/descent contracts stay exact."""
    from ...tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_pairing as pairing
    from ...tasks.manager_based.locomotion.velocity.mdp.path.stair_path_evidence import SETTINGS
    expected_version=pairing.PREVIOUS_VERSION if migrate and source == 'v8' else pairing.VERSION
    if ((migrate and source != 'v8') or prior.get('m20_profile') != 'ascent_entry_v1_stage9'
            or prior.get('refinement_version') != expected_version or target.get('version') != pairing.VERSION):
        raise ValueError('v9 requires explicit --stair_ascent_migrate_v8 from v8, or native v9 resume')
    settings={key:target[key] for key in (
        'stair_path_reward_weights','path_tracking_target','path_velocity_target','contact_slip','merged_entry_slip')}
    if prior.get('stair_path_reward_settings') != settings:
        raise ValueError('v9 cannot change shared path/descent reward settings')
    for key in ('stair_path_reward_version','stair_path_config','control_profile',
                'descent_limits_action_units','descent_authority_schedule'):
        if prior.get(key) != target[key]:raise ValueError('v9 must retain '+key)
    expected=dict(prior.get('reward_weights') or {})
    if not expected:raise ValueError('Source must record all reward weights')
    if migrate:
        if prior.get('ascent_pairing') != pairing.V8_SETTINGS:
            raise ValueError('Source v8 pairing contract mismatch')
        if 'ascent_rear_phase' in expected:raise ValueError('v8 unexpectedly contains rear phase reward')
        expected['ascent_rear_phase']=pairing.REAR_PHASE_WEIGHT
    elif prior.get('ascent_pairing') != pairing.SETTINGS or prior.get('course_evidence') != SETTINGS:
        raise ValueError('Native v9 pairing/course evidence settings mismatch')
    if (weights != expected or target.get('ascent_pairing') != pairing.SETTINGS
            or target.get('course_evidence') != SETTINGS):
        raise ValueError('v9 permits only ascent_rear_phase -0.15; all existing weights must match')


def validate_course_migration(prior, target, source='v4'):
    from ...tasks.manager_based.locomotion.velocity.mdp.path.stair_path_course import VERSION, LEGACY_VERSION
    from ...tasks.manager_based.locomotion.velocity.mdp.path.stair_path_layout import SAMPLING_VERSION
    if (prior.get('m20_profile') != 'ascent_entry_v1_stage9'
            or prior.get('refinement_version') != {'v4':SAMPLING_VERSION,'v5':LEGACY_VERSION}.get(source)
            or target.get('version') != VERSION):
        raise ValueError('Course migration requires the matching v4/v5 source flag and Stage9 v6 target')
    settings = {key:target[key] for key in (
        'stair_path_reward_weights','path_tracking_target','path_velocity_target','contact_slip','merged_entry_slip')}
    if (prior.get('stair_path_reward_settings') != settings
            or prior.get('stair_path_reward_version') != target['stair_path_reward_version']
            or prior.get('stair_path_config') != target['stair_path_config']
            or prior.get('control_profile') != target['control_profile']):
        raise ValueError('Course migration must retain rewards and command feedback')


def validate_sampling_only_migration(prior, profile_manifest):
    """Keep full optimizer resume restricted to this sampling-only revision."""
    from ...tasks.manager_based.locomotion.velocity.mdp.path import stair_path_layout as layout
    if (prior.get('m20_profile') != 'ascent_entry_v1_stage9'
            or prior.get('refinement_version') != layout.LEGACY_SAMPLING_VERSION
            or profile_manifest.get('version') != layout.SAMPLING_VERSION):
        raise ValueError('Sampling migration accepts only Stage9 v3 -> readiness v4')
    settings = {key:profile_manifest[key] for key in (
        'stair_path_reward_weights','path_tracking_target','path_velocity_target','contact_slip','merged_entry_slip')}
    if prior.get('stair_path_reward_settings') != settings or prior.get('stair_path_reward_version') != profile_manifest['stair_path_reward_version']:
        raise ValueError('Sampling-only migration cannot change path rewards')
    if prior.get('stair_path_config') != profile_manifest['stair_path_config'] or prior.get('control_profile') != profile_manifest['control_profile']:
        raise ValueError('Sampling-only migration cannot change command feedback')


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
    from ...tasks.manager_based.locomotion.velocity.mdp.path.stair_path_mdp import descent_course, _sampling_stage, readiness_sampler
    from ...tasks.manager_based.locomotion.velocity.mdp.path import stair_path_layout as layout
    from ...tasks.manager_based.locomotion.velocity.mdp.path.stair_path_readiness import ReadinessSampler, validate_legacy_migration
    attach_ascent(algorithm, env)
    old_save, old_load = algorithm.save, algorithm.load
    def save(self):
        state = old_save(); state['stair_path_descent_curriculum'] = descent_course(env).state_dict()
        if getattr(env.cfg, 'm20_refinement_profile', None) == 'ascent_entry_v1_stage9':
            state['stair_path_terrain_types'] = env.scene.terrain.terrain_types.detach().cpu().clone()
            state['stair_path_sampling'] = dict(version=layout.SAMPLING_VERSION,
                initialized=bool(getattr(env, '_stair_path_sampling_initialized', False)),
                iteration=_sampling_stage(env)[1],readiness=readiness_sampler(env).state_dict())
        return state
    def load(self, loaded, load_cfg=None, strict=True):
        if load_cfg is None or load_cfg.get('iteration', False):
            columns = (loaded.get('m20_training_manifest') or {}).get('curriculum_terrain_columns', 10)
            if columns != env.scene.terrain.terrain_origins.shape[1]:
                raise ValueError('Full Stage8 resume requires the same terrain columns; use actor initialization to change the course')
            if 'stair_path_descent_curriculum' not in loaded:
                raise ValueError('Full Stage8 resume requires its descent curriculum')
            descent_course(env).load_state_dict(loaded['stair_path_descent_curriculum'])
            if getattr(env.cfg, 'm20_refinement_profile', None) == 'ascent_entry_v1_stage9':
                from ...tasks.manager_based.locomotion.velocity.mdp.path.stair_path_course import VERSION as COURSE_VERSION, LEGACY_VERSION
                from ...tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_pairing as pairing
                prior_version = (loaded.get('m20_training_manifest') or {}).get('refinement_version')
                pairing_enabled = getattr(env.cfg,'stair_ascent_pairing_v9',False)
                target_version = pairing.VERSION if pairing_enabled else COURSE_VERSION
                migrating = prior_version != target_version
                if pairing_enabled:
                    from_v8 = prior_version == pairing.PREVIOUS_VERSION and getattr(env.cfg,'stair_ascent_migrate_v8',False)
                    if migrating and not from_v8:
                        raise ValueError('v9 requires --stair_ascent_migrate_v8 from a v8 checkpoint')
                    if migrating:
                        loaded=dict(loaded)
                        loaded['ascent_adaptive_curriculum']=env._ascent_adaptive_curriculum.migrate_v8(
                            loaded['ascent_adaptive_curriculum'])
                        env._stair_ascent_migration=dict(source_version=prior_version,target_version=target_version,
                            changed='ascent_graded_pair_quality_physical_windows_rear_phase_guidance',
                            retained='actor_critic_optimizer_reference_iteration_mastery_prefix_height_geometry_sampling_ramp',
                            evidence='fresh_physical_and_pair_quality_windows_v8_strict_windows_not_reused',
                            descent='all_rewards_control_and_curriculum_unchanged')
                elif migrating:
                    from_v4 = prior_version == layout.SAMPLING_VERSION and getattr(env.cfg,'stair_course_migrate_v4',False)
                    from_v5 = prior_version == LEGACY_VERSION and getattr(env.cfg,'stair_course_migrate_v5',False)
                    if not (from_v4 or from_v5):
                        raise ValueError('v6 requires --stair_course_migrate_v5 from v5, or --stair_course_migrate_v4 from v4')
                    course = env._ascent_adaptive_curriculum
                    loaded = dict(loaded)
                    migrate = course.migrate_bridge if from_v4 else course.migrate_v5
                    loaded['ascent_adaptive_curriculum'] = migrate(loaded['ascent_adaptive_curriculum'],
                        loaded['ascent_adaptive_terrain_levels'])
                    env._stair_path_course_migration = dict(source_version=prior_version,target_version=COURSE_VERSION,
                        prefix_mapping='actual_target_tread_count_unchanged',
                        mastery_initialization='current_both_speed_windows_or_conservative_4_6_tread_trajectory_evidence',
                        retained='actor_critic_optimizer_reference_rows_assignments_speed_windows_sampling_ramp',
                        readiness_history='new_evidence_required')
                saved_types = loaded.get('stair_path_terrain_types')
                saved_sampling = loaded.get('stair_path_sampling') or {}
                if saved_types is None or saved_types.shape != env.scene.terrain.terrain_types.shape or saved_types.dtype != torch.long:
                    raise ValueError('Full Stage9 resume requires its terrain sampling assignment')
                if (type(saved_sampling.get('initialized')) is not bool
                        or type(saved_sampling.get('iteration')) is not int
                        or saved_sampling['iteration'] < 0):
                    raise ValueError('Full Stage9 resume requires valid sampling state')
                columns = env.scene.terrain.terrain_origins.shape[1]
                if (saved_types < 0).any() or (saved_types >= columns).any():
                    raise ValueError('Invalid Stage9 terrain sampling state')
                if saved_sampling.get('version') == layout.SAMPLING_VERSION:
                    readiness = ReadinessSampler(env.device)
                    readiness.load_state_dict(saved_sampling.get('readiness') or {},migrate_v4=migrating)
                    if migrating:
                        readiness.reset_evidence()
                else:
                    iteration = validate_legacy_migration(saved_sampling,saved_types,
                        bool(getattr(env.cfg,'stair_path_sampling_migrate_v3',False)))
                    readiness = ReadinessSampler(env.device,start_iteration=iteration,
                        mode=getattr(env.cfg,'stair_path_sampling_mode',None) or 'manual')
                    env._stair_path_sampling_migration = dict(source_version=saved_sampling['version'],
                        source_iteration=iteration,target_version=layout.SAMPLING_VERSION,
                        retained='actor_critic_optimizer_reference_all_environment_curricula_and_assignments',
                        readiness_history='fresh_observations_required')
                readiness.request(mode=getattr(env.cfg,'stair_path_sampling_mode',None),
                    stage=getattr(env.cfg,'stair_path_sampling_stage',None),iteration=saved_sampling['iteration'])
                env._stair_path_readiness = readiness
                terrain = env.scene.terrain
                terrain.terrain_types.copy_(saved_types.to(env.device))
                env._stair_path_sampling_population = None
                terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
                env._stair_path_sampling_initialized = saved_sampling['initialized']
                steps = max(int(getattr(env.cfg, 'stair_path_sampling_steps_per_iteration', 24)), 1)
                env._stair_path_sampling_iteration_offset = saved_sampling['iteration']-int(env.common_step_counter)//steps
        env._stair_path_sampling_restoring = True
        try:
            return old_load(loaded, load_cfg, strict)
        finally:
            env._stair_path_sampling_restoring = False
    algorithm.save = MethodType(save, algorithm)
    algorithm.load = MethodType(load, algorithm)
