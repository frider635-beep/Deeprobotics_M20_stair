"""Stage9: stabilize narrow-stair courses and retain shared path feedback."""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
import torch
import trimesh
from isaaclab.managers import RewardTermCfg, CurriculumTermCfg, TerminationTermCfg, SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
from isaaclab.utils import configclass
from . import ascent_unified_profile as previous
from .ascent_alternating_profile import AlternatingVelocityCommand
from .ascent_adaptive_profile import AdaptiveStairsCfg
from .ascent_entry_profile import reset_entry_root
from rl_training.control.stair_path import PathConfig, PATH_VERSION
from rl_training.models.stair_path_actor import DESCENT_LIMITS, DESCENT_WARMUP, DESCENT_RAMP
from rl_training.tasks.manager_based.locomotion.velocity.mdp.path import stair_path_mdp as path
from rl_training.tasks.manager_based.locomotion.velocity.mdp.path import stair_path_layout as layout
from rl_training.tasks.manager_based.locomotion.velocity.mdp.path.stair_path_readiness import ReadinessConfig
from rl_training.tasks.manager_based.locomotion.velocity.mdp.path import stair_path_course as progressive
from rl_training.tasks.manager_based.locomotion.velocity.mdp.path.rectangular_random_grid import random_grid_terrain
from rl_training.tasks.manager_based.locomotion.velocity.mdp.path.stair_path_objectives import REWARD_VERSION, REWARD_WEIGHTS
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_alternating_mdp as alternating
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_entry_mdp as entry
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_adaptive as adaptive
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_pairing as pairing
from rl_training.tasks.manager_based.locomotion.velocity.mdp.path.stair_path_evidence import SETTINGS as EVIDENCE_SETTINGS
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_adaptive import geometry
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_adaptive import HEIGHTS


PROFILE = 'ascent_entry_v1_stage9'
VERSION = pairing.VERSION
STAGE8_PROFILE = 'ascent_entry_v1_stage8_fixed'
STAGE8_VERSION = 'm20_stair_path_fixed247_20261022_v1_stage8_reward_v2'
STAGE8_DEPTHS = (.30,.30,.20,.25,.30,.35,.30,.35,.30,.20,.25,.30,.35)
STAGE8_REFERENCE_COLUMNS = (4, 6)
ASCENT_REFERENCE_COLUMN = len(layout.ASCENT_TASKS)
PATH_DEPTHS = tuple(depth for depth, _ in layout.ASCENT_TASKS) + (layout.REFERENCE_TASK[0],) + tuple(
    [.30]*layout.FLAT_COPIES) + tuple(depth for depth, _ in layout.DESCENT_TASKS) + tuple(
    [.30]*len(layout.GENERAL_COLUMNS))
TERRAIN_COLUMNS = layout.TERRAIN_COLUMNS
if len(PATH_DEPTHS) != TERRAIN_COLUMNS:
    raise RuntimeError('Stage9 geometry table does not cover every terrain column')


def _narrow_ascent_mask(env):
    """Limit extra riser protection to the trained 20/25 cm ascent columns."""
    _, depth = geometry(env)
    ascent = adaptive.ascent_mask(env)
    return ascent & (depth <= .25 + 1.e-6)


def narrow_ascent_clearance_cost(env):
    """Restore the wheel-over-riser clearance signal on narrow treads only."""
    return entry.clearance_cost(env) * _narrow_ascent_mask(env)


def narrow_ascent_riser_contact_cost(env):
    """Make front-edge impacts visible on narrow ascent columns only."""
    return entry.riser_contact_cost(env) * _narrow_ascent_mask(env)


def path_stairs(difficulty, cfg):
    row = int(np.clip(np.floor(float(difficulty)*len(HEIGHTS)), 0, len(HEIGHTS)-1))
    height = .15 if cfg.reference else max(HEIGHTS[row], cfg.minimum_height)
    length, tile_width = cfg.size
    depth, count, start = cfg.tread_depth, cfg.step_count, 3.
    course_width = tile_width if cfg.reference else cfg.course_width_m

    def box(x0, x1, z, width):
        return trimesh.creation.box((x1-x0, width, z+.1),
            trimesh.transformations.translation_matrix(((x0+x1)/2, tile_width/2, (z-.1)/2)))

    meshes = [box(0., length, 0., tile_width)]
    if cfg.descending:
        meshes.append(box(0., start, count*height, course_width))
        for i in range(count):
            meshes.append(box(start+i*depth, start+(i+1)*depth,
                              (count-1-i)*height, course_width))
    else:
        for i in range(count):
            meshes.append(box(start+i*depth, start+(i+1)*depth,
                              (i+1)*height, course_width))
        meshes.append(box(start+count*depth, length, count*height, course_width))
    return meshes, np.array([start-1.25, tile_width/2, 0. if not cfg.descending else count*height])


@configclass
class PathStairsCfg(AdaptiveStairsCfg):
    function = path_stairs
    descending: bool = False
    minimum_height: float = 0.
    course_width_m: float = 1.0


class PathVelocityCommand(AlternatingVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        down = ids[adaptive.descent_mask(self._env)[ids]]
        self.entry_speeds[down] = torch.where(torch.rand(len(down), device=self.device) < .5, .5, .7)
        self._apply_ascent_command()

    def _apply_ascent_command(self):
        super()._apply_ascent_command()
        down = adaptive.descent_mask(self._env)
        self.vel_command_b[down, 0] = self.entry_speeds[down]
        self.vel_command_b[down, 1:] = 0.
        self.is_standing_env[down] = False
        self.is_heading_env[down] = False


def reset_path_root(env, env_ids, pose_range, velocity_range, asset_cfg=SceneEntityCfg('robot')):
    reset_entry_root(env, env_ids, pose_range, velocity_range, asset_cfg)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[adaptive.descent_mask(env)[ids]]
    asset = env.scene[asset_cfg.name]
    pose = asset.data.default_root_state[ids, :7].clone()
    pose[:, :3] += env.scene.env_origins[ids]
    pose[:, 0] += torch.empty(len(ids), device=env.device).uniform_(-.15, .15)
    pose[:, 1] += torch.empty(len(ids), device=env.device).uniform_(-.03, .03)
    yaw = torch.empty(len(ids), device=env.device).uniform_(-.025, .025)
    pose[:, 3:] = 0.; pose[:, 3], pose[:, 6] = torch.cos(yaw/2), torch.sin(yaw/2)
    asset.write_root_pose_to_sim(pose, env_ids=ids)
    asset.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


def apply_profile(env_cfg, agent_cfg, evaluation=False, profile=PROFILE):
    if profile not in (STAGE8_PROFILE, PROFILE):
        raise ValueError(f'Unknown stair path profile: {profile}')
    if getattr(env_cfg, 'm20_refinement_profile', None) != profile:
        if profile == PROFILE:
            preserved_terrains = getattr(env_cfg, '_m20_general_terrain_sources', None)
            if preserved_terrains is None:
                preserved_terrains = env_cfg.scene.terrain.terrain_generator.sub_terrains
            try:
                preserved_terrains = layout.preserve_general_terrains(preserved_terrains)
            except RuntimeError as exc:
                raise RuntimeError(str(exc) + '; initialize the v1 task before applying the Stage9 profile') from exc
        previous.apply_profile(env_cfg, agent_cfg, evaluation=evaluation)
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_teacher
        stair_teacher.TUNING['stair_path_feedback'] = True
        env_cfg.stair_path_config = PathConfig().to_dict()
        env_cfg.stair_path_merge_entry_slip = True
        env_cfg.commands.base_velocity.class_type = PathVelocityCommand
        env_cfg.events.randomize_reset_base.func = reset_path_root
        if profile == PROFILE:
            terrains = {}
            proportion = 1./TERRAIN_COLUMNS
            for column, (depth, width) in enumerate(layout.ASCENT_TASKS):
                terrains[f'path_ascent_{column}'] = PathStairsCfg(proportion=proportion,
                    tread_depth=depth, course_width_m=width)
            terrains['path_ascent_reference'] = PathStairsCfg(proportion=proportion,
                tread_depth=layout.REFERENCE_TASK[0], reference=True)
            for replica, _ in enumerate(layout.FLAT_COLUMNS):
                terrains[f'flat_{replica}'] = MeshPlaneTerrainCfg(proportion=proportion)
            for column, (depth, width) in enumerate(layout.DESCENT_TASKS):
                terrains[f'path_descent_{column}'] = PathStairsCfg(proportion=proportion,
                    tread_depth=depth, descending=True, course_width_m=width)
            for name, columns in layout.GENERAL_COLUMN_GROUPS.items():
                for replica, _ in enumerate(columns):
                    terrain_cfg = deepcopy(preserved_terrains[name])
                    if name == 'boxes':
                        terrain_cfg.function = random_grid_terrain
                    terrain_cfg.proportion = proportion
                    terrains[f'{name}_{replica}'] = terrain_cfg
            env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(size=(12., 5.), num_rows=len(HEIGHTS),
                num_cols=TERRAIN_COLUMNS, curriculum=True, difficulty_range=(0., 1.), use_cache=False, sub_terrains=terrains)
            env_cfg.ascent_adaptive_ascent_columns = layout.ASCENT_COLUMNS
            env_cfg.ascent_adaptive_ascent_column_ids = tuple(range(layout.ASCENT_COLUMNS))
            env_cfg.ascent_adaptive_reference_columns = (layout.ASCENT_REFERENCE_COLUMN,)
            env_cfg.stair_path_flat_column = layout.FLAT_COLUMN
            env_cfg.stair_path_flat_columns = layout.FLAT_COLUMNS
            env_cfg.stair_path_descent_start = layout.DESCENT_START
            env_cfg.stair_path_descent_columns = layout.DESCENT_COLUMNS
            env_cfg.stair_path_general_columns = layout.GENERAL_COLUMNS
            env_cfg.ascent_adaptive_height_grid = [[.15 if c == ASCENT_REFERENCE_COLUMN else h
                                                   for c in range(TERRAIN_COLUMNS)] for h in HEIGHTS]
            # Each column keeps its width and tread depth at every height row.
            env_cfg.ascent_adaptive_depth_grid = [list(PATH_DEPTHS) for _ in HEIGHTS]
            env_cfg.stair_path_sampling_steps_per_iteration = agent_cfg.num_steps_per_env
        else:
            terrains = {}
            proportion = 1./len(STAGE8_DEPTHS)
            for column in range(8):
                terrains[f'path_stairs_{column}'] = PathStairsCfg(proportion=proportion,
                    tread_depth=STAGE8_DEPTHS[column], reference=column in STAGE8_REFERENCE_COLUMNS,
                    minimum_height=.15 if column == 3 else 0., course_width_m=5.0)
            terrains['flat'] = MeshPlaneTerrainCfg(proportion=proportion)
            for label, depth in (('compact',.20),('narrow',.25),('reference',.30),('wide',.35)):
                terrains['straight_descent_'+label] = PathStairsCfg(proportion=proportion,
                    tread_depth=depth, descending=True, course_width_m=5.0)
            env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(size=(12., 5.), num_rows=len(HEIGHTS),
                num_cols=len(STAGE8_DEPTHS), curriculum=True, difficulty_range=(0., 1.), use_cache=False,
                sub_terrains=terrains)
            env_cfg.ascent_adaptive_ascent_columns = 8
            env_cfg.ascent_adaptive_ascent_column_ids = tuple(range(8))
            env_cfg.ascent_adaptive_reference_columns = STAGE8_REFERENCE_COLUMNS
            env_cfg.stair_path_flat_column = 8
            env_cfg.stair_path_flat_columns = (8,)
            env_cfg.stair_path_descent_start = 9
            env_cfg.stair_path_descent_columns = tuple(range(9, 13))
            env_cfg.stair_path_general_columns = ()
            env_cfg.ascent_adaptive_height_grid = [[.15 if c in STAGE8_REFERENCE_COLUMNS else max(h,.15) if c == 3 else h
                                                   for c in range(len(STAGE8_DEPTHS))] for h in HEIGHTS]
            env_cfg.ascent_adaptive_depth_grid = [list(STAGE8_DEPTHS) for _ in HEIGHTS]
        env_cfg.rewards.ascent_alternating_lateral.weight = 0.
        # Replace spatial next-edge clearance with the same pending tread used
        # by landing/hold guidance. No descent or path objective is changed.
        if profile == PROFILE:
            env_cfg.stair_ascent_pairing_v9 = True
            env_cfg.rewards.ascent_entry_clearance.func = pairing.reward
            env_cfg.rewards.ascent_entry_clearance.params = {'name':'clearance'}
            env_cfg.rewards.ascent_entry_clearance.weight = -.30
            env_cfg.rewards.ascent_riser_contact.func = narrow_ascent_riser_contact_cost
            env_cfg.rewards.ascent_riser_contact.weight = -.30
            env_cfg.rewards.ascent_rear_paired = RewardTermCfg(func=pairing.reward,
                weight=pairing.REAR_PAIR_WEIGHT,params={'name':'rear_paired'})
            env_cfg.rewards.ascent_landing_order = RewardTermCfg(func=pairing.reward,
                weight=pairing.LANDING_ORDER_WEIGHT,params={'name':'landing_order'})
            env_cfg.rewards.ascent_rear_phase = RewardTermCfg(func=pairing.reward,
                weight=pairing.REAR_PHASE_WEIGHT,params={'name':'rear_phase'})
        # The retained continuous ascent terms own entry + full-flight slip;
        # the common route objective owns descent lateral position/velocity.
        env_cfg.rewards.ascent_stage2_front_slip.weight = 0.
        env_cfg.rewards.ascent_stage2_rear_slip.weight = 0.
        env_cfg.rewards.stair_descent_lateral_cost.weight = 0.
        for name, weight in REWARD_WEIGHTS.items():
            setattr(env_cfg.rewards, 'stair_path_'+name, RewardTermCfg(func=path.reward, weight=weight, params={'name':name}))
        env_cfg.terminations.stair_path_failed = TerminationTermCfg(func=path.failed)
        env_cfg.terminations.stair_path_descent_finished = TerminationTermCfg(func=path.descent_finished)
        env_cfg.curriculum.stair_path_descent = CurriculumTermCfg(func=path.curriculum)
        env_cfg.curriculum.stair_path = CurriculumTermCfg(func=path.metrics)
        if profile == PROFILE:
            env_cfg.curriculum.terrain_levels = CurriculumTermCfg(
                func=path.general_terrain_curriculum, params={'min_commanded_translation_m': .1})
        env_cfg.m20_refinement_profile = profile
        alternating.gate_positive_rewards(env_cfg.rewards)
    if evaluation:
        env_cfg.ascent_adaptive_training = False
        for terrain in env_cfg.scene.terrain.terrain_generator.sub_terrains.values():
            if isinstance(terrain, PathStairsCfg):
                terrain.reference = True; terrain.tread_depth = .30
        columns = TERRAIN_COLUMNS if profile == PROFILE else len(STAGE8_DEPTHS)
        env_cfg.ascent_adaptive_height_grid = [[.15]*columns for _ in HEIGHTS]
        env_cfg.ascent_adaptive_depth_grid = [[.30]*columns for _ in HEIGHTS]
    manifest = configure_agent(agent_cfg, profile=profile)
    manifest['gated_positive_rewards'] = alternating.audit_positive_rewards(env_cfg.rewards)
    if profile == PROFILE:
        manifest.update(contact_curriculum=dict(prefixes=list(progressive.PREFIXES),
            minimum_task_fractions={'2':.5,'3':.3,'4':.2},
            promote='both_speeds_physical_6_of_8_and_graded_per_axle_pair_quality',
            strict_success='final_full_episode_gait_acceptance',
            demote_consecutive_failures=3,demotion_window='current_speed_8_trials_physical_success_at_most_half',
            prefix_quality_thresholds=deepcopy(EVIDENCE_SETTINGS['prefix_pair_quality']),
            height_quality_threshold=EVIDENCE_SETTINGS['height_pair_quality'],
            height_rear_second_quality=EVIDENCE_SETTINGS['height_rear_second_quality'],
            demotion='shorten_prefix_at_same_height_first_never_below_certified_prefix',
            mastery='per_environment_geometry_and_height_both_speeds_6_of_8_retained_on_failure',
            mastered_floor_height_fallback_failures=progressive.ProgressiveCourse.mastered_height_failure_episodes,
            unmastered_minimum_height_fallback_failures=progressive.ProgressiveCourse.unmastered_height_failure_episodes,
            height_change_promotion_hold_eligible_episodes=progressive.ProgressiveCourse.consolidation_episodes,
            failures_can_demote_during_hold=True,geometry_probe_prefix=6,
            new_height_prefix=4,demotion_retains_width_and_depth=True),
            prefix_active_deadlines_s=list(progressive.DEADLINES),
            course_diagnostics='100_iteration_matched_geometry_row_original_prefix_speed_jsonl_and_tensorboard',
            ppo_diagnostics='approx_kl_clip_fraction_grad_norms_batch_explained_variance')
    return manifest


def configure_agent(agent_cfg, profile=PROFILE):
    manifest = previous.configure_agent(agent_cfg)
    stage9 = profile == PROFILE
    agent_cfg.algorithm.learning_rate = 1.5e-5 if stage9 else 1.e-5
    agent_cfg.algorithm.anchor_coefficient = .15
    agent_cfg.algorithm.critic_warmup_iterations = 30
    if stage9:
        agent_cfg.max_iterations = layout.SAMPLING_TOTAL_ITERATIONS
    profile_version = VERSION if stage9 else STAGE8_VERSION
    columns = TERRAIN_COLUMNS if stage9 else len(STAGE8_DEPTHS)
    reference_columns = [ASCENT_REFERENCE_COLUMN] if stage9 else list(STAGE8_REFERENCE_COLUMNS)
    ascent_tasks = ([dict(depth_m=depth, width_m=width) for depth, width in layout.ASCENT_TASKS]
                    if stage9 else [dict(depth_m=depth, width_m=5.0) for depth in STAGE8_DEPTHS[:8]])
    descent_tasks = ([dict(depth_m=depth, width_m=width) for depth, width in layout.DESCENT_TASKS]
                     if stage9 else [dict(depth_m=depth, width_m=5.0) for depth in STAGE8_DEPTHS[9:]])
    course_widths = list(layout.TASK_WIDTHS_M) if stage9 else [5.0]
    manifest.update(profile=profile, version=profile_version, control_profile='legacy_v1_path_v1',
        initialization='complete_stage7_exact_network_state_fresh_optimizer',
        actor_inputs='policy244_plus_raw_command3',
        critic_inputs='critic285_plus_raw_command3_plus_ascent_control12',
        stair_path_version=PATH_VERSION, stair_path_config=PathConfig().to_dict(),
        stair_path_reward_version=REWARD_VERSION, stair_path_reward_weights=REWARD_WEIGHTS.copy(),
        path_tracking_target='effective_xy_only_during_path_feedback_otherwise_raw',
        path_velocity_target='smooth_bounded_inward_return_then_zero_near_route',
        contact_slip='per_support_wheel_lateral015_both_plus_half_longitudinal020_descent',
        merged_entry_slip='max_entry_and_transition_emphasis_capped_1p5',
        descent_limits_action_units=DESCENT_LIMITS,
        descent_authority_schedule=dict(warmup_iterations=DESCENT_WARMUP, ramp_iterations=DESCENT_RAMP),
        descent_initialization='same_unified_mlp_fixed_bounded_envelope_schedule_30_warmup_800_ramp',
        preservation='default_ascent_1_narrow025_015_compact020_005_descent_002',
        course_width_curriculum=dict(widths_m=course_widths,
            pattern='fixed_column_per_tread_depth_and_width' if stage9 else 'legacy_fixed_wide_stairs',
            ascent_columns=ascent_tasks, descent_columns=descent_tasks,
            reference_columns=reference_columns, reference_width_m=5.0,
            outer_region='flat_ground_to_match_MuJoCo_scan_and_edge_support' if stage9 else None),
        narrow_ascent_contact_protection=(dict(depths_m=[.20,.25], clearance_weight=-.30,
            riser_contact_weight=-.30, scope='ascent_only') if stage9 else None),
        staged_sampling=(dict(total_iterations=layout.SAMPLING_TOTAL_ITERATIONS,
            advancement='manual_by_default_optional_matched_strata_plateau_and_automatic_demotion',
            iteration_boundaries=[], budget_forces_advancement=False,
            transition_iterations=ReadinessConfig().transition_iterations,
            readiness=asdict(ReadinessConfig()),
            phases=[dict(zip(('easy','middle','hard'), shares)) for shares in layout.READINESS_GROUP_SHARES],
            easy_columns=['30cm/1.2m','35cm/1.2m','5m reference'],
            middle_columns=['25cm/1.2m','30cm/1.0m','35cm/1.0m'],
            hard_columns=['20cm/1.0m','20cm/1.2m','25cm/1.0m'],
            ascent_fraction=round(layout.ASCENT_FRACTION,4),
            descent_fraction=round(layout.DESCENT_FRACTION,4),
            non_stair_fraction=round(layout.NON_STAIR_FRACTION,4),
            non_stair_and_descent_absolute_samples_preserved_from_8192=True,
            resample='minimal_surplus_to_deficit_ascent_transfers_on_reset',
            fixed_descent_and_non_stair_assignments=True,
            unchanged_geometry_progress_preserved=True,
            changed_geometry_progress='minimum_pinned_prefix_and_fresh_both_speed_windows') if stage9 else None),
        reward_adjustment=('ascent_only_rear_phase_guidance_and_physical_graded_pair_course' if stage9 else None),
        ascent_pairing=(deepcopy(pairing.SETTINGS) if stage9 else None),
        course_evidence=(deepcopy(EVIDENCE_SETTINGS) if stage9 else None),
        geometry=('fixed_width_and_depth_per_column_lowest_row_for_new_width_tasks' if stage9
                  else 'legacy_fixed_depth_per_column'),
        terrain_columns=columns, ascent_depths_m=[.20,.25,.30,.35],
        ascent_course=(dict(reference_column=ASCENT_REFERENCE_COLUMN, reference_width_m=5.0,
            task_widths_m=[1.0,1.2], demotion_retains_width_and_depth=True) if stage9 else
            dict(reference_columns=list(STAGE8_REFERENCE_COLUMNS), course_width_m=5.0)),
        descent_course=dict(fraction=(layout.DESCENT_FRACTION if stage9 else 4./len(STAGE8_DEPTHS)),
            shape='straight_18_treads', speeds=[.5,.7], depths_m=[.20,.25,.30,.35],
            promote='both_speeds_6_of_8',demote_consecutive_failures=3, success_max_path_error_m=.20),
        mixed_terrain=(dict(flat_columns=list(layout.FLAT_COLUMNS),
            general_columns={name:list(columns) for name,columns in layout.GENERAL_COLUMN_GROUPS.items()},
            general_curriculum='distance_based_with_original_velocity_command_and_rewards',
            path_feedback_scope='ascent_and_descent_columns_only',
            terrain_grid=dict(rows=len(HEIGHTS),columns=TERRAIN_COLUMNS,tile_size_m=[12.,5.],
                environments_per_column_at_10584=round(10584/TERRAIN_COLUMNS)))
            if stage9 else dict(
                flat_columns=[8], general_columns={}, general_curriculum='legacy_stage8',
                path_feedback_scope='legacy_stage8_stair_columns_only',
                terrain_grid=dict(rows=len(HEIGHTS),columns=len(STAGE8_DEPTHS),tile_size_m=[12.,5.]))))
    return manifest
