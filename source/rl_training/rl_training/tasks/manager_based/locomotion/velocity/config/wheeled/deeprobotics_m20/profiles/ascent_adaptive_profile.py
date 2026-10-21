"""Isolated stage5: successful entry contacts before geometry generalization."""
import numpy as np
from isaaclab.managers import RewardTermCfg, CurriculumTermCfg, TerminationTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
from isaaclab.utils import configclass
from . import ascent_sequence_profile as previous
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_adaptive_mdp as adaptive
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_adaptive import HEIGHTS, DEPTHS, REFERENCE_COLUMNS, PREFIXES, DEADLINES, depth_for_row
from .ascent_entry_profile import StraightStairsCfg, straight_stairs
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_reward_audit import audit_conflicting_rewards

PROFILE = 'ascent_entry_v1_stage5'
VERSION = 'm20_ascent_adaptive_20261021_v1_stage5'


def adaptive_stairs(difficulty, cfg):
    # Isaac randomizes difficulty inside each row. Quantize to that same row
    # so reward metadata describes the actual mesh, not the row's midpoint.
    row = int(np.clip(np.floor(float(difficulty)*len(HEIGHTS)), 0, len(HEIGHTS)-1))
    height = .15 if cfg.reference else HEIGHTS[row]
    depth = .30 if cfg.reference else depth_for_row(row, cfg.tread_depth)
    return straight_stairs(difficulty, cfg.replace(step_height=height, tread_depth=depth))


@configclass
class AdaptiveStairsCfg(StraightStairsCfg):
    function = adaptive_stairs
    reference: bool = False


@configclass
class AscentCriticControlCfg(ObservationGroupCfg):
    state = ObservationTermCfg(func=adaptive.critic_state)
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def apply_profile(env_cfg, agent_cfg, evaluation=False):
    if getattr(env_cfg, 'm20_refinement_profile', None) != PROFILE:
        previous.apply_profile(env_cfg, agent_cfg)
        descent = env_cfg.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs_inv']
        terrains = {}
        for column in range(8):
            terrains[f'adaptive_stairs_{column}'] = AdaptiveStairsCfg(
                proportion=.1, tread_depth=DEPTHS[column], reference=column in REFERENCE_COLUMNS)
        terrains['flat'] = MeshPlaneTerrainCfg(proportion=.1)
        terrains['pyramid_stairs_inv'] = descent.replace(proportion=.1)
        # 18*35cm stairs plus approach and platform fit without overlap.
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(12., 5.), num_rows=len(HEIGHTS), num_cols=10, curriculum=True,
            difficulty_range=(0., 1.), use_cache=False, sub_terrains=terrains)
        env_cfg.scene.terrain.max_init_terrain_level = 0
        env_cfg.ascent_adaptive_geometry = True
        env_cfg.ascent_adaptive_height_grid = [[.15 if c in REFERENCE_COLUMNS else h for c in range(10)] for h in HEIGHTS]
        env_cfg.ascent_adaptive_depth_grid = [[depth_for_row(row, d) for d in DEPTHS] for row in range(len(HEIGHTS))]
        env_cfg.ascent_adaptive_training = True
        env_cfg.ascent_sequence_training_ramp = False
        env_cfg.commands.base_velocity.class_type = previous.SequenceVelocityCommand
        env_cfg.observations.ascent_control = AscentCriticControlCfg()
        for name in vars(env_cfg.rewards):
            term = getattr(env_cfg.rewards, name)
            if name.startswith('ascent_sequence_') and term is not None: term.weight = 0.
        env_cfg.rewards.ascent_entry_landing.weight = 0.
        env_cfg.rewards.ascent_entry_clearance.weight = 0.
        env_cfg.rewards.ascent_stage2_stall.weight = 0.
        weights = dict(ordered=.15, missing=-.90, dwell=.05, paired=1., failure=-2., early=-.35,
            cadence=-.02, landing_credit=.08, prepare=-.10, height=-1.20, horizontal=-.25,
            velocity=-.12, balance=-.12, surge=-.20, smoothness=-.10,
            top_front=.30, top_rear=.50, top_forward=.80)
        for name, weight in weights.items():
            setattr(env_cfg.rewards, 'ascent_adaptive_'+name,
                    RewardTermCfg(func=adaptive.reward, weight=weight, params={'name': name}))
        env_cfg.rewards.stair_orientation_cost.func = adaptive.orientation
        env_cfg.rewards.stair_ascent_heading_cost.weight = -5.
        env_cfg.rewards.stair_ascent_straight_motion_cost.weight = -3.
        env_cfg.curriculum.ascent_sequence = None
        env_cfg.curriculum.ascent_adaptive_metrics = CurriculumTermCfg(func=adaptive.metrics)
        env_cfg.curriculum.ascent_adaptive = CurriculumTermCfg(func=adaptive.curriculum)
        env_cfg.terminations.ascent_entry_finished = TerminationTermCfg(func=adaptive.finished, time_out=False)
        env_cfg.terminations.ascent_adaptive_failed = TerminationTermCfg(func=adaptive.failed, time_out=False)
        env_cfg.episode_length_s = 24.
        env_cfg.m20_refinement_profile = PROFILE
    if evaluation:
        # Full tasks at default geometry in Isaac play. Variable-size MuJoCo
        # evaluations are explicit and do not use this training configuration.
        env_cfg.ascent_adaptive_training = False
        gen = env_cfg.scene.terrain.terrain_generator
        for cfg in gen.sub_terrains.values():
            if isinstance(cfg, AdaptiveStairsCfg): cfg.reference = True; cfg.tread_depth = .30
        env_cfg.ascent_adaptive_height_grid = [[.15]*10 for _ in HEIGHTS]
        env_cfg.ascent_adaptive_depth_grid = [[.30]*10 for _ in HEIGHTS]
    audit_conflicting_rewards(env_cfg.rewards)
    return configure_agent(agent_cfg)


def configure_agent(agent_cfg):
    manifest = previous.configure_agent(agent_cfg)
    agent_cfg.algorithm.learning_rate = 3.e-5
    agent_cfg.algorithm.anchor_stair_weight = 0.
    agent_cfg.algorithm.motion_regularization_coefficient = .01
    manifest.update(profile=PROFILE, version=VERSION,
        initialization='validated_baseline_stage3_or_stage4_fresh_optimizer_train_sequence_only',
        geometry={'height_rows_m': list(HEIGHTS), 'depth_columns_m': list(DEPTHS),
                  'reference_columns': list(REFERENCE_COLUMNS), 'reference_m': [.15, .30],
                  'depth_expansion': 'row0_30cm_row1_27p5_to_32p5cm_row2plus_25_to_35cm',
                  'mesh_and_rewards_same_quantized_row': True},
        contact_curriculum={'prefixes': list(PREFIXES), 'promote_consecutive_successes': 8,
                            'demote_consecutive_failures': 3, 'geometry_after_full_flight': True,
                            'new_geometry_prefix': 4,
                            'entry_speeds_m_s': [.5, .7], 'global_slowdown': False},
        termination={'miss_or_early_advance_or_simultaneous': 'true_terminal',
                     'prefix_success': 'true_terminal', 'late_positive_credit_after_failure': False},
        prefix_active_deadlines_s=list(DEADLINES),
        preparation='per_wheel_next_target_without_mate_ready_delay',
        landing_credit='inside_tread_best_error_escrow_paid_only_on_stable_contact',
        actor_inputs='policy244_plus_raw_command3_no_privileged_sequence_state',
        critic_inputs='critic285_plus_raw_command3_plus_ascent_control12',
        adaptive_exploration_action_std=[.015, .12, .12]*2+[.01, .04, .04]*2+[.04]*4)
    manifest['sequence'].update(approach_credit='inside_tread_escrow_paid_only_on_stable_contact',
                                mate_miss_counts_as_success=False, strict_prefix_failure=True)
    manifest['training_entry_speed'] = dict(schedule='fixed_default_speeds', speeds_m_s=[.5, .7],
                                             evaluation_ramp=False)
    manifest['contact_demonstration_schedule'] = dict(coefficient=.5, decay_updates=800)
    return manifest
