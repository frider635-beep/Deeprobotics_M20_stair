"""Stage8: retain Stage7 ascent, share path feedback and learn straight descent."""
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
from .ascent_entry_profile import straight_stairs, reset_entry_root
from rl_training.control.stair_path import PathConfig, PATH_VERSION
from rl_training.models.stair_path_actor import PROFILE, VERSION, DESCENT_LIMITS, DESCENT_WARMUP, DESCENT_RAMP
from rl_training.tasks.manager_based.locomotion.velocity.mdp.path import stair_path_mdp as path
from rl_training.tasks.manager_based.locomotion.velocity.mdp.path.stair_path_objectives import REWARD_VERSION, REWARD_WEIGHTS
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_alternating_mdp as alternating
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_adaptive import HEIGHTS, REFERENCE_COLUMNS


# Eight ascent columns, one flat, four descending widths. Never change the
# <8 / ==8 / >=9 direction contract used by the existing contact observers.
PATH_DEPTHS = (.30,.30,.20,.25,.30,.35,.30,.35,.30,.20,.25,.30,.35)
TERRAIN_COLUMNS = len(PATH_DEPTHS)

def path_stairs(difficulty, cfg):
    row = int(np.clip(np.floor(float(difficulty)*len(HEIGHTS)), 0, len(HEIGHTS)-1))
    height = .15 if cfg.reference else max(HEIGHTS[row], cfg.minimum_height)
    if not cfg.descending:
        return straight_stairs(difficulty, cfg.replace(step_height=height))
    length, width = cfg.size; depth, count, start = cfg.tread_depth, cfg.step_count, 3.
    def box(x0, x1, z):
        return trimesh.creation.box((x1-x0, width, z+.1),
            trimesh.transformations.translation_matrix(((x0+x1)/2, width/2, (z-.1)/2)))
    meshes = [box(0., length, 0.), box(0., start, count*height)]
    for i in range(count):
        meshes.append(box(start+i*depth, start+(i+1)*depth, (count-1-i)*height))
    return meshes, np.array([start-1.25, width/2, count*height])


@configclass
class PathStairsCfg(AdaptiveStairsCfg):
    function = path_stairs
    descending: bool = False
    minimum_height: float = 0.


class PathVelocityCommand(AlternatingVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        down = ids[self._env.scene.terrain.terrain_types[ids] >= 9]
        self.entry_speeds[down] = torch.where(torch.rand(len(down), device=self.device) < .5, .5, .7)
        self._apply_ascent_command()

    def _apply_ascent_command(self):
        super()._apply_ascent_command()
        down = self._env.scene.terrain.terrain_types >= 9
        self.vel_command_b[down, 0] = self.entry_speeds[down]
        self.vel_command_b[down, 1:] = 0.
        self.is_standing_env[down] = False
        self.is_heading_env[down] = False


def reset_path_root(env, env_ids, pose_range, velocity_range, asset_cfg=SceneEntityCfg('robot')):
    reset_entry_root(env, env_ids, pose_range, velocity_range, asset_cfg)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[env.scene.terrain.terrain_types[ids] >= 9]
    asset = env.scene[asset_cfg.name]
    pose = asset.data.default_root_state[ids, :7].clone()
    pose[:, :3] += env.scene.env_origins[ids]
    pose[:, 0] += torch.empty(len(ids), device=env.device).uniform_(-.15, .15)
    pose[:, 1] += torch.empty(len(ids), device=env.device).uniform_(-.03, .03)
    yaw = torch.empty(len(ids), device=env.device).uniform_(-.025, .025)
    pose[:, 3:] = 0.; pose[:, 3], pose[:, 6] = torch.cos(yaw/2), torch.sin(yaw/2)
    asset.write_root_pose_to_sim(pose, env_ids=ids)
    asset.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


def apply_profile(env_cfg, agent_cfg, evaluation=False):
    if getattr(env_cfg, 'm20_refinement_profile', None) != PROFILE:
        previous.apply_profile(env_cfg, agent_cfg, evaluation=evaluation)
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_teacher
        stair_teacher.TUNING['stair_path_feedback'] = True
        env_cfg.stair_path_config = PathConfig().to_dict()
        env_cfg.stair_path_merge_entry_slip = True
        env_cfg.commands.base_velocity.class_type = PathVelocityCommand
        env_cfg.events.randomize_reset_base.func = reset_path_root
        terrains = {}
        for column in range(8):
            terrains[f'path_stairs_{column}'] = PathStairsCfg(proportion=1./TERRAIN_COLUMNS, tread_depth=PATH_DEPTHS[column],
                reference=column in REFERENCE_COLUMNS, minimum_height=.15 if column == 3 else 0.)
        terrains['flat'] = MeshPlaneTerrainCfg(proportion=1./TERRAIN_COLUMNS)
        for label, depth in (('compact',.20),('narrow',.25),('reference',.30),('wide',.35)):
            terrains['straight_descent_'+label] = PathStairsCfg(proportion=1./TERRAIN_COLUMNS, tread_depth=depth, descending=True)
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(size=(12., 5.), num_rows=len(HEIGHTS),
            num_cols=TERRAIN_COLUMNS, curriculum=True, difficulty_range=(0., 1.), use_cache=False, sub_terrains=terrains)
        env_cfg.ascent_adaptive_height_grid = [[.15 if c in REFERENCE_COLUMNS else max(h, .15) if c == 3 else h
                                               for c in range(TERRAIN_COLUMNS)] for h in HEIGHTS]
        # Depth remains fixed in each column on EVERY row: failure no longer
        # converts a narrow stair into an easier, wide-tread task.
        env_cfg.ascent_adaptive_depth_grid = [list(PATH_DEPTHS) for _ in HEIGHTS]
        env_cfg.rewards.ascent_alternating_lateral.weight = 0.
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
        env_cfg.m20_refinement_profile = PROFILE
        alternating.gate_positive_rewards(env_cfg.rewards)
    if evaluation:
        env_cfg.ascent_adaptive_training = False
        for terrain in env_cfg.scene.terrain.terrain_generator.sub_terrains.values():
            if isinstance(terrain, PathStairsCfg):
                terrain.reference = True; terrain.tread_depth = .30
        env_cfg.ascent_adaptive_height_grid = [[.15]*TERRAIN_COLUMNS for _ in HEIGHTS]
        env_cfg.ascent_adaptive_depth_grid = [[.30]*TERRAIN_COLUMNS for _ in HEIGHTS]
    manifest = configure_agent(agent_cfg)
    manifest['gated_positive_rewards'] = alternating.audit_positive_rewards(env_cfg.rewards)
    return manifest


def configure_agent(agent_cfg):
    manifest = previous.configure_agent(agent_cfg)
    agent_cfg.algorithm.learning_rate = 1.e-5
    agent_cfg.algorithm.anchor_coefficient = .15
    agent_cfg.algorithm.critic_warmup_iterations = 30
    manifest.update(profile=PROFILE, version=VERSION+'_reward_v2', control_profile='legacy_v1_path_v1',
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
        geometry='fixed_depth_per_column_narrow_min_height_015',
        terrain_columns=TERRAIN_COLUMNS, ascent_depths_m=[.20,.25,.30,.35],
        descent_course=dict(fraction=4./TERRAIN_COLUMNS, shape='straight_18_treads', speeds=[.5,.7], depths_m=[.20,.25,.30,.35],
            promote='both_speeds_6_of_8',demote_consecutive_failures=3, success_max_path_error_m=.20))
    return manifest
