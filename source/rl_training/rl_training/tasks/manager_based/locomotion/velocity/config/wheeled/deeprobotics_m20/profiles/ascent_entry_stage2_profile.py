"""Explicit second-stage profile under the existing StairTeacher-v1 task."""
from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg, CurriculumTermCfg, SceneEntityCfg
from . import ascent_entry_profile as original
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_entry_stage2 as objectives

PROFILE = 'ascent_entry_v1_stage2'
VERSION = 'm20_ascent_entry_20261020_v1_stage2'


class Stage2VelocityCommand(original.EntryVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        types = self._env.scene.terrain.terrain_types[ids]
        self.entry_speeds[ids[types < 2]] = .5
        self.entry_speeds[ids[(types >= 2) & (types < 4)]] = .7
        self._apply_ascent_command()


def reset_stage2_root(env, env_ids, pose_range, velocity_range, asset_cfg=SceneEntityCfg('robot')):
    original.reset_entry_root(env, env_ids, pose_range, velocity_range, asset_cfg)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[env.scene.terrain.terrain_types[ids] < 4]
    asset = env.scene[asset_cfg.name]
    pose = asset.data.root_state_w[ids, :7].clone()
    # Two columns per hard speed, retaining the original regular-entry,
    # continuous, flat and descent columns. The final reset is before physics.
    pose[:, 0] = asset.data.default_root_state[ids, 0]+env.scene.env_origins[ids, 0]
    pose[:, 0] += torch.empty(len(ids), device=env.device).uniform_(-.13, -.07)
    asset.write_root_pose_to_sim(pose, env_ids=ids)


def apply_profile(env_cfg, agent_cfg):
    if getattr(env_cfg, 'm20_refinement_profile', None) == PROFILE:
        return configure_agent(agent_cfg)
    original.apply_profile(env_cfg, agent_cfg)
    env_cfg.commands.base_velocity.class_type = Stage2VelocityCommand
    env_cfg.events.randomize_reset_base.func = reset_stage2_root
    env_cfg.rewards.ascent_entry_phase.weight = -.12
    env_cfg.rewards.ascent_entry_landing.func = objectives.landing_cost
    env_cfg.rewards.ascent_entry_new_tread.weight = 0.
    env_cfg.rewards.ascent_wheel_slip.weight = 0.
    for name, func, weight in (
        ('ascent_stage2_ordered_tread', objectives.ordered_bonus, .45),
        ('ascent_stage2_missing_tread', objectives.missing_cost, -.90),
        ('ascent_stage2_dwell_credit', objectives.dwell_bonus, .10),
        ('ascent_stage2_front_slip', objectives.front_slip_cost, -.45),
        ('ascent_stage2_rear_slip', objectives.rear_slip_cost, -.45),
        ('ascent_stage2_stall', objectives.stall_cost, -.15),
    ):
        setattr(env_cfg.rewards, name, RewardTermCfg(func=func, weight=weight))
    env_cfg.curriculum.ascent_stage2 = CurriculumTermCfg(func=objectives.metrics)
    env_cfg.m20_refinement_profile = PROFILE
    return configure_agent(agent_cfg)


def configure_agent(agent_cfg):
    manifest = original.configure_agent(agent_cfg)
    agent_cfg.algorithm.critic_warmup_iterations = 40
    if agent_cfg.experiment_name in ('deeprobotics_m20_stair_teacher', 'deeprobotics_m20_stair_teacher_v1'):
        agent_cfg.experiment_name = 'deeprobotics_m20_stair_teacher_v1_stage2'
    manifest.update(profile=PROFILE, version=VERSION, critic_warmup_iterations=40,
        initialization='complete_v1_actor_weights_optional_critic_weights_fresh_optimizer_and_iteration',
        ledger={'target_treads': [1, 2, 3, 4], 'stable_contact_s': .04, 'same_tread_gap_s': .01,
                'wheel_weights': [1.5, 1.5, 1., 1.], 'once_per_wheel_tread': True},
        slip={'front_weight': -.45, 'rear_weight': -.45, 'pair_mean_mix': .5, 'pair_max_mix': .5,
              'third_fourth_gain_max': 1.5},
        entry_sampling={'hard_vx05_dx_minus010': .2, 'hard_vx07_dx_minus010': .2,
                        'regular_entry': .2, 'continuous': .2, 'flat': .1, 'descent': .1,
                        'hard_spawn_x_interval_m': [-.13, -.07]})
    return manifest
