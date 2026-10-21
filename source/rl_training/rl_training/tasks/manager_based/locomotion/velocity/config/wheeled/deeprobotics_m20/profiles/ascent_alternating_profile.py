"""Stage6: mandatory swing turns, low-speed progress and distance-aware slip."""
import torch
from isaaclab.managers import RewardTermCfg, CurriculumTermCfg
from . import ascent_adaptive_profile as previous
from .ascent_sequence_profile import SequenceVelocityCommand
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_alternating_mdp as alternating

PROFILE = 'ascent_entry_v1_stage6'
VERSION = 'm20_ascent_alternating_20261021_v1_stage6_geom_v2'


class AlternatingVelocityCommand(SequenceVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ids = ids[self._env.scene.terrain.terrain_types[ids] < 8]
        # Train both speeds at every ascent geometry; emphasize the regressed
        # low speed. A command remains fixed through a 24 s ascent episode.
        self.entry_speeds[ids] = torch.where(torch.rand(len(ids), device=self.device) < .6, .5, .7)
        self._apply_ascent_command()


def apply_profile(env_cfg, agent_cfg, evaluation=False):
    if getattr(env_cfg, 'm20_refinement_profile', None) != PROFILE:
        previous.apply_profile(env_cfg, agent_cfg, evaluation=evaluation)
        env_cfg.commands.base_velocity.class_type = AlternatingVelocityCommand
        env_cfg.commands.base_velocity.resampling_time_range = (24., 24.)
        # Replacement cadence starts with the first measurable one-tread
        # interval, including 1->2 and 3->4, and detects overdue landings.
        env_cfg.rewards.ascent_adaptive_cadence.weight = 0.
        weights = dict(progress=.6, stall=-.9, front_slip=-.45, rear_slip=-.45,
                       lateral=-.6, heading=-.3, cadence=-.12, overdue=-.12, turn=-1.)
        for name, weight in weights.items():
            setattr(env_cfg.rewards, 'ascent_alternating_'+name,
                    RewardTermCfg(func=alternating.reward, weight=weight, params={'name': name}))
        env_cfg.curriculum.ascent_adaptive = CurriculumTermCfg(func=alternating.curriculum)
        env_cfg.curriculum.ascent_alternating = CurriculumTermCfg(func=alternating.metrics)
        env_cfg.m20_refinement_profile = PROFILE
        alternating.gate_positive_rewards(env_cfg.rewards)
    manifest = configure_agent(agent_cfg)
    manifest['gated_positive_rewards'] = alternating.audit_positive_rewards(env_cfg.rewards)
    return manifest


def configure_agent(agent_cfg):
    manifest = previous.configure_agent(agent_cfg)
    agent_cfg.algorithm.learning_rate = 2.e-5
    manifest.update(profile=PROFILE, version=VERSION,
        swing_alternation=dict(pairs=['FL/FR', 'HL/HR'], first_side='either',
            front_rear_overlap_allowed=True, support_to_arm_s=.04, minimum_air_s=.02,
            lift_m=.02, fast_lift_m=.006, fast_lift_vz_m_s=.12,
            simultaneous='two_confirmed_starts_in_same_physics_sample',
            repeated_or_simultaneous='positive_rewards_off_and_true_terminal_until_reset',
            same_tread_landing_still_required=True, fixed_axle_tread_gap=None),
        positive_reward_gate='all_positive_weight_terms_including_velocity_tracking_and_forward_progress',
        tracking_reward='positive_velocity_tracking_scaled_by_real_forward_motion_after_05s_grace',
        contact_curriculum=dict(prefixes=[2, 4, 6, 18], speeds=[.5, .7],
            promote='both_speeds_at_least_6_successes_of_last_8_episodes_each',
            demote_consecutive_failures=3, geometry_probe_fraction_of_nonreference=.25,
            geometry_probe_prefix=6, geometry_probe_max_row=2,
            geometry_probe_max_height_m=.15, other_geometry_requires_full_flight=True),
        low_speed_sampling_fraction=.6, ascent_command_fixed_episode_s=24.,
        cadence='per_wheel_geometric_period_from_second_landing_plus_overdue_and_interval_consistency',
        path='whole_episode_centerline_error_and_episode_max_diagnostic',
        slip='front_rear_normalized_by_actual_forward_speed_with_extra_3_to_4_emphasis',
        contact_demonstration_schedule=dict(coefficient=0., decay_updates=800),
        old_demonstrations='not_loaded_without_swing_alternation_recertification')
    return manifest
