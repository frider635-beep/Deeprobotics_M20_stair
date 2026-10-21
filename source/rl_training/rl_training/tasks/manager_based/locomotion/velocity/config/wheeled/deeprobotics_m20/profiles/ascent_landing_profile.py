"""Third refinement profile; keep all future v1 logs under the stage2 root."""
from isaaclab.managers import RewardTermCfg, CurriculumTermCfg
from . import ascent_entry_stage2_profile as stage2
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_landing_mdp as landing
from rl_training.models.ascent_landing_actor import LANDING_LIMITS

PROFILE = 'ascent_entry_v1_stage3'
VERSION = 'm20_ascent_landing_20261020_v1_stage3'


def apply_profile(env_cfg, agent_cfg):
    if getattr(env_cfg, 'm20_refinement_profile', None) == PROFILE:
        return configure_agent(agent_cfg)
    stage2.apply_profile(env_cfg, agent_cfg)
    env_cfg.rewards.ascent_entry_phase.func = landing.phase_cost
    for name, func, weight in (
        ('ascent_landing_height', landing.height_cost, -.60),
        ('ascent_landing_horizontal', landing.horizontal_cost, -.10),
        ('ascent_landing_velocity', landing.velocity_cost, -.10),
    ):
        setattr(env_cfg.rewards, name, RewardTermCfg(func=func, weight=weight))
    env_cfg.curriculum.ascent_landing = CurriculumTermCfg(func=landing.metrics)
    env_cfg.m20_refinement_profile = PROFILE
    return configure_agent(agent_cfg)


def configure_agent(agent_cfg):
    manifest = stage2.configure_agent(agent_cfg)
    # More authority only at the entrance: reduce the update size and relax
    # the entrance's old-action anchor. Elsewhere the actor is exactly frozen.
    agent_cfg.algorithm.learning_rate = 2.e-5
    agent_cfg.algorithm.anchor_stair_weight = .03
    manifest.update(profile=PROFILE, version=VERSION,
        initialization='complete_stage2_actor_frozen_plus_zero_landing_branch_fresh_optimizer',
        landing_limits_action_units=LANDING_LIMITS,
        entrance_gate='original_ascent_gate_times_smooth_clamp_2_minus_rear_scan_riser_count',
        dense_guidance={'front_target_treads': [1, 2], 'approach_m': .30,
                        'pre_edge_clearance_m': .04, 'descent_center_offsets_m': [.01, .08],
                        'height_weight': -.60, 'horizontal_weight': -.10, 'velocity_weight': -.10},
        continuous_reference='smoothly_enabled_after_both_front_wheels_pass_tread2',
        frozen_source_includes_entry_mlp=True)
    return manifest
