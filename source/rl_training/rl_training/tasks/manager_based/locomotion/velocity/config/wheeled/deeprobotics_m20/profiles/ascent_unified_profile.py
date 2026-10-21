"""Stage7: consolidate ascent branches, bridge the 3->4 contact bottleneck."""
from isaaclab.managers import RewardTermCfg, CurriculumTermCfg
from . import ascent_alternating_profile as previous
from rl_training.models.ascent_unified_actor import PROFILE, VERSION, UNIFIED_LIMITS, EXPLORATION
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_unified_mdp as unified
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_alternating_mdp as alternating


def apply_profile(env_cfg, agent_cfg, evaluation=False):
    if getattr(env_cfg, 'm20_refinement_profile', None) != PROFILE:
        previous.apply_profile(env_cfg, agent_cfg, evaluation=evaluation)
        env_cfg.curriculum.ascent_unified = CurriculumTermCfg(func=unified.metrics)
        # Paired guidance replaces the old independent-front landing costs.
        for key in ('prepare','height','horizontal','velocity'):
            getattr(env_cfg.rewards, 'ascent_adaptive_'+key).weight = 0.
        env_cfg.rewards.ascent_adaptive_paired.weight = 1.5
        env_cfg.rewards.ascent_adaptive_missing.weight = -1.2
        env_cfg.rewards.ascent_adaptive_failure.weight = -3.
        env_cfg.rewards.ascent_alternating_progress.weight = .2
        env_cfg.rewards.ascent_alternating_front_slip.weight = -.65
        env_cfg.rewards.ascent_alternating_rear_slip.weight = -.55
        env_cfg.rewards.ascent_alternating_lateral.weight = -1.
        env_cfg.rewards.ascent_alternating_heading.weight = -.45
        for name,weight in dict(landing_height=-1.2, landing_horizontal=-.35, landing_velocity=-.15,
                                hold_mate=-.60, overspeed=-.40).items():
            setattr(env_cfg.rewards, 'ascent_unified_'+name, RewardTermCfg(func=unified.reward,weight=weight,params={'name':name}))
        env_cfg.m20_refinement_profile = PROFILE
        alternating.gate_positive_rewards(env_cfg.rewards)
    manifest = configure_agent(agent_cfg)
    manifest['gated_positive_rewards'] = alternating.audit_positive_rewards(env_cfg.rewards)
    return manifest


def configure_agent(agent_cfg):
    manifest = previous.configure_agent(agent_cfg)
    manifest.update(profile=PROFILE, version=VERSION, initialization='distilled_unified_actor_fresh_optimizer',
        actor_inputs='policy244_plus_raw_command3', unified_hidden_dims=[256,128],
        unified_limits_action_units=UNIFIED_LIMITS, adaptive_exploration_action_std=EXPLORATION,
        unified_gate='all_scan_rising_edges_no_entry_to_middle_authority_drop',
        contact_curriculum=dict(prefixes=[2,3,4,6,18], minimum_task_fractions={'2':.5,'3':.3,'4':.2},
            promote='both_speeds_6_of_8_each',demote_consecutive_failures=3,
            geometry_probe_fraction_of_nonreference=.25,geometry_probe_prefix=6,geometry_probe_max_row=2),
        prefix_active_deadlines_s=[8.,10.,12.,16.,23.5],
        paired_guidance='four_wheels_next_tread_and_hold_leader_until_mate_supported',
        rear_pair='sequential_same_tread_and_no_early_advance', lateral_terminal_corridor_m=.40,
        contact_demonstration_schedule=dict(coefficient=0.,decay_updates=800))
    return manifest
