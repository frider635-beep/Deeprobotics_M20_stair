"""Stage4: coupled front sequence, flexible axle spacing, full-flight exit."""
from isaaclab.managers import RewardTermCfg, CurriculumTermCfg
from . import ascent_landing_profile as stage3
from . import ascent_entry_stage2_profile as stage2
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_sequence_mdp as sequence
from rl_training.models.ascent_sequence_actor import SEQUENCE_LIMITS
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_reward_audit import disable_conflicting_rewards, audit_conflicting_rewards, CONFLICTING_REWARDS

PROFILE = 'ascent_entry_v1_stage4'
VERSION = 'm20_ascent_sequence_20261020_v1_stage4'


class SequenceVelocityCommand(stage2.Stage2VelocityCommand):
    def _apply_ascent_command(self):
        super()._apply_ascent_command()
        env = self._env
        if not getattr(env.cfg, 'ascent_sequence_training_ramp', False): return
        step = env.common_step_counter+getattr(env, '_ascent_sequence_step_offset', 0)
        fraction = min(1., max(0., step/env.cfg.ascent_sequence_ramp_steps))
        factor = .6+.4*fraction*fraction*(3.-2.*fraction)
        mask = env.scene.terrain.terrain_types < 6
        self.vel_command_b[mask, 0] = self.entry_speeds[mask]*factor


def apply_profile(env_cfg, agent_cfg):
    if getattr(env_cfg, 'm20_refinement_profile', None) == PROFILE:
        audit_conflicting_rewards(env_cfg.rewards)
        return configure_agent(agent_cfg)
    stage3.apply_profile(env_cfg, agent_cfg)
    disable_conflicting_rewards(env_cfg.rewards)
    env_cfg.ascent_sequence_step_count = 18
    env_cfg.ascent_sequence_training_ramp = True
    env_cfg.ascent_sequence_ramp_steps = agent_cfg.num_steps_per_env*400
    env_cfg.commands.base_velocity.class_type = SequenceVelocityCommand
    # Replace independent early-tread accounting and old dense targets rather
    # than paying them again on top of the full-flight paired ledger.
    for name in ('ascent_stage2_ordered_tread', 'ascent_stage2_missing_tread', 'ascent_stage2_dwell_credit',
                 'ascent_entry_phase', 'ascent_entry_landing', 'ascent_landing_height',
                 'ascent_landing_horizontal', 'ascent_landing_velocity', 'ascent_entry_continuity'):
        getattr(env_cfg.rewards, name).weight = 0.
    for name, weight in [('ordered', .20), ('missing', -.90), ('dwell', .08), ('paired', .60),
                         ('simultaneous', -.15), ('early', -.35), ('cadence', -.03), ('approach', .08),
                         ('height', -.90), ('horizontal', -.15), ('velocity', -.10),
                         ('balance', -.12), ('surge', -.20), ('smoothness', -.10), ('top_front', .30),
                         ('top_rear', .50), ('top_forward', .80)]:
        setattr(env_cfg.rewards, 'ascent_sequence_'+name,
                RewardTermCfg(func=getattr(sequence, name), weight=weight))
    env_cfg.rewards.stair_orientation_cost.func = sequence.orientation
    env_cfg.terminations.ascent_entry_finished.func = sequence.finished
    env_cfg.curriculum.ascent_landing = None
    env_cfg.curriculum.ascent_sequence = CurriculumTermCfg(func=sequence.metrics)
    env_cfg.episode_length_s = 20.
    env_cfg.m20_refinement_profile = PROFILE
    return configure_agent(agent_cfg)


def configure_agent(agent_cfg):
    manifest = stage3.configure_agent(agent_cfg)
    agent_cfg.algorithm.learning_rate = 3.e-5
    agent_cfg.algorithm.critic_warmup_iterations = 40
    agent_cfg.algorithm.anchor_stair_weight = .03
    manifest.update(profile=PROFILE, version=VERSION,
        initialization='complete_stage3_frozen_plus_zero_sequence_branch_fresh_optimizer',
        sequence_limits_action_units=SEQUENCE_LIMITS,
        sequence={'treads': 18, 'stable_s': .04, 'minimum_front_separation_s': .04,
                  'lead_side': 'either', 'fixed_axle_tread_gap': None,
                  'early_advance': 'soft_cost_if_mate_has_not_supported_previous_tread',
                  'approach_credit': 'bounded_best_error_improvement_once_per_wheel_tread_phase',
                  'support': 'projected_COM_0.10s_lookahead_heuristic_not_ZMP_guarantee'},
        training_entry_speed={'initial_factor': .6, 'target_factor': 1., 'ramp_iterations': 400,
                              'continuous_courses_unchanged': True, 'evaluation_ramp': False},
        platform={'stable_front_then_rear': True, 'forward_credit_limit_m': .60,
                  'finish_forward_m': .35, 'finish_stable_s': .20,
                  'pitch_target': 'slope_between_current_front_and_rear_terrain_heights'},
        actor_inputs='policy244_plus_raw_command3_no_privileged_sequence_state')
    manifest['conflicting_gait_rewards_disabled'] = CONFLICTING_REWARDS.copy()
    manifest['incompatible_ascent_style_terms_disabled'] = list(CONFLICTING_REWARDS)
    return manifest
