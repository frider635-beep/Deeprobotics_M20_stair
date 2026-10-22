# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Unified M20 stair teacher: observations, terrain gates, event ledgers and rewards.

Terrain gates are derived from the same yaw-aligned ray scan exposed to the
teacher actor. Contact and actuator state are used only for training signals
and critic observations.

All stair-teacher reward implementations and gait envelopes live here. The env
configuration owns reward weights and entity bindings; TUNING below remains
the numeric source for gait envelope parameters.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from rl_training.utils.teacher_diagnostics import export_teacher_metrics
from .stop import stop_coverage
from rl_training.models import stop_branch

from .commands import UniformThresholdVelocityCommand
from .descent_front_pair import descent_front_pair_metrics, stair_descent_front_pair_cost
from .rewards import joint_pos_penalty


def _detailed_teacher_diagnostics(env) -> bool:
    """A logging choice only; rewards and physical event ledgers never depend on it."""
    return bool(getattr(getattr(env, "cfg", None), "teacher_detailed_diagnostics", False))


# =============================================================================
# 1. Gait envelope parameters (soft constraints, not hardware safety limits)
# =============================================================================

TUNING = {
    "translation_threshold": 0.10, "yaw_threshold": 0.10,
    "turn_posture_scale": 0.20,
    "air_min": 0.10, "air_target": 0.20, "air_max": 0.35,
    "turn_height_flat": 0.08, "turn_height_terrain": 0.16,
    "turn_reach_flat": 0.22, "turn_reach_terrain": 0.30,
    "height_scale": 0.08, "reach_scale": 0.10,
    "rear_min_extension": 0.35, "extension_scale": 0.10,
    "rear_max_knee": 1.90, "knee_scale": 0.50,
    "front_min_extension": 0.35, "front_max_knee": 2.0,
    "front_support_grace": 0.0, "front_fold_weight": -1.0,
    "front_swing_min_extension": 0.22, "front_swing_max_knee": 2.50,
    "front_swing_scale": 0.50,
    "support_grace": 0.12, "same_tread_grace": 0.12,
    "wheel_radius": 0.09, "nearest_valid_distance": 0.12,
    "turn_size_weight": -1.0, "rear_fold_weight": -1.0,
    "same_tread_event_weight": -1.5, "same_tread_dwell_weight": -0.2,
    "task_crossing_weight": 2.0,
    "turn_status_weight": 1.0, "turn_symmetry_weight": 2.0,
    "ascent_direction_weight": -1.0,
    "direction_lateral_tolerance": 0.10, "direction_lateral_scale": 0.25,
    "direction_speed_tolerance": 0.05, "direction_speed_scale": 0.20,
    "direction_heading_tolerance": np.deg2rad(3.0),
    "direction_heading_scale": np.deg2rad(8.0),
    "direction_heading_soft_cap": 8.0,
    "direction_velocity_mix": 1.0, "direction_heading_mix": 2.5,
    "direction_soft_cap": 4.0,
    "direction_command_change": np.deg2rad(15.0),
    "direction_exit_inset": 0.06, "direction_exit_height_tolerance": 0.05,
    "direction_exit_contact_time": 0.06,
    # Persistent physical tread ledger; these are perception/style tolerances.
    "riser_identity_distance": 0.12, "riser_height_tolerance": 0.04,
    "riser_lateral_corridor": 0.45, "riser_min_spacing": 0.15,
    "riser_observation_half_width": 0.35,
    "tread_observation_corridor": 0.80,
    "riser_max_spacing": 0.50, "riser_cross_clearance": 0.01,
    "riser_default_half_width": 0.05,
    "tread_inset": 0.025, "tread_height_tolerance": 0.04,
    "landing_time": 0.04, "prep_distance": 0.60,
    "prep_weight": 0.5, "rear_completion_weight": 3.0,
    "front_touchdown_weight": 0.25,
    "front_a_support_low_deg": 90.0, "front_a_support_high_deg": 80.0,
    "front_a_swing_low_deg": 75.0, "front_a_swing_high_deg": 55.0,
    "pose_low_rise": 0.15, "pose_high_rise": 0.25,
    "pose_angle_scale": np.deg2rad(20.0), "pose_length_scale": 0.10,
    "pose_load_force_scale": 60.0,
    "rear_b_limit": np.deg2rad(25.0), "rear_g_limit": np.deg2rad(20.0),
    "rear_swing_b_limit": np.deg2rad(35.0), "rear_swing_g_limit": np.deg2rad(25.0),
    "rear_swing_mix": 0.75, "rear_load_floor": 0.5,
    "rear_target_x_scale": 0.30, "rear_target_z_scale": 0.20,
    "rear_swing_clearance": 0.04, "rear_swing_target_ceiling": 0.15,
    "rear_recovery_weight": 1.0,
    "rear_forward_relax": np.deg2rad(5.0), "rear_forward_weight": -2.0,
    "front_swing_min_extension_v3": 0.30,
    "front_swing_length_high": 0.26, "front_support_length_high": 0.33,
    "body_gap_target": 0.22, "body_gap_scale": 0.10,
    "body_bottom_z": -0.07, "body_gap_weight": -1.0,
    "pose_tracking_discount": 0.0,
    "pose_phase_radius": 1.50,
    "entry_x_lower": -0.05, "entry_x_upper": 0.15,
    "entry_x_high_rise_relax": 0.02, "entry_angle_upper": np.deg2rad(20.0),
    "entry_x_scale": 0.10, "entry_start_distance": 0.60,
    "entry_ground_tolerance": 0.04, "entry_contact_time": 0.06,
    "entry_forward_speed": 0.05, "entry_wheel_reverse_tolerance": 0.05,
    "entry_rearm_time": 0.50, "entry_retract_weight": 0.75,
    "entry_range_cost_weight": -1.0,
    "entry_rear_pair_swing_cost_weight": -2.0,
    # Refinement: stop asking the front axle to reach the SECOND tread while
    # the rear axle stays on the platform. That objective competes with a
    # compact front posture. Keep the physical ledgers for diagnostics and
    # retain the independent wheel-spacing / rear-leg collision constraints.
    "entry_front_lead_weight": 0.0, "entry_rear_early_transfer_weight": -2.0,
    "entry_front_progress_weight": 0.0, "entry_sequence_weight": 0.0,
    "entry_wheel_spacing_left_weight": -3.0,
    "entry_wheel_spacing_right_weight": -3.0,
    "entry_wheel_spacing_min": 0.28, "entry_wheel_spacing_scale": 0.05,
    "entry_transfer_wheel_guard": 0.34,
    "entry_transfer_shank_guard": 0.24,
    "entry_transfer_lookahead_s": 0.25,
    "descent_wheel_shank_min": 0.20,
    "descent_wheel_shank_scale": 0.05,
    # Keep the compact warning margins, but make the final clearance much
    # more expensive than the old linear-tail envelope. These distances are
    # conservative geometric proxies, not measured collision mesh surfaces.
    "interleg_wheel_critical": 0.25,
    "interleg_shank_critical": 0.18,
    "interleg_critical_scale": 0.02,
    "interleg_critical_gain": 4.0,
    "interleg_closing_speed_scale": 0.25,
    "interleg_closing_gate": 0.03,
    "interleg_braking_mix": 0.5,
    "interleg_shank_proxy_radius": 0.03,
    # On the first three descending transfers, keep the front legs near their
    # measured flat-ground posture. Swing gets more reach and knee freedom than
    # loaded support so the wheels can still clear and land on the next tread.
    "descent_front_support_forward_margin": 0.03,
    "descent_front_swing_forward_margin": 0.07,
    "descent_front_forward_scale": 0.05,
    "descent_front_support_knee_margin": 0.30,
    "descent_front_swing_knee_margin": 0.50,
    "descent_front_knee_scale": 0.35,
    "descent_front_posture_weight": -1.5,
    # Flexible front-axle descent: allow a brief stagger, then discourage a
    # long one-wheel drop and the accompanying roll, without a second-tread goal.
    "descent_front_pair_progress_tolerance": 0.20,
    "descent_front_pair_progress_scale": 0.25,
    "descent_front_pair_forward_tolerance": 0.05,
    "descent_front_pair_forward_scale": 0.15,
    "descent_front_pair_lag_grace": 0.08,
    "descent_front_pair_lag_scale": 0.15,
    "descent_front_pair_lag_mix": 0.6,
    "descent_front_pair_roll_tolerance": np.deg2rad(3.0),
    "descent_front_pair_roll_scale": np.deg2rad(5.0),
    "descent_front_pair_roll_rate_tolerance": 0.3,
    "descent_front_pair_roll_rate_scale": 0.6,
    "descent_front_pair_roll_mix": 0.75,
    "descent_front_pair_weight": -1.5,
    "entry_lead_min_spacing": 0.12, "entry_lead_max_spacing": 0.90,
    "entry_lead_wait_timeout": 3.0,
    "stair_zero_transition_probability": 0.50,
    "flat_zero_transition_probability": 0.50,
    "flat_straight_practice_probability": 0.35,
    "stair_straight_practice_probability": 0.60,
    "stair_exit_zero_probability": 0.50,
    "flat_height_range_max": 0.035,
    "flat_command_duration_min": 3.0,
    "flat_command_duration_max": 5.0,
    "zero_command_duration_min": 6.0,
    "zero_command_duration_max": 8.0,
    "descent_entry_pause_probability": 0.25,
    "descent_entry_pause_duration_min": 2.0,
    "descent_entry_pause_duration_max": 3.0,
    "descent_entry_restart_speed_min": 0.4,
    "descent_entry_restart_speed_max": 0.7,
    "descent_entry_pause_height_tolerance": 0.05,
    "descent_entry_pause_contact_time": 0.06,
    "zero_hold_linear_velocity_scale": 0.05,
    "flat_zero_hold_linear_velocity_scale": 0.05,
    "flat_zero_hold_wheel_velocity_scale": 0.07,
    "zero_hold_brake_grace_s": 0.4,
    "zero_hold_leg_velocity_scale": 0.40,
    "zero_hold_yaw_velocity_scale": 0.06,
    "zero_hold_wheel_linear_velocity_scale": 0.07,
    "zero_hold_wheel_action_linear_scale": 0.10,
    # Physical processed targets; these stop-specific regularizers remain
    # active at curriculum level zero and cannot cancel between wheels.
    "zero_hold_action_ramp_s": 0.4,
    "zero_hold_leg_target_delta_scale": 0.008,
    "zero_hold_leg_target_jerk_scale": 0.006,
    "zero_hold_wheel_target_delta_scale": 0.025,
    "zero_hold_wheel_target_jerk_scale": 0.020,
    "zero_hold_stair_dynamics_gain": 1.0,
    # Bounded, observable position recovery after braking on stairs. These
    # gains are mirrored in the deployment controller, not an action filter.
    "stair_stop_position_kp": 0.5,
    "stair_stop_velocity_kd": 0.5,
    "stair_stop_position_deadband": 0.02,
    "stair_stop_velocity_deadband": 0.015,
    "stair_stop_max_speed": 0.10,
    "stair_stop_slew": 0.30,
    "stair_stop_position_scale": 0.06,
    "descent_path_kp": 0.35,
    "descent_path_kd": 0.4,
    "descent_path_max_lateral_speed": 0.10,
    "descent_path_lateral_deadband": 0.03,
    "descent_path_speed_deadband": 0.03,
    "descent_path_lateral_slew": 0.30,
    "zero_hold_quiet_duration_s": 1.0,
    "quiet_leg_speed_rms": 0.10,
    "quiet_leg_speed_max": 0.20,
    "quiet_leg_target_delta": 0.002,
    "quiet_leg_target_second_difference": 0.003,
    "quiet_wheel_target_delta": 0.010,
    "quiet_wheel_target_second_difference": 0.015,
    "teacher_angular_velocity_noise": 0.03,
    "teacher_joint_velocity_noise": 0.20,
    "teacher_joint_position_noise": 0.005,
    "teacher_projected_gravity_noise": 0.01,
    "zero_hold_heading_tolerance": np.deg2rad(0.5),
    "zero_hold_heading_scale": np.deg2rad(3.0),
    "zero_hold_heading_weight": 1.0,
    "hold_success_base_speed": 0.02,
    "hold_success_wheel_speed": 0.03,
    "hold_success_yaw_rate": 0.03,
    "hold_success_heading": np.deg2rad(2.0),
    "ascent_heading_scale": np.deg2rad(5.0),
    "ascent_heading_tolerance": np.deg2rad(1.5),
    "ascent_yaw_tracking_scale": 0.12,
    "heading_hold_kp": 0.8,
    "heading_hold_kd": 0.4,
    "heading_hold_max_yaw_rate": 0.25,
    "heading_hold_deadband": np.deg2rad(0.5),
    "stair_heading_hold_kp": 1.6,
    "stair_heading_hold_kd": 0.7,
    "stair_heading_hold_max_yaw_rate": 0.50,
    "stair_heading_hold_deadband": np.deg2rad(0.5),
    "heading_hold_filter_tau": 0.04,
    "heading_hold_exit_deadband": np.deg2rad(0.25),
    "heading_hold_rate_enter": 0.03,
    "heading_hold_rate_exit": 0.015,
    "heading_hold_slew_flat": 1.5,
    "heading_hold_slew_stairs": 3.0,
    # Ascent-only path feedback through the existing lateral command channel.
    "ascent_path_kp": 0.35,
    "ascent_path_kd": 0.4,
    "ascent_path_max_lateral_speed": 0.12,
    "ascent_path_lateral_deadband": 0.03,
    "ascent_path_speed_deadband": 0.03,
    "ascent_path_lateral_slew": 0.3,
    "ascent_path_error_tolerance": 0.05,
    "ascent_path_error_scale": 0.15,
    "ascent_path_velocity_error_tolerance": 0.03,
    "ascent_path_velocity_error_scale": 0.12,
    "ascent_forward_overspeed_tolerance": 0.15,
    "ascent_forward_overspeed_scale": 0.25,
    "ascent_straight_motion_weight": -1.0,
    "entry_lead_landing_inset": 0.10, "entry_lead_x_scale": 0.25,
    "entry_lead_z_scale": 0.18,
    "entry_swing_x_lower": -0.10, "entry_swing_x_upper": 0.22,
    "entry_swing_min_extension": 0.30, "entry_swing_min_extension_high": 0.26,
    "descent_cost_soft_cap": 4.0,

}


# =============================================================================
# 2. Velocity command exposure and privileged critic observations
# =============================================================================

class StairTeacherVelocityCommand(UniformThresholdVelocityCommand):
    """Track command exposure and train zero-command holds on stairs and flats."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.episode_commanded_translation_m = torch.zeros(self.num_envs, device=self.device)
        # Integrate the requested world displacement separately from the total
        # requested path. Reversal/turn practice must not look like a failed
        # one-way traversal merely because the episode ends near its start.
        self.episode_commanded_translation_w = torch.zeros((self.num_envs, 2), device=self.device)
        self._stair_zero_transition_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._flat_zero_transition_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stair_exit_zero_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._was_stair_area = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._entry_pause_decided_token = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._entry_pause_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._entry_pause_until_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._entry_pause_timer_expired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._entry_pause_body_ids = None
        self._v3_stair_timer_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._v4_preview_decided = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._v4_preview_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._v4_preview_expired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._v4_preview_direction = torch.ones(self.num_envs, device=self.device)
        self.metrics["stair_zero_transition_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["flat_zero_transition_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stair_exit_zero_fraction"] = torch.zeros(self.num_envs, device=self.device)
        for name in ("descent_entry_pause_opportunities", "descent_entry_pause_events",
                     "descent_entry_restart_events", "descent_entry_pause_time_fraction"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        # Deliberately train directional-to-zero transitions on both stairs
        # and flat terrain, in addition to the global standing command mix.
        ids = (torch.arange(self.num_envs, device=self.device) if isinstance(env_ids, slice)
               else torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
        previous = self.vel_command_b[ids].clone()
        restart_preview = self._v4_preview_pending[ids] & self._v4_preview_expired[ids]
        self._v4_preview_pending[ids] = False
        previous_flat_stop = self._flat_zero_transition_env[ids].clone()
        # Only expiry of our own pause may force a restart. A reset or an
        # unrelated early resample cancels it and keeps normal command sampling.
        restart_entry = (self._entry_pause_pending[ids] & self._entry_pause_timer_expired[ids]
                         & (torch.linalg.vector_norm(previous[:, :2], dim=1) <= TUNING["translation_threshold"])
                         & (previous[:, 2].abs() <= TUNING["yaw_threshold"]))
        self._entry_pause_pending[ids] = False
        super()._resample_command(env_ids)
        self._stair_zero_transition_env[ids] = False
        self._flat_zero_transition_env[ids] = False
        self._stair_exit_zero_env[ids] = False
        if ids.numel() == 0:
            return
        on_stairs = _stair_activity_mask(self._env)
        had_motion = torch.linalg.vector_norm(previous[:, :2], dim=1) > TUNING["translation_threshold"]
        stair_transition = (
            on_stairs[ids]
            & had_motion
            & (torch.rand(ids.numel(), device=self.device)
               < TUNING["stair_zero_transition_probability"])
        )
        flat = _flat_surface_mask(self._env, SceneEntityCfg("height_scanner"))[ids] & ~on_stairs[ids]
        flat_transition = (
            flat
            & had_motion
            & (torch.rand(ids.numel(), device=self.device)
               < TUNING["flat_zero_transition_probability"])
        )
        # The forced stop is followed by a straight motion command, giving
        # each flat-ground episode repeated move/stop practice.
        restart_straight = flat & previous_flat_stop
        sampled_motion = torch.linalg.vector_norm(self.vel_command_b[ids, :2], dim=1) > TUNING["translation_threshold"]
        practice_straight = (
            flat & sampled_motion & ~flat_transition & ~stair_transition
            & (torch.rand(ids.numel(), device=self.device) < TUNING["flat_straight_practice_probability"])
        )
        straight_ids = ids[restart_straight | practice_straight]
        if straight_ids.numel() > 0:
            speed = torch.empty(straight_ids.numel(), device=self.device).uniform_(0.4, 0.8)
            direction = torch.where(
                torch.rand(straight_ids.numel(), device=self.device) < 0.75,
                torch.ones_like(speed), -torch.ones_like(speed),
            )
            self.vel_command_b[straight_ids] = 0.0
            self.vel_command_b[straight_ids, 0] = speed * direction
            self.is_standing_env[straight_ids] = False
            self.is_heading_env[straight_ids] = False
        # Long, straight stair flights match W-only deployment. Keep the
        # remaining samples for turning, reversing and general locomotion.
        stair_straight = on_stairs[ids] & sampled_motion & ~stair_transition & (
            torch.rand(ids.numel(), device=self.device) < TUNING["stair_straight_practice_probability"]
        )
        stair_ids = ids[stair_straight]
        self.vel_command_b[stair_ids] = 0.0
        self.vel_command_b[stair_ids, 0] = torch.empty(stair_ids.numel(), device=self.device).uniform_(0.4, 0.8)
        if TUNING.get("zero_stop_v3", False):
            # Match both W-release and S-release deployment, rather than
            # replacing every sampled stair straight request with forward.
            reverse = torch.rand(stair_ids.numel(), device=self.device) < .25
            self.vel_command_b[stair_ids[reverse], 0] *= -1
        self.is_standing_env[stair_ids] = False
        self.is_heading_env[stair_ids] = False
        zero_ids = ids[stair_transition | (flat_transition & ~restart_straight)]
        self.vel_command_b[zero_ids] = 0.0
        self.is_standing_env[zero_ids] = True
        self.is_heading_env[zero_ids] = False
        self._stair_zero_transition_env[ids[stair_transition]] = True
        self._flat_zero_transition_env[ids[flat_transition & ~restart_straight]] = True
        short_ids = ids[flat & ((restart_straight | practice_straight) | flat_transition)]
        if short_ids.numel() > 0:
            self.time_left[short_ids] = torch.empty(short_ids.numel(), device=self.device).uniform_(
                TUNING["flat_command_duration_min"], TUNING["flat_command_duration_max"]
            )
        hold_ids = ids[(self.vel_command_b[ids].abs() <= 1.0e-6).all(dim=1)]
        self.time_left[hold_ids] = torch.empty(hold_ids.numel(), device=self.device).uniform_(
            TUNING["zero_command_duration_min"], TUNING["zero_command_duration_max"]
        )
        restart_ids = ids[restart_entry]
        self.vel_command_b[restart_ids] = 0.0
        self.vel_command_b[restart_ids, 0] = torch.empty(restart_ids.numel(), device=self.device).uniform_(
            TUNING["descent_entry_restart_speed_min"], TUNING["descent_entry_restart_speed_max"]
        )
        self.is_standing_env[restart_ids] = False
        self.is_heading_env[restart_ids] = False
        self._stair_zero_transition_env[restart_ids] = False
        self._flat_zero_transition_env[restart_ids] = False
        self._stair_exit_zero_env[restart_ids] = False
        self.metrics["descent_entry_restart_events"][restart_ids] += 1.0
        if TUNING.get("zero_stop_v4", False):
            preview_ids = ids[restart_preview]
            self.vel_command_b[preview_ids] = 0.
            self.vel_command_b[preview_ids, 0] = self._v4_preview_direction[preview_ids]*torch.empty(
                preview_ids.numel(), device=self.device).uniform_(.4, .8)
            self.is_standing_env[preview_ids] = False
            self.is_heading_env[preview_ids] = False
            self.time_left[preview_ids] = 2.
        # A restart must not accidentally inherit a freshly sampled zero hold's
        # longer duration. Subsequent changes return to the ordinary sampler.
        self.time_left[restart_ids] = torch.empty(restart_ids.numel(), device=self.device).uniform_(
            *self.cfg.resampling_time_range
        )
        if TUNING.get("zero_stop_v2", False):
            # Resample while still on a flight, rather than usually reaching
            # its platform before the next zero-command opportunity.
            c = self.vel_command_b[ids]
            motion = (on_stairs[ids] & (c[:, 0].abs() > TUNING["translation_threshold"])
                      & (c[:, 1].abs() <= TUNING["translation_threshold"])
                      & (c[:, 2].abs() <= TUNING["yaw_threshold"]))
            selected = ids[motion]
            self.time_left[selected] = torch.empty(selected.numel(), device=self.device).uniform_(1.5, 3.0)

    def compute(self, dt: float):
        # The superclass replaces time_left before invoking _resample_command.
        # Capture the reason here: unrelated resamples must never be treated
        # as an expired targeted pause, even after floating-point timer drift.
        self._entry_pause_timer_expired = self._entry_pause_pending & (self.time_left <= dt)
        self._v4_preview_expired = self._v4_preview_pending & (self.time_left <= dt)
        try:
            super().compute(dt)
        finally:
            self._entry_pause_timer_expired.zero_()
            self._v4_preview_expired.zero_()

    def _update_descent_entry_pause(self):
        """Once per physical entry, practice a first-tread pause and restart."""
        descent = getattr(self._env, "_m20_stair_step_state", None)
        entry = getattr(self._env, "_m20_descent_entry_state", None)
        if descent is None or entry is None:
            return
        if self._entry_pause_body_ids is None:
            names = [f"{leg}_wheel" for leg in ("fl", "fr", "hl", "hr")]
            body_ids, _ = self.robot.find_bodies(names, preserve_order=True)
            contact_ids, _ = self._env.scene.sensors["contact_forces"].find_bodies(names, preserve_order=True)
            if len(body_ids) != 4 or len(contact_ids) != 4:
                raise ValueError("M20 entry-stop practice requires four ordered wheel bodies and contacts")
            self._entry_pause_body_ids = (body_ids, contact_ids)
        body_ids, contact_ids = self._entry_pause_body_ids
        sensor = self._env.scene.sensors["contact_forces"]
        force = sensor.data.net_forces_w[:, contact_ids]
        loaded = (force[..., 2] > 10.0) & (force[..., 2] > 0.5 * torch.linalg.vector_norm(force, dim=-1))
        stable = loaded & (sensor.data.current_contact_time[:, contact_ids]
                           >= TUNING["descent_entry_pause_contact_time"])
        center_z = self.robot.data.body_pos_w[:, body_ids, 2]
        tolerance = TUNING["descent_entry_pause_height_tolerance"]
        front_on_first = ((center_z[:, :2] - descent["target_z"][:, None]
                           - TUNING["wheel_radius"]).abs() <= tolerance).all(-1)
        rear_on_source = ((center_z[:, 2:] - descent["source_z"][:, None]
                           - TUNING["wheel_radius"]).abs() <= tolerance).all(-1)
        # Entry history identifies a flight entrance; the step ledger identifies
        # a tread. Requiring the original top excludes later risers in the flight.
        first_transfer = (descent["source_z"] - entry["top"]).abs() <= tolerance
        command = self.vel_command_b
        straight = ((command[:, 0] > TUNING["translation_threshold"])
                    & (command[:, 1].abs() <= TUNING["translation_threshold"])
                    & (command[:, 2].abs() <= TUNING["yaw_threshold"]))
        eligible = (entry["active"] & (entry["count"] > 0)
                    & (entry["count"] != self._entry_pause_decided_token)
                    & descent["active"] & ~descent["up"] & descent["front_reached"].all(-1)
                    & first_transfer & front_on_first & rear_on_source & stable.all(-1)
                    & straight & ~self._entry_pause_pending & (self._env.episode_length_buf > 1))
        # Decide once, including skipped samples. Repeated contact ticks must
        # not turn a 25% probability into a nearly certain intervention.
        self._entry_pause_decided_token[eligible] = entry["count"][eligible]
        self.metrics["descent_entry_pause_opportunities"] += eligible
        selected = eligible & (torch.rand(self.num_envs, device=self.device)
                               < TUNING["descent_entry_pause_probability"])
        ids = selected.nonzero(as_tuple=False).flatten()
        duration = torch.empty(ids.numel(), device=self.device).uniform_(
            TUNING["descent_entry_pause_duration_min"], TUNING["descent_entry_pause_duration_max"]
        )
        self.vel_command_b[ids] = 0.0
        self.is_standing_env[ids] = True
        self.is_heading_env[ids] = False
        self._entry_pause_pending[ids] = True
        self._entry_pause_until_step[ids] = (int(self._env.common_step_counter)
                                            + torch.ceil(duration / self._env.step_dt).long())
        self.time_left[ids] = duration
        self._stair_zero_transition_env[ids] = True
        self._flat_zero_transition_env[ids] = False
        self._stair_exit_zero_env[ids] = False
        self.metrics["descent_entry_pause_events"][ids] += 1.0

    def _update_command(self):
        # Explicit new turning/reversing requests cancel a pending practice
        # pause before the inherited standing mask could overwrite them.
        overridden = self._entry_pause_pending & (
            (torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=1) > TUNING["translation_threshold"])
            | (self.vel_command_b[:, 2].abs() > TUNING["yaw_threshold"]))
        self._entry_pause_pending[overridden] = False
        self.is_standing_env[overridden] = False
        if TUNING.get("zero_stop_v4", False):
            overridden_preview = self._v4_preview_pending & (self.vel_command_b.abs().amax(-1) > 1.e-4)
            self._v4_preview_pending[overridden_preview] = False
            self.is_standing_env[overridden_preview] = False
        super()._update_command()
        on_stairs = _stair_activity_mask(self._env)
        flat = _flat_surface_mask(self._env, SceneEntityCfg("height_scanner"))
        reset = self._env.episode_length_buf <= 1
        if TUNING.get("zero_stop_v4", False):
            support = _zero_stop_support_context(self._env)
            straight = ((self.vel_command_b[:, 0].abs() > TUNING["translation_threshold"])
                        & (self.vel_command_b[:, 1].abs() <= TUNING["translation_threshold"])
                        & (self.vel_command_b[:, 2].abs() <= TUNING["yaw_threshold"]))
            # Decide once per episode, not once per contact tick. This creates
            # real preview stops before the wheels touch the riser.
            opportunity = support["preview"] & straight & ~self._v4_preview_decided & ~reset
            self._v4_preview_decided |= opportunity
            selected = opportunity & (torch.rand(self.num_envs, device=self.device) < .5)
            self._v4_preview_direction[selected] = self.vel_command_b[selected, 0].sign()
            self.vel_command_b[selected] = 0.
            self.is_standing_env[selected] = True
            self.is_heading_env[selected] = False
            self._v4_preview_pending[selected] = True
            self.time_left[selected] = torch.empty(int(selected.sum()), device=self.device).uniform_(6., 8.)
        if TUNING.get("zero_stop_v3", False):
            straight = ((self.vel_command_b[:, 0].abs() > TUNING["translation_threshold"])
                        & (self.vel_command_b[:, 1].abs() <= TUNING["translation_threshold"])
                        & (self.vel_command_b[:, 2].abs() <= TUNING["yaw_threshold"]))
            motion = on_stairs & straight & ~reset
            entering = motion & ~self._v3_stair_timer_active
            ids = entering.nonzero(as_tuple=False).flatten()
            # Arm once on physical stair entry as well as at resampling. A
            # long command sampled on the platform must not span the flight.
            duration = torch.empty(ids.numel(), device=self.device).uniform_(1.5, 3.)
            self.time_left[ids] = torch.minimum(self.time_left[ids], duration)
            self._v3_stair_timer_active = motion.clone()
        leaving = self._was_stair_area & ~on_stairs & flat & (
            self.vel_command_b[:, 0] > TUNING["translation_threshold"]
        ) & ~reset
        stop = leaving & (torch.rand(self.num_envs, device=self.device) < TUNING["stair_exit_zero_probability"])
        ids = stop.nonzero(as_tuple=False).flatten()
        if ids.numel() > 0:
            self.vel_command_b[ids] = 0.0
            self.is_standing_env[ids] = True
            self.is_heading_env[ids] = False
            self._flat_zero_transition_env[ids] = True
            self._stair_exit_zero_env[ids] = True
            self.time_left[ids] = torch.empty(ids.numel(), device=self.device).uniform_(
                TUNING["zero_command_duration_min"], TUNING["zero_command_duration_max"]
            )
        self._was_stair_area = on_stairs.clone()
        self._was_stair_area[reset] = False
        self._update_descent_entry_pause()

    def _update_metrics(self):
        super()._update_metrics()
        self.episode_commanded_translation_m += (
            torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=1) * self._env.step_dt
        )
        translation = torch.zeros((self.num_envs, 3), device=self.device)
        translation[:, :2] = self.vel_command_b[:, :2]
        self.episode_commanded_translation_w += (
            quat_apply(yaw_quat(self.robot.data.root_quat_w), translation)[:, :2] * self._env.step_dt
        )
        self.metrics["stair_zero_transition_fraction"] += self._stair_zero_transition_env.float()
        self.metrics["flat_zero_transition_fraction"] += self._flat_zero_transition_env.float()
        self.metrics["stair_exit_zero_fraction"] += self._stair_exit_zero_env.float()
        self.metrics["descent_entry_pause_time_fraction"] += self._entry_pause_pending.float()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        ids = (torch.arange(self.num_envs, device=self.device) if isinstance(env_ids, slice)
               else torch.as_tensor(env_ids, device=self.device, dtype=torch.long))
        zero_count = self._metric_step_counter[ids].float().clamp_min(1.0)
        zero_fraction = (
            self.metrics["stair_zero_transition_fraction"][ids] / zero_count
        ).mean().item()
        flat_zero_fraction = (
            self.metrics["flat_zero_transition_fraction"][ids] / zero_count
        ).mean().item()
        exit_zero_fraction = (
            self.metrics["stair_exit_zero_fraction"][ids] / zero_count
        ).mean().item()
        entry_pause_fraction = (
            self.metrics["descent_entry_pause_time_fraction"][ids] / zero_count
        ).mean().item()
        self.episode_commanded_translation_m[env_ids] = 0.0
        self.episode_commanded_translation_w[env_ids] = 0.0
        self._entry_pause_decided_token[ids] = -1
        self._entry_pause_pending[ids] = False
        self._entry_pause_until_step[ids] = 0
        self._entry_pause_timer_expired[ids] = False
        self._v4_preview_decided[ids] = False
        self._v4_preview_pending[ids] = False
        self._v4_preview_expired[ids] = False
        if TUNING.get("zero_stop_v4", False):
            self._env._m20_stop_support = None
        if TUNING.get('zero_stop_v5', False):
            self._env._m20_fast_flat_cache = None
            self._env._m20_stop_control_observation = None
        if TUNING.get("zero_stop_v3", False):
            self._v3_stair_timer_active[ids] = False
        extras = super().reset(env_ids)
        extras["stair_zero_transition_fraction"] = zero_fraction
        extras["flat_zero_transition_fraction"] = flat_zero_fraction
        extras["stair_exit_zero_fraction"] = exit_zero_fraction
        extras["descent_entry_pause_time_fraction"] = entry_pause_fraction
        return extras


def critic_joint_acceleration(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Privileged joint acceleration in the configured joint order."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def critic_applied_torque(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Privileged actuator torque in the configured joint order."""
    asset = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def critic_wheel_contact_force_magnitude(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Four wheel contact-force magnitudes, used by the critic only."""
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return torch.linalg.vector_norm(force, dim=-1)


def critic_material_friction(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mean randomized static and dynamic robot material friction per env.

    The ground material in this M20 scene has unit friction and multiply
    combine mode, so the randomized robot material is also the effective
    contact coefficient. This returns two values per environment.
    """
    asset = env.scene[asset_cfg.name]
    material = asset.root_physx_view.get_material_properties().to(env.device)
    if material.ndim < 2 or material.shape[-1] < 2:
        raise RuntimeError(f"Unexpected PhysX material-property shape: {tuple(material.shape)}")
    material = material[..., :2]
    if material.shape[0] == env.num_envs or material.shape[0] % env.num_envs == 0:
        return material.reshape(env.num_envs, -1, 2).mean(dim=1)
    if material.shape[0] == 1:
        return material.reshape(1, -1, 2).mean(dim=1).expand(env.num_envs, -1)
    raise RuntimeError(
        "PhysX material properties cannot be grouped by environment: "
        f"shape={tuple(material.shape)}, num_envs={env.num_envs}"
    )


# =============================================================================
# 3. Height-scan geometry and terrain gates
# =============================================================================

def _stair_scan_context(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_edge_height: float = 0.06,
    gate_ramp_height: float = 0.04,
    lateral_half_width: float = 0.35,
    edge_min_x: float = 0.05,
    edge_max_x: float = 0.70,
    command_threshold: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Find the nearest usable height discontinuity along positive body-x travel.

    RayCaster's exposed local ray starts identify each grid column without
    assuming a hard-coded flattened grid shape. Invalid or incomplete scans
    produce a closed (zero) reward gate.
    """
    sensor = env.scene[sensor_cfg.name]
    counter = int(getattr(env, "common_step_counter", -1))
    cache = getattr(env, "_m20_stair_scan_context", None)
    if cache is not None and cache["counter"] == counter and cache["sensor_name"] == sensor_cfg.name:
        return cache["context"]

    hits = sensor.data.ray_hits_w
    ray_starts = getattr(sensor, "ray_starts", None)
    if ray_starts is None:
        raise RuntimeError("M20 stair rewards require RayCaster.ray_starts for height-grid geometry.")
    ray_starts = torch.as_tensor(ray_starts, device=hits.device, dtype=hits.dtype)
    if ray_starts.ndim == 3:
        # Isaac Lab stores one ray-start grid per sensor instance; the pattern
        # is identical across environments, so one grid defines x/y columns.
        ray_starts = ray_starts[0]
    if ray_starts.ndim != 2 or ray_starts.shape[0] != hits.shape[1] or ray_starts.shape[1] < 2:
        raise RuntimeError(
            "Height scan geometry does not match ray hits: "
            f"starts={tuple(ray_starts.shape)}, hits={tuple(hits.shape)}"
        )

    x_values = ray_starts[:, 0]
    y_values = ray_starts[:, 1]
    x_levels, x_inverse = torch.unique(x_values, sorted=True, return_inverse=True)
    column_heights = []
    column_valid = []
    hit_z = hits[..., 2]
    hit_valid = torch.isfinite(hit_z) & (hit_z.abs() < 1.0e6)
    for column_idx in range(x_levels.numel()):
        in_column = (x_inverse == column_idx) & (y_values.abs() <= lateral_half_width)
        valid = hit_valid[:, in_column]
        values = hit_z[:, in_column]
        nan_values = torch.where(valid, values, torch.full_like(values, float("nan")))
        medians = torch.nanmedian(nan_values, dim=1).values
        has_data = valid.sum(dim=1) >= 2
        column_heights.append(torch.nan_to_num(medians, nan=0.0))
        column_valid.append(has_data)

    if x_levels.numel() < 2:
        raise RuntimeError("M20 height scan must contain at least two distinct forward-x columns.")

    profile = torch.stack(column_heights, dim=1)
    profile_valid = torch.stack(column_valid, dim=1)
    delta = profile[:, 1:] - profile[:, :-1]
    edge_x = 0.5 * (x_levels[1:] + x_levels[:-1])
    command = env.command_manager.get_command(command_name)
    command_x = command[:, 0]
    geometry = profile_valid[:, 1:] & profile_valid[:, :-1] & (delta.abs() >= min_edge_height)
    # A recently passed edge still matters while the rear wheels cross it.
    # Forward style rewards continue to require a command and an edge ahead.
    nearby = geometry & (edge_x.unsqueeze(0) >= -0.45) & (edge_x.unsqueeze(0) <= edge_max_x)
    edge_strength = ((delta.abs() - min_edge_height) / max(gate_ramp_height, 1.0e-6)).clamp(0.0, 1.0)
    terrain_gate = torch.where(nearby, edge_strength, torch.zeros_like(edge_strength)).max(dim=1).values
    candidates = (
        geometry
        & (edge_x.unsqueeze(0) >= edge_min_x)
        & (edge_x.unsqueeze(0) <= edge_max_x)
        & (command_x.unsqueeze(1) > command_threshold)
    )
    candidate_distance = torch.where(candidates, edge_x.abs().unsqueeze(0), torch.inf)
    selected_idx = candidate_distance.argmin(dim=1)
    has_edge = candidates.any(dim=1)
    selected_delta = delta.gather(1, selected_idx[:, None]).squeeze(1)
    selected_x = edge_x[selected_idx]
    lower_i = selected_idx
    upper_i = selected_idx + 1
    lower_z = profile.gather(1, lower_i[:, None]).squeeze(1)
    upper_z = profile.gather(1, upper_i[:, None]).squeeze(1)
    ramp = ((selected_delta.abs() - min_edge_height) / max(gate_ramp_height, 1.0e-6)).clamp(0.0, 1.0)
    gate = ramp * has_edge.to(ramp.dtype)
    up_gate = gate * (selected_delta > 0.0).to(gate.dtype)
    down_gate = gate * (selected_delta < 0.0).to(gate.dtype)

    context = {
        "gate": gate,
        "terrain_gate": terrain_gate,
        "up_gate": up_gate,
        "down_gate": down_gate,
        # Keep all visible ascending edges for style checks. The nearest edge
        # can still be the riser the front wheels have already climbed.
        "ascending_edges_x": edge_x,
        "edge_geometry": geometry,
        "ascending_edges_valid": nearby & (delta > 0.0),
        "ascending_edges_upper_z": profile[:, 1:],
        "ascending_edges_lower_z": profile[:, :-1],
        "ascending_edges_half_width": 0.5 * (x_levels[1:] - x_levels[:-1]),
        "edge_x": torch.where(has_edge, selected_x, torch.zeros_like(selected_x)),
        "edge_delta": torch.where(has_edge, selected_delta, torch.zeros_like(selected_delta)),
        "lower_z": torch.where(has_edge, lower_z, torch.zeros_like(lower_z)),
        "upper_z": torch.where(has_edge, upper_z, torch.zeros_like(upper_z)),
        "command_x": command_x,
    }
    stats = getattr(env, "_m20_stair_gate_stats", None)
    if stats is None or stats["steps"].shape[0] != env.num_envs:
        stats = {key: torch.zeros(env.num_envs, device=env.device)
                 for key in ("steps", "forward", "nearby")}
        setattr(env, "_m20_stair_gate_stats", stats)
    if hasattr(env, "episode_length_buf"):
        reset = env.episode_length_buf <= 1
        for value in stats.values():
            value[reset] = 0
    stats["steps"] += 1
    stats["forward"] += (gate > 0).to(stats["forward"].dtype)
    stats["nearby"] += (terrain_gate > 0).to(stats["nearby"].dtype)
    setattr(env, "_m20_stair_scan_context", {"counter": counter, "sensor_name": sensor_cfg.name, "context": context})
    return context


def _flat_surface_mask(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Require a valid, nearly level support patch under the body."""
    cache = getattr(env, "_m20_fast_flat_cache", None)
    tick = (int(getattr(env, 'common_step_counter', -1)), sensor_cfg.name)
    if TUNING.get("zero_stop_fast_stats", False) and cache is not None and cache[0] == tick:
        return cache[1]
    sensor = env.scene[sensor_cfg.name]
    starts = torch.as_tensor(sensor.ray_starts, device=env.device)
    if starts.ndim == 3:
        starts = starts[0]
    patch = (starts[:, 0].abs() <= 0.45) & (starts[:, 1].abs() <= 0.35)
    heights = sensor.data.ray_hits_w[:, patch, 2]
    valid = torch.isfinite(heights) & (heights.abs() < 1.0e6)
    low = torch.where(valid, heights, torch.inf).amin(dim=1)
    high = torch.where(valid, heights, -torch.inf).amax(dim=1)
    flat = valid.all(dim=1) & ((high - low) <= TUNING["flat_height_range_max"])
    if TUNING.get("zero_stop_fast_stats", False):
        env._m20_fast_flat_cache = (tick, flat)
    return flat


def _stair_activity_mask(env) -> torch.Tensor:
    """Include nearby risers and active step ledgers while crossing a flight."""
    scan = _stair_scan_context(env, SceneEntityCfg("height_scanner"))
    on_stairs = scan["terrain_gate"] > 0.0
    for name in ("_m20_stair_step_state", "_m20_ascent_state", "_m20_descent_entry_state"):
        state = getattr(env, name, None)
        if state is not None and "active" in state:
            on_stairs |= state["active"].bool()
    return on_stairs


# =============================================================================
# 4. Adjacent-riser event ledger, clearance credit and landing order
# =============================================================================

def _new_state(env):
    n, device = env.num_envs, env.device
    zeros = lambda *shape: torch.zeros((n, *shape), device=device)
    flags = lambda *shape: torch.zeros((n, *shape), device=device, dtype=torch.bool)
    state = {
        "active": flags(),
        "edge_xy": zeros(2),
        "heading": zeros(2),
        "source_z": zeros(),
        "target_z": zeros(),
        "age": zeros(),
        "radius": torch.full((n, 4), float("nan"), device=device),
        "front_prev_x": zeros(2),
        "front_prev_valid": flags(2),
        "front_safe": flags(2),
        "front_failed": flags(2),
        "front_face": flags(2),
        "front_face_event": flags(2),
        "front_lift": zeros(2),
        "front_lift_max": zeros(),
        "front_lift_gain": zeros(),
        "front_lift_lead": torch.full((n,), -1, device=device, dtype=torch.long),
        "front_lift_credit": zeros(),
        "lift_history_xy": zeros(64, 2),
        "lift_history_heading": zeros(64, 2),
        "lift_history_z": zeros(64),
        "lift_history_max": zeros(64),
        "lift_history_count": torch.zeros(n, device=device, dtype=torch.long),
        "lift_history_index": torch.zeros(n, device=device, dtype=torch.long),
        "targets_started": zeros(),
        "retry_count": zeros(),
        "forward_steps": zeros(),
        "retreat_steps": zeros(),
        "stationary_steps": zeros(),
        "abandoned_targets": zeros(),
        "front_physical_time": zeros(2),
        "front_event": flags(2),
        "front_lift_settlement": zeros(2),
        "front_violation_event": flags(),
        "front_expected": torch.full((n,), -1, device=device, dtype=torch.long),
        "front_count": torch.zeros(n, device=device, dtype=torch.long),
        "front_successes": zeros(),
        "front_violations": zeros(),
        "front_history_xy": zeros(64, 2),
        "front_history_heading": zeros(64, 2),
        "front_history_source_z": zeros(64),
        "front_history_target_z": zeros(64),
        "retired_valid": flags(),
        "retired_xy": zeros(2),
        "retired_z": zeros(),
        "rear_target_index": torch.full((n,), -1, device=device, dtype=torch.long),
        "rear_target_age": zeros(),
        "rear_count": torch.zeros(n, device=device, dtype=torch.long),
        "rear_expected": torch.full((n,), -1, device=device, dtype=torch.long),
        "rear_prev_x": zeros(2),
        "rear_prev_valid": flags(2),
        "rear_safe": flags(2),
        "rear_failed": flags(2),
        "rear_physical_time": zeros(2),
        "rear_event": flags(2),
        "rear_violation_event": flags(),
        "rear_successes": zeros(),
        "rear_violations": zeros(),
        "front_same_tread_paid": torch.full((n,), -1, device=device, dtype=torch.long),
        "rear_same_tread_paid": torch.full((n,), -1, device=device, dtype=torch.long),
        "front_same_tread_event": flags(),
        "rear_same_tread_event": flags(),
        "same_tread_time": zeros(2),
        "same_tread_riser": torch.full((n, 2), -1, device=device, dtype=torch.long),
        "same_tread_dwell": zeros(2),
        "same_tread_events_total": zeros(2),
        "same_tread_seconds_total": zeros(2),
    }

    state.update({
        "map_count": torch.zeros(n,device=device,dtype=torch.long),
        "map_xy": zeros(64,2), "map_heading": zeros(64,2),
        "map_z": zeros(64), "map_lower_z": zeros(64),
        "map_next_x": torch.full((n,64),torch.inf,device=device),
        "map_intermediate": flags(64), "map_landed": flags(64,4),
        "map_safe": flags(64,4), "map_failed": flags(64,4),
        "map_clear": flags(64,4), "map_crossed": flags(64,4),
        "map_face_struck": flags(64,4), "map_half_width": zeros(64),
        "map_band_min": zeros(64), "map_band_max": zeros(64),
        "map_stable_time": zeros(64,4), "map_paid": flags(64,2),
        "map_correct": flags(64,2), "map_side": torch.full((n,64,2),-1,device=device,dtype=torch.long),
        "map_duplicate_paid": flags(64,2), "map_prep_max": zeros(64,2),
        "rear_prep_initialized": flags(64), "rear_prep_max": zeros(64,2),
        "rear_prep_paid": zeros(64), "rear_prep_blocked_gain": zeros(),
        "rear_recovery_event": flags(), "rear_recoveries": zeros(),
        "rear_style_refund": zeros(), "rear_style_refund_total": zeros(),
        "map_rear_style_credit": zeros(64),
        "map_lead": torch.full((n,64,2),-1,device=device,dtype=torch.long),
        "map_dwell": zeros(64,2), "map_face_paid": flags(64,2),
        "map_flight": torch.full((n,64),-1,device=device,dtype=torch.long),
        "flight_count": torch.zeros(n,device=device,dtype=torch.long),
        "axis_last_slot": torch.full((n,2),-1,device=device,dtype=torch.long),
        "axis_target_slot": torch.full((n,2),-1,device=device,dtype=torch.long),
        "previous_wheel": zeros(4,3), "wheel_history_valid": flags(),
        "radius_calibrated": flags(4), "rear_lift_gain": zeros(),
        "rear_lift_credit": zeros(), "skip_total": zeros(2),
        "physical_total": zeros(2), "unsafe_total": zeros(2),
        "front_transition_successes": zeros(), "rear_transition_successes": zeros(),
        "axis_run": zeros(2), "axis_max_run": zeros(2),
        "duplicate_event_count": zeros(2),
        "map_task_paid": flags(64), "task_crossing_event": zeros(),
        "task_crossings_total": zeros(),
    })
    state["radius"].fill_(TUNING["wheel_radius"])
    return state


def _reset(state, reset):
    if not reset.any():
        return
    for key, value in state.items():
        if key in {"radius"}:
            value[reset] = TUNING["wheel_radius"]
        elif key in {"front_expected", "rear_expected", "rear_target_index", "front_lift_lead",
                     "front_same_tread_paid", "rear_same_tread_paid", "same_tread_riser",
                     "map_side", "map_lead", "axis_last_slot", "axis_target_slot", "map_flight"}:
            value[reset] = -1
        elif key == "map_next_x":
            value[reset] = torch.inf
        else:
            value[reset] = 0


def _contact_on_tread(wheel_z, target_z, radius, force_z, force_norm, x, edge_x, inset):
    return (
        torch.isfinite(radius)
        & ((wheel_z - target_z - radius).abs() <= 0.05)
        & (x > edge_x + inset)
        & (force_z > 0.5 * force_norm)
        & (force_z > 5.0)
    )


def _visible_higher_riser(context, scanner_xy, heading, last_xy, last_heading, last_z):
    """Check all visible risers on the recorded flight, not only the nearest."""
    if "ascending_edges_x" in context:
        x = context["ascending_edges_x"][None].expand(scanner_xy.shape[0], -1)
        valid = context["ascending_edges_valid"]
        upper = context["ascending_edges_upper_z"]
    else:
        # Compatibility with minimal scan-context fixtures/older callers.
        x = context["edge_x"][:, None]
        valid = (context["up_gate"] > 0)[:, None]
        upper = context["upper_z"][:, None]
    edges = scanner_xy[:, None] + heading[:, None] * x[..., None]
    delta = edges - last_xy[:, None]
    along = (delta * last_heading[:, None]).sum(dim=-1)
    lateral = (delta[..., 0] * last_heading[:, None, 1]
               - delta[..., 1] * last_heading[:, None, 0]).abs()
    rise = upper - last_z[:, None]
    aligned = (heading * last_heading).sum(dim=1) > 0.8
    return (valid & (along >= 0.15) & (along <= 0.50) & (lateral <= 0.20)
            & (rise >= 0.06) & (rise <= 0.30) & aligned[:, None]).any(dim=1)


def _register_ascent_risers(state, scan, scanner, heading, command):
    """Observe every ascending riser; registration never depends on actor success."""
    n, device = heading.shape[0], heading.device
    row = torch.arange(n, device=device)
    xs = scan.get("ascending_edges_x", scan["edge_x"])
    if "ascending_edges_x" in scan:
        valid = scan["ascending_edges_valid"]
        upper = scan["ascending_edges_upper_z"]
        lower = scan["ascending_edges_lower_z"]
        xy = scanner.data.pos_w[:, None, :2] + heading[:, None] * xs[None, :, None]
    else:
        valid = (scan["up_gate"] > 0)[:, None]
        upper, lower = scan["upper_z"][:, None], scan["lower_z"][:, None]
        xy = scanner.data.pos_w[:, None, :2] + heading[:, None] * xs[:, None, None]
    for j in range(valid.shape[1]):
        enabled = valid[:, j] & (command[:, 0] > TUNING["translation_threshold"])
        delta = xy[:, j, None] - state["map_xy"]
        along = (delta * state["map_heading"]).sum(-1)
        lateral_signed = (-delta[...,0]*state["map_heading"][...,1]
                          +delta[...,1]*state["map_heading"][...,0])
        known = torch.arange(64, device=device)[None] < state["map_count"][:, None]
        aligned = (state["map_heading"] * heading[:, None]).sum(-1) > 0.8
        half_band = TUNING["riser_observation_half_width"]
        overlap_band = (lateral_signed-half_band <= state["map_band_max"]) & (
            lateral_signed+half_band >= state["map_band_min"])
        match = known & aligned & (along.abs() < TUNING["riser_identity_distance"]) & overlap_band & (
            (state["map_z"] - upper[:, j, None]).abs() < TUNING["riser_height_tolerance"]) & (
            (state["map_lower_z"]-lower[:,j,None]).abs() < TUNING["riser_height_tolerance"])
        add = enabled & ~match.any(-1) & (state["map_count"] < 64)
        slot = torch.where(match.any(-1), match.long().argmax(-1), state["map_count"]).clamp_max(63)
        ids = torch.nonzero(add, as_tuple=False).squeeze(-1)
        state["map_xy"][ids, slot[ids]] = xy[ids, j]
        state["map_heading"][ids, slot[ids]] = heading[ids]
        state["map_z"][ids, slot[ids]] = upper[ids, j]
        state["map_lower_z"][ids, slot[ids]] = lower[ids, j]
        state["map_band_min"][ids,slot[ids]] = -half_band
        state["map_band_max"][ids,slot[ids]] = half_band
        matched = enabled & match.any(-1)
        mid = torch.nonzero(matched,as_tuple=False).squeeze(-1)
        band_center = lateral_signed[mid,slot[mid]]
        state["map_band_min"][mid,slot[mid]] = torch.minimum(state["map_band_min"][mid,slot[mid]],band_center-half_band)
        state["map_band_max"][mid,slot[mid]] = torch.maximum(state["map_band_max"][mid,slot[mid]],band_center+half_band)
        half_width = scan.get("ascending_edges_half_width")
        state["map_half_width"][ids,slot[ids]] = (
            half_width[j] if half_width is not None else TUNING["riser_default_half_width"])
        state["map_count"][ids] += 1
        # A successor confirms an intermediate tread and bounds its upper end.
        # Update both directions so rear observations/registration order cannot
        # change tread identity. No N*64*64 pairwise allocation is needed.
        own_xy = state["map_xy"][row, slot]
        own_h = state["map_heading"][row, slot]
        own_z = state["map_z"][row, slot]
        delta = own_xy[:, None] - state["map_xy"]
        dist = (delta * state["map_heading"]).sum(-1)
        cross = (delta[..., 0] * state["map_heading"][..., 1]
                 - delta[..., 1] * state["map_heading"][..., 0]).abs()
        known = torch.arange(64, device=device)[None] < state["map_count"][:, None]
        parallel = (state["map_heading"] * own_h[:, None]).sum(-1) > 0.8
        pred = known & parallel & (dist >= TUNING["riser_min_spacing"]) & (
            dist <= TUNING["riser_max_spacing"]) & (cross < TUNING["riser_lateral_corridor"]) & (
            (state["map_z"] - lower[:, j, None]).abs() < TUNING["riser_height_tolerance"]) & enabled[:, None]
        state["map_next_x"] = torch.minimum(state["map_next_x"], torch.where(pred, dist, torch.inf))
        dz = state["map_lower_z"] - own_z[:, None]
        successor = known & parallel & (-dist >= TUNING["riser_min_spacing"]) & (
            -dist <= TUNING["riser_max_spacing"]) & (cross < TUNING["riser_lateral_corridor"]) & (
            dz.abs() < TUNING["riser_height_tolerance"]) & enabled[:, None]
        next_dist = torch.where(successor, -dist, torch.inf).amin(-1)
        state["map_next_x"][row, slot] = torch.minimum(state["map_next_x"][row, slot], next_dist)
        neighbors = pred | successor
        connected = neighbors.any(-1)
        neighbor_slot = neighbors.long().argmax(-1)
        new_flight = add & ~connected
        state["map_flight"][ids, slot[ids]] = torch.where(connected[ids],
            state["map_flight"][ids, neighbor_slot[ids]], state["flight_count"][ids])
        state["flight_count"] += new_flight.long()
    state["map_intermediate"] = torch.isfinite(state["map_next_x"])


def ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name="base_velocity"):
    """Physical tread ledger shared by two independently learning alternating axles.

    Every observed riser is registered before either axle's reward is considered.
    Stable landings, clearance, style, duplicate history and preparation budgets
    are separate records. Missing a tread advances physical progress without
    rewarding the skipped/wrong step or permanently trapping the next target.
    """
    step = int(env.common_step_counter)
    cache = getattr(env, "_m20_ascent_cache", None)
    if cache is not None and cache["step"] == step:
        return cache["state"]
    scan = _stair_scan_context(env, sensor_cfg, command_name)
    robot, scanner = env.scene[asset_cfg.name], env.scene[sensor_cfg.name]
    sensor = env.scene.sensors[contact_sensor_cfg.name]
    state = getattr(env, "_m20_ascent_state", None)
    if state is None or state["active"].shape[0] != env.num_envs:
        state = _new_state(env)
        env._m20_ascent_state = state
    reset = env.episode_length_buf <= 1
    _reset(state, reset)
    wheels = robot.data.body_pos_w[:, asset_cfg.body_ids]
    force = sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids]
    norm = torch.linalg.vector_norm(force, dim=-1)
    loaded = (force[..., 2] > 5) & (force[..., 2] > 0.5 * norm)
    loaded &= sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0
    command = env.command_manager.get_command(command_name)
    enabled = command[:, 0] > TUNING["translation_threshold"]
    row = torch.arange(env.num_envs, device=env.device)
    forward = torch.zeros(env.num_envs, 3, device=env.device); forward[:, 0] = 1
    heading = quat_apply(yaw_quat(scanner.data.quat_w), forward)[:, :2]
    _register_ascent_risers(state, scan, scanner, heading, command)
    known = torch.arange(64, device=env.device)[None] < state["map_count"][:, None]
    # Known model wheel radius is available immediately. Sensor-based estimates
    # only calibrate it on flat loaded support, never disable landing attribution.
    hits = scanner.data.ray_hits_w
    valid_hits = torch.isfinite(hits).all(-1)
    distance = torch.linalg.vector_norm(wheels[:, :, None, :2] - hits[:, None, :, :2], dim=-1)
    distance = torch.where(valid_hits[:, None], distance, torch.inf)
    near, nearest = distance.min(-1)
    surface = hits[..., 2].gather(1, nearest)
    measured = wheels[..., 2] - surface
    local = distance < 0.16
    lo = torch.where(local, hits[:, None, :, 2], torch.inf).amin(-1)
    hi = torch.where(local, hits[:, None, :, 2], -torch.inf).amax(-1)
    flat = (local.sum(-1) >= 2) & ((hi-lo) < 0.025)
    calibrate = ~state["radius_calibrated"] & loaded & flat & (near < 0.08) & (
        (measured - TUNING["wheel_radius"]).abs() < 0.025)
    state["radius"][calibrate] = measured[calibrate]
    state["radius_calibrated"] |= calibrate
    relative = wheels[:, None, :, :2] - state["map_xy"][:, :, None]
    x = (relative * state["map_heading"][:, :, None]).sum(-1)
    cross = (relative[..., 0] * state["map_heading"][:, :, None, 1]
             - relative[..., 1] * state["map_heading"][:, :, None, 0]).abs()
    prev = state["previous_wheel"][:, None, :, :2] - state["map_xy"][:, :, None]
    prev_x = (prev * state["map_heading"][:, :, None]).sum(-1)
    width = state["map_half_width"][:,:,None]
    history = state["wheel_history_valid"][:,None,None] & known[:,:,None]
    advance = x > prev_x
    overlap = history & advance & (prev_x <= width) & (x >= -width)
    # The two rays bracket the real riser. Preserve clearance observed anywhere
    # in that interval; a midpoint can be 5cm too early to see the final lift.
    def interval_height(at_x):
        fraction = ((at_x-prev_x)/(x-prev_x).clamp_min(1e-6)).clamp(0,1)
        return state["previous_wheel"][:,None,:,2]+fraction*(
            wheels[:,None,:,2]-state["previous_wheel"][:,None,:,2])
    high = torch.maximum(interval_height(torch.maximum(prev_x,-width)),
                         interval_height(torch.minimum(x,width)))
    clearance = high-state["map_z"][:,:,None]-state["radius"][:,None]
    state["map_clear"] |= overlap & (clearance >= TUNING["riser_cross_clearance"])
    crossing = history & (prev_x < width) & (x >= width)
    state["map_crossed"] |= crossing
    face_horizontal = torch.linalg.vector_norm(force[...,:2],dim=-1)
    face_all = history & (x >= -0.12) & (x <= width+0.03) & (
        wheels[:,None,:,2] < state["map_z"][:,:,None]+state["radius"][:,None]+0.05) & (
        face_horizontal[:,None]>50) & (face_horizontal[:,None]>1.5*force[:,None,:,2].clamp_min(0))
    state["map_face_struck"] |= face_all
    state["map_face_struck"] &= ~(x < -width-0.06)
    state["map_clear"] &= ~face_all
    safe_cross = state["map_clear"] & state["map_crossed"] & ~state["map_face_struck"]
    state["retry_count"] += (safe_cross & state["map_failed"]).sum((1,2))
    state["map_failed"] |= crossing & ~safe_cross & known[:, :, None]
    state["map_failed"] &= ~safe_cross
    state["map_safe"] |= safe_cross
    state["map_safe"] &= ~face_all
    # A ray pair bounds the real riser. Settle a landing only beyond that
    # interval, so an early upper-tread contact cannot consume its credit
    # before the completed crossing is observable.
    landing_boundary = torch.maximum(width, torch.full_like(width, TUNING["tread_inset"]))
    occupancy = known[:, :, None] & loaded[:, None] & (x > landing_boundary) & (
        x < state["map_next_x"][:, :, None] - TUNING["tread_inset"]) & (
        cross < TUNING["tread_observation_corridor"]) & (
        (wheels[:, None, :, 2] - state["map_z"][:, :, None] - state["radius"][:, None]).abs()
        <= TUNING["tread_height_tolerance"])
    state["map_stable_time"] = torch.where(occupancy, state["map_stable_time"] + env.step_dt, 0)
    stable = state["map_stable_time"] >= TUNING["landing_time"]
    state["map_landed"] |= stable
    for key in ("front_event", "rear_event", "front_face_event", "front_violation_event",
                "rear_violation_event", "front_same_tread_event", "rear_same_tread_event"):
        state[key].zero_()
    state["front_lift_gain"].zero_(); state["rear_lift_gain"].zero_()
    state["front_lift_settlement"].zero_()
    state["rear_recovery_event"].zero_(); state["rear_style_refund"].zero_()
    face_force = torch.linalg.vector_norm(force[:, :2, :2], dim=-1)
    face = known[:,:,None] & (x[:,:,:2] >= -0.12) & (x[:,:,:2] <= 0.03) & (
        wheels[:,None,:2,2] < state["map_z"][:,:,None] + state["radius"][:,None,:2] + 0.05) & (
        face_force[:,None] > 50) & (face_force[:,None] > 1.5*force[:,None,:2,2].clamp_min(0))
    new_face = face & ~state["map_face_paid"] & enabled[:,None,None]
    state["front_face_event"] = new_face.any(1)
    state["map_face_paid"] |= new_face
    for axle, name in enumerate(("front", "rear")):
        sl = slice(2*axle, 2*axle+2)
        pair = state["map_landed"][:, :, sl]
        first = pair.any(-1) & ~state["map_paid"][:, :, axle] & known
        candidate = torch.where(first, state["map_z"], torch.inf).argmin(-1)
        available = first.any(-1) & enabled
        landed = pair[row, candidate]
        single = landed.sum(-1) == 1
        side = landed.long().argmax(-1)
        last = state["axis_last_slot"][:, axle]
        candidate_flight = state["map_flight"][row, candidate]
        new_flight = available & (last >= 0) & (
            candidate_flight != state["map_flight"][row,last.clamp_min(0)])
        last = torch.where(new_flight, -1, last)
        state[name+"_expected"][new_flight] = -1
        state["axis_run"][new_flight,axle] = 0
        last_z = state["map_z"][row, last.clamp_min(0)]
        last_xy = state["map_xy"][row, last.clamp_min(0)]
        last_h = state["map_heading"][row, last.clamp_min(0)]
        advance = ((state["map_xy"][row, candidate] - last_xy) * last_h).sum(-1)
        source = state["map_lower_z"][row, candidate]
        adjacent = (last < 0) | ((source-last_z).abs() < TUNING["riser_height_tolerance"]) & (
            advance >= TUNING["riser_min_spacing"]) & (advance <= TUNING["riser_max_spacing"])
        # If a first observed landing skips a known lower tread, it is a skip,
        # including when neither front leg ever paid a completion reward.
        lower_before = known & (state["map_z"] < state["map_z"][row, candidate, None] - 0.05)
        lower_before &= state["map_flight"] == candidate_flight[:,None]
        lower_before &= (last < 0)[:,None] | (state["map_z"] > last_z[:,None]+0.05)
        lower_unvisited = (lower_before & ~state["map_paid"][:, :, axle]).any(-1)
        skipped = (~adjacent | lower_unvisited) & available
        expected = state[name+"_expected"]
        ordered = (expected < 0) | (side == expected)
        safe = state["map_safe"][row, candidate, sl].gather(1, side[:, None]).squeeze(-1)
        correct = available & single & ordered & safe & ~skipped
        violation = available & ~correct
        previous = last.clamp_min(0)
        previous_side = state["map_side"][row,previous,axle]
        previous_pair = state["map_landed"][row,previous,sl]
        previous_safe = state["map_safe"][row,previous,sl].gather(
            1,previous_side.clamp_min(0)[:,None]).squeeze(-1)
        # Pay the main style reward for a demonstrated alternating transition,
        # rather than paying the first foot before its partner's plan is known.
        # This prevents HL1,HR1,HL2,HR2 from collecting a completion per tread
        # and only facing a temporally discounted same-tread penalty later.
        transition = correct & (last >= 0) & (previous_side >= 0) & (
            previous_pair.sum(-1)==1) & previous_safe & (side!=previous_side) & (
            ~state["map_duplicate_paid"][row,previous,axle])
        if axle == 1:
            # A safe single landing beyond a duplicated rear tread starts a
            # recovery. Its partner must still support that previous tread.
            # This is not counted as a strict alternating transition.
            recovery = correct & (last >= 0) & state["map_duplicate_paid"][row,previous,axle] & (
                stable[row,previous,3-side]) & state["map_intermediate"][row,candidate]
            state["rear_recovery_event"] = recovery
            state["rear_recoveries"] += recovery.float()
            style_credit = (transition.float()*TUNING["rear_completion_weight"]
                            + recovery.float()*TUNING["rear_recovery_weight"])
            state["map_rear_style_credit"][row,candidate] += style_credit
        state[name+"_event"] = landed & transition[:, None]
        state[name+"_transition_successes"] += transition.float()
        state[name+"_violation_event"] = violation
        state[name+"_successes"] += correct.float()
        state[name+"_violations"] += violation.float()
        state["skip_total"][:, axle] += skipped.float()
        state["physical_total"][:, axle] += available.float()
        state["unsafe_total"][:, axle] += (available & ~safe).float()
        state["axis_run"][:, axle] = torch.where(available,
            torch.where(correct, state["axis_run"][:, axle]+1, 0), state["axis_run"][:, axle])
        state["axis_max_run"][:, axle] = torch.maximum(state["axis_max_run"][:, axle], state["axis_run"][:, axle])
        ids = torch.nonzero(available, as_tuple=False).squeeze(-1)
        slots = candidate[ids]
        state["map_paid"][ids, slots, axle] = True
        state["map_correct"][ids, slots, axle] = correct[ids]
        state["map_side"][ids, slots, axle] = torch.where(single[ids], side[ids], -1)
        # Physical progress is monotonic; late lower contacts cannot rewind the
        # sequence or unlock more preparation credit.
        further = available & ((last < 0) | (state["map_z"][row, candidate] > last_z+0.05))
        state["axis_last_slot"][further, axle] = candidate[further]
        state[name+"_expected"][further] = torch.where(single[further], 1-side[further], -1)
        state[name+"_count"] += available.long()
        # All historical treads remain checked, including after the first leg
        # has left. Each axle/tread pays once, and both axles add independently.
        duplicate = pair.all(-1) & state["map_intermediate"] & known
        new_duplicate = duplicate & ~state["map_duplicate_paid"][:, :, axle] & enabled[:, None]
        if axle == 1:
            # A later catch-up on the same tread invalidates the earlier style
            # credit. Return exactly that credit, once, in addition to the
            # existing same-tread cost; task crossing credit stays independent.
            refund = (new_duplicate*state["map_rear_style_credit"]).sum(-1)
            state["rear_style_refund"] = refund
            state["rear_style_refund_total"] += refund
        state["map_duplicate_paid"][:, :, axle] |= new_duplicate
        state["map_correct"][:, :, axle] &= ~duplicate
        state[name+"_same_tread_event"] = new_duplicate.any(-1)
        state["duplicate_event_count"][:, axle] = new_duplicate.sum(-1)
        state["same_tread_events_total"][:, axle] += new_duplicate.sum(-1)
        state["axis_run"][new_duplicate.any(-1), axle] = 0
        current_last = state["axis_last_slot"][:,axle]
        ambiguous_pair = (current_last >= 0) & new_duplicate[row,current_last.clamp_min(0)]
        # After both feet occupy one tread, either may restart the sequence.
        # Requiring an invisible historical leader would make a memoryless
        # actor's identical current posture receive contradictory targets.
        state[name+"_expected"][ambiguous_pair] = -1
        current_pair = stable[:, :, sl].all(-1) & state["map_intermediate"] & known & enabled[:, None]
        state["map_dwell"][:, :, axle] += current_pair.float()*env.step_dt
        dwelling = current_pair & (state["map_dwell"][:, :, axle] > TUNING["same_tread_grace"])
        state["same_tread_dwell"][:, axle] = dwelling.any(-1).float()
        state["same_tread_seconds_total"][:, axle] += dwelling.any(-1)*env.step_dt
        # Geometric next target per axle, independent of the other axle queue.
        last = state["axis_last_slot"][:, axle]
        last_z = state["map_z"][row, last.clamp_min(0)]
        last_flight = state["map_flight"][row,last.clamp_min(0)]
        same_flight = state["map_flight"] == last_flight[:,None]
        wheel_x = x[:,:,sl].mean(-1)
        new_flight_ahead = ~same_flight & (wheel_x > -TUNING["prep_distance"]) & (wheel_x < 0.03)
        ahead = known & ~state["map_paid"][:, :, axle] & (
            (last < 0)[:, None] | (same_flight & (state["map_z"] > last_z[:, None]+0.05)) | new_flight_ahead)
        target = torch.where(ahead, state["map_z"], torch.inf).argmin(-1)
        has_target = ahead.any(-1)
        changed = has_target & (target != state["axis_target_slot"][:, axle])
        state["targets_started"] += changed.float() if axle == 0 else 0
        state["axis_target_slot"][:, axle] = torch.where(has_target, target, -1)
        expected = state[name+"_expected"]
        lead = state["map_lead"][row, target, axle]
        lead = torch.where(expected >= 0, expected, lead)
        hfrac = (wheels[:, sl, 2] - state["radius"][:, sl] - state["map_lower_z"][row, target, None]) / (
            state["map_z"][row, target, None] - state["map_lower_z"][row, target, None] + 0.02).clamp_min(0.06)
        tx = x[row, target, sl]
        progress = hfrac.clamp(0,1) * ((tx+TUNING["prep_distance"])/TUNING["prep_distance"]).clamp(0,1)
        other_support = loaded.sum(-1,keepdim=True) - loaded[:, sl].long()
        eligible = has_target[:, None] & enabled[:, None] & ~loaded[:, sl] & (other_support >= 2) & (
            tx > -TUNING["prep_distance"]) & (tx < 0.03)
        if axle == 1:
            # Aim at the next tread, rather than paying ever higher pre-edge lifts.
            target_x = torch.maximum(state["map_half_width"][row,target],
                wheels.new_full((env.num_envs,),TUNING["tread_inset"])) + TUNING["tread_inset"]
            target_x = torch.minimum(target_x,state["map_next_x"][row,target]-TUNING["tread_inset"])
            bottom = wheels[:,sl,2]-state["radius"][:,sl]
            target_z = state["map_z"][row,target,None]
            source_z = state["map_lower_z"][row,target,None]
            rise = (target_z-source_z).clamp_min(0.06)
            raw = rear_target_score(tx,bottom,target_x[:,None],target_z)
            raw *= ((bottom-source_z)/rise).clamp(0,1)
            eligible = has_target[:,None] & enabled[:,None] & ~loaded[:,sl] & (other_support>=2) & (
                bottom > source_z+TUNING["rear_swing_clearance"]) & (
                bottom <= target_z+TUNING["rear_swing_target_ceiling"]) & (
                tx > -TUNING["prep_distance"]) & (
                tx < state["map_next_x"][row,target,None]-TUNING["tread_inset"])
            gain = rear_preparation_gain(state,row,target,has_target,raw,eligible,expected)
            state["rear_lift_gain"] = gain
            state["rear_lift_credit"] += gain
            continue
        progress = torch.where(eligible, progress, 0)
        lead = torch.where(lead >= 0, lead, progress.argmax(-1))
        current = progress.gather(1,lead[:, None]).squeeze(-1)
        # A target's initial posture earns nothing. Only subsequent new maxima
        # earn credit; returning to an old pose never pays again.
        gain = (current-state["map_prep_max"][row, target, axle]).clamp_min(0)
        budget = (1+state[name+"_successes"]-state[name+"_lift_credit"]).clamp_min(0)
        gain = torch.minimum(gain,budget)*has_target
        state["map_lead"][row, target, axle] = torch.where(gain > 0,lead,state["map_lead"][row,target,axle])
        state["map_prep_max"][row,target,axle] = torch.maximum(state["map_prep_max"][row,target,axle],current)
        state[name+"_lift_gain"] = gain
        state[name+"_lift_credit"] += gain
    # Task credit is independent of stylistic alternation. Both axles must
    # have actually supported the upper tread and the whole robot must pass
    # the riser inside its corridor. History alone or sideways bypass is not
    # enough. Pay once per physical riser, including after a same-tread error.
    task_event = ascent_task_crossing_events(state, known, x, cross, wheels, loaded,
                                            robot.data.root_pos_w, robot.data.root_quat_w) & enabled[:, None]
    state["map_task_paid"] |= task_event
    state["task_crossing_event"] = task_event.sum(-1).float()
    state["task_crossings_total"] += state["task_crossing_event"]
    state["previous_wheel"] = wheels.clone()
    state["wheel_history_valid"].fill_(True)
    # Retain unpaid tread history on departure, but do not apply stair posture
    # shaping on unrelated flat terrain. Rear feet can keep the phase active
    # after the forward scan has cleared, while an actual lateral exit cannot.
    nearby = known[:,:,None] & (cross < TUNING["tread_observation_corridor"]) & (
        x > -TUNING["prep_distance"]) & (
        x < torch.where(torch.isfinite(state["map_next_x"]),
                        state["map_next_x"],TUNING["riser_max_spacing"])[:,:,None]+0.20) & (
        wheels[:,None,:,2]-state["radius"][:,None] >= state["map_lower_z"][:,:,None]-0.10) & (
        wheels[:,None,:,2]-state["radius"][:,None] <= state["map_z"][:,:,None]+0.50)
    state["active"] = enabled & (state["axis_target_slot"] >= 0).any(-1) & nearby.any((1,2))
    target = torch.where(state["axis_target_slot"][:,0]>=0,state["axis_target_slot"][:,0],
        torch.where(state["axis_target_slot"][:,1]>=0,state["axis_target_slot"][:,1],state["axis_last_slot"][:,0])).clamp_min(0)
    state["edge_xy"] = state["map_xy"][row,target].clone()
    state["heading"] = state["map_heading"][row,target].clone()
    state["source_z"] = state["map_lower_z"][row,target].clone()
    state["target_z"] = state["map_z"][row,target].clone()
    state["forward_steps"] += state["active"].float()
    state["retreat_steps"] += (state["active"] & (robot.data.root_lin_vel_b[:,0] < -0.05)).float()
    state["stationary_steps"] += (state["active"] & (robot.data.root_lin_vel_b[:,0].abs() < 0.05)).float()
    env._m20_ascent_cache = {"step":step,"state":state}
    return state



def rear_preparation_gain(state, row, target, has_target, raw, eligible, expected):
    """Per-foot geometric high-water marks, independent of support eligibility.

    Initial pose, unloading, changes of supporting partners and changing the
    selected leg never unlock previously observed scores. Total credit is at
    most one unit per physical target, across both rear feet and all retries.
    """
    previous = state["rear_prep_max"][row,target]
    initialized = state["rear_prep_initialized"][row,target]
    delta = (raw-previous).clamp_min(0)*initialized[:,None]*has_target[:,None]
    potential = delta*eligible
    side = torch.where(expected>=0,expected,potential.argmax(-1))
    gain = potential.gather(1,side[:,None]).squeeze(-1)
    gain = torch.minimum(gain,(1-state["rear_prep_paid"][row,target]).clamp_min(0))
    state["rear_prep_blocked_gain"] += (delta.sum(-1)-gain).clamp_min(0)
    state["rear_prep_max"][row,target] = torch.where(has_target[:,None],
        torch.maximum(previous,raw),previous)
    state["rear_prep_initialized"][row,target] |= has_target
    state["rear_prep_paid"][row,target] += gain
    state["map_lead"][row,target,1] = torch.where(gain>0,side,state["map_lead"][row,target,1])
    return gain


def rear_target_score(x, bottom_z, target_x, target_z):
    """Bounded next-tread proximity; over-lift and overshoot reduce the score."""
    error = ((x-target_x)/TUNING["rear_target_x_scale"]).square()
    error += ((bottom_z-target_z)/TUNING["rear_target_z_scale"]).square()
    return torch.exp(-0.5*error)


def _update_rear_swing_stats(env,down,b,g,mask,cost):
    state = getattr(env,"_m20_rear_swing_stats",None)
    if state is None:
        state = {key:torch.zeros(env.num_envs,2,device=env.device) for key in
                 ("samples","b_sum","g_sum","bad","b_max","g_max")}
        env._m20_rear_swing_stats = state
    reset = env.episode_length_buf <= 1
    for key,value in state.items(): value[reset] = 0
    state["samples"] += mask
    state["b_sum"] += b.clamp_min(0)*mask; state["g_sum"] += g.clamp_min(0)*mask
    state["bad"] += (cost>0)&mask
    state["b_max"] = torch.maximum(state["b_max"],torch.where(mask,b,0))
    state["g_max"] = torch.maximum(state["g_max"],torch.where(mask,g,0))

def _update_rear_angle_histograms(env,b,g,loaded,swing):
    hist = getattr(env,"_m20_rear_angle_histograms",None)
    if hist is None:
        hist = {f"{phase}_{coord}":torch.zeros(env.num_envs,2,19,device=env.device)
                for phase in ("loaded","swing") for coord in ("b","g")}
        env._m20_rear_angle_histograms = hist
    reset = env.episode_length_buf <= 1
    for phase,mask in (("loaded",loaded),("swing",swing)):
        for coord,angle in (("b",b),("g",g)):
            value = hist[f"{phase}_{coord}"]; value[reset] = 0
            bins = (torch.rad2deg(angle).clamp(0,90)/5).long().clamp_max(18)
            value.scatter_add_(2,bins[:,:,None],mask.float()[:,:,None])


def stair_rear_wheel_lift(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name="base_velocity"):
    """One bounded preparation budget per rear target; no cyclic leg-dance pay."""
    gain = ascent_state(env,asset_cfg,sensor_cfg,contact_sensor_cfg,command_name)["rear_lift_gain"]
    return gain * _stair_forward_command_gate(env, command_name) / env.step_dt




# =============================================================================
# 5. Stair task rewards, posture, contact and energy costs
# =============================================================================

def stair_terrain_aware_orientation_cost(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    flat_weight: float = 50.0,
    stair_weight: float = 5.0,
) -> torch.Tensor:
    """Relax posture near stairs and permit pitch during descent rear follow."""
    asset = env.scene[asset_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    gravity = asset.data.projected_gravity_b
    state = getattr(env, "_m20_stair_step_state", None)
    descent_rear_follow = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if state is not None:
        descent_rear_follow = (
            state["active"] & ~state["up"] & state["front_reached"].all(dim=1)
            & ~state["rear_support_paid"].all(dim=1)
        )
    stair_gate = torch.maximum(context["terrain_gate"], descent_rear_follow.float())
    if TUNING.get("zero_stop_v4", False):
        raw_zero = env.command_manager.get_command(command_name).abs().amax(-1) <= 1.e-4
        flat_stop = raw_zero & _zero_stop_support_context(env)["flat"]
        stair_gate = torch.where(flat_stop, 0., stair_gate)
        descent_rear_follow &= ~flat_stop
    effective_weight = flat_weight * (1.0 - stair_gate) + stair_weight * stair_gate
    # During rear follow, pitch is needed to transfer support down a step.
    tilt = torch.where(descent_rear_follow, gravity[:, 1].square(), gravity[:, :2].square().sum(dim=1))
    return effective_weight * tilt


def stair_joint_pos_cost(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
    stair_sensor_cfg: SceneEntityCfg,
    stair_penalty_scale: float,
    disable_turn_side_cmd: bool = False,
    ang_cmd_threshold: float = 0.1,
    y_cmd_threshold: float = 0.1,
    xy_norm_max: float = 0.1,
    xz_norm_max: float = 0.1,
    turn_penalty_scale: float = 0.0,
) -> torch.Tensor:
    """Keep the M20 posture cost, relaxing it near a visible stair edge."""
    cost = joint_pos_penalty(
        env, command_name, asset_cfg, stand_still_scale, velocity_threshold, command_threshold
    )
    if TUNING.get("zero_stop_v2", False):
        cmd = env.command_manager.get_command(command_name)
        raw_zero = cmd.abs().amax(-1) <= 1.e-4
        slow = torch.linalg.vector_norm(env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2], dim=-1) <= velocity_threshold
        # Cancel only the inherited stationary amplification, not joint limits,
        # terrain posture envelopes or any nonzero command's original cost.
        release = raw_zero & slow & _stair_activity_mask(env)
        if TUNING.get("zero_stop_v4", False):
            release &= ~_zero_stop_support_context(env)["flat"]
        cost = torch.where(release, cost / max(float(stand_still_scale), 1.e-6), cost)
    if disable_turn_side_cmd:
        cmd = env.command_manager.get_command(command_name)
        turn = (cmd[:, 2].abs() > ang_cmd_threshold) & (torch.linalg.vector_norm(cmd[:, :2], dim=1) < xy_norm_max)
        side = (cmd[:, 1].abs() > y_cmd_threshold) & (
            torch.linalg.vector_norm(cmd[:, [0, 2]], dim=1) < xz_norm_max
        )
        # Sideways relaxation remains available. Pure yaw keeps a weak posture
        # reference rather than removing the hip/knee constraint altogether.
        scale = torch.where(turn, turn_penalty_scale, 1.0)
        cost = cost * torch.where(side, 0.0, scale)
    gate = _stair_scan_context(env, stair_sensor_cfg, command_name)["terrain_gate"]
    if TUNING.get("zero_stop_v4", False):
        raw_zero = env.command_manager.get_command(command_name).abs().amax(-1) <= 1.e-4
        gate = torch.where(raw_zero & _zero_stop_support_context(env)["flat"], 0., gate)
    if TUNING.get('zero_stop_v5', False):
        # A fixed flat default pose must not compete with a captured support
        # pose. Limits/contact costs remain active, and motion is untouched.
        cost = cost*(env.command_manager.get_command(command_name).abs().amax(-1) > 1.e-4)
    return cost * (1.0 - (1.0 - stair_penalty_scale) * gate)


def _stair_new_progress(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, command_name: str):
    """New forward distance along the heading fixed at the first stair edge.

    A per-episode high-water mark makes reversing and crossing the same ground
    again worth zero. The returned value has velocity units for rate rewards.
    """
    counter = int(getattr(env, "common_step_counter", -1))
    cache = getattr(env, "_m20_stair_progress_cache", None)
    if cache is not None and cache["counter"] == counter:
        return cache["rate"], cache["distance"]

    asset = env.scene[asset_cfg.name]
    scanner = env.scene[sensor_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    state = getattr(env, "_m20_stair_progress_state", None)
    if state is None or state["origin"].shape[0] != env.num_envs:
        state = {
            "origin": torch.zeros((env.num_envs, 2), device=env.device),
            "direction": torch.zeros((env.num_envs, 2), device=env.device),
            "maximum": torch.zeros(env.num_envs, device=env.device),
            "desired_lateral": torch.zeros(env.num_envs, device=env.device),
            "active": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }
        setattr(env, "_m20_stair_progress_state", state)
    if hasattr(env, "episode_length_buf"):
        reset = env.episode_length_buf <= 1
        state["active"][reset] = False
        state["maximum"][reset] = 0.0
        state["desired_lateral"][reset] = 0.0

    start = (context["gate"] > 0) & ~state["active"]
    forward = torch.zeros((env.num_envs, 3), device=env.device)
    forward[:, 0] = 1.0
    heading = quat_apply(yaw_quat(scanner.data.quat_w), forward)[:, :2]
    state["origin"][start] = asset.data.root_pos_w[start, :2]
    state["direction"][start] = heading[start]
    state["desired_lateral"][start] = 0.0
    state["active"][start] = True

    command = env.command_manager.get_command(command_name)
    commanded_body = torch.zeros((env.num_envs, 3), device=env.device)
    commanded_body[:, :2] = command[:, :2]
    commanded_world = quat_apply(yaw_quat(scanner.data.quat_w), commanded_body)[:, :2]
    route = state["direction"]
    commanded_side = -commanded_world[:, 0] * route[:, 1] + commanded_world[:, 1] * route[:, 0]
    state["desired_lateral"] += commanded_side * env.step_dt * state["active"].float()

    displacement = ((asset.data.root_pos_w[:, :2] - state["origin"]) * state["direction"]).sum(dim=1)
    new_maximum = torch.maximum(state["maximum"], displacement.clamp(min=0.0))
    gained = new_maximum - state["maximum"]
    state["maximum"] = new_maximum
    rate = context["gate"] * (gained / env.step_dt).clamp(max=1.0)
    setattr(env, "_m20_stair_progress_cache", {"counter": counter, "rate": rate, "distance": new_maximum})
    return rate, new_maximum


def _stair_route_quality(env, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity") -> torch.Tensor:
    """Gate stair rewards by a route fixed when the first riser is seen."""
    _stair_new_progress(env, SceneEntityCfg("robot"), sensor_cfg, command_name)
    context = _stair_scan_context(env, sensor_cfg, command_name)
    state = env._m20_stair_progress_state
    root_xy = env.scene["robot"].data.root_pos_w[:, :2]
    offset = root_xy - state["origin"]
    direction = state["direction"]
    lateral_actual = -offset[:, 0] * direction[:, 1] + offset[:, 1] * direction[:, 0]
    lateral = torch.abs(lateral_actual - state.get("desired_lateral", 0.0))
    excess = (lateral - 0.25).clamp_min(0.0)
    quality = torch.exp(-torch.square(excess / 0.25))
    quality = torch.where(lateral >= 0.75, torch.zeros_like(quality), quality)
    active = state["active"] & (context["command_x"] > 0.10)
    return torch.where(active, quality, torch.ones_like(quality))


def _stair_forward_command_gate(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Enable forward stair shaping only for a current forward-x command."""
    command = env.command_manager.get_command(command_name)
    return (command[:, 0] > TUNING["translation_threshold"]).to(command.dtype)


def _ascent_direction_context(env, asset_cfg, sensor_cfg, contact_sensor_cfg,
                              command_name="base_velocity"):
    """Track ascent drift against a command-integrated heading, once per step.

    Current body yaw must not redefine the desired route: a policy can otherwise
    turn and drive sideways while retaining perfect body-frame velocity tracking.
    Cross-track increments use the previous commanded path tangent, so slowing
    down on a stair does not create an along-track position error penalty.
    """
    step = int(env.common_step_counter)
    cached = getattr(env, "_m20_ascent_direction_cache", None)
    if cached is not None and cached["step"] == step:
        return cached
    robot = env.scene[asset_cfg.name]
    scanner = env.scene[sensor_cfg.name]
    sensor = env.scene.sensors[contact_sensor_cfg.name]
    scan = _stair_scan_context(env, sensor_cfg, command_name)
    command = env.command_manager.get_command(command_name)
    root_xy = robot.data.root_pos_w[:, :2]
    forward = torch.zeros(env.num_envs, 3, device=env.device)
    forward[:, 0] = 1.0
    body_heading = quat_apply(yaw_quat(robot.data.root_quat_w), forward)[:, :2]
    actual_yaw = torch.atan2(body_heading[:, 1], body_heading[:, 0])
    moving_up = command[:, 0] > TUNING["translation_threshold"]
    state = getattr(env, "_m20_ascent_direction_state", None)
    if state is None or state["engaged"].shape[0] != env.num_envs:
        zero = lambda *shape: torch.zeros(env.num_envs, *shape, device=env.device)
        state = dict(engaged=zero().bool(), reference_yaw=actual_yaw.clone(),
                     previous_xy=root_xy.clone(), previous_command=command.clone(),
                     cross_track=zero(), edge_xy=zero(2), edge_heading=zero(2),
                     top_z=zero(), stats={key: zero() for key in
                         ("samples", "lateral_sum", "speed_sum", "heading_sum", "heading_signed_sum",
                          "outside_samples", "heading_outside_samples", "yaw_rate_sum", "yaw_rate_signed_sum", "yaw_rate_max",
                          "cost_sum", "max_lateral", "eligible_samples", "tail_samples",
                          "max_heading", "right_peak", "left_peak", "heading_over10", "heading_over20",
                          "exit_heading_sum", "exit_samples", "yaw_tracking_sum", "yaw_tracking_samples")})
        env._m20_ascent_direction_state = state
    reset = env.episode_length_buf <= 1
    state["engaged"][reset] = False
    state["cross_track"][reset] = 0.0
    for value in state["stats"].values():
        value[reset] = 0.0

    # Remember the furthest/highest ascending riser. A turned-away or missing
    # scan cannot clear the latch. Only four loaded wheels beyond the known
    # riser on its upper platform can end a flight.
    edges_valid = scan["ascending_edges_valid"]
    seen_up = edges_valid.any(dim=1)
    heights = torch.where(edges_valid, scan["ascending_edges_upper_z"], -torch.inf)
    top_z, edge_i = heights.max(dim=1)
    start = ~state["engaged"] & seen_up & moving_up
    update_edge = seen_up & (start | (state["engaged"] & (top_z > state["top_z"] + 0.02)))
    scanner_heading = quat_apply(yaw_quat(scanner.data.quat_w), forward)[:, :2]
    edge_x = scan["ascending_edges_x"][edge_i]
    state["edge_xy"][update_edge] = (scanner.data.pos_w[:, :2]
        + scanner_heading * edge_x[:, None])[update_edge]
    state["edge_heading"][update_edge] = scanner_heading[update_edge]
    state["top_z"][update_edge] = top_z[update_edge]
    previous = state["previous_command"]
    reference = state["reference_yaw"]
    mid_yaw = reference + 0.5 * previous[:, 2] * env.step_dt
    tangent = torch.stack((torch.cos(mid_yaw) * previous[:, 0] - torch.sin(mid_yaw) * previous[:, 1],
                           torch.sin(mid_yaw) * previous[:, 0] + torch.cos(mid_yaw) * previous[:, 1]), -1)
    tangent /= torch.linalg.vector_norm(tangent, dim=1).clamp_min(1e-6)[:, None]
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), -1)
    integrate = state["engaged"] & (previous[:, 0] > TUNING["translation_threshold"]) & ~reset
    state["cross_track"] += ((root_xy - state["previous_xy"]) * normal).sum(-1) * integrate
    reference = reference + previous[:, 2] * env.step_dt * state["engaged"]
    reference = torch.atan2(torch.sin(reference), torch.cos(reference))
    # User changes in travel direction rebase accumulated cross-track error;
    # actual body yaw changes alone never do so. Finishing a commanded in-place
    # turn establishes the new body-relative forward direction.
    old_norm = torch.linalg.vector_norm(previous[:, :2], dim=1)
    new_norm = torch.linalg.vector_norm(command[:, :2], dim=1)
    similarity = (previous[:, :2] * command[:, :2]).sum(-1) / (old_norm * new_norm).clamp_min(1e-6)
    direction_change = (old_norm > 0.1) & (new_norm > 0.1) & (
        similarity < np.cos(TUNING["direction_command_change"]))
    after_turn = (old_norm <= TUNING["translation_threshold"]) & (
        previous[:, 2].abs() > TUNING["yaw_threshold"]) & moving_up
    heading_state = getattr(env, "_m20_zero_hold_heading_state", None)
    straight_up = moving_up & (command[:, 1].abs() <= TUNING["translation_threshold"]) & (
        command[:, 2].abs() <= TUNING["yaw_threshold"]
    )
    start_yaw = actual_yaw if heading_state is None else torch.where(
        straight_up, heading_state["straight_target_yaw"], actual_yaw
    )
    reference = torch.where(start, start_yaw, reference)
    reference = torch.where(direction_change & straight_up, start_yaw, reference)
    reference = torch.where(after_turn, actual_yaw, reference)
    state["cross_track"][start | direction_change | after_turn] = 0.0
    state["engaged"] |= start
    state["reference_yaw"] = reference

    wheels = robot.data.body_pos_w[:, asset_cfg.body_ids]
    forces = sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids]
    loaded = (forces[..., 2] > 5.0) & (
        forces[..., 2] > 0.5 * torch.linalg.vector_norm(forces, dim=-1))
    loaded &= sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] >= TUNING["direction_exit_contact_time"]
    past = ((wheels[..., :2] - state["edge_xy"][:, None]) * state["edge_heading"][:, None]).sum(-1)
    on_top = (wheels[..., 2] - state["top_z"][:, None] - TUNING["wheel_radius"]).abs() <= TUNING["direction_exit_height_tolerance"]
    clear = state["engaged"] & ~seen_up & loaded.all(-1) & on_top.all(-1) & (
        past > TUNING["direction_exit_inset"]).all(-1)
    heading_error = torch.atan2(torch.sin(actual_yaw - reference), torch.cos(actual_yaw - reference))
    flight_exit = clear & state["engaged"]
    state["stats"]["exit_heading_sum"] += heading_error.abs() * flight_exit
    state["stats"]["exit_samples"] += flight_exit
    measured_yaw_rate, _ = _automatic_ascent_heading_rate(env, command, robot, scan)
    yaw_rate = measured_yaw_rate.abs()
    # A commanded turn leading onto descending terrain also ends ascent shaping;
    # an uncommanded turn away from the scan must retain its cost.
    clear |= state["engaged"] & (scan["down_gate"] > 0) & ~seen_up & (
        heading_error.abs() < TUNING["direction_heading_tolerance"])
    state["engaged"][clear] = False
    path = torch.stack((torch.cos(reference) * command[:, 0] - torch.sin(reference) * command[:, 1],
                        torch.sin(reference) * command[:, 0] + torch.cos(reference) * command[:, 1]), -1)
    path /= new_norm.clamp_min(1e-6)[:, None]
    velocity = robot.data.root_lin_vel_w[:, :2]
    lateral_speed = -velocity[:, 0] * path[:, 1] + velocity[:, 1] * path[:, 0]
    active = state["engaged"] & moving_up
    straight_active = active & (command[:, 1].abs() <= TUNING["translation_threshold"]) & (
        command[:, 2].abs() <= TUNING["yaw_threshold"])
    effective_command = heading_hold_velocity_commands(env, command_name)
    yaw_tracking_error = measured_yaw_rate - effective_command[:, 2]
    effective_body = torch.zeros_like(robot.data.root_lin_vel_b)
    effective_body[:, :2] = effective_command[:, :2]
    effective_world = quat_apply(robot.data.root_quat_w, effective_body)[:, :2]
    desired_lateral_speed = -effective_world[:, 0] * path[:, 1] + effective_world[:, 1] * path[:, 0]
    _, ascent_straight = _automatic_ascent_heading_rate(env, command, robot, scan)
    lateral_speed_error = torch.where(ascent_straight, lateral_speed - desired_lateral_speed, lateral_speed)
    def excess(value, tolerance, scale):
        u = (value.abs() - tolerance).clamp_min(0.0) / scale
        huber = 2.0 * (torch.sqrt(1.0 + u.square()) - 1.0)
        return huber / (1.0 + huber / TUNING["direction_soft_cap"])
    cost = excess(state["cross_track"], TUNING["direction_lateral_tolerance"], TUNING["direction_lateral_scale"])
    cost += TUNING["direction_velocity_mix"] * excess(lateral_speed_error, TUNING["direction_speed_tolerance"], TUNING["direction_speed_scale"])
    cost += TUNING["direction_heading_mix"] * excess(heading_error, TUNING["direction_heading_tolerance"], TUNING["direction_heading_scale"])
    cost *= active
    stats = state["stats"]
    stats["samples"] += active
    stats["eligible_samples"] += moving_up
    stats["tail_samples"] += active & (state["cross_track"].abs() >
        TUNING["direction_lateral_tolerance"] + TUNING["direction_lateral_scale"]*TUNING["direction_soft_cap"])
    stats["lateral_sum"] += state["cross_track"].abs() * active
    stats["speed_sum"] += lateral_speed.abs() * active
    stats["heading_sum"] += heading_error.abs() * active
    stats["heading_signed_sum"] += heading_error * active
    stats["outside_samples"] += (state["cross_track"].abs() > TUNING["direction_lateral_tolerance"]) & active
    stats["heading_outside_samples"] += (heading_error.abs() > TUNING["direction_heading_tolerance"]) & active
    stats["yaw_rate_sum"] += yaw_rate * active
    stats["yaw_rate_signed_sum"] += measured_yaw_rate * active
    stats["yaw_rate_max"] = torch.maximum(stats["yaw_rate_max"], yaw_rate * active)
    stats["cost_sum"] += cost
    stats["max_lateral"] = torch.maximum(stats["max_lateral"], state["cross_track"].abs() * active)
    stats["max_heading"] = torch.maximum(stats["max_heading"], heading_error.abs() * straight_active)
    stats["right_peak"] = torch.maximum(stats["right_peak"], (-heading_error).clamp_min(0.0) * straight_active)
    stats["left_peak"] = torch.maximum(stats["left_peak"], heading_error.clamp_min(0.0) * straight_active)
    stats["heading_over10"] += (heading_error.abs() > np.deg2rad(10.0)) & active
    stats["heading_over20"] += (heading_error.abs() > np.deg2rad(20.0)) & active
    stats["yaw_tracking_sum"] += yaw_tracking_error.abs() * straight_active
    stats["yaw_tracking_samples"] += straight_active
    state["previous_xy"] = root_xy.clone()
    state["previous_command"] = command.clone()
    cached = dict(step=step, cost=cost, active=active, cross_track=state["cross_track"].clone(),
                  lateral_speed=lateral_speed, heading_error=heading_error,
                  straight_active=straight_active, yaw_tracking_error=yaw_tracking_error)
    env._m20_ascent_direction_cache = cached
    return cached


def stair_ascent_direction_cost(env, asset_cfg, sensor_cfg, contact_sensor_cfg,
                                command_name="base_velocity"):
    """Bounded ascent cost for cross-track drift, lateral speed and unwanted yaw."""
    return _ascent_direction_context(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)["cost"]


def stair_ascent_heading_cost(env, asset_cfg, sensor_cfg, contact_sensor_cfg,
                              command_name="base_velocity"):
    """Keep large ascent heading errors distinguishable to PPO."""
    context = _ascent_direction_context(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    error = context["heading_error"].abs()
    excess = (error - TUNING["ascent_heading_tolerance"]).clamp_min(0.0)
    excess = excess / TUNING["ascent_heading_scale"]
    huber = 2.0 * (torch.sqrt(1.0 + excess.square()) - 1.0)
    # Linear tails keep 20/30 degree errors distinguishable without the
    # exploding scale of a squared error. PPO compares these rewards; it
    # does not differentiate through this expression.
    return huber * context["active"]


def stair_ascent_yaw_tracking_cost(env, asset_cfg, sensor_cfg, contact_sensor_cfg,
                                  command_name="base_velocity"):
    """Penalize failure to follow the same corrective yaw target seen by the actor."""
    c = _ascent_direction_context(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    u = c["yaw_tracking_error"] / TUNING["ascent_yaw_tracking_scale"]
    return 2.0 * (torch.sqrt(1.0 + u.square()) - 1.0) * c["straight_active"]


def straight_flat_heading_cost(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Keep the latched route after the last riser and on level approaches."""
    command = env.command_manager.get_command(command_name)
    robot = env.scene["robot"]
    state, actual_yaw, _ = _zero_hold_heading_context(env, command, robot)
    straight = (
        (command[:, 0].abs() > TUNING["translation_threshold"])
        & (command[:, 1].abs() <= TUNING["translation_threshold"])
        & (command[:, 2].abs() <= TUNING["yaw_threshold"])
    )
    error = torch.atan2(
        torch.sin(actual_yaw - state["straight_target_yaw"]),
        torch.cos(actual_yaw - state["straight_target_yaw"]),
    ).abs()
    scaled = (error - TUNING["direction_heading_tolerance"]).clamp_min(0.0) / TUNING["direction_heading_scale"]
    huber = 2.0 * (torch.sqrt(1.0 + scaled.square()) - 1.0)
    flat = _flat_surface_mask(env, SceneEntityCfg("height_scanner"))
    return huber / (1.0 + huber / TUNING["direction_heading_soft_cap"]) * (straight & flat).to(error.dtype)


def _ascent_direction_metrics(env, env_ids):
    state = getattr(env, "_m20_ascent_direction_state", None)
    if state is None:
        return {"samples": torch.zeros((), device=env.device)}
    s = {key: value[env_ids].sum() for key, value in state["stats"].items()}
    count = s["samples"].clamp_min(1)
    observed = (state["stats"]["yaw_tracking_samples"][env_ids] > 0).sum().clamp_min(1)
    return dict(samples=s["samples"], lateral_error_mean_m=s["lateral_sum"] / count,
                lateral_error_max_m=state["stats"]["max_lateral"][env_ids].max(),
                lateral_speed_mean_m_s=s["speed_sum"] / count,
                heading_error_mean_deg=torch.rad2deg(s["heading_sum"] / count),
                heading_error_signed_mean_deg=torch.rad2deg(s["heading_signed_sum"] / count),
                heading_outside_tolerance_fraction=s["heading_outside_samples"] / count,
                yaw_rate_mean_rad_s=s["yaw_rate_sum"] / count,
                yaw_rate_signed_mean_rad_s=s["yaw_rate_signed_sum"] / count,
                yaw_rate_max_rad_s=state["stats"]["yaw_rate_max"][env_ids].max(),
                outside_tolerance_fraction=s["outside_samples"] / count,
                cost_mean=s["cost_sum"] / count,
                activation_fraction=s["samples"] / s["eligible_samples"].clamp_min(1),
                large_error_fraction=s["tail_samples"] / count,
                heading_error_peak_mean_deg=torch.rad2deg(state["stats"]["max_heading"][env_ids].sum() / observed),
                right_heading_peak_mean_deg=torch.rad2deg(state["stats"]["right_peak"][env_ids].sum() / observed),
                left_heading_peak_mean_deg=torch.rad2deg(state["stats"]["left_peak"][env_ids].sum() / observed),
                heading_over10deg_fraction=s["heading_over10"] / count,
                heading_over20deg_fraction=s["heading_over20"] / count,
                flight_exit_samples=s["exit_samples"],
                flight_exit_heading_error_mean_deg=torch.rad2deg(s["exit_heading_sum"] / s["exit_samples"].clamp_min(1)),
                yaw_tracking_error_mean_rad_s=s["yaw_tracking_sum"] / s["yaw_tracking_samples"].clamp_min(1))


def ascent_direction_metrics(env, env_ids):
    core = {
        "samples", "heading_error_mean_deg", "heading_error_signed_mean_deg",
        "heading_outside_tolerance_fraction", "heading_over10deg_fraction",
        "heading_over20deg_fraction", "yaw_rate_mean_rad_s", "yaw_rate_signed_mean_rad_s",
        "lateral_error_mean_m", "flight_exit_samples", "flight_exit_heading_error_mean_deg",
        "yaw_tracking_error_mean_rad_s",
    }
    result = _ascent_direction_metrics(env, env_ids)
    straight_metrics = ascent_straight_motion_metrics(env, env_ids)
    result.update(straight_metrics)
    core.update(straight_metrics)
    return export_teacher_metrics(result, core, _detailed_teacher_diagnostics(env))


def stair_forward_progress(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward only new ground covered while a stair edge is visible."""
    rate, _ = _stair_new_progress(env, asset_cfg, sensor_cfg, command_name)
    return rate * _stair_route_quality(env, sensor_cfg, command_name) * _stair_forward_command_gate(env, command_name)


def _stair_front_sequence_events(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return per-riser ascent events and command-aware route quality."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    return state, _stair_route_quality(env, sensor_cfg, command_name)


def stair_pure_yaw_air_time(
    env, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
    threshold: float, stair_sensor_cfg: SceneEntityCfg,
    cmd_threshold: float = 0.1, foot_height_threshold: float = 0.05,
) -> torch.Tensor:
    """Keep the original turning reward out of commanded stair traversal."""
    from .rewards import feet_air_time_ang_z_cmd_M20
    native = feet_air_time_ang_z_cmd_M20(
        env, command_name, sensor_cfg, threshold, cmd_threshold,
        foot_height_threshold, asset_cfg,
    )
    command = env.command_manager.get_command(command_name)
    terrain_gate = _stair_scan_context(env, stair_sensor_cfg, command_name)["terrain_gate"]
    state = getattr(env, "_m20_stair_step_state", None)
    if state is not None:
        terrain_gate = torch.maximum(terrain_gate, state["active"].float())
    ascent = getattr(env, "_m20_ascent_state", None)
    if ascent is not None:
        terrain_gate = torch.maximum(terrain_gate, ascent["active"].float())
    stair_forward = (terrain_gate > 0) & (torch.linalg.vector_norm(command[:, :2], dim=1) > 0.10)
    return native * (~stair_forward).float()


def stair_front_wheel_lift(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> torch.Tensor:
    """Pay bounded lift-height improvement before the expected leg lands."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    return state["front_lift_gain"] * _stair_forward_command_gate(env, command_name) / env.step_dt


def stair_forward_motion_cost(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Bounded forward-command shortfall near a visible or tracked stair.

    A target stays active after retreat makes it disappear from the scan.
    Backward commands and deliberate turns do not activate this cost.
    """
    asset = env.scene[asset_cfg.name]
    scanner = env.scene[sensor_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    gate = context["gate"].clone()
    forward = torch.zeros((env.num_envs, 3), device=env.device)
    forward[:, 0] = 1.0
    heading = quat_apply(yaw_quat(scanner.data.quat_w), forward)[:, :2]
    for name in ("_m20_ascent_state", "_m20_stair_step_state"):
        state = getattr(env, name, None)
        if state is not None:
            aligned = (heading * state["heading"]).sum(dim=1) > 0.8
            nearby = torch.linalg.vector_norm(asset.data.root_pos_w[:, :2] - state["edge_xy"], dim=1) <= 1.5
            gate = torch.maximum(gate, (state["active"] & aligned & nearby).float())
    command = env.command_manager.get_command(command_name)
    traversing = (command[:, 0] > 0.1) & (command[:, 0] > command[:, 1].abs())
    shortfall = ((command[:, 0] - asset.data.root_lin_vel_b[:, 0])
                 / command[:, 0].clamp_min(0.3)).clamp(0.0, 2.0)
    return gate * traversing.float() * shortfall


def _zero_hold_heading_context(env, command, robot):
    """Update and return the latched yaw target used during stationary commands."""
    zero_command = (
        (torch.linalg.vector_norm(command[:, :2], dim=1) <= TUNING["translation_threshold"])
        & (command[:, 2].abs() <= TUNING["yaw_threshold"])
    )
    forward = torch.zeros((env.num_envs, 3), device=env.device)
    forward[:, 0] = 1.0
    body_heading = quat_apply(yaw_quat(robot.data.root_quat_w), forward)[:, :2]
    actual_yaw = torch.atan2(body_heading[:, 1], body_heading[:, 0])

    # Latch the body heading on the first zero-command frame after motion. Keep
    # it fixed until another translation/turn command arrives. This gives the
    # stop state its own heading target instead of reusing a stair-only target.
    state = getattr(env, "_m20_zero_hold_heading_state", None)
    moving_command = ~zero_command
    straight_command = (
        (command[:, 0].abs() > TUNING["translation_threshold"])
        & (command[:, 1].abs() <= TUNING["translation_threshold"])
        & (command[:, 2].abs() <= TUNING["yaw_threshold"])
    )
    if state is None or state["active"].shape[0] != env.num_envs:
        state = {
            "active": zero_command.clone(),
            "was_moving": moving_command.clone(),
            "target_yaw": actual_yaw.clone(),
            "hold_start_step": torch.full((env.num_envs,), int(env.common_step_counter),
                                           device=env.device, dtype=torch.long),
            "reset_step": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
            "straight_active": straight_command.clone(),
            "straight_target_yaw": actual_yaw.clone(),
            "straight_origin_xy": robot.data.root_pos_w[:, :2].clone(),
            "stop_origin_xy": robot.data.root_pos_w[:, :2].clone(),
            "stop_position_latched": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
            "straight_direction": torch.sign(command[:, 0]),
            "error_sum": torch.zeros(env.num_envs, device=env.device),
            "error_signed_sum": torch.zeros(env.num_envs, device=env.device),
            "samples": torch.zeros(env.num_envs, device=env.device),
            "max_error": torch.zeros(env.num_envs, device=env.device),
            "base_speed_sum": torch.zeros(env.num_envs, device=env.device),
            "base_speed_max": torch.zeros(env.num_envs, device=env.device),
            "yaw_rate_sum": torch.zeros(env.num_envs, device=env.device),
            "yaw_rate_signed_sum": torch.zeros(env.num_envs, device=env.device),
            "yaw_rate_max": torch.zeros(env.num_envs, device=env.device),
            "wheel_speed_sum": torch.zeros(env.num_envs, device=env.device),
            "wheel_speed_max": torch.zeros(env.num_envs, device=env.device),
            "hold_xy": robot.data.root_pos_w[:, :2].clone(),
            "previous_hold_xy": robot.data.root_pos_w[:, :2].clone(),
            "position_latched": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
            "quiet_run_s": torch.zeros(env.num_envs, device=env.device),
            "settled_seen": {group: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
                             for group in ("flat", "stair")},
            "quiet_seen": {group: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
                           for group in ("flat", "stair")},
            "settled_stats": {
                group: {key: torch.zeros(env.num_envs, device=env.device) for key in (
                    "samples", "success", "base_speed", "wheel_speed", "heading", "yaw_rate", "drift", "path", "holds",
                    "quiet_samples", "quiet_holds")}
                for group in ("flat", "stair")
            },
        }
        env._m20_zero_hold_heading_state = state
    reset = (env.episode_length_buf <= 1) & (state["reset_step"] != int(env.common_step_counter))
    state["reset_step"][reset] = int(env.common_step_counter)
    state["active"][reset] = zero_command[reset]
    state["was_moving"][reset] = moving_command[reset]
    state["target_yaw"][reset] = actual_yaw[reset]
    state["hold_start_step"][reset] = int(env.common_step_counter)
    state["straight_active"][reset] = straight_command[reset]
    state["straight_target_yaw"][reset] = actual_yaw[reset]
    state["straight_origin_xy"][reset] = robot.data.root_pos_w[reset, :2]
    state["straight_direction"][reset] = torch.sign(command[reset, 0])
    for key in (
        "error_sum", "error_signed_sum", "samples", "max_error", "base_speed_sum", "base_speed_max",
        "yaw_rate_sum", "yaw_rate_signed_sum", "yaw_rate_max", "wheel_speed_sum", "wheel_speed_max",
    ):
        state[key][reset] = 0.0
    for stats in state["settled_stats"].values():
        for value in stats.values():
            value[reset] = 0.0
    state["position_latched"][reset | moving_command] = False
    state["stop_position_latched"][reset | moving_command] = False
    for seen in state["settled_seen"].values():
        seen[reset | moving_command] = False
    for seen in state["quiet_seen"].values():
        seen[reset | moving_command] = False
    state["quiet_run_s"][reset | moving_command] = 0.0
    just_stopped = state["was_moving"] & zero_command & ~reset
    state["target_yaw"][just_stopped] = actual_yaw[just_stopped]
    state["hold_start_step"][just_stopped] = int(env.common_step_counter)
    state["active"][just_stopped] = True
    state["active"][moving_command] = False
    # Also support episodes that begin with a zero command.
    first_zero = zero_command & ~state["active"] & ~state["was_moving"] & ~reset
    state["target_yaw"][first_zero] = actual_yaw[first_zero]
    state["hold_start_step"][first_zero] = int(env.common_step_counter)
    state["active"][first_zero] = True
    state["was_moving"] = moving_command
    straight_start = straight_command & ~state["straight_active"] & ~reset
    reversed_direction = straight_command & state["straight_active"] & (
        torch.sign(command[:, 0]) != state["straight_direction"])
    path_start = straight_start if TUNING.get("v1_motion_baseline", False) else straight_start | reversed_direction
    state["straight_target_yaw"][path_start] = actual_yaw[path_start]
    state["straight_origin_xy"][path_start] = robot.data.root_pos_w[path_start, :2]
    state["straight_direction"] = torch.sign(command[:, 0])
    state["straight_active"] = straight_command.clone()
    hold_active = zero_command & state["active"]
    # Ignore the braking displacement, then latch once for the whole hold.
    # Repeated observation/reward consumers never move this reference.
    age = (int(env.common_step_counter) - state["hold_start_step"]) * env.step_dt
    latch = hold_active & (age >= TUNING["zero_hold_brake_grace_s"] - 1.e-7) & ~state["stop_position_latched"]
    state["stop_origin_xy"][latch] = robot.data.root_pos_w[latch, :2]
    state["stop_position_latched"] |= latch
    return state, actual_yaw, hold_active


def _descent_direction_context(env, command_name: str = "base_velocity"):
    """Latch a straight descent heading and accumulate per-episode yaw telemetry."""
    step = int(env.common_step_counter)
    cached = getattr(env, "_m20_descent_direction_cache", None)
    if cached is not None and cached["step"] == step:
        return cached

    command = env.command_manager.get_command(command_name)
    robot = env.scene["robot"]
    scan = _stair_scan_context(env, SceneEntityCfg("height_scanner"), command_name)
    forward = torch.zeros((env.num_envs, 3), device=env.device)
    forward[:, 0] = 1.0
    body_heading = quat_apply(yaw_quat(robot.data.root_quat_w), forward)[:, :2]
    actual_yaw = torch.atan2(body_heading[:, 1], body_heading[:, 0])
    straight_command = (
        (command[:, 0] > TUNING["translation_threshold"])
        & (command[:, 1].abs() <= TUNING["translation_threshold"])
        & (command[:, 2].abs() <= TUNING["yaw_threshold"])
    )
    on_descent = scan["down_gate"] > 0.0
    eligible = on_descent & straight_command

    state = getattr(env, "_m20_descent_direction_state", None)
    if state is None or state["active"].shape[0] != env.num_envs:
        zero = lambda: torch.zeros(env.num_envs, device=env.device)
        state = {
            "active": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "reference_yaw": actual_yaw.clone(),
            "reference_xy": robot.data.root_pos_w[:, :2].clone(),
            "quiet_time": zero(),
            "stats": {key: zero() for key in (
                "samples", "heading_sum", "heading_signed_sum", "heading_outside",
                "yaw_rate_sum", "yaw_rate_signed_sum", "yaw_rate_max",
            )},
        }
        env._m20_descent_direction_state = state

    reset = env.episode_length_buf <= 1
    state["active"][reset] = False
    state["reference_yaw"][reset] = actual_yaw[reset]
    state["reference_xy"][reset] = robot.data.root_pos_w[reset, :2]
    state["quiet_time"][reset] = 0.0
    for value in state["stats"].values():
        value[reset] = 0.0

    start = eligible & ~state["active"]
    state["reference_yaw"][start] = actual_yaw[start]
    state["reference_xy"][start] = robot.data.root_pos_w[start, :2]
    state["active"] |= start
    state["quiet_time"] = torch.where(on_descent, torch.zeros_like(state["quiet_time"]),
                                       state["quiet_time"] + env.step_dt * state["active"])
    intentional_turn = (command[:, 1].abs() > TUNING["translation_threshold"]) | (
        command[:, 2].abs() > TUNING["yaw_threshold"]
    ) | (command[:, 0] < -TUNING["translation_threshold"])
    expired = state["active"] & (state["quiet_time"] > 0.30)
    state["active"] &= ~intentional_turn & ~expired

    signed_heading_error = torch.atan2(
        torch.sin(actual_yaw - state["reference_yaw"]),
        torch.cos(actual_yaw - state["reference_yaw"]),
    )
    heading_error = signed_heading_error.abs()
    signed_yaw_rate = robot.data.root_ang_vel_b[:, 2]
    yaw_rate = signed_yaw_rate.abs()
    measured = eligible & state["active"] & ~reset
    stats = state["stats"]
    stats["samples"] += measured
    stats["heading_sum"] += heading_error * measured
    stats["heading_signed_sum"] += signed_heading_error * measured
    stats["heading_outside"] += (heading_error > TUNING["direction_heading_tolerance"]) & measured
    stats["yaw_rate_sum"] += yaw_rate * measured
    stats["yaw_rate_signed_sum"] += signed_yaw_rate * measured
    stats["yaw_rate_max"] = torch.maximum(stats["yaw_rate_max"], yaw_rate * measured)

    cost_u = (heading_error - TUNING["direction_heading_tolerance"]).clamp_min(0.0)
    cost_u = cost_u / TUNING["direction_heading_scale"]
    huber = 2.0 * (torch.sqrt(1.0 + cost_u.square()) - 1.0)
    cost = huber / (1.0 + huber / TUNING["direction_soft_cap"])
    cached = {"step": step, "active": state["active"], "measured": measured,
              "heading_error": heading_error, "yaw_rate": yaw_rate, "cost": cost * measured}
    env._m20_descent_direction_cache = cached
    return cached


def stair_descent_direction_cost(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Penalize heading drift from the latched route while descending straight."""
    return _descent_direction_context(env, command_name)["cost"]


def stair_descent_lateral_cost(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Track lateral position and velocity against the latched descent lane."""
    context = _descent_direction_context(env, command_name)
    state = env._m20_descent_direction_state
    robot = env.scene["robot"]
    yaw = state["reference_yaw"]
    lateral = torch.stack((-torch.sin(yaw), torch.cos(yaw)), -1)
    displacement = robot.data.root_pos_w[:, :2] - state["reference_xy"]
    position_error = (displacement * lateral).sum(-1).abs()
    lateral_speed = (robot.data.root_lin_vel_w[:, :2] * lateral).sum(-1).abs()
    position = ((position_error - .04).clamp_min(0.) / .10)
    speed = ((lateral_speed - .03).clamp_min(0.) / .12)
    return (position.square() + .5 * speed.square()) * context["measured"]


def _descent_direction_metrics(env, env_ids):
    state = getattr(env, "_m20_descent_direction_state", None)
    if state is None:
        zero = torch.zeros((), device=env.device)
        return {
            "samples": zero,
            "heading_error_mean_deg": zero,
            "heading_error_signed_mean_deg": zero,
            "heading_outside_tolerance_fraction": zero,
            "yaw_rate_mean_rad_s": zero,
            "yaw_rate_signed_mean_rad_s": zero,
            "yaw_rate_max_rad_s": zero,
        }
    stats = state["stats"]
    samples = stats["samples"][env_ids].sum()
    count = samples.clamp_min(1.0)
    return {
        "samples": samples,
        "heading_error_mean_deg": torch.rad2deg(stats["heading_sum"][env_ids].sum() / count),
        "heading_error_signed_mean_deg": torch.rad2deg(stats["heading_signed_sum"][env_ids].sum() / count),
        "heading_outside_tolerance_fraction": stats["heading_outside"][env_ids].sum() / count,
        "yaw_rate_mean_rad_s": stats["yaw_rate_sum"][env_ids].sum() / count,
        "yaw_rate_signed_mean_rad_s": stats["yaw_rate_signed_sum"][env_ids].sum() / count,
        "yaw_rate_max_rad_s": stats["yaw_rate_max"][env_ids].max(),
    }


def descent_direction_metrics(env, env_ids):
    result = _descent_direction_metrics(env, env_ids)
    result.update(_descent_motion_metrics(env, env_ids))
    return export_teacher_metrics(result, result.keys())


def _automatic_descent_mask(env, command, scan):
    """Detect straight descent without changing ascent or manual commands."""
    descending = scan.get("down_gate", torch.zeros(env.num_envs, device=env.device)) > 0.0
    ledger = getattr(env, "_m20_stair_step_state", None)
    if ledger is not None:
        descending = descending | (ledger["active"].bool() & ~ledger["up"].bool())
    return (descending & (command[:, 0] > TUNING["translation_threshold"])
            & (command[:, 1].abs() <= TUNING["translation_threshold"])
            & (command[:, 2].abs() <= TUNING["yaw_threshold"]))


def _automatic_ascent_heading_rate(env, command, robot, scan=None):
    """Use horizontal heading rate for automatic straight ascent only.

    Body-z angular velocity is not the derivative of world yaw when the base
    pitches and rolls on a riser. Preserve the existing body-z convention for
    flat travel, descent, zero holds and explicit turning commands. The actor's
    angular-velocity observation also retains its original body frame.
    """
    if TUNING.get("v1_motion_baseline", False):
        return robot.data.root_ang_vel_b[:, 2], torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if scan is None:
        scan = _stair_scan_context(env, SceneEntityCfg("height_scanner"), "base_velocity")
    ascending = scan["up_gate"] > 0.0
    ascent = getattr(env, "_m20_ascent_state", None)
    if ascent is not None:
        # A stale last-riser latch must not enable ascent corrections after
        # the user has turned around and is now approaching descending stairs.
        ascending = ascending | (ascent["active"].bool() & (scan["down_gate"] <= 0.0))
    straight_up = (
        (command[:, 0] > TUNING["translation_threshold"])
        & (command[:, 1].abs() <= TUNING["translation_threshold"])
        & (command[:, 2].abs() <= TUNING["yaw_threshold"])
        & ascending
    )
    forward_body = torch.zeros_like(robot.data.root_ang_vel_b)
    forward_body[:, 0] = 1.0
    forward = quat_apply(robot.data.root_quat_w, forward_body)
    omega = quat_apply(robot.data.root_quat_w, robot.data.root_ang_vel_b)
    horizontal_squared = forward[:, :2].square().sum(-1)
    # Heading is undefined near a vertical forward axis. Keep the original
    # body-rate convention outside the ordinary stair posture envelope.
    straight_up = straight_up & (horizontal_squared >= 0.25)
    heading_rate = omega[:, 2] - forward[:, 2] * (
        forward[:, :2] * omega[:, :2]
    ).sum(-1) / horizontal_squared.clamp_min(0.25)
    return torch.where(straight_up, heading_rate, robot.data.root_ang_vel_b[:, 2]), straight_up


def heading_hold_velocity_commands(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Expose a yaw-rate correction in the existing command observation channels.

    The actor input shape stays unchanged. Straight travel keeps one heading
    through approach, stairs, and exit; zero holds latch the release heading.
    """
    raw_command = env.command_manager.get_command(command_name)
    command = raw_command.clone()
    robot = env.scene["robot"]
    state, actual_yaw, hold_active = _zero_hold_heading_context(env, raw_command, robot)

    straight = (
        (raw_command[:, 0].abs() > TUNING["translation_threshold"])
        & (raw_command[:, 1].abs() <= TUNING["translation_threshold"])
        & (raw_command[:, 2].abs() <= TUNING["yaw_threshold"])
    )
    target_yaw = torch.where(straight, state["straight_target_yaw"], state["target_yaw"])
    correction_active = hold_active | straight
    scan = _stair_scan_context(env, SceneEntityCfg("height_scanner"), command_name)
    stair_correction = correction_active & (scan["terrain_gate"] > 0.0)
    for name in ("_m20_stair_step_state", "_m20_ascent_state", "_m20_descent_entry_state"):
        stair_state = getattr(env, name, None)
        if stair_state is not None and "active" in stair_state:
            stair_correction |= correction_active & stair_state["active"].bool()

    heading_error = torch.atan2(
        torch.sin(target_yaw - actual_yaw),
        torch.cos(target_yaw - actual_yaw),
    )
    yaw_rate, world_heading_rate_active = _automatic_ascent_heading_rate(env, raw_command, robot, scan)
    automatic_descent = _automatic_descent_mask(env, raw_command, scan)
    forward_axis = torch.zeros_like(robot.data.root_ang_vel_b)
    forward_axis[:, 0] = 1.0
    automatic_descent &= quat_apply(robot.data.root_quat_w, forward_axis)[:, :2].square().sum(-1) >= 0.25
    path_active = world_heading_rate_active | automatic_descent
    stop_feedback = hold_active & stair_correction & state["stop_position_latched"]
    kp = torch.where(
        stair_correction,
        torch.full_like(heading_error, TUNING["stair_heading_hold_kp"]),
        torch.full_like(heading_error, TUNING["heading_hold_kp"]),
    )
    kd = torch.where(
        stair_correction,
        torch.full_like(heading_error, TUNING["stair_heading_hold_kd"]),
        torch.full_like(heading_error, TUNING["heading_hold_kd"]),
    )
    max_yaw_rate = torch.where(
        stair_correction,
        torch.full_like(heading_error, TUNING["stair_heading_hold_max_yaw_rate"]),
        torch.full_like(heading_error, TUNING["heading_hold_max_yaw_rate"]),
    )
    deadband = torch.where(
        stair_correction,
        torch.full_like(heading_error, TUNING["stair_heading_hold_deadband"]),
        torch.full_like(heading_error, TUNING["heading_hold_deadband"]),
    )
    if TUNING.get("v1_motion_baseline", False):
        # Same direct body-z PD convention as deployment's legacy_v1 mode.
        # No motion-route vy, world-yaw conversion, filtering or XY stop input.
        active = correction_active & ((heading_error.abs() > deadband) | (yaw_rate.abs() > 0.03))
        correction = (kp * heading_error - kd * yaw_rate).clamp(-max_yaw_rate, max_yaw_rate)
        command[:, 2] = torch.where(correction_active, torch.where(active, correction, 0.0), raw_command[:, 2])
        if TUNING.get("stair_path_feedback", False):
            from .path.stair_path_mdp import apply_path_command
            command = apply_path_command(env, command, raw_command, command_name)
        return command
    # Observation and several rewards request this command in the same tick.
    # Recompute from one pre-tick snapshot, so filters never advance once per
    # consumer. This also handles a command resample in the same env tick.
    mode = torch.where(straight, 1, torch.where(hold_active, 2, 0))
    filt = getattr(env, "_m20_heading_command_filter", None)
    step = int(env.common_step_counter)
    if filt is None or filt["mode"].shape != mode.shape:
        filt = dict(step=-1, mode=torch.zeros_like(mode), filtered_rate=yaw_rate.clone(),
                    straight_direction=torch.zeros_like(raw_command[:, 0]),
                    world_heading_rate_active=torch.zeros_like(world_heading_rate_active),
                    correction=torch.zeros_like(yaw_rate), lateral_correction=torch.zeros_like(yaw_rate),
                    path_active=torch.zeros_like(path_active),
                    stop_correction=torch.zeros(env.num_envs, 2, device=env.device),
                    active=torch.zeros_like(correction_active))
        env._m20_heading_command_filter = filt
    if filt["step"] != step:
        filt["before"] = {key: filt[key].clone() for key in
                          ("mode", "filtered_rate", "correction", "active", "world_heading_rate_active",
                           "lateral_correction", "straight_direction", "path_active", "stop_correction")}
        filt["step"] = step
    before = filt["before"]
    reset = env.episode_length_buf <= 1
    entering = (mode != before["mode"]) & correction_active
    reversing = straight & (before["mode"] == 1) & (
        torch.sign(raw_command[:, 0]) != before["straight_direction"])
    frame_changed = world_heading_rate_active != before["world_heading_rate_active"]
    rate_before = torch.where(reset | entering | reversing | frame_changed, yaw_rate, before["filtered_rate"])
    alpha = 1.0 - float(np.exp(-env.step_dt / TUNING["heading_hold_filter_tau"]))
    filtered_rate = rate_before + alpha * (yaw_rate - rate_before)
    was_active = before["active"] & ~entering & ~reversing & ~reset
    enter = ((heading_error.abs() > deadband)
             | (filtered_rate.abs() > TUNING["heading_hold_rate_enter"]))
    remain = ((heading_error.abs() > TUNING["heading_hold_exit_deadband"])
              | (filtered_rate.abs() > TUNING["heading_hold_rate_exit"]))
    active = correction_active & torch.where(was_active, remain, enter)
    desired = (kp * heading_error - kd * filtered_rate).clamp(-max_yaw_rate, max_yaw_rate)
    desired = torch.where(active, desired, torch.zeros_like(desired))
    previous = torch.where(reset | (before["mode"] == 0), torch.zeros_like(yaw_rate), before["correction"])
    slew = torch.where(stair_correction, TUNING["heading_hold_slew_stairs"], TUNING["heading_hold_slew_flat"])
    yaw_correction = previous + (desired - previous).clamp(-slew * env.step_dt, slew * env.step_dt)
    # A stairs->flat transition decays smoothly from the stair limit rather
    # than making a discontinuous second clip to the smaller flat limit.
    yaw_correction = torch.where(correction_active, yaw_correction, torch.zeros_like(yaw_correction))
    # Hold the same world route on either flight. A small heading error does not
    # detect sideways wheel slip, so feedback also uses cross-track position
    # and route-normal velocity. It only changes the existing vy channel.
    route_normal = torch.stack((-torch.sin(state["straight_target_yaw"]),
                                 torch.cos(state["straight_target_yaw"])), dim=-1)
    path_error = ((robot.data.root_pos_w[:, :2] - state["straight_origin_xy"])
                  * route_normal).sum(-1)
    path_speed = (robot.data.root_lin_vel_w[:, :2] * route_normal).sum(-1)
    def path_parameter(suffix):
        return torch.where(automatic_descent, TUNING["descent_path_" + suffix],
                           TUNING["ascent_path_" + suffix])
    position_excess = path_error.sign() * (path_error.abs() - path_parameter("lateral_deadband")).clamp_min(0.0)
    velocity_excess = path_speed.sign() * (path_speed.abs() - path_parameter("speed_deadband")).clamp_min(0.0)
    lateral_limit = path_parameter("max_lateral_speed")
    route_correction = (-path_parameter("kp") * position_excess
                        - path_parameter("kd") * velocity_excess).clamp(-lateral_limit, lateral_limit)
    body_lateral = (route_correction * torch.cos(actual_yaw - state["straight_target_yaw"])).clamp(-lateral_limit, lateral_limit)
    lateral_before = torch.where(
        reset | reversing | ~before["path_active"], torch.zeros_like(yaw_rate), before["lateral_correction"])
    lateral_slew = path_parameter("lateral_slew") * env.step_dt
    lateral_correction = lateral_before + (body_lateral - lateral_before).clamp(-lateral_slew, lateral_slew)
    lateral_correction = torch.where(path_active, lateral_correction, torch.zeros_like(lateral_correction))
    stop_error = robot.data.root_pos_w[:, :2] - state["stop_origin_xy"]
    position_excess_xy = stop_error.sign() * (stop_error.abs() - TUNING["stair_stop_position_deadband"]).clamp_min(0.0)
    velocity_xy = robot.data.root_lin_vel_w[:, :2]
    velocity_excess_xy = velocity_xy.sign() * (velocity_xy.abs() - TUNING["stair_stop_velocity_deadband"]).clamp_min(0.0)
    desired_world = (-TUNING["stair_stop_position_kp"] * position_excess_xy
                     - TUNING["stair_stop_velocity_kd"] * velocity_excess_xy)
    desired_world *= (TUNING["stair_stop_max_speed"]
                      / torch.linalg.vector_norm(desired_world, dim=-1).clamp_min(TUNING["stair_stop_max_speed"]))[:, None]
    cos_yaw, sin_yaw = torch.cos(actual_yaw), torch.sin(actual_yaw)
    desired_body = torch.stack((desired_world[:, 0] * cos_yaw + desired_world[:, 1] * sin_yaw,
                                -desired_world[:, 0] * sin_yaw + desired_world[:, 1] * cos_yaw), -1)
    stop_before = torch.where((reset | entering)[:, None], 0.0, before["stop_correction"])
    stop_slew = TUNING["stair_stop_slew"] * env.step_dt
    stop_delta = desired_body - stop_before
    stop_delta *= (stop_slew / torch.linalg.vector_norm(stop_delta, dim=-1).clamp_min(stop_slew))[:, None]
    stop_correction = stop_before + stop_delta
    stop_correction = torch.where(stop_feedback[:, None], stop_correction, 0.0)
    filt.update(mode=mode.clone(), filtered_rate=filtered_rate, correction=yaw_correction, active=active,
                world_heading_rate_active=world_heading_rate_active.clone(), lateral_correction=lateral_correction,
                path_error=path_error, path_speed=path_speed,
                path_active=path_active.clone(), automatic_descent=automatic_descent,
                stop_correction=stop_correction, stop_feedback=stop_feedback,
                translation_active=path_active | stop_feedback, stop_error=stop_error,
                straight_direction=torch.sign(raw_command[:, 0]))
    command[:, 2] = torch.where(correction_active, yaw_correction, raw_command[:, 2])
    command[:, 1] = torch.where(path_active, lateral_correction, raw_command[:, 1])
    command[:, :2] = torch.where(stop_feedback[:, None], stop_correction, command[:, :2])
    return command


def heading_aware_track_ang_vel_z_exp(
    env, command_name: str = "base_velocity", std: float = np.sqrt(0.5),
) -> torch.Tensor:
    """Track explicit yaw commands plus the heading-hold correction seen by the actor."""
    command = heading_hold_velocity_commands(env, command_name)
    robot = env.scene["robot"]
    measured_yaw_rate, _ = _automatic_ascent_heading_rate(
        env, env.command_manager.get_command(command_name), robot,
        _stair_scan_context(env, SceneEntityCfg("height_scanner"), command_name),
    )
    error = measured_yaw_rate - command[:, 2]
    return torch.exp(-torch.square(error / std))


def zero_stop_metadata(env, command_name="base_velocity"):
    """Exact raw-stop flag stored for PPO; excluded from the 244D actor input."""
    return (env.command_manager.get_command(command_name).abs().amax(-1) <= 1.e-4).float()[:, None]


def raw_velocity_command_observation(env, command_name="base_velocity"):
    """Unmodified command, for strictly gating a residual motion policy."""
    return env.command_manager.get_command(command_name).clone()


def zero_stop_control_observation(env, command_name="base_velocity"):
    """Shared 24D stop reference errors; used by actor and privileged critic."""
    cache = getattr(env, '_m20_stop_control_observation', None)
    step = int(env.common_step_counter)
    if cache is not None and cache[0] == step:
        return cache[1]
    robot = env.scene['robot']
    contact = env.scene.sensors['contact_forces']
    ids = getattr(env, '_m20_stop_control_ids', None)
    if ids is None:
        legs = ('fl', 'fr', 'hl', 'hr')
        joints = [f'{leg}_{joint}_joint' for leg in legs for joint in ('hipx', 'hipy', 'knee')]+[f'{leg}_wheel_joint' for leg in legs]
        names = [f'{leg}_wheel' for leg in legs]
        ids = (robot.find_joints(joints, preserve_order=True)[0], robot.find_bodies(names, preserve_order=True)[0],
               contact.find_bodies(names, preserve_order=True)[0])
        env._m20_stop_control_ids = ids
    data = robot.data
    q = data.root_quat_w
    yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2].square()+q[:, 3].square()))
    support = stop_branch.stable_support(data.body_pos_w[:, ids[1]], contact.data.net_forces_w[:, ids[2]],
                                         data.root_pos_w[:, :2], data.root_lin_vel_w[:, :2])
    reference = getattr(env, '_m20_stop_control_reference', None)
    if reference is None:
        reference = stop_branch.StopReference(env.step_dt)
        env._m20_stop_control_reference = reference
    value = reference.update(env.command_manager.get_command(command_name), data.joint_pos[:, ids[0]],
                             data.root_pos_w[:, :2], yaw, data.root_lin_vel_b[:, :2],
                             reset=env.episode_length_buf <= 1, support=support,
                             joint_velocity=data.joint_vel[:, ids[0]],
                             planar_support=(data.body_pos_w[:, ids[1], 2].amax(-1)-data.body_pos_w[:, ids[1], 2].amin(-1)) <= .03,
                             angular_velocity=data.root_ang_vel_b)
    env._m20_stop_control_observation = (step, value)
    return value


def zero_stop_pose_hold_cost(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name="base_velocity"):
    c = _zero_stop_context(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name)
    reference = getattr(env, '_m20_stop_control_reference', None)
    if reference is None or reference.state is None:
        return torch.zeros(env.num_envs, device=env.device)
    state = reference.state
    current = env.scene[asset_cfg.name].data.joint_pos[:, leg_joint_cfg.joint_ids]
    deviation = (current-state['joints'][:, :12])/.08
    return _zero_stop_robust(deviation).mean(-1)*state['legs_held']*(state['leg_age'] >= .4)*c['ramp']


def zero_stop_settled_reward(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name="base_velocity"):
    c = _zero_stop_context(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name)
    cost = (c['velocity']/.03).square()+.25*(c['wheels']/.05).square().mean(-1)
    cost += (c['drift']/.04).square()+.25*(c['leg_velocity']/.2).square().mean(-1)
    return torch.exp(-cost)*c['ramp']


def _zero_stop_support_context(env):
    """Physical four-wheel planar support, independent of visible stair edges.

    A flat scan alone cannot prove support; all four wheels must bear mostly
    vertical loads for >=40ms, at one height, with a nearly upright trunk.
    Mixed-tread, riser-only and airborne states never receive a flat pose target.
    """
    step = int(env.common_step_counter)
    cache = getattr(env, "_m20_stop_support", None)
    if cache is not None and cache["step"] == step:
        return cache
    robot = env.scene["robot"]
    contact = env.scene.sensors["contact_forces"]
    ids = getattr(env, "_m20_stop_support_ids", None)
    if ids is None:
        names = [f"{leg}_wheel" for leg in ("fl", "fr", "hl", "hr")]
        ids = (robot.find_bodies(names, preserve_order=True)[0],
               contact.find_bodies(names, preserve_order=True)[0])
        env._m20_stop_support_ids = ids
    wheels = robot.data.body_pos_w[:, ids[0]]
    force = contact.data.net_forces_w[:, ids[1]]
    loaded = (force[..., 2] > 5.) & (force[..., 2] > .7*force.norm(dim=-1))
    loaded &= contact.data.current_contact_time[:, ids[1]] >= .04
    upright = robot.data.projected_gravity_b[:, :2].square().sum(-1) < .12**2
    planar = wheels[..., 2].amax(-1)-wheels[..., 2].amin(-1) <= .03
    flat = loaded.all(-1) & planar & upright
    hits = env.scene["height_scanner"].data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits) & (hits.abs() < 1.e6)
    high = torch.where(valid, hits, -torch.inf).amax(-1)
    low = torch.where(valid, hits, torch.inf).amin(-1)
    visible = valid.all(-1) & ((high-low) > .06)
    ground_z = wheels[..., 2].mean(-1)-TUNING["wheel_radius"]
    flat_scan = (robot.data.root_pos_w[:, 2]-ground_z-.5).clamp(-1., 1.)
    cache = dict(step=step, flat=flat, preview=flat & visible, visible=visible,
                 flat_scan=flat_scan)
    env._m20_stop_support = cache
    return cache


def zero_stop_support_metadata(env, command_name="base_velocity"):
    """PPO-only [flat support, preview, post-brake eligible, flat scan value]."""
    c = _zero_stop_support_context(env)
    zero = env.command_manager.get_command(command_name).abs().amax(-1) <= 1.e-4
    state = getattr(env, "_m20_zero_stop", None)
    eligible = torch.zeros_like(zero)
    if state is not None:
        age = (int(env.common_step_counter)-state["start"])*env.step_dt
        eligible = zero & state["was_zero"] & (age >= .8) & (env.episode_length_buf > 1)
    return torch.stack((c["flat"].float(), c["preview"].float(), eligible.float(), c["flat_scan"]), -1)


def _zero_stop_groups():
    base = ("flat", "stair", "rough")
    groups = base + ("stair_ascent", "stair_descent", "stair_unknown") if TUNING.get("zero_stop_v2", False) else base
    return groups + ("support_flat", "support_preview", "support_stair") if TUNING.get("zero_stop_v4", False) else groups


def _zero_stop_entry_terrain(env, flat, stairs):
    scan = _stair_scan_context(env, SceneEntityCfg("height_scanner"))
    if TUNING.get("zero_stop_v3", False):
        return stop_coverage.classify_terrain(env, scan, flat, stairs)
    up, down = scan["up_gate"] > 0, scan["down_gate"] > 0
    ascent = getattr(env, "_m20_ascent_state", None)
    descent = getattr(env, "_m20_stair_step_state", None)
    if ascent is not None:
        up = up | (ascent["active"].bool() & ~down)
    if descent is not None:
        down = down | (descent["active"].bool() & ~descent["up"].bool() & ~up)
    kind = torch.full((env.num_envs,), 3, device=env.device, dtype=torch.long)
    kind[flat & ~stairs] = 0
    kind[stairs] = 4  # Ambiguous scans remain observable rather than mislabeled.
    kind[stairs & up & ~down] = 1
    kind[stairs & down & ~up] = 2
    return kind


def _zero_stop_context(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg,
                       command_name="base_velocity"):
    """One stop ledger per tick; exact raw zero, brake, then gradual shaping.

    The post-braking world position is a training measurement, never an extra
    policy input or an XY recovery command. The actor learns to resist drift.
    """
    step = int(env.common_step_counter)
    state = getattr(env, "_m20_zero_stop", None)
    if state is not None and state["step"] == step:
        zero_now = env.command_manager.get_command(command_name).abs().amax(-1) <= 1.e-4
        return dict(state["context"], ramp=state["context"]["ramp"] * zero_now)
    robot = env.scene[asset_cfg.name].data
    command = env.command_manager.get_command(command_name)
    zero = command.abs().amax(-1) <= 1.e-4
    xy = robot.root_pos_w[:, :2]
    legs = env.action_manager.get_term("joint_pos").processed_actions
    wheels = -env.action_manager.get_term("joint_vel").processed_actions * TUNING["wheel_radius"]
    targets = torch.cat((legs, wheels), -1)
    if state is None:
        z = lambda: torch.zeros(env.num_envs, device=env.device)
        state = dict(step=-1, was_zero=torch.zeros_like(zero),
                     start=torch.full_like(env.episode_length_buf, step),
                     origin=xy.clone(), latched=torch.zeros_like(zero),
                     previous=targets.clone(), older=targets.clone(),
                     history=torch.zeros_like(env.episode_length_buf), quiet_s=z(),
                     entry_terrain=torch.full_like(env.episode_length_buf, 3),
                     seen=torch.zeros_like(zero), quiet_seen=torch.zeros_like(zero),
                     stats={group: {key: z() for key in (
                         "samples", "holds", "quiet_holds", "quiet_samples", "success",
                         "speed", "wheel_speed", "yaw_rate", "drift", "leg_speed_sq",
                         "leg_delta_sq", "wheel_delta_sq", "leg_second_sq", "wheel_second_sq", "wheel_target_bias_sq")}
                            for group in _zero_stop_groups()})
        env._m20_zero_stop = state
        if TUNING.get('zero_stop_fast_stats', False):
            groups = tuple(state['stats'])
            fields = tuple(state['stats'][groups[0]])
            buffer = targets.new_zeros((env.num_envs, len(groups), len(fields)))
            state['stats_buffer'], state['stats_fields'] = buffer, fields
            state['stats'] = {g: {f: buffer[:, i, j] for j, f in enumerate(fields)} for i, g in enumerate(groups)}
    if TUNING.get("zero_stop_v4", False) and "support_seen" not in state:
        state["support_seen"] = {group: torch.zeros_like(zero) for group in ("support_flat", "support_preview", "support_stair")}
        state["support_quiet_seen"] = {group: torch.zeros_like(zero) for group in state["support_seen"]}
    reset = env.episode_length_buf <= 1
    entering = zero & (~state["was_zero"] | reset)
    if TUNING.get("zero_stop_v4", False):
        for fields in (state["support_seen"], state["support_quiet_seen"]):
            for seen in fields.values():
                seen[entering | ~zero | reset] = False
    state["start"][entering] = step
    state["latched"][entering | ~zero | reset] = False
    state["seen"][entering | ~zero | reset] = False
    state["quiet_seen"][entering | ~zero | reset] = False
    state["quiet_s"][entering | ~zero | reset] = 0.0
    age = (step - state["start"]) * env.step_dt
    ramp = ((age - 0.4) / 0.4).clamp(0., 1.) * zero
    latch = zero & (age >= 0.4 - 1.e-7) & ~state["latched"]
    state["origin"][latch] = xy[latch]
    state["latched"] |= latch
    valid_rate = zero & state["was_zero"] & ~reset & (state["history"] >= 1)
    valid_second = valid_rate & (state["history"] >= 2)
    delta = (targets - state["previous"]) * valid_rate[:, None]
    second = (targets - 2 * state["previous"] + state["older"]) * valid_second[:, None]
    flat = _flat_surface_mask(env, SceneEntityCfg("height_scanner"))
    stairs = _stair_activity_mask(env)
    if TUNING.get("zero_stop_v2", False):
        kind = _zero_stop_entry_terrain(env, flat, stairs)
        state["entry_terrain"][entering] = kind[entering]
        # Keep the stop's original terrain even if the policy rolls off stairs.
        flat = state["entry_terrain"] == 0
        stairs = (state["entry_terrain"] == 1) | (state["entry_terrain"] == 2) | (state["entry_terrain"] == 4)
    effective = heading_hold_velocity_commands(env, command_name)
    desired_wheels = effective[:, :1] + effective[:, 2:3] * command.new_tensor([[-.28, .28, -.28, .28]])
    speed = torch.linalg.vector_norm(robot.root_lin_vel_w[:, :2], dim=-1)
    signed_wheels = -robot.joint_vel[:, wheel_joint_cfg.joint_ids] * TUNING["wheel_radius"]
    wheel_speed = signed_wheels.abs().amax(-1)
    yaw_rate = robot.root_ang_vel_b[:, 2].abs()
    leg_velocity = robot.joint_vel[:, leg_joint_cfg.joint_ids]
    leg_speed_sq = leg_velocity.square().mean(-1)
    drift = torch.linalg.vector_norm(xy - state["origin"], dim=-1)
    n = legs.shape[-1]
    hold = env._m20_zero_hold_heading_state
    q = robot.root_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2].square() + q[:, 3].square()))
    heading_error = torch.atan2(torch.sin(yaw - hold["target_yaw"]), torch.cos(yaw - hold["target_yaw"]))
    success = ((speed < .02) & (wheel_speed < .03) & (yaw_rate < .03)
               & (heading_error.abs() < np.deg2rad(2)) & (drift < .03))
    quiet = (success & (leg_speed_sq.sqrt() < .1) & (leg_velocity.abs().amax(-1) < .2)
             & valid_second & (delta[:, :n].abs().amax(-1) < .002)
             & (delta[:, n:].abs().amax(-1) < .010)
             & (second[:, :n].abs().amax(-1) < .003)
             & (second[:, n:].abs().amax(-1) < .015))
    measured = zero & (age >= .8 - 1.e-7) & ~reset
    state["quiet_s"] = torch.where(measured & quiet, state["quiet_s"] + env.step_dt, 0.)
    sustained = state["quiet_s"] >= 1. - 1.e-6
    terrains = {"flat": flat & ~stairs, "stair": stairs, "rough": ~flat & ~stairs}
    if TUNING.get("zero_stop_v2", False):
        terrains.update(stair_ascent=state["entry_terrain"] == 1,
                        stair_descent=state["entry_terrain"] == 2,
                        stair_unknown=state["entry_terrain"] == 4)
    support = None
    if TUNING.get("zero_stop_v4", False):
        support = _zero_stop_support_context(env)
        # Live physical groups complement, rather than overwrite, entry labels.
        terrains.update(support_flat=support["flat"] & ~support["visible"],
                        support_preview=support["preview"],
                        support_stair=~support["flat"] & _stair_activity_mask(env))
    if TUNING.get('zero_stop_fast_stats', False):
        groups, fields = tuple(state['stats']), state['stats_fields']
        masks = torch.stack([measured & terrains[g] for g in groups], -1)
        values = dict(samples=torch.ones_like(speed), holds=~state['seen'],
                      quiet_holds=sustained & ~state['quiet_seen'], quiet_samples=sustained,
                      success=success, speed=speed, wheel_speed=wheel_speed, yaw_rate=yaw_rate,
                      drift=drift, leg_speed_sq=leg_speed_sq,
                      leg_delta_sq=delta[:, :n].square().mean(-1), wheel_delta_sq=delta[:, n:].square().mean(-1),
                      leg_second_sq=second[:, :n].square().mean(-1), wheel_second_sq=second[:, n:].square().mean(-1),
                      wheel_target_bias_sq=(wheels-desired_wheels).square().mean(-1))
        increments = torch.stack([values[f].float() for f in fields], -1)[:, None, :]*masks[:, :, None]
        for i, group in enumerate(groups):
            if group.startswith('support_'):
                increments[:, i, fields.index('holds')] = masks[:, i] & ~state['support_seen'][group]
                increments[:, i, fields.index('quiet_holds')] = masks[:, i] & sustained & ~state['support_quiet_seen'][group]
                state['support_seen'][group] |= masks[:, i]
                state['support_quiet_seen'][group] |= masks[:, i] & sustained
        state['stats_buffer'].masked_fill_(reset[:, None, None], 0.)
        state['stats_buffer'].add_(increments)
    else:
        for group, terrain in terrains.items():
            stats = state["stats"][group]
            mask = measured & terrain
            for value in stats.values():
                value[reset] = 0.
            values = dict(samples=torch.ones_like(speed), holds=(~state["seen"]),
                          quiet_holds=sustained & ~state["quiet_seen"], quiet_samples=sustained,
                          success=success, speed=speed, wheel_speed=wheel_speed, yaw_rate=yaw_rate,
                          drift=drift, leg_speed_sq=leg_speed_sq,
                          leg_delta_sq=delta[:, :n].square().mean(-1), wheel_delta_sq=delta[:, n:].square().mean(-1),
                          leg_second_sq=second[:, :n].square().mean(-1), wheel_second_sq=second[:, n:].square().mean(-1),
                          wheel_target_bias_sq=(wheels - desired_wheels).square().mean(-1))
            if group.startswith("support_"):
                # A stop can enter another physical group after its first sample.
                # Count one observed hold per group, not only in the initial one.
                values["holds"] = ~state["support_seen"][group]
                values["quiet_holds"] = sustained & ~state["support_quiet_seen"][group]
                state["support_seen"][group] |= mask
                state["support_quiet_seen"][group] |= mask & sustained
            for key, value in values.items():
                stats[key] += value * mask
    state["seen"] |= measured
    state["quiet_seen"] |= measured & sustained
    state["history"] = torch.where(zero & ~entering, state["history"] + 1, zero.long())
    if TUNING.get("zero_stop_v3", False):
        coverage = getattr(env, "_m20_stop_coverage", None)
        if coverage is None:
            coverage = stop_coverage.StopCoverage(env)
            env._m20_stop_coverage = coverage
        coverage.update(env, command, kind, state["entry_terrain"], measured, sustained)
    state.update(step=step, was_zero=zero.clone(), older=state["previous"].clone(), previous=targets.clone())
    # Losing one contact on otherwise plain ground must not remove the old
    # flat quiet constraint. Physical evidence adds the visible-edge case.
    quiet_flat = (support["flat"] | (_flat_surface_mask(env, SceneEntityCfg("height_scanner"))
                                   & ~support["visible"])) if support is not None else flat & ~stairs
    context = dict(ramp=ramp, flat=quiet_flat, velocity=speed, wheels=signed_wheels - desired_wheels,
                   yaw=robot.root_ang_vel_b[:, 2] - effective[:, 2], heading=heading_error,
                   leg_velocity=leg_velocity, delta=delta, second=second, leg_count=n, drift=drift,
                   wheel_target_residual=wheels - desired_wheels)
    state["context"] = context
    return context


def _zero_stop_robust(value):
    return 2 * (torch.sqrt(1 + value.square()) - 1)


def zero_stop_velocity_cost(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name="base_velocity"):
    c = _zero_stop_context(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name)
    tolerance = .002 if TUNING.get("zero_stop_v4", False) else .005
    cost = _zero_stop_robust((c["velocity"] - tolerance).clamp_min(0.) / .05)
    cost += .5 * _zero_stop_robust(c["wheels"] / .07).mean(-1)
    cost += _zero_stop_robust(c["yaw"] / .06)
    cost += _zero_stop_robust((c["heading"].abs() - np.deg2rad(.5)).clamp_min(0.) / np.deg2rad(3))
    return cost * c["ramp"]


def zero_stop_position_cost(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name="base_velocity"):
    c = _zero_stop_context(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name)
    tolerance = .005 if TUNING.get("zero_stop_v4", False) else .02
    return _zero_stop_robust((c["drift"] - tolerance).clamp_min(0.) / .06) * c["ramp"]


def zero_stop_wheel_target_cost(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name="base_velocity"):
    """Discourage constant drive and opposing-wheel targets after braking.

    A soft 0.01m/s allowance and the existing yaw target retain balancing and
    corrective authority. This is not an actuator clamp or a frozen pose.
    """
    c = _zero_stop_context(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name)
    excess = (c["wheel_target_residual"].abs() - .01).clamp_min(0.) / .05
    return _zero_stop_robust(excess).mean(-1) * c["ramp"]


def zero_stop_chatter_cost(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name="base_velocity"):
    c = _zero_stop_context(env, asset_cfg, leg_joint_cfg, wheel_joint_cfg, command_name)
    n = c["leg_count"]
    # Stair balance corrections retain freedom; no default joint pose is latched.
    gain = torch.where(c["flat"], 1., .25)
    if TUNING.get('zero_stop_v5', False):
        gain = torch.ones_like(gain)
    cost = gain * _zero_stop_robust(c["leg_velocity"] / .4).mean(-1)
    cost += .5 * _zero_stop_robust(c["delta"][:, :n] / .008).mean(-1)
    cost += .5 * _zero_stop_robust(c["delta"][:, n:] / .025).mean(-1)
    cost += _zero_stop_robust(c["second"][:, :n] / .006).mean(-1)
    cost += _zero_stop_robust(c["second"][:, n:] / .020).mean(-1)
    return cost * c["ramp"]


def zero_stop_flat_stand_still(env, asset_cfg, command_name="base_velocity", command_threshold=.06):
    """The old flat reference pose must not pull a stair stop off its supports."""
    robot = env.scene[asset_cfg.name].data
    deviation = (robot.joint_pos[:, asset_cfg.joint_ids] - robot.default_joint_pos[:, asset_cfg.joint_ids]).abs().sum(-1)
    command = env.command_manager.get_command(command_name)
    flat = _flat_surface_mask(env, SceneEntityCfg("height_scanner")) & ~_stair_activity_mask(env)
    if TUNING.get("zero_stop_v4", False):
        raw_zero = command.abs().amax(-1) <= 1.e-4
        flat = torch.where(raw_zero, flat | _zero_stop_support_context(env)["flat"], flat)
    if TUNING.get('zero_stop_v5', False):
        flat &= command.abs().amax(-1) > 1.e-4
    return deviation * (torch.linalg.vector_norm(command, dim=-1) < command_threshold) * flat


def zero_stop_metrics(env, env_ids):
    state = getattr(env, "_m20_zero_stop", None)
    result = {}
    for group in _zero_stop_groups():
        raw = state["stats"][group] if state is not None else {}
        s = {k: v[env_ids].sum() for k, v in raw.items()}
        count = s.get("samples", torch.zeros((), device=env.device))
        result[f"{group}_samples"] = count
        holds = s.get("holds", torch.zeros_like(count))
        result[f"{group}_holds"] = holds
        for key, name in (("speed", "base_speed_m_s"), ("wheel_speed", "max_wheel_speed_m_s"),
                          ("yaw_rate", "yaw_rate_rad_s"), ("drift", "drift_mean_m"),
                          ("success", "success_fraction"), ("quiet_samples", "quiet_1s_fraction"),
                          ("quiet_holds", "holds_reached_quiet_1s_fraction"),
                          ("leg_speed_sq", "leg_speed_rms_rad_s"),
                          ("leg_delta_sq", "leg_target_delta_rms_rad"), ("wheel_delta_sq", "wheel_target_delta_rms_m_s"),
                          ("leg_second_sq", "leg_target_second_difference_rms_rad"), ("wheel_second_sq", "wheel_target_second_difference_rms_m_s"),
                          ("wheel_target_bias_sq", "wheel_target_bias_rms_m_s")):
            denom = holds if key == "quiet_holds" else count
            total = s.get(key, torch.zeros_like(count))
            rms = key.endswith("_sq")
            value = total / denom.clamp_min(1.)
            metric = f"{group}_{name}"
            result[metric] = value.sqrt() if rms else value
            prefix = "__rms" if rms else "__weighted"
            result[f"{prefix}_total__/{metric}"] = total
            result[f"{prefix}_count__/{metric}"] = denom
    coverage = getattr(env, "_m20_stop_coverage", None)
    if TUNING.get("zero_stop_v3", False) and coverage is not None:
        result.update(coverage.metrics(env_ids))
    return export_teacher_metrics(result, result.keys())


def _zero_hold_targets_quiet(env) -> torch.Tensor:
    """Read target history without advancing the action-dynamics consumer."""
    state = getattr(env, "_m20_stop_action_dynamics", None)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    before = state["before"] if state["step"] == int(env.common_step_counter) else state
    legs = env.action_manager.get_term("joint_pos").processed_actions
    wheels = -env.action_manager.get_term("joint_vel").processed_actions * TUNING["wheel_radius"]
    targets = torch.cat((legs, wheels), dim=-1)
    delta = (targets - before["previous"]).abs()
    second = (targets - 2.0 * before["previous"] + before["older"]).abs()
    n = legs.shape[-1]
    return ((before["history"] >= 2) & (env.episode_length_buf > 1)
            & (delta[:, :n].amax(-1) < TUNING["quiet_leg_target_delta"])
            & (delta[:, n:].amax(-1) < TUNING["quiet_wheel_target_delta"])
            & (second[:, :n].amax(-1) < TUNING["quiet_leg_target_second_difference"])
            & (second[:, n:].amax(-1) < TUNING["quiet_wheel_target_second_difference"]))


def zero_command_hold_cost(
    env, asset_cfg: SceneEntityCfg, wheel_joint_cfg: SceneEntityCfg,
    command_name: str = "base_velocity", leg_joint_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize drift, wheel activity, and heading error while a zero command is held."""
    command = env.command_manager.get_command(command_name)
    robot = env.scene[asset_cfg.name]
    state, actual_yaw, hold_active = _zero_hold_heading_context(env, command, robot)

    body_speed = torch.linalg.vector_norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    yaw_speed = robot.data.root_ang_vel_b[:, 2].abs()
    wheel_speed = robot.data.joint_vel[:, wheel_joint_cfg.joint_ids].abs() * TUNING["wheel_radius"]
    wheel_speed_mean = wheel_speed.mean(dim=1)
    wheel_speed_max = wheel_speed.max(dim=1).values
    flat = _flat_surface_mask(env, SceneEntityCfg("height_scanner"))
    linear_scale = TUNING["zero_hold_linear_velocity_scale"] + flat.to(body_speed.dtype) * (
        TUNING["flat_zero_hold_linear_velocity_scale"] - TUNING["zero_hold_linear_velocity_scale"]
    )
    wheel_scale = TUNING["zero_hold_wheel_linear_velocity_scale"] + flat.to(body_speed.dtype) * (
        TUNING["flat_zero_hold_wheel_velocity_scale"] - TUNING["zero_hold_wheel_linear_velocity_scale"]
    )
    def robust(value):
        return 2.0 * (torch.sqrt(1.0 + value.square()) - 1.0)

    # A corrective turn must remain possible. Penalize yaw-rate tracking
    # error rather than all angular motion, using the actor's effective goal.
    effective = heading_hold_velocity_commands(env, command_name)
    effective_yaw = effective[:, 2]
    yaw_error = robot.data.root_ang_vel_b[:, 2] - effective_yaw
    cost = robust(body_speed / linear_scale)
    cost += robust(yaw_error / TUNING["zero_hold_yaw_velocity_scale"])
    # Allow the differential wheel speed needed for that same correction.
    # URDF and MJCF wheel axes are (0, -1, 0): NEGATIVE joint velocity
    # rolls toward body +X. Convert before comparing to the yaw correction.
    desired_wheel_speed = (effective[:, :1] + effective_yaw[:, None]
                           * command.new_tensor([-0.28, 0.28, -0.28, 0.28]))
    signed_wheel_speed = -robot.data.joint_vel[:, wheel_joint_cfg.joint_ids] * TUNING["wheel_radius"]
    cost += 0.5 * robust((signed_wheel_speed - desired_wheel_speed) / wheel_scale[:, None]).mean(dim=1)
    heading_error = torch.atan2(
        torch.sin(actual_yaw - state["target_yaw"]),
        torch.cos(actual_yaw - state["target_yaw"]),
    ).abs()
    heading_excess = (
        (heading_error - TUNING["zero_hold_heading_tolerance"]).clamp_min(0.0)
        / TUNING["zero_hold_heading_scale"]
    )
    cost += TUNING["zero_hold_heading_weight"] * robust(heading_excess)
    if state.get("telemetry_step") == int(env.common_step_counter):
        return cost * hold_active.to(cost.dtype)
    state["telemetry_step"] = int(env.common_step_counter)
    if _detailed_teacher_diagnostics(env):
        # The per-terrain settled counters below are the core measurements.
        # Skip this redundant whole-hold summary before the braking grace.
        state["error_sum"] += heading_error * hold_active
        signed_heading_error = torch.atan2(
            torch.sin(actual_yaw - state["target_yaw"]),
            torch.cos(actual_yaw - state["target_yaw"]),
        )
        state["error_signed_sum"] += signed_heading_error * hold_active
        state["samples"] += hold_active
        state["max_error"] = torch.maximum(state["max_error"], heading_error * hold_active)
        state["base_speed_sum"] += body_speed * hold_active
        state["base_speed_max"] = torch.maximum(state["base_speed_max"], body_speed * hold_active)
        state["yaw_rate_sum"] += yaw_speed * hold_active
        state["yaw_rate_signed_sum"] += robot.data.root_ang_vel_b[:, 2] * hold_active
        state["yaw_rate_max"] = torch.maximum(state["yaw_rate_max"], yaw_speed * hold_active)
        state["wheel_speed_sum"] += wheel_speed_mean * hold_active
        state["wheel_speed_max"] = torch.maximum(state["wheel_speed_max"], wheel_speed_max * hold_active)
    age = (int(env.common_step_counter) - state["hold_start_step"]) * env.step_dt
    settled = hold_active & (age >= TUNING["zero_hold_brake_grace_s"])
    latch = settled & ~state["position_latched"]
    xy = robot.data.root_pos_w[:, :2]
    state["hold_xy"][latch] = xy[latch]
    state["previous_hold_xy"][latch] = xy[latch]
    state["position_latched"] |= latch
    drift = torch.linalg.vector_norm(xy - state["hold_xy"], dim=1)
    path = torch.linalg.vector_norm(xy - state["previous_hold_xy"], dim=1)
    success = ((body_speed < TUNING["hold_success_base_speed"])
               & (wheel_speed_max < TUNING["hold_success_wheel_speed"])
               & (yaw_speed < TUNING["hold_success_yaw_rate"])
               & (heading_error < TUNING["hold_success_heading"]))
    quiet_motion = success
    if leg_joint_cfg is not None:
        leg_velocity = robot.data.joint_vel[:, leg_joint_cfg.joint_ids]
        quiet_motion = (quiet_motion
                        & (leg_velocity.square().mean(-1).sqrt() < TUNING["quiet_leg_speed_rms"])
                        & (leg_velocity.abs().amax(-1) < TUNING["quiet_leg_speed_max"])
                        & _zero_hold_targets_quiet(env))
    state["quiet_run_s"] = torch.where(settled & quiet_motion,
                                       state["quiet_run_s"] + env.step_dt,
                                       torch.zeros_like(body_speed))
    quiet = state["quiet_run_s"] >= TUNING["zero_hold_quiet_duration_s"] - 1.0e-6
    on_stairs = _stair_activity_mask(env)
    for group, mask in (("flat", settled & flat & ~on_stairs), ("stair", settled & on_stairs)):
        stats = state["settled_stats"][group]
        stats["holds"] += mask & ~state["settled_seen"][group]
        state["settled_seen"][group] |= mask
        stats["quiet_holds"] += mask & quiet & ~state["quiet_seen"][group]
        state["quiet_seen"][group] |= mask & quiet
        for key, value in (("samples", torch.ones_like(body_speed)), ("success", success),
                           ("base_speed", body_speed), ("wheel_speed", wheel_speed_mean),
                           ("heading", heading_error), ("yaw_rate", yaw_speed),
                           ("drift", drift), ("path", path), ("quiet_samples", quiet)):
            stats[key] += value * mask
    state["previous_hold_xy"] = xy.clone()
    return cost * hold_active.to(cost.dtype)


def zero_command_flat_leg_motion_cost(env, asset_cfg: SceneEntityCfg,
                                      command_name: str = "base_velocity") -> torch.Tensor:
    """Discourage continued stepping after a flat-ground stop has settled."""
    command = env.command_manager.get_command(command_name)
    robot = env.scene[asset_cfg.name]
    state, _, hold_active = _zero_hold_heading_context(env, command, robot)
    age = (int(env.common_step_counter) - state["hold_start_step"]) * env.step_dt
    settled = hold_active & (age >= TUNING["zero_hold_brake_grace_s"])
    flat = _flat_surface_mask(env, SceneEntityCfg("height_scanner"))
    leg_speed = robot.data.joint_vel[:, asset_cfg.joint_ids]
    scaled = leg_speed / TUNING["zero_hold_leg_velocity_scale"]
    cost = (2.0 * (torch.sqrt(1.0 + scaled.square()) - 1.0)).mean(dim=1)
    return cost * (settled & flat).to(cost.dtype)


def zero_command_flat_support_cost(env, asset_cfg: SceneEntityCfg,
                                    contact_sensor_cfg: SceneEntityCfg,
                                    command_name: str = "base_velocity") -> torch.Tensor:
    """Favor four loaded wheels after the braking interval on level ground."""
    command = env.command_manager.get_command(command_name)
    robot = env.scene[asset_cfg.name]
    state, _, hold_active = _zero_hold_heading_context(env, command, robot)
    age = (int(env.common_step_counter) - state["hold_start_step"]) * env.step_dt
    settled = hold_active & (age >= TUNING["zero_hold_brake_grace_s"])
    flat = _flat_surface_mask(env, SceneEntityCfg("height_scanner"))
    sensor = env.scene.sensors[contact_sensor_cfg.name]
    loaded = sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, 2] > 10.0
    return (1.0 - loaded.float().mean(dim=1)) * (settled & flat).to(command.dtype)


def _zero_command_hold_metrics(env, env_ids):
    """Report the mean and peak heading error accumulated during zero holds."""
    state = getattr(env, "_m20_zero_hold_heading_state", None)
    if state is None:
        zero = torch.zeros((), device=env.device)
        return {
            "samples": zero,
            "heading_error_mean_deg": zero,
            "heading_error_signed_mean_deg": zero,
            "heading_error_max_deg": zero,
            "base_speed_mean_m_s": zero,
            "base_speed_max_m_s": zero,
            "yaw_rate_mean_rad_s": zero,
            "yaw_rate_signed_mean_rad_s": zero,
            "yaw_rate_max_rad_s": zero,
            "wheel_speed_mean_m_s": zero,
            "wheel_speed_max_m_s": zero,
        }
    result = {}
    if _detailed_teacher_diagnostics(env):
        count = state["samples"][env_ids].sum().clamp_min(1.0)
        result = {
            "samples": state["samples"][env_ids].sum(),
            "heading_error_mean_deg": torch.rad2deg(state["error_sum"][env_ids].sum() / count),
            "heading_error_signed_mean_deg": torch.rad2deg(state["error_signed_sum"][env_ids].sum() / count),
            "heading_error_max_deg": torch.rad2deg(state["max_error"][env_ids].max()),
            "base_speed_mean_m_s": state["base_speed_sum"][env_ids].sum() / count,
            "base_speed_max_m_s": state["base_speed_max"][env_ids].max(),
            "yaw_rate_mean_rad_s": state["yaw_rate_sum"][env_ids].sum() / count,
            "yaw_rate_signed_mean_rad_s": state["yaw_rate_signed_sum"][env_ids].sum() / count,
            "yaw_rate_max_rad_s": state["yaw_rate_max"][env_ids].max(),
            "wheel_speed_mean_m_s": state["wheel_speed_sum"][env_ids].sum() / count,
            "wheel_speed_max_m_s": state["wheel_speed_max"][env_ids].max(),
        }
    for group, stats in state["settled_stats"].items():
        s = {key: value[env_ids].sum() for key, value in stats.items()}
        count = s["samples"].clamp_min(1.0)
        result.update({
            f"{group}_settled_samples": s["samples"],
            f"{group}_settled_success_fraction": s["success"] / count,
            f"{group}_settled_base_speed_m_s": s["base_speed"] / count,
            f"{group}_settled_wheel_speed_m_s": s["wheel_speed"] / count,
            f"{group}_settled_yaw_rate_rad_s": s["yaw_rate"] / count,
            f"{group}_settled_heading_error_deg": torch.rad2deg(s["heading"] / count),
            f"{group}_settled_drift_mean_m": s["drift"] / count,
            f"{group}_settled_holds": s["holds"],
            f"{group}_settled_path_per_hold_m": s["path"] / s["holds"].clamp_min(1.0),
            f"{group}_quiet_1s_sample_fraction": s["quiet_samples"] / count,
            f"{group}_holds_reached_quiet_1s_fraction": s["quiet_holds"] / s["holds"].clamp_min(1.0),
        })
        # RSL-RL averages reset batches of different sizes. Preserve numerator
        # and denominator so the project logger can pool samples exactly.
        numerator = {
            f"{group}_settled_success_fraction": s["success"],
            f"{group}_settled_base_speed_m_s": s["base_speed"],
            f"{group}_settled_wheel_speed_m_s": s["wheel_speed"],
            f"{group}_settled_yaw_rate_rad_s": s["yaw_rate"],
            f"{group}_settled_heading_error_deg": torch.rad2deg(s["heading"]),
            f"{group}_settled_drift_mean_m": s["drift"],
            f"{group}_quiet_1s_sample_fraction": s["quiet_samples"],
        }
        for name, total in numerator.items():
            result[f"__weighted_total__/{name}"] = total
            result[f"__weighted_count__/{name}"] = s["samples"]
        for name, total in ((f"{group}_settled_path_per_hold_m", s["path"]),
                            (f"{group}_holds_reached_quiet_1s_fraction", s["quiet_holds"])):
            result[f"__weighted_total__/{name}"] = total
            result[f"__weighted_count__/{name}"] = s["holds"]
    dynamics = getattr(env, "_m20_stop_action_stats", None)
    if dynamics is not None:
        s = {key: value[env_ids].sum() for key, value in dynamics.items()}
        result["action_dynamics_samples"] = s["samples"]
        for field, name in (
            ("leg_delta_sq", "leg_target_delta_rms_rad"),
            ("wheel_delta_sq", "wheel_target_delta_rms_m_s"),
            ("leg_jerk_sq", "leg_target_second_difference_rms_rad"),
            ("wheel_jerk_sq", "wheel_target_second_difference_rms_m_s"),
            ("leg_speed_sq", "leg_speed_rms_rad_s"),
        ):
            result[name] = torch.sqrt(s[field] / s["samples"].clamp_min(1.0))
            result[f"__rms_total__/{name}"] = s[field]
            result[f"__rms_count__/{name}"] = s["samples"]
    return result


def zero_command_hold_metrics(env, env_ids):
    core = {"action_dynamics_samples", "leg_target_delta_rms_rad", "wheel_target_delta_rms_m_s",
            "leg_target_second_difference_rms_rad", "wheel_target_second_difference_rms_m_s",
            "leg_speed_rms_rad_s"}
    for group in ("flat", "stair"):
        core.update(f"{group}_{name}" for name in (
            "settled_samples", "settled_holds", "settled_success_fraction",
            "settled_base_speed_m_s", "settled_wheel_speed_m_s", "settled_yaw_rate_rad_s",
            "settled_heading_error_deg", "settled_drift_mean_m", "settled_path_per_hold_m",
            "quiet_1s_sample_fraction", "holds_reached_quiet_1s_fraction",
        ))
    return export_teacher_metrics(
        _zero_command_hold_metrics(env, env_ids), core, _detailed_teacher_diagnostics(env)
    )


def zero_command_wheel_action_cost(
    env, command_name: str = "base_velocity", action_term_name: str = "joint_vel",
) -> torch.Tensor:
    """Penalize rolling drift while leaving room for heading correction."""
    command = env.command_manager.get_command(command_name)
    zero_command = (
        (torch.linalg.vector_norm(command[:, :2], dim=1) <= TUNING["translation_threshold"])
        & (command[:, 2].abs() <= TUNING["yaw_threshold"])
    )
    wheel_targets = env.action_manager.get_term(action_term_name).processed_actions
    wheel_speed = -wheel_targets * TUNING["wheel_radius"]
    effective = heading_hold_velocity_commands(env, command_name)
    desired = (effective[:, :1] + effective[:, 2:3]
               * wheel_speed.new_tensor([[-0.28, 0.28, -0.28, 0.28]]))
    scale = TUNING["zero_hold_wheel_action_linear_scale"]
    # Penalize each axle independently: [+v,+v,-v,-v] must not disappear
    # from the cost just because its mean and left/right differential vanish.
    residual = (wheel_speed - desired) / scale
    cost = (2.0 * (torch.sqrt(1.0 + residual.square()) - 1.0)).mean(dim=1)
    return cost * zero_command.to(cost.dtype)


def zero_command_stair_position_cost(env, asset_cfg: SceneEntityCfg,
                                     command_name: str = "base_velocity") -> torch.Tensor:
    """Hold the post-braking stair position while allowing balance corrections.

    Position error is observable through the bounded vx/vy command feedback.
    No joint target is frozen, and the first braking interval is unpenalized.
    """
    heading_hold_velocity_commands(env, command_name)
    feedback = env._m20_heading_command_filter
    error = feedback["stop_error"]
    u = (error.abs() - TUNING["stair_stop_position_deadband"]).clamp_min(0.0)
    u = u / TUNING["stair_stop_position_scale"]
    cost = (2.0 * (torch.sqrt(1.0 + u.square()) - 1.0)).sum(-1)
    leg_speed = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    scaled = leg_speed / TUNING["zero_hold_leg_velocity_scale"]
    cost += 0.1 * (2.0 * (torch.sqrt(1.0 + scaled.square()) - 1.0)).mean(-1)
    return cost * feedback["stop_feedback"]


def zero_command_action_dynamics_cost(
    env, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity",
    leg_action_name: str = "joint_pos", wheel_action_name: str = "joint_vel",
) -> torch.Tensor:
    """Suppress stop-state target chatter without freezing a stair balance pose.

    Processed leg targets are radians; processed wheel targets become m/s.
    History validity depends on elapsed control ticks, including exactly-zero
    actions, and never on the terrain curriculum level.
    """
    step = int(env.common_step_counter)
    legs = env.action_manager.get_term(leg_action_name).processed_actions
    wheels = (-env.action_manager.get_term(wheel_action_name).processed_actions
              * TUNING["wheel_radius"])
    targets = torch.cat((legs, wheels), dim=-1)
    state = getattr(env, "_m20_stop_action_dynamics", None)
    if state is None or state["previous"].shape != targets.shape:
        state = dict(step=-1, previous=targets.clone(), older=targets.clone(),
                     history=torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
        env._m20_stop_action_dynamics = state
    if state["step"] != step:
        state["before"] = {key: state[key].clone() for key in ("previous", "older", "history")}
        state["step"] = step
    before = state["before"]
    reset = env.episode_length_buf <= 1
    valid_rate = (before["history"] >= 1) & ~reset
    valid_jerk = (before["history"] >= 2) & ~reset
    delta = targets - before["previous"]
    jerk = targets - 2.0 * before["previous"] + before["older"]
    leg_count = legs.shape[-1]

    def robust_mean(value, scale):
        u = value / scale
        return (2.0 * (torch.sqrt(1.0 + u.square()) - 1.0)).mean(-1)

    rate_cost = (robust_mean(delta[:, :leg_count], TUNING["zero_hold_leg_target_delta_scale"])
                 + robust_mean(delta[:, leg_count:], TUNING["zero_hold_wheel_target_delta_scale"]))
    jerk_cost = (robust_mean(jerk[:, :leg_count], TUNING["zero_hold_leg_target_jerk_scale"])
                 + robust_mean(jerk[:, leg_count:], TUNING["zero_hold_wheel_target_jerk_scale"]))
    command = env.command_manager.get_command(command_name)
    robot = env.scene[asset_cfg.name]
    hold, _, active = _zero_hold_heading_context(env, command, robot)
    age = (step - hold["hold_start_step"]) * env.step_dt
    ramp = ((age - TUNING["zero_hold_brake_grace_s"])
            / TUNING["zero_hold_action_ramp_s"]).clamp(0.0, 1.0)
    flat = _flat_surface_mask(env, SceneEntityCfg("height_scanner"))
    terrain_gain = torch.where(flat, 1.0, TUNING["zero_hold_stair_dynamics_gain"])
    result = (0.5 * rate_cost * valid_rate + jerk_cost * valid_jerk) * active * ramp * terrain_gain
    # Keep the state valid when an episode starts at a random episode age.
    state.update(previous=targets.clone(), older=before["previous"].clone(),
                 history=torch.where(reset, torch.ones_like(before["history"]), before["history"] + 1))
    measured = active & (ramp >= 1.0) & valid_jerk
    stats = getattr(env, "_m20_stop_action_stats", None)
    if stats is None:
        stats = {key: torch.zeros(env.num_envs, device=env.device) for key in (
            "samples", "leg_delta_sq", "wheel_delta_sq", "leg_jerk_sq", "wheel_jerk_sq", "leg_speed_sq")}
        env._m20_stop_action_stats = stats
    # Only one telemetry update per physical tick, regardless of reward consumers.
    if state.get("stats_step") != step:
        for value in stats.values():
            value[reset] = 0.0
        values = dict(samples=torch.ones_like(result),
                      leg_delta_sq=delta[:, :leg_count].square().mean(-1),
                      wheel_delta_sq=delta[:, leg_count:].square().mean(-1),
                      leg_jerk_sq=jerk[:, :leg_count].square().mean(-1),
                      wheel_jerk_sq=jerk[:, leg_count:].square().mean(-1),
                      leg_speed_sq=robot.data.joint_vel[:, asset_cfg.joint_ids].square().mean(-1))
        for key, value in values.items():
            stats[key] += value * measured
        state["stats_step"] = step
    return result


def stair_front_wheel_touchdown(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> torch.Tensor:
    """Credit one safely landed front wheel per new ascending riser, plus descent support."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    _stair_step_completion_events(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    down_state = env._m20_stair_step_state
    down_front = down_state["front_support_event"].float().sum(dim=1)
    return ((state["front_event"].float().sum(dim=1) + down_front)
            * _stair_forward_command_gate(env, command_name) / env.step_dt)


def stair_front_single_support(
    env,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Prefer single front support only when the robot covers new ground."""
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    progress_rate, _ = _stair_new_progress(env, asset_cfg, sensor_cfg, command_name)
    in_contact = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0.0
    one_support = in_contact.sum(dim=1) == 1
    descent = getattr(env, "_m20_stair_step_state", None)
    if descent is not None and not TUNING.get("v1_motion_baseline", False):
        command = env.command_manager.get_command(command_name)
        paired_descent = (
            descent["active"] & ~descent["up"] & (descent["down_successes"] < 3)
            & (command[:, 0] > TUNING["translation_threshold"])
            & (command[:, 1].abs() <= TUNING["translation_threshold"])
            & (command[:, 2].abs() <= TUNING["yaw_threshold"])
        )
        # Do not pay the one-front-loaded pattern while the descent term
        # explicitly asks for a short, balanced front-axle transfer.
        one_support &= ~paired_descent
    return (progress_rate * one_support.to(progress_rate.dtype)
            * _stair_route_quality(env, sensor_cfg, command_name)
            * _stair_forward_command_gate(env, command_name))


def stair_diagonal_support_symmetry(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Prefer diagonal support only during new forward stair progress."""
    sensor = env.scene.sensors[contact_sensor_cfg.name]
    progress_rate, _ = _stair_new_progress(env, asset_cfg, sensor_cfg, command_name)
    contact = sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0.0
    # body_ids are configured in FL, FR, HL, HR order.
    matches = (contact[:, 0] == contact[:, 3]) & (contact[:, 1] == contact[:, 2])
    return (progress_rate * matches.to(progress_rate.dtype)
            * _stair_route_quality(env, sensor_cfg, command_name)
            * _stair_forward_command_gate(env, command_name))


def stair_front_alternation(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> torch.Tensor:
    """Bonus after the first, only for the opposite front leg on the next riser."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    event = (state["front_event"].any(dim=1) & (state["front_count"] > 1)).float()
    return event * _stair_forward_command_gate(env, command_name) / env.step_dt


def _stair_step_completion_events(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    tread_inset: float = 0.06,
    stable_time_s: float = 0.06,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track a visible edge until both rear wheels reach its destination tread.

    Front wheels may already be on a later tread, so requiring all four wheels
    to stand on one tread would reject valid traversals of narrow stairs.
    """
    counter = int(getattr(env, "common_step_counter", -1))
    cache = getattr(env, "_m20_stair_step_completion_cache", None)
    if cache is not None and cache["counter"] == counter:
        return cache["up"], cache["down"]

    asset = env.scene[asset_cfg.name]
    scanner = env.scene[sensor_cfg.name]
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    state = getattr(env, "_m20_stair_step_state", None)
    if state is None or state["active"].shape[0] != env.num_envs:
        state = {
            "active": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "edge_xy": torch.zeros(env.num_envs, 2, device=env.device),
            "heading": torch.zeros(env.num_envs, 2, device=env.device),
            "target_z": torch.zeros(env.num_envs, device=env.device),
            "source_z": torch.zeros(env.num_envs, device=env.device),
            "rear_source_clearance": torch.full((env.num_envs, 2), float("nan"), device=env.device),
            "up": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "front_reached": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_source_clearance": torch.full((env.num_envs, 2), float("nan"), device=env.device),
            "front_previous_distance": torch.zeros(env.num_envs, 2, device=env.device),
            "front_previous_valid": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_crossed_safe": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_crossing_failed": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_face_collision": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_face_event": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_lift_escrow": torch.zeros(env.num_envs, 2, device=env.device),
            "front_support_time": torch.zeros(env.num_envs, 2, device=env.device),
            "front_support_paid": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_support_event": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_pair_paid": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "front_pair_event": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "front_lead_paid": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "front_lead_event": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "rear_support_time": torch.zeros(env.num_envs, 2, device=env.device),
            "rear_support_paid": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "rear_support_event": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "stable_time": torch.zeros(env.num_envs, device=env.device),
            "age": torch.zeros(env.num_envs, device=env.device),
            "completed_edge_xy": torch.zeros(env.num_envs, 64, 2, device=env.device),
            "completed_heading": torch.zeros(env.num_envs, 64, 2, device=env.device),
            "completed_target_z": torch.zeros(env.num_envs, 64, device=env.device),
            "completed_count": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "up_attempts": torch.zeros(env.num_envs, device=env.device),
            "down_attempts": torch.zeros(env.num_envs, device=env.device),
            "up_successes": torch.zeros(env.num_envs, device=env.device),
            "down_successes": torch.zeros(env.num_envs, device=env.device),
        }
        setattr(env, "_m20_stair_step_state", state)
    if hasattr(env, "episode_length_buf"):
        reset = env.episode_length_buf <= 1
        for key in ("active", "front_reached", "front_previous_valid", "front_crossed_safe", "front_crossing_failed",
                    "front_face_collision", "front_face_event", "front_lift_escrow", "front_support_time", "front_support_paid", "front_support_event",
                    "front_pair_paid", "front_pair_event", "front_lead_paid", "front_lead_event", "rear_support_time", "rear_support_paid",
                    "rear_support_event", "completed_count", "up_attempts", "down_attempts",
                    "up_successes", "down_successes", "stable_time", "age"):
            state[key][reset] = 0
        state["rear_source_clearance"][reset] = float("nan")
        state["front_source_clearance"][reset] = float("nan")

    forward = torch.zeros(env.num_envs, 3, device=env.device)
    forward[:, 0] = 1.0
    heading = quat_apply(yaw_quat(scanner.data.quat_w), forward)[:, :2]
    candidate_edge_xy = scanner.data.pos_w[:, :2] + heading * context["edge_x"].unsqueeze(1)
    # Remember every completed edge in this episode. Checking only the last
    # target would allow up-down-up oscillation to collect the first bonus twice.
    history_valid = (torch.arange(64, device=env.device).unsqueeze(0)
                     < state["completed_count"].unsqueeze(1))
    along_edge = (((candidate_edge_xy.unsqueeze(1) - state["completed_edge_xy"])
                   * state["completed_heading"]).sum(dim=-1)).abs()
    aligned = ((heading.unsqueeze(1) * state["completed_heading"]).sum(dim=-1)).abs() > 0.8
    same_edge = (history_valid & aligned & (along_edge < 0.15)
                 & ((context["upper_z"].unsqueeze(1) - state["completed_target_z"]).abs() < 0.06))
    start = (~state["active"] & (context["down_gate"] > 0) & ~same_edge.any(dim=1)
             & (state["completed_count"] < 64))
    state["edge_xy"][start] = candidate_edge_xy[start]
    state["heading"][start] = heading[start]
    state["target_z"][start] = context["upper_z"][start]
    state["source_z"][start] = context["lower_z"][start]
    state["rear_source_clearance"][start] = float("nan")
    state["front_source_clearance"][start] = float("nan")
    for key in ("front_previous_valid", "front_crossed_safe", "front_crossing_failed",
                "front_face_collision", "front_lift_escrow",
                "front_support_time", "front_support_paid", "rear_support_time", "rear_support_paid"):
        state[key][start] = 0
    state["front_pair_paid"][start] = False
    state["front_lead_paid"][start] = False
    state["up"][start] = context["edge_delta"][start] > 0
    state["front_reached"][start] = False
    state["stable_time"][start] = 0
    state["age"][start] = 0
    state["active"][start] = True
    state["up_attempts"][start & state["up"]] += 1
    state["down_attempts"][start & ~state["up"]] += 1

    wheel_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    signed_distance = ((wheel_pos[:, :, :2] - state["edge_xy"].unsqueeze(1))
                       * state["heading"].unsqueeze(1)).sum(dim=-1)
    wheel_height = wheel_pos[:, :, 2] - state["target_z"].unsqueeze(1)
    forces = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :]
    contact = (contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0) & (
        forces[:, :, 2] > 0.5 * torch.linalg.vector_norm(forces, dim=-1)
    )
    # Learn the wheel-center offset from a real source-tread contact instead
    # of assuming the reference model's wheel radius.
    front_baseline = state["front_source_clearance"]
    source_front = (
        state["active"][:, None] & contact[:, :2] & (signed_distance[:, :2] < -tread_inset)
        & ~torch.isfinite(front_baseline)
    )
    source_front_height = wheel_pos[:, :2, 2] - state["source_z"][:, None]
    source_front &= (source_front_height >= 0.02) & (source_front_height <= 0.25)
    front_baseline[source_front] = source_front_height[source_front]

    before = signed_distance[:, :2] < 0.0
    front_force_xy = torch.linalg.vector_norm(forces[:, :2, :2], dim=-1)
    face_now = (
        state["active"][:, None] & state["up"][:, None]
        & (signed_distance[:, :2] >= -0.12) & (signed_distance[:, :2] <= 0.03)
        & (wheel_height[:, :2] < front_baseline + 0.05)
        & (front_force_xy > 50.0)
        & (front_force_xy > 1.5 * forces[:, :2, 2].clamp_min(0.0))
    )
    state["front_face_event"] = face_now & ~state["front_face_collision"]
    state["front_face_collision"] |= face_now
    upward_speed = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[:2], 2].clamp_min(0.0)
    lift = (
        state["active"][:, None] & state["up"][:, None] & before
        & ~contact[:, :2] & torch.isfinite(front_baseline) & (upward_speed >= 0.10)
    )
    state["front_lift_escrow"] = torch.maximum(
        state["front_lift_escrow"],
        torch.where(lift, (upward_speed / 0.5).clamp_max(1.0), torch.zeros_like(upward_speed)),
    )
    # Height is checked on the control step that crosses the physical riser.
    crossing = (
        state["active"][:, None] & state["up"][:, None] & state["front_previous_valid"]
        & (state["front_previous_distance"] < 0.0) & (signed_distance[:, :2] >= 0.0)
    )
    clear = (
        torch.isfinite(front_baseline)
        & (wheel_height[:, :2] - front_baseline >= 0.05)
        & ~state["front_face_collision"]
    )
    state["front_crossing_failed"] |= crossing & ~clear
    state["front_crossed_safe"] |= crossing & clear & ~state["front_crossing_failed"]
    state["front_lift_escrow"][crossing & ~clear] = 0.0
    state["front_previous_distance"] = signed_distance[:, :2].clone()
    state["front_previous_valid"] = state["active"][:, None].expand_as(state["front_previous_valid"]).clone()

    # Reward upper/lower support only after a true tread contact persisted.
    destination_height = (
        torch.isfinite(front_baseline)
        & (wheel_height[:, :2] - front_baseline <= 0.05)
    )
    valid_crossing = torch.where(
        state["up"][:, None], state["front_crossed_safe"],
        state["front_previous_valid"] & (signed_distance[:, :2] > 0.0),
    )
    front_now = (
        state["active"][:, None] & contact[:, :2]
        & (signed_distance[:, :2] > tread_inset)
        & destination_height & valid_crossing
    )
    state["front_support_time"] = torch.where(
        front_now, state["front_support_time"] + env.step_dt,
        torch.zeros_like(state["front_support_time"]),
    )
    confirmed_front = state["front_support_time"] >= 0.04
    state["front_support_event"] = confirmed_front & ~state["front_support_paid"]
    state["front_support_paid"] |= state["front_support_event"]
    state["front_reached"] |= confirmed_front
    one_lead = (state["front_support_event"].sum(dim=1) == 1) & ~state["front_lead_paid"]
    state["front_lead_event"] = state["front_support_event"] & one_lead[:, None]
    state["front_lead_paid"] |= one_lead
    state["front_pair_event"] = (
        state["active"] & confirmed_front.all(dim=1) & ~state["front_pair_paid"]
    )
    state["front_pair_paid"] |= state["front_pair_event"]

    # Calibrate each rear wheel's center height while it contacts the source
    # tread. A fixed 2-25 cm range would mistake an upper-source contact for a
    # successful descent of a 10 cm step.
    source_clearance = wheel_pos[:, 2:, 2] - state["source_z"].unsqueeze(1)
    baseline = state["rear_source_clearance"]
    source_contact = (state["active"].unsqueeze(1) & contact[:, 2:]
                      & (signed_distance[:, 2:] < -tread_inset)
                      & (source_clearance >= 0.02) & (source_clearance <= 0.25)
                      & ~torch.isfinite(baseline))
    baseline[source_contact] = source_clearance[source_contact]
    target_contact = (contact[:, 2:] & (signed_distance[:, 2:] > tread_inset)
                      & torch.isfinite(baseline)
                      & (wheel_height[:, 2:] - baseline <= 0.05))
    state["rear_support_time"] = torch.where(
        target_contact & state["front_reached"].all(dim=1, keepdim=True),
        state["rear_support_time"] + env.step_dt,
        torch.zeros_like(state["rear_support_time"]),
    )
    rear_confirmed = state["rear_support_time"] >= 0.04
    state["rear_support_event"] = rear_confirmed & ~state["rear_support_paid"]
    state["rear_support_paid"] |= state["rear_support_event"]
    rear_now = state["rear_support_paid"].all(dim=1) & target_contact.all(dim=1)
    upright = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1) < 0.57
    stable = state["active"] & state["front_reached"].all(dim=1) & rear_now & upright
    state["stable_time"] = torch.where(stable, state["stable_time"] + env.step_dt,
                                       torch.zeros_like(state["stable_time"]))
    completed = stable & (state["stable_time"] >= stable_time_s)
    up = completed & state["up"]
    down = completed & ~state["up"]
    state["up_successes"][up] += 1
    state["down_successes"][down] += 1
    completed_ids = torch.nonzero(completed, as_tuple=False).squeeze(-1)
    completed_slots = state["completed_count"][completed_ids]
    state["completed_edge_xy"][completed_ids, completed_slots] = state["edge_xy"][completed_ids]
    state["completed_heading"][completed_ids, completed_slots] = state["heading"][completed_ids]
    state["completed_target_z"][completed_ids, completed_slots] = state["target_z"][completed_ids]
    state["completed_count"][completed_ids] += 1
    state["active"][completed] = False

    state["age"][state["active"]] += env.step_dt
    root_distance = ((asset.data.root_pos_w[:, :2] - state["edge_xy"]) * state["heading"]).sum(dim=1)
    stale = state["active"] & ((state["age"] > 10.0) | (root_distance < -0.70))
    state["active"][stale] = False
    setattr(env, "_m20_stair_step_completion_cache", {"counter": counter, "up": up, "down": down})
    return up, down


def _ascent_completion_speed_scale(env, command_name="base_velocity"):
    """Reduce the extra completion payout from racing above the requested speed.

    This adjusts two positive task-completion terms, leaving style-credit
    accounting, safety costs and every descent/zero/manual mode unchanged.
    """
    if TUNING.get("v1_motion_baseline", False):
        return torch.ones(env.num_envs, device=env.device)
    command = env.command_manager.get_command(command_name)
    robot = env.scene["robot"]
    _, active = _automatic_ascent_heading_rate(env, command, robot)
    allowed = (command[:, 0] + TUNING["ascent_forward_overspeed_tolerance"]).clamp_min(0.1)
    scale = allowed / torch.maximum(robot.data.root_lin_vel_b[:, 0], allowed)
    return torch.where(active, scale, torch.ones_like(scale))


def stair_up_step_completion(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Credit a safe front landing on the next ascending riser, one leg only."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return (state["front_event"].any(dim=1).float() * _stair_forward_command_gate(env)
            * _ascent_completion_speed_scale(env) / env.step_dt)


def ascent_task_crossing_events(state, known, x, cross, wheels, loaded, root_pos, root_quat):
    """Whole-body ascent, regardless of gait style; no pay for bypass/retreat."""
    # map_safe requires the stylistic airborne edge crossing. Task completion
    # instead requires genuine stable upper-tread support, and remains payable
    # after a recovered face contact. Impact has a separate negative reward.
    supported = state["map_landed"]
    both_axles = supported[:, :, :2].any(-1) & supported[:, :, 2:].any(-1)
    beyond = (x > state["map_half_width"][:, :, None]).all(-1)
    corridor = (cross < TUNING["tread_observation_corridor"]).all(-1)
    bottom = wheels[:, None, :, 2] - state["radius"][:, None]
    above = (bottom >= state["map_z"][:, :, None] - TUNING["tread_height_tolerance"]).all(-1)
    upright = (1 - 2 * root_quat[:, 1:3].square().sum(-1)) > 0.35
    body_above = root_pos[:, 2, None] >= state["map_z"] + 0.22
    return known & both_axles & beyond & corridor & above & body_above & upright[:, None] & (
        loaded.sum(-1) >= 2)[:, None] & ~state["map_task_paid"]


def stair_up_task_crossing(env, asset_cfg, sensor_cfg, contact_sensor_cfg):
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return (state["task_crossing_event"] * _stair_forward_command_gate(env)
            * _ascent_completion_speed_scale(env) / env.step_dt)


def stair_up_rear_step_completion(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Credit one alternating rear landing per front-confirmed ascending riser."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return state["rear_event"].any(dim=1).float() * _stair_forward_command_gate(env) / env.step_dt


def stair_rear_transition_recovery(env, asset_cfg, sensor_cfg, contact_sensor_cfg):
    """Small once-only signal for a safe first step out of rear same-tread gait."""
    event = ascent_state(env,asset_cfg,sensor_cfg,contact_sensor_cfg)["rear_recovery_event"].float()
    return event * _stair_forward_command_gate(env) / env.step_dt


def stair_rear_style_refund(env, asset_cfg, sensor_cfg, contact_sensor_cfg):
    """Withdraw earlier rear style credit when that tread later becomes shared."""
    return ascent_state(env,asset_cfg,sensor_cfg,contact_sensor_cfg)["rear_style_refund"]/env.step_dt


def stair_up_order_violation_cost(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """One-shot cost for unsafe, simultaneous, or wrong-side riser landings."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return (state["front_violation_event"].float() + state["rear_violation_event"].float()) / env.step_dt


def stair_up_same_tread_cost(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Cost once per riser if both front or both rear wheels share its tread."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return state["duplicate_event_count"].sum(-1) / env.step_dt


def stair_up_same_tread_dwell_cost(env, asset_cfg, sensor_cfg, contact_sensor_cfg):
    """Bounded time cost on intermediate treads; short support transfers are free."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return state["same_tread_dwell"].mean(dim=1)


def stair_down_step_completion(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """One event when a whole riser has been descended and rear support is stable."""
    _, down = _stair_step_completion_events(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return down.to(torch.float32) * _stair_forward_command_gate(env) / env.step_dt


def stair_front_riser_face_cost(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> torch.Tensor:
    """One-shot cost for a loaded front wheel striking the rising face."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    return state["front_face_event"].float().sum(dim=1) / env.step_dt


def stair_descent_rear_support(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> torch.Tensor:
    """Small one-shot credit for each rear wheel safely supporting the lower tread."""
    _stair_step_completion_events(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    state = env._m20_stair_step_state
    return (
        state["rear_support_event"].float().sum(dim=1) * (~state["up"]).float()
        * _stair_forward_command_gate(env, command_name) / env.step_dt
    )


def stair_descent_entry_rear_pair_swing_cost(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize both rear wheels swinging together during the first descent transfer.

    This only applies after both front wheels have confirmed support on the lower
    tread, while the latched physical entry is active and before either rear
    wheel has established lower-tread support. Keeping one rear wheel loaded
    gives the policy a clear staggered-transfer target without changing later
    stair steps.
    """
    _stair_step_completion_events(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    step = getattr(env, "_m20_stair_step_state", None)
    entry = getattr(env, "_m20_descent_entry_state", None)
    cost = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if step is None or entry is None:
        return cost

    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :]
    rear_loaded = (
        (contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0)
        & (forces[:, :, 2] > 0.5 * torch.linalg.vector_norm(forces, dim=-1))
    )[:, 2:]
    first_transfer = (
        entry["active"] & ~step["up"] & step["active"]
        & step["front_reached"].all(dim=1)
        & ~step["rear_support_paid"].any(dim=1)
        & ~entry["settled"].any(dim=1)
    )
    both_rears_unloaded = ~rear_loaded.any(dim=1)
    return (first_transfer & both_rears_unloaded).to(torch.float32)


def stair_route_track_lin_vel_xy_exp(
    env, std: float, command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hip_cfg=None, knee_cfg=None, contact_sensor_cfg=None,
) -> torch.Tensor:
    """Retain most native tracking; unsafe posture forfeits a bounded fraction.

    Do not suppress driving throughout an entire swing/target or require a
    periodic gait clock. Legacy saved calls without gait entities stay native.
    """
    from .rewards import track_lin_vel_xy_exp
    if TUNING.get('stair_path_feedback', False):
        from .path.stair_path_mdp import controller
        from .path.stair_path_objectives import velocity_tracking_reward
        raw = env.command_manager.get_command(command_name)
        effective = heading_hold_velocity_commands(env, command_name)
        control = controller(env)
        feedback = control.straight & (control.active | (control.correction.abs() > 1.e-6))
        return velocity_tracking_reward(raw, effective,
            env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2], feedback, std)
    native = track_lin_vel_xy_exp(env,std,command_name,asset_cfg)
    if TUNING.get("v1_motion_baseline", False):
        # v1's recorded pose_tracking_discount is zero; target is raw XY.
        return native
    robot = env.scene[asset_cfg.name]
    effective = heading_hold_velocity_commands(env, command_name)
    effective_error = (effective[:, :2] - robot.data.root_lin_vel_b[:, :2]).square().sum(-1)
    # The actor receives bounded XY corrections. Reward those same targets;
    # modes outside feedback retain their original velocity target.
    native = torch.where(env._m20_heading_command_filter["translation_active"],
                         torch.exp(-effective_error / std**2), native)
    if hip_cfg is None or knee_cfg is None or contact_sensor_cfg is None:
        return native
    c = _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg,command_name)
    severity = torch.maximum((c["front_fold"]+c["front_swing_fold"]).mean(-1),c["rear_forward"].mean(-1))
    severity = torch.maximum(severity,c["body_gap"])
    return native*(1-TUNING["pose_tracking_discount"]*severity)


def stair_ascent_straight_motion_cost(
    env, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> torch.Tensor:
    """Separate ascent side-slip and overspeed from small heading errors.

    The path target is exposed through the actor's existing lateral command.
    Penalize velocity residuals against that correction, so returning to the
    line is permitted. No term here applies to zero holds or descending travel.
    """
    tick = int(env.common_step_counter)
    cached = getattr(env, "_m20_ascent_straight_motion_cache", None)
    if cached is not None and cached["tick"] == tick:
        return cached["cost"]
    robot = env.scene[asset_cfg.name]
    raw = env.command_manager.get_command(command_name)
    effective = heading_hold_velocity_commands(env, command_name)
    _, active = _automatic_ascent_heading_rate(env, raw, robot)
    active &= env.episode_length_buf > 1
    heading = env._m20_zero_hold_heading_state
    yaw = heading["straight_target_yaw"]
    normal = torch.stack((-torch.sin(yaw), torch.cos(yaw)), -1)
    path_error = ((robot.data.root_pos_w[:, :2] - heading["straight_origin_xy"]) * normal).sum(-1)
    lateral_speed = (robot.data.root_lin_vel_w[:, :2] * normal).sum(-1)
    desired_body = torch.zeros_like(robot.data.root_lin_vel_b)
    desired_body[:, :2] = effective[:, :2]
    desired_world = quat_apply(robot.data.root_quat_w, desired_body)
    desired_side = (desired_world[:, :2] * normal).sum(-1)
    lateral_error = lateral_speed - desired_side
    forward_speed = robot.data.root_lin_vel_b[:, 0]
    overspeed = (forward_speed - raw[:, 0] - TUNING["ascent_forward_overspeed_tolerance"]).clamp_min(0.0)

    def smooth_excess(value, tolerance, scale):
        u = (value.abs() - tolerance).clamp_min(0.0) / scale
        return 2.0 * (torch.sqrt(1.0 + u.square()) - 1.0)

    cost = smooth_excess(path_error, TUNING["ascent_path_error_tolerance"], TUNING["ascent_path_error_scale"])
    cost += 0.5 * smooth_excess(lateral_error, TUNING["ascent_path_velocity_error_tolerance"],
                              TUNING["ascent_path_velocity_error_scale"])
    cost += smooth_excess(overspeed, 0.0, TUNING["ascent_forward_overspeed_scale"])
    cost *= active
    stats = getattr(env, "_m20_ascent_straight_motion_stats", None)
    values = {
        "samples": torch.ones_like(cost), "path_abs": path_error.abs(), "path_signed": path_error,
        "side_abs": lateral_speed.abs(), "side_signed": lateral_speed,
        "side_error_abs": lateral_error.abs(), "forward": forward_speed,
        "overspeed": (overspeed > 0).float(), "correction": (effective[:, 1] - raw[:, 1]).abs(),
        "cost": cost,
    }
    if stats is None:
        stats = {key: torch.zeros(env.num_envs, device=env.device) for key in values}
        env._m20_ascent_straight_motion_stats = stats
    reset = env.episode_length_buf <= 1
    for key, value in values.items():
        stats[key][reset] = 0.0
        stats[key] += value * active
    env._m20_ascent_straight_motion_cache = {"tick": tick, "cost": cost}
    return cost


def ascent_straight_motion_metrics(env, env_ids):
    """Report signed lateral drift and speed separately from heading error."""
    stats = getattr(env, "_m20_ascent_straight_motion_stats", None)
    if stats is None:
        return {}
    sums = {key: value[env_ids].sum() for key, value in stats.items()}
    count = sums["samples"]
    result = {"straight_motion_samples": count}
    for name, field in (
        ("path_error_mean_m", "path_abs"), ("path_error_signed_mean_m", "path_signed"),
        ("lateral_speed_mean_m_s", "side_abs"), ("lateral_speed_signed_mean_m_s", "side_signed"),
        ("lateral_tracking_error_mean_m_s", "side_error_abs"), ("forward_speed_mean_m_s", "forward"),
        ("overspeed_fraction", "overspeed"), ("lateral_correction_mean_m_s", "correction"),
        ("straight_motion_cost_mean", "cost"),
    ):
        result[name] = sums[field] / count.clamp_min(1.0)
        result[f"__weighted_total__/{name}"] = sums[field]
        result[f"__weighted_count__/{name}"] = count
    return result


def stair_descent_straight_motion_cost(env, asset_cfg: SceneEntityCfg,
                                       command_name: str = "base_velocity") -> torch.Tensor:
    """Penalize descent side slip against the same route target the actor sees."""
    tick = int(env.common_step_counter)
    cached = getattr(env, "_m20_descent_motion_cache", None)
    if cached is not None and cached["tick"] == tick:
        return cached["cost"]
    robot = env.scene[asset_cfg.name]
    raw = env.command_manager.get_command(command_name)
    effective = heading_hold_velocity_commands(env, command_name)
    feedback = env._m20_heading_command_filter
    active = feedback["automatic_descent"] & (env.episode_length_buf > 1)
    yaw = env._m20_zero_hold_heading_state["straight_target_yaw"]
    normal = torch.stack((-torch.sin(yaw), torch.cos(yaw)), -1)
    desired = torch.zeros_like(robot.data.root_lin_vel_b)
    desired[:, :2] = effective[:, :2]
    desired_side = (quat_apply(robot.data.root_quat_w, desired)[:, :2] * normal).sum(-1)
    residual = feedback["path_speed"] - desired_side
    def smooth(value, tolerance, scale):
        u = (value.abs() - tolerance).clamp_min(0.0) / scale
        return 2.0 * (torch.sqrt(1.0 + u.square()) - 1.0)
    cost = (smooth(feedback["path_error"], 0.04, 0.12)
            + 0.5 * smooth(residual, 0.025, 0.10)) * active
    values = dict(samples=torch.ones_like(cost), path_abs=feedback["path_error"].abs(),
                  path_signed=feedback["path_error"], side_abs=feedback["path_speed"].abs(),
                  side_signed=feedback["path_speed"], tracking=residual.abs(),
                  correction=(effective[:, 1] - raw[:, 1]).abs(), forward=robot.data.root_lin_vel_b[:, 0])
    stats = getattr(env, "_m20_descent_motion_stats", None)
    if stats is None:
        stats = {key: torch.zeros(env.num_envs, device=env.device) for key in values}
        env._m20_descent_motion_stats = stats
    for key, value in values.items():
        stats[key][env.episode_length_buf <= 1] = 0.0
        stats[key] += value * active
    env._m20_descent_motion_cache = dict(tick=tick, cost=cost)
    return cost


def _descent_motion_metrics(env, env_ids):
    stats = getattr(env, "_m20_descent_motion_stats", None)
    if stats is None:
        return {}
    sums = {key: value[env_ids].sum() for key, value in stats.items()}
    count = sums["samples"]
    result = {"straight_motion_samples": count}
    for name, field in (("path_error_mean_m", "path_abs"), ("path_error_signed_mean_m", "path_signed"),
                        ("lateral_speed_mean_m_s", "side_abs"), ("lateral_speed_signed_mean_m_s", "side_signed"),
                        ("lateral_tracking_error_mean_m_s", "tracking"),
                        ("lateral_correction_mean_m_s", "correction"), ("forward_speed_mean_m_s", "forward")):
        result[name] = sums[field] / count.clamp_min(1.0)
        result[f"__weighted_total__/{name}"] = sums[field]
        result[f"__weighted_count__/{name}"] = count
    return result


def _legacy_step_completion_metrics(env, env_ids) -> dict[str, torch.Tensor]:
    """Report front and rear ascent sequence results alongside descent results."""
    down_state = getattr(env, "_m20_stair_step_state", None)
    up_state = getattr(env, "_m20_ascent_state", None)
    zero = torch.zeros((), device=env.device)
    result = {}
    if up_state is None:
        result.update({key: zero for key in ("up_attempts", "up_successes", "up_success_rate",
                 "up_rear_successes", "up_front_violations", "up_rear_violations")})
    else:
        attempts = up_state["front_count"][env_ids].float()
        successes = up_state["front_successes"][env_ids]
        result["up_attempts"] = attempts.mean()
        result["up_successes"] = successes.mean()
        result["up_success_rate"] = successes.sum() / attempts.sum().clamp(min=1.0)
        result["up_rear_successes"] = up_state["rear_successes"][env_ids].mean()
        result["up_front_violations"] = up_state["front_violations"][env_ids].mean()
        result["up_rear_violations"] = up_state["rear_violations"][env_ids].mean()
        for axle, name in enumerate(("front", "rear")):
            result[f"up_{name}_same_tread_events"] = up_state["same_tread_events_total"][env_ids, axle].mean()
            result[f"up_{name}_same_tread_dwell_s"] = up_state["same_tread_seconds_total"][env_ids, axle].mean()
        for metric, field in (("up_targets_started", "targets_started"),
                              ("up_lift_credit", "front_lift_credit"), ("up_retries", "retry_count"),
                              ("up_abandoned_targets", "abandoned_targets")):
            result[metric] = up_state[field][env_ids].mean()
        forward_steps = up_state["forward_steps"][env_ids].sum().clamp_min(1.0)
        result["up_retreat_fraction"] = up_state["retreat_steps"][env_ids].sum() / forward_steps
        result["up_stationary_fraction"] = up_state["stationary_steps"][env_ids].sum() / forward_steps
    if down_state is None:
        result.update({key: zero for key in ("down_attempts", "down_successes", "down_success_rate")})
    else:
        attempts = down_state["down_attempts"][env_ids]
        successes = down_state["down_successes"][env_ids]
        result["down_attempts"] = attempts.mean()
        result["down_successes"] = successes.mean()
        result["down_success_rate"] = successes.sum() / attempts.sum().clamp(min=1.0)
    gate_stats = getattr(env, "_m20_stair_gate_stats", None)
    if gate_stats is not None:
        steps = gate_stats["steps"][env_ids].sum().clamp(min=1.0)
        result["forward_gate_fraction"] = gate_stats["forward"][env_ids].sum() / steps
        result["nearby_gate_fraction"] = gate_stats["nearby"][env_ids].sum() / steps
    else:
        result["forward_gate_fraction"] = zero
        result["nearby_gate_fraction"] = zero
    return result


def stair_impact_cost(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    free_fall_speed: float = 0.10,
) -> torch.Tensor:
    """Penalize hard wheel touchdowns during a stair traversal."""
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, contact_sensor_cfg.body_ids].to(torch.float32)
    downward_speed = (-asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2] - free_fall_speed).clamp(min=0.0)
    return context["terrain_gate"] * torch.sum(first_contact * downward_speed.square(), dim=1)


def stair_non_wheel_contact_cost(
    env, sensor_cfg: SceneEntityCfg, stair_sensor_cfg: SceneEntityCfg,
    force_threshold: float = 20.0, force_scale: float = 180.0,
) -> torch.Tensor:
    """Bounded cost for shank or leg contact with a stair."""
    sensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.vector_norm(
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1
    ).amax(dim=1)
    excess = ((force - force_threshold).clamp_min(0.0) / force_scale).square()
    gate = _stair_scan_context(env, stair_sensor_cfg)["terrain_gate"]
    state = getattr(env, "_m20_stair_step_state", None)
    if state is not None:
        gate = torch.maximum(gate, state["active"].float())
    ascent = getattr(env, "_m20_ascent_state", None)
    if ascent is not None:
        gate = torch.maximum(gate, ascent["active"].float())
    return gate * excess.sum(dim=1).clamp_max(1.0)


def stair_non_wheel_scrape_cost(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
    stair_sensor_cfg: SceneEntityCfg, force_threshold: float = 8.0,
    speed_threshold: float = 0.08, speed_scale: float = 0.55,
) -> torch.Tensor:
    """Bounded cost when a loaded non-wheel link slides against a stair."""
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    force = torch.linalg.vector_norm(
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1
    ).amax(dim=1)
    speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    scrape = ((speed - speed_threshold).clamp_min(0.0) / speed_scale).square()
    gate = _stair_scan_context(env, stair_sensor_cfg)["terrain_gate"]
    state = getattr(env, "_m20_stair_step_state", None)
    if state is not None:
        gate = torch.maximum(gate, state["active"].float())
    ascent = getattr(env, "_m20_ascent_state", None)
    if ascent is not None:
        gate = torch.maximum(gate, ascent["active"].float())
    return gate * (scrape * (force > force_threshold).float()).sum(dim=1).clamp_max(1.0)


def stair_joint_effort_cost(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Normalized squared actuator effort while stepping on stairs."""
    asset = env.scene[asset_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    limits = getattr(env, "_m20_stair_actuator_effort_limits", None)
    if limits is None or limits.shape != asset.data.applied_torque.shape:
        limits = torch.zeros_like(asset.data.applied_torque)
        assigned = torch.zeros(limits.shape[1], dtype=torch.bool, device=limits.device)
        for actuator in asset.actuators.values():
            joint_ids = actuator.joint_indices
            limits[:, joint_ids] = actuator.effort_limit.abs()
            assigned[joint_ids] = True
        if not torch.all(assigned) or not torch.all(torch.isfinite(limits)) or not torch.all(limits > 0):
            raise RuntimeError("M20 stair effort cost requires finite physical limits for every actuator joint.")
        setattr(env, "_m20_stair_actuator_effort_limits", limits)
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return context["terrain_gate"] * torch.mean((torque / limits[:, asset_cfg.joint_ids]).square(), dim=1)


def stair_joint_acceleration_cost(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    normalization_acceleration: float = 60.0,
) -> torch.Tensor:
    """Bounded leg acceleration cost; wheel collisions cannot dominate it."""
    asset = env.scene[asset_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    acceleration = asset.data.joint_acc[:, asset_cfg.joint_ids]
    normalized = acceleration / max(normalization_acceleration, 1.0)
    return context["terrain_gate"] * torch.mean(normalized.square(), dim=1).clamp_max(1.0)


def stair_mechanical_power_cost(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    normalization_power: float = 1800.0,
) -> torch.Tensor:
    """Normalized absolute joint mechanical power while traversing stairs."""
    asset = env.scene[asset_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    power = torch.abs(torque * velocity).mean(dim=1) / max(normalization_power, 1.0)
    return context["terrain_gate"] * power




# =============================================================================
# 6. Phase-aware leg folding, compact turns and gait metrics
# =============================================================================

def pure_turn(command):
    return ((torch.linalg.vector_norm(command[:, :2], dim=1) < TUNING["translation_threshold"])
            & (command[:, 2].abs() > TUNING["yaw_threshold"]))


def turn_tracking_quality(yaw, command):
    """Stationary leg dancing or turning opposite to the command earns zero."""
    progress = (yaw * command.sign() / command.abs().clamp_min(TUNING["yaw_threshold"])).clamp(0, 1)
    return progress * torch.exp(-((yaw - command) / 0.5).square())


def bounded_air_score(air, low, target, high):
    """Time-valued triangle; zero for nonflight or prolonged flight."""
    peak = target - low
    return torch.minimum((air - low).clamp_min(0),
                         ((high - air) / (high - target) * peak).clamp_min(0))


def fold_excess(extension, knee, min_extension=None, max_knee=None):
    min_extension = TUNING["rear_min_extension"] if min_extension is None else min_extension
    max_knee = TUNING["rear_max_knee"] if max_knee is None else max_knee
    length = ((min_extension - extension).clamp_min(0)
              / TUNING["extension_scale"]).square()
    angle = ((knee.abs() - max_knee).clamp_min(0)
             / TUNING["knee_scale"]).square()
    return torch.maximum(length, angle).clamp_max(1)


def turn_size_excess(clearance, reach, terrain):
    height_limit = TUNING["turn_height_flat"] + terrain[:, None] * (
        TUNING["turn_height_terrain"] - TUNING["turn_height_flat"])
    reach_limit = TUNING["turn_reach_flat"] + terrain[:, None] * (
        TUNING["turn_reach_terrain"] - TUNING["turn_reach_flat"])
    return (((clearance - height_limit).clamp_min(0) / TUNING["height_scale"]).square()
            + ((reach - reach_limit).clamp_min(0) / TUNING["reach_scale"]).square()).clamp_max(1)


def _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg,
             command_name="base_velocity"):
    step = int(env.common_step_counter)
    cached = getattr(env, "_m20_gait_context", None)
    if cached is not None and cached["step"] == step:
        return cached
    scan = _stair_scan_context(env, sensor_cfg, command_name)
    robot = env.scene[asset_cfg.name]
    contact = env.scene.sensors[contact_sensor_cfg.name]
    wheels = robot.data.body_pos_w[:, asset_cfg.body_ids]
    hips = robot.data.body_pos_w[:, hip_cfg.body_ids]
    hits = env.scene[sensor_cfg.name].data.ray_hits_w
    valid = torch.isfinite(hits).all(dim=-1)
    distance = torch.linalg.vector_norm(wheels[:, :, None, :2] - hits[:, None, :, :2], dim=-1)
    distance = torch.where(valid[:, None], distance, torch.inf)
    closest, index = distance.min(dim=-1)
    ground = hits[..., 2].gather(1, index)
    ground_valid = (closest < TUNING["nearest_valid_distance"]) & torch.isfinite(ground)
    # Missing terrain cannot turn into a fabricated large clearance reward/cost.
    ground = torch.where(ground_valid, ground, wheels[..., 2] - TUNING["wheel_radius"])
    force = contact.data.net_forces_w[:, contact_sensor_cfg.body_ids]
    airborne = contact.data.current_contact_time[:, contact_sensor_cfg.body_ids] <= 0
    support = ((force[..., 2] > 5.0)
               & (force[..., 2] > 0.5 * torch.linalg.vector_norm(force, dim=-1))
               & (contact.data.current_contact_time[:, contact_sensor_cfg.body_ids] >= TUNING["support_grace"]))
    clearance = wheels[..., 2] - ground - TUNING["wheel_radius"]
    support &= ground_valid & (clearance.abs() <= 0.05)
    command = env.command_manager.get_command(command_name)
    down = scan["down_gate"] > 0
    descent = getattr(env, "_m20_stair_step_state", None)
    down_follow = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    if descent is not None:
        close = torch.linalg.vector_norm(wheels[:,:,:2]-descent["edge_xy"][:,None],dim=-1).amin(-1) < TUNING["pose_phase_radius"]
        down_follow = descent["active"] & ~descent["up"] & (env.episode_length_buf > 1) & close
        down |= down_follow
    phase = getattr(env, "_m20_descent_pose_phase", None)
    if phase is None:
        phase = dict(active=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device),
                     xy=torch.zeros(env.num_envs,2,device=env.device),
                     heading=torch.zeros(env.num_envs,2,device=env.device),
                     lower=torch.zeros(env.num_envs,device=env.device))
        env._m20_descent_pose_phase = phase
    phase["active"][env.episode_length_buf <= 1] = False
    seen_down = scan["down_gate"] > 0
    forward = wheels.new_zeros(env.num_envs,3); forward[:,0] = 1
    heading = quat_apply(yaw_quat(env.scene[sensor_cfg.name].data.quat_w),forward)[:,:2]
    edge_xy = env.scene[sensor_cfg.name].data.pos_w[:,:2] + heading*scan["edge_x"][:,None]
    phase["xy"][seen_down] = edge_xy[seen_down]
    phase["heading"][seen_down] = heading[seen_down]
    # Scan lower_z/upper_z name the pre/post-edge columns, not min/max height.
    phase["lower"][seen_down] = torch.minimum(scan["lower_z"],scan["upper_z"])[seen_down]
    phase["active"] |= seen_down
    dist = torch.linalg.vector_norm(wheels[:,:,:2]-phase["xy"][:,None],dim=-1).amin(-1)
    past = ((wheels[:,:,:2]-phase["xy"][:,None])*phase["heading"][:,None]).sum(-1)
    rear_loaded = (force[:,2:,2]>5) & (force[:,2:,2]>.5*torch.linalg.vector_norm(force[:,2:],dim=-1)) & ~airborne[:,2:]
    rear_low = wheels[:,2:,2]-TUNING["wheel_radius"] <= phase["lower"][:,None]+.05
    finished = (past[:,2:]>.06).all(-1) & rear_low.all(-1) & rear_loaded.all(-1)
    aligned = (heading*phase["heading"]).sum(-1) > 0.5
    phase["active"] &= (dist < TUNING["pose_phase_radius"]) & aligned & ~(finished & ~seen_down)
    down |= phase["active"]
    down &= command[:, 0] > TUNING["translation_threshold"]
    # All visible upward edges include the one recently passed by the trunk.
    # An active target also keeps the gate through a temporary scan transition.
    rising_edges = scan.get("ascending_edges_valid")
    up = (scan["up_gate"] > 0) if rising_edges is None else rising_edges.any(dim=1)
    ascent = ascent_state(env,asset_cfg,sensor_cfg,contact_sensor_cfg,command_name)
    if ascent is not None:
        up |= ascent["active"] & (env.episode_length_buf > 1)
    up &= (command[:, 0] > TUNING["translation_threshold"]) & ~down
    extension = torch.linalg.vector_norm(wheels - hips, dim=-1)
    reach = torch.linalg.vector_norm((wheels - hips)[..., :2], dim=-1)
    knee = robot.data.joint_pos[:, knee_cfg.joint_ids]
    turn = pure_turn(command)
    size = turn_size_excess(clearance, reach, scan["terrain_gate"])
    size = torch.where(ground_valid & airborne, size, 0.0)
    # Force-supported posture is measurable without a terrain-nearest point.
    # Ground validity remains required only by ground-relative turn/clearance.
    load = (force[...,2] > 5) & (force[...,2] > 0.5*torch.linalg.vector_norm(force,dim=-1)) & ~airborne
    load_gain = (force[...,2] / TUNING["pose_load_force_scale"]).clamp(0,1)
    fold = torch.where(load[:,2:] & down[:,None], fold_excess(extension[:,2:],knee[:,2:]),0)
    rise = scan["edge_delta"].abs()
    if ascent is not None:
        rise = torch.maximum(rise,torch.where(ascent["active"],
            (ascent["target_z"]-ascent["source_z"]).abs(),0))
    if descent is not None:
        rise = torch.maximum(rise,torch.where(down_follow,
            (descent["target_z"]-descent["source_z"]).abs(),0))
    relax = ((rise-TUNING["pose_low_rise"]) / (TUNING["pose_high_rise"]-TUNING["pose_low_rise"])).clamp(0,1)
    a_support = TUNING["front_a_support_low_deg"] + relax*(TUNING["front_a_support_high_deg"]-TUNING["front_a_support_low_deg"])
    a_swing = TUNING["front_a_swing_low_deg"] + relax*(TUNING["front_a_swing_high_deg"]-TUNING["front_a_swing_low_deg"])
    front_a = (torch.pi-knee[:,:2].abs()).clamp_min(0)
    front_support = load[:,:2] & up[:,None]
    # Face brushing cannot pretend to be valid support and escape the wider
    # swing envelope; no initial 0.2s free deep fold and no nearest-map veto.
    front_swing = ~load[:,:2] & up[:,None]
    def bounded(value):
        square = value.clamp_min(0).square()
        return square/(1+square)
    front_loaded_angle = bounded((torch.deg2rad(a_support)[:,None]-front_a)/TUNING["pose_angle_scale"])
    front_air_angle = bounded((torch.deg2rad(a_swing)[:,None]-front_a)/TUNING["pose_angle_scale"])
    support_length = TUNING["front_min_extension"]+relax*(TUNING["front_support_length_high"]-TUNING["front_min_extension"])
    swing_length = TUNING["front_swing_min_extension_v3"]+relax*(TUNING["front_swing_length_high"]-TUNING["front_swing_min_extension_v3"])
    front_loaded_length = bounded((support_length[:,None]-extension[:,:2])/TUNING["pose_length_scale"])
    front_air_length = bounded((swing_length[:,None]-extension[:,:2])/TUNING["pose_length_scale"])
    # Use the worse of angle/length, since both describe the same flexion.
    front_fold = torch.maximum(front_loaded_angle,front_loaded_length)*front_support*load_gain[:,:2]
    front_swing_fold = torch.maximum(front_air_angle,front_air_length)*front_swing
    delta = wheels-hips
    qb = robot.data.root_quat_w[:,None].expand(-1,4,-1).reshape(-1,4)
    body_delta = quat_apply_inverse(qb,delta.reshape(-1,3)).reshape(-1,4,3)
    descent_front_posture_cost = torch.zeros(env.num_envs, 2, device=env.device)
    descent_step = getattr(env, "_m20_stair_step_state", None)
    if descent_step is not None:
        first_three_down = (descent_step["active"] & ~descent_step["up"]
                            & (descent_step["down_successes"] < 3)
                            & (command[:, 0] > TUNING["translation_threshold"]))
        level_index = descent_step["down_successes"].long().clamp(0, 2)
        reference = getattr(env, "_m20_descent_front_posture_reference", None)
        if reference is None or reference["forward"].shape[0] != env.num_envs:
            reference = {
                "forward": body_delta[:, :2, 0].clone(),
                "knee": robot.data.default_joint_pos[:, knee_cfg.joint_ids[:2]].clone(),
            }
            env._m20_descent_front_posture_reference = reference
        reset = env.episode_length_buf <= 1
        reference["forward"][reset] = body_delta[reset, :2, 0]
        reference["knee"][reset] = robot.data.default_joint_pos[reset][:, knee_cfg.joint_ids[:2]]
        flat_straight = (
            _flat_surface_mask(env, sensor_cfg)
            & (scan["down_gate"] <= 0.0) & (scan["up_gate"] <= 0.0)
            & (command[:, 0] > TUNING["translation_threshold"])
            & (command[:, 1].abs() <= TUNING["translation_threshold"])
            & (command[:, 2].abs() <= TUNING["yaw_threshold"])
            & ~reset
        )
        reference["forward"] = torch.where(
            flat_straight[:, None],
            0.98 * reference["forward"] + 0.02 * body_delta[:, :2, 0],
            reference["forward"],
        )
        reference["knee"] = torch.where(
            flat_straight[:, None],
            0.98 * reference["knee"] + 0.02 * knee[:, :2],
            reference["knee"],
        )

        front_forward = body_delta[:, :2, 0]
        front_extension = torch.linalg.vector_norm(body_delta[:, :2], dim=-1)
        front_knee_error = (knee[:, :2] - reference["knee"]).abs()
        descent_front_support = first_three_down[:, None] & load[:, :2]
        descent_front_swing = first_three_down[:, None] & ~load[:, :2]

        def smooth_excess(value):
            scaled = value.clamp_min(0.0)
            return 2.0 * (torch.sqrt(1.0 + scaled.square()) - 1.0)

        support_forward_excess = (
            front_forward - reference["forward"]
            - TUNING["descent_front_support_forward_margin"]
        ) / TUNING["descent_front_forward_scale"]
        swing_forward_excess = (
            front_forward - reference["forward"]
            - TUNING["descent_front_swing_forward_margin"]
        ) / TUNING["descent_front_forward_scale"]
        support_knee_excess = (
            front_knee_error - TUNING["descent_front_support_knee_margin"]
        ) / TUNING["descent_front_knee_scale"]
        swing_knee_excess = (
            front_knee_error - TUNING["descent_front_swing_knee_margin"]
        ) / TUNING["descent_front_knee_scale"]
        support_cost = torch.maximum(
            smooth_excess(support_forward_excess), smooth_excess(support_knee_excess)
        )
        swing_cost = torch.maximum(
            smooth_excess(swing_forward_excess), smooth_excess(swing_knee_excess)
        )
        descent_front_posture_cost = (
            support_cost * descent_front_support + swing_cost * descent_front_swing
        )
        _update_descent_front_posture_stats(
            env, level_index, descent_front_support, descent_front_swing,
            front_extension, front_forward, knee[:, :2].abs(), front_knee_error,
            descent_front_posture_cost, reference["forward"],
        )
    qy = yaw_quat(robot.data.root_quat_w)[:,None].expand(-1,4,-1).reshape(-1,4)
    gravity_delta = quat_apply_inverse(qy,delta.reshape(-1,3)).reshape(-1,4,3)
    body_angle = torch.atan2(body_delta[...,0],(-body_delta[...,2]).clamp_min(1e-6))
    gravity_angle = torch.atan2(gravity_delta[...,0],(-gravity_delta[...,2]).clamp_min(1e-6))
    rear_limit = relax[:,None]*TUNING["rear_forward_relax"]
    excess_b = (body_angle[:,2:]-TUNING["rear_b_limit"]-rear_limit)/TUNING["pose_angle_scale"]
    excess_g = (gravity_angle[:,2:]-TUNING["rear_g_limit"]-rear_limit)/TUNING["pose_angle_scale"]
    rear_load = load[:,2:] & down[:,None]
    rear_swing = ~load[:,2:] & down[:,None]
    support_gain = TUNING["rear_load_floor"] + (1-TUNING["rear_load_floor"])*load_gain[:,2:]
    rear_support_cost = descent_forward_excess(torch.maximum(excess_b,excess_g))*rear_load*support_gain
    swing_b = (body_angle[:,2:]-TUNING["rear_swing_b_limit"]-rear_limit)/TUNING["pose_angle_scale"]
    swing_g = (gravity_angle[:,2:]-TUNING["rear_swing_g_limit"]-rear_limit)/TUNING["pose_angle_scale"]
    rear_swing_cost = descent_forward_excess(torch.maximum(swing_b,swing_g))*rear_swing*TUNING["rear_swing_mix"]
    rear_forward = rear_support_cost + rear_swing_cost
    entry = _descent_entry_retract_context(env,scan,down,wheels,body_delta,ground,ground_valid,
        load,contact.data.current_contact_time[:,contact_sensor_cfg.body_ids],
        robot.data.body_lin_vel_w[:,asset_cfg.body_ids],robot.data.root_lin_vel_b,
        robot.data.root_quat_w,command,relax,env.scene[sensor_cfg.name].data.pos_w[:,:2])
    _update_rear_swing_stats(env,down,body_angle[:,2:],gravity_angle[:,2:],rear_swing,rear_swing_cost)
    if _detailed_teacher_diagnostics(env):
        _update_rear_angle_histograms(env,body_angle[:,2:],gravity_angle[:,2:],rear_load,rear_swing)
    # A gravity-aligned safety lower envelope is a height-map proxy, not an
    # exact collision mesh distance. It alone requires local map validity.
    points = wheels.new_tensor([[x,y,TUNING["body_bottom_z"]] for x in (0.10,0.22,0.34) for y in (-0.16,0,0.16)])
    count = points.shape[0]
    q = robot.data.root_quat_w[:,None].expand(-1,count,-1).reshape(-1,4)
    world = quat_apply(q,points[None].expand(env.num_envs,-1,-1).reshape(-1,3)).reshape(env.num_envs,count,3)+robot.data.root_pos_w[:,None]
    distances = torch.linalg.vector_norm(world[:,:,None,:2]-hits[:,None,:,:2],dim=-1)
    local = valid[:,None] & (distances < TUNING["nearest_valid_distance"])
    floor = torch.where(local,hits[:,None,:,2],-torch.inf).amax(-1)
    gap_valid = local.any(-1).any(-1) & up
    gap = torch.where(local.any(-1),world[...,2]-floor,torch.inf).amin(-1)
    body_gap = torch.where(gap_valid,bounded((TUNING["body_gap_target"]-gap)/TUNING["body_gap_scale"]),0)
    _update_pose_stats(env,up,down,front_a,front_support,front_swing,front_loaded_angle,front_air_angle,
                       body_angle[:,2:],gravity_angle[:,2:],rear_load,rear_forward,gap,gap_valid,load,ground_valid)
    tracking = turn_tracking_quality(robot.data.root_ang_vel_b[:, 2], command[:, 2])
    cached = dict(step=step, turn=turn, tracking=tracking, support=support, airborne=airborne, clearance=clearance,
                  ground_valid=ground_valid, extension=extension, knee=knee, reach=reach,
                  size=size, fold=fold, down=down, up=up,
                  front_fold=front_fold, front_support=front_support,
                  front_swing=front_swing, front_swing_fold=front_swing_fold,
                  rear_forward=rear_forward, body_angle=body_angle, gravity_angle=gravity_angle,
                  entry_retract_gain=entry["gain"], entry_range_cost=entry["cost"],
                  entry_front_lead_reward=entry["lead_reward"],
                  entry_front_progress_reward=entry["lead_progress_reward"],
                  entry_sequence_reward=entry["sequence_reward"],
                  entry_wheel_spacing_left_cost=entry["wheel_spacing_cost"][:, 0],
                  entry_wheel_spacing_right_cost=entry["wheel_spacing_cost"][:, 1],
                  descent_wheel_shank_cost=0.5 * (entry["shank_cost"].mean(-1) + entry["shank_cost"].amax(-1)),
                  descent_front_posture_cost=descent_front_posture_cost,
                  entry_rear_early_transfer_cost=entry["early_rear_transfer_cost"],
                  body_gap=body_gap, body_gap_m=gap, body_gap_valid=gap_valid, load=load, rear_load=rear_load)
    env._m20_gait_context = cached
    # Once per control step, independently of how many reward terms use it.
    detailed = _detailed_teacher_diagnostics(env)
    stats = getattr(env, "_m20_gait_stats", None)
    if stats is None:
        names = ("rear_support_samples", "rear_extension_sum", "rear_fold_samples", "down_steps")
        if detailed:
            names += ("turn_steps", "turn_clearance_sum", "turn_reach_sum", "turn_size_sum",
                      "turn_yaw_error_sum", "up_steps", "front_support_samples",
                      "front_extension_sum", "front_knee_abs_sum", "front_fold_samples",
                      "front_swing_samples", "front_swing_fold_samples")
        stats = {name: torch.zeros(env.num_envs, device=env.device) for name in names}
        env._m20_gait_stats = stats
    reset = env.episode_length_buf <= 1
    for value in stats.values():
        value[reset] = 0
    rear_samples = load[:, 2:] & down[:, None]
    stats["rear_support_samples"] += rear_samples.sum(dim=1)
    stats["rear_extension_sum"] += (extension[:, 2:] * rear_samples.float()).sum(dim=1)
    stats["rear_fold_samples"] += ((fold > 0) & rear_samples).sum(dim=1)
    stats["down_steps"] += down.float()
    if not detailed:
        return cached
    valid_turn = turn & ground_valid.all(dim=1)
    stats["turn_steps"] += valid_turn.float()
    stats["turn_clearance_sum"] += torch.where(valid_turn, clearance.clamp_min(0).mean(dim=1), 0)
    stats["turn_reach_sum"] += torch.where(valid_turn, reach.mean(dim=1), 0)
    stats["turn_size_sum"] += size.mean(dim=1) * turn.float()
    stats["turn_yaw_error_sum"] += (robot.data.root_ang_vel_b[:, 2] - command[:, 2]).abs() * valid_turn.float()
    stats["up_steps"] += up.float()
    stats["front_support_samples"] += front_support.sum(dim=1)
    stats["front_extension_sum"] += (extension[:, :2] * front_support.float()).sum(dim=1)
    stats["front_knee_abs_sum"] += (knee[:, :2].abs() * front_support.float()).sum(dim=1)
    stats["front_fold_samples"] += ((front_fold > 0) & front_support).sum(dim=1)
    stats["front_swing_samples"] += front_swing.sum(dim=1)
    stats["front_swing_fold_samples"] += ((front_swing_fold > 0) & front_swing).sum(dim=1)
    return cached


def turn_swing_size_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    c = _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)
    return c["size"].mean(dim=1) * c["turn"].float()


def descent_rear_fold_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    return _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)["fold"].mean(dim=1)


def ascent_front_fold_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    """Bounded cost for excessive front flexion with separate support/swing envelopes.

    Support and swing have height-adaptive soft envelopes. Genuine vertical
    load activates posture immediately, independently of nearest-map validity.
    """
    c = _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)
    return (c["front_fold"] + c["front_swing_fold"]).mean(dim=1)


def descent_front_posture_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    """Limit excess front reach during the first three descending transfers."""
    _stair_step_completion_events(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg,
                    contact_sensor_cfg)["descent_front_posture_cost"].mean(dim=1)


def compact_turn_air_time(env, command_name, sensor_cfg, asset_cfg, threshold,
                          stair_sensor_cfg, cmd_threshold=0.1, foot_height_threshold=0.05):
    """Bound airtime at touchdown; moving forward cannot activate this term.

    Keep the inherited call signature so saved config parameters still resolve.
    Height shaping is handled by turn_swing_size_cost using local ground.
    """
    from .rewards import get_gait_level_tensor
    sensor = env.scene.sensors[sensor_cfg.name]
    first = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    air = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    command = env.command_manager.get_command(command_name)
    score = bounded_air_score(air, TUNING["air_min"], threshold, TUNING["air_max"])
    yaw = env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2]
    tracking = turn_tracking_quality(yaw, command[:, 2])
    return (score * first).sum(dim=1) * pure_turn(command).float() * tracking * get_gait_level_tensor(env)


def compact_turn_status(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    """Only actual airborne diagonal pairs count, measured above local terrain."""
    from .rewards import get_gait_level_tensor
    c = _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)
    pattern = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for grounded, lifted in (((0, 3), (1, 2)), ((1, 2), (0, 3))):
        lo = c["clearance"][:, lifted]
        airborne = c["airborne"][:, lifted]
        pattern |= (c["support"][:, grounded].all(dim=1) & airborne.all(dim=1)
                    & c["ground_valid"][:, lifted].all(dim=1)
                    & (lo > 0.02).all(dim=1) & (lo < TUNING["turn_height_terrain"]).all(dim=1))
    return pattern.float() * c["turn"].float() * c["tracking"] * get_gait_level_tensor(env)


def _legacy_gait_quality_metrics(env, env_ids):
    stats = getattr(env, "_m20_gait_stats", None)
    if stats is None:
        return {"samples": torch.zeros((), device=env.device)}
    s = {key: value[env_ids].sum() for key, value in stats.items()}
    turns = s["turn_steps"].clamp_min(1)
    rear = s["rear_support_samples"].clamp_min(1)
    front = s["front_support_samples"].clamp_min(1)
    swing = s["front_swing_samples"].clamp_min(1)
    return {"turn_samples": s["turn_steps"], "descent_samples": s["down_steps"],
            "rear_support_samples": s["rear_support_samples"],
            "turn_clearance_mean_m": s["turn_clearance_sum"] / turns,
            "turn_reach_mean_m": s["turn_reach_sum"] / turns,
            "turn_yaw_error_mean": s["turn_yaw_error_sum"] / turns,
            "rear_extension_mean_m": s["rear_extension_sum"] / rear,
            "rear_fold_support_fraction": s["rear_fold_samples"] / rear,
            "ascent_samples": s["up_steps"], "front_support_samples": s["front_support_samples"],
            "front_extension_mean_m": s["front_extension_sum"] / front,
            "front_knee_abs_mean_rad": s["front_knee_abs_sum"] / front,
            "front_fold_support_fraction": s["front_fold_samples"] / front,
            "front_swing_samples": s["front_swing_samples"],
            "front_extreme_swing_fold_fraction": s["front_swing_fold_samples"] / swing}



def descent_forward_excess(value):
    """Monotonic forward-posture cost with useful severe-error sensitivity."""
    huber = 2*(torch.sqrt(1+value.clamp_min(0).square())-1)
    return huber/(1+huber/TUNING["descent_cost_soft_cap"])


def descent_entry_position_band(depth,relax):
    """Soft hip-relative X band, limited by both measured reach and leg tilt."""
    lower = torch.full_like(depth,TUNING["entry_x_lower"])
    upper = TUNING["entry_x_upper"] + relax[:,None]*TUNING["entry_x_high_rise_relax"]
    upper = torch.minimum(upper,depth.clamp_min(0)*torch.tan(
        depth.new_tensor(TUNING["entry_angle_upper"])+relax[:,None]*TUNING["rear_forward_relax"]))
    return lower,upper


def _same_side_interleg_distances(env, wheels):
    """Wheel-center and rear-wheel-to-front-shank distances (capsule proxy).

    The segment joins knee and axle origins; the 0.20 m threshold includes
    the wheel radius, shank thickness/offset and a margin. This is a soft
    conservative envelope, not a mesh contact detector.
    """
    robot = env.scene["robot"]
    ids = getattr(env, "_m20_front_knee_body_ids", None)
    if ids is None:
        ids, _ = robot.find_bodies(["fl_knee", "fr_knee"], preserve_order=True)
        if len(ids) != 2:
            raise ValueError("M20 interleg guard requires fl_knee and fr_knee bodies")
        env._m20_front_knee_body_ids = ids
    knee = robot.data.body_pos_w[:, ids]
    segment = wheels[:, :2] - knee
    rear = wheels[:, 2:]
    t = ((rear - knee) * segment).sum(-1) / segment.square().sum(-1).clamp_min(1.0e-6)
    nearest = knee + t.clamp(0.0, 1.0)[..., None] * segment
    return (torch.linalg.vector_norm(wheels[:, :2] - rear, dim=-1),
            torch.linalg.vector_norm(rear - nearest, dim=-1))


def _same_side_interleg_closing_speeds(env, wheels, wheel_velocity):
    """Positive speed means separation is shrinking, including front motion.

    A moving knee changes the nearest point on the shank segment. Using that
    point's interpolated velocity catches rear-wheel/front-calf approaches
    which wheel-center closing speed alone can miss.
    """
    robot = env.scene["robot"]
    knee_ids = env._m20_front_knee_body_ids
    knee = robot.data.body_pos_w[:, knee_ids]
    knee_velocity = robot.data.body_lin_vel_w[:, knee_ids]
    wheel_delta = wheels[:, :2] - wheels[:, 2:]
    pair_distance = torch.linalg.vector_norm(wheel_delta, dim=-1).clamp_min(1.0e-6)
    pair_closing = ((wheel_velocity[:, 2:] - wheel_velocity[:, :2]) * wheel_delta).sum(-1) / pair_distance
    segment = wheels[:, :2] - knee
    t = (((wheels[:, 2:] - knee) * segment).sum(-1)
         / segment.square().sum(-1).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    nearest = knee + t[..., None] * segment
    nearest_velocity = knee_velocity + t[..., None] * (wheel_velocity[:, :2] - knee_velocity)
    shank_delta = nearest - wheels[:, 2:]
    shank_distance = torch.linalg.vector_norm(shank_delta, dim=-1).clamp_min(1.0e-6)
    shank_closing = ((wheel_velocity[:, 2:] - nearest_velocity) * shank_delta).sum(-1) / shank_distance
    return pair_closing, shank_closing


def interleg_clearance_cost(distance, margin, critical_margin, scale):
    """Smooth warning followed by an uncapped quadratic near collision.

    The compact safe band pays zero. A short wheelbase cannot cheaply trade
    away the last centimetres of clearance after the warning has activated.
    """
    warning = (margin - distance).clamp_min(0.0) / scale
    critical = ((critical_margin - distance).clamp_min(0.0)
                / TUNING["interleg_critical_scale"])
    return (2.0 * (torch.sqrt(1.0 + warning.square()) - 1.0)
            + TUNING["interleg_critical_gain"] * critical.square())


def descent_transfer_risk(pair_distance, shank_distance, rear_departing, closing_speed=None,
                          shank_closing_speed=None):
    """Allow safe following; brake closing wheel/wheel and wheel/calf motion.

    ``rear_departing`` is the physical transfer/motion mask, never a prescribed
    front-tread number. Neither forward commands nor fixed support order are
    required, so a pause or reversal cannot disable a live clearance warning.
    """
    if closing_speed is None:
        closing_speed = torch.zeros_like(pair_distance)
    if shank_closing_speed is None:
        shank_closing_speed = closing_speed
    wheel_lookahead = closing_speed.clamp_min(0.0) * TUNING["entry_transfer_lookahead_s"]
    if TUNING.get("v1_motion_baseline", False):
        # v1 used wheel-pair closing speed for both predictive proxies.
        shank_closing_speed = closing_speed
    shank_lookahead = shank_closing_speed.clamp_min(0.0) * TUNING["entry_transfer_lookahead_s"]
    # The larger warning envelope is predictive, not a permanent demand for
    # a long wheelbase: only a closing rear wheel needs the extra clearance.
    wheel_limit = TUNING["entry_wheel_spacing_min"] + wheel_lookahead.clamp_max(
        TUNING["entry_transfer_wheel_guard"] - TUNING["entry_wheel_spacing_min"])
    shank_limit = TUNING["descent_wheel_shank_min"] + shank_lookahead.clamp_max(
        TUNING["entry_transfer_shank_guard"] - TUNING["descent_wheel_shank_min"])
    wheel_risk = interleg_clearance_cost(pair_distance, wheel_limit,
        TUNING["interleg_wheel_critical"], TUNING["entry_wheel_spacing_scale"])
    shank_risk = interleg_clearance_cost(shank_distance, shank_limit,
        TUNING["interleg_shank_critical"], TUNING["descent_wheel_shank_scale"])
    wheel_near = ((TUNING["entry_transfer_wheel_guard"] - pair_distance)
                  / (TUNING["entry_transfer_wheel_guard"] - TUNING["entry_wheel_spacing_min"])).clamp(0.0, 1.0)
    shank_near = ((TUNING["entry_transfer_shank_guard"] - shank_distance)
                  / (TUNING["entry_transfer_shank_guard"] - TUNING["descent_wheel_shank_min"])).clamp(0.0, 1.0)
    wheel_braking = wheel_near * (closing_speed.clamp_min(0.0) / TUNING["interleg_closing_speed_scale"]).square()
    shank_braking = shank_near * (shank_closing_speed.clamp_min(0.0) / TUNING["interleg_closing_speed_scale"]).square()
    risk = torch.maximum(wheel_risk, shank_risk)
    risk += TUNING["interleg_braking_mix"] * torch.maximum(wheel_braking, shank_braking)
    per_side = risk * rear_departing
    cost = 0.5 * (per_side.mean(-1) + per_side.amax(-1))
    return cost, ((risk > 0.0) & rear_departing).any(-1)


def _descent_entry_retract_context(env,scan,down,wheels,body_delta,ground,valid,loaded,
                                    contact_time,wheel_velocity,root_velocity,root_quat,command,relax,scanner_pos=None):
    """One bounded hip-relative adjustment per physical descending flight entry.

    World-ground contacts may stay planted while the trunk advances. No bonus
    is paid for reverse rolling, stopped/backward trunks or repeated poses.
    The first edge is latched, never switched to each successive stair riser.
    """
    n=env.num_envs;device=env.device
    state=getattr(env,"_m20_descent_entry_state",None)
    if state is None:
        zero=lambda *shape:torch.zeros(n,*shape,device=device)
        state=dict(settled=torch.zeros(n,2,dtype=torch.bool,device=device),active=zero().bool(),ready=torch.ones(n,dtype=torch.bool,device=device),
            top=zero(),xy=zero(2),heading=zero(2),best=zero(2),quiet=zero(),
            lead_xy=zero(2),lead_target_z=zero(),lead_valid=zero().bool(),
            lead_front_time=zero(2),lead_front_paid=torch.zeros(n,2,dtype=torch.bool,device=device),
            lead_progress_best=zero(2),lead_progress_initialized=zero().bool(),
            lead_progress_total=zero(2),lead_front_event_total=zero(2),
            lead_pair_paid=zero().bool(),lead_pair_event=zero().bool(),lead_phase_started=zero().bool(),lead_wait_time=zero(),
            lead_candidate_search_steps=zero(),
            lead_phase_not_started_steps=zero(),lead_scan_empty_steps=zero(),
            lead_no_matching_height_steps=zero(),lead_spacing_rejected_steps=zero(),
            lead_attempts=zero(),lead_successes=zero(),lead_phase_samples=zero(),early_rear_transfer_steps=zero(),
            count=torch.zeros(n,dtype=torch.long,device=device),history_xy=zero(64,2),history_heading=zero(64,2),history_top=zero(64),
            stats={k:zero(2) for k in ("samples","x_sum","in_band","too_forward","too_rearward","credit","blocked","transfer_samples","transfer_excess","transfer_swing_samples","transfer_cost_sum","transfer_compact")},
            safety_samples=zero(),safety_side_samples=zero(2),pair_distance_sum=zero(2),shank_distance_sum=zero(2),
            pair_distance_min=torch.full((n,2),torch.inf,device=device),
            shank_distance_min=torch.full((n,2),torch.inf,device=device),
            wheel_low_samples=zero(2),shank_low_samples=zero(2),
            wheel_critical_samples=zero(2),shank_critical_samples=zero(2),
            wheel_overlap_proxy_samples=zero(2),shank_overlap_proxy_samples=zero(2),
            closing_risk_samples=zero(),closing_risk_cost_sum=zero(),closing_motion_samples=zero(),
            unsafe_transfer_steps=zero(),rear_transfer_steps=zero(),started=zero())
        env._m20_descent_entry_state=state
    reset=env.episode_length_buf<=1
    for key,value in state.items():
        if key=="stats":
            for item in value.values():item[reset]=0
        else:value[reset]=True if key=="ready" else (torch.inf if key.endswith("_min") else 0)
    lower,upper=descent_entry_position_band(-body_delta[:,2:,2],relax)
    rear_x=body_delta[:,2:,0]
    error=(lower-rear_x).clamp_min(0)+(rear_x-upper).clamp_min(0)
    normalized=error/TUNING["entry_x_scale"]
    score=torch.exp(-0.5*normalized.square())
    direction=wheels.new_zeros(n,3);direction[:,0]=1
    heading=quat_apply(yaw_quat(root_quat),direction)[:,:2]
    # The scan and body share yaw, the actual raycaster position is supplied
    # through the canonical scan cache by _context before this call.
    if scanner_pos is None:
        scanner_pos=env.scene["height_scanner"].data.pos_w[:,:2]
    edge_xy=scanner_pos+heading*scan["edge_x"][:,None]
    seen=scan["down_gate"]>0
    edge_known=seen.clone()
    phase=getattr(env,"_m20_descent_pose_phase",None)
    if phase is not None:
        retained=~seen&phase["active"]
        edge_xy=torch.where(retained[:,None],phase["xy"],edge_xy)
        anchor_heading=torch.where(retained[:,None],phase["heading"],heading)
        edge_known|=retained
    else:
        anchor_heading=heading
    # No current or latched physical edge means no entrance identity: fail closed.
    rear_top=ground[:,2:].amax(-1)
    rear_same=valid[:,2:].all(-1)&((ground[:,2:]-rear_top[:,None]).abs()<TUNING["entry_ground_tolerance"]).all(-1)
    stable=loaded&(contact_time>=TUNING["entry_contact_time"])
    pre_edge=(scan["down_gate"]>0)&(scan["edge_x"]>0)&(scan["edge_x"]<TUNING["entry_start_distance"])
    front_lower=(valid[:,:2]&stable[:,:2]&(ground[:,:2]<rear_top[:,None]-TUNING["entry_ground_tolerance"])).any(-1)
    known=torch.arange(64,device=device)[None]<state["count"][:,None]
    same=(((state["history_xy"]-edge_xy[:,None])*state["history_heading"]).sum(-1).abs()<.20)&(
        (state["history_heading"]*anchor_heading[:,None]).sum(-1).abs()>.8)&(
        (state["history_top"]-rear_top[:,None]).abs()<TUNING["entry_ground_tolerance"])&known
    start=state["ready"]&~state["active"]&down&edge_known&rear_same&stable[:,2:].all(-1)&(
        stable.sum(-1)>=3)&(pre_edge|front_lower)&~same.any(-1)&(state["count"]<64)
    ids=torch.nonzero(start,as_tuple=False).squeeze(-1);slots=state["count"][ids]
    state["history_xy"][ids,slots]=edge_xy[ids];state["history_top"][ids,slots]=rear_top[ids]
    state["history_heading"][ids,slots]=anchor_heading[ids]
    state["count"][ids]+=1;state["started"]+=start
    state["top"][start]=rear_top[start];state["xy"][start]=edge_xy[start]
    state["heading"][start]=anchor_heading[start];state["best"][start]=score[start];state["settled"][start]=False
    state["lead_xy"][start]=0;state["lead_target_z"][start]=0;state["lead_valid"][start]=False
    state["lead_front_time"][start]=0;state["lead_front_paid"][start]=False
    state["lead_progress_best"][start]=0;state["lead_progress_initialized"][start]=False
    state["lead_pair_paid"][start]=False;state["lead_pair_event"][start]=False
    state["lead_phase_started"][start]=False;state["lead_wait_time"][start]=0
    state["lead_candidate_search_steps"][start]=0
    state["lead_phase_not_started_steps"][start]=0;state["lead_scan_empty_steps"][start]=0
    state["lead_no_matching_height_steps"][start]=0;state["lead_spacing_rejected_steps"][start]=0
    state["lead_phase_samples"][start]=0;state["early_rear_transfer_steps"][start]=0
    state["active"]|=start;state["ready"][start]=False
    # Each rear foot loses its entrance bonus permanently on leaving this top.
    on_top=valid[:,2:]&((ground[:,2:]-state["top"][:,None]).abs()<TUNING["entry_ground_tolerance"])
    gone=valid[:,2:]&(ground[:,2:]<state["top"][:,None]-TUNING["entry_ground_tolerance"])
    state["best"]=torch.where(gone,torch.ones_like(state["best"]),state["best"])
    near=torch.linalg.vector_norm(wheels[:,:,:2]-state["xy"][:,None],dim=-1).amin(-1)<TUNING["pose_phase_radius"]
    aligned=(heading*state["heading"]).sum(-1)>.5
    # A lower ray hit means the wheel has crossed the edge, not that it has
    # landed. Keep the physical entry phase through the airborne transfer and
    # across command pauses; reward gates never erase its one-shot ledger.
    lower_support = gone & stable[:,2:] & (
        (wheels[:,2:,2]-TUNING["wheel_radius"]-ground[:,2:]).abs()<TUNING["entry_ground_tolerance"])
    state["settled"] |= lower_support & state["active"][:,None]

    # The first descending edge is tracked by the step ledger. Find and latch
    # the following edge whose source tread is the first edge's destination.
    # The front axle gets credit for reaching that next tread while both rear
    # wheels remain stable on the original platform.
    descent=getattr(env,"_m20_stair_step_state",None)
    lead_front_event=torch.zeros(n,2,dtype=torch.bool,device=device)
    lead_progress_gain=torch.zeros(n,2,device=device)
    lead_sequence_event=torch.zeros(n,dtype=torch.bool,device=device)
    early_rear_transfer=torch.zeros(n,dtype=torch.bool,device=device)
    state["lead_pair_event"].zero_()
    if descent is not None:
        candidate_delta=scan["ascending_edges_upper_z"]-scan["ascending_edges_lower_z"]
        candidate_xy=scanner_pos[:,None,:]+heading[:,None,:]*scan["ascending_edges_x"][None,:,None]
        candidate_distance=((candidate_xy-descent["edge_xy"][:,None,:])*descent["heading"][:,None,:]).sum(-1)
        descending_edge=scan["edge_geometry"]&(candidate_delta < 0.0)
        matching_height=descending_edge&(
            (scan["ascending_edges_lower_z"]-descent["target_z"][:,None]).abs()
            < TUNING["riser_height_tolerance"])
        in_spacing=((candidate_distance >= TUNING["entry_lead_min_spacing"])
                    &(candidate_distance <= TUNING["entry_lead_max_spacing"]))
        next_edge=matching_height&in_spacing
        nearest=torch.where(next_edge,candidate_distance,torch.inf).argmin(dim=1)
        found=next_edge.any(dim=1)&state["active"]&descent["active"]&~descent["up"]
        entry_active=state["active"]&descent["active"]&~descent["up"]
        front_phase_ready=entry_active&descent["front_reached"].all(-1)
        searching=front_phase_ready&~state["lead_valid"]
        state["lead_candidate_search_steps"]+=searching
        state["lead_phase_not_started_steps"]+=entry_active&~descent["front_reached"].all(-1)
        state["lead_scan_empty_steps"]+=searching&~scan["edge_geometry"].any(-1)
        state["lead_no_matching_height_steps"]+=(searching&scan["edge_geometry"].any(-1)
                                                    &~matching_height.any(-1))
        state["lead_spacing_rejected_steps"]+=(searching&matching_height.any(-1)
                                                 &~(matching_height&in_spacing).any(-1))
        latch=found&~state["lead_valid"]
        state["lead_xy"][latch]=candidate_xy[latch,nearest[latch]]
        state["lead_target_z"][latch]=scan["ascending_edges_upper_z"][latch,nearest[latch]]
        state["lead_valid"]|=found
        state["lead_attempts"]+=latch

        state["lead_phase_started"]|=(state["active"]&descent["active"]&~descent["up"]&
            descent["front_reached"].all(-1))

        rear_on_top=(valid[:,2:]&((ground[:,2:]-state["top"][:,None]).abs()<TUNING["entry_ground_tolerance"]))
        rear_stable_top=rear_on_top&loaded[:,2:]&(contact_time[:,2:]>=TUNING["entry_contact_time"])
        lead_phase=(state["active"]&state["lead_phase_started"]&state["lead_valid"]&~state["lead_pair_paid"])
        lead_front_x=((wheels[:,:2,:2]-state["lead_xy"][:,None,:])*state["heading"][:,None,:]).sum(-1)
        desired_front_x=TUNING["tread_inset"]+TUNING["entry_lead_landing_inset"]
        front_radius=torch.where(torch.isfinite(descent["front_source_clearance"]),
            descent["front_source_clearance"],TUNING["wheel_radius"])
        lead_x_error=(lead_front_x-desired_front_x).abs()
        lead_z_error=(wheels[:,:2,2]-(state["lead_target_z"][:,None]+front_radius)).abs()
        lead_score=torch.exp(-0.5*((lead_x_error/TUNING["entry_lead_x_scale"]).square()+
            (lead_z_error/TUNING["entry_lead_z_scale"]).square()))
        # The optional historical touchdown objective still requires stable
        # rear support. Its reward weights are zero in the compact-posture
        # teacher; the independent transfer cost protects physical clearance.
        progress_active=lead_phase
        initialize_progress=progress_active&~state["lead_progress_initialized"]
        state["lead_progress_best"][initialize_progress]=lead_score[initialize_progress]
        state["lead_progress_initialized"]|=progress_active
        lead_progress_gain=(lead_score-state["lead_progress_best"]).clamp_min(0)*progress_active[:,None]
        state["lead_progress_best"]=torch.where(progress_active[:,None],
            torch.maximum(state["lead_progress_best"],lead_score),state["lead_progress_best"])
        lead_front_now=(lead_phase[:,None]&rear_stable_top.all(-1)[:,None]&valid[:,:2]&loaded[:,:2]&
            (contact_time[:,:2]>=TUNING["entry_contact_time"])&
            ((ground[:,:2]-state["lead_target_z"][:,None]).abs()<TUNING["tread_height_tolerance"])&
            (lead_front_x>TUNING["tread_inset"]))
        state["lead_front_time"]=torch.where(lead_front_now,state["lead_front_time"]+env.step_dt,
            torch.zeros_like(state["lead_front_time"]))
        lead_confirmed=state["lead_front_time"]>=TUNING["entry_contact_time"]
        lead_front_event=lead_confirmed&~state["lead_front_paid"]
        state["lead_front_paid"]|=lead_front_event
        state["lead_progress_total"]+=lead_progress_gain
        state["lead_front_event_total"]+=lead_front_event
        state["lead_pair_event"]=state["active"]&state["lead_front_paid"].all(-1)&rear_stable_top.all(-1)&~state["lead_pair_paid"]
        state["lead_pair_paid"]|=state["lead_pair_event"]
        state["lead_successes"]+=state["lead_pair_event"]
        lead_sequence_event=state["lead_pair_event"]
        lead_phase_before_pair=(state["active"]&state["lead_phase_started"]&
            state["lead_valid"]&~state["lead_pair_paid"])
        early_rear_transfer=lead_phase_before_pair&~rear_stable_top.all(-1)
        state["lead_phase_samples"]+=lead_phase_before_pair
        state["early_rear_transfer_steps"]+=early_rear_transfer
        waiting=state["active"]&state["lead_phase_started"]&state["settled"].all(-1)&~state["lead_pair_paid"]
        state["lead_wait_time"]+=waiting*env.step_dt
    lead_expired=state["lead_phase_started"]&(state["lead_wait_time"]>=TUNING["entry_lead_wait_timeout"])
    transfer_done=state["settled"].all(-1)&(~state["lead_phase_started"]|state["lead_pair_paid"]|lead_expired)
    second_tread_objective = any(TUNING[key] > 0 for key in (
        "entry_front_lead_weight", "entry_front_progress_weight", "entry_sequence_weight"))
    if not second_tread_objective:
        # With that objective disabled, two safely landed rear wheels complete
        # entry immediately; do not keep its old second-tread waiting timeout.
        transfer_done = state["settled"].all(-1)
    state["active"] &= near&aligned&~transfer_done
    # Only a new flat landing with no active descending transfer rearms the
    # detector. The physical-entry history also blocks retreat/reentry farming.
    flat=valid.all(-1)&stable.all(-1)&((ground.amax(-1)-ground.amin(-1))<TUNING["entry_ground_tolerance"])&~down
    state["quiet"]=torch.where(flat,state["quiet"]+env.step_dt,0)
    state["ready"]|=~state["active"]&(state["quiet"]>=TUNING["entry_rearm_time"])
    other=stable.sum(-1,keepdim=True)-stable[:,2:].long()
    support=state["active"][:,None]&down[:,None]&on_top&stable[:,2:]&(other>=2)
    forward=(command[:,0]>TUNING["translation_threshold"])&(root_velocity[:,0]>TUNING["entry_forward_speed"])
    forward_wheel=(wheel_velocity[:,2:,:2]*state["heading"][:,None]).sum(-1)>=-TUNING["entry_wheel_reverse_tolerance"]
    upright=1-2*(root_quat[:,1].square()+root_quat[:,2].square())>.5
    extension=torch.linalg.vector_norm(body_delta[:,2:],dim=-1)
    healthy_reach=(extension>=TUNING["rear_min_extension"])&(
        -body_delta[:,2:,2]>=TUNING["rear_min_extension"]-TUNING["tread_inset"]*2)
    allowed=support&forward[:,None]&forward_wheel&upright[:,None]&healthy_reach
    gain=(score-state["best"]).clamp_min(0)*allowed
    # Record even unsupported/blocked progress while still on the original top;
    # it cannot be banked and collected later simply by switching the gate on.
    observed=state["active"][:,None]&on_top&valid[:,2:]
    state["best"]=torch.where(observed,torch.maximum(state["best"],score),state["best"])
    # Continuous guard, including the airborne gap. Swing has a wider band
    # so reaching the lower tread remains possible; it cannot evade all cost
    # by briefly unloading a forward-leaning rear leg.
    depth=(-body_delta[:,2:,2]).clamp_min(0)
    swing_upper=torch.minimum(depth*torch.tan(depth.new_tensor(TUNING["rear_swing_b_limit"])+
        relax[:,None]*TUNING["rear_forward_relax"]),
        depth.new_full(depth.shape,TUNING["entry_swing_x_upper"])+
        relax[:,None]*TUNING["entry_x_high_rise_relax"])
    guard_lower=torch.where(loaded[:,2:],lower,TUNING["entry_swing_x_lower"])
    guard_upper=torch.where(loaded[:,2:],upper,swing_upper)
    guard_error=(guard_lower-rear_x).clamp_min(0)+(rear_x-guard_upper).clamp_min(0)
    guard=state["active"][:,None]&down[:,None]
    guard_gain=torch.where(loaded[:,2:],1.0,TUNING["rear_swing_mix"])
    swing_extension=TUNING["entry_swing_min_extension"]+relax[:,None]*(
        TUNING["entry_swing_min_extension_high"]-TUNING["entry_swing_min_extension"])
    compact_cost=descent_forward_excess((swing_extension-extension)/TUNING["pose_length_scale"])
    position_cost=descent_forward_excess(guard_error/TUNING["entry_x_scale"])
    # Rearward placement must not be achieved only by collapsing the swing leg.
    cost=torch.maximum(position_cost,compact_cost*(~loaded[:,2:]))*guard*guard_gain
    stats=state["stats"];stats["samples"]+=support;stats["x_sum"]+=rear_x*support
    stats["in_band"]+=support&(error<=1e-6);stats["too_forward"]+=support&(rear_x>upper)
    stats["too_rearward"]+=support&(rear_x<lower);stats["credit"]+=gain;stats["blocked"]+=support&~allowed
    stats["transfer_samples"]+=guard
    stats["transfer_excess"]+=guard&(guard_error>1e-6)
    stats["transfer_swing_samples"]+=guard&~loaded[:,2:]
    stats["transfer_cost_sum"]+=cost
    stats["transfer_compact"]+=guard&~loaded[:,2:]&(compact_cost>0)
    # Keep the three learning signals separate so TensorBoard can show whether
    # shaping, stable touchdown, or the complete front-before-rear sequence fires.
    lead_progress_reward=lead_progress_gain.sum(-1)
    lead_reward=0.5*lead_front_event.float().sum(-1)
    sequence_reward=2.0*lead_sequence_event.float()
    # Soft center-distance margin for each same-side front/rear wheel pair.
    # 0.28 m leaves 10 cm over the 0.18 m sum of nominal wheel radii.
    pair_distance, shank_distance = _same_side_interleg_distances(env, wheels)
    spacing_cost=interleg_clearance_cost(pair_distance,TUNING["entry_wheel_spacing_min"],
        TUNING["interleg_wheel_critical"],TUNING["entry_wheel_spacing_scale"])
    # Keep safety shaping throughout descent, including the same-tread phase
    # after the entry ledger has completed.
    safety_down = down.clone()
    if phase is not None:
        safety_down |= phase["active"]
    spacing_cost=spacing_cost*safety_down[:, None].float()
    shank_cost = interleg_clearance_cost(shank_distance,TUNING["descent_wheel_shank_min"],
        TUNING["interleg_shank_critical"],TUNING["descent_wheel_shank_scale"]) * safety_down[:, None]
    rear_departing = safety_down[:, None] & (~loaded[:, 2:] | ~on_top)
    closing_speed, shank_closing_speed = _same_side_interleg_closing_speeds(env,wheels,wheel_velocity)
    closing_motion = safety_down[:, None] & (
        torch.maximum(closing_speed,shank_closing_speed)>TUNING["interleg_closing_gate"])
    # A front wheel moving backwards toward a planted rear wheel is just as
    # hazardous as rear swing. Physical relative motion covers both cases.
    motion_guard = rear_departing if TUNING.get("v1_motion_baseline", False) else rear_departing | closing_motion
    unsafe_transfer_cost, unsafe_transfer = descent_transfer_risk(
        pair_distance, shank_distance, motion_guard, closing_speed, shank_closing_speed)
    _, unsafe_rear_transfer = descent_transfer_risk(
        pair_distance, shank_distance, rear_departing, closing_speed, shank_closing_speed)
    safety_side = safety_down[:,None] & torch.isfinite(pair_distance) & torch.isfinite(shank_distance)
    state["safety_samples"] += safety_down
    state["safety_side_samples"] += safety_side
    state["pair_distance_sum"] += torch.where(safety_side,pair_distance,0.0)
    state["shank_distance_sum"] += torch.where(safety_side,shank_distance,0.0)
    state["pair_distance_min"] = torch.minimum(state["pair_distance_min"],torch.where(safety_side,pair_distance,torch.inf))
    state["shank_distance_min"] = torch.minimum(state["shank_distance_min"],torch.where(safety_side,shank_distance,torch.inf))
    state["wheel_low_samples"] += safety_side & (pair_distance<TUNING["entry_wheel_spacing_min"])
    state["shank_low_samples"] += safety_side & (shank_distance<TUNING["descent_wheel_shank_min"])
    state["wheel_critical_samples"] += safety_side & (pair_distance<TUNING["interleg_wheel_critical"])
    state["shank_critical_samples"] += safety_side & (shank_distance<TUNING["interleg_shank_critical"])
    state["wheel_overlap_proxy_samples"] += safety_side & (pair_distance<2*TUNING["wheel_radius"])
    state["shank_overlap_proxy_samples"] += safety_side & (shank_distance<TUNING["wheel_radius"]+TUNING["interleg_shank_proxy_radius"])
    state["closing_motion_samples"] += motion_guard.any(-1)
    state["closing_risk_samples"] += unsafe_transfer
    state["closing_risk_cost_sum"] += unsafe_transfer_cost
    state["rear_transfer_steps"] += rear_departing.any(-1)
    state["unsafe_transfer_steps"] += unsafe_rear_transfer
    return dict(gain=gain.mean(-1),cost=cost.mean(-1),lead_reward=lead_reward,
        lead_progress_reward=lead_progress_reward,
        sequence_reward=sequence_reward, wheel_spacing_cost=spacing_cost, shank_cost=shank_cost,
        early_rear_transfer_cost=unsafe_transfer_cost)


def stair_descent_entry_retract(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg):
    """Once-only progress toward a supported hip-relative rear-wheel band."""
    gain = _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["entry_retract_gain"]
    return gain * _stair_forward_command_gate(env) / env.step_dt


def _descent_entry_guide_scale(env) -> float:
    """Linear training-iteration schedule for the optional front-lead shaping."""
    schedule = getattr(env, "_m20_descent_entry_guide_schedule", None)
    if not schedule:
        return 1.0
    steps_per_iteration = max(int(schedule["steps_per_iteration"]), 1)
    iteration = int(schedule.get("iteration_offset", 0)) + int(env.common_step_counter) // steps_per_iteration
    guided_iterations = max(int(schedule["guided_iterations"]), 0)
    decay_iterations = max(int(schedule["decay_iterations"]), 0)
    if iteration < guided_iterations:
        return 1.0
    if decay_iterations == 0:
        return 0.0
    return max(0.0, 1.0 - (iteration - guided_iterations) / decay_iterations)


def stair_descent_entry_position_cost(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg):
    """Continuous entry guard until both rear wheels actually settle below."""
    return _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["entry_range_cost"]


def stair_descent_entry_front_lead_reward(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg):
    """Reward each front wheel's stable touchdown on the second tread."""
    reward = _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["entry_front_lead_reward"]
    return reward * _descent_entry_guide_scale(env) * _stair_forward_command_gate(env) / env.step_dt


def stair_descent_entry_front_progress_reward(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg):
    """Dense, non-repeating potential gain as the front wheels approach tread two."""
    reward = _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["entry_front_progress_reward"]
    return reward * _descent_entry_guide_scale(env) * _stair_forward_command_gate(env) / env.step_dt


def stair_descent_entry_sequence_reward(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg):
    """One-time reward when both fronts stabilize on tread two with both rears on top."""
    reward = _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["entry_sequence_reward"]
    return reward * _descent_entry_guide_scale(env) * _stair_forward_command_gate(env) / env.step_dt


def stair_descent_entry_wheel_spacing_left_cost(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg):
    """Penalize left front/rear wheel centers entering the collision margin."""
    return _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["entry_wheel_spacing_left_cost"]


def stair_descent_entry_wheel_spacing_right_cost(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg):
    """Penalize right front/rear wheel centers entering the collision margin."""
    return _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["entry_wheel_spacing_right_cost"]


def stair_descent_entry_rear_early_transfer_cost(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg):
    """Retain transfer protection when wheel or shank separation is unsafe."""
    return _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["entry_rear_early_transfer_cost"]


def stair_descent_wheel_shank_cost(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg):
    return _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["descent_wheel_shank_cost"]


def descent_rear_forward_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    """Separate loaded/swing forward envelopes throughout the descent transfer."""
    return _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["rear_forward"].mean(-1)


def ascent_front_body_clearance_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    """Map-based front belly clearance; invalid cells cannot fabricate a cost."""
    return _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["body_gap"]


def _update_descent_front_posture_stats(env, level_index, support, swing,
                                        extension, forward, knee_abs, knee_error, cost, reference_forward):
    """Track front-leg reach and posture cost separately for the first three drops."""
    state = getattr(env, "_m20_descent_front_posture_stats", None)
    if state is None or state["support_samples"].shape[0] != env.num_envs:
        state = {}
        metrics = ("samples", "forward_sum", "cost_sum", "reference_sum",
                   "reach_excess_sum", "reach_excess_samples", "extension_sum",
                   "knee_abs_sum", "knee_error_sum")
        for phase in ("support", "swing"):
            for metric in metrics:
                state[f"{phase}_{metric}"] = torch.zeros(env.num_envs, 3, device=env.device)
        env._m20_descent_front_posture_stats = state
    reset = env.episode_length_buf <= 1
    for value in state.values():
        value[reset] = 0.0
    for level in range(3):
        level_mask = level_index == level
        for phase, mask in (("support", support), ("swing", swing)):
            selected = mask & level_mask[:, None]
            state[f"{phase}_samples"][:, level] += selected.sum(dim=1)
            state[f"{phase}_forward_sum"][:, level] += (forward * selected).sum(dim=1)
            state[f"{phase}_extension_sum"][:, level] += (extension * selected).sum(dim=1)
            state[f"{phase}_knee_abs_sum"][:, level] += (knee_abs * selected).sum(dim=1)
            state[f"{phase}_knee_error_sum"][:, level] += (knee_error * selected).sum(dim=1)
            state[f"{phase}_cost_sum"][:, level] += (cost * selected).sum(dim=1)
            margin = TUNING[f"descent_front_{phase}_forward_margin"]
            excess = (forward - reference_forward - margin).clamp_min(0.0)
            state[f"{phase}_reference_sum"][:, level] += (reference_forward * selected).sum(dim=1)
            state[f"{phase}_reach_excess_sum"][:, level] += (excess * selected).sum(dim=1)
            state[f"{phase}_reach_excess_samples"][:, level] += ((excess > 0.0) & selected).sum(dim=1)


def _update_pose_stats(env,up,down,front_a,front_load,front_air,loaded_cost,air_cost,
                       rear_b,rear_g,rear_load,rear_cost,gap,gap_valid,load,ground_valid):
    detailed = _detailed_teacher_diagnostics(env)
    names = ("rear_loaded_samples","rear_b_sum","rear_g_sum","rear_bad",
             "rear_entry_samples","rear_entry_b_sum","rear_entry_g_sum")
    if detailed:
        names += ("front_loaded_samples","front_loaded_a_sum","front_loaded_bad",
                  "front_swing_samples","front_swing_a_sum","front_swing_bad")
    state = getattr(env,"_m20_pose_stats",None)
    if state is None:
        state = {name:torch.zeros(env.num_envs,2,device=env.device) for name in names}
        if detailed:
            state.update({name:torch.zeros(env.num_envs,device=env.device) for name in
                          ("gap_samples","gap_sum","gap_bad","up_steps","down_steps","loaded_without_local_map")})
            state["front_loaded_a_min"] = torch.full((env.num_envs,2),torch.pi,device=env.device)
            state["front_swing_a_min"] = torch.full((env.num_envs,2),torch.pi,device=env.device)
        state["rear_b_max"] = torch.full((env.num_envs,2),-torch.pi,device=env.device)
        state["rear_g_max"] = torch.full((env.num_envs,2),-torch.pi,device=env.device)
        env._m20_pose_stats = state
    reset = env.episode_length_buf <= 1
    for key,value in state.items():
        value[reset] = torch.pi if key.endswith("_min") else (-torch.pi if key.endswith("_max") else 0)
    if detailed:
        state["up_steps"] += up; state["down_steps"] += down
        for phase,mask,cost in (("loaded",front_load,loaded_cost),("swing",front_air,air_cost)):
            state[f"front_{phase}_samples"] += mask
            state[f"front_{phase}_a_sum"] += front_a*mask
            state[f"front_{phase}_bad"] += (cost>0)&mask
            state[f"front_{phase}_a_min"] = torch.minimum(state[f"front_{phase}_a_min"],torch.where(mask,front_a,torch.pi))
    state["rear_loaded_samples"] += rear_load
    state["rear_b_sum"] += rear_b*rear_load; state["rear_g_sum"] += rear_g*rear_load
    state["rear_bad"] += (rear_cost>0)&rear_load
    state["rear_b_max"] = torch.maximum(state["rear_b_max"],torch.where(rear_load,rear_b,-torch.pi))
    state["rear_g_max"] = torch.maximum(state["rear_g_max"],torch.where(rear_load,rear_g,-torch.pi))
    descent = getattr(env,"_m20_stair_step_state",None)
    entry = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    if descent is not None:
        # Mixed support: at least one front wheel has transferred down but rear
        # support has not completed, rather than time since first seeing stairs.
        entry = descent["active"] & ~descent["up"] & descent["front_reached"].any(-1) & ~descent["rear_support_paid"].all(-1)
    mask = rear_load & entry[:,None]
    state["rear_entry_samples"] += mask
    state["rear_entry_b_sum"] += rear_b*mask; state["rear_entry_g_sum"] += rear_g*mask
    if detailed:
        state["gap_samples"] += gap_valid
        state["gap_sum"] += torch.where(gap_valid,gap,0)
        state["gap_bad"] += gap_valid&(gap<TUNING["body_gap_target"])
        state["loaded_without_local_map"] += (load&~ground_valid&(up|down)[:,None]).sum(-1)


def _detailed_gait_quality_metrics(env,env_ids):
    result = _legacy_gait_quality_metrics(env,env_ids)
    def weighted(name,total,count):
        # The training logger pools these sufficient statistics across reset
        # batches before writing a scalar. Empty third-drop batches therefore
        # cannot masquerade as perfect posture or dilute an observed mean.
        result[name] = total/count.clamp_min(1)
        result[f"__weighted_total__/{name}"] = total
        result[f"__weighted_count__/{name}"] = count

    def minimum(name,values,valid):
        count = valid.sum().float()
        value = torch.where(valid,values,torch.inf).amin() if values.numel() else count.new_zeros(())
        result[name] = torch.where(count>0,value,0.0)
        result[f"__minimum_value__/{name}"] = result[name]
        result[f"__minimum_count__/{name}"] = count

    state = getattr(env,"_m20_pose_stats",None)
    if state is None:
        return result
    sums = {key:value[env_ids].sum(0) for key,value in state.items()}
    def average(total,count):
        return torch.rad2deg(total.sum()/count.sum().clamp_min(1))
    for phase in ("loaded","swing"):
        count = sums[f"front_{phase}_samples"]
        result[f"front_{phase}_a_mean_deg"] = average(sums[f"front_{phase}_a_sum"],count)
        result[f"front_{phase}_a_low_fraction"] = sums[f"front_{phase}_bad"].sum()/count.sum().clamp_min(1)
        observed = state[f"front_{phase}_a_min"][env_ids]
        result[f"front_{phase}_a_min_deg"] = torch.rad2deg(observed.amin()) if observed.numel() and count.sum()>0 else count.new_zeros(())
        for leg,i in (("fl",0),("fr",1)):
            result[f"{leg}_{phase}_samples"] = count[i]
            result[f"{leg}_{phase}_a_mean_deg"] = torch.rad2deg(sums[f"front_{phase}_a_sum"][i]/count[i].clamp_min(1))
            result[f"{leg}_{phase}_a_low_fraction"] = sums[f"front_{phase}_bad"][i]/count[i].clamp_min(1)
    rear = sums["rear_loaded_samples"]
    result["rear_forward_samples"] = rear.sum()
    result["rear_forward_excess_fraction"] = sums["rear_bad"].sum()/rear.sum().clamp_min(1)
    result["rear_body_b_mean_deg"] = average(sums["rear_b_sum"],rear)
    result["rear_gravity_g_mean_deg"] = average(sums["rear_g_sum"],rear)
    for coord in ("b","g"):
        peak = state[f"rear_{coord}_max"][env_ids]
        result[f"rear_{coord}_max_deg"] = torch.rad2deg(peak.amax()) if peak.numel() and rear.sum()>0 else rear.new_zeros(())
    for leg,i in (("hl",0),("hr",1)):
        result[f"{leg}_loaded_samples"] = rear[i]
        result[f"{leg}_body_b_mean_deg"] = torch.rad2deg(sums["rear_b_sum"][i]/rear[i].clamp_min(1))
        result[f"{leg}_gravity_g_mean_deg"] = torch.rad2deg(sums["rear_g_sum"][i]/rear[i].clamp_min(1))
        result[f"{leg}_forward_excess_fraction"] = sums["rear_bad"][i]/rear[i].clamp_min(1)
    entry = sums["rear_entry_samples"]
    result["rear_entry_samples"] = entry.sum()
    result["rear_entry_b_mean_deg"] = average(sums["rear_entry_b_sum"],entry)
    result["rear_entry_g_mean_deg"] = average(sums["rear_entry_g_sum"],entry)
    result["front_body_gap_samples"] = sums["gap_samples"]
    result["front_body_gap_mean_m"] = sums["gap_sum"]/sums["gap_samples"].clamp_min(1)
    result["front_body_gap_low_fraction"] = sums["gap_bad"]/sums["gap_samples"].clamp_min(1)
    result["loaded_without_local_map_samples"] = sums["loaded_without_local_map"]
    descent_pose = getattr(env, "_m20_descent_front_posture_stats", None)
    if descent_pose is not None:
        descent_sums = {key: value[env_ids].sum(0) for key, value in descent_pose.items()}
        for level in range(3):
            for phase in ("support", "swing"):
                count = descent_sums[f"{phase}_samples"][level]
                prefix = f"descent_step{level + 1}_front_{phase}"
                result[f"{prefix}_samples"] = count
                result[f"{prefix}_valid"] = (count>0).float()
                result[f"{prefix}_observed_episodes"] = (
                    descent_pose[f"{phase}_samples"][env_ids,level]>0).sum().float()
                for name,numerator in (
                    ("hip_wheel_length_mean_m","extension_sum"),
                    ("forward_reach_mean_m","forward_sum"),
                    ("knee_abs_mean_rad","knee_abs_sum"),
                    ("knee_deviation_mean_rad","knee_error_sum"),
                    ("posture_cost_mean","cost_sum"),
                    ("reference_forward_mean_m","reference_sum"),
                    ("reach_excess_mean_m","reach_excess_sum"),
                    ("reach_excess_fraction","reach_excess_samples"),
                ):
                    weighted(f"{prefix}_{name}",descent_sums[f"{phase}_{numerator}"][level],count)
    swing = getattr(env,"_m20_rear_swing_stats",None)
    if swing is not None:
        ss = {k:v[env_ids].sum(0) for k,v in swing.items()}
        count = ss["samples"].sum().clamp_min(1)
        result["rear_swing_samples"] = ss["samples"].sum()
        result["rear_swing_excess_fraction"] = ss["bad"].sum()/count
        for coord in ("b","g"):
            result[f"rear_swing_{coord}_positive_mean_deg"] = torch.rad2deg(ss[f"{coord}_sum"].sum()/count)
            result[f"rear_swing_{coord}_max_deg"] = torch.rad2deg(swing[f"{coord}_max"][env_ids].amax())
        for i,leg in enumerate(("hl","hr")):
            result[f"{leg}_swing_excess_fraction"] = ss["bad"][i]/ss["samples"][i].clamp_min(1)
    hist = getattr(env,"_m20_rear_angle_histograms",None)
    if hist is not None:
        for key,value in hist.items():
            counts = value[env_ids].sum(0)
            for i,leg in enumerate(("hl","hr")):
                total = counts[i].sum()
                # Upper edge of a five-degree bin; negative angles enter zero bin.
                q = ((counts[i].cumsum(-1) < .95*total).sum()+1).clamp_max(18)*5
                result[f"{leg}_{key}_p95_upper_deg"] = torch.where(total>0,q,0).float()
    entry=getattr(env,"_m20_descent_entry_state",None)
    if entry is not None:
        result["descent_entry_started"]=entry["started"][env_ids].sum()
        lead_attempts=entry["lead_attempts"][env_ids].sum()
        lead_successes=entry["lead_successes"][env_ids].sum()
        lead_phase=entry["lead_phase_samples"][env_ids].sum().clamp_min(1)
        result["descent_entry_front_lead_attempts"]=lead_attempts
        result["descent_entry_front_lead_successes"]=lead_successes
        result["descent_entry_front_lead_success_rate"]=lead_successes/lead_attempts.clamp_min(1)
        # Legacy style metric: leaving the top before both fronts settle on
        # tread two is not itself unsafe, and that style objective is disabled.
        legacy_early = entry["early_rear_transfer_steps"][env_ids].sum()/lead_phase
        result["descent_entry_rear_early_transfer_fraction"]=legacy_early
        result["descent_legacy_front_second_tread_early_follow_fraction"]=legacy_early
        result["descent_legacy_front_second_tread_objective_enabled"]=float(any(
            TUNING[key]>0 for key in ("entry_front_lead_weight","entry_front_progress_weight","entry_sequence_weight")))
        transfers = entry["rear_transfer_steps"][env_ids].sum()
        result["descent_rear_transfer_samples"] = transfers
        result["descent_rear_transfer_valid"] = (transfers>0).float()
        weighted("descent_unsafe_rear_transfer_fraction",entry["unsafe_transfer_steps"][env_ids].sum(),transfers)
        motion_count = entry["closing_motion_samples"][env_ids].sum()
        result["descent_transfer_or_closing_samples"] = motion_count
        weighted("descent_transfer_or_closing_risk_fraction",entry["closing_risk_samples"][env_ids].sum(),motion_count)
        weighted("descent_transfer_or_closing_cost_mean",entry["closing_risk_cost_sum"][env_ids].sum(),motion_count)
        result["descent_safety_samples"] = entry["safety_samples"][env_ids].sum()
        for side, i in (("left", 0), ("right", 1)):
            safety_count = entry["safety_side_samples"][env_ids,i].sum()
            observed = entry["safety_side_samples"][env_ids,i]>0
            result[f"descent_{side}_safety_samples"] = safety_count
            result[f"descent_{side}_safety_valid"] = (safety_count>0).float()
            result[f"descent_{side}_safety_observed_episodes"] = observed.sum().float()
            for name,numerator in (
                ("wheel_center_distance_mean_m","pair_distance_sum"),
                ("rear_wheel_front_shank_distance_mean_m","shank_distance_sum"),
                ("wheel_margin_violation_fraction","wheel_low_samples"),
                ("shank_margin_violation_fraction","shank_low_samples"),
                ("wheel_critical_clearance_fraction","wheel_critical_samples"),
                ("shank_critical_clearance_fraction","shank_critical_samples"),
                ("wheel_overlap_proxy_fraction","wheel_overlap_proxy_samples"),
                ("shank_overlap_proxy_fraction","shank_overlap_proxy_samples"),
            ):
                weighted(f"descent_{side}_{name}",entry[numerator][env_ids,i].sum(),safety_count)
            for name,buffer in (("wheel_center_distance","pair_distance_min"),
                                ("rear_wheel_front_shank_distance","shank_distance_min")):
                values = entry[buffer][env_ids,i]
                minimum(f"descent_{side}_{name}_min_m",values,observed)
                weighted(f"descent_{side}_{name}_episode_min_mean_m",
                    torch.where(observed,values,0.0).sum(),observed.sum().float())
        candidate_search=entry["lead_candidate_search_steps"][env_ids].sum().clamp_min(1)
        result["descent_entry_lead_phase_not_started_steps"]=entry["lead_phase_not_started_steps"][env_ids].sum()
        result["descent_entry_lead_scan_empty_steps"]=entry["lead_scan_empty_steps"][env_ids].sum()
        result["descent_entry_lead_no_matching_height_steps"]=entry["lead_no_matching_height_steps"][env_ids].sum()
        result["descent_entry_lead_spacing_rejected_steps"]=entry["lead_spacing_rejected_steps"][env_ids].sum()
        result["descent_entry_lead_scan_empty_fraction"]=entry["lead_scan_empty_steps"][env_ids].sum()/candidate_search
        result["descent_entry_lead_no_matching_height_fraction"]=entry["lead_no_matching_height_steps"][env_ids].sum()/candidate_search
        result["descent_entry_lead_spacing_rejected_fraction"]=entry["lead_spacing_rejected_steps"][env_ids].sum()/candidate_search
        progress=entry["lead_progress_total"][env_ids].sum(0)
        front_events=entry["lead_front_event_total"][env_ids].sum(0)
        for i,leg in enumerate(("fl","fr")):
            result[f"{leg}_entry_lead_progress_gain"]=progress[i]
            result[f"{leg}_entry_lead_stable_events"]=front_events[i]
        for i,leg in enumerate(("hl","hr")):
            ss={k:v[env_ids,i].sum() for k,v in entry["stats"].items()}
            count=ss["samples"].clamp_min(1)
            result[f"{leg}_entry_samples"]=ss["samples"]
            result[f"{leg}_entry_x_mean_m"]=ss["x_sum"]/count
            result[f"{leg}_entry_in_band_fraction"]=ss["in_band"]/count
            result[f"{leg}_entry_too_forward_fraction"]=ss["too_forward"]/count
            result[f"{leg}_entry_too_rearward_fraction"]=ss["too_rearward"]/count
            result[f"{leg}_entry_retract_credit"]=ss["credit"]
            result[f"{leg}_entry_blocked_fraction"]=ss["blocked"]/count
            transfer=ss["transfer_samples"].clamp_min(1)
            result[f"{leg}_entry_transfer_samples"]=ss["transfer_samples"]
            result[f"{leg}_entry_transfer_excess_fraction"]=ss["transfer_excess"]/transfer
            result[f"{leg}_entry_transfer_swing_fraction"]=ss["transfer_swing_samples"]/transfer
            result[f"{leg}_entry_transfer_cost_mean"]=ss["transfer_cost_sum"]/transfer
            result[f"{leg}_entry_transfer_compact_fraction"]=ss["transfer_compact"]/ss["transfer_swing_samples"].clamp_min(1)
    return result


def _detailed_stair_step_completion_metrics(env,env_ids):
    result = _legacy_step_completion_metrics(env,env_ids)
    state = getattr(env,"_m20_ascent_state",None)
    if state is None:
        return result
    ids = torch.arange(env.num_envs,device=env.device)[env_ids]
    known = torch.arange(64,device=env.device)[None] < state["map_count"][ids,None]
    intermediate = state["map_intermediate"][ids]&known
    result["registered_risers"] = state["map_count"][ids].float().mean()
    result["up_whole_body_crossings"] = state["task_crossings_total"][ids].mean()
    result["up_whole_body_crossing_coverage"] = (
        (state["map_task_paid"][ids] & known).sum().float() / known.sum().clamp_min(1))
    for axle,name in enumerate(("front","rear")):
        paid = state["map_paid"][ids,:,axle]&known
        strict = state["map_correct"][ids,:,axle]&~state["map_duplicate_paid"][ids,:,axle]&intermediate
        result[f"up_{name}_registered_intermediate_risers"] = intermediate.sum().float()
        result[f"up_{name}_landing_coverage"] = paid.sum().float()/known.sum().clamp_min(1)
        result[f"up_{name}_strict_coverage"] = strict.sum().float()/intermediate.sum().clamp_min(1)
        result[f"up_{name}_physical_landings"] = state["physical_total"][ids,axle].float().mean()
        result[f"up_{name}_transition_successes"] = state[name+"_transition_successes"][ids].mean()
        result[f"up_{name}_transition_per_landing"] = state[name+"_transition_successes"][ids].sum()/state["physical_total"][ids,axle].sum().clamp_min(1)
        result[f"up_{name}_skipped_riser_events"] = state["skip_total"][ids,axle].float().mean()
        result[f"up_{name}_unsafe_events"] = state["unsafe_total"][ids,axle].float().mean()
        result[f"up_{name}_skip_fraction"] = (
            state["skip_total"][ids,axle].sum()/state["physical_total"][ids,axle].sum().clamp_min(1))
        visited = paid & intermediate
        result[f"up_{name}_visited_intermediate_risers"] = visited.sum().float()
        result[f"up_{name}_same_tread_fraction"] = (
            (state["map_duplicate_paid"][ids,:,axle]&visited).sum().float()/visited.sum().clamp_min(1))
        # Longest surviving strict run. A later duplicate invalidates an earlier
        # tread; future detected-but-unlanded treads contribute no success.
        order = (state["map_flight"][ids].float()*1e6+state["map_z"][ids]).argsort(dim=-1,stable=True)
        good = strict.gather(1,order)
        flight = state["map_flight"][ids].gather(1,order)
        group_start = torch.ones_like(good); group_start[:,1:] = flight[:,1:]!=flight[:,:-1]
        total = good.long().cumsum(-1)
        restart = torch.where(~good,total,torch.where(group_start,total-good.long(),0)).cummax(-1).values
        runs = (total-restart).amax(-1)
        result[f"up_{name}_longest_strict_run"] = runs.float().mean()
        result[f"up_{name}_six_step_strict_fraction"] = (runs>=6).float().mean()
        for side,leg in enumerate((("fl","fr"),("hl","hr"))[axle]):
            result[f"{leg}_stable_treads"] = (state["map_landed"][ids,:,axle*2+side]&known).sum(-1).float().mean()
    result["up_rear_lift_credit"] = state["rear_lift_credit"][ids].mean()
    result["up_rear_prep_blocked_gain"] = state["rear_prep_blocked_gain"][ids].mean()
    result["up_rear_recovery_events"] = state["rear_recoveries"][ids].mean()
    result["up_rear_style_refund"] = state["rear_style_refund_total"][ids].mean()
    result["up_rear_recovery_per_landing"] = state["rear_recoveries"][ids].sum()/state["physical_total"][ids,1].sum().clamp_min(1)
    return result


def _descent_gait_metrics(env, env_ids):
    """Keep descent sequence, loaded posture and rear swing visible by default."""
    result = {}

    def weighted(name, total, count):
        result[name] = total / count.clamp_min(1)
        result[f"__weighted_total__/{name}"] = total
        result[f"__weighted_count__/{name}"] = count

    gait = getattr(env, "_m20_gait_stats", None)
    if gait is not None:
        s = {key: gait[key][env_ids].sum() for key in (
            "down_steps", "rear_support_samples", "rear_extension_sum", "rear_fold_samples")}
        result["descent_samples"] = s["down_steps"]
        result["rear_support_samples"] = s["rear_support_samples"]
        weighted("rear_extension_mean_m", s["rear_extension_sum"], s["rear_support_samples"])
        weighted("rear_fold_support_fraction", s["rear_fold_samples"], s["rear_support_samples"])

    pose = getattr(env, "_m20_pose_stats", None)
    if pose is not None:
        samples = pose["rear_loaded_samples"][env_ids].sum(0)
        count = samples.sum()
        result["rear_forward_samples"] = count
        weighted("rear_forward_excess_fraction", pose["rear_bad"][env_ids].sum(), count)
        for coord, label in (("b", "body_b"), ("g", "gravity_g")):
            angles = torch.rad2deg(pose[f"rear_{coord}_sum"][env_ids].sum(0))
            weighted(f"rear_{label}_mean_deg", angles.sum(), count)
            peaks = pose[f"rear_{coord}_max"][env_ids]
            peak = peaks.amax() if peaks.numel() else count.new_zeros(())
            result[f"rear_{coord}_max_deg"] = torch.rad2deg(torch.where(count > 0, peak, 0.0))
            for i, leg in enumerate(("hl", "hr")):
                weighted(f"{leg}_{label}_mean_deg", angles[i], samples[i])
        for i, leg in enumerate(("hl", "hr")):
            result[f"{leg}_loaded_samples"] = samples[i]
            weighted(f"{leg}_forward_excess_fraction", pose["rear_bad"][env_ids, i].sum(), samples[i])
        entry_samples = pose["rear_entry_samples"][env_ids].sum()
        result["rear_entry_samples"] = entry_samples
        for coord in ("b", "g"):
            weighted(f"rear_entry_{coord}_mean_deg",
                     torch.rad2deg(pose[f"rear_entry_{coord}_sum"][env_ids].sum()), entry_samples)

    swing = getattr(env, "_m20_rear_swing_stats", None)
    if swing is not None:
        samples = swing["samples"][env_ids].sum(0)
        count = samples.sum()
        result["rear_swing_samples"] = count
        weighted("rear_swing_excess_fraction", swing["bad"][env_ids].sum(), count)
        for coord in ("b", "g"):
            weighted(f"rear_swing_{coord}_positive_mean_deg",
                     torch.rad2deg(swing[f"{coord}_sum"][env_ids].sum()), count)
            peaks = swing[f"{coord}_max"][env_ids]
            result[f"rear_swing_{coord}_max_deg"] = (
                torch.rad2deg(peaks.amax()) if peaks.numel() else count.new_zeros(()))
        for i, leg in enumerate(("hl", "hr")):
            weighted(f"{leg}_swing_excess_fraction", swing["bad"][env_ids, i].sum(), samples[i])

    entry = getattr(env, "_m20_descent_entry_state", None)
    if entry is not None:
        result["descent_entry_started"] = entry["started"][env_ids].sum()
        attempts = entry["lead_attempts"][env_ids].sum()
        successes = entry["lead_successes"][env_ids].sum()
        phase = entry["lead_phase_samples"][env_ids].sum()
        result["descent_entry_front_lead_attempts"] = attempts
        result["descent_entry_front_lead_successes"] = successes
        result["descent_entry_front_lead_phase_samples"] = phase
        weighted("descent_entry_front_lead_success_rate", successes, attempts)
        # This is a sequence/style measurement, independently of safety.
        early = entry["early_rear_transfer_steps"][env_ids].sum()
        weighted("descent_entry_rear_early_transfer_fraction", early, phase)
        weighted("descent_legacy_front_second_tread_early_follow_fraction", early, phase)
        result["descent_legacy_front_second_tread_objective_enabled"] = float(any(
            TUNING[key] > 0 for key in ("entry_front_lead_weight", "entry_front_progress_weight", "entry_sequence_weight")))
        for i, leg in enumerate(("fl", "fr")):
            result[f"{leg}_entry_lead_progress_gain"] = entry["lead_progress_total"][env_ids, i].sum()
            result[f"{leg}_entry_lead_stable_events"] = entry["lead_front_event_total"][env_ids, i].sum()
        for i, leg in enumerate(("hl", "hr")):
            s = {key: value[env_ids, i].sum() for key, value in entry["stats"].items()}
            result[f"{leg}_entry_samples"] = s["samples"]
            result[f"{leg}_entry_retract_credit"] = s["credit"]
            result[f"{leg}_entry_transfer_samples"] = s["transfer_samples"]
            for name, field in (("x_mean_m", "x_sum"), ("in_band_fraction", "in_band"),
                                ("too_forward_fraction", "too_forward"), ("too_rearward_fraction", "too_rearward"),
                                ("blocked_fraction", "blocked")):
                weighted(f"{leg}_entry_{name}", s[field], s["samples"])
            for name, field in (("excess_fraction", "transfer_excess"),
                                ("swing_fraction", "transfer_swing_samples"),
                                ("cost_mean", "transfer_cost_sum")):
                weighted(f"{leg}_entry_transfer_{name}", s[field], s["transfer_samples"])
            weighted(f"{leg}_entry_transfer_compact_fraction", s["transfer_compact"], s["transfer_swing_samples"])
    return result


def _core_gait_quality_metrics(env, env_ids):
    """Reduce descent gait, posture and collision-risk statistics.

    Detailed ascent/turn angles and histograms remain optional. Descent
    sequence and rear support/swing statistics are always available.
    """
    result = {}

    def weighted(name, total, count):
        result[name] = total / count.clamp_min(1)
        result[f"__weighted_total__/{name}"] = total
        result[f"__weighted_count__/{name}"] = count

    posture = getattr(env, "_m20_descent_front_posture_stats", None)
    if posture is not None:
        for phase in ("support", "swing"):
            fields = ("samples", "forward_sum", "reference_sum", "reach_excess_sum",
                      "reach_excess_samples", "cost_sum", "extension_sum", "knee_abs_sum", "knee_error_sum")
            sums = {key: posture[f"{phase}_{key}"][env_ids].sum(0) for key in fields}
            episodes = (posture[f"{phase}_samples"][env_ids] > 0).sum(0).float()
            for level in range(3):
                prefix = f"descent_step{level + 1}_front_{phase}"
                count = sums["samples"][level]
                result[f"{prefix}_samples"] = count
                result[f"{prefix}_valid"] = (count > 0).float()
                result[f"{prefix}_observed_episodes"] = episodes[level]
                for name, field in (
                    ("forward_reach_mean_m", "forward_sum"),
                    ("reference_forward_mean_m", "reference_sum"),
                    ("reach_excess_mean_m", "reach_excess_sum"),
                    ("reach_excess_fraction", "reach_excess_samples"),
                    ("posture_cost_mean", "cost_sum"),
                    ("hip_wheel_length_mean_m", "extension_sum"),
                    ("knee_abs_mean_rad", "knee_abs_sum"),
                    ("knee_deviation_mean_rad", "knee_error_sum"),
                ):
                    weighted(f"{prefix}_{name}", sums[field][level], count)

    entry = getattr(env, "_m20_descent_entry_state", None)
    if entry is not None:
        transfers = entry["rear_transfer_steps"][env_ids].sum()
        result["descent_rear_transfer_samples"] = transfers
        weighted("descent_unsafe_rear_transfer_fraction",
                 entry["unsafe_transfer_steps"][env_ids].sum(), transfers)
        closing = entry["closing_motion_samples"][env_ids].sum()
        result["descent_transfer_or_closing_samples"] = closing
        weighted("descent_transfer_or_closing_risk_fraction",
                 entry["closing_risk_samples"][env_ids].sum(), closing)
        weighted("descent_transfer_or_closing_cost_mean",
                 entry["closing_risk_cost_sum"][env_ids].sum(), closing)
        for side, i in (("left", 0), ("right", 1)):
            samples = entry["safety_side_samples"][env_ids, i]
            count = samples.sum()
            observed = samples > 0
            result[f"descent_{side}_safety_samples"] = count
            for name, field in (
                ("wheel_center_distance_mean_m", "pair_distance_sum"),
                ("rear_wheel_front_shank_distance_mean_m", "shank_distance_sum"),
                ("wheel_margin_violation_fraction", "wheel_low_samples"),
                ("shank_margin_violation_fraction", "shank_low_samples"),
                ("wheel_critical_clearance_fraction", "wheel_critical_samples"),
                ("shank_critical_clearance_fraction", "shank_critical_samples"),
                ("wheel_overlap_proxy_fraction", "wheel_overlap_proxy_samples"),
                ("shank_overlap_proxy_fraction", "shank_overlap_proxy_samples"),
            ):
                weighted(f"descent_{side}_{name}", entry[field][env_ids, i].sum(), count)
            for name, field in (("wheel_center_distance", "pair_distance_min"),
                                ("rear_wheel_front_shank_distance", "shank_distance_min")):
                values = entry[field][env_ids, i]
                observations = observed.sum().float()
                minimum = (torch.where(observed, values, torch.inf).amin()
                           if values.numel() else observations.new_zeros(()))
                key = f"descent_{side}_{name}_min_m"
                result[key] = torch.where(observations > 0, minimum, 0.0)
                result[f"__minimum_value__/{key}"] = result[key]
                result[f"__minimum_count__/{key}"] = observations
                weighted(f"descent_{side}_{name}_episode_min_mean_m",
                         torch.where(observed, values, 0.0).sum(), observations)
    result.update(_descent_gait_metrics(env, env_ids))
    return result


def gait_quality_metrics(env, env_ids):
    detailed = _detailed_teacher_diagnostics(env)
    result = (_detailed_gait_quality_metrics(env, env_ids) if detailed
              else _core_gait_quality_metrics(env, env_ids))
    result.update(descent_front_pair_metrics(env, env_ids))
    return export_teacher_metrics(result, result.keys())


def stair_step_completion_metrics(env, env_ids):
    if _detailed_teacher_diagnostics(env):
        result = _detailed_stair_step_completion_metrics(env, env_ids)
    else:
        # Keep traversal progress without sorting a 64-riser ledger for old
        # strict-sequence/longest-run diagnostics at every environment reset.
        result = {}
        up = getattr(env, "_m20_ascent_state", None)
        if up is not None:
            attempts = up["front_count"][env_ids].float()
            result["up_attempts"] = attempts.mean()
            result["up_success_rate"] = up["front_successes"][env_ids].sum() / attempts.sum().clamp_min(1)
            result["up_whole_body_crossings"] = up["task_crossings_total"][env_ids].mean()
            known = torch.arange(64, device=env.device)[None] < up["map_count"][env_ids, None]
            result["up_whole_body_crossing_coverage"] = (
                (up["map_task_paid"][env_ids] & known).sum().float() / known.sum().clamp_min(1)
            )
            for axle, name in enumerate(("front", "rear")):
                result[f"up_{name}_unsafe_events"] = up["unsafe_total"][env_ids, axle].float().mean()
                result[f"up_{name}_landing_coverage"] = (
                    (up["map_paid"][env_ids, :, axle] & known).sum().float() / known.sum().clamp_min(1)
                )
        down = getattr(env, "_m20_stair_step_state", None)
        if down is not None:
            attempts = down["down_attempts"][env_ids]
            successes = down["down_successes"][env_ids]
            result["down_attempts"] = attempts.mean()
            result["down_successes"] = successes.mean()
            result["down_success_rate"] = successes.sum() / attempts.sum().clamp_min(1)
    return export_teacher_metrics(result, result.keys())


# =============================================================================
# 7. Terrain curriculum and per-terrain level metrics
# =============================================================================

def stair_terrain_level_mean(env, env_ids, terrain_name: str, detailed_only: bool = False) -> torch.Tensor | None:
    """Mean current row level for one terrain type, or the full population.

    Isaac Lab assigns terrain types to columns using cumulative proportions.
    Recreate that assignment from the active generator configuration so a
    type's curve follows the same environments through level changes.
    """
    if detailed_only and not _detailed_teacher_diagnostics(env):
        return None
    terrain = env.scene.terrain
    levels = terrain.terrain_levels.float()
    if terrain_name == "overall":
        return levels.mean()

    generator = terrain.cfg.terrain_generator
    names = list(generator.sub_terrains)
    if terrain_name not in names:
        raise ValueError(f"Unknown terrain type for level logging: {terrain_name}")
    proportions = np.array(
        [generator.sub_terrains[name].proportion for name in names], dtype=np.float64
    )
    if proportions.sum() <= 0:
        raise RuntimeError("Cannot log terrain levels without positive terrain proportions")
    proportions /= proportions.sum()
    column_samples = np.arange(generator.num_cols) / generator.num_cols + 0.001
    column_type_ids = np.searchsorted(np.cumsum(proportions), column_samples, side="right")
    if np.any(column_type_ids >= len(names)):
        raise RuntimeError("Terrain type columns do not match Isaac Lab's generator proportions")

    columns = np.flatnonzero(column_type_ids == names.index(terrain_name))
    selected = torch.isin(
        terrain.terrain_types,
        torch.as_tensor(columns, device=levels.device, dtype=torch.long),
    )
    if not torch.any(selected):
        return levels.new_tensor(float("nan"))
    return levels[selected].mean()


def teacher_action_smooth_l2(env) -> torch.Tensor:
    """Second action difference with episode age, including exact-zero targets.

    A zero action is a valid controller target. Masking history by its numeric
    value makes a repeated nonzero/zero oscillation escape this cost. Keep the
    rough-task curriculum scaling, but isolate this correction to the teacher.
    """
    from .rewards import get_gait_level_tensor

    step = int(env.common_step_counter)
    cached = getattr(env, "_m20_action_smooth_cache", None)
    if cached is not None and cached["step"] == step:
        return cached["cost"]
    action = env.action_manager.action
    previous = env.action_manager.prev_action
    previous_previous = getattr(env, "_m20_action_smooth_previous_previous", None)
    if previous_previous is None or previous_previous.shape != action.shape:
        previous_previous = torch.zeros_like(action)
        env._m20_action_smooth_history_steps = torch.zeros_like(env.episode_length_buf)
    history_steps = env._m20_action_smooth_history_steps
    history_steps[env.episode_length_buf <= 1] = 0
    second_difference = action - 2.0 * previous + previous_previous
    # RSL-RL can randomize the initial episode age. A large age at the first
    # call does not mean this reward already has a complete action history.
    valid_history = (env.episode_length_buf >= 3) & (history_steps >= 2)
    cost = second_difference.square().sum(dim=1) * valid_history
    cost *= get_gait_level_tensor(env)
    history_steps.add_(1).clamp_max_(2)
    env._m20_action_smooth_previous_previous = previous.clone()
    env._m20_action_smooth_cache = {"step": step, "cost": cost}
    return cost


def stair_distance_terrain_levels(
    env,
    env_ids,
    min_commanded_translation_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Adjust difficulty from the full motion opportunity.

    Promote at half the requested episode path; demote only below one quarter
    (up to half a tile). Stops and intentional reversals otherwise retain the
    level. The net-request coherence gate excludes out-and-back / turning
    practice, whose small final displacement is intentional. Net displacement
    still prevents back-and-forth physical motion from farming promotions.
    The explicit zero_stop_v2 stage additionally requires observed, successful
    stair holds before promotion and caps its training terrain level at three.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    distance = torch.linalg.vector_norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    command_term = env.command_manager.get_term("base_velocity")
    commanded_translation = command_term.episode_commanded_translation_m[env_ids]
    commanded_net = torch.linalg.vector_norm(
        command_term.episode_commanded_translation_w[env_ids], dim=1
    )
    coherence = commanded_net / commanded_translation.clamp_min(1.0e-6)
    has_translation_opportunity = commanded_translation >= max(float(min_commanded_translation_m), 1.0)
    coherent_request = coherence >= 0.5
    eligible = has_translation_opportunity & coherent_request
    promotion_distance = 0.5 * commanded_translation
    move_up = eligible & (distance >= promotion_distance)
    if TUNING.get("zero_stop_v5", False):
        stop = getattr(env, '_m20_zero_stop', None)
        exposed = torch.zeros_like(move_up)
        if stop is not None:
            exposed = stop['stats']['stair']['samples'][env_ids] >= 50
        types = terrain.terrain_types[env_ids]
        terrain_names = tuple(terrain.cfg.terrain_generator.sub_terrains)
        proportions = command_term.vel_command_b.new_tensor([
            terrain.cfg.terrain_generator.sub_terrains[name].proportion for name in terrain_names])
        cuts = proportions.cumsum(0)/proportions.sum()
        column_fraction = types.float()/terrain.cfg.terrain_generator.num_cols+.001
        kind = torch.searchsorted(cuts, column_fraction)
        stair_kind = torch.zeros_like(move_up)
        for i, name in enumerate(terrain_names):
            if name in ('pyramid_stairs', 'pyramid_stairs_inv'):
                stair_kind |= kind == i
        move_up &= (~stair_kind | exposed) & (terrain.terrain_levels[env_ids] < TUNING['zero_stop_max_terrain_level'])
    elif TUNING.get("zero_stop_v2", False):
        stop = getattr(env, "_m20_zero_stop", None)
        ready = torch.zeros_like(move_up)
        if stop is not None:
            s = stop["stats"]["stair"]
            count = s["samples"][env_ids]
            ready = ((count >= 50) & (s["success"][env_ids] / count.clamp_min(1.) >= .5)
                     & (s["drift"][env_ids] / count.clamp_min(1.) <= .05))
        # Stage 1 is deliberately conservative: motion alone must not promote
        # environments whose stair stops have never been observed or passed.
        move_up &= ready & (terrain.terrain_levels[env_ids] < TUNING["zero_stop_max_terrain_level"])
    tile_half_length = terrain.cfg.terrain_generator.size[0] * 0.5
    demotion_distance = torch.minimum(
        0.25 * commanded_translation, torch.full_like(commanded_translation, tile_half_length)
    )
    move_down = eligible & (distance < demotion_distance) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    mean_level = torch.mean(terrain.terrain_levels.float())
    from .rewards import update_gait_level_from_terrain_mean

    update_gait_level_from_terrain_mean(mean_level)
    return mean_level
