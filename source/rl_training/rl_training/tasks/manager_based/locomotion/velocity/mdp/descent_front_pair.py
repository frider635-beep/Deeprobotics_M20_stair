# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Soft coordination of the two front wheels on the first descending treads.

The physical edge ledger owns tread identity and confirmed touchdown. This
module only shapes the difference between the two fronts; it prescribes
neither an airborne contact pattern nor a front-to-rear step sequence.
"""

from __future__ import annotations

import torch


_SUM_FIELDS = (
    "samples", "valid_samples", "progress_gap_sum", "forward_gap_sum",
    "pending_samples", "lag_excess_samples", "roll_abs_sum", "high_roll_samples",
    "roll_rate_squared_sum", "cost_sum", "attempts", "left_first", "right_first",
    "simultaneous_first", "completed_pairs", "landing_lag_sum", "late_pairs",
    "timing_interruptions",
)


def _smooth_excess(value: torch.Tensor) -> torch.Tensor:
    positive = value.clamp_min(0.0)
    return 2.0 * (torch.sqrt(1.0 + positive.square()) - 1.0)


def stair_descent_front_pair_cost(
    env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name="base_velocity",
) -> torch.Tensor:
    """Penalize prolonged or strongly unequal front descent, with timing grace.

    Touchdown is latched by the existing ledger. Unloading the first landed
    wheel therefore cannot erase a pending pair. Timing counts only continuous
    forward straight commands: pauses and deliberate turns reset its grace.
    """
    # Local import avoids a cycle when stair_teacher exposes this reward.
    from . import stair_teacher

    tuning = stair_teacher.TUNING
    tick = int(env.common_step_counter)
    state = getattr(env, "_m20_descent_front_pair_state", None)
    if state is not None and state["tick"] == tick:
        return state["cost"]

    stair_teacher._stair_step_completion_events(
        env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name,
    )
    ledger = env._m20_stair_step_state
    robot = env.scene[asset_cfg.name]
    device = env.device
    n = env.num_envs
    if state is None or state["cost"].shape[0] != n:
        # One contiguous accumulator keeps diagnostic updates to a stacked
        # scatter, instead of launching one addition for every field and tread.
        stats_tensor = torch.zeros(n, 3, len(_SUM_FIELDS), device=device)
        state = {
            "tick": -1,
            "cost": torch.zeros(n, device=device),
            "tracked": torch.zeros(n, dtype=torch.bool, device=device),
            "edge_xy": torch.zeros(n, 2, device=device),
            "source_z": torch.zeros(n, device=device),
            "target_z": torch.zeros(n, device=device),
            "edge_count": torch.zeros(n, dtype=torch.long, device=device),
            "level": torch.zeros(n, dtype=torch.long, device=device),
            "previous_paid": torch.zeros(n, 2, dtype=torch.bool, device=device),
            "attempt_started": torch.zeros(n, dtype=torch.bool, device=device),
            "timing": torch.zeros(n, dtype=torch.bool, device=device),
            "lag": torch.zeros(n, device=device),
            "stats_tensor": stats_tensor,
            "stats": {key: stats_tensor[:, :, index] for index, key in enumerate(_SUM_FIELDS)},
        }
        env._m20_descent_front_pair_state = state

    reset = env.episode_length_buf <= 1
    for key in ("tracked", "previous_paid", "attempt_started", "timing", "lag"):
        state[key][reset] = 0
    state["stats_tensor"][reset] = 0

    physical = ledger["active"] & ~ledger["up"] & (ledger["down_successes"] < 3)
    changed = (
        (torch.linalg.vector_norm(ledger["edge_xy"] - state["edge_xy"], dim=-1) > 0.02)
        | ((ledger["source_z"] - state["source_z"]).abs() > 1.0e-4)
        | ((ledger["target_z"] - state["target_z"]).abs() > 1.0e-4)
        | (ledger["completed_count"] != state["edge_count"])
    )
    new_edge = physical & (~state["tracked"] | changed)
    for key in ("previous_paid", "attempt_started", "timing", "lag"):
        state[key][new_edge] = 0
    state["edge_xy"][new_edge] = ledger["edge_xy"][new_edge]
    state["source_z"][new_edge] = ledger["source_z"][new_edge]
    state["target_z"][new_edge] = ledger["target_z"][new_edge]
    state["edge_count"][new_edge] = ledger["completed_count"][new_edge]
    state["level"][new_edge] = ledger["down_successes"][new_edge].long().clamp(0, 2)
    state["tracked"] = physical.clone()

    command = env.command_manager.get_command(command_name)
    eligible = (
        physical & ~reset
        & (command[:, 0] > tuning["translation_threshold"])
        & (command[:, 1].abs() <= tuning["translation_threshold"])
        & (command[:, 2].abs() <= tuning["yaw_threshold"])
    )
    paid = ledger["front_support_paid"]
    any_paid = paid.any(dim=-1)
    both_paid = paid.all(dim=-1)
    phase = eligible & ~both_paid
    pending = phase & any_paid
    interrupted = state["timing"] & ~eligible & physical
    next_lag = torch.where(state["timing"], state["lag"] + env.step_dt, 0.0)
    # Ending a pause or turn starts a new grace period, even on the same edge.
    lag = torch.where(pending, next_lag, 0.0)
    # A pair already completed during an inactive command is unobserved here,
    # rather than incorrectly credited as a simultaneous zero-lag touchdown.
    first = (
        eligible & any_paid & ~state["attempt_started"]
        & (~both_paid | ~state["previous_paid"].any(dim=-1))
    )
    completed = (
        eligible & both_paid & ~state["previous_paid"].all(dim=-1)
        & (state["attempt_started"] | first)
    )
    completion_lag = torch.where(state["timing"], next_lag, 0.0)
    state["attempt_started"] |= first

    wheels = robot.data.body_pos_w[:, asset_cfg.body_ids[:2]]
    radius = torch.where(
        torch.isfinite(ledger["front_source_clearance"]),
        ledger["front_source_clearance"],
        wheels.new_tensor(tuning["wheel_radius"]),
    )
    drop_height = ledger["source_z"] - ledger["target_z"]
    valid = (
        torch.isfinite(wheels).all(dim=(-2, -1))
        & torch.isfinite(drop_height) & (drop_height > 0.03)
        & torch.isfinite(ledger["source_z"])
        & torch.isfinite(ledger["heading"]).all(dim=-1)
    )
    progress = (
        (ledger["source_z"][:, None] + radius - wheels[:, :, 2])
        / drop_height.clamp_min(0.03)[:, None]
    ).clamp(0.0, 1.0)
    progress_gap = torch.nan_to_num((progress[:, 0] - progress[:, 1]).abs(), nan=0.0, posinf=0.0)
    along = (wheels[:, :, :2] * ledger["heading"][:, None]).sum(dim=-1)
    forward_gap = torch.nan_to_num((along[:, 0] - along[:, 1]).abs(), nan=0.0, posinf=0.0)
    progress_cost = _smooth_excess(
        (progress_gap - tuning["descent_front_pair_progress_tolerance"])
        / max(tuning["descent_front_pair_progress_scale"], 1.0e-6)
    )
    forward_cost = _smooth_excess(
        (forward_gap - tuning["descent_front_pair_forward_tolerance"])
        / max(tuning["descent_front_pair_forward_scale"], 1.0e-6)
    )
    lag_cost = _smooth_excess(
        (lag - tuning["descent_front_pair_lag_grace"])
        / max(tuning["descent_front_pair_lag_scale"], 1.0e-6)
    )

    quat = robot.data.root_quat_w
    roll = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]),
        1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()),
    ).abs()
    roll_rate = robot.data.root_ang_vel_b[:, 0].abs()
    finite_pose = torch.isfinite(roll) & torch.isfinite(roll_rate)
    roll = torch.nan_to_num(roll, nan=0.0, posinf=0.0, neginf=0.0)
    roll_rate = torch.nan_to_num(roll_rate, nan=0.0, posinf=0.0, neginf=0.0)
    roll_cost = _smooth_excess(
        (roll - tuning["descent_front_pair_roll_tolerance"])
        / max(tuning["descent_front_pair_roll_scale"], 1.0e-6)
    )
    roll_rate_cost = _smooth_excess(
        (roll_rate - tuning["descent_front_pair_roll_rate_tolerance"])
        / max(tuning["descent_front_pair_roll_rate_scale"], 1.0e-6)
    )
    valid_phase = phase & valid & finite_pose
    geometry_cost = torch.maximum(progress_cost, forward_cost)
    # Delay must still have a gradient when geometric mismatch dominates.
    coordination_cost = geometry_cost + tuning["descent_front_pair_lag_mix"] * lag_cost
    cost = torch.where(
        valid_phase,
        coordination_cost + tuning["descent_front_pair_roll_mix"]
        * torch.maximum(roll_cost, roll_rate_cost),
        0.0,
    )

    fields = {
        "samples": phase.float(),
        "valid_samples": valid_phase.float(),
        "progress_gap_sum": progress_gap * valid_phase,
        "forward_gap_sum": forward_gap * valid_phase,
        "pending_samples": pending.float(),
        "lag_excess_samples": (pending & (lag > tuning["descent_front_pair_lag_grace"])).float(),
        "roll_abs_sum": roll * valid_phase,
        "high_roll_samples": (valid_phase & (roll > tuning["descent_front_pair_roll_tolerance"])).float(),
        "roll_rate_squared_sum": roll_rate.square() * valid_phase,
        "cost_sum": cost,
        "attempts": first.float(),
        "left_first": (first & paid[:, 0] & ~paid[:, 1]).float(),
        "right_first": (first & paid[:, 1] & ~paid[:, 0]).float(),
        "simultaneous_first": (first & both_paid).float(),
        "completed_pairs": completed.float(),
        "landing_lag_sum": completion_lag * completed,
        "late_pairs": (completed & (completion_lag > tuning["descent_front_pair_lag_grace"])).float(),
        "timing_interruptions": interrupted.float(),
    }
    observations = torch.stack(tuple(fields[key] for key in _SUM_FIELDS), dim=-1)
    level_index = state["level"][:, None, None].expand(-1, 1, len(_SUM_FIELDS))
    state["stats_tensor"].scatter_add_(1, level_index, observations[:, None, :])

    state["previous_paid"] = paid.clone()
    state["timing"] = pending.clone()
    state["lag"] = lag
    state["cost"] = cost
    state["tick"] = tick
    return cost


def descent_front_pair_metrics(env, env_ids) -> dict[str, torch.Tensor]:
    """Return first-three-tread diagnostics and their weighted pooling metadata.

    Landing lag measures active straight-command time after the latest timing
    interruption. Same-control-step confirmations count as simultaneous at the
    policy sampling resolution; they are not exact impact-time measurements.
    """
    state = getattr(env, "_m20_descent_front_pair_state", None)
    if state is None:
        return {}
    if env_ids is None:
        env_ids = slice(None)
    sums = {key: value[env_ids].sum(dim=0) for key, value in state["stats"].items()}
    result = {}

    def weighted(name, total, count):
        result[name] = total / count.clamp_min(1.0)
        result[f"__weighted_total__/{name}"] = total
        result[f"__weighted_count__/{name}"] = count

    for level in range(3):
        prefix = f"descent_step{level + 1}_front_pair"
        count = sums["valid_samples"][level]
        attempts = sums["attempts"][level]
        completed = sums["completed_pairs"][level]
        result[f"{prefix}_samples"] = sums["samples"][level]
        result[f"{prefix}_valid_samples"] = count
        result[f"{prefix}_attempts"] = attempts
        result[f"{prefix}_completed_pairs"] = completed
        result[f"{prefix}_timing_interruptions"] = sums["timing_interruptions"][level]
        for name, field in (
            ("progress_gap_mean", "progress_gap_sum"),
            ("forward_gap_mean_m", "forward_gap_sum"),
            ("high_roll_fraction", "high_roll_samples"),
            ("cost_mean", "cost_sum"),
        ):
            weighted(f"{prefix}_{name}", sums[field][level], count)
        weighted(f"{prefix}_roll_abs_mean_deg", torch.rad2deg(sums["roll_abs_sum"][level]), count)
        weighted(f"{prefix}_pending_fraction", sums["pending_samples"][level], sums["samples"][level])
        weighted(f"{prefix}_lag_grace_exceeded_fraction", sums["lag_excess_samples"][level],
                 sums["pending_samples"][level])
        for name, field in (
            ("left_first_fraction", "left_first"),
            ("right_first_fraction", "right_first"),
            ("simultaneous_first_fraction", "simultaneous_first"),
        ):
            weighted(f"{prefix}_{name}", sums[field][level], attempts)
        weighted(f"{prefix}_landing_lag_mean_s", sums["landing_lag_sum"][level], completed)
        weighted(f"{prefix}_late_pair_fraction", sums["late_pairs"][level], completed)
        name = f"{prefix}_roll_rate_rms_rad_s"
        total = sums["roll_rate_squared_sum"][level]
        result[name] = torch.sqrt(total / count.clamp_min(1.0))
        result[f"__rms_total__/{name}"] = total
        result[f"__rms_count__/{name}"] = count
    return result
