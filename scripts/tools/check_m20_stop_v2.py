#!/usr/bin/env python3
"""CPU counterexamples for stop rewards, exact metric pooling and controller parity.

Uses production reward bodies and the real MuJoCo deployment controller. These
checks establish attribution and state consistency, not trained-policy success.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from check_m20_curriculum import production_namespace
from check_stair_rewards import Entity, make_env


ROOT = Path(__file__).resolve().parents[2]


def fixture(ns, initial_step=0, initial_age=2, flat=True):
    env = make_env()
    env.common_step_counter = initial_step
    env.episode_length_buf.fill_(initial_age)
    data = env.scene["robot"].data
    data.root_lin_vel_w = torch.zeros(1, 3)
    data.root_ang_vel_b = torch.zeros(1, 3)
    data.joint_vel = torch.zeros(1, 16)
    command = env.command_manager.get_command("base_velocity")
    command.zero_()
    leg = SimpleNamespace(processed_actions=torch.zeros(1, 12))
    wheel = SimpleNamespace(processed_actions=torch.zeros(1, 4))
    terms = {"joint_pos": leg, "joint_vel": wheel}
    env.action_manager = SimpleNamespace(get_term=terms.__getitem__)
    ns["_flat_surface_mask"] = lambda *args: torch.full((1,), flat, dtype=torch.bool)
    ns["_stair_activity_mask"] = lambda *args: torch.full((1,), not flat, dtype=torch.bool)
    ns["_stair_scan_context"] = lambda *args: {
        "terrain_gate": torch.full((1,), float(not flat)),
        "up_gate": torch.zeros(1), "down_gate": torch.zeros(1)}
    return env, leg, wheel


def advance(env):
    env.common_step_counter += 1
    env.episode_length_buf += 1


def check_wheel_cancellation(ns):
    env, _, wheel = fixture(ns)
    cost = ns["zero_command_wheel_action_cost"]
    still = float(cost(env))
    # Physical rolling targets front +0.1, rear -0.1 have zero mean/differential.
    wheel.processed_actions[:] = -torch.tensor([[.1, .1, -.1, -.1]]) / ns["TUNING"]["wheel_radius"]
    cancellation = float(cost(env))
    assert still == 0 and cancellation > .5
    # The four physical targets needed for a legitimate correction remain free.
    advance(env)
    yaw = -.10
    env.scene["robot"].data.root_quat_w[:] = torch.tensor([[math.cos(yaw/2), 0, 0, math.sin(yaw/2)]])
    effective = ns["heading_hold_velocity_commands"](env)[:, 2:3]
    wheel.processed_actions[:] = -(effective * torch.tensor([[-.28, .28, -.28, .28]])) / ns["TUNING"]["wheel_radius"]
    correction = float(cost(env))
    assert correction < 1e-6
    return {"per_wheel_cancellation_counterexample_cost": cancellation,
            "stationary_and_corrective_targets_cost": [still, correction]}


def check_action_dynamics(ns):
    fn = ns["zero_command_action_dynamics_cost"]
    cfg = Entity("robot", joint_ids=list(range(12)))

    def run(alternating):
        env, leg, wheel = fixture(ns)
        values = []
        for i in range(100):
            if alternating:
                leg.processed_actions.fill_(.02 if i % 2 else 0.)
                wheel.processed_actions.fill_(.5 if i % 2 else 0.)
            value = float(fn(env, cfg))
            before = {key: tensor.clone() for key, tensor in env._m20_stop_action_stats.items()}
            duplicate = float(fn(env, cfg))
            assert duplicate == value
            assert all(torch.equal(tensor, before[key]) for key, tensor in env._m20_stop_action_stats.items())
            values.append(value)
            advance(env)
        return env, values

    stationary, zero_cost = run(False)
    oscillating, alternating_cost = run(True)
    assert max(zero_cost) == 0
    assert min(alternating_cost[45:]) > 1
    assert max(alternating_cost[:21]) == 0  # 0.4 s braking grace.
    assert alternating_cost[30] > 0 and alternating_cost[30] < alternating_cost[45]
    assert oscillating._m20_stop_action_stats["samples"].item() > 0

    # Reset clears history and per-episode statistics, even after large targets.
    oscillating.episode_length_buf.fill_(1)
    oscillating.common_step_counter += 1
    value = float(fn(oscillating, cfg))
    assert value == 0 and oscillating._m20_stop_action_stats["samples"].item() == 0
    # At startup, randomized episode age must not provide imaginary history.
    late, leg, wheel = fixture(ns, initial_step=100, initial_age=500)
    leg.processed_actions.fill_(.2)
    assert float(fn(late, cfg)) == 0
    age = late.common_step_counter - int(late._m20_zero_hold_heading_state["hold_start_step"][0])
    assert age == 0, "Fresh zero hold incorrectly inherits global-step age"
    advance(late)
    leg.processed_actions.fill_(-.2)
    assert float(fn(late, cfg)) == 0  # Still braking, despite target change.
    return {"stationary_dynamics_cost": 0.,
            "alternating_dynamics_cost_after_ramp": alternating_cost[45],
            "braking_grace_then_gradual_ramp": True,
            "duplicate_dynamics_consumers_do_not_advance_history_or_stats": True,
            "random_initial_age_and_reset_history": True}


def check_quiet_metrics(ns):
    env, _, _ = fixture(ns)
    cfg = Entity("robot", joint_ids=list(range(12, 16)))
    fn = ns["zero_command_hold_cost"]
    for i in range(100):
        fn(env, Entity("robot"), cfg)
        state = env._m20_zero_hold_heading_state
        samples = state["samples"].clone()
        quiet = state["quiet_run_s"].clone()
        fn(env, Entity("robot"), cfg)
        assert torch.equal(samples, state["samples"]), "Duplicate reward read increments sample count"
        assert torch.equal(quiet, state["quiet_run_s"]), "Duplicate reward read advances quiet clock"
        advance(env)
    result = ns["zero_command_hold_metrics"](env, slice(None))
    assert result["flat_settled_success_fraction"] == 1
    assert result["flat_holds_reached_quiet_1s_fraction"] == 1
    # A short disturbance must immediately break consecutive quiet duration.
    env.scene["robot"].data.joint_vel[0, 12] = 1.
    fn(env, Entity("robot"), cfg)
    assert env._m20_zero_hold_heading_state["quiet_run_s"].item() == 0
    return {"quiet_duration_and_hold_metrics": True,
            "duplicate_hold_consumers_do_not_inflate_quiet_time": True}


def check_configured_quiet_chatter(ns):
    env, leg, wheel = fixture(ns)
    wheel_cfg = Entity("robot", joint_ids=list(range(12, 16)))
    leg_cfg = Entity("robot", joint_ids=list(range(12)))
    hold, dynamics = ns["zero_command_hold_cost"], ns["zero_command_action_dynamics_cost"]

    def tick():
        # Match configured reward order: hold observes read-only target history,
        # then the dynamics reward records the current control tick.
        hold(env, Entity("robot"), wheel_cfg, leg_joint_cfg=leg_cfg)
        dynamics(env, leg_cfg)
        advance(env)

    for _ in range(100):
        tick()
    assert env._m20_zero_hold_heading_state["quiet_run_s"].item() > 1
    # Historical pointwise velocity success stays comparable; quiet additionally
    # rejects visible leg movement even when base/wheel velocities are zero.
    env.scene["robot"].data.joint_vel[:, :12] = .5
    tick()
    assert env._m20_zero_hold_heading_state["quiet_run_s"].item() == 0
    env.scene["robot"].data.joint_vel.zero_()
    for step in range(80):
        leg.processed_actions.fill_(.02 if step % 2 else 0.)
        tick()
        if step > 1:
            assert env._m20_zero_hold_heading_state["quiet_run_s"].item() == 0
    leg.processed_actions.zero_()
    for step in range(80):
        wheel.processed_actions.fill_(.5 if step % 2 else 0.)
        tick()
        if step > 1:
            assert env._m20_zero_hold_heading_state["quiet_run_s"].item() == 0
    wheel.processed_actions.zero_()
    for _ in range(65):
        tick()
    assert env._m20_zero_hold_heading_state["quiet_run_s"].item() > 1
    return {"configured_quiet_rejects_leg_motion_and_target_chatter": True,
            "configured_quiet_recovers_after_actual_targets_settle": True}


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_pooling():
    pool = import_file("m20_pooling_counterexamples", ROOT / "source/rl_training/rl_training/utils/metric_aggregation.py").pool_episode_metrics
    prefix = "Curriculum/zero_command_hold/"
    mean, rms, minimum = prefix + "speed", prefix + "delta", prefix + "distance"
    rows = [
        {"ordinary": torch.tensor(1.),
         prefix + "__weighted_total__/speed": torch.tensor(0.), prefix + "__weighted_count__/speed": torch.tensor(0.),
         prefix + "__rms_total__/delta": torch.tensor(0.), prefix + "__rms_count__/delta": torch.tensor(0.),
         prefix + "__minimum_value__/distance": torch.tensor(float("inf")), prefix + "__minimum_count__/distance": torch.tensor(0.)},
        {"ordinary": torch.tensor(3.), mean: torch.tensor(2.),
         prefix + "__weighted_total__/speed": torch.tensor(4.), prefix + "__weighted_count__/speed": torch.tensor(2.),
         prefix + "__rms_total__/delta": torch.tensor(8.), prefix + "__rms_count__/delta": torch.tensor(2.),
         prefix + "__minimum_value__/distance": torch.tensor(.31), prefix + "__minimum_count__/distance": torch.tensor(5.)},
        {"ordinary": torch.tensor(5.), mean: torch.tensor(4.),
         prefix + "__weighted_total__/speed": torch.tensor(12.), prefix + "__weighted_count__/speed": torch.tensor(3.),
         prefix + "__rms_total__/delta": torch.tensor(27.), prefix + "__rms_count__/delta": torch.tensor(3.),
         prefix + "__minimum_value__/distance": torch.tensor(.28), prefix + "__minimum_count__/distance": torch.tensor(1.)},
    ]
    pooled = pool(rows)
    for row in pooled:
        assert math.isclose(float(row[mean]), 3.2, rel_tol=1e-6)
        assert math.isclose(float(row[rms]), math.sqrt(7), rel_tol=1e-6)
        assert math.isclose(float(row[minimum]), .28, rel_tol=1e-6)
        assert not any("__" in key for key in row)
    assert [float(row["ordinary"]) for row in pooled] == [1., 3., 5.]
    # The first logger row lacked these targets; pooling must add all three.
    assert mean in pooled[0] and rms in pooled[0] and minimum in pooled[0]
    empty = pool([{mean: torch.tensor(0.), rms: torch.tensor(0.), minimum: torch.tensor(float("inf")),
                   prefix + "__weighted_total__/speed": torch.tensor(0.),
                   prefix + "__weighted_count__/speed": torch.tensor(0.),
                   prefix + "__rms_total__/delta": torch.tensor(0.),
                   prefix + "__rms_count__/delta": torch.tensor(0.),
                   prefix + "__minimum_value__/distance": torch.tensor(float("inf")),
                   prefix + "__minimum_count__/distance": torch.tensor(0.)}])
    assert not any(key in empty[0] for key in (mean, rms, minimum))
    assert pool([]) == []
    assert mean in rows[1] and any("__" in key for key in rows[0])  # Input untouched.
    return {"pooled_sample_weighted_mean": 3.2, "pooled_rms": math.sqrt(7),
            "pooled_minimum": .28, "zero_samples_omit_diagnostic": True,
            "first_row_missing_target_is_populated": True, "ordinary_metrics_preserved": True}


def check_heading_parity(ns):
    sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
    deploy = import_file("m20_stop_v2_deploy", ROOT / "deploy/deploy_mujoco/deploy_mujoco.py")
    args = SimpleNamespace(terrain="flat", terrain_xml=None)
    model = deploy.load_model(ROOT / "deep_robotics_model/M20/mjcf/M20.xml", args)
    contract = deploy.M20Contract(model, args.terrain_config)
    data = deploy.mujoco.MjData(model)
    contract.reset(data)
    root_qpos = int(contract.root_joint.qposadr[0])
    root_qvel = int(contract.root_joint.dofadr[0])
    controller = deploy.HeadingHoldController(contract)
    env, _, _ = fixture(ns)
    terrain_state = {"stairs": False}
    ns["_stair_scan_context"] = lambda *args: {
        "terrain_gate": torch.tensor([float(terrain_state["stairs"])]),
        "up_gate": torch.tensor([float(terrain_state["stairs"])]), "down_gate": torch.zeros(1)}
    command = env.command_manager.get_command("base_velocity")
    max_error = 0.
    requests = ([0., 0., 0.], [.7, 0, 0], [0., 0., 0.], [0., 0., .6], [-.5, 0, 0], [0., 0., 0.])
    for step in range(240):
        terrain_state["stairs"] = 60 <= step < 150
        contract.terrain = SimpleNamespace(kind="ascent" if terrain_state["stairs"] else "flat",
                                           parameters={"start_x": [0.]}, goal_x=10.)
        requested = np.array(requests[step // 40], dtype=np.float32)
        yaw = .1 * math.sin(step * .13) + .0002 * step
        yaw_rate = .15 * math.sin(step * .61)
        command[:] = torch.from_numpy(requested)[None]
        env.scene["robot"].data.root_quat_w[:] = torch.tensor([[math.cos(yaw/2), 0, 0, math.sin(yaw/2)]])
        env.scene["robot"].data.root_ang_vel_b[:, 2] = yaw_rate
        env.common_step_counter = step
        env.episode_length_buf.fill_(step + 2)
        data.time = step * deploy.POLICY_DT
        data.qpos[root_qpos] = 0.
        data.qpos[root_qpos + 3:root_qpos + 7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        data.qvel[root_qvel + 5] = yaw_rate
        deploy.mujoco.mj_forward(model, data)
        train = ns["heading_hold_velocity_commands"](env).numpy()[0]
        actual, _ = controller.update(data, requested)
        error = float(np.max(np.abs(train - actual)))
        max_error = max(max_error, error)
        assert error < 2e-6, f"Training/deploy divergence at step {step}: {train} vs {actual}"
        assert np.array_equal(ns["heading_hold_velocity_commands"](env).numpy()[0], train)
        assert np.array_equal(controller.update(data, requested)[0], actual)
    threshold = np.float32(.1)
    above = np.nextafter(threshold, np.float32(1.))
    below = np.nextafter(threshold, np.float32(0.))
    boundary_cases = ([threshold, 0, 0], [-threshold, 0, 0], [above, 0, 0], [-above, 0, 0],
                      [below, 0, 0], [0, threshold, 0], [0, above, 0], [0, 0, threshold],
                      [0, 0, -threshold], [0, 0, above], [0, 0, -above], [.08, .06, 0])
    for values in boundary_cases:
        for dtype in (np.float32, np.float64):
            requested = np.array(values, dtype=dtype)
            boundary_env, _, _ = fixture(ns)
            boundary_env.command_manager.get_command("base_velocity")[:] = torch.tensor(values)[None]
            train = ns["heading_hold_velocity_commands"](boundary_env).numpy()[0]
            deploy.mujoco.mj_forward(model, data)
            controller = deploy.HeadingHoldController(contract)
            actual, info = controller.update(data, requested)
            train_mode = int(boundary_env._m20_heading_command_filter["mode"][0])
            expected_mode = {0: "manual", 1: "straight", 2: "zero"}[train_mode]
            assert info["mode"] == expected_mode, f"Boundary mode mismatch for {requested}: {info['mode']} vs {expected_mode}"
            # The initial poses differ, but mode-dependent passthrough XY is identical.
            assert np.array_equal(train[:2], actual[:2].astype(np.float32))
    return {"training_deploy_multitick_heading_max_error": max_error,
            "heading_duplicate_consumers_idempotent": True,
            "heading_flat_stair_flat_handover_parity": True,
            "heading_float32_float64_threshold_boundary_parity": True}


if __name__ == "__main__":
    checks = (check_wheel_cancellation, check_action_dynamics, check_quiet_metrics,
              check_configured_quiet_chatter, check_heading_parity)
    results, failures = {}, []
    for check in checks:
        try:
            results.update(check(production_namespace()))
        except Exception as error:
            failures.append(f"{check.__name__}: {type(error).__name__}: {error}")
    try:
        results.update(check_pooling())
    except Exception as error:
        failures.append(f"check_pooling: {type(error).__name__}: {error}")
    print(json.dumps({"checks": results, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)
