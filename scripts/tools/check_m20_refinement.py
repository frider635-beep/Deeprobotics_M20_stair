#!/usr/bin/env python3
"""CPU counterexamples for refinement rewards and the real MuJoCo controller.

Checks wiring/signs/monotonicity, not learned locomotion or collision freedom.
Isaac imports are substituted only for the reward checks; MuJoCo uses the MJCF.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from check_stair_rewards import Entity, MDP, make_env
from check_m20_curriculum import production_namespace


ROOT = Path(__file__).resolve().parents[2]


def check_rewards():
    ns = production_namespace()
    tuning = ns["TUNING"]
    result = {}
    assert tuning["entry_front_lead_weight"] == tuning["entry_front_progress_weight"] == tuning["entry_sequence_weight"] == 0
    assert tuning["entry_rear_early_transfer_weight"] < 0
    risk = ns["descent_transfer_risk"]
    def transfer(wheel, shank, moving=True):
        return float(risk(torch.full((1, 2), wheel), torch.full((1, 2), shank),
                          torch.full((1, 2), moving, dtype=torch.bool))[0])
    assert transfer(.40, .30) == 0
    assert transfer(.26, .30) > 0
    assert transfer(.40, .18) > 0
    assert transfer(.20, .18) > transfer(.26, .30)
    assert transfer(.20, .18, False) == 0
    approaching = risk(torch.full((1, 2), .30), torch.full((1, 2), .25),
                       torch.ones(1, 2, dtype=torch.bool), torch.full((1, 2), .25))[0]
    assert transfer(.30, .25) == 0 and float(approaching) > 0
    result["safe_same_tread_transfer_allowed"] = True
    result["wheel_and_shank_transfer_guards"] = True
    result["predictive_guard_only_expands_for_closing_rear_wheel"] = True

    # Large heading errors must continue to receive distinguishable costs.
    context = {"heading_error": torch.zeros(1), "active": torch.ones(1, dtype=torch.bool)}
    ns["_ascent_direction_context"] = lambda *args: context
    values = []
    for angle in (3, 10, 20, 30, 60):
        context["heading_error"].fill_(math.radians(angle))
        values.append(float(ns["stair_ascent_heading_cost"](None, None, None, None)))
    assert all(b > a for a, b in zip(values, values[1:]))
    result["heading_costs_3_10_20_30_60deg"] = values

    ns["_flat_surface_mask"] = lambda env, *args: torch.ones(env.num_envs, dtype=torch.bool)
    ns["_stair_activity_mask"] = lambda env: torch.zeros(env.num_envs, dtype=torch.bool)
    ns["_stair_scan_context"] = lambda env, *args: {
        "terrain_gate": torch.zeros(env.num_envs), "up_gate": torch.zeros(env.num_envs),
        "down_gate": torch.zeros(env.num_envs)}
    env = make_env()
    robot = env.scene["robot"].data
    robot.root_lin_vel_w = torch.zeros(1, 3)
    robot.root_ang_vel_b = torch.zeros(1, 3)
    robot.joint_vel = torch.zeros(1, 4)
    command = env.command_manager.get_command("base_velocity")
    command.zero_()
    wheel_cfg = Entity("robot", joint_ids=list(range(4)))
    hold = ns["zero_command_hold_cost"]
    assert float(hold(env, Entity("robot"), wheel_cfg)) == 0
    env.common_step_counter = 30
    env.episode_length_buf += 1
    hold(env, Entity("robot"), wheel_cfg)
    metrics = ns["zero_command_hold_metrics"](env, slice(None))
    assert float(metrics["flat_settled_success_fraction"]) == 1
    costs = []
    for velocity in (.02, .08, .3, .8):
        robot.root_lin_vel_w[:, 0] = velocity
        costs.append(float(hold(env, Entity("robot"), wheel_cfg)))
    assert all(b > a for a, b in zip(costs, costs[1:]))
    result["zero_hold_costs_002_008_03_08m_s"] = costs

    # A rightward heading error calls for a positive (left) yaw correction.
    # Negative-Y wheel axes require positive left-wheel joint velocities.
    robot.root_lin_vel_w.zero_()
    yaw = math.radians(-5)
    robot.root_quat_w[:] = torch.tensor([[math.cos(yaw/2), 0, 0, math.sin(yaw/2)]])
    effective = ns["heading_hold_velocity_commands"](env)
    assert effective[0, 2] > 0
    qvel = effective[:, 2:3] * torch.tensor([[.28, -.28, .28, -.28]]) / tuning["wheel_radius"]
    action_term = SimpleNamespace(processed_actions=qvel)
    env.action_manager = SimpleNamespace(get_term=lambda name: action_term)
    action_cost = ns["zero_command_wheel_action_cost"]
    good = float(action_cost(env))
    action_term.processed_actions = -qvel
    bad = float(action_cost(env))
    assert good < 1e-6 and bad > good
    result["wheel_yaw_sign_cost_correct_vs_reversed"] = [good, bad]
    return result, tuning


def check_mujoco(tuning):
    deploy = ROOT / "deploy/deploy_mujoco"
    sys.path.insert(0, str(deploy))
    spec = importlib.util.spec_from_file_location("m20_refinement_deploy", deploy / "deploy_mujoco.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(terrain=None, terrain_xml=deploy / "terrains/stairs_descent.xml")
    model = module.load_model(ROOT / "deep_robotics_model/M20/mjcf/M20.xml", args)
    contract = module.M20Contract(model, args.terrain_config)
    data = module.mujoco.MjData(model)
    contract.reset(data)
    assert np.allclose(model.jnt_axis[[model.joint(n).id for n in module.JOINT_NAMES[12:]]], [0, -1, 0])
    for values, prefix in ((module.HEADING_HOLD_FLAT, ""), (module.HEADING_HOLD_STAIRS, "stair_")):
        expected = tuple(tuning[f"{prefix}heading_hold_{name}"] for name in ("kp", "kd", "max_yaw_rate", "deadband"))
        assert np.allclose(values, expected)
    controller = module.HeadingHoldController(contract)
    request = np.array([.7, 0, 0], dtype=np.float32)
    controller.update(data, request)
    root = int(contract.root_joint.qposadr[0])
    data.qpos[root+3:root+7] = [math.cos(-.1/2), 0, 0, math.sin(-.1/2)]
    data.time += module.POLICY_DT
    module.mujoco.mj_forward(model, data)
    effective, info = controller.update(data, request)
    assert effective[2] > 0 and info["mode"] == "straight"
    controller.update(data, np.zeros(3, dtype=np.float32))
    target = controller.target_yaw
    controller.update(data, np.zeros(3, dtype=np.float32))
    assert controller.target_yaw == target
    disabled = module.HeadingHoldController(contract, enabled=False)
    assert np.array_equal(disabled.update(data, request)[0], request)
    sample = module.StairTelemetry(contract).sample(data)
    for key in ("wheel_pair_distances", "rear_wheel_front_shank_distances", "hip_wheel_forward"):
        assert np.isfinite(sample[key]).all()
    segments = [{"duration_s": 6, "command": [0, 0, 0]}, {"duration_s": 7, "command": [.7, 0, 0]}]
    assert module.command_profile_at(segments, 5.98)[0] == 0
    assert abs(module.command_profile_at(segments, 6)[0]-.7) < 1e-6
    assert module.command_profile_at(segments, 13)[0] == 0
    # Exercise the complete production gait cache, including its new keys,
    # on named bodies from the real MJCF. Only scan/ledger inputs are supplied.
    ns = production_namespace()
    env = make_env()
    robot = env.scene["robot"]
    robot.data.body_pos_w = torch.tensor(data.xpos[None], dtype=torch.float32)
    robot.data.body_lin_vel_w = torch.zeros_like(robot.data.body_pos_w)
    robot.data.root_quat_w = torch.tensor(data.xquat[contract.base_id][None], dtype=torch.float32)
    robot.data.root_pos_w = torch.tensor(data.xpos[contract.base_id][None], dtype=torch.float32)
    robot.data.root_ang_vel_b = torch.zeros(1, 3)
    robot.data.joint_pos = torch.tensor(data.qpos[contract.qpos_ids][None], dtype=torch.float32)
    robot.data.default_joint_pos = robot.data.joint_pos.clone()
    robot.find_bodies = lambda names, preserve_order: ([model.body(n).id for n in names], names)
    z = torch.zeros(1)
    scan = dict(gate=z, terrain_gate=torch.ones(1), up_gate=z, down_gate=torch.ones(1),
                edge_x=torch.tensor([.25]), edge_delta=torch.tensor([-.15]),
                lower_z=torch.tensor([3.]), upper_z=torch.tensor([2.85]),
                ascending_edges_valid=torch.zeros(1, 1, dtype=torch.bool),
                ascending_edges_upper_z=torch.tensor([[2.85]]), ascending_edges_lower_z=torch.tensor([[3.]]),
                ascending_edges_x=torch.tensor([.25]), edge_geometry=torch.zeros(1, 1, dtype=torch.bool))
    ns["_stair_scan_context"] = lambda *args: scan
    ns["_flat_surface_mask"] = lambda *args: torch.zeros(1, dtype=torch.bool)
    ns["ascent_state"] = lambda *args: dict(active=torch.zeros(1, dtype=torch.bool), source_z=z, target_z=z)
    ns["_stair_step_completion_events"] = lambda *args: None
    env._m20_stair_step_state = dict(active=torch.ones(1, dtype=torch.bool), up=torch.zeros(1, dtype=torch.bool),
        edge_xy=robot.data.root_pos_w[:, :2].clone(), heading=torch.tensor([[1., 0.]]),
        source_z=torch.tensor([3.]), target_z=torch.tensor([2.85]), front_reached=torch.zeros(1, 2, dtype=torch.bool),
        rear_support_paid=torch.zeros(1, 2, dtype=torch.bool),
        down_successes=z, front_source_clearance=torch.full((1, 2), float('nan')))
    bindings = dict(asset_cfg=Entity('robot', body_ids=contract.wheel_body_ids.tolist()),
                    hip_cfg=Entity('robot', body_ids=contract.hip_body_ids.tolist()),
                    knee_cfg=Entity('robot', joint_ids=[2,5,8,11]), sensor_cfg=Entity('height_scanner'),
                    contact_sensor_cfg=Entity('contact_forces'))
    before = float(ns['descent_front_posture_cost'](env, **bindings))
    for level in range(3):
        env._m20_stair_step_state['down_successes'].fill_(level)
        env.common_step_counter += 1
        robot.data.body_pos_w[:, contract.wheel_body_ids[:2], 0] += .04
        after = float(ns['descent_front_posture_cost'](env, **bindings))
        assert after > before
        before = after
        assert torch.isfinite(ns['stair_descent_entry_rear_early_transfer_cost'](env, **bindings)).all()
        assert torch.isfinite(ns['stair_descent_wheel_shank_cost'](env, **bindings)).all()
        metrics = ns['gait_quality_metrics'](env, slice(None))
        assert f'descent_step{level+1}_front_support_reach_excess_fraction' in metrics
    # A pause cannot switch off the proximity guard while descent is latched.
    env.command_manager.get_command('base_velocity').zero_()
    env.common_step_counter += 1
    robot.data.body_pos_w[:, contract.wheel_body_ids[2:]] = (
        robot.data.body_pos_w[:, contract.wheel_body_ids[:2]] - torch.tensor([.20, 0., 0.]))
    assert float(ns['stair_descent_entry_wheel_spacing_left_cost'](env, **bindings)) > 0
    return {"real_mjcf_axes_and_heading_parameters_match": True,
            "controller_positive_correction_for_right_drift": float(effective[2]),
            "telemetry_geometry_finite": True, "profile_boundaries": True,
            "full_gait_cache_and_three_descent_posture_levels": True,
            "descent_proximity_guard_survives_zero_command": True}


if __name__ == "__main__":
    torch.set_num_threads(1)
    rewards, tuning = check_rewards()
    print(json.dumps({"reward_checks": rewards, "mujoco_checks": check_mujoco(tuning)}, indent=2))
