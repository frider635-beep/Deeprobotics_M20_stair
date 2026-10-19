#!/usr/bin/env python3
"""Real-MJCF checks for heading chatter, mode handover, and telemetry cadence.

These controlled disturbances check controller behavior, not learned gait quality.
Run with the same Python environment used for deploy_mujoco.py.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
spec = importlib.util.spec_from_file_location("m20_heading_deploy", ROOT / "deploy/deploy_mujoco/deploy_mujoco.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def main():
    args = SimpleNamespace(terrain="flat", terrain_xml=None)
    model = deploy.load_model(ROOT / "deep_robotics_model/M20/mjcf/M20.xml", args)
    contract = deploy.M20Contract(model, args.terrain_config)
    data = deploy.mujoco.MjData(model)
    contract.reset(data)
    root_qpos = int(contract.root_joint.qposadr[0])
    root_qvel = int(contract.root_joint.dofadr[0])
    zero = np.zeros(3, dtype=np.float32)
    controller = deploy.HeadingHoldController(contract)

    def observe(yaw_degrees=0.0, yaw_rate=0.0, command=zero, advance=True):
        if advance:
            data.time += deploy.POLICY_DT
        angle = math.radians(yaw_degrees)
        data.qpos[root_qpos + 3:root_qpos + 7] = [math.cos(angle/2), 0, 0, math.sin(angle/2)]
        data.qvel[root_qvel + 5] = yaw_rate
        deploy.mujoco.mj_forward(model, data)
        return controller.update(data, command)

    # Exact float32 thresholds and their immediate neighbors match training,
    # including callers that construct a float64 request array.
    above = float(np.nextafter(np.float32(.1), np.float32(np.inf)))
    for dtype in (np.float32, np.float64):
        for sign in (-1., 1.):
            for axis in range(3):
                requested = np.zeros(3, dtype=dtype)
                requested[axis] = sign * .1
                _, info = observe(command=requested)
                assert info["mode"] == "zero", (dtype, requested, info)
                requested[axis] = sign * above
                _, info = observe(command=requested)
                assert info["mode"] == ("straight" if axis == 0 else "manual")
        _, info = observe(command=np.array([.7, .1, -.1], dtype=dtype))
        assert info["mode"] == "straight"
    # Keep all later disturbances independent of these mode transitions.
    controller.reset()

    # Small motion inside the entry deadband must not create wheel commands.
    observe()
    for angle in (0.2, -0.2, 0.45, -0.45, 0.2):
        effective, info = observe(angle)
        assert effective[2] == 0.0 and not info["active"]
    effective, info = observe(-0.6)
    assert effective[2] > 0 and info["active"]
    effective, info = observe(-0.4)
    assert info["active"]  # Exit threshold differs from the entry threshold.
    effective, info = observe(-0.2)
    assert not info["active"] and effective[2] == 0

    # A measurement disturbance cannot instantaneously reverse correction.
    outputs, filtered = [], []
    for step in range(100):
        effective, info = observe(-5.0, .3 if step % 2 else -.3)
        outputs.append(float(effective[2]))
        filtered.append(info["filtered_yaw_rate_rad_s"])
        # Observation/reward duplicate reads must not advance filter state.
        duplicate, duplicate_info = controller.update(data, zero)
        assert np.array_equal(effective, duplicate) and info == duplicate_info
    slew = np.max(np.abs(np.diff(outputs))) / deploy.POLICY_DT
    assert slew <= deploy.HEADING_HOLD_SLEW_FLAT + 1e-6
    assert max(abs(value) for value in filtered[-20:]) < .10
    assert max(abs(value) for value in outputs) <= deploy.HEADING_HOLD_FLAT[2] + 1e-6

    # Manual turning bypasses correction, then re-latches at the present yaw.
    manual = np.array([0., 0., .6], dtype=np.float32)
    effective, info = observe(-12., .1, manual)
    assert np.array_equal(effective, manual) and info["mode"] == "manual"
    effective, info = observe(-12., 0.)
    assert abs(info["error_deg"]) < 1e-6 and effective[2] == 0
    target = controller.target_yaw
    observe(-13., 0.)
    assert controller.target_yaw == target
    # Stopping during an established correction preserves the rate bound.
    moving = np.array([.7, 0, 0], dtype=np.float32)
    observe(-13., 0., moving)
    last = 0.0
    for _ in range(10):
        effective, _ = observe(-30., 0., moving)
        last = float(effective[2])
    assert last > .1
    effective, info = observe(-30., 0., zero)
    assert abs(float(effective[2]) - last) <= deploy.HEADING_HOLD_SLEW_FLAT * deploy.POLICY_DT + 1e-6
    assert abs(info["error_deg"]) < 1e-6
    # Time reset must discard targets and filter history from the old episode.
    data.time = 0
    effective, info = observe(17., 0., advance=False)
    assert abs(info["error_deg"]) < 1e-6 and effective[2] == 0

    # Same-time command changes must not integrate the filter repeatedly.
    target_before = controller.target_yaw
    first, _ = controller.update(data, zero)
    for _ in range(10):
        repeated, _ = controller.update(data, zero)
        assert np.array_equal(first, repeated) and controller.target_yaw == target_before

    # Action/contact histories are read once per policy interval, reset on gaps.
    monitor = deploy.StairTelemetry(contract)
    action = np.zeros(deploy.ACTION_DIM)
    for _ in range(4):
        contract.apply_pd(data, action)
        deploy.mujoco.mj_step(model, data)
        monitor.observe_physics_step(data)
    sample = monitor.sample(data, action)
    assert sample["physics_substeps"] == 4 and sample["leg_action_delta_rms"] == 0
    action[12:] = np.array([.2, -.2, .2, -.2])
    action[0] = .2
    for _ in range(4):
        contract.apply_pd(data, action)
        deploy.mujoco.mj_step(model, data)
        monitor.observe_physics_step(data)
    sample = monitor.sample(data, action)
    assert sample["physics_substeps"] == 4 and abs(sample["wheel_action_delta_rms"] - .2) < 1e-9
    assert abs(sample["wheel_target_delta_rms_m_s"] - .2 * 5 * deploy.WHEEL_RADIUS) < 1e-9
    assert abs(sample["leg_target_delta_rms_rad"] - .2 * .125 / math.sqrt(12)) < 1e-9
    assert sample["rear_front_wheel_contact_physics_substeps"] == 0
    data.time += 1.0
    action[12:] *= -1
    sample = monitor.sample(data, action)
    assert sample["physics_substeps"] == 0 and sample["wheel_action_delta_rms"] == 0

    # Positive-force contacts must be counted, even if they end before sampling.
    collision_model = deploy.mujoco.MjModel.from_xml_string('''
    <mujoco><option timestep="0.005" gravity="0 0 0"/>
    <worldbody>
      <body name="fl_wheel"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
      <body name="fr_wheel" pos="0 1 0"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
      <body name="hl_wheel" pos="0.15 0 0"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
      <body name="hr_wheel" pos="0.5 1 0"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
    </worldbody></mujoco>''')
    collision_data = deploy.mujoco.MjData(collision_model)
    collision_contract = SimpleNamespace(model=collision_model,
        wheel_body_ids=np.array([collision_model.body(name).id for name in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")]))
    collision_monitor = deploy.StairTelemetry(collision_contract)
    deploy.mujoco.mj_step(collision_model, collision_data)
    collision_monitor.observe_physics_step(collision_data)
    assert collision_monitor._physics_window["rear_front_wheel_contact_physics_substeps"] == 1
    assert collision_monitor._physics_window["rear_front_peak_contact_force_N"] > 1
    assert collision_monitor._physics_window["rear_front_contact_impulse_Ns"] > 0

    print(json.dumps({"real_mjcf": True, "deadband_hysteresis": True,
                      "float32_command_threshold_parity": True,
                      "filter_noise_peak_rad_s": max(abs(x) for x in filtered[-20:]),
                      "measured_slew_max_rad_s2": slew, "duplicate_updates_idempotent": True,
                      "manual_bypass_and_target_latching": True,
                      "simulation_reset_clears_state": True,
                      "stop_transition_slew_bound": True,
                      "physics_substep_and_action_telemetry": True,
                      "positive_force_contact_counting": True}, indent=2))


if __name__ == "__main__":
    main()
