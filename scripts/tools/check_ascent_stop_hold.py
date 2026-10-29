#!/usr/bin/env python3
"""Check ascent stop state transitions and optional telemetry-state physics replay.

Never loads a policy checkpoint. Replay uses approximate states reconstructed
from CSV and recorded open-loop actions, so it is not a closed-loop policy eval.
"""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
spec = importlib.util.spec_from_file_location("ascent_hold_deploy", ROOT / "deploy/deploy_mujoco/deploy_mujoco.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def state_checks():
    control = deploy.AscentStopHold(deploy.DEFAULT_JOINT_POS, deploy.ACTION_SCALE)
    joints = deploy.DEFAULT_JOINT_POS.copy()
    action = np.linspace(-.2, .2, 16)
    args = dict(requested=np.zeros(3), action=action, position=np.zeros(2),
                velocity=np.zeros(2), yaw=0., yaw_rate=0., joints=joints,
                joint_velocity=np.zeros(16), support=np.ones(4, dtype=bool),
                wheel_lateral=np.array([.2, -.2, .2, -.2]), eligible=True)
    control.update(now=0., **args)
    assert control.info["ascent_stop_hold_phase"] == "brake"
    assert not control.legs_latched
    effort = np.linspace(-10., 10., 16)
    held = control.update(now=.02, hold_effort=effort, **args)
    assert control.info["ascent_stop_hold_phase"] == "hold"
    np.testing.assert_allclose(held[:12]*deploy.ACTION_SCALE[:12]+deploy.DEFAULT_JOINT_POS[:12],
                               joints[:12]+effort[:12]/80.)
    np.testing.assert_array_equal(control.update(now=.02, **args), held)
    # A positive displacement must drive the wheels in the restoring direction.
    shifted = dict(args, position=np.array([.02, 0.]))
    result = control.update(now=.04, **shifted)
    assert np.all(result[12:] > 0.), result
    # Forward handover fades monotonically and finishes with exact policy output.
    moving = dict(args, requested=np.array([.5, 0., 0.]))
    blends = []
    for now in (.06, .08, .12, .16, .20):
        result = control.update(now=now, **moving)
        blends.append(control.info["ascent_stop_hold_blend"])
    assert np.all(np.diff(blends) <= 0.) and blends[0] == 1. and blends[-1] == 0.
    np.testing.assert_array_equal(result, action)
    for command in ([-.5, 0., 0.], [0., .3, 0.], [0., 0., .4], [.05, 0., 0.]):
        control.reset()
        control.update(now=1., hold_effort=effort, **args)
        result = control.update(now=1.02, **dict(args, requested=np.array(command)))
        np.testing.assert_array_equal(result, action)
        assert not control.active
    control.update(now=2., hold_effort=effort, **args)
    control.update(now=.1, **dict(args, position=np.array([1., 0.])))
    np.testing.assert_array_equal(control.target_xy, [1., 0.])
    assert not control.legs_latched  # Time rollback cannot reuse the old stance.
    control.reset()
    assert not control.active and control.last_time is None
    for override in (dict(eligible=False),):
        np.testing.assert_array_equal(control.update(now=3., **dict(args, **override)), action)
    control.enabled = False
    np.testing.assert_array_equal(control.update(now=4., **args), action)
    return "passed: brake/hold/release, wheel sign, manual bypass, repeated tick, reset/time rollback"


def passthrough_checks():
    action = np.linspace(-.3, .3, 16, dtype=np.float32)
    for kind in ("flat", "descent", "ascent"):
        options = NS(terrain=kind, terrain_xml=None)
        model = deploy.load_model(ROOT / "deep_robotics_model/M20/mjcf/M20.xml", options)
        contract = deploy.M20Contract(model, options.terrain_config)
        data = deploy.mujoco.MjData(model)
        contract.reset(data)
        heading = deploy.make_heading_hold(contract, NS(uses_merged_ascent=True), legacy=True)
        if kind != "ascent":
            result, info = deploy.apply_ascent_stop_hold(contract, data, np.zeros(3), action, {})
            np.testing.assert_array_equal(result, action)
            assert not info["ascent_stop_hold_active"]
        # Every moving tick outside an existing hold is an exact action bypass.
        for command in ([.5, 0., 0.], [-.5, 0., 0.], [0., .2, .3]):
            result, _ = deploy.apply_ascent_stop_hold(contract, data, np.array(command), action, {})
            np.testing.assert_array_equal(result, action)
        if kind == "ascent":
            # Within an ascent scene, approaching flat ground is also excluded.
            result, info = deploy.apply_ascent_stop_hold(contract, data, np.zeros(3), action, {})
            np.testing.assert_array_equal(result, action)
            assert not info["ascent_stop_hold_active"]
            assert deploy.ascent_static_support_effort(contract, data, np.zeros(4, dtype=bool)) is None
            deploy.make_heading_hold(contract, NS(uses_merged_ascent=False), legacy=True)
            result, info = deploy.apply_ascent_stop_hold(contract, data, np.zeros(3), action, {})
            np.testing.assert_array_equal(result, action)
            assert not info["ascent_stop_hold_active"]
            deploy.make_heading_hold(contract, NS(uses_merged_ascent=True), legacy=True,
                                     ascent_stop_enabled=False)
            result, _ = deploy.apply_ascent_stop_hold(contract, data, np.zeros(3), action, {})
            np.testing.assert_array_equal(result, action)
            heading.reset()
    return "passed: real MJCF flat/descent/moving/unsupported actor/disabled bypass"


def telemetry_check():
    options = NS(terrain="flat", terrain_xml=None)
    model = deploy.load_model(ROOT / "deep_robotics_model/M20/mjcf/M20.xml", options)
    contract = deploy.M20Contract(model, options.terrain_config)
    data = deploy.mujoco.MjData(model)
    contract.reset(data)
    heading = deploy.make_heading_hold(contract, NS(uses_merged_ascent=True), legacy=True)
    command, info = heading.update(data, np.zeros(3))
    action, info = deploy.apply_ascent_stop_hold(contract, data, np.zeros(3), np.zeros(16), info)
    sample = deploy.StairTelemetry(contract).sample(data, action)
    sample.update(time_s=0., policy_step=0, command=np.zeros(3), effective_command=command,
                  heading_hold=info, base_xyz=data.xpos[contract.base_id].copy())
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        # Export hashes an arbitrary fixture file; no model is read or loaded.
        fixture = directory / "metadata_fixture.txt"
        fixture.write_text("telemetry export fixture")
        options.checkpoint = fixture
        options.model = ROOT / "deep_robotics_model/M20/mjcf/M20.xml"
        options.output = directory / "test.npz"
        options.disable_heading_hold = False
        csv_path, md_path, _ = deploy.export_stair_telemetry([sample], options)
        with csv_path.open() as stream:
            row = next(csv.DictReader(stream))
        assert row["ascent_stop_hold_phase"] == "inactive"
        assert row["ascent_stop_hold_held_legs"] == "0"
        assert deploy.ASCENT_STOP_VERSION in md_path.read_text()
    return "passed: independent ascent hold CSV fields and Markdown metadata"


def quaternion(row):
    r, p, y = [math.radians(float(row[k+"_deg"]))/2 for k in ("roll", "pitch", "yaw")]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]


def physics_replay(prefix):
    with prefix.with_name(prefix.name+"_telemetry.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    recorded = np.load(prefix.with_suffix(".npz"), allow_pickle=False)
    times = np.array([float(row["time_s"]) for row in rows])
    options = NS(terrain="ascent", terrain_xml=None, stair_height=.15, tread_depth=.30,
                 course_width=1.2, stair_start=.25, stair_count=20)
    model = deploy.load_model(ROOT / "deep_robotics_model/M20/mjcf/M20.xml", options)
    results = []
    # Four representative snapshots, eight neighboring snapshots, and two long holds.
    cases = [(t, 10.) for t in (12.38, 12.56, 81.52, 81.8, 12.46, 12.50, 12.70,
                                13., 81.60, 81.70, 82., 82.5)] + [(12.56, 30.), (81.8, 30.)]
    for timestamp, duration in cases:
        row = rows[int(np.argmin(abs(times-timestamp)))]
        contract = deploy.M20Contract(model, options.terrain_config)
        data = deploy.mujoco.MjData(model)
        contract.reset(data)
        q, v = int(contract.root_joint.qposadr[0]), int(contract.root_joint.dofadr[0])
        data.qpos[q:q+7] = [float(row["base_"+axis+"_m"]) for axis in "xyz"]+quaternion(row)
        for i, name in enumerate(deploy.JOINT_NAMES[:12]):
            stem = name[:-6]
            data.qpos[contract.qpos_ids[i]] = float(row[stem+"_angle_rad"])
            data.qvel[contract.qvel_ids[i]] = float(row[stem+"_velocity_rad_s"])
            data.ctrl[contract.actuator_ids[i]] = float(row[stem+"_torque_Nm"])
        for i, name in enumerate(("FL", "FR", "HL", "HR")):
            data.qvel[contract.qvel_ids[12+i]] = float(row[name+"_wheel_linear_speed_m_s"])/deploy.WHEEL_RADIUS
            data.ctrl[contract.actuator_ids[12+i]] = float(row[name+"_motor_torque_Nm"])
        data.qvel[v:v+3] = [float(row["base_velocity_"+axis+"_world_m_s"]) for axis in "xyz"]
        # Original CSV has no roll/pitch angular rates or wheel angles.
        data.qvel[v+3:v+6] = [0., 0., float(row["body_yaw_rate_rad_s"])]
        data.time = float(row["time_s"])
        deploy.mujoco.mj_forward(model, data)
        deploy.make_heading_hold(contract, NS(uses_merged_ascent=True), legacy=True)
        initial = data.xpos[contract.base_id].copy()
        positions, hold_frames, first_hold = [], 0, None
        peak_leg = peak_wheel = 0.
        for step in range(round(duration/deploy.POLICY_DT)):
            index = int(np.argmin(abs(recorded["time"]-data.time)))
            action, info = deploy.apply_ascent_stop_hold(
                contract, data, np.zeros(3), recorded["action"][index].copy(), {})
            if info["ascent_stop_hold_phase"] == "hold":
                hold_frames += 1
                if first_hold is None:
                    first_hold = step*deploy.POLICY_DT
            for _ in range(round(deploy.POLICY_DT/deploy.PHYSICS_DT)):
                contract.apply_pd(data, action)
                peak_leg = max(peak_leg, float(abs(data.ctrl[contract.actuator_ids[:12]]).max()))
                peak_wheel = max(peak_wheel, float(abs(data.ctrl[contract.actuator_ids[12:]]).max()))
                deploy.mujoco.mj_step(model, data)
            positions.append(data.xpos[contract.base_id].copy())
        positions = np.array(positions)
        delta = positions[-1]-initial
        tail_span = np.ptp(positions[-100:], axis=0)
        result = dict(snapshot_time_s=timestamp, duration_s=duration,
                      final_delta_xyz_m=delta.tolist(), last_2s_span_xyz_m=tail_span.tolist(),
                      first_hold_after_s=first_hold, hold_fraction=hold_frames/len(positions),
                      peak_leg_torque_Nm=peak_leg, peak_wheel_torque_Nm=peak_wheel)
        results.append(result)
        print(json.dumps(result), flush=True)
        assert first_hold is not None and first_hold < .8, result
        assert hold_frames/len(positions) > .90, result
        assert abs(delta[0]) < .12 and abs(delta[1]) < .10, result
        assert positions[:, 2].min() > initial[2]-.06, result
        assert np.linalg.norm(tail_span[:2]) < .04, result
        assert peak_leg <= deploy.LEG_EFFORT_LIMIT+1.e-6, result
        assert peak_wheel <= deploy.WHEEL_EFFORT_LIMIT+1.e-6, result
    recorded.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-prefix", type=Path,
                        help="Original ascent_yaw_telemetry stem; assumes original .15/.30/1.2 m terrain")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = dict(controller_version=deploy.ASCENT_STOP_VERSION,
                  state_machine=state_checks(), passthrough=passthrough_checks(),
                  telemetry=telemetry_check(),
                  validation_scope="No checkpoint loaded; approximate CSV state reconstruction and open-loop action replay")
    if args.telemetry_prefix:
        result["physics_replay"] = physics_replay(args.telemetry_prefix)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    print("PASS", result["state_machine"], result["passthrough"], flush=True)


if __name__ == "__main__":
    main()
