#!/usr/bin/env python3
"""Collect solved-contact M20 ascent data without changing the deployment policy.

Run the companion analyze_m20_gait_reference.py on the resulting manifest.
All lengths/velocities are SI; wheel order is FL, FR, HL, HR. No training.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_mujoco as d

LEGS = ("FL", "FR", "HL", "HR")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact_measurements(model, data, contract, qvel, sample_index, time_s):
    """Use contact geometry and pre-integration qvel from the same mj_step.

    mj_step solves contacts at t and then integrates qpos/qvel to t+dt. Its
    xpos/contact/Jacobian fields still describe t. Do not combine those fields
    with the integrated qvel or call an extra forward solve in the rollout.
    """
    wheel_index = {int(b): i for i, b in enumerate(contract.wheel_body_ids)}
    forces = np.zeros((4, 3))
    top_force, face_force = np.zeros(4), np.zeros(4)
    slip_sum, slip_max = np.zeros(4), np.zeros(4)
    slip_xy = np.zeros((4, 2))
    level_weight = [dict() for _ in LEGS]
    self_peak = nonwheel_peak = 0.0
    rows = []
    jac = np.zeros((3, model.nv))
    height = contract.terrain.parameters["step_height"][0]
    for ci, contact in enumerate(data.contact):
        b1, b2 = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, ci, wrench)
        magnitude = float(np.linalg.norm(wrench[:3]))
        if b1 and b2:
            self_peak = max(self_peak, magnitude)
            continue
        body = b2 if b1 == 0 else b1
        if body not in wheel_index:
            if body:
                nonwheel_peak = max(nonwheel_peak, magnitude)
            continue
        if wrench[0] <= 1.0:  # solved active contact, not a geometric proximity
            continue
        i = wheel_index[body]
        sign = 1.0 if b2 == body else -1.0
        frame = contact.frame.reshape(3, 3)
        normal = sign * frame[0]
        force = sign * (frame.T @ wrench[:3])
        forces[i] += force
        mujoco.mj_jac(model, data, jac, None, contact.pos, body)
        velocity = jac @ qvel  # ground is static worldbody in this terrain
        tangent = velocity - normal * np.dot(normal, velocity)
        slip = float(np.linalg.norm(tangent))
        is_top = bool(normal[2] >= 0.7)
        level = int(round(float(contact.pos[2]) / height)) if is_top else -1
        if is_top:
            top_force[i] += wrench[0]
            slip_sum[i] += wrench[0] * slip
            slip_max[i] = max(slip_max[i], slip)
            slip_xy[i] += wrench[0] * tangent[:2]
            level_weight[i][level] = level_weight[i].get(level, 0.0) + wrench[0]
        else:
            face_force[i] += wrench[0]
        rows.append(dict(sample_index=sample_index, time_s=time_s, wheel=LEGS[i],
                         contact_id=ci, x_m=contact.pos[0], y_m=contact.pos[1], z_m=contact.pos[2],
                         normal_x=normal[0], normal_y=normal[1], normal_z=normal[2],
                         normal_force_N=wrench[0], force_x_N=force[0], force_y_N=force[1], force_z_N=force[2],
                         tangent_vx_m_s=tangent[0], tangent_vy_m_s=tangent[1], tangent_vz_m_s=tangent[2],
                         slip_m_s=slip, tread_top=int(is_top), tread_id=level,
                         friction_mu=contact.friction[0]))
    levels = np.array([max(weights, key=weights.get) if weights else -1 for weights in level_weight])
    norm = np.linalg.norm(forces, axis=1)
    support = (top_force > 5.0) & (forces[:, 2] > 5.0) & (forces[:, 2] > .5 * norm)
    levels[~support] = -1
    return dict(contact_force=forces, top_normal_force=top_force, face_normal_force=face_force,
                support=support, support_tread=levels,
                slip=np.divide(slip_sum, top_force, out=np.zeros(4), where=top_force > 0),
                slip_max=slip_max,
                slip_xy=np.divide(slip_xy, top_force[:, None], out=np.zeros((4, 2)), where=top_force[:, None] > 0),
                self_contact_peak_force=self_peak, nonwheel_contact_peak_force=nonwheel_peak), rows


def run_case(actor, args, speed, offset):
    name = f"vx{speed:.2f}_dx{offset:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    folder = args.output_dir / name
    folder.mkdir(parents=True, exist_ok=False)
    cfg = SimpleNamespace(terrain=None, terrain_xml=args.terrain_xml,
                          stair_height=args.stair_height, tread_depth=args.tread_depth,
                          stair_count=args.stair_count, stair_start=None)
    model = d.load_model(args.model, cfg)
    contract = d.M20Contract(model, cfg.terrain_config)
    if contract.terrain.kind not in {"ascent", "descent"}:
        raise ValueError("Reference acquisition requires a staircase")
    data = mujoco.MjData(model)
    contract.reset(data)
    data.qpos[int(contract.root_joint.qposadr[0])] += offset
    mujoco.mj_forward(model, data)
    legacy = d.use_checkpoint_control_profile(SimpleNamespace(legacy_heading_hold=False), actor)
    hold = d.make_heading_hold(contract, actor, enabled=True, legacy=legacy)
    records, policies = {}, {}
    previous = np.zeros(d.ACTION_DIM, dtype=np.float32)
    jac = np.zeros((3, model.nv))
    run_status = "duration_limit"
    contact_path = folder / "contacts.csv.gz"
    with gzip.open(contact_path, "wt", newline="") as stream:
        writer = None
        for step in range(int(np.ceil(args.duration / d.POLICY_DT))):
            requested = np.array([speed if data.time >= args.settle - 1.e-8 else 0., 0., 0.], np.float32)
            effective, info = hold.update(data, requested)
            with torch.inference_mode():
                obs = d.actor_inputs(contract, data, effective, previous, actor, requested)
                action = actor(obs).squeeze(0).cpu().numpy()
                bounded = (torch.tanh(actor.residual_mlp(torch.cat((obs["policy"], obs["motion_control"]), -1)))
                           .squeeze(0).cpu().numpy()) if getattr(actor, "uses_motion_control", False) else np.zeros(16)
                entry_bounded = (torch.tanh(actor.entry_mlp(torch.cat((obs['policy'],obs['motion_control']),-1)))
                                 .squeeze(0).cpu().numpy()) if hasattr(actor,'entry_mlp') else np.zeros(16)
                landing_telemetry = ({'landing_bounded': actor._landing_bounded[0].cpu().numpy(),
                                      'landing_gate': float(actor._landing_gate[0])}
                                     if getattr(actor, 'uses_landing', False) else {})
                if getattr(actor, 'uses_merged_ascent', False):
                    landing_telemetry.update(unified_bounded=actor._unified_bounded[0].cpu().numpy(),
                                             unified_gate=float(actor._unified_gate[0]))
                if getattr(actor, 'uses_path_control', False) or getattr(actor, 'uses_path_feedback', False):
                    landing_telemetry.update(path_control=contract._path_control[0].cpu().numpy(),
                                             descent_gate=float(actor._descent_gate[0]))
                if getattr(actor, 'uses_sequence', False):
                    landing_telemetry.update(sequence_bounded=actor._sequence_bounded[0].cpu().numpy(),
                                             sequence_gate=float(actor._sequence_gate[0]))
            if action.shape != (16,) or not np.isfinite(action).all():
                raise RuntimeError(f"Nonfinite actor output in {name}")
            action = np.clip(action, -100., 100.)
            policy = dict(time=float(data.time), observation=obs["policy"][0].numpy(),
                          requested_command=requested.copy(), effective_command=effective.copy(),
                          previous_action=previous.copy(), action=action.copy(), residual_bounded=bounded.copy(),
                          entry_bounded=entry_bounded.copy(),
                          heading_error_deg=info["error_deg"], **landing_telemetry)
            for key, value in policy.items():
                policies.setdefault(key, []).append(value)
            for _ in range(round(d.POLICY_DT / d.PHYSICS_DT)):
                t = float(data.time)
                qpos, qvel = data.qpos.copy(), data.qvel.copy()
                contract.apply_pd(data, action)
                mujoco.mj_step(model, data)
                index = len(records.get("time", []))
                measurement, contacts = contact_measurements(model, data, contract, qvel, index, t)
                if contacts:
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(contacts[0]))
                        writer.writeheader()
                    writer.writerows(contacts)
                rotation = data.xmat[contract.base_id].reshape(3, 3)
                wheel_pos = data.xpos[contract.wheel_body_ids].copy()
                wheel_vel = []
                for body in contract.wheel_body_ids:
                    mujoco.mj_jac(model, data, jac, None, data.xpos[body], int(body))
                    wheel_vel.append(jac @ qvel)
                mujoco.mj_jac(model, data, jac, None, data.xpos[contract.base_id], contract.base_id)
                base_velocity = jac @ qvel
                ground = np.array([contract.ground_height(data, *point[:2]) for point in wheel_pos])
                rpy = [np.arctan2(rotation[2, 1], rotation[2, 2]),
                       np.arctan2(-rotation[2, 0], np.linalg.norm(rotation[2, 1:])),
                       np.arctan2(rotation[1, 0], rotation[0, 0])]
                row = dict(time=t, policy_step=step, qpos=qpos, qvel=qvel,
                           root_pos=data.xpos[contract.base_id].copy(), root_quat=data.xquat[contract.base_id].copy(),
                           base_velocity=base_velocity, rpy=rpy,
                           joint_pos=qpos[contract.qpos_ids], joint_vel=qvel[contract.qvel_ids],
                           actuator_force=data.actuator_force[contract.actuator_ids].copy(),
                           wheel_pos=wheel_pos, wheel_velocity=wheel_vel,
                           wheel_from_hip_body=(wheel_pos - data.xpos[contract.hip_body_ids]) @ rotation,
                           wheel_ground=ground, wheel_clearance=wheel_pos[:, 2] - ground - d.WHEEL_RADIUS,
                           requested_command=requested.copy(), effective_command=effective.copy(), action=action.copy(),
                           residual_bounded=bounded.copy(), entry_bounded=entry_bounded.copy(),
                           **landing_telemetry, **measurement)
                for key, value in row.items():
                    records.setdefault(key, []).append(value)
            previous = action.astype(np.float32)
            pos = data.xpos[contract.base_id]
            ground_base = contract.ground_height(data, *pos[:2])
            if pos[2] < ground_base + .22 or rotation[2, 2] < .35:
                run_status = "fell"
                break
            if np.max(np.abs(wheel_pos[:, 1])) + d.WHEEL_RADIUS >= contract.terrain.course_width / 2:
                run_status = "left_course"
                break
            if np.all(wheel_pos[:, 0] > contract.terrain.goal_x + getattr(args, "exit_distance", .4)):
                run_status = "cleared_stairs"
                break
    np.savez_compressed(folder / "physics.npz", **{key: np.asarray(value) for key, value in records.items()})
    np.savez_compressed(folder / "policy.npz", **{key: np.asarray(value) for key, value in policies.items()})
    summary = dict(case=name, speed_m_s=speed, spawn_x_offset_m=offset, status=run_status,
                   settle_s=args.settle, requested_duration_s=args.duration, duration_s=float(data.time),
                   physics_samples=len(records["time"]), policy_samples=len(policies["time"]),
                   final_xyz=pos.tolist(), control_profile=("legacy_v1_path_v1" if (getattr(actor, "uses_path_control", False) or getattr(actor, "uses_path_feedback", False)) else "legacy_v1" if legacy else "filtered_v4"),
                   terrain=cfg.terrain_config.summary(), model_nv=model.nv, model_nq=model.nq,
                   final_qpos=data.qpos.tolist(), final_qvel=data.qvel.tolist())
    (folder / "case.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("case", "status", "duration_s", "physics_samples", "final_xyz")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pre_train/teacher_baseline_.pt")
    parser.add_argument("--model", type=Path, default=ROOT / "deep_robotics_model/M20/mjcf/M20.xml")
    parser.add_argument("--terrain-xml", type=Path, default=ROOT / "deploy/deploy_mujoco/terrains/stairs_ascent.xml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speeds", type=float, nargs="+", default=[.5, .7])
    parser.add_argument("--spawn-x-offsets", type=float, nargs="+", default=[-.1, 0., .1])
    parser.add_argument("--settle", type=float, default=6.)
    parser.add_argument("--duration", type=float, default=32.)
    parser.add_argument("--stair-height", type=float)
    parser.add_argument("--tread-depth", type=float)
    parser.add_argument("--stair-count", type=int)
    args = parser.parse_args()
    if (not np.isfinite([*args.speeds, *args.spawn_x_offsets, args.settle, args.duration]).all()
            or min(args.speeds) <= .1 or args.settle < 0 or args.duration <= args.settle
            or any(value is not None and (not np.isfinite(value) or value <= 0)
                   for value in (args.stair_height, args.tread_depth, args.stair_count))):
        parser.error("Use finite positive speeds >0.1, nonnegative settle, duration > settle and positive terrain sizes")
    if len(set(args.speeds)) != len(args.speeds) or len(set(args.spawn_x_offsets)) != len(args.spawn_x_offsets):
        parser.error("Duplicate scenarios are not independent data")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose an empty output directory; existing acquisition data/manifest will not be overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    actor = d.load_actor(args.checkpoint)
    manifest = dict(version="m20_gait_reference_1", checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_sha256=digest(args.checkpoint), checkpoint_manifest=actor.m20_training_manifest,
                    argv=sys.argv, physics_dt_s=d.PHYSICS_DT, policy_dt_s=d.POLICY_DT,
                    mujoco_version=mujoco.__version__, torch_version=torch.__version__,
                    wheel_order=list(LEGS), joint_order=list(d.JOINT_NAMES),
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                                   (Path(__file__).resolve(), ROOT / "deploy/deploy_mujoco/deploy_mujoco.py",
                                    ROOT / "deploy/deploy_mujoco/terrain_loader.py", args.model.resolve(),
                                    args.terrain_xml.resolve(), ROOT / "source/rl_training/rl_training/models/motion_residual_actor.py")},
                    measurement_alignment="pre-integration time/qpos/qvel paired with solved contact geometry/forces; no extra forward solve",
                    cases=[])
    path = args.output_dir / "manifest.json"
    if getattr(actor,'uses_ascent_entry',False):
        source=ROOT/'source/rl_training/rl_training/models/ascent_entry_actor.py'
        manifest['source_sha256'][str(source.relative_to(ROOT))]=digest(source)
    if getattr(actor, 'uses_sequence', False):
        for name in ('models/ascent_sequence_actor.py', 'algorithms/initialization/ascent_sequence_init.py', 'models/ascent_landing_actor.py', 'algorithms/initialization/ascent_landing_init.py'):
            source = ROOT/'source/rl_training/rl_training'/name
            manifest['source_sha256'][str(source.relative_to(ROOT))] = digest(source)
    if getattr(actor, 'uses_adaptive_ascent', False):
        source = ROOT/'source/rl_training/rl_training/models/ascent_adaptive_actor.py'
        manifest['source_sha256'][str(source.relative_to(ROOT))] = digest(source)
    for name in manifest["source_sha256"]:
        target = args.output_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    for speed in args.speeds:
        for offset in args.spawn_x_offsets:
            manifest["cases"].append(run_case(actor, args, speed, offset))
            path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
