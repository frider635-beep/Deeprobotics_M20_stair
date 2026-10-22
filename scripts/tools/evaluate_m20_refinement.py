#!/usr/bin/env python3
"""Bounded, deterministic M20 checkpoint comparisons with the deployment code.

Run from any directory with the deploy Python environment. This evaluates a
finite set of scenarios, not collision freedom over all initial conditions.
--steps caps each case for a pipeline smoke check; incomplete cases cannot pass
the optional acceptance gate. Neither checkpoints nor training state are edited.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
spec = importlib.util.spec_from_file_location("m20_evaluation_deploy", ROOT / "deploy/deploy_mujoco/deploy_mujoco.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

CASES = {
    "flat_zero": ("flat", [(6., 0.)]),
    "flat_move_stop": ("flat", [(6., 0.), (3., .5), (6., 0.)]),
    "flat_reverse_stop": ("flat", [(6., 0.), (3., .5), (6., 0.), (3., -.5), (6., 0.)]),
    "ascent_hold": ("ascent", [(6., 0.), (7., .7), (6., 0.)]),
    "descent_hold": ("descent", [(6., 0.), (11.5, .7), (6., 0.)]),
    "descent_first_tread_pause": ("descent", None),
}


def aggregate(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"samples": 0, "mean": None, "p95": None, "max": None}
    return {"samples": int(values.size), "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def interleg_distances(data, contract):
    wheels = data.xpos[contract.wheel_body_ids]
    knees = data.xpos[contract.front_knee_body_ids]
    segment = wheels[:2] - knees
    projection = np.sum((wheels[2:] - knees) * segment, axis=1) / np.maximum(np.sum(segment**2, axis=1), 1.e-9)
    nearest = knees + np.clip(projection, 0, 1)[:, None] * segment
    return np.linalg.norm(wheels[:2] - wheels[2:], axis=1), np.linalg.norm(wheels[2:] - nearest, axis=1)


def run_case(actor, name, cli):
    terrain, timed = CASES[name]
    terrain_file = cli.terrain_dir / {"flat": "flat.xml", "ascent": "stairs_ascent.xml",
                                      "descent": "stairs_descent.xml"}[terrain]
    args = SimpleNamespace(terrain=None, terrain_xml=terrain_file)
    model = d.load_model(cli.model, args)
    contract = d.M20Contract(model, args.terrain_config)
    data = d.mujoco.MjData(model)
    contract.reset(data)
    yaw_offset = math.radians(getattr(cli, "spawn_yaw_offset_deg", 0.))
    if yaw_offset:
        root_q = int(contract.root_joint.qposadr[0]) + 3
        w, x, y, z = data.qpos[root_q:root_q+4].copy()
        c, s = math.cos(yaw_offset/2), math.sin(yaw_offset/2)
        data.qpos[root_q:root_q+4] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
        d.mujoco.mj_forward(model, data)
    hold = d.make_heading_hold(contract, actor, enabled=not cli.disable_heading_hold,
                                   legacy=d.use_checkpoint_control_profile(cli, actor))
    telemetry = d.StairTelemetry(contract)
    rotation = data.xmat[contract.base_id].reshape(3, 3)
    nominal_forward = ((data.xpos[contract.wheel_body_ids] - data.xpos[contract.hip_body_ids]) @ rotation)[:2, 0].copy()
    initial_levels = None
    rows, events = [], []
    previous_action = np.zeros(d.ACTION_DIM, dtype=np.float32)
    previous_request = np.zeros(3, dtype=np.float32)
    min_wheels, min_shanks = np.full(2, np.inf), np.full(2, np.inf)
    phase, phase_start, stable = "initial", 0., 0
    trigger = None
    expected_duration = sum(duration for duration, _ in timed) if timed else 32.
    max_steps = math.ceil(expected_duration / d.POLICY_DT)
    if cli.steps is not None:
        max_steps = min(max_steps, cli.steps)
    fell = False
    completed = False
    segments = [{"duration_s": duration, "command": [speed, 0, 0]} for duration, speed in timed] if timed else None
    for step in range(max_steps):
        now = float(data.time)
        if timed:
            command = d.command_profile_at(segments, step * d.POLICY_DT)
            phase = "zero" if np.linalg.norm(command) <= .1 else "moving"
        else:
            if phase == "initial" and now >= 6. - 1.e-8:
                phase, phase_start = "approach", now
                events.append({"time_s": now, "phase": phase})
            elif phase == "pause" and now - phase_start >= 2. - 1.e-8:
                phase, phase_start = "resume", now
                events.append({"time_s": now, "phase": phase})
            elif phase == "resume" and now - phase_start >= 11.5 - 1.e-8:
                phase, phase_start = "final", now
                events.append({"time_s": now, "phase": phase})
            elif phase == "final" and now - phase_start >= 6. - 1.e-8:
                completed = True
                break
            command = np.array([.7 if phase in ("approach", "resume") else 0, 0, 0], dtype=np.float32)
        effective, info = hold.update(data, command)
        with torch.inference_mode():
            observation = d.actor_inputs(contract, data, effective, previous_action, actor, command)
            action = actor(observation).squeeze(0).cpu().numpy()
        if action.shape != (d.ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid actor output in {name}, step {step}")
        action = np.clip(action, -100., 100.)
        for _ in range(round(d.POLICY_DT / d.PHYSICS_DT)):
            contract.apply_pd(data, action)
            d.mujoco.mj_step(model, data)
            telemetry.observe_physics_step(data)
            wheels, shanks = interleg_distances(data, contract)
            min_wheels, min_shanks = np.minimum(min_wheels, wheels), np.minimum(min_shanks, shanks)
        sample = telemetry.sample(data, action)
        sample.update(time_s=float(data.time), command=command.copy(), effective_command=effective.copy(),
                      previous_command=previous_request.copy(),
                      heading_hold=info, base_xyz=data.xpos[contract.base_id].copy(), phase=phase)
        previous_request = command.copy()
        rows.append(sample)
        if initial_levels is None:
            initial_levels = np.asarray(sample["wheel_levels"]).copy()
        if timed is None and phase == "approach":
            levels = np.asarray(sample["wheel_levels"])
            ready = bool(np.all(levels[:2] == initial_levels[:2] - 1) and np.all(levels[2:] == initial_levels[2:])
                         and np.all(sample["wheel_forces"][:, 2] > 5) and np.all(sample["wheel_contact_counts"] > 0))
            stable = stable + 1 if ready else 0
            if stable >= 3:
                phase, phase_start = "pause", float(data.time)
                trigger = phase_start
                events.append({"time_s": trigger, "phase": phase})
        position = data.xpos[contract.base_id]
        ground = contract.ground_height(data, float(position[0]), float(position[1]))
        upright = data.xmat[contract.base_id].reshape(3, 3)[2, 2] > .35
        fell = bool(position[2] < ground + .22 or not upright)
        previous_action = action.astype(np.float32)
        if fell:
            break
    if timed and len(rows) == math.ceil(expected_duration / d.POLICY_DT):
        completed = True
    zero_phases, group = [], []
    def finish_zero():
        if not group:
            return
        clean = []
        for sample in group:
            if sample["time_s"] - group[0]["time_s"] < .4 - 1.e-8:
                continue
            if sample["near_lateral_edge"] or sample["nonwheel_ground_contacts"]:
                break
            clean.append(sample)
        path = float(sum(np.linalg.norm(b["base_xyz"][:2] - a["base_xyz"][:2]) for a, b in zip(clean, clean[1:])))
        drift = [float(np.linalg.norm(x["base_xyz"][:2] - clean[0]["base_xyz"][:2])) for x in clean]
        stair_samples = 0
        if terrain != "flat":
            start_x = args.terrain_config.parameters["start_x"][0]
            stair_samples = sum(bool(np.any((x["wheel_positions_world_m"][:, 0] > start_x + .05)
                                       & (x["wheel_positions_world_m"][:, 0] < args.terrain_config.goal_x - .05))) for x in clean)
        quiet_run, max_quiet = 0., 0.
        for sample in clean:
            quiet = (sample["base_speed_m_s"] < .02 and np.max(np.abs(sample["wheel_linear_speeds"])) < .03
                     and abs(sample["yaw_rate_rad_s"]) < .03
                     and np.sqrt(np.mean(sample["joint_velocities"][:12]**2)) < .1
                     and np.max(np.abs(sample["joint_velocities"][:12])) < .2
                     and sample["leg_target_delta_rms_rad"] < .002
                     and sample["leg_target_second_difference_rms_rad"] < .003
                     and sample["wheel_target_delta_rms_m_s"] < .010
                     and sample["wheel_target_second_difference_rms_m_s"] < .015)
            quiet_run = quiet_run + d.POLICY_DT if quiet else 0.
            max_quiet = max(max_quiet, quiet_run)
        zero_phases.append({"start_s": group[0]["time_s"], "clean_samples_after_brake": len(clean),
            "total_samples_after_brake": sum(x['time_s']-group[0]['time_s'] >= .4-1.e-8 for x in group),
            "previous_command": group[0]["previous_command"].tolist(),
            "stair_samples_after_brake": stair_samples, "max_quiet_duration_s": max_quiet,
            "max_position_drift_m": max(drift) if drift else None,
            "leg_speed_rms_rad_s": aggregate([np.sqrt(np.mean(x["joint_velocities"][:12]**2)) for x in clean]),
            "base_speed_m_s": aggregate([x["base_speed_m_s"] for x in clean]),
            "max_wheel_speed_m_s": aggregate([np.max(np.abs(x["wheel_linear_speeds"])) for x in clean]),
            "abs_yaw_rate_rad_s": aggregate([abs(x["yaw_rate_rad_s"]) for x in clean]),
            "path_m": path,
            "heading_change_deg": math.degrees(d.wrap_angle(math.radians(clean[-1]["yaw_deg"] - clean[0]["yaw_deg"]))) if clean else None,
            "leg_action_second_difference_rms": aggregate([x["leg_action_second_difference_rms"] for x in clean]),
            "wheel_action_second_difference_rms": aggregate([x["wheel_action_second_difference_rms"] for x in clean]),
            "leg_target_delta_rms_rad": aggregate([x["leg_target_delta_rms_rad"] for x in clean]),
            "leg_target_second_difference_rms_rad": aggregate([x["leg_target_second_difference_rms_rad"] for x in clean]),
            "wheel_target_delta_rms_m_s": aggregate([x["wheel_target_delta_rms_m_s"] for x in clean]),
            "wheel_target_second_difference_rms_m_s": aggregate([x["wheel_target_second_difference_rms_m_s"] for x in clean])})
    for row in rows:
        if np.linalg.norm(row["command"]) <= .1:
            group.append(row)
        else:
            finish_zero()
            group = []
    finish_zero()
    motion = [row for row in rows if row["command"][0] > .1]
    # Heading alone misses side-slip. Fix a line per requested motion segment,
    # and retain progress so refusing to move cannot pass as perfect tracking.
    route_errors, route_progress, route_origin, was_moving = [], [], None, False
    moving_segments = []
    for row in rows:
        moving = abs(row["command"][0]) > .1
        if moving:
            direction = float(np.sign(row["command"][0]))
            if not was_moving or direction != moving_segments[-1]["direction"]:
                route_origin = row["base_xyz"][:2].copy()
                target = math.radians(row["heading_hold"]["target_yaw_deg"])
                route_forward = np.array([math.cos(target), math.sin(target)])
                route_normal = np.array([-math.sin(target), math.cos(target)])
                moving_segments.append({"direction": direction, "end_progress_m": 0.})
            delta = row["base_xyz"][:2] - route_origin
            route_errors.append(abs(float(delta @ route_normal)))
            progress = float(delta @ route_forward) * direction
            route_progress.append(progress)
            moving_segments[-1]["end_progress_m"] = progress
        was_moving = moving
    reach = {}
    for level in (1, 2, 3):
        for front in (0, 1):
            for support in (True, False):
                selected = [row for row in rows if initial_levels[front] - row["wheel_levels"][front] == level
                            and bool(row["wheel_forces"][front, 2] > 5 and row["wheel_contact_counts"][front] > 0) == support]
                reach[f"step{level}_{('FL', 'FR')[front]}_{'support' if support else 'swing'}"] = {
                    "hip_wheel_forward_m": aggregate([row["hip_wheel_forward"][front] for row in selected]),
                    "nominal_excess_m": aggregate([max(0., row["hip_wheel_forward"][front] - nominal_forward[front]) for row in selected])}
    metrics = {"case": name, "terrain": args.terrain_config.summary(),
        "terrain_xml_sha256": hashlib.sha256(terrain_file.read_bytes()).hexdigest(),
        "samples": len(rows), "complete_profile": completed, "fell": fell,
        "controller": "disabled" if cli.disable_heading_hold else "legacy_v1" if d.use_checkpoint_control_profile(cli, actor) else "filtered_v4",
        "spawn_yaw_offset_deg": getattr(cli, "spawn_yaw_offset_deg", 0.),
        "moving_route_error_m": aggregate(route_errors),
        "moving_route_progress_m": aggregate(route_progress),
        "moving_segments": moving_segments,
        "moving_end_x_m": float(motion[-1]["base_xyz"][0]) if motion else None,
        "first_tread_pause_trigger_s": trigger, "events": events,
        "min_wheel_center_distances_m": min_wheels.tolist(),
        "min_rear_wheel_front_shank_distances_m": min_shanks.tolist(),
        "physics_contact_substeps": {key: sum(row[key] for row in rows) for key in
            ("self_contact_physics_substeps", "rear_front_wheel_contact_physics_substeps", "rear_wheel_front_leg_contact_physics_substeps")},
        "rear_front_peak_contact_force_N": max((row["rear_front_peak_contact_force_N"] for row in rows), default=0.),
        "rear_front_contact_impulse_Ns": sum(row["rear_front_contact_impulse_Ns"] for row in rows),
        "zero_phases": zero_phases,
        "moving_max_right_yaw_deg": max([0.] + [-row["yaw_from_policy_start_deg"] for row in motion]),
        "moving_max_abs_yaw_deg": max([0.] + [abs(row["yaw_from_policy_start_deg"]) for row in motion]),
        "moving_end_yaw_deg": motion[-1]["yaw_from_policy_start_deg"] if motion else None,
        "near_edge_samples": sum(bool(row["near_lateral_edge"]) for row in rows),
        "nonwheel_ground_contact_samples": sum(bool(row["nonwheel_ground_contacts"]) for row in rows),
        "action_jitter": {key: aggregate([row[key] for row in rows]) for key in
            ("leg_action_delta_rms", "wheel_action_delta_rms", "leg_action_second_difference_rms", "wheel_action_second_difference_rms",
             "leg_target_delta_rms_rad", "leg_target_second_difference_rms_rad", "wheel_target_delta_rms_m_s", "wheel_target_second_difference_rms_m_s")},
        "nominal_front_forward_m": nominal_forward.tolist(),
        "first_three_descent_reach": reach if terrain == "descent" else {},
        **contract.crossing_summary(data, fell)}
    gates = {"complete_profile": completed, "no_fall": not fell,
             "no_rear_front_contact_gt1N": not any(metrics["physics_contact_substeps"][key] for key in
                 ("rear_front_wheel_contact_physics_substeps", "rear_wheel_front_leg_contact_physics_substeps")),
             "wheel_clearance_ge_028m": bool(np.min(min_wheels) >= .28),
             "shank_proxy_clearance_ge_020m": bool(np.min(min_shanks) >= .20),
             "no_lateral_edge_contamination": metrics["near_edge_samples"] == 0}
    valid_zero = [phase for phase in zero_phases if phase["clean_samples_after_brake"] >= 50]
    gates["zero_hold_samples_available"] = len(valid_zero) == len(zero_phases) and bool(valid_zero)
    gates["zero_hold_full_window_clean"] = bool(zero_phases) and all(
        phase['clean_samples_after_brake'] == phase['total_samples_after_brake'] for phase in zero_phases)
    gates["zero_speed_p95_le_002m_s"] = bool(valid_zero) and all(phase["base_speed_m_s"]["p95"] <= .02 for phase in valid_zero)
    gates["zero_wheel_p95_le_003m_s"] = bool(valid_zero) and all(phase["max_wheel_speed_m_s"]["p95"] <= .03 for phase in valid_zero)
    gates["zero_yaw_rate_p95_le_003rad_s"] = bool(valid_zero) and all(phase["abs_yaw_rate_rad_s"]["p95"] <= .03 for phase in valid_zero)
    gates["zero_path_le_005m"] = bool(valid_zero) and all(phase["path_m"] <= .05 for phase in valid_zero)
    gates["zero_heading_drift_le_2deg"] = bool(valid_zero) and all(abs(phase["heading_change_deg"]) <= 2 for phase in valid_zero)
    for field, threshold, label in (
        ("leg_target_delta_rms_rad", .002, "zero_leg_target_delta_p95"),
        ("leg_target_second_difference_rms_rad", .003, "zero_leg_target_second_difference_p95"),
        ("wheel_target_delta_rms_m_s", .010, "zero_wheel_target_delta_p95"),
        ("wheel_target_second_difference_rms_m_s", .015, "zero_wheel_target_second_difference_p95"),
    ):
        gates[label] = bool(valid_zero) and all(phase[field]["p95"] <= threshold for phase in valid_zero)
    if terrain == "ascent":
        gates.update(ascent_max_right_le_5deg=metrics["moving_max_right_yaw_deg"] <= 5,
                     ascent_max_abs_yaw_le_5deg=metrics["moving_max_abs_yaw_deg"] <= 5,
                     ascent_end_abs_yaw_le_3deg=metrics["moving_end_yaw_deg"] is not None and abs(metrics["moving_end_yaw_deg"]) <= 3,
                     crossed_finish=metrics["crossed_finish"])
    if terrain == "descent":
        supports = [stats["nominal_excess_m"] for key, stats in reach.items() if key.endswith("support")]
        swings = [stats["nominal_excess_m"] for key, stats in reach.items() if key.endswith("swing") and stats["nominal_excess_m"]["samples"]]
        coverage = all(stats["samples"] for stats in supports)
        gates.update(crossed_finish=metrics["crossed_finish"], descent_first_three_support_levels_observed=coverage,
                     descent_max_abs_yaw_le_5deg=metrics["moving_max_abs_yaw_deg"] <= 5,
                     descent_support_reach_excess_p95_le_003m=coverage and all(stats["p95"] <= .03 for stats in supports),
                     descent_swing_reach_excess_p95_le_007m=all(stats["p95"] <= .07 for stats in swings))
    if timed is None:
        gates["first_tread_pause_exercised"] = trigger is not None
    return {"metrics": metrics, "gates": gates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path,
                        help="Optional paired comparison; omit to evaluate only the new candidate")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--terrain-dir", type=Path, default=ROOT / "deploy/deploy_mujoco/terrains",
                        help="Pin flat.xml, stairs_ascent.xml and stairs_descent.xml used for both models")
    parser.add_argument("--case", action="append", choices=tuple(CASES), help="Repeat to select a subset; default all six")
    parser.add_argument("--steps", type=int, help="Cap each case; incomplete profiles cannot pass")
    parser.add_argument("--disable-heading-hold", action="store_true")
    parser.add_argument("--legacy-heading-hold", action="store_true")
    parser.add_argument("--require-pass", action="store_true", help="Exit2 unless every candidate gate passes")
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    for path in [args.model, args.candidate] + ([args.baseline] if args.baseline is not None else []):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    selected_terrains = {CASES[name][0] for name in args.case or CASES}
    for terrain in selected_terrains:
        path = args.terrain_dir / {"flat": "flat.xml", "ascent": "stairs_ascent.xml",
                                   "descent": "stairs_descent.xml"}[terrain]
        if not path.is_file():
            parser.error(f"Terrain XML not found: {path}")
    torch.set_num_threads(1)
    report = {"deploy_version": d.TELEMETRY_VERSION, "deploy_sha256": d.DEPLOY_SCRIPT_SHA256,
        "model": str(args.model.resolve()), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "case_selection": args.case or list(CASES), "steps_cap": args.steps,
        "controller": "disabled" if args.disable_heading_hold else "legacy_v1" if args.legacy_heading_hold else "filtered_v4",
        "limitations": ["Finite deterministic initial states do not establish collision freedom.",
            "Zero-command phases exclude first0.4s and stop clean-prefix analysis at edge/nonwheel terrain contact.",
            "Shank clearance is a centerline segment proxy, not mesh separation.",
            "Descent level is each front wheel's queried tread drop, not whole-robot completed-step count.",
            "Action jitter fields are normalized; target jitter fields use radians or m/s, per policy tick.",
            "Current checks use unmodified MJCF collision settings and count solved self-contact forces>1N."],
        "runs": {}}
    runs = ([('baseline', args.baseline)] if args.baseline is not None else []) + [('candidate', args.candidate)]
    for label, checkpoint in runs:
        print(f"[EVALUATE] {label}: {checkpoint}", flush=True)
        actor = d.load_actor(checkpoint)
        cases = {}
        for name in report["case_selection"]:
            print(f"[EVALUATE] {label}/{name}", flush=True)
            cases[name] = run_case(actor, name, args)
        report["runs"][label] = {"checkpoint": str(checkpoint.resolve()),
            "controller": "disabled" if args.disable_heading_hold else "legacy_v1" if d.use_checkpoint_control_profile(args, actor) else "filtered_v4",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "cases": cases}
    for name in report["case_selection"] if args.baseline is not None else []:
        base = report["runs"]["baseline"]["cases"][name]["metrics"]
        candidate = report["runs"]["candidate"]["cases"][name]
        comparison = {}
        for key in ("leg_action_second_difference_rms", "wheel_action_second_difference_rms"):
            old, new = base["action_jitter"][key]["mean"], candidate["metrics"]["action_jitter"][key]["mean"]
            comparison[f"{key}_candidate_minus_baseline"] = new - old
            candidate["gates"][f"{key}_not_worse_by_10pct"] = new <= max(old * 1.10, old + 1.e-4)
        candidate["comparison"] = comparison
    report["candidate_all_gates_pass"] = all(all(case["gates"].values()) for case in report["runs"]["candidate"]["cases"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output.resolve()), "candidate_all_gates_pass": report["candidate_all_gates_pass"]}, indent=2))
    if args.require_pass and not report["candidate_all_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
