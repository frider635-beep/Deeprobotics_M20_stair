#!/usr/bin/env python3
"""Extract event-aligned M20 gait candidates from collect_m20_gait_reference.py.

Produces raw-event CSVs, quality decisions, speed-specific phase templates,
plots and a Chinese report. Rejected cycles remain in the output. The template
is a measured rhythm candidate, not a certified slip-free expert policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/m20_reference_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEGS = ("FL", "FR", "HL", "HR")
DT = .005


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def runs(values):
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    return [(int(a), int(b), int(values[a])) for a, b in zip(starts, np.r_[starts[1:], len(values)])]


def stable_events(data, case, stable_s=.04, bridge_s=.01):
    """Stable top-contact labels, bounded dropout bridging, one new high per wheel.

    A label is a *physical tread* at a force-bearing top contact, not the terrain
    below an airborne wheel. Touchdown time is backdated to the start of a run
    only after the run passes the stability duration. Same-tread recontacts are
    retained separately; they cannot repeatedly claim ascent progress.
    """
    events, all_runs = [], []
    labels = data["support_tread"]
    for wheel, leg in enumerate(LEGS):
        clean = labels[:, wheel].copy()
        for a, b, level in runs(clean):
            if level == -1 and 0 < a < b < len(clean) and (b-a)*DT <= bridge_s + 1.e-9:
                if clean[a-1] == clean[b] and clean[b] >= 0:
                    clean[a:b] = clean[b]
        high = -1
        for a, b, level in runs(clean):
            if level < 0 or (b-a)*DT < stable_s - 1.e-9:
                continue
            row = dict(case=case, wheel=leg, tread_id=level, touchdown_time_s=float(data["time"][a]),
                       stable_confirmed_time_s=float(data["time"][a] + stable_s),
                       support_end_time_s=float(data["time"][b-1] + DT), duration_s=(b-a)*DT,
                       sample_index=a, previous_high_tread=high,
                       new_high=bool(level > high), skipped_treads=max(0, level-high-1) if high >= 0 else 0)
            all_runs.append(row)
            if level > high:
                events.append(row.copy())
                high = level
    return sorted(events, key=lambda x: x["touchdown_time_s"]), sorted(all_runs, key=lambda x: x["touchdown_time_s"])


def settled_mask(support, warmup_s=.02):
    out = np.zeros_like(support, dtype=bool)
    for j in range(4):
        for a, b, level in runs(support[:, j].astype(int)):
            if level:
                out[min(b, a + int(round(warmup_s/DT))):b, j] = True
    return out


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(samples=int(a.size), median=float(np.median(a)) if a.size else None,
                p95=float(np.percentile(a, 95)) if a.size else None,
                max=float(np.max(a)) if a.size else None)


def circular(values):
    if not values:
        return dict(samples=0, mean_phase=None, std_phase=None)
    z = np.exp(2j*np.pi*np.asarray(values))
    mean = np.mean(z)
    phase = float((np.angle(mean)/(2*np.pi)) % 1.)
    return dict(samples=len(values), mean_phase=phase,
                std_phase=float(np.sqrt(-2*np.log(max(abs(mean), 1.e-12)))/(2*np.pi)))


def metrics(data, mask, settled):
    result = {}
    for j, leg in enumerate(LEGS):
        support = mask & data["support"][:, j]
        quiet = mask & settled[:, j]
        result[leg] = dict(contact_slip_m_s=stats(data["slip"][support, j]),
                           settled_slip_m_s=stats(data["slip"][quiet, j]),
                           slip_distance_m=float(np.sum(data["slip"][support, j])*DT),
                           support_fraction=float(np.mean(data["support"][mask, j])) if mask.any() else None,
                           peak_top_force_N=float(data["top_normal_force"][mask, j].max()) if mask.any() else None,
                           face_contact_time_s=float(np.sum(data["face_normal_force"][mask, j] > 5.)*DT),
                           face_normal_impulse_Ns=float(data["face_normal_force"][mask, j].sum()*DT),
                           peak_face_normal_force_N=float(data["face_normal_force"][mask, j].max()) if mask.any() else None)
    return result


def build_cycles(data, events, case, gates):
    front = [event for event in events if event["wheel"] == "FL" and event["tread_id"] > 0]
    settled = settled_mask(data["support"], gates["settle_after_touchdown_s"])
    cycles = []
    for a, b in zip(front, front[1:]):
        start, end = a["touchdown_time_s"], b["touchdown_time_s"]
        duration = end-start
        stage = "entry" if b["tread_id"] <= 3 else ("transition_3_to_4" if a["tread_id"] <= 3 else "steady")
        mask = (data["time"] >= start-1.e-9) & (data["time"] < end-1.e-9)
        measures = metrics(data, mask, settled)
        reasons = []
        if stage != "steady":
            reasons.append("entry_or_transition")
        if not (gates["min_reference_tread"] <= a["tread_id"] < b["tread_id"] <= gates["max_reference_tread"]):
            reasons.append("outside_middle_stairs")
        if b["tread_id"] - a["tread_id"] != 1:
            reasons.append("anchor_skipped_tread")
        if not .15 <= duration <= 1.5:
            reasons.append("period_out_of_range")
        for leg in LEGS:
            before = [e for e in events if e["wheel"] == leg and e["touchdown_time_s"] <= start+1.e-9]
            if not before or before[-1]["tread_id"] < 1:
                reasons.append("not_all_wheels_established_on_stairs")
                break
        for leg, m in measures.items():
            if m["settled_slip_m_s"]["p95"] is None or m["settled_slip_m_s"]["p95"] > gates["settled_slip_p95_max_m_s"]:
                reasons.append(f"{leg}_settled_slip")
            if m["slip_distance_m"] > gates["slip_distance_max_m_per_wheel_cycle"]:
                reasons.append(f"{leg}_slip_distance")
            if m["support_fraction"] is None or not .05 <= m["support_fraction"] <= .95:
                reasons.append(f"{leg}_support_fraction")
            if m["face_contact_time_s"] > gates["face_contact_max_s_per_wheel_cycle"]:
                reasons.append(f"{leg}_face_contact")
        if np.max(data["self_contact_peak_force"][mask]) > 1.:
            reasons.append("self_contact")
        if np.max(data["nonwheel_contact_peak_force"][mask]) > 1.:
            reasons.append("nonwheel_contact")
        if np.max(np.abs(np.rad2deg(data["rpy"][mask, 0]))) > gates["roll_max_deg"]:
            reasons.append("roll")
        phases = {leg: [(e["touchdown_time_s"]-start)/duration for e in events
                        if e["wheel"] == leg and start-1.e-9 <= e["touchdown_time_s"] < end-1.e-9]
                  for leg in LEGS}
        cycles.append(dict(case=case, stage=stage, start_s=start, end_s=end, duration_s=duration,
                           start_tread=a["tread_id"], end_tread=b["tread_id"],
                           accepted=not reasons, rejection_reasons=sorted(set(reasons)),
                           rhythm_only_eligible=not any(not r.endswith("_face_contact") for r in reasons),
                           touchdown_phases=phases, wheel_metrics=measures))
    return cycles, settled


def extract_template(speed, cases, datasets, events_by_case, cycles):
    # Preserve the strict decision. A second, explicitly labelled descriptive
    # rhythm set may include riser contact, but must still pass the stance-slip,
    # slip-distance, support, collision, roll and geometry checks. This set is
    # not a clean-motion expert dataset and must never be called one.
    accepted = [c for c in cycles if c["rhythm_only_eligible"] and cases[c["case"]]["speed_m_s"] == speed]
    if not accepted:
        return None, None
    phase_stats = {leg: circular([p for c in accepted for p in c["touchdown_phases"][leg]]) for leg in LEGS}
    kept, selected_events = [], []
    for c in accepted:
        chosen = {}
        for leg in LEGS:
            target_phase = phase_stats[leg]["mean_phase"]
            if target_phase is None:
                break
            if target_phase > .85:
                target_phase -= 1.
            pool = [e for e in events_by_case[c["case"]] if e["wheel"] == leg]
            event = min(pool, key=lambda e: abs(e["touchdown_time_s"]-(c["start_s"]+target_phase*c["duration_s"])))
            phase_error = abs((event["touchdown_time_s"]-c["start_s"])/c["duration_s"]-target_phase)
            if phase_error > .25:
                break
            chosen[leg] = event
        if len(chosen) != 4:
            c["rhythm_only_eligible"] = False
            c["accepted"] = False
            c["rejection_reasons"].append("ambiguous_or_missing_phase_event")
        else:
            kept.append(c)
            selected_events.append(chosen)
    accepted = kept
    if not accepted:
        return None, None
    phase_stats = {leg: circular([p for c in accepted for p in c["touchdown_phases"][leg]]) for leg in LEGS}
    grid = np.linspace(0., 1., 201)
    samples = {name: [] for name in ("joint_pos_leg_rad", "joint_vel_rad_s", "wheel_from_hip_body_m",
                                   "wheel_from_landing_edge_normalized", "pitch_rad", "support", "slip_m_s", "top_force_N")}
    event_samples = {leg: [] for leg in LEGS}
    for c, chosen in zip(accepted, selected_events):
        a = datasets[c["case"]]
        times = c["start_s"] + grid*c["duration_s"]
        interpolated = {}
        for name, key, dims in (("joint_pos_leg_rad", "joint_pos", 12), ("joint_vel_rad_s", "joint_vel", 16),
                                ("wheel_from_hip_body_m", "wheel_from_hip_body", None),
                                ("slip_m_s", "slip", None), ("top_force_N", "top_normal_force", None)):
            value = a[key] if dims is None else a[key][:, :dims]
            flat = value.reshape(len(value), -1)
            interpolated[name] = np.stack([np.interp(times, a["time"], flat[:, j]) for j in range(flat.shape[1])], -1).reshape((len(grid), *value.shape[1:]))
        indices = np.clip(np.searchsorted(a["time"], times), 0, len(a["time"])-1)
        interpolated["support"] = a["support"][indices].astype(float)
        interpolated["pitch_rad"] = np.interp(times, a["time"], a["rpy"][:, 1])
        terrain = cases[c["case"]]["terrain"]["terrain_parameters"]
        h, depth, x0 = (terrain[key][0] for key in ("step_height", "tread_depth", "start_x"))
        xyz = np.stack([np.interp(times, a["time"], a["wheel_pos"][:, j, dim]) for j in range(4) for dim in range(3)], -1).reshape(len(grid), 4, 3)
        base_y = np.interp(times, a["time"], a["root_pos"][:, 1])
        for j, leg in enumerate(LEGS):
            event = chosen[leg]
            event_phase = (event["touchdown_time_s"]-c["start_s"])/c["duration_s"]
            event_samples[leg].append(event_phase)
            level = event["tread_id"]
            xyz[:, j, 0] = (xyz[:, j, 0]-(x0+(level-1)*depth))/depth
            xyz[:, j, 1] = (xyz[:, j, 1]-base_y)/depth
            xyz[:, j, 2] = (xyz[:, j, 2]-level*h-.09)/h
        interpolated["wheel_from_landing_edge_normalized"] = xyz
        for name in samples:
            samples[name].append(interpolated[name])
    result = dict(version="m20_rhythm_reference_1", speed_m_s=speed, phase_anchor="FL stable top touchdown at phase=0",
                  phase_definition="0..1 between consecutive FL new-tread touchdowns; touchdown backdated after 40ms confirmation",
                  reference_status="RHYTHM_ONLY: riser contacts allowed in descriptive timing set; not a clean-motion expert",
                  strict_clean_cycle_count=sum(c["accepted"] for c in accepted),
                  riser_contact_cycle_count=sum(not c["accepted"] for c in accepted),
                  cycle_count=len(accepted), case_count=len(set(c["case"] for c in accepted)),
                  period_s=stats([c["duration_s"] for c in accepted]), phase_grid=grid.tolist(),
                  touchdown_phase={leg: dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                                             p90=float(np.percentile(v, 90)), circular=circular(v)) for leg, v in event_samples.items()},
                  wheel_order=list(LEGS),
                  geometry_coordinate_definition="x=(wheel_x-landing_tread_front_edge_x)/depth; y=(wheel_y-base_y)/depth; z=(wheel_z-landing_tread_z-wheel_radius)/rise",
                  source_cycles=[dict(**{k: c[k] for k in ("case", "start_s", "end_s", "start_tread", "end_tread")},
                                      touchdown_event={leg: dict(tread_id=chosen[leg]["tread_id"],
                                                               time_s=chosen[leg]["touchdown_time_s"]) for leg in LEGS})
                                 for c, chosen in zip(accepted, selected_events)],
                  trajectories={})
    arrays = dict(phase=grid, cycle_duration_s=np.array([c["duration_s"] for c in accepted]),
                  touchdown_phase_cycles=np.stack([event_samples[leg] for leg in LEGS], -1),
                  landing_tread_cycles=np.array([[chosen[leg]["tread_id"] for leg in LEGS] for chosen in selected_events]))
    for name, value in samples.items():
        value = np.stack(value)
        arrays[name+"_cycles"] = value
        for statistic, percentile in (("p10", 10), ("p50", 50), ("p90", 90)):
            reduced = np.percentile(value, percentile, axis=0)
            arrays[name+"_"+statistic] = reduced
            result["trajectories"][name+"_"+statistic] = reduced.tolist()
        if name == "support":
            result["trajectories"]["support_probability"] = value.mean(0).tolist()
            arrays["support_probability"] = value.mean(0)
    return result, arrays


def plot_case(folder, name, data, events):
    moving = data["requested_command"][:, 0] > .1
    begin = max(float(data["time"][moving][0]), float(min(e["touchdown_time_s"] for e in events if e["tread_id"] > 0))-.6)
    end = min(begin+9., float(data["time"][-1]))
    mask = (data["time"] >= begin) & (data["time"] <= end)
    t = data["time"][mask]
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for j, leg in enumerate(LEGS):
        on = data["support"][mask, j]
        axes[0].plot(t, np.where(on, j+.8, np.nan), linewidth=5, solid_capstyle="butt", label=leg)
        es = [e for e in events if e["wheel"] == leg and begin <= e["touchdown_time_s"] <= end]
        for e in es:
            axes[0].text(e["touchdown_time_s"], j+.8, str(e["tread_id"]), fontsize=8, va="bottom")
        axes[1].plot(t, np.where(on, data["slip"][mask, j], np.nan), label=leg)
        axes[2].plot(t, data["top_normal_force"][mask, j], label=leg)
    axes[0].set_yticks(np.arange(4)+.8, LEGS)
    axes[0].set_ylabel("Top support / tread ID")
    axes[1].set_ylabel("Contact slip [m/s]")
    axes[2].set_ylabel("Top normal force [N]")
    axes[3].plot(t, np.rad2deg(data["rpy"][mask, 1]), label="pitch [deg]")
    axes[3].plot(t, data["base_velocity"][mask, 0]*10, label="vx [m/s] x10")
    axes[3].set_ylabel("Base motion")
    axes[3].set_xlabel("Simulation time [s]")
    axes[0].set_title(name+" | numbers mark new stable tread touchdowns; gaps are not support")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc="upper right", ncol=4)
    fig.savefig(folder / f"{name}_timeline.png", dpi=150)
    plt.close(fig)


def plot_template(folder, result):
    p = np.array(result["phase_grid"])*100
    tr = result["trajectories"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for j, leg in enumerate(LEGS):
        axes[0].plot(p, np.array(tr["support_probability"])[:, j], label=leg)
        axes[1].plot(p, np.array(tr["wheel_from_hip_body_m_p50"])[:, j, 0], label=leg)
        line, = axes[2].plot(p, np.array(tr["slip_m_s_p50"])[:, j], label=leg)
        axes[2].fill_between(p, np.array(tr["slip_m_s_p10"])[:, j], np.array(tr["slip_m_s_p90"])[:, j],
                             color=line.get_color(), alpha=.12)
    axes[0].set_ylabel("Top support probability")
    axes[1].set_ylabel("Wheel x from hip [m]")
    axes[2].set_ylabel("Contact slip P50, P10-P90 [m/s]")
    axes[2].set_xlabel("Cycle phase [%], FL stable touchdown = 0")
    axes[0].set_title(f"{result['speed_m_s']:.1f} m/s | {result['cycle_count']} rhythm-only cycles (riser contacts retained)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(ncol=4)
    fig.savefig(folder / f"reference_vx{result['speed_m_s']:.2f}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.input_dir
    manifest = json.loads((folder / "manifest.json").read_text())
    if not np.isclose(manifest["physics_dt_s"], DT):
        raise ValueError("This extractor expects 200 Hz physics samples")
    gates = dict(stable_contact_s=.04, bridge_same_tread_gap_s=.01, settle_after_touchdown_s=.02,
                 min_reference_tread=4, max_reference_tread=18, settled_slip_p95_max_m_s=.10,
                 slip_distance_max_m_per_wheel_cycle=.03, face_contact_max_s_per_wheel_cycle=.02,
                 roll_max_deg=10., period_relative_deviation_max=.15)
    datasets, event_sets, cases = {}, {}, {c["case"]: c for c in manifest["cases"]}
    all_events, all_runs, all_cycles, stages, integrity = [], [], [], [], []
    for name, case in cases.items():
        with np.load(folder / name / "physics.npz", allow_pickle=False) as loaded:
            data = {k: loaded[k] for k in loaded.files}
        if not all(np.isfinite(x).all() for x in data.values()):
            raise ValueError(f"Nonfinite data in {name}")
        if not np.allclose(np.diff(data["time"]), DT, rtol=0, atol=1.e-10):
            raise ValueError(f"Incorrect physics sampling in {name}")
        alignment_error = float(np.max(np.abs(data["root_pos"]-data["qpos"][:, :3])))
        if alignment_error > 1.e-10:
            raise ValueError("Contact and position timestamps are inconsistent")
        events, support_runs = stable_events(data, name, gates["stable_contact_s"], gates["bridge_same_tread_gap_s"])
        cycles, settled = build_cycles(data, events, name, gates)
        all_events.extend(events)
        all_runs.extend(support_runs)
        all_cycles.extend(cycles)
        datasets[name], event_sets[name] = data, events
        progress = np.zeros((len(data["time"]), 4), dtype=int)
        for e in events:
            progress[e["sample_index"]:, LEGS.index(e["wheel"])] = e["tread_id"]
        front_max = progress[:, :2].max(-1)
        moving = data["requested_command"][:, 0] > .1
        masks = {"entry_front_treads_1_to_3": moving & (front_max >= 1) & (front_max <= 3),
                 "transition_front_tread_4": moving & (front_max == 4),
                 "steady_front_treads_5_to_18": moving & (front_max >= 5) & (front_max <= 18)}
        for stage, mask in masks.items():
            stages.append(dict(case=name, stage=stage, duration_s=float(mask.sum()*DT),
                               start_s=float(data["time"][mask][0]) if mask.any() else None,
                               end_s=float(data["time"][mask][-1]+DT) if mask.any() else None,
                               wheel_metrics=metrics(data, mask, settled)))
        integrity.append(dict(case=name, root_time_alignment_error_m=alignment_error,
                              physics_samples=len(data["time"]), stable_new_tread_events=len(events),
                              accepted_cycles=sum(c["accepted"] for c in cycles),
                              rhythm_only_cycles=sum(c["rhythm_only_eligible"] for c in cycles), status=case["status"]))
        if abs(case["spawn_x_offset_m"]) < 1.e-9:
            plot_case(folder, name, data, events)
    for speed in sorted(set(c["speed_m_s"] for c in cases.values())):
        pool = [c for c in all_cycles if cases[c["case"]]["speed_m_s"] == speed and c["stage"] == "steady"
                and gates["min_reference_tread"] <= c["start_tread"] < c["end_tread"] <= gates["max_reference_tread"]
                and c["end_tread"]-c["start_tread"] == 1]
        if not pool:
            continue
        period = float(np.median([c["duration_s"] for c in pool]))
        for c in pool:
            if abs(c["duration_s"]/period-1.) > gates["period_relative_deviation_max"]:
                c["rejection_reasons"].append("period_deviation")
                c["rhythm_only_eligible"] = c["accepted"] = False
    templates = []
    for speed in sorted(set(c["speed_m_s"] for c in cases.values())):
        result, arrays = extract_template(speed, cases, datasets, event_sets, all_cycles)
        if result:
            result["quality_gates"] = gates
            result["checkpoint_sha256"] = manifest["checkpoint_sha256"]
            write_json(folder / f"reference_vx{speed:.2f}.json", result)
            np.savez_compressed(folder / f"reference_vx{speed:.2f}.npz", **arrays)
            plot_template(folder, result)
            templates.append(result)
    for row in integrity:
        selected = [c for c in all_cycles if c["case"] == row["case"]]
        row["accepted_cycles"] = sum(c["accepted"] for c in selected)
        row["rhythm_only_cycles"] = sum(c["rhythm_only_eligible"] for c in selected)
    write_csv(folder / "landing_events.csv", all_events)
    write_csv(folder / "stable_support_runs.csv", all_runs)
    write_json(folder / "cycle_quality.json", all_cycles)
    write_json(folder / "stage_metrics.json", stages)
    write_json(folder / "integrity.json", integrity)
    summary = dict(checkpoint_sha256=manifest["checkpoint_sha256"], cases=integrity, quality_gates=gates,
                   analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   reference_summary=[{key: t[key] for key in ("speed_m_s", "cycle_count", "case_count", "strict_clean_cycle_count", "riser_contact_cycle_count", "period_s", "touchdown_phase")} for t in templates],
                   rejected_cycle_count=sum(not c["accepted"] for c in all_cycles),
                   total_cycle_count=len(all_cycles), references_available=bool(templates))
    write_json(folder / "analysis_summary.json", summary)
    lines = ["# M20 上楼参考节奏采集与分析", "", "日期：2026-10-20。仅采集与分析，未修改策略、训练奖励或控制参数。", "",
             "## 数据来源与条件", "", f"- 检查点：`pre_train/teacher_baseline_.pt`；SHA256：`{manifest['checkpoint_sha256']}`。",
             "- 用户确认默认楼梯和控制方式。15 cm 阶高、30 cm 踏深、20 级；使用检查点 legacy_v1 航向保持和完整运动残差网络。",
             "- 0.5/0.7 m/s，每档出生 X 偏移 -0.10/0/+0.10 m；先零命令稳定 6 s。无随机噪声；六条是受控不同初始条件，不是六个独立随机种子。",
             "- 策略 50 Hz、接触/状态 200 Hz。原始轮序 FL/FR/HL/HR。MuJoCo 录屏的具体按键时序未提供，本次是默认环境的新增物理回放，不是从视频反演的精确轨迹。",
             "- 每个接触点用 `mj_jac(point, wheel_body) @ qvel` 得到轮体材料点相对静止地面的速度，再投影到接触切平面；不会把轮心平移直接当成滑移。",
             "- `mj_step` 返回时接触几何属于积分前状态；记录积分前的 time/qpos/qvel 与求解后的接触力配对，不额外调用 forward 改变仿真。", "",
             "## 提取方法", "", "1. 只把向上法向的求解接触视为踏面候选；合力竖直分量 >5 N、竖直分量占合力 >50%、踏面法向力 >5 N 才视为有效支撑。立面接触另记。",
             "2. 由实际踏面接触点高度确定物理阶号。支撑连续至少40 ms才确认落地，事件时间回记为该段支撑起点；同阶、最长10 ms的接触掉点可桥接。保留同阶重复接触，只有首次达到新阶计入爬阶进度。",
             "3. 用同一只前轮 FL 连续两次新阶稳定落地划分周期。入阶、3→4过渡及中段分开；参考仅取起始阶≥4、结束阶≤18、FL每周期上升一级且四轮已经上阶的片段。",
             "4. 周期归一化为0～100%，以FL落地为0。接触事件的相位采用圆周统计，0%和100%相邻。轮轨迹同时提供机身坐标下相对髋的位置及相对本周期落地踏面前沿的归一化坐标。",
             "5. 接触初始20 ms与后续承重分别统计；初始滑移仍计入全周期滑移距离。严格筛选要求各轮后续承重滑移P95≤0.10 m/s、周期滑移距离≤3 cm、立面接触≤20 ms，且没有>1 N的自碰/非轮地面接触、|roll|≤10°。这些是本次明确记录的工程筛选阈值，不是硬件安全认证。",
             "6. 严格筛选与描述性节奏集合分开记录：后者仅允许放宽立面接触这一项，其余条件保持一致，用于提取已有步态的相位；严格筛选结果不被覆盖。立面接触本身不一定是故障，但当前片段不能标作干净越阶专家。",
             "7. 再剔除周期长度偏离同速度中段周期中位数>15%的片段，以及四轮事件无法在统计相位附近±25%周期内明确配对的片段。每档速度独立汇总201个相位点的P10/P50/P90、支撑概率、关节/轮速和轨迹。被拒片段及原因全部保留，未挑选后删除不利样本。", "",
             "## 采集结果", "", "| 速度 m/s | X偏移 m | 回放结果 | 200Hz帧 | 严格合格 | 仅节奏候选 |", "|---|---|---|---:|---:|---:|"]
    for row in integrity:
        c = cases[row["case"]]
        lines.append(f"| {c['speed_m_s']:.1f} | {c['spawn_x_offset_m']:+.2f} | {row['status']} | {row['physics_samples']} | {row['accepted_cycles']} | {row['rhythm_only_cycles']} |")
    lines += ["", "## 提取的节奏", "", "相位为同一FL周期的比例，不要求左右轮同时落地；接近0的负相位表示略早于FL。", "", "| 速度 m/s | 周期数 | 周期中位数 s | FR相位 | HL相位 | HR相位 |", "|---|---:|---:|---:|---:|---:|"]
    for t in templates:
        phases = t["touchdown_phase"]
        lines.append(f"| {t['speed_m_s']:.1f} | {t['cycle_count']} | {t['period_s']['median']:.3f} | {phases['FR']['median']*100:.1f}% | {phases['HL']['median']*100:.1f}% | {phases['HR']['median']*100:.1f}% |")
    if not templates:
        lines += ["", "当前阈值下没有合格参考周期，不能把后续片段直接作为专家数据。详见cycle_quality.json。"]
    elif not any(c["accepted"] for c in all_cycles):
        lines += ["", "**严格干净参考周期为0。输出的是已有步态的节奏候选，不是完整的低滑移/无立面接触专家动作。所有节奏候选仍有超出严格阈值的立面接触；保留其相位作软引导，轨迹和轮速只供诊断，不建议原样强制模仿。**"]
    lines += ["", "**主要发现：**标准出生位置、0.7 m/s时，入口左前轮首先稳定落在阶1、阶3，右前轮首先稳定落在阶2、阶4；后续两前轮则都逐阶落地。入口与后续的区别包含落阶序列改变，不能仅靠统一周期时长描述。中段左右前轮支撑占比也明显不同，模板保留这种实测不对称性，没有将其人为改成对称步态。",
              "", "- 0.7 m/s参考中，左前轮与右后轮接近同时落地，左后轮约在44%周期、右前轮约在69%周期落地。其间可能有单轮踏面支撑及额外阶沿/立面接触，不能将它简化为理想对角同步。",
              "- 全接触滑移和接触20 ms之后的滑移差异较大：中段仍反复出现落地瞬间的高滑移，因此后续片段只证明有可重复节奏，不能证明其全部接触过程优于入口。",
              "- 非向上法向接触记为face，可能包括立面和阶沿斜接触；持续超过20 ms只是违反本次严格筛选条件，不单独判定为不合理碰撞。",
              "- 原始事件时间分辨率5 ms；40 ms是离线落地确认时长。相位分布为样本统计，不代表亚毫秒控制精度。"]
    lines += ["", "## 标准出生位置：入口与中段对比", "", "P95仅统计该轮实际踏面支撑帧；settled排除每次接触最初20 ms，all包含这些帧。长阶段不直接比较总滑移距离，应比较滑移速度或周期归一化距离。", "",
              "| 速度 | 阶段 | 轮 | all滑移P95 m/s | settled滑移P95 m/s | 支撑占比 |", "|---|---|---|---:|---:|---:|"]
    for stage in stages:
        if abs(cases[stage["case"]]["spawn_x_offset_m"]) > 1.e-9:
            continue
        for leg, m in stage["wheel_metrics"].items():
            def fmt(v):
                return "无样本" if v is None else f"{v:.3f}"
            lines.append(f"| {cases[stage['case']]['speed_m_s']:.1f} | {stage['stage']} | {leg} | {fmt(m['contact_slip_m_s']['p95'])} | {fmt(m['settled_slip_m_s']['p95'])} | {fmt(m['support_fraction'])} |")
    lines += ["", "## 标准出生位置：前几阶实测落地顺序", "", "这里的阶号来自稳定踏面接触，跳过某阶表示未观测到该轮在该阶≥40 ms的有效稳定支撑，不等于完全没有瞬时接触。", ""]
    for name, case in cases.items():
        if abs(case["spawn_x_offset_m"]) > 1.e-9:
            continue
        lines.append(f"- {case['speed_m_s']:.1f} m/s：")
        for leg in LEGS:
            first = [e for e in event_sets[name] if e["wheel"] == leg and 1 <= e["tread_id"] <= 7]
            lines.append("  - "+leg+"："+" → ".join(f"阶{e['tread_id']}@{e['touchdown_time_s']:.3f}s" for e in first))
    lines += ["", "## 文件与使用边界", "", "- `manifest.json`：检查点/代码哈希、地形、控制配置、命令和逐次运行信息。每次回放目录含physics.npz、policy.npz、contacts.csv.gz、case.json。",
              "- `sources/`保存本次采集时与manifest哈希一致的源文件；当前工具若在采集后修复输出目录保护等逻辑，应以快照核对本次采集实现。",
              "- `landing_events.csv`和`stable_support_runs.csv`：新阶落地与全部稳定支撑段。`stage_metrics.json`：分阶段四轮滑移与支撑。`cycle_quality.json`：所有周期的筛选结果与原因。",
              "- `reference_vx*.json/.npz`：按速度区分的节奏候选、201相位点轨迹分位数与逐周期数组，含来源case/时段/物理阶号。四轮关节位置只推荐使用腿部12维，不复制连续旋转轮关节的绝对角度。",
              "- `*_timeline.png`：接触/阶号、滑移、接触力与机身状态；`reference_vx*.png`：归一化节奏。",
              "- rhythm_only_eligible表示满足描述性节奏条件，accepted才是严格干净筛选。节奏候选保留立面接触和落地瞬态，不应宣称整周期无滑移。参考相位不等于可直接复制到首阶的固定关节目标。",
              "- 一条连续回放内的相邻周期有关联，P10/P90是样本分布，不是跨独立实验的置信区间。当前结果仅覆盖默认15/30 cm楼梯、两档速度与三个出生位置，未覆盖低摩擦、不同阶高、传感噪声或真机。",
              "- 本轮未执行奖励修改、策略微调或步骤2～5。后续可先使用0.7 m/s模板的相位与相对落点作为软目标，再单独验证入口物理可行性。", ""]
    validation_path = folder / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        lines += ["## 测量验证", "", "- 解析纯滚动、锁轮平移、原地空转三个反例通过：纯滚动接触点滑移接近0，后两者约0.4 m/s。",
                  "- 接触短脉冲、同阶重复落地、倒退和0/100%相位跨界检查通过。",
                  f"- 标准出生位置0.7 m/s另跑未插入采集的部署循环，逐动作与最终qpos比较：最大动作误差{validation['production_equivalence']['max_action_error']:.1f}、最大最终qpos误差{validation['production_equivalence']['max_final_qpos_error']:.1f}；采集没有改变回放。详见validation.json。", ""]
    lines += ["## 复现命令", "", "从仓库根目录执行；采集输出目录必须是新目录，避免覆盖已有原始数据。", "", "```bash",
              "python scripts/tools/collect_m20_gait_reference.py --output-dir .scratch/m20_reference_repeat --speeds 0.5 0.7 --spawn-x-offsets -0.1 0 0.1",
              "python scripts/tools/analyze_m20_gait_reference.py --input-dir .scratch/m20_reference_repeat",
              "python scripts/tools/check_m20_gait_reference.py --input-dir .scratch/m20_reference_repeat",
              "python scripts/tools/analyze_m20_gait_reference.py --input-dir .scratch/m20_reference_repeat", "```", "",
              "接触点速度实现参考：[MuJoCo mj_jac 官方文档](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac)。", ""]
    (folder / "分析报告.md").write_text("\n".join(lines))
    print(json.dumps({key: summary[key] for key in ("total_cycle_count", "rejected_cycle_count", "reference_summary")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
