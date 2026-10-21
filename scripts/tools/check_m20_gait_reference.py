#!/usr/bin/env python3
"""Measurement counterexamples and production-rollout equivalence, no training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from collect_m20_gait_reference import ROOT, contact_measurements, d
from analyze_m20_gait_reference import circular, stable_events


def check_rolling():
    # A sphere gives an analytic no-slip material-point velocity. The 10 um
    # penetration makes its normal force observable; the contact midpoint
    # introduces a known <0.0001 m/s geometric residual at this test speed.
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <worldbody><geom type="plane" size="100 100 .1"/>
       <body name="fl" pos="0 0 .09999"><freejoint/><geom type="sphere" size=".1" mass="2"/></body>
       <body name="fr" pos="1 0 10"><freejoint/><geom type="sphere" size=".1" mass="2"/></body>
       <body name="hl" pos="2 0 10"><freejoint/><geom type="sphere" size=".1" mass="2"/></body>
       <body name="hr" pos="3 0 10"><freejoint/><geom type="sphere" size=".1" mass="2"/></body>
      </worldbody></mujoco>''')
    contract = SimpleNamespace(wheel_body_ids=np.arange(1, 5), terrain=SimpleNamespace(parameters={"step_height": [.15]}))
    result = {}
    for name, vx, wy in (("pure_rolling", .4, 4.), ("locked_translation", .4, 0.), ("spinning_in_place", 0., 4.)):
        data = mujoco.MjData(model)
        data.qvel[0], data.qvel[4] = vx, wy
        mujoco.mj_forward(model, data)
        m, contacts = contact_measurements(model, data, contract, data.qvel.copy(), 0, 0.)
        assert contacts and m["support"][0], "Analytic rolling test has no load-bearing contact"
        result[name+"_slip_m_s"] = float(m["slip"][0])
    assert result["pure_rolling_slip_m_s"] < 1.e-4
    assert .399 < result["locked_translation_slip_m_s"] < .401
    assert .399 < result["spinning_in_place_slip_m_s"] < .401
    return result


def check_segmentation():
    labels = np.array([0]*20 + [-1]*5 + [1]*6 + [-1]*5 + [2]*12 + [-1]*2 + [2]*8 + [-1]*5 + [1]*12)
    data = dict(time=np.arange(len(labels))*.005, support_tread=np.tile(labels[:, None], (1, 4)))
    events, support_runs = stable_events(data, "synthetic")
    fl = [e for e in events if e["wheel"] == "FL"]
    assert [e["tread_id"] for e in fl] == [0, 2], "Transient, repeated or backwards landings became ascent progress"
    assert fl[-1]["skipped_treads"] == 1
    assert abs(fl[-1]["touchdown_time_s"] - 36*.005) < 1.e-12
    assert abs(fl[-1]["stable_confirmed_time_s"] - (36*.005+.04)) < 1.e-12
    assert any(e["wheel"] == "FL" and e["tread_id"] == 1 and not e["new_high"] for e in support_runs)
    phase = circular([.99, .01])
    assert min(phase["mean_phase"], 1.-phase["mean_phase"]) < 1.e-9
    return dict(transient_recontact_and_backwards_progress_rejected=True, circular_phase_wrap_correct=True)


def check_production_equivalence(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    case = next(c for c in manifest["cases"] if c["speed_m_s"] == .7 and c["spawn_x_offset_m"] == 0.)
    p = np.load(folder / case["case"] / "policy.npz")
    a = np.load(folder / case["case"] / "physics.npz")
    cfg = SimpleNamespace(terrain=None, terrain_xml=Path(case["terrain"]["terrain_xml"]))
    model = d.load_model(ROOT / "deep_robotics_model/M20/mjcf/M20.xml", cfg)
    contract = d.M20Contract(model, cfg.terrain_config)
    actor = d.load_actor(Path(manifest["checkpoint"]))
    data = mujoco.MjData(model)
    contract.reset(data)
    hold = d.HeadingHoldController(contract, enabled=True, legacy=True)
    previous = np.zeros(16, dtype=np.float32)
    action_error = 0.
    # Same production loop, with no contact logging, raycast diagnostics,
    # residual introspection, or additional Jacobian queries.
    for step in range(len(p["time"])):
        raw = p["requested_command"][step]
        effective, _ = hold.update(data, raw)
        with torch.inference_mode():
            action = actor(d.actor_inputs(contract, data, effective, previous, actor, raw)).squeeze(0).numpy()
        action = np.clip(action, -100., 100.)
        action_error = max(action_error, float(np.max(np.abs(action-p["action"][step]))))
        for _ in range(4):
            contract.apply_pd(data, action)
            mujoco.mj_step(model, data)
        previous = action.astype(np.float32)
    state_error = float(np.max(np.abs(data.qpos-np.array(case["final_qpos"]))))
    assert action_error == 0. and state_error == 0., (action_error, state_error)
    assert np.max(np.abs(a["qpos"][:, :3]-a["root_pos"])) == 0.
    return dict(policy_steps=len(p["time"]), max_action_error=action_error, max_final_qpos_error=state_error,
                measurement_did_not_change_production_rollout=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = dict(rolling=check_rolling(), event_detection=check_segmentation(),
                  production_equivalence=check_production_equivalence(args.input_dir))
    (args.input_dir / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
