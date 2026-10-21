#!/usr/bin/env python3
"""CPU local full-body IK diagnostic with recorded support wheels held fixed.

Not a dynamics/controller test: local joint excursions, base lowering and pitch
are optimized to determine whether entrance preparation can geometrically close
the missing landing gap. Collision penetration is reported, not ignored.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--sample-diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = SimpleNamespace(terrain=None, terrain_xml=ROOT/'deploy/deploy_mujoco/terrains/stairs_ascent.xml',
                          stair_height=None, tread_depth=None, stair_count=None, stair_start=None)
    model = deployment.load_model(ROOT/'deep_robotics_model/M20/mjcf/M20.xml', cfg)
    contract = deployment.M20Contract(model, cfg.terrain_config)
    data = mujoco.MjData(model)
    limits = np.array(deployment.ascent_landing_module().LANDING_LIMITS[:12])*deployment.ACTION_SCALE[:12]
    rows = []
    for sample in json.loads(args.sample_diagnostic.read_text())['rows']:
        folder = args.evaluation_dir/sample['case']
        with np.load(folder/'physics.npz') as z: physical = {k: z[k] for k in z.files}
        i = sample['sample_index'];wheel = 0 if sample['wheel'] == 'FL' else 1
        q0 = physical['qpos'][i].copy(); root = int(contract.root_joint.qposadr[0])
        rotation = Rotation.from_quat(q0[root+3:root+7][[1, 2, 3, 0]])
        support = np.flatnonzero(physical['support'][i] & (np.arange(4) != wheel))
        constrained = np.r_[support, wheel]
        desired = physical['wheel_pos'][i, constrained].copy()
        desired[-1, 2] = sample['tread']*.15+.09
        lower, upper = -limits.copy(), limits.copy()
        for j, name in enumerate(deployment.JOINT_NAMES[:12]):
            mid = model.joint(name).id
            if model.jnt_limited[mid]:
                lower[j] = max(lower[j], model.jnt_range[mid, 0]-q0[contract.qpos_ids[j]])
                upper[j] = min(upper[j], model.jnt_range[mid, 1]-q0[contract.qpos_ids[j]])
        bounds = list(zip(lower, upper))+[(-.06, 0.), (-.06, .06)]
        def place(x):
            data.qpos[:] = q0
            data.qpos[contract.qpos_ids[:12]] += x[:12]
            data.qpos[root+2] += x[12]
            q = (Rotation.from_rotvec([0., x[13], 0.])*rotation).as_quat()
            data.qpos[root+3:root+7] = q[[3, 0, 1, 2]]
            mujoco.mj_kinematics(model, data)
            return data.xpos[contract.wheel_body_ids[constrained]].copy()
        best = None
        for dz in [0., -.03, -.055]:
            start = np.zeros(14);start[12] = dz
            opt = minimize(lambda x: float(np.mean((x[:12]/limits)**2)+.2*(x[12]/.06)**2+.2*(x[13]/.06)**2),
                start, bounds=bounds, constraints=[{'type': 'eq', 'fun': lambda x: (place(x)-desired).ravel()}],
                method='SLSQP', options={'maxiter': 150, 'ftol': 1.e-9})
            error = float(np.max(np.abs(place(opt.x)-desired)))
            if best is None or error < best[0]: best = (error, opt.x)
        error, solution = best
        place(solution);mujoco.mj_forward(model, data)
        bad_contacts = []
        for contact in data.contact:
            b1, b2 = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            wheel_ground = ((b1 == 0 and b2 in contract.wheel_body_ids) or
                            (b2 == 0 and b1 in contract.wheel_body_ids))
            if not wheel_ground and contact.dist < -.003:
                bad_contacts.append(float(contact.dist))
        rows.append({'case': sample['case'], 'wheel': sample['wheel'], 'target_tread': sample['tread'],
                     'held_support_wheels': support.tolist(), 'max_position_error_m': error,
                     'local_ik_feasible': error < 1.e-4, 'nonwheel_or_self_penetrations_gt3mm': bad_contacts,
                     'base_lowering_m': float(solution[12]), 'base_pitch_delta_rad': float(solution[13]),
                     'joint_delta_rad': solution[:12].tolist()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'rows': rows,
        'scope': 'CPU local kinematics around recorded actual joints, max6cm base lowering and .06rad pitch; '
                 'not a PD/dynamics reachability guarantee or a trained motion'}, indent=2)+'\n')
    for row in rows:
        print(row['case'], row['wheel'], 'position_error_m', round(row['max_position_error_m'], 6),
              'lower_cm', round(row['base_lowering_m']*100, 2), 'bad_contacts', len(row['nonwheel_or_self_penetrations_gt3mm']))


if __name__ == '__main__': main()
