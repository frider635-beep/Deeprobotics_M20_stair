#!/usr/bin/env python3
"""CPU fixed-base FK diagnostic of missed-tread PD target boxes, not a rollout.

The robot's actual dynamic workspace is NOT the box of instantaneous PD targets.
This diagnostic asks whether allowed targets near recorded missed landings can
even place a wheel near its intended tread with the recorded base pose fixed.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import minimize
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--factors', type=float, nargs='+', default=[1., 2., 4.])
    parser.add_argument('--center', choices=('baseline', 'source'), default='baseline',
                        help='Center target boxes on frozen original motion or full stage2 actions.')
    parser.add_argument('--hipx-factor', type=float, default=None,
                        help='Optional separate hipx factor, preserving small lateral authority.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = SimpleNamespace(terrain=None, terrain_xml=ROOT/'deploy/deploy_mujoco/terrains/stairs_ascent.xml',
                          stair_height=None, tread_depth=None, stair_count=None, stair_start=None)
    model = deployment.load_model(ROOT/'deep_robotics_model/M20/mjcf/M20.xml', cfg)
    contract = deployment.M20Contract(model, cfg.terrain_config)
    data = mujoco.MjData(model)
    report = json.loads((args.evaluation_dir/'evaluation.json').read_text())
    rows = []
    for pair in report['pairs']:
        folder = args.evaluation_dir/pair['case']
        with np.load(folder/'physics.npz') as z: physical = {k: z[k] for k in z.files}
        with np.load(folder/'policy.npz') as z: policy = {k: z[k] for k in z.files}
        for wheel, name, tread in [(0, 'FL', 2), (1, 'FR', 1)]:
            edge = .25+(tread-1)*.30
            inside = (physical['wheel_pos'][:, wheel, 0] >= edge+.07)
            inside &= (physical['wheel_pos'][:, wheel, 0] <= edge+.25) & (physical['time'] >= 6.)
            candidates = np.flatnonzero(inside)
            if not len(candidates):
                continue
            # Lowest observed wheel position inside the allowed landing patch
            # is the recorded trajectory's most favorable vertical opportunity.
            sample = candidates[np.argmin(physical['wheel_pos'][candidates, wheel, 2])]
            step = int(physical['policy_step'][sample])
            observation = torch.from_numpy(policy['observation'][step:step+1])
            command = torch.from_numpy(policy['requested_command'][step:step+1])
            gate = deployment.ascent_entry_module().ascent_gate(observation, command).item()
            limits = np.array([.12, .24, .24]*4+[.12]*4)
            correction = physical['entry_bounded'][sample]*limits*gate
            baseline_action = physical['action'][sample]-(correction if args.center == 'baseline' else 0.)
            joint_slice = slice(wheel*3, wheel*3+3)
            joint_ids = contract.qpos_ids[joint_slice]
            target = deployment.DEFAULT_JOINT_POS[joint_slice]+baseline_action[joint_slice]*deployment.ACTION_SCALE[joint_slice]
            actual = physical['qpos'][sample, joint_ids]
            qpos = physical['qpos'][sample].copy()
            def position(q):
                data.qpos[:] = qpos
                data.qpos[joint_ids] = q
                mujoco.mj_kinematics(model, data)
                return data.xpos[contract.wheel_body_ids[wheel]].copy()
            actual_fk = position(actual)
            assert np.max(np.abs(actual_fk-physical['wheel_pos'][sample, wheel])) < 1.e-7
            for factor in args.factors:
                half = limits[joint_slice]*deployment.ACTION_SCALE[joint_slice]*gate*factor
                if args.hipx_factor is not None:
                    half[0] = limits[joint_slice][0]*deployment.ACTION_SCALE[joint_slice][0]*gate*args.hipx_factor
                lower, upper = target-half, target+half
                for j, joint in enumerate(range(wheel*3, wheel*3+3)):
                    mid = model.joint(deployment.JOINT_NAMES[joint]).id
                    if model.jnt_limited[mid]:
                        lower[j] = max(lower[j], model.jnt_range[mid, 0])
                        upper[j] = min(upper[j], model.jnt_range[mid, 1])
                if np.any(lower > upper):
                    rows.append({'case': pair['case'], 'wheel': name, 'tread': tread,
                                 'factor': factor, 'feasible': False, 'reason': 'PD box outside joint limits'})
                    continue
                constraints = [
                    {'type': 'ineq', 'fun': lambda q: position(q)[0]-(edge+.07)},
                    {'type': 'ineq', 'fun': lambda q: edge+.25-position(q)[0]},
                ]
                starts = [np.clip(actual, lower, upper), (lower+upper)/2]
                starts += [np.where(np.array(bits), upper, lower) for bits in itertools.product([0, 1], repeat=3)]
                best = None
                for start in starts:
                    opt = minimize(lambda q: position(q)[2], start, bounds=list(zip(lower, upper)),
                                   constraints=constraints, method='SLSQP',
                                   options={'maxiter': 60, 'ftol': 1.e-9})
                    point = position(opt.x)
                    if edge+.07-1.e-6 <= point[0] <= edge+.25+1.e-6:
                        if best is None or point[2] < best[0][2]: best = (point, opt.x)
                row = {'case': pair['case'], 'wheel': name, 'tread': tread, 'factor': factor,
                       'sample_index': int(sample), 'time_s': float(physical['time'][sample]),
                       'gate': gate, 'actual_wheel_bottom_minus_tread_m': float(actual_fk[2]-.09-tread*.15),
                       'feasible': best is not None}
                if best is not None:
                    row.update(min_pd_target_bottom_minus_tread_m=float(best[0][2]-.09-tread*.15),
                               minimizing_joint_target=best[1].tolist())
                rows.append(row)
    result = {'evaluation_dir': str(args.evaluation_dir.resolve()), 'center': args.center,
              'hipx_factor': args.hipx_factor, 'rows': rows,
              'scope': 'CPU fixed recorded base and other joints, optimize three leg PD targets inside their box; '
                       '9cm wheel-bottom approximation, not collision-free IK or proof of dynamic reachability'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for factor in args.factors:
        for wheel in ['FL', 'FR']:
            pool = [r for r in rows if r['factor'] == factor and r['wheel'] == wheel]
            print(factor, wheel, 'feasible', sum(r['feasible'] for r in pool), '/', len(pool),
                  'minimum PD target clearance cm', [round(r['min_pd_target_bottom_minus_tread_m']*100, 2)
                                                     if r['feasible'] else None for r in pool], flush=True)


if __name__ == '__main__': main()
