#!/usr/bin/env python3
"""Replay stored kinematics through stage4's real observer using a CPU adapter.

Contact patches are approximated from archived wheel/tread measurements. This
checks bookkeeping/cache/reset plumbing, not PhysX API or slip equivalence.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--evaluation-dir', type=Path)
    inputs.add_argument('--comparison-json', type=Path)
    parser.add_argument('--model-label', default='stage3_1100')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('stage4', 'stage5'), default='stage4')
    args = parser.parse_args(); torch.set_num_threads(1)
    mdp = deployment.training_module('ascent_adaptive_mdp' if args.profile == 'stage5' else 'ascent_sequence_mdp')
    if args.comparison_json:
        entries = [r for r in json.loads(args.comparison_json.read_text())['rows'] if r['model'] == args.model_label]
        report = {'pairs': [dict(case=Path(r['raw_dir']).name, raw_dir=r['raw_dir']) for r in entries]}
    else:
        report = json.loads((args.evaluation_dir/'evaluation.json').read_text())
    rows = []
    for pair in report['pairs']:
        folder = Path(pair['raw_dir']) if 'raw_dir' in pair else args.evaluation_dir/pair['case']
        with np.load(folder/'physics.npz') as z: raw = {k: z[k] for k in z.files}
        cfg = json.loads((folder/'case.json').read_text())
        count = int(cfg['terrain']['terrain_parameters']['step_count'][0])
        start = cfg['terrain']['terrain_parameters']['start_x'][0]
        height = cfg['terrain']['terrain_parameters']['step_height'][0]
        depth = cfg['terrain']['terrain_parameters']['tread_depth'][0]
        class Scene(dict): pass
        scene = Scene()
        scene.env_origins = torch.tensor([[start-1.25, 0., 0.]])
        scene.terrain = NS(terrain_types=torch.tensor([6]), terrain_levels=torch.tensor([0]))
        data = NS(); robot = NS(data=data, find_bodies=lambda *a, **k: ([0, 1, 2, 3], None))
        scene['robot'] = robot; scene.sensors = {}
        for leg in ('fl', 'fr', 'hl', 'hr'):
            scene.sensors['entry_'+leg+'_contact'] = NS(data=NS())
        env = NS(scene=scene, num_envs=1, device='cpu', physics_dt=.005, step_dt=.02,
                 cfg=NS(ascent_entry_depth=depth, ascent_entry_height=height, ascent_sequence_step_count=count),
                 episode_length_buf=torch.zeros(1, dtype=torch.long), _sim_step_counter=0, common_step_counter=0,
                 action_manager=NS(action=torch.zeros(1, 16)))
        command = torch.zeros(1, 3)
        env.command_manager = NS(get_command=lambda name: command)
        if args.profile == 'stage5':
            env.cfg.ascent_adaptive_training = True
            env.cfg.ascent_adaptive_geometry = True
            env.cfg.ascent_adaptive_height_grid = [[height]*10]
            env.cfg.ascent_adaptive_depth_grid = [[depth]*10]
            observer = mdp.AdaptivePhysicsObserver(env)
        else:
            observer = mdp.SequencePhysicsObserver(env)
        env._ascent_entry_observer = observer
        reward_totals = {}; sampled = 0; failure_seen_at_sample = False
        for i in range(len(raw['time'])):
            env._sim_step_counter = i+1
            env.common_step_counter = i//4
            env.episode_length_buf[:] = i//4
            command[:] = torch.from_numpy(raw['requested_command'][i:i+1])
            def value(name): return torch.from_numpy(raw[name][i:i+1]).float()
            data.root_pos_w = value('root_pos'); data.root_com_pos_w = data.root_pos_w
            data.root_quat_w = value('root_quat'); data.root_lin_vel_w = value('base_velocity')
            q = data.root_quat_w
            data.projected_gravity_b = torch.stack((2.*(q[:, 0]*q[:, 2]-q[:, 1]*q[:, 3]),
                -2.*(q[:, 0]*q[:, 1]+q[:, 2]*q[:, 3]), -1.+2.*q[:, 1:3].square().sum(-1)), -1)
            data.body_pos_w = value('wheel_pos'); data.body_com_pos_w = data.body_pos_w
            data.body_com_lin_vel_w = value('wheel_velocity')
            data.body_ang_vel_w = torch.zeros(1, 4, 3)  # No angular body series in old acquisition.
            env.action_manager.action = value('action')
            for wheel, leg in enumerate(('fl', 'fr', 'hl', 'hr')):
                sensor = scene.sensors['entry_'+leg+'_contact'].data
                forces = torch.zeros(1, 1, 1, 3)
                support = bool(raw['support'][i, wheel])
                forces[0, 0, 0, 2 if support else 0] = float(raw['top_normal_force' if support else 'face_normal_force'][i, wheel])
                point = data.body_pos_w[:, wheel].clone()
                point[:, 2] = max(0, int(raw['support_tread'][i, wheel]))*height
                sensor.force_matrix_w = forces; sensor.contact_pos_w = point[:, None, None]
            observer.observe()
            # Duplicate same-tick calls must neither credit events nor time twice.
            before = observer.sequence.time.clone(); observer.observe()
            assert torch.equal(before, observer.sequence.time)
            if i % 4 == 3:
                result = mdp.context(env)
                assert result is mdp.context(env)
                assert all(torch.isfinite(v).all() for v in result.values())
                if args.profile == 'stage5':
                    if failure_seen_at_sample:
                        assert all(result[key].item() == 0. for key in ('ordered', 'paired', 'dwell', 'landing_credit', 'top_front', 'top_rear', 'top_forward'))
                    failure_seen_at_sample = observer.sequence.failed.item()
                for key, v in result.items(): reward_totals[key] = reward_totals.get(key, 0.)+v.item()*.02
                sampled += 1
        metrics = mdp.metrics(env, None)
        selected = mdp.metrics(env, slice(None))
        assert all(torch.equal(metrics[k], selected[k]) for k in metrics)
        assert all(torch.isfinite(v).all() for v in mdp.metrics(env, torch.tensor([], dtype=torch.long)).values())
        assert all(torch.isfinite(v).all() for v in metrics.values())
        assert reward_totals['ordered'] > 0 and reward_totals['missing'] > 0
        if args.profile == 'stage5':
            assert observer.sequence.failed.item()
            assert reward_totals['failure'] == 1.
            if not args.comparison_json: assert reward_totals['paired'] == 0.
            assert reward_totals['landing_credit'] >= 0.  # FL1 can be a real earlier success.
        else:
            assert observer.sequence.covered[:, :2, :4].sum().item() == 6, 'Replay adapter changed known entry coverage'
            assert reward_totals['paired'] > 0 and reward_totals['height'] > 0
        row = {'case': pair['case'], 'physics_frames': len(raw['time']), 'policy_reward_samples': sampled,
               'reward_integrals': reward_totals, 'metrics': {k: float(v) for k, v in metrics.items()}}
        observer.reset(torch.tensor([0]))
        assert not observer.sequence.covered.any() and not observer.top.front_paid.any()
        if args.profile == 'stage5':
            assert observer.stats['adaptive_opportunity'].sum() == 0.
            assert not observer.sequence.failed.any() and observer.adaptive_action_age.sum() == 0
        else:
            assert observer.stats['seq_opportunity'].sum() == 0.
            assert observer.sequence_action_age.sum() == 0
        assert not observer.top_ready.any()
        assert all(v.sum() == 0. for v in observer.window.values())
        row['reset_clears_event_credit_and_metrics'] = True; rows.append(row)
        print(pair['case'], 'OK', sampled, flush=True)
    result = {'rows': rows, 'gpu_test_run': False,
              'scope': 'Real observer/context with CPU scene adapter and approximated archived contact patches; '
                       'not Isaac training startup, exact slip replay, or trained improvement evidence.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
