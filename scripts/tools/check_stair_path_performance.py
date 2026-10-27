#!/usr/bin/env python3
"""Synthetic CPU observer checks; optional source comparison, no policy/Isaac.

--baseline-root accepts an archived rl_training package directory. It compares
all observer tensors at every tick, including sparse/explicit resets, and counts
CPU-dispatched nonzero operations. CPU timings are not a GPU speed estimate.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'tasks.manager_based.locomotion.velocity.mdp.'


def load_source(root, namespace):
    spec = importlib.util.spec_from_file_location(namespace+'_loader', root/'utils/module_loader.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    loader._NAMESPACE = namespace
    return loader


class Scene(dict):
    pass


def fixture(loader, count, progressive=False, pairing=False):
    mdp = loader.load_module(PREFIX+'path.stair_path_mdp')
    scene = Scene()
    scene.env_origins = torch.zeros(count, 3)
    scene.terrain = NS(terrain_types=torch.arange(count) % 28,
                       terrain_levels=torch.zeros(count, dtype=torch.long))
    data = NS()
    scene['robot'] = NS(data=data, find_bodies=lambda *a, **k: ([0, 1, 2, 3], None))
    scene.sensors = {f'entry_{leg}_contact':NS(data=NS()) for leg in ('fl','fr','hl','hr')}
    command = torch.zeros(count, 3)
    command[:, 0] = torch.where(torch.arange(count) % 2 == 0, .5, .7)
    env = NS(scene=scene, num_envs=count, device='cpu', physics_dt=.005, step_dt=.02,
        cfg=NS(ascent_entry_depth=.30, ascent_entry_height=.15, ascent_sequence_step_count=18,
               m20_refinement_profile='ascent_entry_v1_stage9' if progressive else None,
               stair_ascent_pairing_v9=pairing, stair_ascent_pairing_v8=pairing, stair_ascent_pairing_v7=pairing,
               ascent_adaptive_geometry=True, ascent_adaptive_training=True,
               ascent_adaptive_height_grid=[[.1+.03*r]*28 for r in range(6)],
               ascent_adaptive_depth_grid=[[.20+.05*(c%4) for c in range(28)]]*6,
               ascent_adaptive_ascent_column_ids=tuple(range(9)),
               stair_path_descent_columns=tuple(range(12,20)), stair_path_config=None),
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        _sim_step_counter=0, common_step_counter=0,
        action_manager=NS(action=torch.zeros(count,16)),
        command_manager=NS(get_command=lambda _:command))
    observer = mdp.PathPhysicsObserver(env)
    env._ascent_entry_observer = observer
    return env, observer


def step(env, observer, tick):
    n = env.num_envs
    env._sim_step_counter = tick
    env.common_step_counter = (tick-1)//4
    if tick % 4 == 1:
        env.episode_length_buf += 1
    # Exercise fallback reset detection independently of explicit reset events.
    if tick in (23, 51):
        env.episode_length_buf[::3] = 0
    if tick == 37:
        ids = torch.arange(1, n, 4)
        observer.reset(ids)
        env.episode_length_buf[ids] = 0
    data = env.scene['robot'].data
    rows = torch.arange(n).float()
    data.root_pos_w = torch.stack((rows*.001+tick*.0025, .01*torch.sin(rows+tick*.1), rows*0.+.5), -1)
    yaw = .05*torch.sin(rows*.1+tick*.01)
    data.root_quat_w = torch.stack((torch.cos(yaw/2), yaw*0., yaw*0., torch.sin(yaw/2)), -1)
    data.root_lin_vel_w = torch.stack((rows*0.+.5, .01*torch.cos(rows+tick*.1), rows*0.), -1)
    data.projected_gravity_b = torch.tensor([0.,0.,-1.]).expand(n,-1)
    data.body_pos_w = data.root_pos_w[:,None,:]+torch.tensor([
        [.35,.2,-.31],[.35,-.2,-.31],[-.35,.2,-.31],[-.35,-.2,-.31]])
    data.body_com_pos_w = data.body_pos_w
    data.body_com_lin_vel_w = data.root_lin_vel_w[:,None,:].expand(-1,4,-1)
    data.body_ang_vel_w = torch.zeros(n,4,3)
    env.action_manager.action = torch.sin(rows[:,None]+torch.arange(16)[None,:]+tick*.01)
    for wheel, leg in enumerate(('fl','fr','hl','hr')):
        sensor = env.scene.sensors[f'entry_{leg}_contact'].data
        force = torch.zeros(n,1,2,3)
        force[:,0,0,2] = 70.*(((torch.arange(n)+tick+wheel*7)%32)<24)
        point = data.body_pos_w[:,wheel].clone()
        point[:,2] = .1*(1+tick//40)
        sensor.force_matrix_w = force
        sensor.contact_pos_w = point[:,None,None,:].expand(-1,1,2,-1)
    observer.observe()


def tensors(value, prefix=''):
    if isinstance(value, torch.Tensor):
        return {prefix:value}
    fields = value if isinstance(value, dict) else vars(value)
    result = {}
    for key, child in fields.items():
        if key == 'env' or key.startswith('_episode_reset_'):
            continue
        if isinstance(child, (torch.Tensor, dict)) or hasattr(child, '__dict__'):
            result.update(tensors(child, prefix+'.'+key))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--num-envs', type=int, default=56)
    parser.add_argument('--ticks', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    current = load_source(ROOT/'source/rl_training/rl_training', '_m20_perf_current')
    env, observer = fixture(current, args.num_envs)
    adaptive = current.load_module(PREFIX+'ascent.ascent_adaptive')
    # Contiguous ranges and arbitrary sets retain identical membership.
    for columns in ((), (0,), tuple(range(9)), tuple(range(12,20)), (7,2,7)):
        expected = torch.zeros(args.num_envs,dtype=torch.bool)
        for column in columns:
            expected |= env.scene.terrain.terrain_types == column
        assert torch.equal(adaptive.column_mask(env,columns),expected)
    baseline = load_source(args.baseline_root, '_m20_perf_baseline') if args.baseline_root else None
    old_env, old_observer = fixture(baseline,args.num_envs) if baseline else (None,None)
    checked = 0
    for tick in range(1,args.ticks+1):
        step(env,observer,tick)
        # Same-tick repeated consumers cannot advance state a second time.
        before = observer.path_samples.clone()
        observer.observe()
        assert torch.equal(before,observer.path_samples)
        if baseline:
            step(old_env,old_observer,tick)
            left, right = tensors(observer), tensors(old_observer)
            assert left.keys() == right.keys(), (left.keys()-right.keys(),right.keys()-left.keys())
            for key in left:
                torch.testing.assert_close(left[key],right[key],rtol=0.,atol=0.,equal_nan=True,msg=key)
                checked += 1
    assert observer.path_samples.max() > 0
    report = dict(policy_checkpoints_loaded=0, physics_simulation=False,
                  ticks=args.ticks, environments=args.num_envs, tensor_comparisons=checked,
                  same_tick_idempotent=True, column_membership_equivalent=True)
    # Dispatch counts isolate redundant bookkeeping independently of hardware.
    for label, source in [('current',current)]+([('baseline',baseline)] if baseline else []):
        bench_env, bench_observer = fixture(source,args.num_envs)
        for tick in range(1,9):
            step(bench_env,bench_observer,tick)
        start = time.perf_counter()
        for tick in range(9,25):
            step(bench_env,bench_observer,tick)
        report[label+'_cpu_ms_per_tick'] = (time.perf_counter()-start)*1000/16
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            step(bench_env,bench_observer,25)
        report[label+'_nonzero_calls_per_tick'] = sum(
            event.count for event in profile.key_averages() if event.key == 'aten::nonzero')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
