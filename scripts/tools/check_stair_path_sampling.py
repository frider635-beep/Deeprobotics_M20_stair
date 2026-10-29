#!/usr/bin/env python3
"""CPU checks for explicit quota stages and minimal transfers; no checkpoints."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'sampling_check_loader', ROOT/'source/rl_training/rl_training/utils/module_loader.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
PREFIX = 'tasks.manager_based.locomotion.velocity.mdp.'


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    layout = loader.load_module(PREFIX+'path.stair_path_layout')
    sampling = loader.load_module(PREFIX+'path.stair_path_sampling')
    mdp = loader.load_module(PREFIX+'path.stair_path_mdp')
    core = loader.load_module(PREFIX+'ascent.ascent_unified')

    # Merely passing old boundaries or the total budget can never change shares.
    previous = sampling.probabilities_at(0)
    for iteration in range(40002):
        probabilities = sampling.probabilities_at(iteration)
        assert abs(sum(probabilities)-1.) < 1.e-12
        assert min(probabilities) > 0.
        assert probabilities[layout.DESCENT_START:] == previous[layout.DESCENT_START:]
        assert probabilities[layout.FLAT_START:layout.DESCENT_START] == previous[layout.FLAT_START:layout.DESCENT_START]
        assert max(abs(a-b) for a,b in zip(probabilities, previous)) < .0001
        previous = probabilities
    assert previous == sampling.probabilities_at(0)

    count = layout.STAGED_NUM_ENVS
    terrain = NS(terrain_types=torch.floor(torch.arange(count)/(count/28)).long(),
        terrain_levels=torch.full((count,), 3, dtype=torch.long),
        terrain_origins=torch.arange(6*28*3).reshape(6,28,3).float(),
        env_origins=torch.zeros(count,3))
    env = NS(num_envs=count, device='cpu', scene=NS(terrain=terrain),
        cfg=NS(m20_refinement_profile='ascent_entry_v1_stage9',
               ascent_adaptive_ascent_columns=9, stair_path_descent_start=12,
               ascent_sequence_step_count=18),
        _ascent_adaptive_curriculum=core.BridgeCurriculum(count,'cpu'), common_step_counter=0)
    descent = mdp.descent_course(env)
    all_ids = torch.arange(count)
    mdp._resample_reset_terrains(env, all_ids, 0)
    initial = terrain.terrain_types.clone()
    assert torch.bincount(initial, minlength=28).tolist() == layout.integer_sample_counts(sampling.probabilities_at(0), count)
    fixed = initial >= 9
    fixed_types = initial[fixed].clone()
    for size in (1, 28, 101):
        assert len(sampling.initial_assignment(size, 'cpu')) == size

    # Give every environment real-looking progress. Neither stable phases nor
    # later ascent transfers may erase unrelated environment histories.
    state = env._ascent_adaptive_curriculum
    state.skill.fill_(4)
    for name in ('good','bad','completed','window','trials','cursor','speed_bucket'):
        getattr(state,name).fill_(1)
    for name in ('window','trials','cursor','bad'):
        getattr(descent,name).fill_(1)
    terrain.terrain_levels.fill_(3)
    before = state.state_dict()
    down_before = descent.state_dict()
    levels_before = terrain.terrain_levels.clone()
    population = mdp._sampling_population(env)
    for iteration in (0, 599, 600, 1999, 2000, 2999, 3000):
        assert mdp._resample_reset_terrains(env, all_ids, iteration) == 0
        assert mdp._sampling_population(env) is population
    assert all(torch.equal(value,state.state_dict()[name]) for name,value in before.items())

    # Resetting only fixed-pool members cannot move a single ascent task.
    ready = mdp.readiness_sampler(env)
    ready.request(stage=2,iteration=3000)
    assert mdp._resample_reset_terrains(env, all_ids[fixed], 4000) == 0
    counts, target = sampling.ascent_targets(initial, 4000, ready)
    minimum_moves = sum(max(a-b,0) for a,b in zip(counts[:9].tolist(),target))
    moved = mdp._resample_reset_terrains(env, all_ids, 4000)
    changed = terrain.terrain_types != initial
    assert moved == minimum_moves == int(changed.sum())
    assert torch.bincount(terrain.terrain_types,minlength=28)[:9].tolist() == target
    assert torch.equal(terrain.terrain_types[fixed],fixed_types)
    assert torch.equal(terrain.terrain_levels[~changed],levels_before[~changed])
    assert (terrain.terrain_levels[changed] == 0).all()
    assert torch.equal(state.skill[changed],state.minimum_skill[changed])
    for name,value in before.items():
        actual = state.state_dict()[name]
        assert torch.equal(actual[~changed],value[~changed]), name
        if name != 'skill': assert not actual[changed].any(), name
    assert all(torch.equal(value,descent.state_dict()[name]) for name,value in down_before.items())
    assert mdp._resample_reset_terrains(env, all_ids, 4000) == 0
    assert torch.equal(terrain.env_origins[changed], terrain.terrain_origins[0,terrain.terrain_types[changed]])

    # Asynchronous episodes: transfers are restricted to the reset batch and
    # reduce the remaining histogram mismatch by exactly the number moved.
    for stage,start in zip((4,6,8,10),layout.SAMPLING_BOUNDARIES[1:]):
        ready.request(stage=stage,iteration=start)
        for iteration in range(start, start+1001, 10):
            ids = torch.randperm(count)[:127]
            old = terrain.terrain_types.clone()
            counts,target = sampling.ascent_targets(old,iteration,ready)
            pending = sum(max(a-b,0) for a,b in zip(counts[:9].tolist(),target))
            moved = mdp._resample_reset_terrains(env,ids,iteration)
            changed = terrain.terrain_types != old
            assert torch.isin(all_ids[changed],ids).all()
            assert int(changed.sum()) == moved
            after = torch.bincount(terrain.terrain_types,minlength=28)[:9].tolist()
            assert mdp._sampling_population(env).current[:9] == after
            assert sum(max(a-b,0) for a,b in zip(after,target)) == pending-moved
            assert torch.equal(terrain.terrain_types[fixed],fixed_types)
        mdp._resample_reset_terrains(env,all_ids,start+1000)
        _,target = sampling.ascent_targets(terrain.terrain_types,start+1000,ready)
        assert torch.bincount(terrain.terrain_types,minlength=28)[:9].tolist() == target
    assert mdp._resample_reset_terrains(env,all_ids,40000) == 0
    assert torch.equal(terrain.terrain_levels[fixed],levels_before[fixed])
    assert all(torch.equal(value,descent.state_dict()[name]) for name,value in down_before.items())

    # Exercise actual telemetry export after settling the final distribution.
    zero = torch.zeros(count)
    env._ascent_entry_observer = NS(**{name:zero.clone() for name in (
        'path_samples','path_max','path_signed','path_vy_total','path_slip_total',
        'path_longitudinal_slip_total','path_return_error_total','path_saturated')},
        path_failed=zero.bool(), down_success=zero.bool(), prefix_ready=zero.bool(),
        top_ready=zero.bool(), stats={'adaptive_missing':zero.clone()})
    env.command_manager = NS(get_command=lambda _:torch.tensor([[.5,0.,0.]]).expand(count,-1))
    env.common_step_counter = 21000*24
    metrics = mdp.metrics(env,all_ids[:0])
    assert metrics['sampling_stage'] == 10
    assert metrics['sampling_pending_ascent_fraction'] == 0.
    assert mdp._sampling_population(env).current == torch.bincount(terrain.terrain_types,minlength=28).tolist()
    expected_hard = sum(sampling.probabilities_at(21000,ready)[c] for c in layout.ASCENT_HARD_COLUMNS)/layout.ASCENT_FRACTION
    assert abs(metrics['sampling_target_hard_fraction']-expected_hard) < 1.e-12
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    # Exercise the environment -> scheduler adapter with ORIGINAL episode rows
    # and goals, even when the live curriculum has already advanced.
    ready.request(mode='auto',iteration=21000)
    observer=env._ascent_entry_observer
    observer.path_samples.fill_(1)
    observer.path_heading_total=zero.clone()
    observer.episode_goals=state.goals().clone()
    observer.episode_terrain_rows=torch.zeros(count,dtype=torch.long)
    observer.failure_reason=torch.zeros(count,dtype=torch.long)
    observer.sequence=NS(targets=torch.arange(1,19)[None,None,:],
        covered=torch.zeros(count,4,18,dtype=torch.bool),
        pair_paid=torch.zeros(count,18,dtype=torch.bool),
        rear_pair_paid=torch.zeros(count,18,dtype=torch.bool))
    selected=all_ids[terrain.terrain_types<9][:50]
    mdp._observe_sampling_readiness(env,selected,21000)
    bins=ready.totals.reshape(9,6,len(ready.state_dict()['prefixes']),2,9)
    expected = (~torch.isin(terrain.terrain_types[selected],torch.tensor(layout.NARROW_ASCENT_COLUMNS))).sum()
    assert bins[...,0].sum()==expected and bins[:,1:,...,0].sum()==0
    assert not bins[list(layout.NARROW_ASCENT_COLUMNS)].any()
    # Evaluation keeps its own course/goal assignment even at a different phase.
    env.cfg.ascent_adaptive_training = False
    env.common_step_counter = 0
    env._stair_path_sampling_initialized = False
    assignment = terrain.terrain_types.clone()
    mdp.metrics(env,all_ids)
    assert torch.equal(terrain.terrain_types,assignment)
    assert not env._stair_path_sampling_initialized
    print(json.dumps(dict(total_iterations=40000, timed_boundaries=[],
        mode='manual_default',ramp_iterations=1000, first_transition_minimum_moves=minimum_moves,
        ascent_pool=int((initial<9).sum()), fixed_descent_and_non_stair_pool=int(fixed.sum()),
        initial_ascent_group_counts=[int(torch.isin(initial,torch.tensor(columns)).sum()) for columns in (
            layout.ASCENT_EASY_COLUMNS,layout.ASCENT_MIDDLE_COLUMNS,layout.ASCENT_HARD_COLUMNS)],
        continuous_schedule=True, all_geometries_always_sampled=True,
        asynchronous_minimal_transfers=True, untouched_curricula_preserved=True,
        changed_geometry_speed_windows_cleared=True, pinned_prefix_retained=True,
        stable_quota_cache_reused=True, moved_quota_cache_rebuilt=True,
        telemetry_finite=True),indent=2))


if __name__ == '__main__':
    main()
