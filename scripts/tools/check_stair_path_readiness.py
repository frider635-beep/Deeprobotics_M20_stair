#!/usr/bin/env python3
"""Deterministic CPU checks for trend gates, demotion and saved scheduler state.

Uses synthetic episode statistics only, never reads a policy checkpoint and
never initializes Isaac/MuJoCo, a runner, PPO or a full resume.
"""
import importlib.util
import io
import json
from types import SimpleNamespace
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('readiness_check_loader',ROOT/'source/rl_training/rl_training/utils/module_loader.py')
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
module=loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_readiness')
layout=loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_layout')


def evidence(success=.55, hard=None):
    totals=torch.zeros((*module.SHAPE,9),dtype=torch.float64)
    totals[...,0]=100
    totals[...,1:]=torch.tensor([success,.25,.20,.82,.86,.06,.04,.02],dtype=torch.float64)*100
    if hard is not None: totals[list(layout.ASCENT_HARD_COLUMNS),...,1]=hard*100
    return totals


def add_block(state, iteration, totals):
    state.totals.copy_(totals.reshape_as(state.totals))
    state.population_long_fraction=.05
    state.advance(iteration)


def clone(state):
    # Prove scheduler metadata is weights_only-compatible, without policy state.
    stream=io.BytesIO();torch.save(state.state_dict(),stream);stream.seek(0)
    restored=module.ReadinessSampler()
    restored.load_state_dict(torch.load(stream,weights_only=True))
    return restored


def main():
    torch.set_num_threads(1)
    # Manual default never advances, regardless of apparently ideal data/time.
    manual=module.ReadinessSampler()
    for iteration in (3000,6000,10000,40000,120000):
        add_block(manual,iteration,evidence(.9))
        assert manual.stage==0 and manual.schedule(iteration)==(0,1.)
    initial=manual.probabilities(120000)
    manual.request(stage=1,iteration=120000)
    assert manual.probabilities(120000)==initial
    midpoint=manual.probabilities(120500)
    assert midpoint != initial
    # An interrupted ramp continues, and reversing mid-ramp stays continuous.
    restored=clone(manual)
    assert restored.probabilities(120500)==midpoint
    restored.request(stage=0,iteration=120500)
    assert restored.probabilities(120500)==midpoint
    assert restored.probabilities(121500)==initial
    for it in range(120000,121001):
        p=manual.probabilities(it)
        assert min(p)>0 and abs(sum(p)-1)<1.e-9
        assert p[9:]==initial[9:]

    # Moderate success with a stable curve can promote: no high absolute floor.
    state=module.ReadinessSampler(mode='auto')
    for i in range(200,1200,200):add_block(state,i,evidence(.45))
    assert state.stage==0 and state.checks==1
    state2=clone(state)
    for s in (state,state2):add_block(s,1200,evidence(.45))
    assert state.stage==state2.stage==1 and state.transition_start==1200
    assert state.probabilities(1700)==state2.probabilities(1700)
    # Population settling blocks the next hold window until actual quotas agree.
    state.population_pending=.01;state.advance(2200)
    assert state.transition_start is not None
    state.population_pending=0;state.advance(2201)
    assert state.stable_since==2201 and not state.history

    # Clearly improving success, regression and missing comparable data hold.
    for kind in ('improving','regressing','missing'):
        s=module.ReadinessSampler(mode='auto')
        for i in range(200,2401,200):
            score=.30+i/10000 if kind=='improving' else .80-i/10000
            data=evidence(score)
            if kind=='missing':data[:]=0
            add_block(s,i,data)
        assert s.stage==0,kind
    # Changing only the mixture of task strata cannot manufacture a trend.
    s=module.ReadinessSampler(mode='auto')
    for i in range(200,1201,200):
        data=evidence(.45)
        data[:,:,0]*=(1+i/200)
        add_block(s,i,data)
    assert s.stage==1

    # A persistently weak group at a large population share retreats one step.
    weak=module.ReadinessSampler(mode='auto');weak.stage=4
    for i in range(200,801,200):add_block(weak,i,evidence(.6,hard=.10))
    assert weak.stage==3 and weak.transition_from==4 and weak.retry_lock
    weak.advance(1800)
    assert weak.stable_since==1800
    for i in range(2000,4001,200):add_block(weak,i,evidence(.6,hard=.10))
    assert weak.stage==3 and 'retry_cooldown' in weak.reasons
    for i in range(4200,6001,200):add_block(weak,i,evidence(.6,hard=.18))
    assert weak.stage==4 and weak.retry_lock is None

    # Preserve all provided episode labels and sufficient statistics exactly.
    acc=module.ReadinessSampler(mode='auto')
    acc.observe(torch.tensor([1,1]),torch.tensor([0,1]),torch.tensor([2,4]),
                torch.tensor([.5,.7]),torch.tensor([[1,0,0,1,1,.01,.02,.01],[0,1,0,.5,.5,.1,.03,.02]]))
    assert acc.totals[:,0].sum()==2 and (acc.totals[:,0]>0).sum()==2
    assert torch.equal(clone(acc).totals,acc.totals)

    # The ONLY compatible legacy source is an explicitly requested phase-zero
    # assignment. This tests metadata, not checkpoint loading/full resume.
    counts=layout.integer_sample_counts(layout.staged_sampling_probabilities(0),10584)
    types=torch.repeat_interleave(torch.arange(28),torch.tensor(counts))
    saved=dict(version=layout.LEGACY_SAMPLING_VERSION,initialized=True,iteration=2001)
    assert module.validate_legacy_migration(saved,types,True)==2001
    for allowed,iteration in [(False,2001),(True,3000),(True,7200)]:
        try:module.validate_legacy_migration(dict(saved,iteration=iteration),types,allowed)
        except ValueError:pass
        else:raise AssertionError('invalid migration accepted')
    init=loader.load_module('algorithms.initialization.stair_path_init')
    prior=dict(m20_profile='ascent_entry_v1_stage9',refinement_version=layout.LEGACY_SAMPLING_VERSION,
        stair_path_reward_settings=dict(stair_path_reward_weights={'position':-.3},path_tracking_target='effective',
            path_velocity_target='return',contact_slip='measured',merged_entry_slip='max'),
        stair_path_reward_version='unchanged_rewards',stair_path_config={'test_gain':1.},control_profile='legacy_v1_path_v1')
    target=dict(prior['stair_path_reward_settings'],version=layout.SAMPLING_VERSION,
        stair_path_reward_version=prior['stair_path_reward_version'],
        stair_path_config=prior['stair_path_config'],control_profile=prior['control_profile'])
    init.validate_sampling_only_migration(prior,target)
    try:init.validate_sampling_only_migration(prior,dict(target,stair_path_reward_version='changed'))
    except ValueError:pass
    else:raise AssertionError('reward migration accepted')
    # Graceful interruption saves only after the logger boundary, using no PPO
    # runner/checkpoint; the second request remains an immediate forced exit.
    stop=loader.load_module('utils.iteration_stop');events=[]
    runner=SimpleNamespace(current_learning_iteration=123,save=lambda path:events.append('save'))
    runner.logger=SimpleNamespace(log=lambda **kwargs:events.append('log'),writer=SimpleNamespace(flush=lambda:events.append('flush')))
    with stop.IterationStop(runner,'/tmp/unused_sampling_check') as stopping:
        stopping.request()
        assert not events
        runner.logger.log(it=123)
    assert events==['log','save','flush']
    try:stopping.request()
    except KeyboardInterrupt:pass
    else:raise AssertionError('second Ctrl+C did not force exit')
    print(json.dumps(dict(manual_no_timed_switch=True,continuous_reversible_ramps=True,
        moderate_success_plateau_promotes=True,improving_and_regressing_hold=True,
        missing_evidence_holds=True,stratum_mix_does_not_create_progress=True,
        low_success_demotion=True,retry_hysteresis=True,scheduler_state_roundtrip=True,
        legacy_phase_zero_migration_metadata=True,interrupt_saves_at_update_boundary=True,
        policy_checkpoints_loaded=0),indent=2))


if __name__=='__main__':main()
