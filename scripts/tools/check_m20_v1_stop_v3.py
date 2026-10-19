#!/usr/bin/env python3
"""Geometry/coverage counterexamples and configuration ablation contract."""
from __future__ import annotations
import copy
import json
from types import SimpleNamespace

import torch
from check_m20_v1_stop import ROOT, SRC, profile, namespace, functions, check_config, check_controller_parity
from check_m20_stop_v2 import fixture, advance, import_file
from check_stair_rewards import make_env, Entity
from check_m20_entry_pause import fixture as command_fixture, tick

coverage = import_file('stop_coverage_checks', SRC/'stop_coverage.py')


def ns_v3():
    ns=namespace();profile.configure_tuning(ns['TUNING'],'v1_stop_v3')
    ns['stop_coverage']=coverage
    return ns


def check_classification():
    result=[]
    for sign in (1.,-1.):
        ns=ns_v3();env=make_env();env.scene['height_scanner'].data.ray_hits_w[...,2] *= sign
        for vx in (.6,0.,-.6):
            env.command_manager.get_command('base_velocity')[0,0]=vx
            advance(env)
            scan=ns['_stair_scan_context'](env,Entity('height_scanner'))
            kind=ns['_zero_stop_entry_terrain'](env,torch.tensor([False]),torch.tensor([True]))
            assert kind.item()==(1 if sign>0 else 2)
            if vx<=0:assert scan['up_gate'].item()==scan['down_gate'].item()==0
            result.append(dict(signed_height=sign,vx=vx,kind=kind.item()))
    ns=ns_v3();env=make_env();scan=ns['_stair_scan_context'](env,Entity('height_scanner'))
    classify=lambda s,flat=False,stairs=True:coverage.classify_terrain(env,s,torch.tensor([flat]),torch.tensor([stairs]))
    assert classify(scan).item()==1
    missing=copy.deepcopy(scan);missing['edge_geometry'].zero_()
    env.common_step_counter+=40
    assert classify(missing).item()==1  # 0.8s, local nonflat support only.
    env.common_step_counter+=20
    assert classify(missing).item()==4
    assert classify(missing,flat=True,stairs=False).item()==0
    assert classify(scan).item()==1
    env.episode_length_buf.fill_(1)
    assert classify(missing).item()==4
    env.episode_length_buf.fill_(5)
    # Equidistant opposing edges must not be assigned an invented direction.
    mixed=dict(ascending_edges_x=torch.tensor([-.1,.1]),edge_geometry=torch.ones(1,2,dtype=torch.bool),
               ascending_edges_upper_z=torch.tensor([[.1,-.1]]),ascending_edges_lower_z=torch.zeros(1,2))
    assert classify(mixed).item()==4
    return dict(real_zero_reverse_and_forward_geometry=result,
                missing_scan_memory_expires_and_reset_clears_it=True,ambiguous_edges_stay_unknown=True)


def check_coverage_and_rewards():
    for sign,terrain in ((1.,'stair_ascent'),(-1.,'stair_descent')):
        ns=ns_v3();real_scan=ns['_stair_scan_context'];env,leg,wheel=fixture(ns,flat=False)
        ns['_stair_scan_context']=real_scan
        env.scene['height_scanner'].data.ray_hits_w[...,2] *= sign
        fns=functions(ns);cmd=env.command_manager.get_command('base_velocity')
        def step(vx):
            cmd[:]=torch.tensor([[vx,0.,0.]])
            values=[float(fn(env)) for fn in fns]
            before=env._m20_stop_coverage.stats['entries'].clone()
            assert values==[float(fn(env)) for fn in fns]
            assert torch.equal(before,env._m20_stop_coverage.stats['entries'])
            advance(env)
            return values
        for _ in range(10):step(.6)
        wheel.processed_actions[:]=torch.tensor([[2.,2.,-2.,-2.]])
        for _ in range(270):step(0.)
        m=ns['zero_stop_metrics'](env,slice(None));prefix=f'coverage/{terrain}/after_forward/'
        assert m[prefix+'opportunities']==1 and m[prefix+'entries']==1
        assert m[prefix+'observed_4s_holds']==1
        assert m[prefix+'on_terrain_samples']>=200
        # Constant nonzero holding targets are free after their transition;
        # actual motion, drift and changing targets remain penalized.
        assert max(step(0.))==0
        env.scene['robot'].data.root_lin_vel_w[:,0]=.1
        assert step(0.)[0]>0
        env.scene['robot'].data.root_lin_vel_w.zero_()
        step(.6);env.scene['robot'].data.root_pos_w[:,0]+=.21;step(.6)
        m=ns['zero_stop_metrics'](env,slice(None))
        assert m[prefix+'restart_requests']==1 and m[prefix+'restart_progress']==1
        for _ in range(10):step(-.6)
        for _ in range(70):step(0.)
        # Rolling onto a platform retains the entry label in the old ledger,
        # but must not inflate actual-on-stairs exposure to a four-second hold.
        env.scene['height_scanner'].data.ray_hits_w[...,2].zero_()
        ns['_flat_surface_mask']=lambda *args:torch.ones(1,dtype=torch.bool)
        ns['_stair_activity_mask']=lambda *args:torch.zeros(1,dtype=torch.bool)
        for _ in range(230):step(0.)
        m=ns['zero_stop_metrics'](env,slice(None));reverse=f'coverage/{terrain}/after_reverse/'
        assert m[reverse+'entries']==1
        assert m[reverse+'measured_samples']>200
        assert m[reverse+'on_terrain_samples']<100 and m[reverse+'observed_4s_holds']==0
        assert m[reverse+'active_at_episode_end']==1
        env.episode_length_buf.fill_(1);step(0.)
        assert ns['zero_stop_metrics'](env,slice(None))[reverse+'entries']==0
        for request in ([.001,0,0],[0,.4,0],[0,0,.3]):
            cmd[:]=torch.tensor([request]);advance(env)
            assert all(float(fn(env))==0 for fn in fns)
    return dict(both_stair_directions_and_both_prior_commands_covered=True,
                constant_holding_targets_allowed_actual_motion_penalized=True,
                platform_rolloff_cannot_fake_four_second_stair_exposure=True,
                restart_requires_command_and_projected_progress=True,
                reset_and_duplicate_consumers_safe=True,manual_requests_excluded=True)


def check_sampler():
    command,env,_=command_fixture(probability=0.)
    ns=command._resample_command.__func__.__globals__
    profile.configure_tuning(ns['TUNING'],'v1_stop_v3')
    ns['TUNING']['stair_zero_transition_probability']=0.
    ns['TUNING']['stair_straight_practice_probability']=1.
    ns['TUNING']['stair_exit_zero_probability']=0.
    signs=[]
    for _ in range(300):
        command._resample(torch.tensor([0]));signs.append(float(command.vel_command_b[0,0]))
    reverse=sum(s<0 for s in signs)
    assert 35<reverse<115 and all(.4<=abs(s)<=.8 for s in signs)
    # Entry arms a short timer even when a long command was sampled on flat.
    command.time_left.fill_(10.);command.vel_command_b[:]=torch.tensor([[-.6,0.,0.]])
    command._v3_stair_timer_active.zero_();tick(command,env)
    assert 1.5<=command.time_left.item()<=3.
    prior=command.time_left.item();tick(command,env)
    assert abs(command.time_left.item()-(prior-.02))<1.e-5
    # A zero hold retains its full duration; turning requests are not shortened.
    command.vel_command_b.zero_();command.is_standing_env.fill_(True);command.time_left.fill_(7.)
    tick(command,env);assert command.time_left.item()>6.9
    command.is_standing_env.fill_(False);command.vel_command_b[:]=torch.tensor([[.6,0.,.5]])
    command.time_left.fill_(10.);tick(command,env);assert command.time_left.item()>9.9
    return dict(stair_reverse_samples=reverse,out_of=300,
                entry_timer_armed_once_zero_and_turn_not_shortened=True)


def check_ablation():
    check_config('v1_stop_v3');check_config('v1_stop_v3_target_control')
    primary=json.loads((ROOT/'docs/v1_stop_v3_20261017/effective_profile.json').read_text())
    control=json.loads((ROOT/'docs/v1_stop_v3_target_control_20261017/effective_profile.json').read_text())
    assert primary['algorithm']==control['algorithm']
    weights=control['reward_weights'].copy();assert weights.pop('zero_stop_wheel_target')==-.75
    assert weights==primary['reward_weights']
    left,right={},{}
    profile.configure_tuning(left,'v1_stop_v3');profile.configure_tuning(right,'v1_stop_v3_target_control')
    assert left==right
    return dict(only_reward_difference_is_wheel_target_amplitude=True,
                actor_noise_schedule_and_v1_motion_baseline_retained=True)


def check_batched_reset_and_restart():
    env=SimpleNamespace(num_envs=3,device='cpu',step_dt=.02,
        episode_length_buf=torch.tensor([3,3,3]),scene={'robot':SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(3,3),root_quat_w=torch.tensor([[1.,0.,0.,0.]]*3)))})
    cov=coverage.StopCoverage(env);kind=torch.tensor([1,2,0]);no=torch.zeros(3,dtype=torch.bool)
    cmd=torch.tensor([[.5,0.,0.],[-.5,0.,0.],[.5,0.,0.]])
    cov.update(env,cmd,kind,kind,no,no)
    cov.update(env,torch.zeros_like(cmd),kind,kind,no,no)
    entries=cov.stats['entries'].clone()
    env.episode_length_buf[0]=1
    cov.update(env,torch.zeros_like(cmd),kind,kind,no,no)
    assert torch.equal(cov.stats['entries'][1:],entries[1:])
    assert cov.stats['entries'][0,3]==0  # ascent/after_forward cleared
    assert cov.stats['entries'][0,5]==1  # reset stop is 'other'
    # Body +X at yaw pi points toward world -X.
    env.episode_length_buf.fill_(4)
    env.scene['robot'].data.root_quat_w[2]=torch.tensor([0.,0.,0.,1.])
    cov.update(env,cmd,kind,kind,no,no)
    env.scene['robot'].data.root_pos_w[2,0]=-.21
    cov.update(env,cmd,kind,kind,no,no)
    assert cov.stats['restart_progress'][2,0]==1
    return dict(reset_is_per_environment=True,restart_progress_uses_world_heading=True)


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    report=dict(classification=check_classification(),coverage_rewards=check_coverage_and_rewards(),
                sampler=check_sampler(),ablation=check_ablation(),batched=check_batched_reset_and_restart(),
                legacy_controller=check_controller_parity())
    out=ROOT/'docs/v1_stop_v3_20261017/checks.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
