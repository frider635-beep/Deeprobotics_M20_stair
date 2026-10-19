#!/usr/bin/env python3
"""Physical-support counterexamples, dual-reference gradients and real PPO/resume."""
import copy
import json
import math
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage

from check_m20_v1_stop import ROOT, SRC, namespace, profile, check_config, functions
from check_m20_v1_stop_v3 import coverage
from check_m20_stop_v2 import fixture, advance, import_file
from check_stair_rewards import make_env, Entity
from check_stair_preservation import models, StairGuardedPPO
from check_m20_entry_pause import fixture as command_fixture, tick


def ns_v4():
    ns = namespace()
    profile.configure_tuning(ns['TUNING'], 'v1_stop_v4')
    ns['stop_coverage'] = coverage
    return ns


def support_env(ns, flat=True):
    env = make_env()
    env._m20_stop_support_ids = (list(range(4)), list(range(4)))
    if not flat:
        env.scene['robot'].data.body_pos_w[:, :2, 2] += .15
    return env


def check_support():
    ns = ns_v4(); env = support_env(ns)
    fn = ns['_zero_stop_support_context']
    assert fn(env)['flat'].item() and fn(env)['preview'].item()
    env.scene['height_scanner'].data.ray_hits_w[..., 2].zero_();advance(env)
    assert fn(env)['flat'].item() and not fn(env)['preview'].item()
    env.scene['robot'].data.body_pos_w[:, :2, 2] += .15;advance(env)
    assert not fn(env)['flat'].item()  # Mixed treads, even if scan misses them.
    env.scene['robot'].data.body_pos_w[:, :2, 2] -= .15
    contact = env.scene['contact_forces'].data
    for force in ([100.,0.,10.], [0.,0.,0.]):
        contact.net_forces_w[0,0] = torch.tensor(force);advance(env)
        assert not fn(env)['flat'].item()
    contact.net_forces_w[0,0] = torch.tensor([0.,0.,100.])
    contact.current_contact_time[0,0] = .02;advance(env)
    assert not fn(env)['flat'].item()
    contact.current_contact_time[0,0] = .1
    env.scene['robot'].data.projected_gravity_b[0,0] = .2;advance(env)
    assert not fn(env)['flat'].item()
    return dict(planar_preview_and_pure_flat_distinguished=True,
                mixed_treads_riser_contact_air_recent_contact_tilt_excluded=True)


def check_rewards():
    ns = ns_v4();env, leg, wheel = fixture(ns,flat=False)
    ns['_zero_stop_support_context'] = lambda env: dict(flat=torch.ones(1,dtype=torch.bool),
        visible=torch.ones(1,dtype=torch.bool),preview=torch.ones(1,dtype=torch.bool),flat_scan=torch.zeros(1))
    ns['_zero_stop_entry_terrain'] = lambda *args: torch.tensor([1])
    fns = functions(ns)
    for _ in range(60):
        for fn in fns:fn(env)
        advance(env)
    assert env._m20_zero_stop['context']['flat'].item()
    metrics=ns['zero_stop_metrics'](env,slice(None))
    assert metrics['support_preview_samples']>0 and metrics['support_flat_samples']==0
    ns['_zero_stop_support_context'] = lambda env: dict(flat=torch.ones(1,dtype=torch.bool),
        visible=torch.zeros(1,dtype=torch.bool),preview=torch.zeros(1,dtype=torch.bool),flat_scan=torch.zeros(1))
    advance(env)
    for fn in fns:fn(env)
    assert ns['zero_stop_metrics'](env,slice(None))['support_flat_holds']==1
    env.scene['robot'].data.root_lin_vel_w[:,0] = .003
    advance(env)
    assert float(fns[0](env))>0  # Previously below the 5mm/s allowance.
    env.scene['robot'].data.root_pos_w[:,0] += .01;advance(env)
    assert float(fns[1](env))>0  # Previously below the 2cm allowance.
    for request in ([.001,0,0],[.7,0,0],[-.7,0,0],[0,.4,0],[0,0,.3]):
        env.command_manager.get_command('base_velocity')[:] = torch.tensor([request]);advance(env)
        assert all(float(fn(env))==0 for fn in fns)
    # Compare every inherited posture/orientation reward on identical nonzero states.
    ns=ns_v4();env=support_env(ns);asset=Entity('robot',joint_ids=list(range(12)))
    robot=env.scene['robot'].data
    robot.projected_gravity_b[:,0]=.08
    robot.joint_pos=torch.ones(1,16)*.1;robot.default_joint_pos=torch.zeros(1,16)
    ns['joint_pos_penalty']=lambda *args:torch.ones(1)
    kwargs=dict(command_name='base_velocity',asset_cfg=asset,stand_still_scale=5.,
                velocity_threshold=.5,command_threshold=.1,stair_sensor_cfg=Entity('height_scanner'),stair_penalty_scale=.1)
    for request in ([.7,0,0],[-.7,0,0],[0,.4,0],[0,0,.3],[.001,0,0]):
        env.command_manager.get_command('base_velocity')[:] = torch.tensor([request]);advance(env)
        values=[]
        for enabled in (False,True):
            ns['TUNING']['zero_stop_v4']=enabled
            values.append((ns['stair_joint_pos_cost'](env,**kwargs),
                ns['stair_terrain_aware_orientation_cost'](env,asset,Entity('height_scanner')),
                ns['zero_stop_flat_stand_still'](env,asset)))
        assert all(torch.equal(a,b) for a,b in zip(*values))
    return dict(preview_gets_full_quiet_constraint=True,slow_drift_deadbands_tightened=True,
                zero_only_costs_and_nonzero_posture_rewards_unchanged=True)


def check_preview_sampler():
    command,env,_=command_fixture(probability=0.)
    ns=command._resample_command.__func__.__globals__;profile.configure_tuning(ns['TUNING'],'v1_stop_v4')
    ns['TUNING']['stair_zero_transition_probability']=0.
    ns['TUNING']['stair_exit_zero_probability']=0.
    ns['_zero_stop_support_context']=lambda env:dict(preview=torch.ones(1,dtype=torch.bool))
    # Fixed seed gives a selected first opportunity. Expiry resumes prior sign.
    torch.manual_seed(0)
    command.vel_command_b[:]=torch.tensor([[-.6,0.,0.]])
    tick(command,env)
    assert command._v4_preview_pending.item() and command.vel_command_b.abs().sum()==0
    duration=command.time_left.item();assert 6.<=duration<=8.
    tick(command,env);assert abs(command.time_left.item()-(duration-.02))<1.e-4
    command.time_left.fill_(.01);tick(command,env)
    assert -.8<=command.vel_command_b[0,0]<=-.4 and not command._v4_preview_pending.item()
    tick(command,env);assert command.vel_command_b[0,0]<0  # No repeated forced stop.
    command._v4_preview_pending.fill_(True);command.is_standing_env.fill_(True)
    command.vel_command_b[:]=torch.tensor([[0.,.4,.3]])
    tick(command,env)
    assert not command._v4_preview_pending.item() and command.vel_command_b[0,1]==.4
    command.reset(torch.tensor([0]));assert not command._v4_preview_decided.item()
    assert not command._v4_preview_pending.item()
    return dict(preview_pause_6_to_8s_once_per_episode=True,expiry_restarts_previous_direction=True,reset_clears=True)


def check_teacher_and_update():
    obs=TensorDict({'policy':torch.randn(4,244)*.05,'critic':torch.zeros(4,285),
        'stop_teacher':torch.randn(4,244)*.05,'stop_metadata':torch.ones(4,1),
        'stop_support':torch.tensor([[1.,1.,1.,.03],[0.,0.,1.,0.],[1.,0.,0.,0.],[1.,0.,1.,.03]])},batch_size=[4])
    obs['stop_metadata'][3]=0.
    a,c=models(obs)
    alg=StairGuardedPPO(a,c,RolloutStorage('rl',4,2,obs,[16]),
        stop_teacher_coefficient=.1,critic_warmup_iterations=0,num_learning_epochs=1,num_mini_batches=1)
    state=copy.deepcopy(a.state_dict());alg.set_reference(state);alg.set_stop_teacher(state)
    seen=[]
    hook=alg.stop_teacher_actor.register_forward_pre_hook(lambda module,args:seen.append(args[0]['policy'].clone()))
    loss=alg._stop_teacher_loss(obs);loss.backward();hook.remove()
    assert torch.allclose(seen[0][0,57:],torch.full((187,),.03))
    assert all(p.grad is None for p in alg.stop_teacher_actor.parameters())
    assert a.mlp[-1].bias.grad[:12].abs().sum()>0
    assert a.mlp[-1].bias.grad[12:].abs().sum()==0  # Teacher never copies wheel drift.
    excluded=obs[1:].clone()
    assert float(alg._stop_teacher_loss(excluded))==0  # Stair, braking, manual requests.
    for _ in range(2):
        with torch.no_grad():alg.act(obs);alg.process_env_step(obs,torch.ones(4),torch.zeros(4),{})
    alg.compute_returns(obs);result=alg.update()
    assert result['stop_teacher_sample_fraction']==.25
    assert all(math.isfinite(x) for x in result.values())
    a2,c2=models(obs)
    restored=StairGuardedPPO(a2,c2,RolloutStorage('rl',4,2,obs,[16]),stop_teacher_coefficient=.1)
    restored.load(alg.save(),None,True)
    assert restored.stop_teacher_actor is not None
    assert all(torch.equal(v,restored.stop_teacher_actor.state_dict()[k]) for k,v in state.items())
    return dict(clean_flat_scan_teacher_leg_gradients_only=True,
                mixed_stair_braking_manual_samples_excluded=True,real_ppo_update_and_dual_reference_resume=True)


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    config=check_config('v1_stop_v4')
    base=ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-15_11-59-41_refine_stop_heading_posture_v1/model_5100.pt'
    quiet=ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-17_11-50-54_v1_stop_stage1/model_900.pt'
    profile.validate_initialization(quiet,base,'v1_stop_v4',quiet)
    try:profile.validate_initialization(base,base,'v1_stop_v4',quiet)
    except ValueError:pass
    else:raise AssertionError('v4 must start from validated quiet model900')
    result=dict(config=config,support=check_support(),rewards=check_rewards(),
                preview_sampler=check_preview_sampler(),teacher=check_teacher_and_update())
    p=ROOT/'docs/v1_stop_v4_20261017/checks.json';p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
