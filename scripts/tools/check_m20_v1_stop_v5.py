#!/usr/bin/env python3
"""Reference latching, motion isolation, real PPO/resume and fast-stat parity."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from check_m20_v1_stop import ROOT, SRC, namespace, profile, functions
from check_m20_stop_v2 import fixture, advance, import_file
from check_m20_v1_stop_v3 import coverage
from check_stair_preservation import StairGuardedPPO
from check_stair_rewards import Scene

branch=import_file('m20_stop_branch_checks',SRC/'stop_branch.py')


def check_reference():
    n=3;cmd=torch.zeros(n,3);q=torch.zeros(n,16);xy=torch.zeros(n,2);yaw=torch.zeros(n);v=torch.zeros(n,2)
    safe=torch.tensor([True,True,False]);planar=torch.tensor([True,False,False]);ref=branch.StopReference()
    for _ in range(50):
        c=ref.update(cmd,q,xy,yaw,v,support=safe,planar_support=planar)
    assert c.shape==(n,24) and c[:,0].tolist()==[1.,1.,1.]
    assert c[:,1].tolist()==[1.,1.,0.] and c[:,23].tolist()==[1.,0.,0.]
    xy[:,0]+=.02;c=ref.update(cmd,q,xy,yaw,v,support=safe,planar_support=planar)
    assert torch.allclose(c[:,6],torch.full((n,),-.02))
    cmd[0,2]=.001;c=ref.update(cmd,q,xy,yaw,v,support=safe,planar_support=planar)
    assert c[0,0]==0 and not ref.state['held'][0]
    cmd[0]=0;c=ref.update(cmd,q,xy,yaw,v,support=safe,planar_support=planar)
    assert c[0,6]==0 and c[0,1]==0
    xy[:,0]+=.1;reset=torch.tensor([False,True,False])
    c=ref.update(cmd,q,xy,yaw,v,reset=reset,support=safe,planar_support=planar)
    assert c[1,6]==0 and c[2,6]<-.1
    wheels=torch.tensor([[[.4,.2,.1],[.4,-.2,.1],[-.4,.2,.25],[-.4,-.2,.25]]])
    force=torch.tensor([[[0.,0.,100.]]*4]);center=torch.zeros(1,2)
    assert branch.stable_support(wheels,force,center,center).item()
    assert not branch.stable_support(wheels,force,center,torch.ones(1,2)*5).item()
    force[...,2]=1.;assert not branch.stable_support(wheels,force,center,center).item()
    return dict(raw_zero_distinct_from_manual_turn=True,reference_does_not_chase_drift=True,
                per_environment_reset_and_restart_clear=True,mixed_treads_never_force_flat_leg_pose=True,
                unstable_support_excluded=True)


def make_models(obs):
    groups={'actor':['policy','stop_control'],'critic':['critic','stop_control']}
    actor=branch.StopBranchMLPModel(obs,groups,'actor',16,hidden_dims=[32,32],activation='elu',
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.1,'std_type':'log'})
    critic=MLPModel(obs,groups,'critic',1,hidden_dims=[32,32],activation='elu')
    return actor,critic


def check_isolation_and_ppo():
    obs=TensorDict({'policy':torch.randn(8,244)*.05,'critic':torch.randn(8,285)*.05,
                   'stop_control':torch.zeros(8,24)},batch_size=[8])
    obs['stop_control'][:4,0:2]=1.;obs['stop_control'][:2,23]=1.
    a,c=make_models(obs);before=copy.deepcopy(a.state_dict())
    motion_before=a(obs[4:]).detach().clone()
    alg=StairGuardedPPO(a,c,RolloutStorage('rl',8,2,obs,[16]),learning_rate=1.e-4,
        critic_warmup_iterations=1,stop_regularization_coefficient=.01,num_learning_epochs=1,num_mini_batches=1)
    def iteration():
        for _ in range(2):
            with torch.no_grad():alg.act(obs);alg.process_env_step(obs,torch.randn(8),torch.zeros(8),{})
        alg.compute_returns(obs);return alg.update()
    warm=iteration();assert warm['critic_warmup']==1 and all(torch.equal(v,a.state_dict()[k]) for k,v in before.items())
    for _ in range(3):result=iteration()
    assert all(math.isfinite(x) for x in result.values())
    assert all(torch.equal(v,a.state_dict()[k]) for k,v in before.items() if not k.startswith('stop_mlp.'))
    assert any(not torch.equal(v,a.state_dict()[k]) for k,v in before.items() if k.startswith('stop_mlp.'))
    assert torch.equal(motion_before,a(obs[4:]).detach())
    aa,cc=make_models(obs);restored=StairGuardedPPO(aa,cc,RolloutStorage('rl',8,2,obs,[16]),stop_regularization_coefficient=.01)
    restored.load(alg.save(),None,True)
    assert restored.guard_updates==4 and restored.reference_actor is None
    assert torch.equal(a(obs).detach(),aa(obs).detach())
    return dict(nonzero_mean_bitwise_unchanged_after_stop_updates=True,
                motion_mlp_and_sampling_std_frozen=True,real_ppo_warmup_and_stop_update=True,
                complete_resume_preserves_branch_and_counters=True)


def check_fast_statistics():
    configs=[]
    for fast in (False,True):
        ns=namespace();profile.configure_tuning(ns['TUNING'],'v1_stop_v4')
        ns['TUNING']['zero_stop_fast_stats']=fast
        ns['stop_coverage']=coverage
        env,leg,wheel=fixture(ns,flat=False)
        ns['_zero_stop_support_context']=lambda env:dict(flat=torch.ones(1,dtype=torch.bool),
            visible=torch.ones(1,dtype=torch.bool),preview=torch.ones(1,dtype=torch.bool),flat_scan=torch.zeros(1))
        ns['_zero_stop_entry_terrain']=lambda *args:torch.tensor([1])
        fns=functions(ns)
        for step in range(220):
            request=.6 if step<10 else -.5 if 90<=step<100 else 0.
            env.command_manager.get_command('base_velocity')[:]=torch.tensor([[request,0.,0.]])
            robot=env.scene['robot'].data;robot.root_lin_vel_w[:,0]=.01*(step%3)
            robot.root_pos_w[:,0]+=.0001
            leg.processed_actions.fill_(.002*(step%5));wheel.processed_actions.fill_(.01*(step%7))
            if step==150:env.episode_length_buf.fill_(1)
            for fn in fns:fn(env)
            advance(env)
        configs.append({g:{k:v.clone() for k,v in stats.items()} for g,stats in env._m20_zero_stop['stats'].items()})
    for g,stats in configs[0].items():
        for k,v in stats.items():assert torch.allclose(v,configs[1][g][k],rtol=1.e-6,atol=1.e-6),(g,k,v,configs[1][g][k])
    return dict(all_group_statistics_match_scalar_implementation=True,resets_and_direction_changes_included=True)


def check_curriculum():
    ns=namespace();profile.configure_tuning(ns['TUNING'],'v1_stop_v5')
    captured={};n=3
    def update(ids,up,down):captured.update(up=up.clone(),down=down.clone())
    sub={'pyramid_stairs':SimpleNamespace(proportion=.3),'pyramid_stairs_inv':SimpleNamespace(proportion=.3),
         'flat':SimpleNamespace(proportion=.4)}
    terrain=SimpleNamespace(terrain_levels=torch.zeros(n,dtype=torch.long),terrain_types=torch.tensor([0,3,6]),
        cfg=SimpleNamespace(terrain_generator=SimpleNamespace(size=(8.,8.),num_cols=10,sub_terrains=sub)),update_env_origins=update)
    scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[3.,0.,0.]]*n))))
    scene.terrain=terrain;scene.env_origins=torch.zeros(n,3)
    term=SimpleNamespace(episode_commanded_translation_m=torch.ones(n)*4.,
        episode_commanded_translation_w=torch.tensor([[4.,0.]]*n),vel_command_b=torch.zeros(n,3))
    env=SimpleNamespace(scene=scene,command_manager=SimpleNamespace(get_term=lambda _:term),
        _m20_zero_stop={'stats':{'stair':{'samples':torch.tensor([100.,0.,0.])}}})
    ns['stair_distance_terrain_levels'](env,torch.arange(n),.1)
    assert captured['up'].tolist()==[True,False,True]
    terrain.terrain_levels[0]=3
    ns['stair_distance_terrain_levels'](env,torch.arange(n),.1)
    assert captured['up'].tolist()==[False,False,True]
    return dict(flat_does_not_require_stair_samples=True,stairs_need_exposure=True,level_three_cap_preserved=True)


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    logging=import_file('m20_fast_logging_checks',SRC/'fast_logging.py')
    rows=[{'a':torch.tensor([1.,3.]),'b':torch.tensor(7.)},{'a':torch.tensor([5.]),'b':torch.tensor(9.)},{'a':torch.tensor([7.,9.])}]
    means=logging.episode_means(rows)
    assert means=={'a':5.,'b':8.}
    result=dict(reference=check_reference(),ppo=check_isolation_and_ppo(),fast_statistics=check_fast_statistics(),
                curriculum=check_curriculum(),fast_logging={'variable_length_missing_and_scalar_fields_preserved':True})
    out=ROOT/'docs/v1_stop_v5_20261017';out.mkdir(parents=True,exist_ok=True)
    (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
