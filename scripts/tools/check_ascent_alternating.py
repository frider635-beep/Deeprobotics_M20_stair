#!/usr/bin/env python3
"""Stage6 counterexamples, resolved reward audit and native PPO integration."""
import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import tempfile
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deployment
BASE = 'tasks.manager_based.locomotion.velocity.mdp.ascent.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture-checkpoint', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(1); torch.manual_seed(42)
    core = deployment.training_module(BASE+'ascent_alternating')
    mdp = deployment.training_module(BASE+'ascent_alternating_mdp')
    init = deployment.training_module('algorithms.initialization.ascent_alternating_init')
    n = 2; on = torch.ones(n, 4, dtype=torch.bool); z = torch.zeros(n, 4); near = torch.ones(n, 2, dtype=torch.bool); active = torch.ones(n, dtype=torch.bool)
    def settle(book, position=z):
        for _ in range(9): result = book.update(on, position, z, near, active, .005)
        return result
    def swing(book, wheel, position=z):
        support=on.clone();support[:, wheel]=False
        pos=position.clone();pos[:, wheel]+=.04
        for _ in range(5): result=book.update(support,pos,torch.ones_like(z)*.2,near,active,.005)
        return result
    for lead in (0, 1):
        book=core.SwingAlternation(n,'cpu');assert settle(book)['valid'].all()
        for side in (lead, 1-lead, lead, 1-lead):
            result=swing(book,side);assert result['valid'].all() and not book.failed.any()
            settle(book)
        assert book.starts[:,0].tolist()==[4.,4.]
        # Independent rear clock can begin with either side at any front phase.
        for side in (1,0,1,0):
            assert swing(book,side+2)['valid'].all();settle(book)
        assert not book.failed.any()
    book=core.SwingAlternation(n,'cpu');settle(book);swing(book,0);settle(book)
    assert not swing(book,0)['valid'].any() and book.failed.all()
    before=book.violations.clone();swing(book,1);assert torch.equal(before,book.violations)
    book.reset(torch.tensor([0]));assert not book.failed[0] and book.failed[1] and (book.last_side[0]==-1).all()
    # Contact jitter, small vertical bobbing and inactive terrain earn no turns.
    book=core.SwingAlternation(n,'cpu');settle(book)
    off=on.clone();off[:,0]=False
    for _ in range(3):book.update(off,z+.04,z+.2,near,active,.005)
    settle(book)
    for _ in range(20):book.update(off,z,z,near,active,.005)
    assert book.starts.sum()==0 and not book.failed.any()
    book=core.SwingAlternation(n,'cpu');settle(book)
    both=on.clone();both[:,:2]=False
    for _ in range(5):book.update(both,z+.04,z+.2,near,active,.005)
    assert book.failed.all()
    book=core.SwingAlternation(n,'cpu')
    for _ in range(20):book.update(both,z+.04,z+.2,near,~active,.005)
    assert not book.engaged.any() and book.starts.sum()==0
    # One leg staying airborne is one swing, even after many physics ticks.
    book=core.SwingAlternation(n,'cpu');settle(book)
    for _ in range(50):book.update(off,z+.04,z+.2,near,active,.005)
    assert (book.starts[:,0]==1).all() and not book.failed.any()

    state=core.BalancedContactCurriculum(n,'cpu');ids=torch.arange(n);rows=torch.zeros(n,dtype=torch.long);refs=torch.tensor([False,True])
    for _ in range(24):
        state.speed_bucket.fill_(1);state.update(ids,active,active,rows,refs)
    assert not state.skill.any(), 'High-speed success promoted the untested low speed'
    for _ in range(8):
        state.speed_bucket.zero_();state.update(ids,active,active,rows,refs)
    assert (state.skill==1).all()
    snapshot=state.state_dict();restored=core.BalancedContactCurriculum(n,'cpu');restored.load_state_dict(snapshot)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in snapshot.items())
    try:restored.load_state_dict({k:v for k,v in snapshot.items() if k!='window'})
    except ValueError:pass
    else:raise AssertionError('Stage5 curriculum accepted as a Stage6 resume')
    for _ in range(3):state.update(ids,~active,active,rows,refs)
    assert (state.skill==0).all(), 'Failed reference courses could not reduce prefix'

    # Exercise the actual parallel geometry path, its row cap and references.
    count=8;course=core.BalancedContactCurriculum(count,'cpu');all_ids=torch.arange(count)
    all_true=torch.ones(count,dtype=torch.bool);row=torch.zeros(count,dtype=torch.long)
    reference=torch.zeros(count,dtype=torch.bool);reference[4]=True
    course.skill.fill_(2)  # all environments are solving six-tread tasks
    def pass_both_speeds():
        raised=torch.zeros(count,dtype=torch.bool)
        for speed in (0,1):
            course.speed_bucket.fill_(speed)
            for _ in range(8):
                up,down=course.update(all_ids,all_true,all_true,row,reference)
                row.add_(up.long()-down.long());raised |= up
        return raised
    raised=pass_both_speeds()
    assert raised.tolist()==[True,False,False,False,False,False,False,False]
    assert row.tolist()==[1,0,0,0,0,0,0,0]
    assert course.skill[0]==1 and course.skill[1]==3 and course.skill[4]==3
    # At 15 cm, even the practice group must finish the full flight to go higher.
    course=core.BalancedContactCurriculum(count,'cpu');course.skill.fill_(2);row.fill_(2)
    assert not pass_both_speeds().any() and (course.skill==3).all()
    raised=pass_both_speeds()
    assert raised.tolist()==[True,True,True,True,False,True,True,True]
    assert row[4]==2 and course.skill[4]==3  # fixed reference geometry
    # All environments return to four-tread practice after increasing geometry.
    assert (course.skill[~reference]==1).all()
    for _ in range(3):
        up,down=course.update(all_ids,~all_true,all_true,row,reference)
        row.add_(up.long()-down.long())
    assert (row[~reference]==2).all() and row[4]==2
    snapshot=course.state_dict();restored=core.BalancedContactCurriculum(count,'cpu')
    restored.load_state_dict(snapshot)
    assert torch.equal(restored.geometry_probe,course.geometry_probe)
    assert all(torch.equal(value,restored.state_dict()[key]) for key,value in snapshot.items())

    # Exercise the actual manager callback and its separate diagnostic groups.
    terrain=NS(terrain_levels=torch.zeros(8,dtype=torch.long),terrain_types=torch.arange(8))
    def move_origins(ids,up,down):
        terrain.terrain_levels[ids]+=up.long()-down.long()
    terrain.update_env_origins=move_origins
    course=core.BalancedContactCurriculum(8,'cpu');course.skill.fill_(2)
    course.trials.fill_(8);course.window.fill_(1)
    callback_env=NS(num_envs=8,device='cpu',scene=NS(terrain=terrain),
        cfg=NS(ascent_adaptive_training=True,ascent_sequence_step_count=18,
            ascent_adaptive_geometry=True,ascent_adaptive_height_grid=[[.10]*10,[.12]*10,[.15]*10],
            ascent_adaptive_depth_grid=[[.30]*10]*3),
        _ascent_adaptive_curriculum=course,
        _ascent_entry_observer=NS(prefix_ready=torch.ones(8,dtype=torch.bool),
            top_ready=torch.zeros(8,dtype=torch.bool),stats={'physics_samples':torch.ones(8)}),
        command_manager=NS(get_command=lambda _:torch.tensor([[.5,0.,0.]]*8)))
    metrics=mdp.curriculum(callback_env,torch.arange(8))
    assert terrain.terrain_levels.tolist()==[1,0,0,0,0,0,0,0]
    assert metrics['geometry_probe_terrain_level_mean']==1.
    assert metrics['full_flight_group_terrain_level_mean']==0.
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())

    cmd=torch.full((n,),.5);vel=torch.tensor([0.,.5]);zeros=torch.zeros(n)
    costs=core.dense_costs(cmd,vel,zeros,zeros,torch.zeros(n,4),on,cmd*.3,cmd*.6,zeros,torch.ones(n))
    assert costs['stall'][0]>costs['stall'][1] and costs['stall'][1]==0
    costs=core.dense_costs(cmd,cmd,torch.tensor([0.,.44]),zeros,torch.ones(n,4)*.3,on,cmd*.3,cmd*.6,zeros,torch.ones(n))
    assert costs['lateral'][1]>costs['lateral'][0] and (costs['front_slip']>0).all() and (costs['rear_slip']>0).all()
    cadence=core.CadenceTracker(n,'cpu');level=torch.ones(n,4,dtype=torch.long);target=torch.full((n,),.6)
    first=torch.ones(n,4);event,overdue=cadence.update(level,first,torch.ones(n),target,active)
    assert event.sum()==0
    second=first+torch.tensor([[.6]*4,[1.2]*4]);event,_=cadence.update(level+1,second,torch.tensor([1.6,2.2]),target,active)
    assert event[0]==0 and event[1]>0
    _,late=cadence.update(level+1,second,torch.full((n,),5.),target,active,goals=torch.full((n,),2))
    assert late.sum()==0, 'Completed prefix still charged for waiting for its rear wheels'

    # All positive terms, including generic tracking, obey the same condition.
    class Scene(dict): pass
    env=NS(num_envs=4,device='cpu',scene=Scene(),_ascent_entry_observer=NS(positive_gate=torch.tensor([False,True,False,False])))
    env.scene.terrain=NS(terrain_types=torch.tensor([0,0,8,9]));env.command_manager=NS(get_command=lambda _:torch.tensor([[.5,0,0]]*4))
    class Entity:
        def __init__(self):self.calls=0
        def resolve(self,scene):self.calls+=1
    entity=Entity()
    def original(env,asset_cfg):return torch.ones(4)
    params={'asset_cfg':entity}
    assert mdp.positive_reward(env,original,params).tolist()==[0.,1.,1.,1.]
    mdp.positive_reward(env,original,params);assert entity.calls==1
    assert (mdp.positive_reward(env,lambda env:-torch.ones(4),{})==-1).all()
    env.scene['robot']=NS(data=NS(root_lin_vel_w=torch.zeros(4,3)))
    env._ascent_entry_observer.active_time=torch.ones(4)
    assert mdp.positive_reward(env,original,params,require_progress=True).tolist()==[0.,0.,1.,1.]
    env.scene['robot'].data.root_lin_vel_w[:,0]=.5
    assert mdp.positive_reward(env,original,params,require_progress=True).tolist()==[0.,1.,1.,1.]
    weights=json.loads((args.source_checkpoint.parent/'params/launch_manifest.json').read_text())['reward_weights']
    terms=NS(**{name:NS(weight=weight,func=lambda env:torch.ones(4),params={}) for name,weight in weights.items()})
    positives=mdp.gate_positive_rewards(terms);assert positives==mdp.audit_positive_rewards(terms)
    assert 'track_lin_vel_xy_exp' in positives and 'track_ang_vel_z_exp' in positives
    untouched=terms.ascent_stage2_front_slip.func
    mdp.gate_positive_rewards(terms);assert terms.ascent_stage2_front_slip.func is untouched
    terms.test_ungated=NS(weight=1.,func=original,params={})
    try:mdp.audit_positive_rewards(terms)
    except ValueError:pass
    else:raise AssertionError('New ungated positive reward escaped audit')

    # Execute the actual profile bodies, with only Isaac config containers stubbed.
    profile_file=deployment.training_source('tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_alternating_profile.py')
    source=ast.parse(profile_file.read_text());scope=dict(PROFILE='ascent_entry_v1_stage6',VERSION='m20_ascent_alternating_20261021_v1_stage6_geom_v2',torch=torch,
        alternating=mdp,RewardTermCfg=NS,CurriculumTermCfg=NS,
        previous=NS(apply_profile=lambda *a,**k:None,configure_agent=lambda agent:{}),AlternatingVelocityCommand=object)
    for node in source.body:
        if isinstance(node,ast.FunctionDef):exec(compile(ast.Module(body=[node],type_ignores=[]),str(profile_file),'exec'),scope)
    del terms.test_ungated
    cfg=NS(m20_refinement_profile='ascent_entry_v1_stage5',commands=NS(base_velocity=NS()),rewards=terms,curriculum=NS())
    agent=NS(algorithm=NS());manifest=scope['apply_profile'](cfg,agent)
    assert manifest['gated_positive_rewards']==mdp.audit_positive_rewards(cfg.rewards)
    assert cfg.rewards.ascent_alternating_turn.weight<0 and cfg.rewards.ascent_adaptive_cadence.weight==0
    assert manifest['contact_demonstration_schedule']['coefficient']==0
    # Isaac's signature check also includes the explicit progress condition.
    import inspect
    assert set(inspect.signature(mdp.positive_reward).parameters)-{'env'}=={'source_func','source_params','require_progress'}

    # A promotion must not relabel a successful 2-tread episode as 4 treads.
    env.cfg=NS(ascent_sequence_step_count=18)
    env._ascent_adaptive_curriculum=core.BalancedContactCurriculum(4,'cpu')
    env._ascent_adaptive_curriculum.skill[:2]=1
    observer=env._ascent_entry_observer
    observer.episode_goals=torch.full((4,),2,dtype=torch.long)
    observer.prefix_ready=torch.ones(4,dtype=torch.bool);observer.top_ready=torch.zeros(4,dtype=torch.bool)
    observer.stats={'alternating_'+key:torch.zeros(4) for key in (
        'front_swing_starts','rear_swing_starts','front_repeat_or_simultaneous','rear_repeat_or_simultaneous','lateral_max')}
    diagnostic=mdp.metrics(env,None)
    assert diagnostic['__weighted_count__/vx05_prefix2_success']==2
    assert diagnostic['__weighted_count__/vx05_prefix4_success']==0

    provenance=init.validate_source(args.source_checkpoint)
    saved=torch.load(args.source_checkpoint,map_location='cpu',weights_only=True)
    actor=deployment.load_actor(args.source_checkpoint)
    branch=actor.sequence_source_digest()
    obs=TensorDict(dict(policy=torch.randn(16,244),motion_control=torch.tensor([[.5,0.,0.],[.7,0.,0.]]*8),
        critic=torch.randn(16,285),ascent_control=torch.zeros(16,12)),batch_size=[16])
    obs['policy'][:,57:]=torch.tensor([.7]*10+[.55]*3+[.4]*4)[None,None].expand(16,11,17).reshape(16,187)
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    critic=MLPModel(obs,{'critic':['critic','motion_control','ascent_control']},'critic',1,
        hidden_dims=[512,256,128],activation='elu',obs_normalization=False)
    init.load_initialized_critic(critic,saved['critic_state_dict'])
    learner=deployment.ascent_adaptive_module().AscentAdaptiveMLPModel(obs,{'actor':['policy','motion_control']},'actor',16,
        hidden_dims=[512,256,128],activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':1.,'std_type':'log'})
    init.load_initialized_actor(learner,saved)
    with torch.no_grad():assert torch.equal(actor(obs),learner(obs))
    storage=RolloutStorage('rl',16,4,obs,[16],device='cpu')
    ppo=deployment.training_module('stair_guarded_ppo').StairGuardedPPO(learner,critic,storage,device='cpu',
        num_learning_epochs=1,num_mini_batches=2,schedule='fixed',learning_rate=2e-5,critic_warmup_iterations=0)
    ppo.set_reference(learner.state_dict())
    demos=deployment.training_module('ascent_adaptive_demo');demos.attach_demonstrations(ppo,coefficient=0.)
    terrain=NS(terrain_levels=torch.zeros(16,dtype=torch.long),terrain_types=torch.arange(16)%8,
        terrain_origins=torch.zeros(6,10,3),env_origins=torch.zeros(16,3))
    fake_env=NS(num_envs=16,device='cpu',scene=NS(terrain=terrain),cfg=NS(ascent_adaptive_height_grid=[[.1]*10]*6),
        _ascent_adaptive_curriculum=core.BalancedContactCurriculum(16,'cpu'),reset=lambda:None)
    init.attach_curriculum_checkpoint(ppo,fake_env)
    before=learner.sequence_mlp[-1].weight.detach().clone()
    for _ in range(4):
        with torch.inference_mode():
            actions=ppo.act(obs);ppo.process_env_step(obs,torch.randn(16),torch.zeros(16,dtype=torch.long),{})
    with torch.inference_mode():ppo.compute_returns(obs)
    losses=ppo.update();assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert not torch.equal(before,learner.sequence_mlp[-1].weight) and learner.sequence_source_digest()==branch
    fixture=copy.deepcopy(saved);fixture['actor_state_dict']=learner.state_dict();fixture['m20_training_manifest'].update(m20_profile='ascent_entry_v1_stage6',refinement_version=scope['VERSION'])
    args.fixture_checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save(fixture,args.fixture_checkpoint)
    native=deployment.load_actor(args.fixture_checkpoint)
    with torch.no_grad():assert torch.equal(native(obs),learner(obs))
    init.validate_source(args.fixture_checkpoint)
    fake_env._ascent_adaptive_curriculum.skill[0]=2
    fake_env._ascent_adaptive_curriculum.window[0,0,:]=1
    state=ppo.save();assert state['ascent_contact_demonstrations'] is None
    assert 'window' in state['ascent_adaptive_curriculum']
    fake_env._ascent_adaptive_curriculum=core.BalancedContactCurriculum(16,'cpu')
    resumed=deployment.training_module('stair_guarded_ppo').StairGuardedPPO(native,critic,storage,device='cpu',
        num_learning_epochs=1,num_mini_batches=2,schedule='fixed',learning_rate=2e-5,critic_warmup_iterations=0)
    demos.attach_demonstrations(resumed,coefficient=0.)
    init.attach_curriculum_checkpoint(resumed,fake_env)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'resume.pt';torch.save(state,path)
        resumed.load(torch.load(path,map_location='cpu',weights_only=True))
    assert fake_env._ascent_adaptive_curriculum.skill[0]==2
    assert fake_env._ascent_adaptive_curriculum.window[0,0].sum()==8
    assert all(torch.equal(v,resumed.actor.state_dict()[k]) for k,v in learner.state_dict().items())
    result=dict(profile=scope['PROFILE'],source=provenance,alternating_either_first_side=True,front_rear_independent=True,
        repeated_side_and_simultaneous_rejected=True,contact_jitter_ignored=True,airborne_not_recounted=True,
        reset_isolated=True,all_positive_rewards_gated=manifest['gated_positive_rewards'],flat_descent_and_negative_costs_preserved=True,
        nested_scene_parameters_resolved_once=True,actual_profile_and_manager_signature_checked=True,
        both_speed_curriculum_and_resume=True,parallel_geometry_practice_and_row_cap=True,low_speed_slip_cadence_and_path_costs=True,
        actual_cpu_ppo_update=True,source_frozen=True,tracking_idle_reward_suppressed=True,stage6_native_checkpoint_loader_exact=True,
        no_old_demonstrations_loaded=True,gpu_training_started=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
