#!/usr/bin/env python3
"""Check native migration, descent gradients, shared feedback and real PPO."""
import argparse, copy, json, sys
from types import SimpleNamespace as NS
from pathlib import Path
import torch
import numpy as np
import mujoco
from tensordict import TensorDict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(42)
    path=d.training_module('control.stair_path')
    init=d.training_module('algorithms.initialization.stair_path_init')
    init.validate_source(args.checkpoint);actor=d.load_actor(args.checkpoint);source=d.load_actor(args.source)
    assert sum(isinstance(module,torch.nn.Sequential) for module in actor.modules())==3
    trainable=[name for name,param in actor.named_parameters() if param.requires_grad]
    assert all(name.startswith('unified_mlp.') for name in trainable)
    n=32
    obs=TensorDict(dict(policy=torch.randn(n,244), motion_control=torch.tensor([[.5,0.,0.],[.7,0.,0.]]*16),
        critic=torch.randn(n,285),ascent_control=torch.zeros(n,12)),batch_size=[n])
    # Four unambiguous classes: ascent, descent, flat, stopped on descent.
    up=torch.tensor([.7]*10+[.55]*3+[.4]*4)
    down=up.flip(-1)
    for a,b,scan in ((0,8,up),(8,16,down),(16,24,torch.full((17,),.5)),(24,32,down)):
        obs['policy'][a:b,57:]=scan[None,None].expand(b-a,11,17).reshape(b-a,187)
    obs['motion_control'][24:]=0.
    with torch.no_grad():
        expected=source(obs);actual=actor(obs)
        max_error=float((actual-expected).abs().max())
        assert max_error<2.e-5
        assert actor._descent_gate.max()==0.
        assert actor.descent_authority==0.
        assert actor._descent_gate[24:].max()==0.
    labelled=obs.clone()
    labelled['ascent_control'][:3,:2]=torch.tensor([[.6,.20/.35],[.6,.25/.35],[.6,.30/.35]])
    assert torch.allclose(actor.preservation_weights(labelled)[:3],torch.tensor([.05,.15,1.]))
    actor.set_update_iteration(430);assert actor.descent_authority==.5
    actor.set_update_iteration(830)
    actor(obs)[8:16].square().mean().backward()
    assert actor.unified_mlp[0].weight.grad.abs().sum()>0.,'Existing unified MLP must learn descent after gate ramp'
    assert actor._descent_gate[8:16].min()>.99
    actor.zero_grad(set_to_none=True)
    actor=d.load_actor(args.checkpoint)
    # Batch/scalar parity represents training/deployment equation parity.
    batch=path.StairPathController(4,'cpu');scalar=[path.StairPathController(1,'cpu') for _ in range(4)]
    maximum=0.
    for step in range(60):
        pos=torch.tensor([[step*.01,step*.002],[step*.01,-step*.002],[step*.01,step*.001],[0.,0.]])
        vel=torch.tensor([[.5,.10],[.7,-.10],[.5,.05],[0.,0.]])
        command=torch.tensor([[.5,0.,0.],[.7,0.,0.],[.5,0.,0.],[0.,0.,0.]])
        if step>=45: command[1,2]=.3
        scan=torch.stack((up,up,down,down))[:,None].expand(4,11,17).reshape(4,187)
        if step>=30: scan.fill_(.5)
        correction,features=batch.update(pos,vel,torch.zeros(4),command,scan,.02)
        for i,control in enumerate(scalar):
            c,f=control.update(pos[i:i+1],vel[i:i+1],torch.zeros(1),command[i:i+1],scan[i:i+1],.02)
            maximum=max(maximum,float((c-correction[i]).abs().max()),float((f-features[i:i+1]).abs().max()))
        assert correction[3]==0.
        if step>=45: assert correction[1]==0.
    assert maximum<1.e-6
    batch.reset(torch.arange(4));assert not batch.features.any() and not batch.correction.any()
    cfg=NS(terrain=None,terrain_xml=ROOT/'deploy/deploy_mujoco/terrains/stairs_ascent.xml',
        stair_height=None,tread_depth=None,stair_count=None,stair_start=None)
    mj_model=d.load_model(ROOT/'deep_robotics_model/M20/mjcf/M20.xml',cfg)
    contract=d.M20Contract(mj_model,cfg.terrain_config);data=mujoco.MjData(mj_model);contract.reset(data)
    hold=d.make_heading_hold(contract,actor);raw=np.array([.5,0.,0.],np.float32)
    command,_=hold.update(data,raw);features=contract._path_control.clone()
    repeat,_=hold.update(data,raw)
    assert np.array_equal(command,repeat) and torch.equal(features,contract._path_control)
    data.qvel[1]=.2;mujoco.mj_forward(mj_model,data)
    hold.update(data,raw)
    assert abs(float(contract._path_control[0,1]-features[0,1]))>.3
    hold.reset();assert not contract._path_control.any()
    # Native PPO must update the same unified MLP, with frozen inherited motion.
    assert set(actor.state_dict())==set(source.state_dict())
    assert all(actor.state_dict()[k].shape==v.shape for k,v in source.state_dict().items())
    assert sum(p.numel() for p in actor.parameters())==sum(p.numel() for p in source.parameters())
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    critic=MLPModel(obs,{'critic':['critic','motion_control','ascent_control']},'critic',1,
        hidden_dims=[512,256,128],activation='elu',obs_normalization=False)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    init.load_critic(critic,saved['critic_state_dict'])
    storage=RolloutStorage('rl',n,4,obs,[16],device='cpu')
    alg=d.training_module('algorithms.stair_guarded_ppo').StairGuardedPPO(actor,critic,storage,device='cpu',
        num_learning_epochs=2,num_mini_batches=2,learning_rate=1.e-5,schedule='fixed',critic_warmup_iterations=0)
    alg.set_reference(actor.state_dict());alg.guard_updates=830;baseline=init.frozen_digest(actor.state_dict())
    for update in range(2):
        for _ in range(4):
            with torch.inference_mode():
                alg.act(obs);alg.process_env_step(obs,torch.randn(n),torch.zeros(n,dtype=torch.long),{})
        with torch.inference_mode():alg.compute_returns(obs)
        losses=alg.update();assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.equal(actor.unified_mlp[0].weight,source.unified_mlp[0].weight)
    assert alg.reference_actor._reference_mode and alg.reference_actor.descent_authority==0.
    assert baseline==init.frozen_digest(actor.state_dict())
    # Preserve the two curricula, environment rows and native optimizer too.
    core=d.training_module('tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_unified')
    mdp=d.training_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_mdp')
    def make_env():
        terrain=NS(terrain_types=torch.floor(torch.arange(n)/(n/13)).long(),
            terrain_levels=torch.zeros(n,dtype=torch.long),terrain_origins=torch.zeros(6,13,3),
            env_origins=torch.zeros(n,3))
        return NS(num_envs=n,device='cpu',scene=NS(terrain=terrain),
            cfg=NS(ascent_adaptive_height_grid=[[.1]*13]*6),
            _ascent_adaptive_curriculum=core.BridgeCurriculum(n,'cpu'),reset=lambda:None)
    training_env=make_env()
    seeded=init.seed_curriculum(training_env,torch.load(args.source,map_location='cpu',weights_only=True))
    assert seeded['restored'] and training_env.scene.terrain.terrain_levels.max()>0
    compact=training_env.scene.terrain.terrain_types==2
    assert (training_env.scene.terrain.terrain_levels[compact]==0).all()
    assert (training_env._ascent_adaptive_curriculum.goals()[compact]==4).all()
    init.attach_curriculum_checkpoint(alg,training_env)
    course_state=mdp.descent_course(training_env)
    course_state.window[0,0,:4]=1;course_state.trials[0,0]=4;course_state.cursor[0,0]=4
    alg.m20_training_manifest=dict(curriculum_terrain_columns=13)
    state=alg.save()
    resumed=d.training_module('algorithms.stair_guarded_ppo').StairGuardedPPO(d.load_actor(args.checkpoint),
        copy.deepcopy(critic),storage,device='cpu',num_learning_epochs=2,num_mini_batches=2,
        learning_rate=1.e-5,schedule='fixed',critic_warmup_iterations=0)
    resumed_env=make_env();init.attach_curriculum_checkpoint(resumed,resumed_env)
    resumed.load(state)
    assert resumed.actor.descent_authority==1. and resumed.guard_updates==alg.guard_updates
    assert resumed.reference_actor._reference_mode
    assert all(torch.equal(value,resumed.actor.state_dict()[key]) for key,value in actor.state_dict().items())
    assert all(torch.equal(value,mdp.descent_course(resumed_env).state_dict()[key])
               for key,value in course_state.state_dict().items())
    assert torch.equal(training_env.scene.terrain.terrain_levels,resumed_env.scene.terrain.terrain_levels)
    bad_layout=copy.deepcopy(state);bad_layout['m20_training_manifest']['curriculum_terrain_columns']=10
    try:resumed.load(bad_layout)
    except ValueError:pass
    else:raise AssertionError('Mismatched terrain columns were silently resumed')
    mdp=d.training_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_mdp')
    reward_mdp=d.training_module('tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_alternating_mdp')
    observer=NS(reward_gate_key=-1,positive_gate=torch.tensor([True,True,True,True]),
        active_time=torch.ones(4),path_time=torch.ones(4),path_failed=torch.tensor([False,False,False,True]))
    scene=NS(terrain=NS(terrain_types=torch.tensor([0,9,8,9])))
    class Scene:
        terrain=scene.terrain
        def __getitem__(self,key):return NS(data=NS(root_lin_vel_w=torch.zeros(4,3)))
    env=NS(num_envs=4,device='cpu',common_step_counter=1,_ascent_entry_observer=observer,
        scene=Scene(),command_manager=NS(get_command=lambda name:torch.tensor([[.5,0.,0.]]*4)))
    paid=reward_mdp.positive_reward(env,lambda env:torch.ones(4),{},require_progress=True)
    assert paid.tolist()==[0.,0.,1.,0.], 'Stationary descent or a failed route collected positive tracking reward'
    empty_observer=NS(path_samples=torch.zeros(4),path_max=torch.zeros(4),path_signed=torch.zeros(4),
        path_vy_total=torch.zeros(4),path_slip_total=torch.zeros(4),path_saturated=torch.zeros(4),
        path_longitudinal_slip_total=torch.zeros(4),path_return_error_total=torch.zeros(4),
        path_failed=torch.zeros(4,dtype=torch.bool),down_success=torch.zeros(4,dtype=torch.bool))
    env._ascent_entry_observer=empty_observer
    values=mdp.metrics(env,torch.tensor([],dtype=torch.long))
    assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
    course=mdp.DescentCurriculum(2,'cpu');ids=torch.arange(2)
    for speed in (.5,.7):
        for _ in range(7):
            promote,_=course.update(ids,torch.ones(2,dtype=torch.bool),torch.full((2,),speed))
            assert not promote.any()
        promote,_=course.update(ids,torch.ones(2,dtype=torch.bool),torch.full((2,),speed))
    assert promote.all()
    copy_course=mdp.DescentCurriculum(2,'cpu');copy_course.load_state_dict(course.state_dict())
    result=dict(migration_max_action_error=max_error,actual_policy_mlps=3,trainable_parameters=trainable,
        descent_initial_output_zero=True,descent_has_gradient=True,actual_cpu_ppo_update=True,
        network_keys_shapes_count_identical=True,existing_unified_mlp_learns_descent=True,
        actor_input_dim=247,critic_input_dim=300,
        trainable_parameter_count=sum(p.numel() for p in actor.parameters() if p.requires_grad),frozen_motion_exact=True,
        batch_scalar_feedback_max_error=maximum,stop_manual_reset_checked=True,ppo_state_resume=True,
        mujoco_duplicate_and_velocity_edit_cache_checked=True,
        descent_both_speed_curriculum=True,descent_stationary_tracking_not_paid=True,empty_metrics_finite=True,
        curriculum_10_to_13_seed=True,both_course_state_resume=True,mismatched_layout_resume_rejected=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
