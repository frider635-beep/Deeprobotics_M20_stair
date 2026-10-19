#!/usr/bin/env python3
"""Counterexamples for v4 route/stop feedback and pre-touchdown front coordination.

Synthetic state checks and real-MJCF controller parity; not policy acceptance.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from check_m20_curriculum import production_namespace
from check_m20_stop_v2 import fixture, advance, import_file
from check_stair_rewards import Entity

ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp'


def check_feedback():
    ns = production_namespace()
    env, _, wheel = fixture(ns, flat=False)
    cfg = Entity('robot', joint_ids=list(range(12)))
    scan = {'terrain_gate': torch.ones(1), 'up_gate': torch.zeros(1), 'down_gate': torch.ones(1)}
    ns['_stair_scan_context'] = lambda *args: scan
    command = env.command_manager.get_command('base_velocity')
    robot = env.scene['robot'].data
    command[:] = torch.tensor([[.7, 0, 0]])
    call = ns['heading_hold_velocity_commands']
    call(env)
    origin = env._m20_zero_hold_heading_state['straight_origin_xy'].clone()
    robot.root_pos_w[:, 1] += .20
    outputs=[]
    for _ in range(30):
        advance(env);out=call(env);outputs.append(float(out[0, 1]))
        assert torch.equal(call(env), out)
        assert float(out[0, 0]) == float(command[0, 0])
    assert outputs[-1] < 0 and max(abs(x) for x in outputs) <= .100001
    assert np.max(np.abs(np.diff(outputs))) <= .006001
    assert torch.equal(origin, env._m20_zero_hold_heading_state['straight_origin_xy'])
    cost = ns['stair_descent_straight_motion_cost'](env, Entity('robot'))
    assert cost > 0
    samples = env._m20_descent_motion_stats['samples'].clone()
    assert torch.equal(ns['stair_descent_straight_motion_cost'](env, Entity('robot')), cost)
    assert torch.equal(samples, env._m20_descent_motion_stats['samples'])
    robot.root_pos_w[:, 1] -= .40
    for _ in range(40):advance(env);opposite=call(env)
    assert opposite[0,1]>0
    # A human side-step or turn remains unmodified in every velocity channel.
    for request in ([0, .5, .2], [.5, 0, .6], [-.7, 0, 0]):
        command[:] = torch.tensor([request]);advance(env)
        assert torch.equal(call(env)[:, :2], command[:, :2])
    command.zero_();advance(env)
    for i in range(21):
        robot.root_pos_w[:, 0] += .002
        out=call(env)
        if i<20:assert not env._m20_heading_command_filter['stop_feedback'].any()
        advance(env)
    anchor = env._m20_zero_hold_heading_state['stop_origin_xy'].clone()
    robot.root_pos_w[:, :2] += torch.tensor([[.15, -.10]])
    robot.root_lin_vel_w[:, :2] = torch.tensor([[.04, -.03]])
    for _ in range(40):
        out=call(env);advance(env)
    assert out[0,0]<0 and out[0,1]>0
    assert torch.linalg.vector_norm(out[0,:2]) <= .100001
    assert torch.equal(anchor,env._m20_zero_hold_heading_state['stop_origin_xy'])
    assert ns['zero_command_stair_position_cost'](env,cfg)>0
    # The wheel target can execute the same longitudinal position recovery
    # without being punished by the independent zero-command wheel-action term.
    effective=call(env)
    targets=effective[:,:1]+effective[:,2:3]*torch.tensor([[-.28,.28,-.28,.28]])
    wheel.processed_actions[:]=-targets/ns['TUNING']['wheel_radius']
    assert ns['zero_command_wheel_action_cost'](env)<1.e-6
    env.episode_length_buf.fill_(1);advance(env);env.episode_length_buf.fill_(1)
    out=call(env)
    assert not env._m20_heading_command_filter['stop_feedback'].any()
    assert torch.equal(out[:,:2],torch.zeros(1,2))
    return {'descent_recovers_either_sign_without_direction_bias':True,
            'stop_braking_grace_latched_anchor_and_recovery':True,
            'manual_and_reverse_translation_passthrough':True,
            'feedback_targets_and_wheel_cost_agree':True,
            'duplicate_consumers_reset_and_motion_statistics':True}


def check_parity():
    sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
    d=import_file('v4_parity_deploy',ROOT/'deploy/deploy_mujoco/deploy_mujoco.py')
    args=SimpleNamespace(terrain='flat',terrain_xml=None)
    model=d.load_model(ROOT/'deep_robotics_model/M20/mjcf/M20.xml',args)
    contract=d.M20Contract(model,args.terrain_config)
    data=d.mujoco.MjData(model);contract.reset(data)
    controller=d.HeadingHoldController(contract)
    qp=int(contract.root_joint.qposadr[0]);qv=int(contract.root_joint.dofadr[0])
    ns=production_namespace();env,_,_=fixture(ns)
    robot=env.scene['robot'].data;command=env.command_manager.get_command('base_velocity')
    max_error=0.;seen_stop=False;seen_descent=False
    for step in range(360):
        terrain='ascent' if step<140 else 'descent' if step<300 else 'flat'
        request=[.7,0,0] if step<60 or 140<=step<200 else [0,0,0] if step<140 or 200<=step<270 else [0,.5,.6] if step<300 else [-.5,0,0]
        reset=step==320
        time=(step-320 if step>=320 else step)*.02
        yaw=.08*math.sin(step*.13);pitch=.12*math.sin(step*.09);roll=.05*math.sin(step*.12)
        cy,sy,cp,sp,cr,sr=math.cos(yaw/2),math.sin(yaw/2),math.cos(pitch/2),math.sin(pitch/2),math.cos(roll/2),math.sin(roll/2)
        data.qpos[qp:qp+3]=[1+.001*step,.05*math.sin(step*.05),.55]
        data.qpos[qp+3:qp+7]=[cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]
        data.qvel[qv:qv+6]=[.03*math.sin(step*.04),.04*math.cos(step*.03),0,.02,.03,.1*math.sin(step*.2)]
        data.time=time;d.mujoco.mj_forward(model,data)
        contract.terrain=SimpleNamespace(kind=terrain,parameters={'start_x':[0.]},goal_x=10.)
        ns['_stair_scan_context']=lambda *args, t=terrain:{'terrain_gate':torch.tensor([float(t!='flat')]),'up_gate':torch.tensor([float(t=='ascent')]),'down_gate':torch.tensor([float(t=='descent')])}
        rotation=data.xmat[contract.base_id].reshape(3,3);vel=np.zeros(6)
        d.mujoco.mj_objectVelocity(model,data,d.mujoco.mjtObj.mjOBJ_BODY,contract.base_id,vel,1)
        robot.root_pos_w[:]=torch.tensor(data.xpos[contract.base_id][None],dtype=torch.float32)
        robot.root_quat_w[:]=torch.tensor(data.xquat[contract.base_id][None],dtype=torch.float32)
        robot.root_lin_vel_w[:]=torch.tensor((rotation@vel[3:])[None],dtype=torch.float32)
        robot.root_lin_vel_b[:]=torch.tensor(vel[3:][None],dtype=torch.float32)
        robot.root_ang_vel_b[:]=torch.tensor(vel[:3][None],dtype=torch.float32)
        command[:]=torch.tensor([request]);env.common_step_counter=step;env.episode_length_buf.fill_(1 if reset else step+2)
        actual,info=controller.update(data,np.array(request,dtype=np.float32));expected=ns['heading_hold_velocity_commands'](env).numpy()[0]
        error=float(np.max(np.abs(actual-expected)));max_error=max(max_error,error)
        assert error<3.e-6,(step,actual,expected,info)
        assert np.array_equal(controller.update(data,np.array(request,dtype=np.float32))[0],actual)
        assert np.array_equal(ns['heading_hold_velocity_commands'](env).numpy()[0],expected)
        seen_stop |= info['stop_feedback_active'];seen_descent |= terrain=='descent' and info['path_active']
    assert seen_stop and seen_descent
    return {'real_mjcf_360_step_xyz_command_parity_max_error':max_error,
            'parity_covers_pitch_roll_descending_stops_manual_reversal_time_reset':True}


def check_front_pair():
    ns=production_namespace()
    package=ModuleType('_m20_v4_fixture');package.__path__=[]
    teacher=ModuleType('_m20_v4_fixture.stair_teacher');teacher.TUNING=ns['TUNING']
    teacher._stair_step_completion_events=lambda *args:None
    package.stair_teacher=teacher;sys.modules[package.__name__]=package;sys.modules[teacher.__name__]=teacher
    spec=importlib.util.spec_from_file_location(package.__name__+'.descent_front_pair',MDP/'descent_front_pair.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def scenario(drop=0.,roll=0.,side=0):
        env,_,_=fixture(ns,flat=False);env.command_manager.get_command('base_velocity')[:]=torch.tensor([[.7,0,0]])
        robot=env.scene['robot'].data
        robot.body_pos_w[:,:2,2]=.24;robot.body_pos_w[:,side,2]-=drop
        robot.root_quat_w[:]=torch.tensor([[math.cos(roll/2),math.sin(roll/2),0,0]])
        env._m20_stair_step_state=dict(active=torch.tensor([True]),up=torch.tensor([False]),down_successes=torch.zeros(1),edge_xy=torch.zeros(1,2),source_z=torch.tensor([.15]),target_z=torch.zeros(1),completed_count=torch.zeros(1,dtype=torch.long),front_support_paid=torch.zeros(1,2,dtype=torch.bool),front_source_clearance=torch.full((1,2),.09),heading=torch.tensor([[1.,0.]]))
        return env
    cfg=Entity('robot');sensor=Entity('height_scanner');contact=Entity('contact_forces')
    fn=lambda env:module.stair_descent_front_pair_cost(env,cfg,sensor,contact)
    aligned=scenario();assert fn(aligned)==0
    left=scenario(.05,math.radians(5),0);right=scenario(.05,-math.radians(5),1)
    lc,rc=float(fn(left)),float(fn(right));assert lc>0 and abs(lc-rc)<1.e-6
    # Both effects precede any touchdown: 5deg was entirely inside the old 6deg band.
    assert float(fn(scenario(0,math.radians(5))))>0
    assert float(fn(scenario(.05,0)))>0
    env=scenario();fn(env);env._m20_stair_step_state['front_support_paid'][0,1]=True
    for _ in range(20):advance(env);fn(env)
    assert env._m20_descent_front_pair_state['lag']>.08
    before=env._m20_descent_front_pair_state['stats_tensor'].clone();v=fn(env)
    assert torch.equal(before,env._m20_descent_front_pair_state['stats_tensor'])
    env.command_manager.get_command('base_velocity').zero_();advance(env);assert fn(env)==0
    assert env._m20_descent_front_pair_state['lag']==0
    env.command_manager.get_command('base_velocity')[:,0]=.7;advance(env);fn(env)
    assert env._m20_descent_front_pair_state['lag']==0
    env._m20_stair_step_state['front_support_paid'][:]=True;advance(env);assert fn(env)==0
    return {'pre_touchdown_asymmetry_and_5deg_roll_now_cost':True,
            'front_pair_left_right_mirror_cost':lc,
            'pair_pause_restart_grace_latched_contacts_and_duplicate_reads':True}


if __name__=='__main__':
    results={}
    for check in (check_feedback,check_parity,check_front_pair):results.update(check())
    print(json.dumps(results,indent=2))
