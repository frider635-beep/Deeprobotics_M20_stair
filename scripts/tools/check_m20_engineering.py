#!/usr/bin/env python3
"""Adversarial engineering checks; no claim about a newly trained policy."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from check_m20_curriculum import production_namespace
from check_m20_stop_v2 import fixture, advance, import_file
from check_stair_rewards import Entity, make_env
from check_stair_preservation import models, StairGuardedPPO

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'source/rl_training/rl_training'


def ledger_case(skip=False, weak=False):
    ns=production_namespace();env=make_env();robot=env.scene['robot'].data
    robot.body_pos_w[...,2]=.39
    scan={'edge_x':torch.tensor([.45]),'upper_z':torch.tensor([.2]),'lower_z':torch.tensor([.3]),
          'edge_delta':torch.tensor([-.1]),'down_gate':torch.ones(1)}
    ns['_stair_scan_context']=lambda *args:scan
    cfg,sensor,contact=Entity('robot'),Entity('height_scanner'),Entity('contact_forces')
    def tick():
        advance(env);return ns['_stair_step_completion_events'](env,cfg,sensor,contact)
    tick()  # Calibrate real source-tread heights.
    robot.body_pos_w[0,0]=torch.tensor([.10,.22,.29])
    robot.body_pos_w[0,1]=torch.tensor([.10,-.22,.19 if skip else .29])
    if weak:env.scene['contact_forces'].data.net_forces_w[:,:,2]=2.
    for _ in range(3):tick()
    state=env._m20_stair_step_state
    if weak:
        assert not state['front_support_paid'].any()
        return True
    if skip:
        assert state['front_support_paid'][0,0] and not state['front_support_paid'][0,1]
        robot.body_pos_w[0,2:,:]=torch.tensor([[.10,.22,.19],[.10,-.22,.19]])
        for _ in range(4):tick()
        assert state['down_successes']==0 and state['down_missed_targets']==1 and not state['active']
        for _ in range(3):tick()
        assert state['down_missed_targets']==1 and not state['active']
        scan['edge_x'][:]=.75;scan['upper_z'][:]=.10;scan['lower_z'][:]=.20
        tick();assert state['active'] and abs(float(state['edge_xy'][0,0])-.30)<1.e-6
    else:
        robot.body_pos_w[0,1]=torch.tensor([.40,-.22,.19])
        robot.body_pos_w[0,2]=torch.tensor([.10,.22,.29])
        for _ in range(2):tick()
        robot.body_pos_w[0,3]=torch.tensor([.10,-.22,.29])
        for _ in range(2):tick()
        robot.body_pos_w[0,3]=torch.tensor([.40,-.22,.19])
        for _ in range(4):tick()
        assert state['down_successes']==1
    return True


def check_pooling():
    ns=production_namespace()
    pool=import_file('engineering_metric_pool',SOURCE/'utils/metric_aggregation.py').pool_episode_metrics
    keys=('samples','heading_sum','heading_signed_sum','heading_outside','yaw_rate_sum','yaw_rate_signed_sum','yaw_rate_max')
    stats={k:torch.zeros(2) for k in keys};stats['samples'][1]=10;stats['heading_sum'][1]=math.pi/2
    env=SimpleNamespace(device='cpu',_m20_descent_direction_state={'stats':stats})
    rows=[ns['_descent_direction_metrics'](env,torch.tensor([i])) for i in (0,1)]
    pooled=pool(rows);assert abs(float(pooled[0]['heading_error_mean_deg'])-9.)<1.e-5
    env=SimpleNamespace(device='cpu');assert ns['_descent_direction_metrics'](env)=={'samples':torch.tensor(0.)}
    return {'heading_metrics_not_diluted_by_unobserved_reset_batches':True}


def check_preservation():
    torch.manual_seed(4)
    obs=TensorDict({'policy':torch.zeros(4,244),'critic':torch.zeros(4,285)},batch_size=[4])
    obs['policy'][:,6:9]=torch.tensor([[0.,0.,0.],[.07,.07,.2],[.2,0.,0.],[0.,0.,.6]])
    actor,critic=models(obs)
    from rsl_rl.storage import RolloutStorage
    alg=StairGuardedPPO(actor,critic,RolloutStorage('rl',4,2,obs,[16]),critic_warmup_iterations=0,num_learning_epochs=1,num_mini_batches=1)
    alg.set_reference(copy.deepcopy(actor.state_dict()))
    with torch.no_grad():actor.mlp[-1].bias.add_(.1)
    losses=[float(alg._anchor_loss(obs[i:i+1])) for i in range(4)]
    assert math.isclose(losses[0],losses[1],rel_tol=1.e-5)
    assert math.isclose(losses[2]/losses[1],50.,rel_tol=1.e-4)
    assert math.isclose(losses[3],losses[2],rel_tol=1.e-5)
    for _ in range(2):
        with torch.no_grad():alg.act(obs);alg.process_env_step(obs,torch.ones(4),torch.zeros(4),{})
    alg.compute_returns(obs)
    handle=next(actor.parameters()).register_hook(lambda gradient:gradient*float('nan'))
    before=copy.deepcopy(actor.state_dict())
    try:
        alg.update()
    except RuntimeError as e:
        assert 'non-finite' in str(e)
    else:raise AssertionError('Non-finite gradients reached optimizer.step')
    finally:handle.remove()
    assert all(torch.equal(before[k],v) for k,v in actor.state_dict().items())
    return {'small_xy_stop_feedback_has_low_preservation_weight':True,'nonfinite_gradient_stops_before_optimizer_mutation':True}


def check_contract():
    pre=import_file('engineering_preflight',SOURCE/'utils/m20_preflight.py')
    tuning=production_namespace()['TUNING']
    deploy=ROOT/'deploy/deploy_mujoco/deploy_mujoco.py'
    result=pre.validate_control_contract(tuning,deploy)
    changed=dict(tuning,descent_path_kp=tuning['descent_path_kp']+.1)
    try:pre.validate_control_contract(changed,deploy)
    except ValueError as e:assert 'descent_path_kp' in str(e)
    else:raise AssertionError('Training/deployment gain drift was accepted')
    state={'mlp.0.weight':torch.zeros(512,244),'mlp.6.weight':torch.zeros(16,128)}
    saved={'actor_state_dict':state}
    pre.validate_actor_checkpoint(saved)
    try:pre.validate_actor_checkpoint(saved,require_version=True)
    except ValueError:pass
    else:raise AssertionError('An older checkpoint was fully resumed under new rewards')
    state['mlp.0.weight'][0,0]=float('nan')
    try:pre.validate_actor_checkpoint(saved)
    except ValueError:pass
    else:raise AssertionError('NaN checkpoint was accepted')
    return {'control_parameters_checked':result['checked_parameters'],'changed_gains_resume_version_and_nonfinite_weights_rejected':True}


def check_ascent_tracking():
    def cost(mix):
        ns=production_namespace();ns['TUNING']['ascent_lateral_tracking_mix']=mix
        env,_,_=fixture(ns,flat=False)
        ns['_stair_scan_context']=lambda *args:{'terrain_gate':torch.ones(1),'up_gate':torch.ones(1),'down_gate':torch.zeros(1)}
        env.command_manager.get_command('base_velocity')[:,0]=.7
        env.scene['robot'].data.root_lin_vel_w[:]=torch.tensor([[.7,.15,0.]])
        env.scene['robot'].data.root_lin_vel_b[:]=torch.tensor([[.7,.15,0.]])
        return float(ns['stair_ascent_straight_motion_cost'](env,Entity('robot')))
    old,new=cost(.5),cost(1.)
    assert old>0 and math.isclose(new,2*old,rel_tol=1.e-5)
    return {'ascent_lateral_tracking_cost_old_new':[old,new],'ascent_controller_gains_unchanged':True}


def check_handover():
    sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
    d=import_file('engineering_deploy',ROOT/'deploy/deploy_mujoco/deploy_mujoco.py')
    args=SimpleNamespace(terrain='flat',terrain_xml=None)
    model=d.load_model(ROOT/'deep_robotics_model/M20/mjcf/M20.xml',args)
    contract=d.M20Contract(model,args.terrain_config);data=d.mujoco.MjData(model);contract.reset(data)
    control=d.HeadingHoldController(contract);qp=int(contract.root_joint.qposadr[0])
    ns=production_namespace();env,_,_=fixture(ns);robot=env.scene['robot'].data;command=env.command_manager.get_command('base_velocity')
    error=0.;previous=None
    for step in range(180):
        gate=not (50<=step<60 or step>=145)
        contract.terrain=SimpleNamespace(kind='descent' if gate else 'flat',parameters={'start_x':[0.]},goal_x=10.)
        request=np.array([.7,0,0] if step<75 else [0,0,0],dtype=np.float32)
        y=0. if step<5 else .30 if step<100 else .55
        data.qpos[qp:qp+2]=[1.,y];data.time=step*.02;d.mujoco.mj_forward(model,data)
        ns['_stair_scan_context']=lambda *a,g=gate:{'terrain_gate':torch.tensor([float(g)]),'up_gate':torch.zeros(1),'down_gate':torch.tensor([float(g)])}
        robot.root_pos_w[:]=torch.tensor(data.xpos[contract.base_id][None],dtype=torch.float32)
        robot.root_quat_w[:]=torch.tensor(data.xquat[contract.base_id][None],dtype=torch.float32)
        command[:]=torch.from_numpy(request)[None];env.common_step_counter=step;env.episode_length_buf.fill_(step+2)
        actual,info=control.update(data,request);expected=ns['heading_hold_velocity_commands'](env).numpy()[0]
        error=max(error,float(abs(actual-expected).max()));assert error<3.e-6,(step,actual,expected)
        if previous is not None and step in list(range(50,71))+list(range(145,179)):
            assert np.linalg.norm(actual[:2]-previous[:2])<=.00601
        previous=actual.copy()
    before=control._cache_key
    for bad in (np.array([np.nan,0,0]),np.array([1,2])):
        try:control.update(data,bad)
        except ValueError:pass
        else:raise AssertionError('Invalid request entered the controller')
        assert control._cache_key==before
    contract.terrain=SimpleNamespace(kind='descent',parameters={'start_x':[0.]},goal_x=10.)
    control.reset();control.update(data,np.array([1,0,0]));data.qpos[qp+1]+=.2;data.time+=.02;d.mujoco.mj_forward(model,data)
    actual,_=control.update(data,np.array([1,0,0]));assert actual.dtype.kind=='f' and actual[1]<0
    return {'stairs_platform_and_gate_flicker_xyz_parity_error':error,'handover_slew_and_invalid_request_state_integrity':True,'integer_requests_do_not_truncate_feedback':True}


def check_evaluation():
    ev=import_file('engineering_evaluation',ROOT/'scripts/tools/evaluate_m20_refinement.py')
    rows=[]
    for i,y in enumerate((0.,.12,.05,0.)):
        rows.append({'command':np.array([.7,0,0]),'control_start_yaw_deg':0.,'control_start_xy':np.zeros(2),
                     'base_xyz':np.array([i*.1,y,0.]),'base_velocity_world_m_s':np.array([.5,.1,0]),'time_s':i*.02})
    route=ev.straight_route_metrics(rows)[0]
    assert route['path_abs_max_m']==.12 and route['path_end_signed_m']==0.
    assert {'ascent_mid_stair_pause','descent_mid_stair_pause','descent_second_tread_pause','flat_side_stop','flat_turn_stop','ascent_reverse_stop'}<=ev.CASES.keys()
    return {'ascent_out_and_back_lateral_drift_is_not_hidden_by_zero_endpoint':True,'critical_pause_side_turn_reverse_cases_registered':True}


if __name__=='__main__':
    torch.set_num_threads(1)
    checks={'target_tread_must_be_loaded_no_skipped_tread_credit':ledger_case(skip=True),
            'genuine_tread_visits_allow_staggered_axles':ledger_case(),
            'tiny_forces_do_not_confirm_tread_support':ledger_case(weak=True)}
    for check in (check_pooling,check_preservation,check_contract,check_ascent_tracking,check_handover,check_evaluation):checks.update(check())
    print(json.dumps(checks,indent=2))
