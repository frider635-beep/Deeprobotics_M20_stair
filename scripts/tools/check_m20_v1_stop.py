#!/usr/bin/env python3
"""Stop-only attribution, reset/history and v1-control parity; no learning claim."""
from __future__ import annotations
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage
from check_m20_curriculum import production_namespace
from check_m20_stop_v2 import fixture, advance, import_file
from check_stair_rewards import Entity
from check_stair_preservation import models, StairGuardedPPO

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'source/rl_training/rl_training'
profile = import_file('m20_v1_stop_profile_checks', SRC / 'v1_stop_profile.py')


def namespace():
    ns = production_namespace()
    profile.configure_tuning(ns['TUNING'])
    return ns


def functions(ns):
    entities = dict(asset_cfg=Entity('robot'), leg_joint_cfg=Entity('robot', joint_ids=list(range(12))),
                    wheel_joint_cfg=Entity('robot', joint_ids=list(range(12, 16))))
    return [lambda env, f=ns[name]: f(env, **entities) for name in
            ('zero_stop_velocity_cost', 'zero_stop_position_cost', 'zero_stop_chatter_cost')]


def check_attribution():
    for flat in (True, False):
        ns = namespace()
        env, leg, wheel = fixture(ns, flat=flat)
        fns = functions(ns)
        for _ in range(60):
            for fn in fns: fn(env)
            advance(env)
        # Bad motion in a stop receives a signal, including opposing wheels.
        robot = env.scene['robot'].data
        robot.root_lin_vel_w[:, 1] = .08
        robot.joint_vel[:, 12:] = torch.tensor([[2., 2., -2., -2.]])
        leg.processed_actions.fill_(.1)
        assert sum(float(fn(env)) for fn in fns) > 0
        env.command_manager.get_command('base_velocity')[0,0] = .7
        assert all(float(fn(env)) == 0 for fn in fns)  # Same-tick resample is safe.
        for request in ([.7, 0, 0], [-.7, 0, 0], [0, .5, 0], [0, 0, .5], [.01, 0, 0], [0, 0, .01]):
            advance(env)
            env.command_manager.get_command('base_velocity')[:] = torch.tensor([request])
            assert all(float(fn(env)) == 0 for fn in fns), request
        command = env.command_manager.get_command('base_velocity')
        command.zero_()
        # A fresh stop cannot inherit the age of the previous one.
        advance(env)
        assert all(float(fn(env)) == 0 for fn in fns)
        assert env._m20_zero_stop['history'].item() == 1
    return {'new_costs_exactly_zero_for_forward_reverse_side_turn_and_small_manual_requests': True,
            'new_stop_does_not_inherit_previous_stop_age': True}


def check_history():
    ns = namespace()
    env, leg, wheel = fixture(ns, initial_step=500, initial_age=700, flat=False)
    fns = functions(ns)
    for i in range(80):
        leg.processed_actions.fill_(.02 if i % 2 else 0.)
        wheel.processed_actions.fill_(.5 if i % 2 else 0.)
        values = [float(fn(env)) for fn in fns]
        stats = copy.deepcopy(env._m20_zero_stop['stats'])
        assert values == [float(fn(env)) for fn in fns]
        assert all(torch.equal(v, env._m20_zero_stop['stats'][g][k]) for g, s in stats.items() for k, v in s.items())
        if i <= 20: assert max(values) == 0
        if i >= 41: assert values[2] > 0
        advance(env)
    origin = env._m20_zero_stop['origin'].clone()
    env.scene['robot'].data.root_pos_w[:, 0] += .1
    assert float(fns[1](env)) > 0
    assert torch.equal(origin, env._m20_zero_stop['origin'])
    # Episode reset must clear phase age, quiet counters and action history.
    advance(env); env.episode_length_buf.fill_(1)
    assert all(float(fn(env)) == 0 for fn in fns)
    assert env._m20_zero_stop['stats']['stair']['samples'].item() == 0
    assert env._m20_zero_stop['history'].item() == 1
    return {'brake_04s_ramp_history_random_initial_age_duplicate_consumers': True,
            'post_brake_position_anchor_never_chases_drift': True,
            'episode_reset_clears_history_and_statistics': True}


def check_quiet_and_pose():
    ns = namespace(); env, leg, wheel = fixture(ns, flat=False); fns = functions(ns)
    for _ in range(105):
        for fn in fns: fn(env)
        advance(env)
    metrics = ns['zero_stop_metrics'](env, slice(None))
    assert metrics['stair_holds_reached_quiet_1s_fraction'] == 1
    assert metrics['stair_success_fraction'] == 1
    leg.processed_actions.fill_(.02)
    for fn in fns: fn(env)
    assert env._m20_zero_stop['quiet_s'].item() == 0
    assert ns['zero_stop_metrics'](env, slice(None))['flat_samples'] == 0
    # A net-zero wheel mean cannot cancel physical wheel speed cost.
    advance(env)
    env.scene['robot'].data.joint_vel[:, 12:] = torch.tensor([[1., 1., -1., -1.]])
    assert float(fns[0](env)) > 0
    robot = env.scene['robot'].data
    robot.joint_pos = torch.ones(1,16) * .2
    robot.default_joint_pos = torch.zeros(1,16)
    pose = ns['zero_stop_flat_stand_still']
    cfg = Entity('robot', joint_ids=list(range(12)))
    assert float(pose(env, cfg)) == 0
    ns['_flat_surface_mask'] = lambda *args: torch.ones(1,dtype=torch.bool)
    ns['_stair_activity_mask'] = lambda *args: torch.zeros(1,dtype=torch.bool)
    assert float(pose(env, cfg)) > 0
    return {'quiet_one_second_requires_actual_leg_and_target_quiet': True,
            'per_wheel_motion_cannot_cancel': True, 'stairs_do_not_pull_legs_to_flat_default_pose': True}


def check_config(profile_name='v1_stop'):
    saved = profile.baseline()
    ns = namespace()
    profile.configure_tuning(ns['TUNING'], profile_name)
    ns['stair_descent_front_pair_cost'] = import_file('v1_stop_disabled_pair',
        SRC/'tasks/manager_based/locomotion/velocity/mdp/descent_front_pair.py').stair_descent_front_pair_cost
    # Build terms from the actual reward declarations, retaining inherited terms
    # from the recorded v1 config. No Isaac/Kit is required for this wiring check.
    class Term(SimpleNamespace):
        pass
    class EC:
        def __init__(self, name, **kw): self.name=name; self.__dict__.update(kw)
    import ast
    source = SRC/'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'M20StairTeacherRewardsCfg')
    bindings = dict(RewTerm=Term, SceneEntityCfg=EC, stair_teacher=SimpleNamespace(**ns),
                    _gait_entities=lambda: dict(asset_cfg=EC('robot'), hip_cfg=EC('robot'), knee_cfg=EC('robot'),
                                               sensor_cfg=EC('height_scanner'), contact_sensor_cfg=EC('contact_forces')))
    terms = {k:Term(func=saved['reward_functions'][k], weight=w, params={}) for k,w in saved['reward_weights'].items()}
    for node in cls.body:
        if isinstance(node, ast.Assign):
            terms[node.targets[0].id] = eval(compile(ast.Expression(node.value), str(source), 'eval'), bindings)
    cfg = SimpleNamespace(rewards=SimpleNamespace(**terms), curriculum=SimpleNamespace(),
        observations=SimpleNamespace(policy=SimpleNamespace(**{k:SimpleNamespace(noise=SimpleNamespace(n_min=-99,n_max=99)) for k in saved['policy_noise']})))
    agent = SimpleNamespace(algorithm=SimpleNamespace())
    manager = ModuleType('isaaclab.managers')
    manager.CurriculumTermCfg=manager.RewardTermCfg=Term;manager.SceneEntityCfg=EC
    manager.ObservationGroupCfg=manager.ObservationTermCfg=Term
    utils = ModuleType('isaaclab.utils');utils.configclass=lambda cls:cls
    mdp = ModuleType('rl_training.tasks.manager_based.locomotion.velocity.mdp')
    mdp.stair_teacher=SimpleNamespace(**ns)
    old_smooth=object();mdp.rewards=SimpleNamespace(action_smooth_l2=old_smooth)
    with patch.dict(sys.modules, {'isaaclab.managers':manager, 'isaaclab.utils':utils,
              'rl_training.tasks.manager_based.locomotion.velocity.mdp':mdp}):
        manifest = profile.apply_profile(cfg, agent, profile_name)
    removed = ['stair_ascent_straight_motion_cost', 'stair_descent_straight_motion_cost',
               'stair_descent_front_pair_cost', 'zero_command_action_dynamics_cost', 'zero_command_stair_position_cost']
    assert all(getattr(cfg.rewards,n,None) is None for n in removed)
    assert cfg.rewards.action_smooth_l2.func is old_smooth
    assert cfg.rewards.stair_descent_wheel_shank_cost.weight == -3.
    assert cfg.rewards.stair_descent_entry_rear_early_transfer_cost.weight == -3.
    assert cfg.rewards.stand_still.func is ns['zero_stop_flat_stand_still']
    assert cfg.rewards.zero_stop_chatter.weight == -.1
    assert cfg.rewards.zero_stop_position.weight == -.5
    assert agent.algorithm.anchor_stair_weight == 1. and agent.algorithm.anchor_stop_weight == .02
    active = {n:t.weight for n,t in vars(cfg.rewards).items() if t is not None and t.weight != 0}
    path = ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-15_11-59-41_refine_stop_heading_posture_v1/model_5100.pt'
    profile.validate_initialization(str(path), str(path))
    weights_enabled = profile_name in ('v1_stop_v2', 'v1_stop_v3_target_control')
    assert (getattr(cfg.rewards,'zero_stop_wheel_target',None) is not None) == weights_enabled
    if profile.uses_stop_metadata(profile_name):
        assert cfg.observations.stop_metadata.raw_zero.func is ns['zero_stop_metadata']
    out = ROOT/f'docs/{profile_name}_20261017/effective_profile.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(manifest=manifest,reward_weights=active,algorithm=vars(agent.algorithm)),indent=2)+'\n')
    return {'current_reward_declarations_restored_to_recorded_v1_weights': True,
            'later_motion_pair_and_stacked_stop_rewards_disabled': removed,
            'recorded_model5100_sha256_checked': True, 'active_reward_terms':len(active)}


def check_preservation():
    obs = TensorDict({'policy':torch.zeros(3,244),'critic':torch.zeros(3,285)},batch_size=[3])
    obs['policy'][0,6]=.7;obs['policy'][1,6]=.7
    obs['policy'][1,57]=.2;obs['policy'][2,57]=.2
    a,c=models(obs)
    alg=StairGuardedPPO(a,c,RolloutStorage('rl',3,2,obs,[16]),anchor_stair_weight=1.,anchor_stop_weight=.02,
                        critic_warmup_iterations=0,num_learning_epochs=1,num_mini_batches=1)
    alg.set_reference(copy.deepcopy(a.state_dict()))
    with torch.no_grad(): a.mlp[-1].bias.add_(.1)
    losses=[float(alg._anchor_loss(obs[i:i+1])) for i in range(3)]
    assert math.isclose(losses[0],losses[1],rel_tol=1.e-4)
    assert math.isclose(losses[1]/losses[2],50.,rel_tol=1.e-4)
    for _ in range(2):
        with torch.no_grad():
            alg.act(obs);alg.process_env_step(obs,torch.ones(3),torch.zeros(3),{})
    alg.compute_returns(obs)
    updated=alg.update()
    assert updated['anchor']>0 and all(math.isfinite(x) for x in updated.values())
    return {'moving_stair_preservation_equals_flat_movement': True,'stop_changes_remain_possible':True,
            'real_ppo_update_with_profile_sample_weights':True}


def check_controller_parity():
    sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
    d=import_file('v1_stop_parity_deploy',ROOT/'deploy/deploy_mujoco/deploy_mujoco.py')
    args=SimpleNamespace(terrain='flat',terrain_xml=None)
    model=d.load_model(ROOT/'deep_robotics_model/M20/mjcf/M20.xml',args)
    contract=d.M20Contract(model,args.terrain_config);data=d.mujoco.MjData(model);contract.reset(data)
    controller=d.HeadingHoldController(contract,legacy=True)
    ns=namespace();env,_,_=fixture(ns)
    robot=env.scene['robot'].data;command=env.command_manager.get_command('base_velocity')
    qp=int(contract.root_joint.qposadr[0]);qv=int(contract.root_joint.dofadr[0]);error=0.
    for step in range(240):
        terrain='flat' if step<60 else 'ascent' if step<150 else 'descent'
        request=np.array([.7,0,0] if step<30 or 60<=step<100 or 150<=step<180 else [0,0,0] if step<60 or 100<=step<150 or 180<=step<210 else [0,.5,.6] if step<225 else [-.7,0,0],dtype=np.float32)
        yaw=.1*math.sin(step*.07);pitch=.15*math.sin(step*.04);roll=.06*math.sin(step*.09)
        cy,sy,cp,sp,cr,sr=math.cos(yaw/2),math.sin(yaw/2),math.cos(pitch/2),math.sin(pitch/2),math.cos(roll/2),math.sin(roll/2)
        data.qpos[qp:qp+3]=[1.,.03*math.sin(step*.05),.55]
        data.qpos[qp+3:qp+7]=[cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]
        data.qvel[qv:qv+6]=[.02,.01,0,.02,.03,.1*math.sin(step*.2)]
        data.time=step*.02;d.mujoco.mj_forward(model,data)
        contract.terrain=SimpleNamespace(kind=terrain,parameters={'start_x':[0.]},goal_x=10.)
        ns['_stair_scan_context']=lambda *args,t=terrain:{'terrain_gate':torch.tensor([float(t!='flat')]),'up_gate':torch.tensor([float(t=='ascent')]),'down_gate':torch.tensor([float(t=='descent')])}
        rotation=data.xmat[contract.base_id].reshape(3,3);vel=np.zeros(6)
        d.mujoco.mj_objectVelocity(model,data,d.mujoco.mjtObj.mjOBJ_BODY,contract.base_id,vel,1)
        robot.root_pos_w[:]=torch.tensor(data.xpos[contract.base_id][None],dtype=torch.float32)
        robot.root_quat_w[:]=torch.tensor(data.xquat[contract.base_id][None],dtype=torch.float32)
        robot.root_lin_vel_w[:]=torch.tensor((rotation@vel[3:])[None],dtype=torch.float32)
        robot.root_ang_vel_b[:]=torch.tensor(vel[:3][None],dtype=torch.float32)
        command[:]=torch.tensor(request[None]);env.common_step_counter=step;env.episode_length_buf.fill_(step+2)
        actual,info=controller.update(data,request)
        expected=ns['heading_hold_velocity_commands'](env).numpy()[0]
        error=max(error,float(np.abs(actual-expected).max()))
        assert error<3.e-6,(step,actual,expected)
        assert np.array_equal(actual[:2],request[:2])
        assert not info['path_active'] and not info['stop_feedback_active']
    assert d.use_checkpoint_control_profile(SimpleNamespace(legacy_heading_hold=False),SimpleNamespace(m20_training_manifest={'control_profile':'legacy_v1'}))
    return {'real_mjcf_240_step_legacy_control_parity_max_error':error,
            'no_route_or_stop_XY_command_injection':True,'checkpoint_automatically_selects_matching_motion_controller':True}


def check_acceptance():
    ev=import_file('v1_stop_acceptance_counterexamples',ROOT/'scripts/tools/evaluate_m20_v1_stop.py')
    m=dict(complete_profile=True,fell=False,near_edge_samples=0,nonwheel_ground_contact_samples=0,
           zero_phases=[dict(stair_samples_after_brake=150,max_quiet_duration_s=2.,max_position_drift_m=.01)],
           moving_route_error_m=dict(samples=100,max=.1,mean=.05),moving_max_abs_yaw_deg=3.,
           moving_route_progress_m=dict(max=7.),moving_end_x_m=7.,
           moving_segments=[dict(direction=1.,end_progress_m=7.)],crossed_finish=True,
           physics_contact_substeps=dict(rear_front_wheel_contact_physics_substeps=0,rear_wheel_front_leg_contact_physics_substeps=0))
    base=dict(metrics=m,gates={'zero_hold_samples_available':True})
    assert all(ev.acceptance(base,copy.deepcopy(base),'ascent_mid_stair_pause').values())
    stopped=copy.deepcopy(base)
    stopped['metrics']['moving_route_progress_m']['max']=0.
    stopped['metrics']['moving_segments'][0]['end_progress_m']=0.
    assert not ev.acceptance(base,stopped,'ascent_hold')['each_motion_and_restart_progress_preserved']
    drift=copy.deepcopy(base);drift['metrics']['moving_route_error_m']['max']=.4
    assert not ev.acceptance(base,drift,'ascent_hold')['moving_max_lateral_not_worse_002m']
    no_stair=copy.deepcopy(base);no_stair['metrics']['zero_phases'][0]['stair_samples_after_brake']=0
    assert not ev.acceptance(base,no_stair,'ascent_mid_stair_pause')['stair_stop_actually_exercised']
    return {'acceptance_rejects_refusal_to_move_lateral_regression_and_missing_stair_hold':True}


def main():
    torch.set_num_threads(1)
    results={}
    for check in (check_attribution,check_history,check_quiet_and_pose,check_config,check_preservation,check_controller_parity,check_acceptance):
        results.update(check())
    (ROOT/'docs/v1_stop_20261017/checks.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
