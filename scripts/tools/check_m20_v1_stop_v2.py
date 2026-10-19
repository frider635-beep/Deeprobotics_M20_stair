#!/usr/bin/env python3
"""Production-body counterexamples and real RSL-RL checks; no learning claim."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from check_m20_v1_stop import ROOT, SRC, profile, namespace, functions, check_controller_parity
from check_m20_stop_v2 import fixture, advance, import_file
from check_stair_rewards import Entity, Scene
from check_stair_preservation import StairGuardedPPO

Actor = import_file('stop_aware_actor_checks', SRC/'stop_aware_actor.py').StopAwareMLPModel


def ns_v2():
    ns = namespace()
    profile.configure_tuning(ns['TUNING'], 'v1_stop_v2')
    return ns


def check_rewards():
    ns = ns_v2()
    env, leg, wheel = fixture(ns, flat=False)
    fn = lambda: ns['zero_stop_wheel_target_cost'](env, Entity('robot'),
        Entity('robot', joint_ids=list(range(12))), Entity('robot', joint_ids=list(range(12,16))))
    ns['_stair_scan_context'] = lambda *args: dict(terrain_gate=torch.ones(1), up_gate=torch.ones(1), down_gate=torch.zeros(1))
    for _ in range(70):
        fn(); advance(env)
    wheel.processed_actions.fill_(-.1/ns['TUNING']['wheel_radius'])
    constant = float(fn())
    assert constant > 0  # No target delta and no actual motion is needed.
    advance(env)
    wheel.processed_actions[:] = torch.tensor([[-1.,1.,-1.,1.]])
    assert float(fn()) > 0
    advance(env)
    wheel.processed_actions.zero_()
    yaw = -.1
    import math
    env.scene['robot'].data.root_quat_w[:] = torch.tensor([[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]])
    effective = ns['heading_hold_velocity_commands'](env)[:,2:3]
    wheel.processed_actions[:] = -effective*torch.tensor([[-.28,.28,-.28,.28]])/ns['TUNING']['wheel_radius']
    assert float(fn()) < 1.e-6
    # The terrain of a stop is latched even when it rolls onto a platform.
    advance(env)
    ns['_flat_surface_mask'] = lambda *args: torch.ones(1,dtype=torch.bool)
    ns['_stair_activity_mask'] = lambda *args: torch.zeros(1,dtype=torch.bool)
    ns['_stair_scan_context'] = lambda *args: dict(terrain_gate=torch.zeros(1),up_gate=torch.zeros(1),down_gate=torch.zeros(1))
    fn()
    assert env._m20_zero_stop['stats']['stair_ascent']['samples'].item() > 0
    assert env._m20_zero_stop['stats']['flat']['samples'].item() == 0
    command = env.command_manager.get_command('base_velocity')
    for request in ([.7,0,0],[-.7,0,0],[0,.5,0],[0,0,.3],[.001,0,0]):
        command[:] = torch.tensor([request]); advance(env)
        assert float(fn()) == 0
        assert ns['zero_stop_metadata'](env).item() == 0
    command.zero_(); advance(env); fn()
    assert env._m20_zero_stop['entry_terrain'].item() == 0
    assert ns['zero_stop_metadata'](env).item() == 1
    return dict(constant_drive_cost=constant, opposing_wheels_detected=True,
                yaw_correction_allowed=True, nonzero_commands_excluded=True, stop_terrain_latched=True)


def check_pose():
    ns = ns_v2(); env,_,_ = fixture(ns, flat=False)
    env.scene['robot'].data.root_lin_vel_b = torch.zeros(1,3)
    # Supply the inherited helper's known output; test only the new multiplier release.
    ns['joint_pos_penalty'] = lambda *args: torch.tensor([5.])
    cfg=Entity('robot',joint_ids=list(range(12)))
    def cost():
        return float(ns['stair_joint_pos_cost'](env,command_name='base_velocity',asset_cfg=cfg,
            stand_still_scale=5.,velocity_threshold=.5,command_threshold=.1,
            stair_sensor_cfg=Entity('height_scanner'),stair_penalty_scale=1.))
    assert cost() == 1.
    command=env.command_manager.get_command('base_velocity')
    command[0,0]=.7
    assert cost() == 5.
    command.zero_()
    ns['_stair_activity_mask']=lambda *args: torch.zeros(1,dtype=torch.bool)
    assert cost() == 5.
    return dict(only_raw_zero_stair_stationary_amplification_released=True)


def check_curriculum():
    ns=ns_v2(); n=5; captured={}
    term=SimpleNamespace(episode_commanded_translation_m=torch.ones(n)*4,
        episode_commanded_translation_w=torch.tensor([[4.,0.]]*n))
    terrain=SimpleNamespace(cfg=SimpleNamespace(terrain_generator=SimpleNamespace(size=(8.,8.))),
        terrain_levels=torch.tensor([1,1,1,3,1]),
        update_env_origins=lambda ids,up,down: captured.update(up=up,down=down))
    scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[3.,0.,0.]]*n))))
    scene.terrain=terrain;scene.env_origins=torch.zeros(n,3)
    env=SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_term=lambda name:term),
        _m20_zero_stop=dict(stats=dict(stair=dict(samples=torch.tensor([100.,0.,100.,100.,100.]),
            success=torch.tensor([80.,0.,10.,80.,80.]),drift=torch.tensor([2.,0.,2.,2.,10.])))))
    ns['stair_distance_terrain_levels'](env,torch.arange(n),min_commanded_translation_m=.1)
    assert captured['up'].tolist() == [True,False,False,False,False]
    assert not captured['down'].any()
    return dict(coverage_success_drift_and_stage_cap_required_for_promotion=True)


def make_models(obs, large=False):
    groups=dict(actor=['policy'],critic=['critic']); dims=[512,256,128] if large else [32,32]
    kw=dict(hidden_dims=dims,activation='elu',distribution_cfg=dict(class_name='GaussianDistribution',init_std=.35,std_type='log'))
    actor=Actor(obs,groups,'actor',16,**copy.deepcopy(kw))
    standard=MLPModel(obs,groups,'actor',16,**copy.deepcopy(kw))
    critic=MLPModel(obs,groups,'critic',1,hidden_dims=dims,activation='elu')
    return actor,standard,critic


def check_actor_and_ppo():
    obs=TensorDict(dict(policy=torch.randn(8,244)*.1,critic=torch.randn(8,285)*.1,
        stop_metadata=torch.tensor([[1.],[0.]]*4)),batch_size=[8])
    actor,standard,critic=make_models(obs)
    standard.load_state_dict(actor.state_dict(),strict=True)
    assert torch.equal(actor(obs),standard(obs))
    for iteration,factor in ((0,.5),(250,.3),(500,.1),(1000,.1)):
        actor.set_update_iteration(iteration)
        actions=actor(obs,stochastic_output=True)
        old=actor.get_output_log_prob(actions).detach()
        actual_std=actor.output_distribution_params[1]
        assert torch.allclose(actual_std[::2],actual_std[1::2]*factor)
        actor(obs,stochastic_output=True)
        assert torch.equal(old,actor.get_output_log_prob(actions))
        assert torch.equal(actor(obs),standard(obs))
    storage=RolloutStorage('rl',8,4,obs,[16])
    # Match rollout sampling: distributions cached before a reference copy
    # must not retain an autograd graph from these standalone probes.
    with torch.no_grad(): actor(obs,stochastic_output=True)
    alg=StairGuardedPPO(actor,critic,storage,critic_warmup_iterations=1,
        anchor_stair_weight=1.,anchor_stop_weight=.02,
        num_learning_epochs=1,num_mini_batches=1,learning_rate=1.e-5,schedule='fixed')
    alg.set_reference(copy.deepcopy(standard.state_dict()))
    with torch.no_grad():
        actor.mlp[-1].bias.add_(.1)
        # Identical effective commands can represent a raw stop with yaw
        # correction or a manual turn; only the former gets the stop weight.
        paired=obs[:2].clone()
        paired['policy'][1]=paired['policy'][0]
        losses=[float(alg._anchor_loss(paired[i:i+1])) for i in range(2)]
        assert abs(losses[1]/losses[0]-50.)<1.e-3
        actor.load_state_dict(standard.state_dict())
    initial=copy.deepcopy(actor.state_dict())
    for update in range(3):
        for _ in range(4):
            with torch.inference_mode():
                alg.act(obs)
                alg.process_env_step(obs,torch.ones(8),torch.zeros(8),{})
        assert 'stop_metadata' in storage.observations
        alg.compute_returns(obs); result=alg.update()
        assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
        assert 'stop_sample_noise_rms_rad' in result
        if update==0:
            assert all(torch.equal(initial[k],v) for k,v in actor.state_dict().items())
    a2,_,c2=make_models(obs)
    restored=StairGuardedPPO(a2,c2,RolloutStorage('rl',8,4,obs,[16]))
    restored.load(alg.save(),None,True)
    assert restored.guard_updates==3
    with torch.inference_mode(): restored.act(obs)
    assert abs(a2.stop_std_factor-(.5-.4*3/500))<1.e-8
    # Actual 5100 actor remains strictly loadable with exactly equal deterministic means.
    large,old,_=make_models(obs,large=True)
    checkpoint=ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-15_11-59-41_refine_stop_heading_posture_v1/model_5100.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)['actor_state_dict']
    large.load_state_dict(saved,strict=True);old.load_state_dict(saved,strict=True)
    assert torch.equal(large(obs),old(obs))
    return dict(sampling_and_ppo_log_probability_consistent=True,movement_std_unchanged=True,
        raw_stop_only_releases_preservation=True,
        deterministic_actor_and_state_keys_compatible_with_model5100=True,
        metadata_preserved_in_storage=True,actual_cpu_ppo_updates=3,checkpoint_schedule_restored=True,
        final_losses=result)


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    result=dict(rewards=check_rewards(),pose=check_pose(),curriculum=check_curriculum(),
                actor_ppo=check_actor_and_ppo(),controller=check_controller_parity())
    out=ROOT/'docs/v1_stop_v2_20261017/checks.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__': main()
