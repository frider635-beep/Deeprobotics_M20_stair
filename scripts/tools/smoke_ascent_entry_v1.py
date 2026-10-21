#!/usr/bin/env python3
"""Bounded real Isaac v1 registration, inference and contact-objective smoke."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/rl_training'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--steps',type=int,default=250)
parser.add_argument('--num_envs',type=int,default=32)
parser.add_argument('--profile', choices=('ascent_entry_v1', 'ascent_entry_v1_stage2', 'ascent_entry_v1_stage3', 'ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'), default=None,
                    help='Defaults to checkpoint metadata; override to check stage2 with a stage1 actor.')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import gymnasium as gym
import torch
import rl_training.tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rl_training.models.ascent_entry_actor import AscentEntryMLPModel

task='Rough-Deeprobotics-M20-StairTeacher-v1'
saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
profile=args.profile or (saved.get('m20_training_manifest') or {}).get('m20_profile', 'ascent_entry_v1')
cfg=parse_env_cfg(task,device=args.device or 'cuda:0',num_envs=args.num_envs)
if profile in ('ascent_entry_v1_stage2', 'ascent_entry_v1_stage3', 'ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
    if profile == 'ascent_entry_v1_stage5':
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.ascent_adaptive_profile import apply_profile
    elif profile == 'ascent_entry_v1_stage4':
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.ascent_sequence_profile import apply_profile
    elif profile == 'ascent_entry_v1_stage3':
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.ascent_landing_profile import apply_profile
    else:
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.ascent_entry_stage2_profile import apply_profile
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.stair_teacher_v1_ppo_cfg import DeeproboticsM20StairTeacherV1PPORunnerCfg
    apply_profile(cfg, DeeproboticsM20StairTeacherV1PPORunnerCfg())
    if profile == 'ascent_entry_v1_stage4': cfg.ascent_sequence_training_ramp = False
cfg.seed=42
env=gym.make(task,cfg=cfg)
assert type(env.unwrapped).__name__=='AscentEntryRLEnv'
wrapper=RslRlVecEnvWrapper(env,clip_actions=100)
obs=wrapper.get_observations()
actor_type = AscentEntryMLPModel
if profile == 'ascent_entry_v1_stage5':
    from rl_training.models.ascent_adaptive_actor import AscentAdaptiveMLPModel
    actor_type = AscentAdaptiveMLPModel
elif profile == 'ascent_entry_v1_stage4':
    from rl_training.models.ascent_sequence_actor import AscentSequenceMLPModel
    actor_type = AscentSequenceMLPModel
elif profile == 'ascent_entry_v1_stage3':
    from rl_training.models.ascent_landing_actor import AscentLandingMLPModel
    actor_type = AscentLandingMLPModel
actor=actor_type(obs,{'actor':['policy','motion_control']},'actor',16,
    hidden_dims=[512,256,128],activation='elu',obs_normalization=False,
    distribution_cfg={'class_name':'GaussianDistribution','init_std':1.,'std_type':'log'}).to(cfg.sim.device)
if profile == 'ascent_entry_v1_stage5' and (saved.get('m20_training_manifest') or {}).get('m20_profile') != profile:
    from rl_training.algorithms.initialization.ascent_adaptive_init import validate_source, load_initialized_actor
    validate_source(args.checkpoint)
    load_initialized_actor(actor, saved)
elif profile == 'ascent_entry_v1_stage4' and (saved.get('m20_training_manifest') or {}).get('m20_profile') == 'ascent_entry_v1_stage3':
    actor.load_stage3_checkpoint(saved['actor_state_dict'])
elif profile == 'ascent_entry_v1_stage3' and (saved.get('m20_training_manifest') or {}).get('m20_profile') == 'ascent_entry_v1_stage2':
    actor.load_stage2_checkpoint(saved['actor_state_dict'])
else:
    actor.load_state_dict(saved['actor_state_dict'],strict=True)
actor.eval()
base=env.unwrapped
terms=base.reward_manager.active_terms
peaks=torch.zeros(len(terms),device=base.device)
for step in range(args.steps):
    with torch.inference_mode():
        action=actor(obs)
        obs,reward,done,extra=wrapper.step(action)
    assert torch.isfinite(action).all() and torch.isfinite(reward).all()
    assert all(torch.isfinite(v).all() for v in obs.values())
    peaks=torch.maximum(peaks,base.reward_manager._step_reward.abs().amax(0))
observer=base._ascent_entry_observer
result={'task':task,'profile':profile,'registered_environment_type':type(base).__name__,
        'checkpoint':str(args.checkpoint),'steps':args.steps,'num_envs':args.num_envs,
        'observation_shapes':{k:list(v.shape) for k,v in obs.items()},
        'filtered_support_samples':int(observer.lifetime_support_samples.item()),
        'new_tread_landings':int(observer.lifetime_landings.item()),
        'reward_abs_peaks':{name:float(peaks[i]) for i,name in enumerate(terms) if name.startswith('ascent_')},
        'finite':True,'trained':False}
assert result['filtered_support_samples']>0 and result['new_tread_landings']>0
required = (('ascent_stage2_ordered_tread', 'ascent_stage2_missing_tread',
             'ascent_stage2_front_slip', 'ascent_stage2_rear_slip') if profile in ('ascent_entry_v1_stage2', 'ascent_entry_v1_stage3')
            else ('ascent_entry_phase', 'ascent_wheel_slip', 'ascent_entry_landing'))
if profile == 'ascent_entry_v1_stage4':
    required = ('ascent_sequence_ordered', 'ascent_sequence_missing', 'ascent_sequence_height',
                'ascent_stage2_front_slip', 'ascent_stage2_rear_slip')
if profile == 'ascent_entry_v1_stage5':
    required = ('ascent_adaptive_height', 'ascent_adaptive_failure', 'ascent_stage2_front_slip')
for name in required:
    assert result['reward_abs_peaks'][name]>0,name
if profile in ('ascent_entry_v1_stage2', 'ascent_entry_v1_stage3'):
    result['ordered_tread_events'] = int(observer.lifetime_ordered.item())
    result['missing_tread_events'] = int(observer.lifetime_missing.item())
    assert result['ordered_tread_events'] > 0 and result['missing_tread_events'] > 0
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print('V1_SMOKE_RESULT='+json.dumps(result),flush=True)
wrapper.close()
if args.headless:
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
app.close()
