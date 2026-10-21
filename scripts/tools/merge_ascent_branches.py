#!/usr/bin/env python3
"""Consolidate three old ascent branches into one MLP using archived states.

This is action-function migration, not certification of an expert gait. Case
holdout and subsequent closed-loop evaluations are required; no PPO is started.
"""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deployment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--data',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=2400)
    parser.add_argument('--warm-start',type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new path')
    torch.set_num_threads(2);torch.manual_seed(42)
    saved=torch.load(args.source,map_location='cpu',weights_only=True)
    deployment.training_module('algorithms.initialization.ascent_alternating_init').validate_source(args.source)
    teacher=deployment.load_actor(args.source).eval().requires_grad_(False)
    unified=deployment.training_module('models.ascent_unified_actor')
    sample=TensorDict(dict(policy=torch.zeros(1,244),motion_control=torch.zeros(1,3)),batch_size=[1])
    student=unified.AscentUnifiedMLPModel(sample,{'actor':['policy','motion_control']},'actor',16,
        hidden_dims=[512,256,128],activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':1.,'std_type':'log'})
    student.load_motion_checkpoint(saved['actor_state_dict'])
    if args.warm_start:
        warm=torch.load(args.warm_start,map_location='cpu',weights_only=True)
        student.load_state_dict(warm['actor_state_dict'],strict=True)
    policies=[];commands=[];targets=[];heldout=[];cases=[]
    for folder in args.data:
        for path in sorted(folder.rglob('policy.npz')):
            cfg=json.loads((path.parent/'case.json').read_text())
            with np.load(path) as z:
                p=torch.from_numpy(z['observation'].astype(np.float32))
                c=torch.from_numpy(z['requested_command'].astype(np.float32))
            obs=TensorDict(dict(policy=p,motion_control=c),batch_size=[len(p)])
            with torch.no_grad():target=teacher(obs)
            policies.append(p);commands.append(c);targets.append(target)
            # Every zero-offset rollout is held out as a complete case.
            heldout.append(torch.full((len(p),),abs(cfg['spawn_x_offset_m'])<1.e-8))
            cases.append(dict(file=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              samples=len(p),heldout=abs(cfg['spawn_x_offset_m'])<1.e-8))
    if not policies:parser.error('No policy.npz data')
    p=torch.cat(policies);c=torch.cat(commands);target=torch.cat(targets);hold=torch.cat(heldout)
    obs=TensorDict(dict(policy=p,motion_control=c),batch_size=[len(p)])
    gate=unified.unified_gate(p,c)
    train=(~hold & (gate>.02)).nonzero().flatten();valid=(hold & (gate>.02)).nonzero().flatten()
    if len(train)<100 or len(valid)<100:parser.error('Insufficient active training/heldout states')
    # Normalize using action units corresponding to 0.25 rad leg / 5 rad/s wheel
    # control; fit the final composed action without dividing by a tiny gate.
    scale=torch.tensor([.125,.25,.25]*4+[.45]*4)
    optimizer=torch.optim.Adam(student.unified_mlp.parameters(),lr=2.e-4 if args.warm_start else 1.e-3)
    best=float('inf');best_state=None;history=[]
    for step in range(args.updates):
        if step==args.updates*2//3:
            for group in optimizer.param_groups:group['lr']=2.e-4
        ids=train[torch.randint(len(train),(min(1024,len(train)),))]
        predicted=student(obs[ids]);error=(predicted-target[ids])*scale
        loss=error.square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%100==0 or step==args.updates-1:
            with torch.no_grad():
                errors=torch.cat([(student(obs[chunk])-target[chunk])*scale for chunk in valid.split(2048)])
                score=float(errors.square().mean())
            history.append(dict(update=step+1,train_mse=float(loss),heldout_mse=score))
            if score<best:best=score;best_state=copy.deepcopy(student.unified_mlp.state_dict())
            print(json.dumps(history[-1]),flush=True)
    student.unified_mlp.load_state_dict(best_state)
    with torch.no_grad():
        err=torch.cat([student(obs[chunk])-target[chunk] for chunk in valid.split(2048)])
        physical=err*torch.tensor([.125,.25,.25]*4+[5.]*4)
        metrics=dict(leg_rms_rad=float(physical[:,:12].square().mean().sqrt()),
                     wheel_rms_rad_s=float(physical[:,12:].square().mean().sqrt()),
                     leg_abs_p99_rad=float(torch.quantile(physical[:,:12].abs().flatten(),.99)),
                     action_rms=float(err.square().mean().sqrt()))
    base_original={k:v for k,v in saved['actor_state_dict'].items() if k.startswith(('mlp.','residual_mlp.','distribution.'))}
    assert all(torch.equal(v,student.state_dict()[k]) for k,v in base_original.items())
    if metrics['leg_rms_rad']>.035 or metrics['wheel_rms_rad_s']>.20:
        raise RuntimeError('Migration fit did not meet thresholds: '+json.dumps(metrics))
    consolidation=dict(warm_start=str(args.warm_start.resolve()) if args.warm_start else None,source=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        teacher_profile=saved['m20_training_manifest']['m20_profile'],purpose='action_function_migration_not_expert_imitation',
        cases=cases,train_active_samples=len(train),heldout_active_samples=len(valid),metrics=metrics,history=history)
    manifest=dict(m20_profile=unified.PROFILE,refinement_version=unified.VERSION,control_profile='legacy_v1',
        actor_model_config=dict(unified_hidden_dims=[256,128]),unified_limits_action_units=unified.UNIFIED_LIMITS,
        adaptive_exploration_action_std=unified.EXPLORATION,frozen_full_baseline_sha256=saved['m20_training_manifest']['frozen_full_baseline_sha256'],
        frozen_motion_state_sha256=unified.baseline_digest(student.state_dict()),consolidation=consolidation,
        actor_input_groups={'policy':244,'motion_control':3},critic_input_groups={'critic':285,'motion_control':3,'ascent_control':12})
    result=dict(actor_state_dict=student.state_dict(),critic_state_dict=saved['critic_state_dict'],
                m20_training_manifest=manifest,iter=0)
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,args.output)
    args.output.with_suffix('.json').write_text(json.dumps(consolidation,indent=2)+'\n')
    deployment.training_module('algorithms.initialization.ascent_unified_init').validate_source(args.output)
    loaded=deployment.load_actor(args.output)
    with torch.no_grad():assert torch.equal(loaded(obs[:16]),student(obs[:16]))
    print(json.dumps(dict(output=str(args.output),metrics=metrics,baseline_exact=True),indent=2))


if __name__=='__main__':main()
