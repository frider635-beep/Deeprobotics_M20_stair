#!/usr/bin/env python3
"""CPU counterexamples for v1 actor preservation, gating and physical slip."""
import argparse
import json
from pathlib import Path
import sys
import torch
from tensordict import TensorDict

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'.scratch/ascent_entry_v1/cpu_checks.json')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    old=d.load_actor(ROOT/'pre_train/teacher_baseline_.pt')
    module=d.ascent_entry_module()
    observations=TensorDict({'policy':torch.randn(64,244),'motion_control':torch.zeros(64,3)},batch_size=[64])
    observations['motion_control'][:,0]=.7
    actor=module.AscentEntryMLPModel(observations,{'actor':['policy','motion_control']},'actor',16,
        hidden_dims=[512,256,128],activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':1.,'std_type':'log'})
    saved=torch.load(ROOT/'pre_train/teacher_baseline_.pt',map_location='cpu',weights_only=True)
    actor.load_motion_checkpoint(saved['actor_state_dict'])
    with torch.no_grad():
        assert torch.equal(actor(observations),old(observations)), 'Initialization changed the full baseline'
    trainable=[k for k,p in actor.named_parameters() if p.requires_grad]
    assert trainable and all(k.startswith('entry_mlp.') for k in trainable)
    # Give the branch a nonzero output, then exercise the zero, flat and
    # descending gates. These must remain exact regardless of learned weights.
    with torch.no_grad(): actor.entry_mlp[-1].bias.fill_(.5)
    for name in ('raw_zero','flat','descent','reverse','turn'):
        obs=observations.clone()
        grid=torch.linspace(.8,-.8,17)[None,None].expand(64,11,17)
        obs['policy'][:,57:]=grid.reshape(64,187)
        if name=='raw_zero': obs['motion_control'].zero_()
        if name=='flat': obs['policy'][:,57:]=.1
        if name=='descent': obs['policy'][:,57:]=-grid.reshape(64,187)
        if name=='reverse': obs['motion_control'][:,0]=-.7
        if name=='turn': obs['motion_control'][:]=torch.tensor([0.,0.,.6])
        with torch.no_grad(): assert torch.equal(actor(obs),old(obs)),name
    obs=observations.clone()
    obs['policy'][:,57:]=torch.linspace(.8,-.8,17)[None,None].expand(64,11,17).reshape(64,187)
    new=actor(obs)
    with torch.no_grad(): target=old(obs)
    assert (new-target).abs().max()>0.
    assert torch.all((new-target).abs()<=actor.entry_limits+1.e-6)
    loss=(new-target-.01).square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in actor.entry_mlp.parameters())
    assert all(p.grad is None for k,p in actor.named_parameters() if not k.startswith('entry_mlp.'))
    mdp=d.training_module('ascent_entry_mdp')
    com=torch.tensor([[[0.,0.,.1]]]);v=torch.tensor([[[.4,0.,0.]]]);omega=torch.tensor([[[0.,4.,0.]]])
    point=torch.zeros_like(com);force=torch.tensor([[[0.,0.,100.]]])
    slip,support,face,_=mdp.contact_patch_velocity(com,v,omega,point,force)
    assert slip.max()<1.e-6 and support.all() and not face.any()
    slip,*_=mdp.contact_patch_velocity(com,v,omega*0,point,force)
    assert torch.allclose(slip,torch.tensor([[.4]]))
    slip,support,face,_=mdp.contact_patch_velocity(com,v,omega,point*float('nan'),force)
    assert slip.max()==0 and not support.any() and not face.any()
    _,support,face,_=mdp.contact_patch_velocity(com,v,omega,point,torch.tensor([[[-100.,0.,0.]]]))
    assert face.all() and not support.any()
    assert mdp.period_error(torch.tensor([.43]),torch.tensor([.43])).item()==0.
    assert mdp.period_error(torch.tensor([.86]),torch.tensor([.43])).item()>0.,'A missing whole cycle was hidden by phase wrapping'
    # Boundary weighting is continuous at tread three and four.
    x=torch.tensor([2.999,3.,3.001,3.999,4.,4.001])
    weight=1.-mdp.smooth01((x-2.)/3.)
    assert (weight[1:]-weight[:-1]).abs().max()<.5
    result={'full_motion_checkpoint_initialization_exact':True,'frozen_parameter_names':[k for k,p in actor.named_parameters() if not p.requires_grad],
            'trainable_parameter_names':trainable,'zero_flat_descent_reverse_turn_exact':True,
            'bounded_correction_and_branch_only_gradients':True,'rolling_translation_missing_patch_and_face_counterexamples':True,
            'whole_cycle_aliasing_counterexample':True,
            'initialization_sha256':__import__('hashlib').sha256((ROOT/'pre_train/teacher_baseline_.pt').read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
