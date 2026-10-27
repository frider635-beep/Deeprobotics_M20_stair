#!/usr/bin/env python3
"""Synthetic optimizer diagnostic integration; no policy files, runner or sim."""
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ppo_diag_loader',ROOT/'source/rl_training/rl_training/utils/module_loader.py')
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
module=loader.load_module('algorithms.stair_guarded_ppo')


class Observations(dict):
    @property
    def batch_size(self):return (len(self['policy']),)


class Actor(torch.nn.Module):
    is_recurrent=False
    def __init__(self):
        super().__init__();self.net=torch.nn.Linear(244,16)
        self.distribution=NS(log_std_param=torch.zeros(16))
    def forward(self,obs,**kwargs):
        mean=self.net(obs['policy'])
        self.dist=torch.distributions.Normal(mean,torch.ones_like(mean))
        self.output_distribution_params=(mean,torch.ones_like(mean))
        self.output_entropy=self.dist.entropy().sum(-1)
        return mean
    def get_output_log_prob(self,actions):return self.dist.log_prob(actions).sum(-1)


class Critic(torch.nn.Module):
    is_recurrent=False
    def __init__(self):super().__init__();self.net=torch.nn.Linear(244,1)
    def forward(self,obs,**kwargs):return self.net(obs['policy'])


def make(sample):
    torch.manual_seed(42)
    actor=Actor();critic=Critic();reference=copy.deepcopy(actor).requires_grad_(False)
    with torch.no_grad():actor.net.bias.add_(.03)
    observations=Observations(policy=torch.randn(32,244))
    with torch.no_grad():
        actions=actor(observations)+torch.randn(32,16)
        old_log=actor.get_output_log_prob(actions)
        values=critic(observations)
    batch=NS(observations=observations,actions=actions,old_actions_log_prob=old_log,
             values=values,returns=values+torch.randn(32,1),advantages=torch.randn(32),
             masks=None,hidden_states=(None,None))
    # Exercise the real vendored update using tiny synthetic actors/storage.
    alg=object.__new__(module.StairGuardedPPO)
    alg.actor=actor;alg.critic=critic;alg.reference_actor=reference
    alg.storage=NS(mini_batch_generator=lambda *_:iter([batch,batch]),clear=lambda:None)
    alg.optimizer=torch.optim.Adam([*actor.parameters(),*critic.parameters()],lr=1.e-5)
    for key,value in dict(guard_updates=100 if sample else 101,num_mini_batches=2,num_learning_epochs=1,
        normalize_advantage_per_mini_batch=False,symmetry=None,rnd=None,rnd_optimizer=None,
        is_multi_gpu=False,desired_kl=None,schedule='fixed',clip_param=.1,use_clipped_value_loss=True,
        value_loss_coef=1.,entropy_coef=.0005,critic_warmup_iterations=30,
        anchor_coefficient=.15,anchor_stair_weight=1.,anchor_stop_weight=.02,
        anchor_stop_command_epsilon=1.e-4,stop_teacher_coefficient=0.,
        stop_regularization_coefficient=0.,motion_regularization_coefficient=0.,max_grad_norm=1.).items():
        setattr(alg,key,value)
    alg._anchor_losses=[]
    return alg


def main():
    torch.set_num_threads(1)
    observed=make(True);control=make(False)
    observed_metrics=observed._guarded_update();control_metrics=control._guarded_update()
    for a,b in zip(observed.actor.parameters(),control.actor.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(observed.critic.parameters(),control.critic.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    expected=('approx_kl','clip_fraction','ppo_grad_norm','weighted_anchor_grad_norm',
              'ppo_anchor_grad_cosine','actor_parameter_update_l2','actor_parameter_relative_update')
    assert all(name in observed_metrics and math.isfinite(observed_metrics[name]) for name in expected)
    assert observed_metrics['actor_parameter_update_l2']>0
    assert observed_metrics['weighted_anchor_grad_norm']>0
    assert observed_metrics['gradient_cosine_valid']==1
    assert 'ppo_grad_norm' not in control_metrics
    print(json.dumps(dict(diagnostic_sampling_preserves_optimizer_result=True,
        synthetic_actor=True,policy_checkpoints_loaded=0,diagnostics=observed_metrics),indent=2))


if __name__=='__main__':main()
