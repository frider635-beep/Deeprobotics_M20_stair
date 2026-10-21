#!/usr/bin/env python3
"""Validate stage5 initialization, dimensions and demonstrations without Isaac."""
import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--profile', choices=('stage5', 'stage6'), default='stage5')
    parser.add_argument('--demonstration', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.profile == 'stage5' and not args.demonstration:
        parser.error('Stage5 preflight requires --demonstration')
    if args.profile == 'stage6' and args.demonstration:
        parser.error('Stage6 does not reuse demonstrations without swing-alternation verification')
    torch.set_num_threads(1)
    module = deployment.ascent_adaptive_module()
    init = deployment.training_module('algorithms.initialization.ascent_alternating_init' if args.profile == 'stage6' else 'ascent_adaptive_init')
    demo = deployment.training_module('ascent_adaptive_demo')
    provenance = init.validate_source(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    obs = TensorDict(dict(policy=torch.zeros(2, 244), motion_control=torch.tensor([[.5, 0., 0.], [.7, 0., 0.]]),
                          critic=torch.zeros(2, 285), ascent_control=torch.zeros(2, 12)), batch_size=[2])
    from rsl_rl.models import MLPModel
    with redirect_stdout(StringIO()):
        actor = module.AscentAdaptiveMLPModel(obs, {'actor': ['policy', 'motion_control']}, 'actor', 16,
            hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False,
            distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'log'})
        init.load_initialized_actor(actor, saved)
        source = deployment.load_actor(args.checkpoint)
        critic = MLPModel(obs, {'critic': ['critic', 'motion_control', 'ascent_control']}, 'critic', 1,
            hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False)
        init.load_initialized_critic(critic, saved['critic_state_dict'])
    with torch.no_grad():
        assert torch.equal(actor(obs), source(obs))
        assert torch.isfinite(critic(obs)).all()
    holder = SimpleNamespace(actor=actor, device='cpu', update=lambda: {}, save=lambda: {}, load=lambda *args: None)
    demo.attach_demonstrations(holder, args.demonstration, coefficient=0. if args.profile == 'stage6' else .5)
    authority = demo.validate_demo_authority(holder)
    result = dict(profile='ascent_entry_v1_'+args.profile, source=provenance, actor_dimensions=[247, 16],
        critic_dimensions=[300, 1], trainable_branch='sequence_mlp',
        frozen_source_exact=True, demonstrations=holder._contact_demo_manifest,
        demonstration_authority=authority, isaac_started=False, training_started=False,
        scope='CPU preflight only; not Isaac/PhysX training startup or learned gait acceptance.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
