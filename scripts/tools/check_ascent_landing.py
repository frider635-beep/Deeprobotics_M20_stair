#!/usr/bin/env python3
"""CPU checks for the entrance-only landing branch and dense pre-contact cost."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deployment
from check_ascent_entry_stage2 import local_module


def make_actor(module, obs):
    return module.AscentLandingMLPModel(obs, {'actor': ['policy', 'motion_control']}, 'actor', 16,
        hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False,
        distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'log'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--source-evaluation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zero-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    module = deployment.ascent_landing_module()
    init = local_module('ascent_landing_init')
    mdp = local_module('ascent_landing_mdp')
    provenance = init.validate_source(args.source_checkpoint)
    saved = torch.load(args.source_checkpoint, map_location='cpu', weights_only=True)
    old = deployment.load_actor(args.source_checkpoint)
    obs = TensorDict({'policy': torch.randn(64, 244), 'motion_control': torch.zeros(64, 3)}, batch_size=[64])
    obs['motion_control'][:, 0] = .7
    entry_profile = torch.tensor([.6]*11+[.45]*3+[.3]*3)
    continuous_profile = torch.tensor([.8, .8, .65, .65, .65, .5, .5, .5, .35, .35, .35, .2, .2, .2, .05, .05, .05])
    def scan(values): return values[None, None].expand(64, 11, 17).reshape(64, 187)
    obs['policy'][:, 57:] = scan(entry_profile)
    actor = make_actor(module, obs)
    actor.load_stage2_checkpoint(saved['actor_state_dict'])
    with torch.no_grad(): assert torch.equal(actor(obs), old(obs))
    assert module.entrance_gate(obs['policy'], obs['motion_control']).min() > 0.
    trainable = [name for name, parameter in actor.named_parameters() if parameter.requires_grad]
    assert trainable and all(name.startswith('landing_mlp.') for name in trainable)
    zero = copy.deepcopy(actor.state_dict())
    # Build a clearly marked CPU-only inference fixture, never a promoted model.
    fixture_manifest = copy.deepcopy(saved['m20_training_manifest'])
    fixture_manifest.update(m20_profile='ascent_entry_v1_stage3',
        refinement_version='m20_ascent_landing_20261020_v1_stage3',
        frozen_stage2_source_sha256=provenance['sha256'], landing_limits_action_units=module.LANDING_LIMITS,
        frozen_stage2_state_sha256=actor.source_state_digest(),
        test_fixture_only=True)
    fixture_manifest['actor_model_config'].update(class_name='rl_training.models.ascent_landing_actor:AscentLandingMLPModel',
                                                landing_hidden_dims=[128, 64])
    args.zero_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'actor_state_dict': zero, 'critic_state_dict': saved['critic_state_dict'],
                'm20_training_manifest': fixture_manifest}, args.zero_checkpoint)
    native = deployment.load_actor(args.zero_checkpoint)
    with torch.no_grad(): assert torch.equal(native(obs), old(obs))
    with torch.no_grad(): actor.landing_mlp[-1].bias.fill_(.4)
    for kind in ['zero', 'flat', 'descent', 'reverse', 'turn', 'continuous']:
        sample = obs.clone()
        if kind == 'zero': sample['motion_control'].zero_()
        if kind == 'flat': sample['policy'][:, 57:] = .2
        if kind == 'descent': sample['policy'][:, 57:] = scan(-entry_profile)
        if kind == 'reverse': sample['motion_control'][:, 0] = -.7
        if kind == 'turn': sample['motion_control'][:, 1] = .3
        if kind == 'continuous': sample['policy'][:, 57:] = scan(continuous_profile)
        with torch.no_grad(): assert torch.equal(actor(sample), old(sample)), kind
    with torch.no_grad(): inherited = old(obs)
    prediction = actor(obs)
    assert (prediction-inherited).abs().max() > 0.
    assert torch.all((prediction-inherited).abs() <= actor.landing_limits+1.e-6)
    frozen = {name: value.detach().clone() for name, value in actor.named_parameters() if not value.requires_grad}
    before = actor.landing_mlp[-1].weight.detach().clone()
    optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=2.e-5)
    optimizer.zero_grad()
    (prediction-inherited-.01).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, actor.landing_mlp[-1].weight)
    assert all(torch.equal(value, dict(actor.named_parameters())[name]) for name, value in frozen.items())
    assert actor.source_state_digest() == fixture_manifest['frozen_stage2_state_sha256']

    # In the landing patch, correct support has no dense penalty; hovering
    # higher costs more, with an analytic gradient pointing down toward tread.
    positions = torch.tensor([[[.14, .2, .24], [.44, -.2, .39]]])
    targets = torch.tensor([[1, 2]])
    unresolved = torch.ones(1, 2, dtype=torch.bool)
    enabled = torch.ones(1)
    def guide(points, velocities=None, valid=unresolved, active=enabled, target=targets):
        return mdp.landing_guidance(points, torch.zeros(1, 2) if velocities is None else velocities,
            target, valid, torch.zeros(1), torch.zeros(1), active)
    good = guide(positions)
    assert good['height'].max() < 1.e-10 and good['velocity'].max() < 1.e-10
    high = positions.clone(); high[:, :, 2] += .12; high.requires_grad_()
    high_cost = guide(high)
    grad = torch.autograd.grad(high_cost['height'].sum(), high)[0]
    assert torch.all(grad[:, :, 2] > 0.)
    closer = positions.clone(); closer[:, :, 2] += .04
    assert guide(closer)['height'].item() < high_cost['height'].item()
    assert guide(high.detach(), high_cost['desired_vz'])['velocity'].max() < 1.e-10
    before_edge = positions.clone(); before_edge[:, :, 0] -= .19; before_edge[:, :, 2] += .04
    assert guide(before_edge)['height'].max() < 1.e-10
    below = before_edge.clone(); below[:, :, 2] -= .06; below.requires_grad_()
    grad = torch.autograd.grad(guide(below)['height'].sum(), below)[0]
    assert torch.all(grad[:, :, 2] < 0.), 'Guide pulled the wheel down into the riser'
    for info in [guide(high.detach(), active=enabled*0), guide(high.detach(), valid=~unresolved),
                 guide(high.detach(), target=targets+2)]:
        assert all(info[key].max() == 0 for key in ['height', 'horizontal', 'velocity'])
    assert mdp.metrics(SimpleNamespace(), None) == {}, 'Initial curriculum logging accessed an uninitialized observer'

    report = json.loads((args.source_evaluation_dir/'evaluation.json').read_text())
    assert report['candidate_sha256'] == provenance['sha256']
    gates = []
    for pair in report['pairs']:
        folder = args.source_evaluation_dir/pair['case']
        with np.load(folder/'policy.npz') as z: policy = {key: z[key] for key in z.files}
        with np.load(folder/'physics.npz') as z: physical = {key: z[key] for key in z.files}
        with torch.no_grad(): gate = module.entrance_gate(torch.from_numpy(policy['observation']),
                                                        torch.from_numpy(policy['requested_command'])).numpy()
        gate_physics = gate[physical['policy_step'].astype(int)]
        progress = np.zeros((len(physical['time']), 2), dtype=int)
        for event in pair['candidate']['landing_events']:
            if event['wheel'] in ('FL', 'FR'):
                j = 0 if event['wheel'] == 'FL' else 1
                progress[event['sample_index']:, j] = event['tread_id']
        middle = (progress.max(-1) >= 5) & (progress.max(-1) <= 18)
        row = {'case': pair['case'], 'middle_max_gate': float(gate_physics[middle].max())}
        for wheel, target in [(0, 2), (1, 1)]:
            edge = .25+(target-1)*.30
            inside = (physical['wheel_pos'][:, wheel, 0] >= edge+.07)
            inside &= (physical['wheel_pos'][:, wheel, 0] <= edge+.25) & (physical['time'] >= 6.)
            row['FL2_min_gate' if wheel == 0 else 'FR1_min_gate'] = float(gate_physics[inside].min())
            assert gate_physics[inside].min() > 0., 'Entrance gate missed the intended landing opportunity'
        assert row['middle_max_gate'] < 1.e-5, 'Entrance authority leaked into continuous climbing'
        gates.append(row)
    # Exercise the real PPO/anchor/warmup code on CPU with synthetic rewards.
    # This verifies optimizer integration, not the Isaac physics environment
    # or gait quality. One test warmup update stands in for the configured40.
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    guarded = local_module('stair_guarded_ppo')
    batch = obs[:32].clone()
    batch['critic'] = torch.randn(32, 285)
    learner = make_actor(module, batch)
    learner.load_stage2_checkpoint(saved['actor_state_dict'])
    critic = MLPModel(batch, {'critic': ['critic', 'motion_control']}, 'critic', 1,
                     hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False)
    critic.load_state_dict(saved['critic_state_dict'], strict=True)
    storage = RolloutStorage('rl', 32, 4, batch, [16], device='cpu')
    algorithm = guarded.StairGuardedPPO(learner, critic, storage, device='cpu', num_learning_epochs=1,
        num_mini_batches=2, schedule='fixed', learning_rate=2.e-5, critic_warmup_iterations=1,
        anchor_coefficient=.1, anchor_stair_weight=.03, motion_regularization_coefficient=.02)
    algorithm.set_reference(learner.state_dict())
    ppo_checks = []
    for iteration in range(3):
        before = learner.landing_mlp[-1].weight.detach().clone()
        for _ in range(4):
            with torch.inference_mode():
                actions = algorithm.act(batch)
                rewards = torch.randn(32)-.02*actions.square().mean(-1)
                algorithm.process_env_step(batch, rewards, torch.zeros(32, dtype=torch.long), {})
        with torch.inference_mode(): algorithm.compute_returns(batch)
        losses = algorithm.update()
        assert all(math.isfinite(float(value)) for value in losses.values())
        changed = not torch.equal(before, learner.landing_mlp[-1].weight)
        assert changed == (iteration > 0), 'PPO warmup or landing update failed'
        assert learner.source_state_digest() == fixture_manifest['frozen_stage2_state_sha256']
        ppo_checks.append({'test_iteration': iteration, 'landing_updated': changed, 'finite_losses': True,
                           'complete_source_unchanged': True, 'warmup': bool(losses['critic_warmup'])})
    result = {'source': provenance, 'complete_stage2_zero_initialization_exact': True,
              'native_loader_exact': True, 'only_landing_mlp_trainable': trainable,
              'entire_source_actor_frozen_after_update': True,
              'zero_flat_descent_reverse_turn_and_continuous_exact': True,
              'bounded_entrance_authority': True, 'pre_contact_gradient_and_riser_clearance_checks': True,
              'inactive_resolved_and_later_treads_do_not_pay': True,
              'initial_curriculum_metrics_safe': True, 'archived_gate_checks': gates,
              'cpu_ppo_synthetic_integration': ppo_checks,
              'zero_checkpoint_fixture': str(args.zero_checkpoint), 'gpu_test_run': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
