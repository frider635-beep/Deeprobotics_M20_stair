#!/usr/bin/env python3
"""CPU counterexamples and real-PPO integration for stage4; never launches Isaac."""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deployment
from check_ascent_entry_stage2 import local_module


def make_actor(module, obs):
    return module.AscentSequenceMLPModel(obs, {'actor': ['policy', 'motion_control']}, 'actor', 16,
        hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False,
        distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'log'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zero-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(42)
    module = deployment.ascent_sequence_module()
    core = deployment.training_module('ascent_sequence')
    mdp = deployment.training_module('ascent_sequence_mdp')
    init = deployment.training_module('ascent_sequence_init')
    provenance = init.validate_source(args.source_checkpoint)
    saved = torch.load(args.source_checkpoint, map_location='cpu', weights_only=True)
    with tempfile.TemporaryDirectory(prefix='m20_sequence_provenance_') as folder:
        for kind in ['baseline_changed', 'frozen_entry_changed', 'landing_nonfinite', 'incomplete']:
            bad = copy.deepcopy(saved)
            state = bad['actor_state_dict']
            if kind == 'baseline_changed': state['mlp.0.bias'][0] += .01
            if kind == 'frozen_entry_changed': state['entry_mlp.4.bias'][0] += .01
            if kind == 'landing_nonfinite': state['landing_mlp.4.bias'][0] = float('nan')
            if kind == 'incomplete': state.pop('landing_mlp.4.weight')
            path = Path(folder)/f'{kind}.pt'; torch.save(bad, path)
            try: init.validate_source(path)
            except ValueError: pass
            else: raise AssertionError('Invalid source accepted: '+kind)
    old = deployment.load_actor(args.source_checkpoint)
    obs = TensorDict({'policy': torch.randn(32, 244), 'motion_control': torch.zeros(32, 3)}, batch_size=[32])
    obs['motion_control'][:, 0] = .7
    entry = torch.tensor([.6]*11+[.45]*3+[.3]*3)
    continuous = torch.tensor([.8, .8, .65, .65, .65, .5, .5, .5, .35, .35, .35, .2, .2, .2, .05, .05, .05])
    top_scan = torch.tensor([.35, .35, .20, .20, .20]+[.05]*12)
    def scan(values): return values[None, None].expand(32, 11, 17).reshape(32, 187)
    obs['policy'][:, 57:] = scan(entry)
    actor = make_actor(module, obs); actor.load_stage3_checkpoint(saved['actor_state_dict'])
    with torch.no_grad(): assert torch.equal(actor(obs), old(obs))
    assert actor.sequence_source_digest() == provenance['frozen_source_state_sha256']
    trainable = [name for name, p in actor.named_parameters() if p.requires_grad]
    assert trainable and all(k.startswith('sequence_mlp.') for k in trainable)
    fixture = copy.deepcopy(saved['m20_training_manifest'])
    fixture.update(m20_profile='ascent_entry_v1_stage4', refinement_version='m20_ascent_sequence_20261020_v1_stage4',
                   frozen_stage3_state_sha256=actor.sequence_source_digest(),
                   frozen_stage3_source_sha256=provenance['sha256'], sequence_limits_action_units=module.SEQUENCE_LIMITS,
                   test_fixture_only=True)
    fixture['actor_model_config'].update(class_name='rl_training.models.ascent_sequence_actor:AscentSequenceMLPModel',
                                        sequence_hidden_dims=[128, 64])
    args.zero_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'actor_state_dict': actor.state_dict(), 'critic_state_dict': saved['critic_state_dict'],
                'm20_training_manifest': fixture}, args.zero_checkpoint)
    native = deployment.load_actor(args.zero_checkpoint)
    with torch.no_grad(): assert torch.equal(native(obs), old(obs))
    with torch.no_grad(): actor.sequence_mlp[-1].bias.fill_(.4)
    for kind in ('zero', 'flat', 'descent', 'reverse', 'turn'):
        sample = obs.clone()
        if kind == 'zero': sample['motion_control'].zero_()
        if kind == 'flat': sample['policy'][:, 57:] = .2
        if kind == 'descent': sample['policy'][:, 57:] = scan(-entry)
        if kind == 'reverse': sample['motion_control'][:, 0] = -.7
        if kind == 'turn': sample['motion_control'][:, 1] = .3
        with torch.no_grad(): assert torch.equal(actor(sample), old(sample)), kind
    for profile, factor in [(continuous, .2), (top_scan, .2), (entry, 1.)]:
        sample = obs.clone(); sample['policy'][:, 57:] = scan(profile)
        gate = module.sequence_gate(sample['policy'], sample['motion_control'])
        assert torch.allclose(gate, torch.full_like(gate, factor))
        with torch.no_grad():
            delta = actor(sample)-old(sample)
            assert delta.abs().max() > 0.
            assert torch.all(delta.abs() <= actor.sequence_limits*factor+1.e-6)

    # Either front side may lead; contact <40ms and simultaneous landings must
    # not earn a sequential-pair bonus. Retreat and repeated contact cannot farm it.
    active = torch.ones(1, dtype=torch.bool); first = torch.zeros(1)
    def new_ledger(): return core.SequenceLedger(1, 'cpu', targets=4)
    def tick(ledger, levels, support=None, x=None):
        levels = torch.tensor([levels], dtype=torch.long)
        support = (levels >= 0) if support is None else torch.tensor([support])
        x = (levels.float()-1.)*.3+.12 if x is None else torch.tensor([x])
        return ledger.update(levels, support, x, first, active, .005, torch.tensor([.43]))
    for leader in (0, 1):
        ledger = new_ledger(); levels = [0, 0, 0, 0]; levels[leader] = 1
        for _ in range(7): assert tick(ledger, levels)['ordered'].sum() == 0
        tick(ledger, levels)
        for _ in range(10): tick(ledger, levels)
        levels[1-leader] = 1
        pair = sum(tick(ledger, levels)['paired'].item() for _ in range(8))
        assert pair == 1. and ledger.pair_paid[0, 0]
        for _ in range(30): assert tick(ledger, levels)['paired'].item() == 0.
        levels[leader] = 2
        for _ in range(8): tick(ledger, levels)
        levels[leader] = 3
        assert sum(tick(ledger, levels)['early_advance'].item() for _ in range(8)) == 1.
    ledger = new_ledger()
    events = [tick(ledger, [1, 1, 0, 0]) for _ in range(8)]
    assert sum(v['paired'].item() for v in events) == 0.
    assert sum(v['simultaneous'].item() for v in events) == 1.
    ledger = new_ledger()
    for _ in range(8): tick(ledger, [3, 2, 0, 0])
    assert ledger.missed[0, 0, :2].all() and ledger.missed[0, 1, 0]
    # Same front sequence has identical pair/advance accounting regardless of
    # whether the rear axle is on ground, tread1 or tread3 (no fixed gap).
    outcomes = []
    for rear in (0, 1, 3):
        ledger = new_ledger()
        for _ in range(20): tick(ledger, [1, 0, rear, rear])
        rows = [tick(ledger, [1, 1, rear, rear]) for _ in range(8)]
        outcomes.append((sum(r['paired'].item() for r in rows), sum(r['early_advance'].item() for r in rows)))
    assert outcomes == [(1., 0.)]*3

    positions = torch.tensor([[[.14, .2, .24], [.44, -.2, .39]]])
    targets = torch.tensor([[1, 2]]); enabled = torch.ones(1, 2, dtype=torch.bool)
    def guide(p): return core.front_guidance(p, torch.zeros(1, 2), targets, enabled, first, first)
    assert guide(positions)['height'].item() == 0.
    hovering = positions.clone(); hovering[..., 2] += .12; hovering.requires_grad_()
    assert (torch.autograd.grad(guide(hovering)['height'].sum(), hovering)[0][..., 2] > 0).all()
    before = positions.clone(); before[..., 0] -= .19; before[..., 2] -= .05; before.requires_grad_()
    assert (torch.autograd.grad(guide(before)['height'].sum(), before)[0][..., 2] < 0).all()
    ledger = new_ledger()
    opportunity = torch.ones(1, 2); phase = torch.ones(1, 2, dtype=torch.bool)
    assert ledger.approach_credit(targets, opportunity*2, opportunity, phase).item() == 0.
    assert ledger.approach_credit(targets, opportunity, opportunity, phase).item() == 2.
    for _ in range(5):
        assert ledger.approach_credit(targets, opportunity*2, opportunity, phase).item() == 0.
        assert ledger.approach_credit(targets, opportunity, opportunity, phase).item() == 0.
    xy = torch.tensor([[[.4, .2], [.4, -.2], [-.4, .2], [-.4, -.2]]])
    cost, _ = core.support_margin_cost(xy, torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 2), torch.zeros(1, 2))
    outside, _ = core.support_margin_cost(xy, torch.ones(1, 4, dtype=torch.bool), torch.tensor([[1., 0.]]), torch.zeros(1, 2))
    assert cost.item() == 0. and outside.item() > 0.
    segment, _ = core.support_margin_cost(xy, torch.tensor([[True, False, False, True]]), torch.tensor([[2., 1.]]), torch.zeros(1, 2))
    assert segment.item() > 0., 'Two contacts were treated as an infinite support line'
    # As rear wheels reach the same platform, the target pitch tends smoothly
    # to zero; it stays nonzero while the rear axle is below the front axle.
    pose = torch.tensor([[[1.0, .2, .6], [1.0, -.2, .6], [.3, .2, .2], [.3, -.2, .2]]])
    tilted = core.terrain_pitch_target(pose, first, count=3)
    pose[:, 2:, 0] = 1.0
    assert tilted.item() > .1 and core.terrain_pitch_target(pose, first, count=3).item() == 0.
    top = core.TopTransition(1, 'cpu')
    def top_tick(on, x=0., angle=0.):
        return top.update(torch.tensor([on]), torch.tensor([x]), torch.tensor([angle]), torch.tensor([.5]), active, .005)
    assert top_tick([True, True, False, False])['front'].item() == 1.
    assert top_tick([True, True, True, False])['rear'].item() == 0.
    assert top_tick([True]*4, .1)['rear'].item() == 1.
    for _ in range(50): assert top_tick([True]*4, .6, .3)['forward'].item() == 0.
    credit = 0.
    for i in range(100): credit += top_tick([True]*4, .1+.006*i)['forward'].item()
    assert .35 < credit <= .60+1.e-5
    assert top_tick([True]*4, .7)['ready'].item()
    assert not top_tick([True]*4, .2)['ready'].item(), 'Retreat triggered completion'
    assert top_tick([True]*4, .7)['forward'].item() == 0.
    assert not top_tick([True, True, False, False], 1.)['ready'].item()
    top.reset(slice(None))
    for _ in range(20): assert top_tick([True]*4)['rear'].item() == 0., 'No front-before-rear history'
    assert mdp.metrics(SimpleNamespace(), None) == {}
    # Execute the actual command class body without importing Isaac/Kit.
    # The parent adapter supplies only the existing fixed target command.
    tree = ast.parse((ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_sequence_profile.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SequenceVelocityCommand')
    class Parent:
        def _apply_ascent_command(self): self.vel_command_b[:, 0] = self.entry_speeds
    scope = {'stage2': SimpleNamespace(Stage2VelocityCommand=Parent)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual_stage4_command_class>', 'exec'), scope)
    command = scope['SequenceVelocityCommand']()
    command._env = SimpleNamespace(common_step_counter=0,
        cfg=SimpleNamespace(ascent_sequence_training_ramp=True, ascent_sequence_ramp_steps=9600),
        scene=SimpleNamespace(terrain=SimpleNamespace(terrain_types=torch.tensor([0, 2, 6]))))
    command.entry_speeds = torch.tensor([.5, .7, .7]); command.vel_command_b = torch.zeros(3, 3)
    command._apply_ascent_command()
    assert torch.allclose(command.vel_command_b[:, 0], torch.tensor([.3, .42, .7]))
    command._env._ascent_sequence_step_offset = 9600
    command._apply_ascent_command(); assert torch.equal(command.vel_command_b[:, 0], command.entry_speeds)
    command._env._ascent_sequence_step_offset = 0
    command._env.cfg.ascent_sequence_training_ramp = False
    command._apply_ascent_command(); assert torch.equal(command.vel_command_b[:, 0], command.entry_speeds)

    # Actual StairGuardedPPO, synthetic CPU rewards: one warmup then two
    # optimizer updates; all stage3 source parameters remain exactly frozen.
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    guarded = local_module('stair_guarded_ppo')
    batch = obs.clone(); batch['critic'] = torch.randn(32, 285)
    learner = make_actor(module, batch); learner.load_stage3_checkpoint(saved['actor_state_dict'])
    critic = MLPModel(batch, {'critic': ['critic', 'motion_control']}, 'critic', 1,
                     hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False)
    critic.load_state_dict(saved['critic_state_dict'], strict=True)
    storage = RolloutStorage('rl', 32, 4, batch, [16], device='cpu')
    algorithm = guarded.StairGuardedPPO(learner, critic, storage, device='cpu', num_learning_epochs=1,
        num_mini_batches=2, schedule='fixed', learning_rate=3.e-5, critic_warmup_iterations=1,
        anchor_coefficient=.1, anchor_stair_weight=.03, motion_regularization_coefficient=.02)
    algorithm.set_reference(learner.state_dict()); ppo = []
    for iteration in range(3):
        before = learner.sequence_mlp[-1].weight.detach().clone()
        for _ in range(4):
            with torch.inference_mode():
                actions = algorithm.act(batch)
                algorithm.process_env_step(batch, torch.randn(32)-.02*actions.square().mean(-1),
                                           torch.zeros(32, dtype=torch.long), {})
        with torch.inference_mode(): algorithm.compute_returns(batch)
        losses = algorithm.update()
        assert all(math.isfinite(float(v)) for v in losses.values())
        changed = not torch.equal(before, learner.sequence_mlp[-1].weight)
        assert changed == (iteration > 0)
        assert learner.sequence_source_digest() == fixture['frozen_stage3_state_sha256']
        ppo.append({'iteration': iteration, 'sequence_updated': changed, 'finite': True, 'complete_source_frozen': True})
    result = {'source': provenance, 'complete_stage3_initialization_exact': True, 'native_loader_exact': True,
              'only_new_branch_trainable': trainable, 'zero_flat_reverse_turn_descent_preserved': True,
              'entry_continuous_exit_authority_bounded': True, 'either_front_can_lead': True,
              'invalid_frozen_incomplete_and_nonfinite_sources_rejected': True,
              'same_tread_order_and_40ms_confirmation': True, 'early_advance_and_skip_counterexamples': True,
              'no_fixed_axle_tread_gap': True, 'riser_clearance_and_descent_direction': True,
              'approach_credit_not_repeatable': True, 'support_hull_and_finite_segment_checks': True,
              'top_front_rear_level_forward_sequence_and_retreat_checks': True,
              'top_pitch_target_gradually_flattens_after_rear_progress': True,
              'actual_command_class_speed_ramp_resume_and_evaluation_checks': True,
              'cpu_ppo_synthetic_integration': ppo, 'gpu_test_run': False,
              'zero_checkpoint_fixture': str(args.zero_checkpoint)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
