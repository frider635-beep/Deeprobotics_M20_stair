#!/usr/bin/env python3
"""Counterexamples for stage-two tread credit, misses and full v1 warm start."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deployment


def local_module(name):
    return deployment.training_module(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    mdp = local_module('ascent_entry_stage2')
    init = local_module('ascent_entry_stage2_init')
    ledger = mdp.TreadLedger(1, 'cpu')
    active = torch.ones(1, dtype=torch.bool)
    on = torch.ones(1, 4, dtype=torch.bool)
    off = ~on
    def tick(level, support=on, x=None, enabled=active):
        levels = torch.full((1, 4), level, dtype=torch.long)
        position = torch.full((1, 4), (level-1)*.30+.12 if x is None else x)
        return ledger.update(levels, support, position, torch.zeros(1), enabled, .005)
    totals = {k: 0. for k in ('ordered', 'missing', 'dwell')}
    for level in range(1, 5):
        for i in range(8):
            events = tick(level)
            if i < 7:
                assert events['ordered'].sum() == 0, 'Contact shorter than 40ms qualified'
            for key in totals:
                totals[key] += events[key].sum().item()
        for i in range(20):
            events = tick(level)
            assert all(value.sum() == 0 for value in events.values()), 'Same tread paid twice'
    assert ledger.covered.all() and not ledger.missed.any()
    assert totals['ordered'] == 16. and totals['missing'] == 0.
    assert abs(totals['dwell']-16.) < 1.e-5
    ledger.reset(slice(None))
    for _ in range(8): tick(1)
    missing = ordered = 0.
    for _ in range(8):
        events = tick(3)
        missing += events['missing'].sum().item()
        ordered += events['ordered'].sum().item()
    assert missing == 4. and ordered == 4.
    assert ledger.missed[:, :, 1].all() and not ledger.covered[:, :, 1].any()
    for _ in range(30):
        assert tick(3)['missing'].sum() == 0, 'Irrecoverable miss kept paying a cost'
    ledger.reset(slice(None))
    missing = 0.
    for _ in range(8): missing += tick(2)['missing'].sum().item()
    assert missing == 4., 'Skipping the first physical tread was not detected'
    ledger.reset(slice(None))
    for _ in range(3): tick(1)
    for _ in range(2): tick(1, off)
    for _ in range(3): event = tick(1)
    assert event['ordered'].sum() == 4., '10ms same-tread gap should bridge'
    ledger.reset(slice(None))
    for _ in range(3): tick(1)
    for _ in range(3): tick(1, off)
    for _ in range(3): event = tick(1)
    assert not ledger.covered.any(), '15ms gap incorrectly bridged'
    ledger.reset(slice(None))
    dwell = 0.
    for _ in range(15):
        for _ in range(3): dwell += tick(1)['dwell'].sum().item()
        for _ in range(3): tick(1, off)
    assert dwell <= 4. and not ledger.covered.any(), 'Contact chatter farmed dwell rewards'
    ledger.reset(slice(None))
    for _ in range(40):
        assert all(v.sum() == 0 for v in tick(0, x=-.1).values()), 'Flat support earned stair credit'
    event = tick(0, off, x=2.)
    assert event['missing'].sum() == 16.
    assert tick(0, off, x=2.)['missing'].sum() == 0
    ledger.reset(slice(None))
    assert all(v.sum() == 0 for v in tick(3, x=2., enabled=~active).values())
    assert not ledger.missed.any() and not ledger.covered.any()
    two = mdp.TreadLedger(2, 'cpu')
    two.covered[:] = True
    two.reset(torch.tensor([0]))
    assert not two.covered[0].any() and two.covered[1].all()
    assert torch.allclose(mdp.pair_slip_cost(torch.tensor([[.8, 0.]])), torch.tensor([.6]))
    assert mdp.pair_slip_cost(torch.zeros(1, 2)).item() == 0.

    provenance = init.validate_checkpoint(args.source_checkpoint)
    source = deployment.load_actor(args.source_checkpoint)
    obs = TensorDict({'policy': torch.randn(64, 244), 'motion_control': torch.zeros(64, 3)}, batch_size=[64])
    obs['motion_control'][:, 0] = .7
    obs['policy'][:, 57:] = torch.linspace(.8, -.8, 17)[None, None].expand(64, 11, 17).reshape(64, 187)
    actor = deployment.ascent_entry_module().AscentEntryMLPModel(obs, {'actor': ['policy', 'motion_control']}, 'actor', 16,
        hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False,
        distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'log'})
    saved = torch.load(args.source_checkpoint, map_location='cpu', weights_only=True)
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    with torch.no_grad(): target = source(obs)
    assert torch.equal(actor(obs), target), 'Warm start lost learned v1 correction'
    frozen = {k: p.detach().clone() for k, p in actor.named_parameters() if not p.requires_grad}
    before = actor.entry_mlp[-1].weight.detach().clone()
    optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=5.e-5)
    optimizer.zero_grad()
    (actor(obs)-target-.01).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, actor.entry_mlp[-1].weight)
    assert all(torch.equal(value, dict(actor.named_parameters())[key]) for key, value in frozen.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        checkpoint = Path(temporary)/'fixture.pt'
        altered = copy.deepcopy(saved)
        altered['actor_state_dict']['mlp.0.weight'][0, 0] += 1.
        torch.save(altered, checkpoint)
        try: init.validate_checkpoint(checkpoint)
        except ValueError: pass
        else: raise AssertionError('Corrupt frozen policy passed provenance check')
        altered = copy.deepcopy(saved)
        altered['m20_training_manifest']['m20_profile'] = 'ascent_entry_v1_stage2'
        altered['m20_training_manifest']['refinement_version'] = 'm20_ascent_entry_20261020_v1_stage2'
        torch.save(altered, checkpoint)
        init.validate_checkpoint(checkpoint)
        deployed = deployment.load_actor(checkpoint)
        assert torch.equal(deployed(obs), target), 'Stage2 native deployment loader differs'
    result = {'source': provenance, 'complete_actor_warm_start_exact': True,
              'only_entry_branch_updates': True, 'original_frozen_parameters_exact': True,
              'one_credit_per_wheel_tread': True, 'first_and_second_missing_treads_detected': True,
              'bounded_gap_bridge_and_chatter_counterexamples': True,
              'missing_events_do_not_repeat': True, 'flat_and_inactive_do_not_pay': True,
              'subset_reset_isolated': True, 'pair_worst_wheel_slip_counterexample': True,
              'corrupt_frozen_checkpoint_rejected': True, 'stage2_deployment_loader_exact': True}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
