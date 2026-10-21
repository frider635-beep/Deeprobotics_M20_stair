#!/usr/bin/env python3
"""CPU audit of inherited alternation weights and desired-vs-skipped tread credit."""
import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    audit = deployment.training_module('ascent_reward_audit')
    core = deployment.training_module('ascent_sequence')
    manifest = json.loads((args.source_run/'params/launch_manifest.json').read_text())
    terms = {k: NS(weight=v) for k, v in manifest['reward_weights'].items()}
    before = {k: getattr(terms.get(k), 'weight', None) for k in audit.CONFLICTING_REWARDS}
    try: audit.audit_conflicting_rewards(terms)
    except ValueError: pass
    else: raise AssertionError('Historical residual alternation was not detected')
    disabled = audit.disable_conflicting_rewards(terms)
    assert all(row['weight'] in (None, 0.) for row in disabled.values())
    for name in audit.CONFLICTING_REWARDS:
        terms[name] = NS(weight=1.)
        try: audit.audit_conflicting_rewards(terms)
        except ValueError: pass
        else: raise AssertionError('Conflicting reward could be re-enabled: '+name)
        terms[name].weight = 0.
    for name in ['stair_up_task_crossing', 'ascent_stage2_front_slip', 'ascent_stage2_rear_slip',
                 'stair_front_riser_face_cost', 'undesired_contacts', 'bad_orientation_penalty']:
        assert terms[name].weight == manifest['reward_weights'][name], name

    def trace(schedule):
        ledger = core.SequenceLedger(1, 'cpu', targets=4)
        level = torch.zeros(1, 4, dtype=torch.long)
        totals = {k: 0. for k in ['ordered', 'missing', 'paired', 'simultaneous', 'early_advance', 'dwell']}
        wheel_weights = torch.tensor([1.5, 1.5, 1., 1.])
        for wheel, target in schedule:
            level[0, wheel] = target
            for _ in range(8):
                result = ledger.update(level, torch.ones(1, 4, dtype=torch.bool),
                    (level.float()-1.)*.30+.12, torch.zeros(1), torch.ones(1, dtype=torch.bool), .005)
                for key in totals:
                    value = result[key]*wheel_weights if key in ['ordered', 'missing', 'dwell'] else result[key]
                    totals[key] += float(value.sum())
        score = (.20*totals['ordered']-.90*totals['missing']+.60*totals['paired']-
                 .15*totals['simultaneous']-.35*totals['early_advance']+.08*totals['dwell'])
        return {'events': totals, 'event_reward_sum': score,
                'front_coverage': float(ledger.covered[:, :2].float().mean()),
                'rear_coverage': float(ledger.covered[:, 2:].float().mean())}
    desired = trace([(wheel, step) for step in range(1, 5) for wheel in (0, 1, 2, 3)])
    crossed = trace([(wheel, step) for step in range(1, 5) for wheel in ((step-1) % 2, 2+(step-1) % 2)])
    assert desired['front_coverage'] == desired['rear_coverage'] == 1.
    assert desired['events']['missing'] == desired['events']['early_advance'] == 0.
    assert desired['events']['paired'] == 4.
    assert crossed['events']['missing'] > 0. and crossed['events']['paired'] == 0.
    assert desired['event_reward_sum'] > crossed['event_reward_sum']

    # Execute the actual gait-neutral whole-body crossing function without Kit:
    # same-tread support and old style-invalid flags must not suppress credit.
    path = ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ascent_task_crossing_events')
    scope = {'torch': torch, 'TUNING': manifest['stair_tuning']}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual_task_crossing_function>', 'exec'), scope)
    state = {'map_landed': torch.ones(1, 1, 4, dtype=torch.bool), 'map_half_width': torch.full((1, 1), .05),
             'map_z': torch.full((1, 1), .15), 'radius': torch.full((1, 4), .09),
             'map_task_paid': torch.zeros(1, 1, dtype=torch.bool),
             'map_correct': torch.zeros(1, 1, 2, dtype=torch.bool),
             'map_duplicate_paid': torch.ones(1, 1, 2, dtype=torch.bool)}
    wheels = torch.zeros(1, 4, 3); wheels[:, :, 2] = .24
    result = scope['ascent_task_crossing_events'](state, torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1, 4), .2), torch.zeros(1, 1, 4), wheels, torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[0., 0., .5]]), torch.tensor([[1., 0., 0., 0.]]))
    assert result.item(), 'Paired same-tread support suppressed neutral task crossing'
    result = {'historical_source_profile': manifest['m20_profile'],
              'historical_conflicting_nonzero': {k: v for k, v in before.items() if v not in (None, 0.)},
              'disabled_rewards': disabled, 're_enable_guard_all_terms_passed': True,
              'safety_progress_and_front_rear_slip_preserved': True,
              'desired_same_tread_trace': desired, 'cross_tread_alternation_trace': crossed,
              'actual_task_crossing_pays_with_same_tread_support': True,
              'scope': 'Actual pure helper and source function plus synthetic event traces; not trained gait acceptance',
              'gpu_test_run': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
