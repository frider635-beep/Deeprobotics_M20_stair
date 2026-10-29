#!/usr/bin/env python3
"""CPU rear-entry counterexamples and v9 manifest checks; no policy loading."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rear_entry_checks', ROOT/'scripts/tools/check_stair_ascent_pairing.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p = m.pair
e = m.loader.load_module(m.BASE+'path.stair_path_evidence')
turns_module = m.loader.load_module(m.BASE+'ascent.ascent_alternating')
YES = torch.tensor([True])


def run(depth=.25, targets=6):
    result = m.LandingRun(depth=depth, targets=targets)
    result.ledger = p.RearEntryPairLedger(1,'cpu',depth=depth,targets=targets)
    return result


def land_pair(r, axle, tread, side):
    r.land(2*axle+side,tread);r.land(2*axle+1-side,tread)


def sequences():
    cases = 0
    for depth in (.20,.25,.30,.35):
        for front_side in (0,1):
            for tail_side in (0,1):
                r = run(depth)
                for tread in range(1,7):land_pair(r,0,tread,front_side)
                # Two independent entry choices, then exactly one tail anchor.
                land_pair(r,1,1,1-tail_side)
                land_pair(r,1,2,tail_side)
                assert r.ledger.first_side[0,1] == -1
                for tread in range(3,7):land_pair(r,1,tread,tail_side)
                assert not r.ledger.failed.any() and not r.ledger.style_failed.any()
                assert r.ledger.first_side[0,0] == front_side and r.ledger.first_side[0,1] == tail_side
                assert r.ledger.covered.all() and r.front == r.rear == 6
                quality,tail = e.pair_quality(r.ledger,torch.tensor([6]))
                assert quality.eq(1).all() and tail.item() == 1
                # A fourth-tread reversal is penalized; fifth does not reanchor.
                bad = run(depth)
                land_pair(bad,1,1,tail_side);land_pair(bad,1,2,1-tail_side)
                land_pair(bad,1,3,tail_side);land_pair(bad,1,4,1-tail_side)
                land_pair(bad,1,5,tail_side)
                assert not bad.ledger.failed.any() and bad.ledger.style_failed.item()
                assert bad.ledger.style_bad_treads[0,1,3] and not bad.ledger.style_bad_treads[0,1,4]
                assert bad.ledger.first_side[0,1] == tail_side
                # Front entry reversals keep the old strict gait rule.
                front = run(depth);land_pair(front,0,1,front_side);land_pair(front,0,2,1-front_side)
                assert front.ledger.style_bad_treads[0,0,1] and front.ledger.style_failed.item()
                cases += 1
        for wheel in (2,3):
            missing = run(depth);missing.land(wheel,2,offset=.05)
            assert missing.ledger.failed.item()  # no skipped tread1
            early = run(depth);early.land(wheel,1);early.land(wheel,2)
            assert early.ledger.failed.item()  # mate must finish the previous tread
        simultaneous = run(depth)
        simultaneous.level[0,2:] = 1;simultaneous.x[0,2:] = .10
        simultaneous.land(2,1)
        assert simultaneous.ledger.failed.item()
    short = run(targets=2)
    land_pair(short,1,1,0);land_pair(short,1,2,1)
    assert not short.ledger.style_failed.item() and short.ledger.first_side[0,1] == -1
    assert e.pair_quality(short.ledger,torch.tensor([2]))[1].item() == 0
    short.ledger.reset(torch.tensor([0]))
    assert short.ledger.first_side.eq(-1).all() and not short.ledger.rear_swing_style_failed.any()
    return cases


def swing_softening():
    near = torch.ones(1,2,dtype=torch.bool)
    def trial(wheel,grace=False,soft=False,simultaneous=False):
        turns = turns_module.SwingAlternation(1,'cpu')
        support = torch.ones(1,4,dtype=torch.bool);z = torch.zeros(1,4);vz = torch.zeros_like(z)
        repeat_grace = torch.tensor([[False,grace]])
        soft_repeat = torch.tensor([[False,soft]])
        for _ in range(10):turns.update(support,z,vz,near,YES,.005)
        turns.last_side[0,wheel//2] = wheel%2
        chosen = [2,3] if simultaneous else [wheel]
        support[:,chosen] = False;z[:,chosen] = .03
        soft_events = 0
        for _ in range(5):
            result = turns.update(support,z,vz,near,YES,.005,repeat_grace=repeat_grace,soft_repeat=soft_repeat)
            soft_events += result['soft_violation'].sum().item()
        return turns,soft_events
    turns,events = trial(2,grace=True,soft=True)
    assert not turns.failed.item() and events == 0
    turns,events = trial(2,soft=True)
    assert not turns.failed.item() and events == 1 and turns.violations[0,1] == 1
    turns,_ = trial(0,grace=True,soft=True);assert turns.failed.item()  # front unchanged
    turns,_ = trial(2,grace=True,soft=True,simultaneous=True);assert turns.failed.item()


def guidance():
    for depth in (.20,.25,.30,.35):
        for side in (0,1):
            r = run(depth);land_pair(r,1,1,side)
            turns = turns_module.SwingAlternation(1,'cpu')
            turns.seen_support.fill_(True);turns.armed.fill_(True)
            turns.last_side[:,1] = side  # same side may start tread2
            position = torch.zeros(1,4,3);position[...,0] = .1;position[...,2] = .19
            support = torch.ones(1,4,dtype=torch.bool);stable = torch.ones(1,4)*.1
            def measure():
                return p.pairing_guidance(r.ledger,turns,support,stable,position,torch.zeros(1,4),
                    torch.zeros(1),torch.zeros(1),torch.tensor([depth]),torch.tensor([.1]),r.goals,YES)
            _,_,_,st = measure()
            assert not st['rear_wait'].any() and st['rear_phase'].item() == 0
            # If one wheel already landed, the unfinished same-side wheel stays eligible.
            r.land(3-side,2)
            _,_,_,st = measure()
            assert st['permission'][0,2+side]
            r.land(2+side,2);land_pair(r,1,3,1-side)
            position[0,2+side,0] = 3*depth+.05
            position[0,2+side,2] = .52
            _,_,_,st = measure()
            assert st['rear_wait'][0,2+side] and st['rear_phase'].item() > 0
            r.land(3-side,4)
            _,_,_,st = measure()
            assert not st['rear_wait'].any() and st['rear_phase'].item() == 0


def course_and_migration():
    ids = torch.tensor([11]);row = torch.tensor([1]);no = ~YES
    def course(front=1.,rear=0.,tail=0.):
        state = e.RearEntryEvidenceCourse(40,'cpu')
        state.skill[ids] = state.prefixes.index(6);state.geometry_probe[ids] = True
        for bucket in (0,1):
            state.speed_bucket[ids] = bucket
            for _ in range(8):
                up,down = state.update(ids,YES,YES,row,no,
                    pair_quality=torch.tensor([[front,rear]]),rear_tail_quality=torch.tensor([tail]))
        return state,up,down
    state,up,down = course()
    assert up.item() and not down.item()  # rear style cannot gate terrain height
    _,up,_ = course(front=.8);assert not up.item()  # old front quality gate retained
    legacy = e.EvidenceCourse(40,'cpu').state_dict()
    legacy['skill'][ids] = 3;legacy['mastered'][ids,:2] = 3
    legacy['window'].fill_(1);legacy['quality_window'].fill_(1);legacy['trials'].fill_(8)
    mapped = state.migrate_v9(legacy)
    assert legacy['quality_window'].all() and legacy['window'].all()
    assert torch.equal(mapped['window'],legacy['window'])
    assert torch.equal(mapped['quality_window'][...,0],legacy['quality_window'][...,0])
    assert not mapped['quality_window'][...,1].any() and not mapped['rear_tail_window'].any()
    for key in ('skill','mastered','bad','promotion_hold','event_counts','speed_bucket'):
        assert torch.equal(legacy[key],mapped[key])
    state.load_state_dict(mapped)
    clone = e.RearEntryEvidenceCourse(40,'cpu');clone.load_state_dict(state.state_dict())
    assert torch.equal(state.mastered,clone.mastered)
    manifest = json.loads((ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher_v1_stage2/2026-10-27_16-40-32_ascent_entry_v1_stage9_v9/params/launch_manifest.json').read_text())
    target = dict(manifest['stair_path_reward_settings'],version=p.VERSION,ascent_pairing=deepcopy(p.SETTINGS),
        narrow_ascent_practice=deepcopy(m.loader.load_module(m.BASE+'path.stair_path_layout').NARROW_ASCENT_SETTINGS),
        ascent_height_practice=deepcopy(m.loader.load_module(m.BASE+'path.stair_path_height_practice').SETTINGS),
        course_evidence=deepcopy(e.TRANSITION_SETTINGS),**{k:manifest[k] for k in (
        'stair_path_reward_version','stair_path_config','control_profile','descent_limits_action_units','descent_authority_schedule')})
    init = m.loader.load_module('algorithms.initialization.stair_path_init')
    weights=dict(manifest['reward_weights'],ascent_rear_paired=p.REAR_PAIR_WEIGHT,
        ascent_landing_order=p.LANDING_ORDER_WEIGHT,ascent_rear_phase=p.REAR_PHASE_WEIGHT,
        ascent_rear_landing_credit=p.LANDING_QUALITY_SETTINGS['rear_landing_credit_weight'])
    init.validate_ascent_pairing_resume(manifest,target,weights,migrate=True,source='v9')
    bad = dict(weights);bad['stair_path_slip'] *= .5
    try:init.validate_ascent_pairing_resume(manifest,target,bad,migrate=True,source='v9')
    except ValueError:pass
    else:raise AssertionError('descent/path weight changed')


def observer_and_descent(baseline_root):
    fixture_rng = torch.get_rng_state()
    env,observer = m.perf.fixture(m.loader,56,progressive=True,pairing=True,rear_entry=True)
    assert isinstance(observer.sequence,p.RearEntryPairLedger)
    assert isinstance(env._ascent_adaptive_curriculum,e.RearEntryEvidenceCourse)
    prior_loader = m.perf.load_source(baseline_root,'_m20_rear_entry_v9_baseline')
    torch.set_rng_state(fixture_rng)
    old_env,old = m.perf.fixture(prior_loader,56,progressive=True,pairing=True)
    comparisons = 0
    masks = m.loader.load_module(m.BASE+'ascent.ascent_adaptive')
    for tick in range(1,65):
        step_rng = torch.get_rng_state()
        m.perf.step(env,observer,tick)
        torch.set_rng_state(step_rng)
        m.perf.step(old_env,old,tick)
        down = masks.descent_mask(env)
        before,after = m.perf.tensors(old),m.perf.tensors(observer)
        for key,value in before.items():
            if value.ndim and value.shape[0] == env.num_envs:
                assert torch.equal(value[down],after[key][down]),f'descent state changed {tick}:{key}'
                comparisons += 1
        for name in ('ascent_entry_mdp','ascent_adaptive_mdp','ascent_alternating_mdp','ascent_unified_mdp'):
            left = prior_loader.load_module(m.BASE+'ascent.'+name)
            right = m.loader.load_module(m.BASE+'ascent.'+name)
            for term in getattr(right,'NAMES',getattr(right,'EXTRA_NAMES',())):
                if hasattr(right,'reward'):
                    torch.testing.assert_close(left.reward(old_env,term)[down],right.reward(env,term)[down],rtol=0,atol=0)
        for term in ('clearance','rear_paired','landing_order','rear_phase'):
            torch.testing.assert_close(p.reward(env,term)[down],torch.zeros_like(p.reward(env,term)[down]),rtol=0,atol=0)
    # Inject the already separately checked soft event through the real
    # observer caller: it must charge a penalty without latching task failure.
    env,observer = m.perf.fixture(m.loader,56,progressive=True,pairing=True,rear_entry=True)
    original_update = observer.turns.update
    def soft_event(*args,**kwargs):
        result = original_update(*args,**kwargs)
        result['soft_violation'][0,1] = True;result['violation'][0,1] = True
        return result
    observer.turns.update = soft_event
    m.perf.step(env,observer,1)
    assert observer.sequence.style_failed[0] and observer.sequence.rear_swing_style_failed[0]
    assert not observer.sequence.failed[0] and not observer.turns.failed[0]
    assert observer.failure_reason[0] == 0 and observer.window['alternating_turn'][0] > 0
    observer.prefix_ready[0] = True
    adaptive = m.loader.load_module(m.BASE+'ascent.ascent_adaptive_mdp')
    assert adaptive.finished(env)[0]  # style must not rewrite physical completion
    return comparisons


def diagnostics():
    r = run(targets=18)
    for tread in range(1,7):land_pair(r,0,tread,0)
    land_pair(r,1,1,0);land_pair(r,1,2,1)
    for tread in range(3,7):land_pair(r,1,tread,1)
    diag = p.PairingDiagnostics(1,'cpu')
    observer = NS(sequence=r.ledger,episode_goals=torch.tensor([6]),failure_reason=torch.tensor([0]),
        prefix_ready=YES,top_ready=~YES)
    def values():
        return dict(zip(p.DIAGNOSTIC_FIELDS,diag.values(observer,torch.tensor([0]))[0].tolist()))
    result = values()
    assert result['rear_tail_evaluated'] == result['rear_tail_strict_success'] == 1
    assert result['rear_tail_ordered_pairs'] == result['rear_tail_required_pairs'] == 4
    assert result['rear_tread2_order_violation'] == 0
    r.ledger.rear_swing_style_failed[0] = True;r.ledger.style_failed[0] = True
    result = values()
    assert result['rear_tail_strict_success'] == result['strict_success'] == 0
    observer.episode_goals.fill_(2)
    assert values()['rear_tail_evaluated'] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = sequences();swing_softening();guidance();course_and_migration();diagnostics()
    comparisons = observer_and_descent(args.baseline_root)
    print(json.dumps(dict(version=p.VERSION,mirrored_sequence_cases=cases,
        rear_entry_free=True,tail_anchor_fixed=True,front_order_unchanged=True,
        rear_repeat_penalty_not_physical_failure=True,rear_style_not_terrain_gate=True,
        no_skipping_or_early_axle_advance=True,simultaneous_swing_and_landing_still_fail=True,
        descent_tensor_comparisons=comparisons,descent_reward_outputs_identical=True,
        manifest_and_synthetic_curriculum_checks=True,policy_checkpoints_loaded=0,
        physics_simulation=False),indent=2))


if __name__ == '__main__':main()
