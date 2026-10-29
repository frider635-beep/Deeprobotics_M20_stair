#!/usr/bin/env python3
"""CPU geometry/course counterexamples and JSON contracts; never load a policy."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('short_tread_fixture',ROOT/'scripts/tools/check_stair_ascent_pairing.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
load = m.loader.load_module
BASE = m.BASE
p = m.pair
short = load(BASE+'ascent.ascent_short_tread')
layout = load(BASE+'path.stair_path_layout')
height = load(BASE+'path.stair_path_height_practice')
evidence = load(BASE+'path.stair_path_evidence')


def clearance():
    enabled = torch.ones(1,2,dtype=torch.bool)
    permission = torch.tensor([[True,False]])
    def cost(bottom, vx=.3, depth=.25, x=-.10):
        return short.rear_riser_cost(torch.full((1,2),x),torch.full((1,2),bottom),
            torch.full((1,2),.25),torch.tensor([depth]),enabled,permission,
            torch.full((1,2),vx))[0]
    # High risers must retain a useful lift signal while the old square clips.
    bottom = torch.zeros(1,2,requires_grad=True)
    value,_ = short.rear_riser_cost(torch.full((1,2),-.1),bottom,
        torch.full((1,2),.25),torch.tensor([.25]),enabled,permission,torch.zeros(1,2))
    value.sum().backward()
    assert bottom.grad[0,0] < 0 and bottom.grad[0,1] == 0
    assert cost(.01).sum() < cost(0.).sum()
    # Waiting followers have no lift penalty, but cannot push low into a riser.
    assert cost(0.,vx=0.)[0,1] == 0 and cost(0.,vx=.3)[0,1] > 0
    assert cost(0.,vx=-.3)[0,1] == 0
    assert cost(.30).eq(0).all()  # no reward for excessive lifting
    for depth in (.30,.35):assert cost(0.,depth=depth).eq(0).all()
    assert cost(0.,x=-.60).eq(0).all()
    return float(bottom.grad[0,0])


def worst_rear_aggregation():
    cost = short.aggregate_rear_cost(torch.tensor([[2., 0.], [1., 1.]]))
    assert torch.allclose(cost, torch.tensor([3.5, 2.75]))
    assert cost[0] > cost[1]
    return cost.tolist()


def quotas():
    for shares in layout.READINESS_GROUP_SHARES:
        before = layout.probabilities_for_shares(shares,practice_share=None)
        after = layout.probabilities_for_shares(shares)
        assert after[9:] == before[9:] and after[8] == before[8]
        assert abs(sum(after)-1.) < 1.e-12
        for c in layout.NARROW_ASCENT_COLUMNS:
            assert abs(after[c]/layout.ASCENT_FRACTION-.0125) < 1.e-12
            assert abs(after[c]+after[c+1]-before[c]-before[c+1]) < 1.e-12
        assert min(after) > 0


def height_course():
    n=280;types=torch.arange(n)%28
    course=evidence.RearEntryEvidenceCourse(n,'cpu')
    narrow=torch.isin(types,torch.tensor(layout.NARROW_ASCENT_COLUMNS))
    terrain=NS(terrain_types=types,terrain_levels=torch.zeros(n,dtype=torch.long),
        terrain_origins=torch.arange(6*28*3).reshape(6,28,3).float(),env_origins=torch.zeros(n,3))
    ids=torch.arange(n);yes=torch.ones(n,dtype=torch.bool)
    before=deepcopy(course.state_dict())
    torch.manual_seed(42)
    height.resample_height_practice(terrain,course,ids,yes)
    rows=terrain.terrain_levels.clone()
    assert course.skill[narrow].eq(8).all()
    assert set(rows[narrow].tolist()) == set(range(6))
    for name,value in before.items():
        if isinstance(value,torch.Tensor):
            assert torch.equal(course.state_dict()[name][~narrow],value[~narrow]),name
    # Neither own nor wider mastery/outcomes can change the practice schedule.
    course.mastered.fill_(8);course.bad.fill_(12);course.window.fill_(1)
    torch.manual_seed(42)
    height.resample_height_practice(terrain,course,ids,yes)
    assert torch.equal(terrain.terrain_levels,rows)
    assert not course.window[narrow].any()
    assert torch.equal(terrain.env_origins[narrow],terrain.terrain_origins[rows[narrow],types[narrow]])
    adapter=load(BASE+'ascent.ascent_adaptive_mdp')
    def forbid_update(*args):raise AssertionError('practice entered terrain promotion/demotion')
    terrain.update_env_origins=forbid_update
    cfg=NS(ascent_practice_only_columns=layout.NARROW_ASCENT_COLUMNS,
        ascent_adaptive_reference_columns=(8,),ascent_adaptive_ascent_columns=9,
        ascent_adaptive_training=True,stair_ascent_pairing_v9=True,
        ascent_adaptive_geometry=True,ascent_adaptive_height_grid=[[h]*28 for h in (.1,.12,.15,.18,.20,.25)],
        ascent_adaptive_depth_grid=[[.25]*28]*6,ascent_sequence_step_count=18)
    ledger=p.RearEntryPairLedger(n,'cpu',targets=18)
    observer=NS(sequence=ledger,stats={'physics_samples':torch.ones(n)},
        prefix_ready=torch.ones(n,dtype=torch.bool),top_ready=torch.ones(n,dtype=torch.bool),episode_goals=course.goals())
    env=NS(cfg=cfg,num_envs=n,device='cpu',scene=NS(terrain=terrain),
        _ascent_entry_observer=observer,_ascent_adaptive_curriculum=course)
    selected=ids[narrow];before=deepcopy(course.state_dict())
    for i in range(32):
        observer.prefix_ready.fill_(i%2==0)
        adapter.curriculum(env,selected)
        assert course.skill[narrow].eq(8).all()
        assert not course.last_events[narrow].any() and not course.evidence_last[narrow].any()
    for name in ('completed','good','bad','mastered','event_counts','height_age','promotion_hold'):
        assert torch.equal(course.state_dict()[name],before[name]),name
    # Practice cannot influence the automatic global sampling course either.
    ready=load(BASE+'path.stair_path_readiness').ReadinessSampler(mode='auto')
    ready.observe(types[selected],rows[selected],course.goals()[selected],
        torch.full((len(selected),),.5),torch.ones(len(selected),8))
    assert not ready.totals.any()
    return dict(practice_fraction_of_ascent_pool=.05,height_rows_all_sampled=True,
        no_prefix_height_or_sampling_course_updates=True)


def manifest():
    # Validate the actual launch JSON without opening a checkpoint.
    run=ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher_v1_stage2/2026-10-28_12-56-25_ascent_entry_v1_stage10_v1'
    prior=json.loads((run/'params/launch_manifest.json').read_text())
    target=dict(prior['stair_path_reward_settings'],version=p.VERSION,
        ascent_pairing=deepcopy(p.SETTINGS),course_evidence=deepcopy(evidence.TRANSITION_SETTINGS),
        narrow_ascent_practice=deepcopy(layout.NARROW_ASCENT_SETTINGS),ascent_height_practice=deepcopy(height.SETTINGS),
        **{k:prior[k] for k in ('stair_path_reward_version','stair_path_config','control_profile',
            'descent_limits_action_units','descent_authority_schedule')})
    validator=load('algorithms.initialization.stair_path_init').validate_ascent_pairing_resume
    weights=dict(prior['reward_weights'],ascent_rear_paired=p.REAR_PAIR_WEIGHT,
        ascent_landing_order=p.LANDING_ORDER_WEIGHT,ascent_rear_phase=p.REAR_PHASE_WEIGHT,
        ascent_rear_landing_credit=p.LANDING_QUALITY_SETTINGS['rear_landing_credit_weight'])
    validator(prior,target,weights,migrate=True,source='v10')
    try:validator(prior,target,weights)
    except ValueError:pass
    else:raise AssertionError('v10 migration must be explicit')
    native=dict(prior,refinement_version=p.VERSION,reward_weights=weights,ascent_pairing=deepcopy(p.SETTINGS),
        narrow_ascent_practice=deepcopy(layout.NARROW_ASCENT_SETTINGS),ascent_height_practice=deepcopy(height.SETTINGS))
    validator(native,target,weights)
    for key in ('stair_path_config','ascent_height_practice','narrow_ascent_practice'):
        bad=deepcopy(target);bad[key]={}
        try:validator(prior,bad,weights,migrate=True,source='v10')
        except ValueError:pass
        else:raise AssertionError('protected contract accepted: '+key)


if __name__=='__main__':
    torch.set_num_threads(1);torch.manual_seed(42)
    gradient=clearance();aggregation=worst_rear_aggregation();quotas();heights=height_course();manifest()
    print(json.dumps(dict(rear_lift_gradient_on_25cm_riser=gradient,
        worst_rear_aggregation=aggregation,**heights,
        narrow_outcomes_do_not_control_any_course=True,all_heights_retained=True,
        changed_height_evidence_isolated=True,unchanged_depth_quotas=True,
        actual_v10_launch_json_validated=True,policy_checkpoints_loaded=0),indent=2))
