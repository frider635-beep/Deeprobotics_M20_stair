#!/usr/bin/env python3
"""CPU landing-quality counterexamples and real manifest checks; no policy loads."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('quality_fixture',ROOT/'scripts/tools/check_stair_ascent_pairing.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
q=m.loader.load_module(m.BASE+'ascent.ascent_landing_quality')
p=m.pair


def trajectory(wheel=3,depth=.20,scrub=False,sliding=False,low=False,load=40.,vz=0.,
               active=True,delay_physical=False,denied=False):
    tracker=q.LandingQuality(1,'cpu',targets=4)
    state=dict(target=torch.ones(1,4,dtype=torch.long),enabled=torch.zeros(1,4,dtype=torch.bool),
        hold=torch.zeros(1,4,dtype=torch.bool),transition_hold=torch.zeros(1,4,dtype=torch.bool))
    state['enabled'][0,wheel]=active
    permission=state['enabled'].clone()
    if denied:permission.zero_()
    ledger=NS(covered=torch.zeros(1,4,4,dtype=torch.bool),targets=torch.arange(1,5)[None,None],
        failed=torch.tensor([False]),style_bad_treads=torch.zeros(1,2,4,dtype=torch.bool),
        pair_paid=torch.zeros(1,4,dtype=torch.bool),rear_pair_paid=torch.zeros(1,4,dtype=torch.bool))
    total=torch.zeros(4);credit=0.;events=[]
    def step(x,bottom,supporting=False,face=False,slip=0.,force_z=0.,physical=False):
        nonlocal credit
        pos=torch.zeros(1,4,3);pos[...,0]=-.6;pos[...,2]=.09
        pos[0,wheel]=torch.tensor([x,0.,bottom+.09])
        support=torch.zeros(1,4,dtype=torch.bool);support[0,wheel]=supporting
        faces=torch.zeros_like(support);faces[0,wheel]=face
        force=torch.zeros(1,4,3);force[0,wheel]=torch.tensor([25. if face else 0.,0.,force_z])
        slips=torch.zeros(1,4);slips[0,wheel]=slip
        velocity=torch.zeros(1,4);velocity[0,wheel]=vz if supporting else 0.
        level=torch.zeros(1,4,dtype=torch.long);level[0,wheel]=1 if supporting else 0
        tracker.observe(state,permission,pos,velocity,support,faces,force,slips,level,
            torch.zeros(1),torch.zeros(1),torch.tensor([depth]),torch.tensor([.15]),torch.tensor([active]),.005)
        if physical:ledger.covered[0,wheel,0]=True
        event=tracker.credit(ledger,torch.tensor([active]),torch.tensor([4]))
        total.add_(event['ordered'][0]);credit+=event['front_credit'].item()+event['rear_credit'].item()
        events.append(event)
    step(-.15,.0)
    step(-.08,.02 if low else .20)
    if scrub:step(-.03,.08,face=True)
    step(.02,.10 if low else .19)
    if sliding:step(.07,.15,True,slip=.2,force_z=40.)
    # A 40ms physical landing can precede stricter quality qualification.
    for i in range(24):step(.08,.15,True,slip=.01,force_z=load,
        physical=(i >= (12 if delay_physical else 7)))
    return tracker,ledger,total,credit,events


def counterexamples():
    cases=0
    for wheel in range(4):
        for depth in (.20,.25,.30,.35):
            tracker,ledger,total,credit,_=trajectory(wheel,depth)
            assert ledger.covered[0,wheel,0] and tracker.paid[0,wheel,0]
            assert 0. < total[wheel] <= 1. and 0. < credit <= 2.
            assert (total > 0).sum() == 1
            # Staying on the same tread pays once, not every physics frame.
            assert tracker.paid.sum() == 1
            for options in ({'scrub':True},{'sliding':True},{'low':True},
                            {'load':5.},{'vz':.3},{'denied':True}):
                bad,physical,paid,improvement,_=trajectory(wheel,depth,**options)
                assert physical.covered[0,wheel,0]  # physical evidence retained
                assert not bad.paid.any() and not paid.any() and improvement == 0.,options
                cases+=1
    for wheel in (2,3):
        tracker,ledger,total,credit,_=trajectory(wheel,delay_physical=True)
        assert tracker.paid[0,wheel,0] and total[wheel] > 0
        inactive,_,paid,credit,_=trajectory(wheel,active=False)
        assert not inactive.attempted.any() and not paid.any() and credit == 0
        tracker.reset(torch.tensor([0]))
        assert not tracker.paid.any() and not tracker.scrubbed.any()
        assert tracker.peak.eq(-.5).all() and tracker.best_error.isinf().all()
    # Clean pair credit cannot be replaced by physical scraped coverage.
    tracker,ledger,_,_,_=trajectory(2)
    ledger.covered[0,3,0]=True;ledger.rear_pair_paid[0,0]=True
    tracker.scrubbed[0,3,0]=True
    assert tracker.credit(ledger,torch.tensor([True]),torch.tensor([4]))['rear_paired'].item()==0
    tracker.scrubbed[0,3,0]=False;tracker.paid[0,3,0]=True;tracker.score[0,3,0]=1.
    assert tracker.credit(ledger,torch.tensor([True]),torch.tensor([4]))['rear_paired'].item()>0
    assert tracker.credit(ledger,torch.tensor([True]),torch.tensor([4]))['rear_paired'].item()==0
    # No upward target increase beyond the clearance objective; once crossed,
    # the horizontal objective stays active while landing back on the tread.
    bottom=torch.tensor([[.10,.15,.17,.19,.30]])
    factor=q.forward_factor(bottom,torch.full_like(bottom,.15),torch.full_like(bottom,-.05),torch.ones_like(bottom,dtype=torch.bool))
    assert factor[0,0]==factor[0,1]==0 and 0<factor[0,2]<1 and factor[0,3]==factor[0,4]==1
    assert q.forward_factor(torch.tensor([[.15]]),torch.tensor([[.15]]),torch.tensor([[.08]]),torch.tensor([[True]])).item()==1
    return cases


def migration():
    run=ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher_v1_stage2/2026-10-29_12-31-27_ascent_entry_v1_stage11_rear_worst_v1'
    prior=json.loads((run/'params/launch_manifest.json').read_text())
    evidence=m.loader.load_module(m.BASE+'path.stair_path_evidence')
    layout=m.loader.load_module(m.BASE+'path.stair_path_layout')
    height=m.loader.load_module(m.BASE+'path.stair_path_height_practice')
    target=dict(prior['stair_path_reward_settings'],version=p.VERSION,ascent_pairing=deepcopy(p.SETTINGS),
        course_evidence=deepcopy(evidence.TRANSITION_SETTINGS),narrow_ascent_practice=deepcopy(layout.NARROW_ASCENT_SETTINGS),
        ascent_height_practice=deepcopy(height.SETTINGS),**{k:prior[k] for k in (
            'stair_path_reward_version','stair_path_config','control_profile','descent_limits_action_units','descent_authority_schedule')})
    weights=dict(prior['reward_weights'],ascent_rear_paired=p.REAR_PAIR_WEIGHT,
        ascent_rear_landing_credit=q.SETTINGS['rear_landing_credit_weight'])
    validate=m.loader.load_module('algorithms.initialization.stair_path_init').validate_ascent_pairing_resume
    validate(prior,target,weights,migrate=True,source='v11')
    for key in ('joint_mirror','stair_path_slip','ascent_unified_hold_mate'):
        bad=dict(weights);bad[key]=bad[key]+.1
        try:validate(prior,target,bad,migrate=True,source='v11')
        except ValueError:pass
        else:raise AssertionError('unapproved weight change: '+key)
    try:validate(prior,target,weights)
    except ValueError:pass
    else:raise AssertionError('v11 migration accepted without explicit flag')
    native=dict(prior,refinement_version=p.VERSION,reward_weights=weights,ascent_pairing=deepcopy(p.SETTINGS))
    validate(native,target,weights)


def observer_parity(baseline):
    # Enable the actual new observer path, including sparse and explicit reset.
    rng=torch.get_rng_state()
    env,observer=m.perf.fixture(m.loader,56,progressive=True,pairing=True,rear_entry=True,landing_quality=True)
    source=m.perf.load_source(baseline,'_m20_before_quality')
    torch.set_rng_state(rng)
    old_env,old=m.perf.fixture(source,56,progressive=True,pairing=True,rear_entry=True)
    masks=m.loader.load_module(m.BASE+'ascent.ascent_adaptive')
    comparisons=0
    for tick in range(1,65):
        rng=torch.get_rng_state();m.perf.step(env,observer,tick)
        torch.set_rng_state(rng);m.perf.step(old_env,old,tick)
        down=masks.descent_mask(env)
        after=m.perf.tensors(observer)
        for key,value in m.perf.tensors(old).items():
            if value.ndim and value.shape[0]==56:
                assert torch.equal(value[down],after[key][down]),(tick,key)
                comparisons+=1
        for name in ('ascent_adaptive_mdp','ascent_alternating_mdp','ascent_unified_mdp'):
            left=source.load_module(m.BASE+'ascent.'+name);right=m.loader.load_module(m.BASE+'ascent.'+name)
            for term in getattr(right,'NAMES',getattr(right,'EXTRA_NAMES',())):
                torch.testing.assert_close(left.reward(old_env,term)[down],right.reward(env,term)[down],rtol=0,atol=0)
        for term in ('clearance','rear_paired','rear_landing_credit','landing_order','rear_phase'):
            assert not p.reward(env,term)[down].any()
        snapshot=observer.landing_quality.near_s.clone();observer.observe()
        assert torch.equal(snapshot,observer.landing_quality.near_s)
    return comparisons


def observer_landings(scrub=False):
    env,observer=m.perf.fixture(m.loader,28,progressive=True,pairing=True,rear_entry=True,landing_quality=True)
    m.perf.step(env,observer,1)
    data=env.scene['robot'].data
    offset=torch.full((28,4),-.15);bottom=torch.zeros(28,4)
    supported=torch.ones(28,4,dtype=torch.bool)
    tick=1
    ordered=rear_pairs=rear_credit=0.
    def frames(count,face_wheel=None):
        nonlocal tick,ordered,rear_pairs,rear_credit
        for _ in range(count):
            tick+=1;env._sim_step_counter=tick;env.common_step_counter=tick
            env.episode_length_buf+=1
            data.body_pos_w=torch.stack((offset+1.25,torch.zeros_like(offset),bottom+.09),-1)
            data.body_com_pos_w=data.body_pos_w
            data.body_com_lin_vel_w=torch.zeros(28,4,3)
            for wheel,leg in enumerate(('fl','fr','hl','hr')):
                sensor=env.scene.sensors[f'entry_{leg}_contact'].data
                force=torch.zeros(28,1,2,3);force[:,0,0,2]=supported[:,wheel]*70.
                if face_wheel==wheel:force[0,0,0]=torch.tensor([25.,0.,0.])
                point=data.body_pos_w[:,wheel].clone();point[:,2]=bottom[:,wheel]
                sensor.force_matrix_w=force;sensor.contact_pos_w=point[:,None,None,:].expand(-1,1,2,-1)
            observer.observe()
            ordered+=observer.window['adaptive_ordered'][0].item()
            rear_pairs+=observer.window['ascent_pairing_rear_paired'][0].item()
            rear_credit+=observer.window['ascent_pairing_rear_landing_credit'][0].item()
    frames(16)
    for axle in (0,1):
        for _ in range(2):
            first=torch.zeros(28)+1.25;depth=torch.tensor([.20+.05*(c%4) for c in range(28)])
            state=p.paired_targets(observer.sequence,observer.episode_goals,torch.ones(28,dtype=torch.bool))
            permission=p.swing_permissions(state,observer.turns,observer.support,observer.stable_time,
                data.body_pos_w,first,depth)
            choices=permission[0,2*axle:2*axle+2].nonzero().flatten()
            assert len(choices)==1,(axle,permission[0],observer.turns.armed[0])
            wheel=2*axle+int(choices[0])
            supported[0,wheel]=False;offset[0,wheel]=-.08;bottom[0,wheel]=.15
            frames(6)
            if scrub and wheel==3:
                offset[0,wheel]=-.03;bottom[0,wheel]=.05;frames(1,face_wheel=wheel)
            offset[0,wheel]=.02;bottom[0,wheel]=.14;frames(4)
            offset[0,wheel]=.08;bottom[0,wheel]=.10;supported[0,wheel]=True;frames(12)
    assert not observer.sequence.failed[0]
    assert observer.sequence.covered[0,:,0].all()
    if scrub:
        assert observer.landing_quality.scrubbed[0,3,0]
        assert not observer.landing_quality.paid[0,3,0] and rear_pairs==0
    else:
        assert observer.landing_quality.paid[0,:,0].all()
        assert ordered>0 and rear_pairs>0 and rear_credit>0
    alternating=m.loader.load_module(m.BASE+'ascent.ascent_alternating_mdp')
    crossing=alternating.positive_reward(env,lambda env:torch.ones(28),{},require_clean_landing=True)
    assert crossing[0].item()==(0. if scrub else 1.)
    course=m.loader.load_module(m.BASE+'path.stair_path_course_diagnostics').CourseDiagnostics('cpu')
    course.observe(env,torch.tensor([0,1]),0)
    logs={};writer=NS(add_scalar=lambda k,v,i:logs.update({k:v}))
    with TemporaryDirectory() as directory:
        course.write(writer,directory,99,1)
        assert logs['Course/practice/quality_HR_tread1_clean_landing_rate']==(0. if scrub else 1.)
        assert logs['Course/practice/quality_HR_tread1_scrub_rate']==(1. if scrub else 0.)
        assert logs['Course/practice/quality_HR_tread1_permission_fraction']>0
        assert abs(logs['Course/practice/quality_HR_tread1_support_force_mean_n']-70.) < 1.e-4
    return dict(ordered=ordered,rear_pair=rear_pairs,rear_credit=rear_credit)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline-root',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(42)
    cases=counterexamples();migration();compared=observer_parity(args.baseline_root)
    clean=observer_landings();scrubbed=observer_landings(scrub=True)
    print(json.dumps(dict(counterexamples=cases,descent_tensor_comparisons=compared,
        clean_landings_pay_once=True,scraped_or_sliding_landings_pay_zero=True,
        physical_coverage_retained=True,rear_credit_added=True,explicit_v11_manifest_migration=True,
        descent_reward_outputs_identical=True,clean_observer_rewards=clean,scrubbed_observer_rewards=scrubbed,
        policy_checkpoints_loaded=0,physics_simulation=False),indent=2))
