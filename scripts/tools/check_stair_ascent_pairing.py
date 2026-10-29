#!/usr/bin/env python3
"""CPU ascent gait/counterexample checks; optional frozen-source descent parity.

No policy deserialization, Isaac startup, or full resume test is performed.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('pairing_fixture',ROOT/'scripts/tools/check_stair_path_performance.py')
perf = importlib.util.module_from_spec(spec); spec.loader.exec_module(perf)
loader = perf.load_source(ROOT/'source/rl_training/rl_training','_m20_pairing_checks')
BASE = perf.PREFIX
pair = loader.load_module(BASE+'ascent.ascent_pairing')


class LandingRun:
    def __init__(self,depth=.20,dt=.005,targets=4):
        self.ledger = pair.AlternatingPairLedger(1,'cpu',depth=depth,targets=targets)
        self.level = torch.zeros(1,4,dtype=torch.long)
        self.x = torch.full((1,4),-.15)
        self.front = self.rear = self.style = 0.
        self.dt = dt; self.depth = depth; self.goals = torch.tensor([targets])

    def land(self,wheel,tread,offset=.10):
        self.level[0,wheel] = tread
        self.x[0,wheel] = (tread-1)*self.depth+offset
        for _ in range(round(.06/self.dt)):
            event = self.ledger.update_strict(self.level,torch.ones(1,4,dtype=torch.bool),self.x,
                torch.zeros(1),torch.tensor([True]),self.dt,self.goals)
            self.front += event['paired'].item()
            self.rear += self.ledger.rear_paired_event.item()
            self.style += self.ledger.style_event.item()
        return event


def ledger_checks():
    for depth in (.20,.25,.30,.35):
        for dt in (.005,.01):
            for first_side in (0,1):
                run = LandingRun(depth,dt)
                # Different axle progress is legal; no fixed front/rear gap.
                for tread in range(1,5):
                    for wheel in (first_side,1-first_side):run.land(wheel,tread)
                for tread in range(1,5):
                    for wheel in (3-first_side,2+first_side):run.land(wheel,tread)
                assert not run.ledger.failed.any()
                assert run.ledger.covered.all() and run.front == 4 and run.rear == 4
                # Continued contact pays no second event.
                run.land(3,4)
                assert run.front == run.rear == 4
                run.ledger.reset(torch.tensor([0]))
                assert run.ledger.last_landing_side.eq(-1).all()
    for wheels in ((0,1,1),(2,3,3)):
        run = LandingRun()
        run.land(wheels[0],1);run.land(wheels[1],1)
        event = run.land(wheels[2],2)
        assert not run.ledger.failed.item() and run.ledger.failure_reason.item() == 0
        assert run.ledger.style_failed.item() and run.style == 1
        assert not run.ledger.style_reward_allowed.item()
        assert event['paired'].item() == 0 and run.ledger.rear_paired_event.item() == 0
        # Finish the reversed pair; it never gets pair credit. Correct order
        # on the next pair recovers credit using the ORIGINAL first leader.
        run.land(wheels[0],2)
        assert run.front + run.rear == 1 and run.style == 1
        run.land(wheels[0],3);run.land(wheels[1],3)
        assert run.front + run.rear == 2 and run.ledger.style_reward_allowed.item()
        assert not run.ledger.failed.item() and run.ledger.style_failed.item()
        assert run.ledger.first_style_tread.max().item() == 2
    run = LandingRun();run.land(0,1);run.land(0,2)
    assert run.ledger.failed.item()  # advancing before the axle mate
    run = LandingRun();run.land(2,2,offset=.05)
    assert run.ledger.failed.item() and run.ledger.first_fault_jumped[0,2]
    assert run.ledger.first_fault_treads[0,2] == 1
    run = LandingRun()
    run.level[0,:2] = 1;run.x[0,:2] = .10
    run.land(0,1)
    assert run.ledger.failed.item()  # simultaneous same-tread front support


def guidance_checks():
    turns_module = loader.load_module(BASE+'ascent.ascent_alternating')
    for depth,offset in ((.20,.10),(.25,.15),(.30,.10),(.35,.15)):
        ledger = pair.AlternatingPairLedger(1,'cpu',depth=depth,targets=4)
        turns = turns_module.SwingAlternation(1,'cpu')
        turns.last_side[:] = 0;turns.seen_support.fill_(True);turns.armed.fill_(True)
        turns.armed[:,[0,2]] = False  # legal in-flight leaders
        position = torch.tensor([[[offset,0.,.19]]]).repeat(1,4,1)
        support = torch.tensor([[False,True,False,True]])
        zeros = torch.zeros(1);d=torch.tensor([depth]);h=torch.tensor([.10])
        guide,hold,clearance,state = pair.pairing_guidance(ledger,turns,support,
            torch.ones(1,4)*.1,position,torch.zeros(1,4),zeros,zeros,d,h,
            torch.tensor([4]),torch.tensor([True]))
        assert state['target'].eq(1).all() and clearance.item() == 0.
        assert guide['height'].item() < 1.e-8 and hold.item() == 0.
        # Before the edge, only the legal side receives preparation guidance.
        position[...,0] = -.05;position[...,2] = .09
        _,_,clearance,state = pair.pairing_guidance(ledger,turns,support,
            torch.ones(1,4)*.1,position,torch.zeros(1,4),zeros,zeros,d,h,
            torch.tensor([4]),torch.tensor([True]))
        assert state['permission'].tolist() == [[True,False,True,False]]
        assert (clearance.item() > 0) == (depth <= .25)
        # Stable leader must stay on1 while mate approaches1, never prepare2.
        ledger.covered[:,[0,2],0] = True
        state = pair.paired_targets(ledger,torch.tensor([4]),torch.tensor([True]))
        assert state['target'].eq(1).all() and state['hold'].tolist() == [[True,False,True,False]]
        permission = pair.swing_permissions(state,turns,torch.ones(1,4,dtype=torch.bool),
            torch.ones(1,4)*.1,position,zeros,d)
        assert not permission[:,[0,2]].any()
        # A full-flight task is done, not wrapped around to tread1.
        ledger.covered.fill_(True)
        assert not pair.paired_targets(ledger,torch.tensor([4]),torch.tensor([True]))['enabled'].any()
    # First side free to validate, mirrored starts encouraged across envs.
    ledger = pair.AlternatingPairLedger(2,'cpu',targets=4)
    turns = turns_module.SwingAlternation(2,'cpu');turns.armed.fill_(True);turns.seen_support.fill_(True)
    position = torch.zeros(2,4,3)
    state=pair.paired_targets(ledger,torch.tensor([4,4]),torch.tensor([True,True]))
    permission=pair.swing_permissions(state,turns,torch.ones(2,4,dtype=torch.bool),
        torch.ones(2,4)*.1,position,torch.zeros(2),torch.ones(2)*.2)
    assert permission.reshape(2,2,2).sum(-1).eq(1).all()
    assert not torch.equal(permission[0],permission[1])
    # The v10 transition guard holds both axles briefly after tread2 becomes
    # stable, then releases the next target without changing physical ledger
    # coverage.
    locked = pair.AlternatingPairLedger(1,'cpu',targets=6)
    locked.covered[:, :, :2] = True
    locked.landing_time[:, :, 1] = .40
    locked.time.fill_(.45)
    state = pair.paired_targets(locked, torch.tensor([4]), torch.tensor([True]))
    assert state['target'].eq(3).all() and state['transition_hold'].all()
    assert state['hold_target'].eq(2).all()
    locked.time.fill_(.55)
    assert not pair.paired_targets(locked, torch.tensor([4]), torch.tensor([True]))['transition_hold'].any()


def bridge_checks():
    turns_module=loader.load_module(BASE+'ascent.ascent_alternating')
    for depth in (.20,.25,.30,.35):
        ledger=pair.AlternatingPairLedger(1,'cpu',depth=depth,targets=4)
        turns=turns_module.SwingAlternation(1,'cpu')
        support=torch.ones(1,4,dtype=torch.bool)
        z=torch.full((1,4),.09);vz=torch.zeros_like(z)
        active=torch.tensor([True]);near=torch.ones(1,2,dtype=torch.bool)
        for _ in range(10):turns.update(support,z,vz,near,active,.005)
        position=torch.zeros(1,4,3);position[...,0]=-.05;position[...,2]=.09
        def measure():
            return pair.pairing_guidance(ledger,turns,support,torch.where(support,.1,0.),
                position,vz,torch.zeros(1),torch.zeros(1),torch.tensor([depth]),
                torch.tensor([.1]),torch.tensor([4]),active)
        initial=measure();chosen=initial[3]['permission'].nonzero()[:,1]
        support[:,chosen]=False
        # Actual state machine: contact loss, unconfirmed air, then real lift.
        values=[]
        for tick in range(5):
            z[:,chosen]=.12;vz[:,chosen]=.3
            turns.update(support,z,vz,near,active,.005)
            current=measure();values.append(current)
            assert current[3]['permission'][:,chosen].all()
            assert current[0]['horizontal'].item() > 0
        assert torch.allclose(initial[0]['horizontal'],values[0][0]['horizontal'])
        if depth <= .25:
            assert initial[2].item() > 0
            assert all(torch.equal(initial[2],v[2]) for v in values)


def course_style_checks():
    module=loader.load_module(BASE+'path.stair_path_course')
    course=module.ProgressiveCourse(40,'cpu')
    ids=torch.tensor([11]);yes=torch.tensor([True]);no=~yes;row=torch.tensor([1])
    course.skill[ids]=course.prefixes.index(6)
    course.mastered[ids,:2]=course.prefixes.index(6)
    remembered=course.mastered.clone()
    for _ in range(40):
        up,down=course.update(ids,yes,yes,row,no,promotion_success=no)
        assert not up.any() and not down.any() and course.goals()[ids] == 6
        assert course.bad[ids].item() == 0
    assert torch.equal(course.mastered,remembered)
    # Strict passes from both speed buckets promote, same as the old evidence threshold.
    for bucket in (0,1):
        course.speed_bucket[ids]=bucket
        for _ in range(8):course.update(ids,yes,yes,row,no,promotion_success=yes)
    assert course.goals()[ids] == 8
    # Real failure still shortens before lowering height, and respects mastered6.
    for _ in range(3):
        up,down=course.update(ids,no,yes,row,no,promotion_success=no)
        assert not down.any()
    assert course.goals()[ids] == 6
    for index in range(12):
        _,down=course.update(ids,no,yes,row,no,promotion_success=no)
        assert down.item() == (index == 11)
    # New semantics clear only old promotion evidence, not physical mastery or assignments.
    saved=course.state_dict();saved['window'].fill_(1);saved['trials'].fill_(8)
    migrated=course.fresh_promotion_evidence(saved)
    for field in ('window','cursor','trials','good'):assert not migrated[field].any()
    for field in ('mastered','skill','bad','speed_bucket','height_age','event_counts'):
        assert torch.equal(saved[field],migrated[field])
    assert saved['window'].all()  # migration cannot mutate source tensors


def swing_checks():
    turns_module = loader.load_module(BASE+'ascent.ascent_alternating')
    for wheels in ((0,1,0,1),(1,0,1,0),(2,3,2,3),(3,2,3,2)):
        turns=turns_module.SwingAlternation(1,'cpu')
        z=torch.ones(1,4)*.2;vz=torch.zeros_like(z);support=torch.ones(1,4,dtype=torch.bool)
        for wheel in wheels:
            for _ in range(10):turns.update(support,z,vz,torch.ones(1,2,dtype=torch.bool),torch.tensor([True]),.005)
            support[0,wheel]=False;z[0,wheel]+=.03
            for _ in range(5):turns.update(support,z,vz,torch.ones(1,2,dtype=torch.bool),torch.tensor([True]),.005)
            support[0,wheel]=True
        assert not turns.failed.any()
    turns=turns_module.SwingAlternation(1,'cpu')
    z=torch.zeros(1,4);support=torch.ones(1,4,dtype=torch.bool)
    for _ in range(10):turns.update(support,z,z,torch.ones(1,2,dtype=torch.bool),torch.tensor([True]),.005)
    support[:,2:]=False;z[:,2:]=.03
    for _ in range(5):turns.update(support,z,torch.zeros_like(z),torch.ones(1,2,dtype=torch.bool),torch.tensor([True]),.005)
    assert turns.failed.any()


def migration_checks():
    init=loader.load_module('algorithms.initialization.stair_path_init')
    evidence=loader.load_module(BASE+'path.stair_path_evidence')
    settings=dict(stair_path_reward_weights={'slip':-.4,'down_complete':2.},
        path_tracking_target='effective',path_velocity_target='return',contact_slip='support',merged_entry_slip='max')
    prior=dict(m20_profile='ascent_entry_v1_stage9',refinement_version=pair.V9_VERSION,
        ascent_pairing=deepcopy(pair.V9_SETTINGS),course_evidence=deepcopy(evidence.SETTINGS),stair_path_reward_settings=settings,
        stair_path_reward_version='same',stair_path_config={'gain':.6},control_profile='same',
        descent_limits_action_units=[.18],descent_authority_schedule={'ramp':800},
        reward_weights={'ascent_entry_clearance':-.3,'heading':-.45,'front_slip':-.65,
            'rear_slip':-.55,'stair_path_slip':-.4,'down_complete':2.,'ascent_rear_paired':.75,'ascent_landing_order':-.5,'ascent_rear_phase':-.15})
    target=dict(settings,version=pair.VERSION,ascent_pairing=deepcopy(pair.SETTINGS),course_evidence=deepcopy(evidence.TRANSITION_SETTINGS),
        narrow_ascent_practice=deepcopy(loader.load_module(BASE+'path.stair_path_layout').NARROW_ASCENT_SETTINGS),
        ascent_height_practice=deepcopy(loader.load_module(BASE+'path.stair_path_height_practice').SETTINGS),
        **{k:prior[k] for k in ('stair_path_reward_version','stair_path_config','control_profile',
                              'descent_limits_action_units','descent_authority_schedule')})
    weights=dict(prior['reward_weights'],ascent_rear_paired=pair.REAR_PAIR_WEIGHT,
        ascent_landing_order=pair.LANDING_ORDER_WEIGHT,ascent_rear_phase=pair.REAR_PHASE_WEIGHT,
        ascent_rear_landing_credit=pair.LANDING_QUALITY_SETTINGS['rear_landing_credit_weight'])
    init.validate_ascent_pairing_resume(prior,target,weights,migrate=True,source='v9')
    for field in ('heading','rear_slip','stair_path_slip','down_complete','ascent_landing_order'):
        bad=dict(weights);bad[field]*=.5
        try:init.validate_ascent_pairing_resume(prior,target,bad,migrate=True)
        except ValueError:pass
        else:raise AssertionError('protected reward changed: '+field)
    for field,value in (('stair_path_config',{'gain':.7}),('descent_authority_schedule',{'ramp':1000})):
        try:init.validate_ascent_pairing_resume(prior,dict(target,**{field:value}),weights,migrate=True)
        except ValueError:pass
        else:raise AssertionError('protected control contract changed')
    native=dict(prior,refinement_version=pair.VERSION,reward_weights=weights,
                narrow_ascent_practice=deepcopy(target['narrow_ascent_practice']),
                ascent_height_practice=deepcopy(target['ascent_height_practice']),
                ascent_pairing=deepcopy(pair.SETTINGS),course_evidence=deepcopy(evidence.TRANSITION_SETTINGS))
    init.validate_ascent_pairing_resume(native,target,weights)
    # The first v10 checkpoint predates the transition-hold manifest field;
    # it must remain resumable into the hold-enabled v10 target.
    old_native=dict(native,refinement_version=pair.V10_VERSION,ascent_pairing=deepcopy(pair.PREVIOUS_V10_SETTINGS))
    old_native['ascent_pairing']['rear_entry_transition'].pop('stable_transition_hold_s',None)
    init.validate_ascent_pairing_resume(old_native,target,weights,migrate=True,source='v10')
    for source in ('v5','v6','v8'):
        try:init.validate_ascent_pairing_resume(prior,target,weights,migrate=True,source=source)
        except ValueError:pass
        else:raise AssertionError('wrong migration accepted')
    try:init.validate_ascent_pairing_resume(prior,target,weights)
    except ValueError:pass
    else:raise AssertionError('missing migration flag accepted')


def diagnostics_checks():
    n=3;ledger=pair.AlternatingPairLedger(n,'cpu',depth=.2,targets=18)
    diag=pair.PairingDiagnostics(n,'cpu');goals=torch.full((n,),2)
    position=torch.zeros(n,4,3);position[...,0]=-.10
    position[1,2:,0]=-.60  # front fails before rear gets an opportunity
    depth=torch.full((n,),.20);first=torch.zeros(n);active=torch.ones(n,dtype=torch.bool)
    state=pair.paired_targets(ledger,goals,active)
    state['offset']=position[...,0]-first[:,None]-(state['target']-1)*depth[:,None]
    diag.observe(state,ledger,position,first,depth,torch.zeros(n),.005)
    ledger.failed[1:]=True
    state=pair.paired_targets(ledger,goals,active)
    state['offset']=position[...,0]-first[:,None]-(state['target']-1)*depth[:,None]
    # Env2 approached but was terminated before its40ms observation window.
    diag.observe(state,ledger,position,first,depth,torch.tensor([.06,.02,.02]),.005)
    ledger.covered[0,2:,0]=True
    observer=NS(sequence=ledger,episode_goals=goals,failure_reason=torch.tensor([0,1,1]),
        pairing_diagnostics=diag,episode_terrain_rows=torch.zeros(n,dtype=torch.long),
        prefix_ready=torch.zeros(n,dtype=torch.bool),top_ready=torch.zeros(n,dtype=torch.bool),
        path_forward_max=torch.zeros(n),path_max=torch.zeros(n),path_heading_total=torch.zeros(n),path_samples=torch.ones(n))
    values=diag.values(observer,torch.arange(n))
    field={name:i for i,name in enumerate(pair.DIAGNOSTIC_FIELDS)}
    assert values[0,field['rear_covered_mature_attempts']]==2
    assert values[1,field['rear_no_opportunity_at_failure']]==2
    assert values[2,field['rear_truncated_attempts']]==2
    course=loader.load_module(BASE+'path.stair_path_course')
    mdp=loader.load_module(BASE+'path.stair_path_course_diagnostics')
    scene=perf.Scene();scene.terrain=NS(terrain_types=torch.ones(n,dtype=torch.long))
    env=NS(_ascent_entry_observer=observer,_ascent_adaptive_curriculum=course.ProgressiveCourse(n,'cpu'),
        scene=scene,command_manager=NS(get_command=lambda _:torch.tensor([[.5,0.,0.]]).expand(n,-1)))
    aggregate=mdp.CourseDiagnostics('cpu');aggregate.observe(env,torch.arange(n),12000)
    logs={};writer=NS(add_scalar=lambda k,v,i:logs.update({k:v}))
    with TemporaryDirectory() as directory:
        aggregate.write(writer,directory,12099,1)
        record=json.loads((Path(directory)/'course_strata.jsonl').read_text())
        assert record['version']==pair.VERSION
        assert logs['Course/rear_opportunity_coverage']==.5
        assert logs['Course/rear_mature_opportunity_coverage']==1.
    diag.reset(torch.tensor([0]))
    assert not torch.isfinite(diag.started[0]).any() and not diag.mature[0].any()


def observer_checks(baseline_root=None):
    fixture_rng=torch.get_rng_state()
    env,observer=perf.fixture(loader,56,progressive=True,pairing=True)
    baseline=None
    if baseline_root:
        old=perf.load_source(baseline_root,'_m20_pairing_frozen_v6')
        torch.set_rng_state(fixture_rng)
        old_env,old_observer=perf.fixture(old,56,progressive=True,pairing=True)
        baseline=(old,old_env,old_observer)
    modules = {name:loader.load_module(BASE+'ascent.'+name) for name in (
        'ascent_entry_mdp','ascent_adaptive_mdp','ascent_alternating_mdp','ascent_unified_mdp')}
    path=loader.load_module(BASE+'path.stair_path_mdp')
    masks=loader.load_module(BASE+'ascent.ascent_adaptive')
    comparisons=0
    for tick in range(1,65):
        step_rng=torch.get_rng_state()
        perf.step(env,observer,tick)
        down=masks.descent_mask(env)
        assert pair.reward(env,'rear_paired')[down].eq(0).all()
        assert pair.reward(env,'clearance')[down].eq(0).all()
        assert pair.reward(env,'landing_order')[down].eq(0).all()
        assert pair.reward(env,'rear_phase')[down].eq(0).all()
        if baseline:
            old,old_env,old_observer=baseline
            torch.set_rng_state(step_rng)
            perf.step(old_env,old_observer,tick)
            before=perf.tensors(old_observer);after=perf.tensors(observer)
            for key,value in before.items():
                if value.ndim and value.shape[0] == env.num_envs:
                    current_key=key.replace('ascent_v7_','ascent_pairing_')
                    assert torch.equal(value[down],after[current_key][down]),f'descent state changed: {tick}:{key}'
                    comparisons+=1
            for name,module in modules.items():
                prior=old.load_module(BASE+'ascent.'+name)
                if name in ('ascent_entry_mdp','ascent_adaptive_mdp'):
                    current=module.context(env);previous=prior.context(old_env)
                    for key,value in previous.items():
                        assert torch.equal(value[down],current[key][down]),f'descent reward changed: {name}.{key}'
                else:
                    names=module.EXTRA_NAMES if name=='ascent_alternating_mdp' else module.NAMES
                    for key in names:
                        assert torch.equal(module.reward(env,key)[down],prior.reward(old_env,key)[down])
            old_path=old.load_module(BASE+'path.stair_path_mdp')
            for key in ('position','velocity','slip','down_stall','down_complete'):
                assert torch.equal(path.reward(env,key)[down],old_path.reward(old_env,key)[down])
    # A late failure must also erase earlier substep credit at reward reading.
    observer.window['ascent_pairing_rear_paired']=torch.ones(56)
    observer.positive_gate.zero_()
    assert pair.reward(env,'rear_paired').eq(0).all()
    observer.positive_gate.fill_(True);observer.sequence.failed.zero_();observer.path_failed.fill_(True)
    assert pair.reward(env,'rear_paired').eq(0).all()
    assert torch.isfinite(observer.pairing_diagnostics.values(observer,torch.arange(56))).all()
    return comparisons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    ledger_checks();guidance_checks();bridge_checks();course_style_checks();swing_checks();migration_checks();diagnostics_checks()
    comparisons=observer_checks(args.baseline_root)
    print(json.dumps(dict(version=pair.VERSION,legal_mirrored_axle_sequences=True,
        reversed_landing_soft_penalty_and_local_credit_gate=True,correct_next_pair_recovers=True,
        physical_success_vs_strict_promotion=True,physical_demotion_retained=True,prelift_guidance_continuous=True,skip_early_simultaneous_rejected=True,
        current_tread_target_and_hold=True,legal_preparation_only=True,rear_pairs_paid_once=True,
        protected_reward_migration=True,late_failure_clears_credit=True,
        opportunity_censoring_and_matched_logging=True,
        frozen_descent_tensor_comparisons=comparisons,policy_checkpoints_loaded=0),indent=2))


if __name__=='__main__':main()
