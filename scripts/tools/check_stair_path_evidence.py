#!/usr/bin/env python3
"""CPU v9 curriculum and rear-phase counterexamples; no policy or full resume."""
import importlib.util
from pathlib import Path
import json
from copy import deepcopy
import torch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('evidence_checks_loader',ROOT/'scripts/tools/check_stair_ascent_pairing.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
e=m.loader.load_module(m.BASE+'path.stair_path_evidence')
p=m.pair
ids=torch.tensor([11]);yes=torch.tensor([True]);no=~yes;rows=torch.tensor([1])


def state(prefix=6):
    s=e.EvidenceCourse(40,'cpu');s.skill[ids]=s.prefixes.index(prefix);s.geometry_probe[ids]=False
    s.mastered[ids,:2]=s.prefixes.index(6)
    return s


def feed(s,success=True,quality=(1.,1.),second=1.,bucket=0):
    s.speed_bucket[ids]=bucket
    return s.update(ids,torch.tensor([success]),yes,rows,no,
        pair_quality=torch.tensor([quality]),rear_second_quality=torch.tensor([second]))


def fill(s,quality=(1.,1.),second=1.):
    for bucket in (0,1):
        for success in (False,False,True,True,True,True,True,True):
            result=feed(s,success,quality,second,bucket)
    return result


def courses():
    # A single bad pair does not erase all six treads of course evidence.
    s=state();up,down=fill(s,(1.,5/6),0.)
    assert s.goals()[ids] == 8 and not up.any() and not down.any()
    assert s.mastered[ids,1] == s.prefixes.index(6)
    # Missing the other speed, or completely reversed rear sequences, cannot promote.
    s=state()
    for _ in range(40):feed(s,quality=(1.,5/6))
    assert s.goals()[ids] == 6
    s=state();fill(s,(1.,1/6),0.)
    assert s.goals()[ids] == 6
    assert s.mastered[ids,1] == s.prefixes.index(6)  # physical ability protected
    # Height requires better quality plus repeated rear-second correctness.
    s=state();s.geometry_probe[ids]=True
    up,_=fill(s,(1.,5/6),0.);assert not up.any() and s.goals()[ids] == 8
    s=state();s.geometry_probe[ids]=True
    up,_=fill(s,(1.,1.),0.);assert not up.any() and s.goals()[ids] == 8
    s=state();s.geometry_probe[ids]=True
    up,_=fill(s,(1.,1.),1.);assert up.all() and s.goals()[ids] == 4
    # Reference never raises height, even with perfect quality.
    s=state();s.geometry_probe[ids]=True
    for bucket in (0,1):
        for _ in range(8):
            s.speed_bucket[ids]=bucket
            up,_=s.update(ids,yes,yes,rows,yes,pair_quality=torch.ones(1,2),rear_second_quality=torch.ones(1))
            assert not up.any()
    # An occasional three-loss run after eight passes must not shorten10.
    s=state(10);s.window[ids]=1;s.trials[ids]=8
    for _ in range(3):
        _,down=feed(s,False);assert not down.any() and s.goals()[ids] == 10
    assert s.evidence_last[ids,3].item() == 1
    # Continuing failure lowers the current-speed window to4/8: shorten10->8.
    _,down=feed(s,False);assert s.goals()[ids] == 8 and not down.any()
    # Missing evidence never triggers a retreat merely on an inherited streak.
    s=state(10);s.bad[ids]=12
    for _ in range(7):feed(s,False);assert s.goals()[ids] == 10
    feed(s,False);assert s.goals()[ids] == 8
    # Still failing: same-height8->6; protected6 remains until12 floor failures.
    for _ in range(8):_,down=feed(s,False);assert not down.any()
    assert s.goals()[ids] == 6
    for tick in range(12):
        _,down=feed(s,False)
        assert down.item() == (tick == 11)
    assert s.goals()[ids] == 6 and s.mastered[ids,1] == s.prefixes.index(6)
    # Hold prevents bursts of promotions; failures remain available during hold.
    s=state();fill(s);assert s.promotion_hold[ids] == 16
    before=s.state_dict();s.update(ids,no,yes,rows,no,training=False)
    assert all(torch.equal(v,s.state_dict()[k]) for k,v in before.items() if isinstance(v,torch.Tensor))


def persistence():
    s=state();legacy=m.loader.load_module(m.BASE+'path.stair_path_course').ProgressiveCourse(40,'cpu')
    saved=legacy.state_dict();saved['window'].fill_(1);saved['trials'].fill_(8)
    saved['skill'][ids]=legacy.prefixes.index(10);saved['mastered'][ids,:2]=legacy.prefixes.index(6)
    saved['bad'][ids]=3;saved['promotion_hold'][ids]=12
    mapped=s.migrate_v8(saved)
    assert saved['window'].all() and not mapped['window'].any()
    for field in ('mastered','skill','bad','speed_bucket','height_age','promotion_hold','event_counts'):
        assert torch.equal(saved[field],mapped[field])
    s.load_state_dict(mapped);feed(s,True,(.9,.8),1.)
    clone=state();clone.load_state_dict(s.state_dict())
    assert all(torch.equal(v,clone.state_dict()[k]) for k,v in s.state_dict().items() if isinstance(v,torch.Tensor))
    bad=deepcopy(s.state_dict());bad['quality_window'][0,0,0,0]=float('nan')
    try:clone.load_state_dict(bad)
    except ValueError:pass
    else:raise AssertionError('nonfinite evidence accepted')
    preserved=s.mastered.clone();s.reset_geometry(ids)
    assert (s.mastered[ids] == -1).all() and not s.quality_window[ids].any()
    assert torch.equal(s.mastered[12:],preserved[12:])


def phase():
    turns_mod=m.loader.load_module(m.BASE+'ascent.ascent_alternating')
    for depth in (.20,.25,.30,.35):
        for side in (0,1):
            run=m.LandingRun(depth=depth)
            run.land(2+side,1);run.land(3-side,1)
            ledger=run.ledger;goals=run.goals;active=yes
            turns=turns_mod.SwingAlternation(1,'cpu');turns.seen_support.fill_(True);turns.armed.fill_(True)
            turns.last_side[:,1]=1-side
            pos=torch.zeros(1,4,3);pos[...,0]=.1;pos[...,2]=.19
            support=torch.ones(1,4,dtype=torch.bool);stable=torch.ones(1,4)*.1
            zeros=torch.zeros(1);d=torch.tensor([depth]);h=torch.tensor([.1])
            guide,_,_,st=p.pairing_guidance(ledger,turns,support,stable,pos,torch.zeros(1,4),zeros,zeros,d,h,goals,active)
            assert st['rear_wait'][0,3-side] and not st['rear_wait'][0,2+side]
            base=st['rear_phase'].item()
            # Preparing the correct leader incurs no follower phase penalty.
            pos[0,2+side,2]+=.08
            _,_,_,st=p.pairing_guidance(ledger,turns,support,stable,pos,torch.zeros(1,4),zeros,zeros,d,h,goals,active)
            assert st['rear_phase'].item() == base
            # The follower moving first is penalized before stable landing.
            pos[0,3-side,2]+=.08
            _,_,_,st=p.pairing_guidance(ledger,turns,support,stable,pos,torch.zeros(1,4),zeros,zeros,d,h,goals,active)
            assert base < st['rear_phase'].item() <= 4
            run.land(2+side,2)
            _,_,_,st=p.pairing_guidance(ledger,turns,support,stable,pos,torch.zeros(1,4),zeros,zeros,d,h,goals,active)
            assert not st['rear_wait'].any() and st['rear_phase'].item() == 0
            run.land(3-side,2)
            quality,second=e.pair_quality(ledger,torch.tensor([2]))
            assert quality[0,1] == 1 and second.item() == 1
    # First side remains free; neither initial step nor inactive/down slots wait.
    ledger=p.AlternatingPairLedger(1,'cpu',targets=4)
    st=p.paired_targets(ledger,torch.tensor([4]),yes)
    pos=torch.zeros(1,4,3)
    mask,cost=p.rear_phase_state(ledger,st,pos,torch.zeros(1),torch.zeros(1),torch.ones(1)*.2,torch.ones(1)*.1,yes)
    assert not mask.any() and cost.item() == 0
    ledger.covered[:,2:,:1]=True;ledger.first_side[:,1]=0
    st=p.paired_targets(ledger,torch.tensor([4]),yes)
    mask,cost=p.rear_phase_state(ledger,st,pos,torch.zeros(1),torch.zeros(1),torch.ones(1)*.2,torch.ones(1)*.1,no)
    assert not mask.any() and cost.item() == 0


def integration():
    env,observer=m.perf.fixture(m.loader,56,progressive=True,pairing=True)
    course=env._ascent_adaptive_curriculum
    assert isinstance(course,e.EvidenceCourse)
    index=torch.tensor([0]);course.skill[index]=course.prefixes.index(6)
    course.geometry_probe[index]=False
    course.mastered[index,0]=course.prefixes.index(6)
    observer.episode_goals[index]=6;observer.stats['physics_samples'][index]=1
    observer.prefix_ready[index]=True
    observer.sequence.covered[index,:,:6]=True
    observer.sequence.pair_paid[index,:6]=True;observer.sequence.rear_pair_paid[index,:6]=True
    observer.sequence.style_bad_treads[index,1,1]=True;observer.sequence.style_failed[index]=True
    # Exercise the real environment curriculum caller, not only the pure class.
    calls=[]
    env.scene.terrain.update_env_origins=lambda selected,up,down:calls.append((selected.clone(),up.clone(),down.clone()))
    alternating=m.loader.load_module(m.BASE+'ascent.ascent_alternating_mdp')
    for bucket in (0,1):
        env.command_manager.get_command('base_velocity')[index,0]=.5+.2*bucket
        for _ in range(8):result=alternating.curriculum(env,index)
    assert course.goals()[index] == 8 and len(calls) == 16
    assert all(not (up.any() or down.any()) for _,up,down in calls)
    assert 'at_protected_floor_fraction' in result
    diag=m.loader.load_module(m.BASE+'path.stair_path_course_diagnostics').CourseDiagnostics('cpu')
    diag.observe(env,index,12300)
    assert diag.totals[:,diag.totals.shape[-1]-6].sum() == 1
    assert diag.totals[:,1].sum() == 1  # old episode physical success
    # Validate actual launch metadata without loading any policy file.
    import json
    from copy import deepcopy
    run=ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher_v1_stage2/2026-10-27_13-52-28_ascent_entry_v1_stage9_v8'
    manifest=run/'params/launch_manifest.json'
    if manifest.exists():
        prior=json.loads(manifest.read_text())
        target=dict(prior['stair_path_reward_settings'],version=p.VERSION,
            ascent_pairing=deepcopy(p.SETTINGS),course_evidence=deepcopy(e.SETTINGS),
            **{k:prior[k] for k in ('stair_path_reward_version','stair_path_config','control_profile',
                                  'descent_limits_action_units','descent_authority_schedule')})
        weights=dict(prior['reward_weights'],ascent_rear_phase=p.REAR_PHASE_WEIGHT)
        init=m.loader.load_module('algorithms.initialization.stair_path_init')
        init.validate_ascent_pairing_resume(prior,target,weights,migrate=True,source='v8')


def main():
    torch.set_num_threads(1);courses();persistence();phase();integration()
    print(json.dumps(dict(version=e.VERSION,graded_prefix_promotion=True,both_speeds_required=True,
        height_requires_quality_and_rear_second=True,isolated_three_losses_do_not_demote=True,
        poor_window_demotes_before_height=True,mastered_floor_preserved=True,
        migration_and_roundtrip=True,real_curriculum_caller_and_course_logging=True,geometry_clears_only_changed_evidence=True,
        mirrored_rear_follower_waits_for_original_leader=True,first_side_free=True,
        no_front_rear_gap=True,policy_checkpoints_loaded=0),indent=2))
if __name__=='__main__':main()
