#!/usr/bin/env python3
"""Synthetic mastered-floor regression cases; no policy checkpoint or sim."""
import importlib.util
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('mastery_loader',ROOT/'source/rl_training/rl_training/utils/module_loader.py')
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
course=loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_course')


def main():
    torch.set_num_threads(1)
    ids=torch.tensor([11]);yes=torch.tensor([True]);no=~yes;row=torch.tensor([1]);ref=no
    s=course.ProgressiveCourse(40,'cpu')
    s.skill[ids]=s.prefixes.index(6)
    s.window[ids]=1;s.trials[ids]=8
    up,down=s.update(ids,yes,yes,row,ref)
    assert s.goals()[ids]==8 and s.mastered[ids,1]==s.prefixes.index(6)
    assert s.mastered[ids,0]==s.prefixes.index(6) and s.mastered[ids,2]==-1
    #12 keeps failing: retreat12->10->8->6 on SAME HEIGHT.6 is never lost.
    s.skill[ids]=s.prefixes.index(12)
    for target in (10,8,6):
        for attempt in range(3):
            up,down=s.update(ids,no,yes,row,ref)
            assert not down.any()
        assert s.goals()[ids]==target
    # Reaching the floor starts a NEW sustained-failure observation.
    for _ in range(11):
        up,down=s.update(ids,no,yes,row,ref)
        assert not down.any() and s.goals()[ids]==6
    up,down=s.update(ids,no,yes,row,ref)
    assert down.all() and s.goals()[ids]==6
    assert s.mastered[ids,1]==s.prefixes.index(6) # row1 evidence retained
    row[:]=0
    for _ in range(100):
        up,down=s.update(ids,no,yes,row,ref)
        assert not down.any() and s.goals()[ids]==6
    assert s.bad[ids]==12
    # Ineligible episodes and evaluation must not change failure/mastery.
    saved=s.state_dict()
    for _ in range(10):s.update(ids,no,no,row,ref)
    assert torch.equal(s.bad,saved['bad'])
    s.update(ids,yes,yes,row,ref,training=False)
    assert all(torch.equal(saved[k],s.state_dict()[k]) for k in saved if isinstance(saved[k],torch.Tensor))
    # A newly raised, unproven height is not mistaken for learned6.
    fresh=course.ProgressiveCourse(40,'cpu');fresh.skill[ids]=2;row[:]=2
    for _ in range(5):
        _,down=fresh.update(ids,no,yes,row,ref);assert not down.any()
    _,down=fresh.update(ids,no,yes,row,ref);assert down.all()
    # Restore a known row: do not restart below its learned6 floor.
    s.bad[ids]=0;s.promotion_hold[ids]=0;s.skill[ids]=3;row[:]=0
    s.window[ids]=1;s.trials[ids]=8
    s.geometry_probe[ids]=True
    up,down=s.update(ids,yes,yes,row,ref)
    assert up.all() and s.goals()[ids]==6
    # Changing width/depth invalidates only that environment's floor.
    remembered=s.mastered.clone();sampling=loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_sampling')
    sampling.reset_ascent_progress(s,ids)
    assert (s.mastered[ids]==-1).all() and s.skill[ids]==s.minimum_skill[ids]
    other=torch.arange(40)!=11;assert torch.equal(s.mastered[other],remembered[other])
    # Native state retains learned floors and floor failure streaks.
    clone=course.ProgressiveCourse(40,'cpu');clone.load_state_dict(s.state_dict())
    assert torch.equal(clone.mastered,s.mastered)
    bad=s.state_dict();bad['mastered'][0,0]=99
    try:clone.load_state_dict(bad)
    except ValueError:pass
    else:raise AssertionError('invalid mastered prefix accepted')
    # v5 has no mastery history. Infer only4/6 from its target trajectory,
    # or the current prefix if BOTH recorded speed windows certify it.
    legacy=s.state_dict();legacy.pop('mastered');legacy['event_counts']=legacy['event_counts'][:,:8]
    legacy['curriculum_version']=course.LEGACY_VERSION;legacy['bad'].zero_()
    legacy['skill'][5:10]=torch.tensor([2,3,4,6,8])
    legacy['window'].zero_();legacy['trials'].zero_()
    legacy['window'][9]=1;legacy['trials'][9]=8
    rows=torch.ones(40,dtype=torch.long)
    mapped=clone.migrate_v5(legacy,rows);clone.load_state_dict(mapped)
    assert clone.mastered[5:10,1].tolist()==[-1,2,3,3,8]
    assert (clone.mastered[:,2:]==-1).all()
    assert torch.equal(clone.skill,legacy['skill'])
    for key in ('window','trials','cursor','good','bad','completed','promotion_hold'):
        assert torch.equal(mapped[key],legacy[key]),key
    assert torch.equal(mapped['event_counts'][:,:8],legacy['event_counts'])
    # Migration must not modify its input state or accept an implicit load.
    assert legacy['curriculum_version']==course.LEGACY_VERSION and 'mastered' not in legacy
    try:clone.load_state_dict(legacy)
    except ValueError:pass
    else:raise AssertionError('implicit v5 migration accepted')
    readiness=loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_readiness')
    r=readiness.ReadinessSampler();r.request(stage=1,iteration=1000)
    expected=r.probabilities(1400);r.checks=2;r.totals.fill_(1)
    r.reset_evidence()
    assert r.stage==1 and r.probabilities(1400)==expected and not r.totals.any() and r.checks==0
    init=loader.load_module('algorithms.initialization.stair_path_init')
    settings={k:'same' for k in ('stair_path_reward_weights','path_tracking_target','path_velocity_target','contact_slip','merged_entry_slip')}
    prior=dict(m20_profile='ascent_entry_v1_stage9',refinement_version=course.LEGACY_VERSION,
        stair_path_reward_settings=settings,stair_path_reward_version='same',stair_path_config={},control_profile='same')
    target=dict(settings,version=course.VERSION,stair_path_reward_version='same',stair_path_config={},control_profile='same')
    init.validate_course_migration(prior,target,source='v5')
    try:init.validate_course_migration(prior,target,source='v4')
    except ValueError:pass
    else:raise AssertionError('wrong migration flag accepted')
    print(json.dumps(dict(learned6_retreat_12_10_8_6=True,floor_never_erased_by_failure=True,
        sustained_floor_failures_can_reduce_height=True,new_height_requires_new_evidence=True,
        lower_height_keeps_learned_prefix=True,geometry_change_clears_only_affected_floor=True,
        native_state_roundtrip=True,explicit_v5_migration=True,legacy_windows_retained=True,
        sampling_ramp_retained=True,policy_checkpoints_loaded=0),indent=2))


if __name__=='__main__':main()
