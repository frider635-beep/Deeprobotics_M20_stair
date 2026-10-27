#!/usr/bin/env python3
"""Synthetic course/migration/diagnostic checks; no policy checkpoint or PPO."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('course_loader',ROOT/'source/rl_training/rl_training/utils/module_loader.py')
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
BASE='tasks.manager_based.locomotion.velocity.mdp.'
course=loader.load_module(BASE+'path.stair_path_course')
old=loader.load_module(BASE+'ascent.ascent_unified')
readiness=loader.load_module(BASE+'path.stair_path_readiness')


def prime(state, ids):
    state.window[ids]=1;state.trials[ids]=8


def main():
    torch.set_num_threads(1)
    n=80;state=course.ProgressiveCourse(n,'cpu');legacy=old.BridgeCurriculum(n,'cpu')
    legacy.skill[5:10]=torch.tensor([0,1,2,3,4])
    legacy.window.fill_(1);legacy.trials.fill_(7);legacy.cursor.fill_(3)
    legacy.good.fill_(2);legacy.completed.fill_(12)
    before=legacy.state_dict();adapted=state.migrate_bridge(before);state.load_state_dict(adapted)
    assert torch.equal(state.goals(),legacy.goals())
    for key,value in before.items():
        if key!='skill':assert torch.equal(state.state_dict()[key],value),key
    expected=torch.tensor([0.,.25,.5,.75,1.])[legacy.skill]
    assert torch.equal(state.critic_skill(),expected)
    clone=course.ProgressiveCourse(n,'cpu');clone.load_state_dict(state.state_dict())
    assert torch.equal(clone.skill,state.skill)
    # Actual Stage9 observer inheritance must use the expanded goals/deadlines.
    perf_spec=importlib.util.spec_from_file_location('course_observer_fixture',ROOT/'scripts/tools/check_stair_path_performance.py')
    perf=importlib.util.module_from_spec(perf_spec);perf_spec.loader.exec_module(perf)
    env_live,observer=perf.fixture(loader,56,progressive=True)
    assert isinstance(env_live._ascent_adaptive_curriculum,course.ProgressiveCourse)
    env_live._ascent_adaptive_curriculum.skill.fill_(8)
    observer.episode_goals.fill_(18)
    for tick in range(1,25):perf.step(env_live,observer,tick)
    assert torch.isfinite(observer.path_max).all()
    # Metrics must use ORIGINAL goals after live curriculum promotion.
    mdp=loader.load_module(BASE+'ascent.ascent_adaptive_mdp')
    observer.prefix_ready.fill_(True);observer.top_ready.fill_(False)
    env_live._ascent_adaptive_curriculum.skill.fill_(8);observer.episode_goals.fill_(6)
    assert mdp.finished(env_live).all()
    observer.episode_goals.fill_(18)
    assert not mdp.finished(env_live).any()
    try:clone.load_state_dict(before)
    except ValueError:pass
    else:raise AssertionError('implicit legacy migration')
    # Non-probe environments bridge every prefix while preserving depth/width.
    ids=torch.tensor([11]);ok=torch.tensor([True]);no=torch.tensor([False]);row=torch.tensor([0]);ref=torch.tensor([False])
    state.skill[ids]=3
    for goal in (8,10,12,15,18):
        prime(state,ids);up,down=state.update(ids,ok,ok,row,ref)
        assert state.goals()[ids]==goal and not up.any() and not down.any()
    prime(state,ids);up,down=state.update(ids,ok,ok,row,ref)
    assert up.all() and state.goals()[ids]==4 and state.promotion_hold[ids]==24
    row[:]=1
    # Hold delays promotion even when both speed windows already qualify.
    prime(state,ids);up,down=state.update(ids,ok,ok,row,ref)
    assert state.goals()[ids]==4 and state.last_events[ids,5]==1
    # This4-tread task was certified while held. A certified floor needs12
    # consecutive failures for height fallback, and its prefix is retained.
    for _ in range(11):
        up,down=state.update(ids,no,ok,row,ref)
        assert not down.any() and state.goals()[ids]==4
    up,down=state.update(ids,no,ok,row,ref)
    assert down.all() and state.last_events[ids,4]==1
    assert state.height_age[ids]==0 and state.promotion_hold[ids]==24
    state.reset_geometry(ids) # independent unmastered18-tread case
    state.promotion_hold[ids]=0;state.skill[ids]=8;state.bad[ids]=0;row[:]=0
    for _ in range(3):up,down=state.update(ids,no,ok,row,ref)
    assert state.goals()[ids]==15 and not down.any()
    # Reference still never changes physical height, long task remains18.
    state.skill[ids]=8;state.promotion_hold[ids]=0;prime(state,ids)
    up,_=state.update(ids,ok,ok,row,torch.tensor([True]));assert not up.any() and state.goals()[ids]==18
    # Geometry moves clear only moved course timers; telemetry totals persist.
    untouched=state.skill.clone();state.promotion_hold[:]=12
    state.reset_geometry(ids);assert state.promotion_hold[ids]==0 and state.promotion_hold[12]==12
    assert torch.equal(state.skill,untouched)
    # Legacy readiness evidence is reset explicitly; quotas/ramp remain exact.
    r=readiness.ReadinessSampler();r.request(stage=1,iteration=4000)
    saved=r.state_dict();saved.pop('prefixes');saved['totals']=torch.zeros(9*6*5*2,9,dtype=torch.float64)
    migrated=readiness.ReadinessSampler();migrated.load_state_dict(saved,migrate_v4=True)
    assert migrated.stage==r.stage and migrated.probabilities(4500)==r.probabilities(4500)
    assert not migrated.history and migrated.totals.shape==(9*6*9*2,9)
    init=loader.load_module('algorithms.initialization.stair_path_init')
    settings=dict(stair_path_reward_weights={'position':-1.5},path_tracking_target='effective',
                  path_velocity_target='return',contact_slip='support',merged_entry_slip='max')
    prior=dict(m20_profile='ascent_entry_v1_stage9',refinement_version=readiness.layout.SAMPLING_VERSION,
               stair_path_reward_settings=settings,stair_path_reward_version='same',
               stair_path_config={'gain':.6},control_profile='same')
    target=dict(settings,version=course.VERSION,stair_path_reward_version='same',stair_path_config={'gain':.6},control_profile='same')
    init.validate_course_migration(prior,target)
    try:init.validate_course_migration(prior,dict(target,stair_path_config={'gain':.7}))
    except ValueError:pass
    else:raise AssertionError('controller change allowed in course migration')
    # Test real gradient diagnostics, including anti-alignment and no .grad writes.
    opt=loader.load_module('utils.ppo_diagnostics');p=torch.nn.Parameter(torch.tensor([1.,2.]))
    ppo=p.square().sum();anchor=-2*p.square().sum()
    values=opt.gradient_comparison(ppo,anchor,[p])
    assert values['ppo_anchor_grad_cosine']==-1 and values['anchor_to_ppo_grad_ratio']==2
    assert p.grad is None
    snap=opt.parameter_snapshot([p]);(ppo+anchor).backward()
    grad=p.grad.clone()
    with torch.no_grad():p.add_(-.01*p.grad)
    change=opt.parameter_change(snap);assert change['actor_parameter_update_l2']>0 and torch.equal(p.grad,grad)
    # Matched telemetry uses original episode row/goal even after live promotion.
    diag=loader.load_module(BASE+'path.stair_path_course_diagnostics').CourseDiagnostics('cpu')
    zero=torch.zeros(n);goals=torch.full((n,),6,dtype=torch.long)
    covered=torch.zeros(n,4,18,dtype=torch.bool);covered[:,:,:3]=True
    obs=NS(episode_goals=goals,episode_terrain_rows=torch.ones(n,dtype=torch.long),
        sequence=NS(targets=torch.arange(1,19)[None,None,:],covered=covered,rear_pair_paid=covered[:,0]),
        prefix_ready=zero.bool(),top_ready=zero.bool(),path_max=zero+.1,
        path_heading_total=zero+.2,path_samples=zero+2,path_forward_max=zero+1,
        failure_reason=torch.ones(n,dtype=torch.long))
    class Scene(dict):pass
    scene=Scene(robot=NS(data=NS(root_pos_w=torch.ones(n,3))))
    scene.env_origins=torch.zeros(n,3);scene.terrain=NS(terrain_types=torch.zeros(n,dtype=torch.long))
    env=NS(_ascent_entry_observer=obs,_ascent_adaptive_curriculum=state,scene=scene,
           command_manager=NS(get_command=lambda _:torch.tensor([[.5,0.,0.]]).expand(n,-1)))
    diag.observe(env,torch.tensor([0,1]),4001)
    assert diag.totals.reshape(9,6,9,2,-1)[0,1,3,0,0]==2
    assert diag.first_uncovered[3]==2
    logs={};writer=NS(add_scalar=lambda k,v,i:logs.update({k:v}))
    with TemporaryDirectory() as folder:
        diag.write(writer,folder,4099,1)
        record=json.loads((Path(folder)/'course_strata.jsonl').read_text())
        assert record['cells'][0]['row']==1 and record['cells'][0]['prefix']==6
        assert logs['Course/rear_pair_coverage']==.5 and logs['Course/four_wheel_coverage']==.5
        assert not diag.totals.any()
    print(json.dumps(dict(prefixes=course.PREFIXES,legacy_targets_and_windows_retained=True,
        legacy_critic_feature_retained=True,consolidation_promotion_only=True,failures_can_demote=True,
        native_state_roundtrip=True,readiness_ramp_retained=True,matched_episode_labels=True,
        gradient_diagnostics_do_not_modify_gradients=True,policy_checkpoints_loaded=0),indent=2))


if __name__=='__main__':main()
