#!/usr/bin/env python3
"""Paired default-stair acceptance for v1; keep reports even on rejection.

This evaluates physical completion, early per-wheel tread coverage, reference
support timing, normalized entry slip and middle-flight regression together.
No checkpoint is promoted or overwritten by this tool.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from collect_m20_gait_reference import ROOT, d, run_case
from analyze_m20_gait_reference import LEGS, stable_events, settled_mask, stats


def load_data(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def sequence_summary(data, events, count=20):
    times = np.full((4, count), np.nan)
    for event in events:
        index = event['tread_id']-1
        if 0 <= index < count:
            wheel = LEGS.index(event['wheel'])
            if not np.isfinite(times[wheel, index]): times[wheel, index] = event['touchdown_time_s']
    paired = np.isfinite(times[:2]).all(0)
    sequential = paired & (np.abs(times[0]-times[1]) >= .04-1.e-6)
    early = 0
    for wheel in [0, 1]:
        for index in range(1, count):
            if np.isfinite(times[wheel, index]) and (not np.isfinite(times[1-wheel, index-1]) or
                                                   times[wheel, index] < times[1-wheel, index-1]+.005-1.e-6):
                early += 1
    front_top = float(np.max(times[:2, -1])) if np.isfinite(times[:2, -1]).all() else None
    rear_top = float(np.max(times[2:, -1])) if np.isfinite(times[2:, -1]).all() else None
    all_top = (data['support_tread'] == count).all(-1) & (data['time'] >= 6.)
    ready = False
    if front_top is not None and rear_top is not None:
        root_origin = float(np.interp(rear_top+.04, data['time'], data['root_pos'][:, 0]))
        valid = all_top & (np.abs(data['rpy'][:, :2]).sum(-1) < .15)
        valid &= (data['root_pos'][:, 0]-root_origin >= .35)
        valid &= data['base_velocity'][:, 0] > .1
        consecutive = longest = 0
        for value in valid:
            consecutive = consecutive+1 if value else 0; longest = max(longest, consecutive)
        ready = longest*.005 >= .20-1.e-6
    return {'front_same_tread_sequential_first_four': float(sequential[:4].mean()),
            'front_same_tread_sequential_full_flight': float(sequential.mean()),
            'front_early_advance_count': early,
            'front_top_touchdown_s': front_top, 'rear_top_touchdown_s': rear_top,
            'front_then_rear_platform': bool(front_top is not None and rear_top is not None and rear_top-front_top >= .04-1.e-6),
            'platform_level_forward_035m_stable_020s': bool(ready)}


def summarize(data,case,reference):
    events,_=stable_events(data,case)
    n=len(data['time']);progress=np.zeros((n,4),dtype=int)
    for e in events:
        progress[e['sample_index']:,LEGS.index(e['wheel'])]=e['tread_id']
    front=progress[:,:2].max(-1)
    entry=(front>=1)&(front<=4)
    middle=(front>=5)&(front<=18)
    period=reference['period_s']['median']
    phases=np.array(reference['phase_grid'])
    probability=np.array(reference['trajectories']['support_probability'])
    fl=np.array([e['touchdown_time_s'] for e in events if e['wheel']=='FL' and e['tread_id']>0])
    timing=None
    if entry.any() and len(fl):
        t=data['time'][entry];latest=fl[np.maximum(0,np.searchsorted(fl,t,side='right')-1)]
        phase=np.remainder((t-latest)/period,1.)
        expected=np.stack([np.interp(phase,phases,probability[:,j]) for j in range(4)],-1)
        timing=float(np.mean((expected-data['support'][entry])**2))
    early=[e for e in events if e['wheel'] in ('FL','FR') and 1<=e['tread_id']<=4]
    coverage=sum(len({e['tread_id'] for e in early if e['wheel']==leg}) for leg in ('FL','FR'))/8.
    skipped=sum(e['skipped_treads'] for e in early)
    quiet=settled_mask(data['support'])
    def window(mask):
        if not mask.any():
            return None
        xyz=data['root_pos'][mask]
        distance=float(xyz[-1,0]-xyz[0,0])
        support=data['support'][mask]
        slip=data['slip'][mask]
        integral=(slip*support).sum(0)*.005
        return {'duration_s':float(mask.sum()*.005),'forward_progress_m':distance,
                'wheel_slip_distance_m':integral.tolist(),
                'sum_slip_distance_per_forward_m':float(integral.sum()/max(distance,.05)),
                'supported_slip_m_s':{leg:stats(data['slip'][mask&data['support'][:,j],j]) for j,leg in enumerate(LEGS)},
                'settled_slip_m_s':{leg:stats(data['slip'][mask&quiet[:,j],j]) for j,leg in enumerate(LEGS)},
                'peak_abs_roll_deg':float(np.rad2deg(np.abs(data['rpy'][mask,0])).max()),
                'nonwheel_contact_frames':int(np.sum(data['nonwheel_contact_peak_force'][mask]>1.)),
                'self_contact_frames':int(np.sum(data['self_contact_peak_force'][mask]>1.))}
    return {'front_tread_1_to_4_coverage':coverage,'front_skipped_stable_treads':skipped,
            'sequence': sequence_summary(data, events),
            'entry_support_timing_mse':timing,'entry':window(entry),'middle':window(middle),
            'landing_events':events,'new_branch_saturation_fraction':float((np.abs(data.get('landing_bounded',data.get('entry_bounded',np.zeros((n,16)))))>.95).mean()),
            'entry_branch_saturation_fraction':float((np.abs(data.get('entry_bounded',np.zeros((n,16))))>.95).mean()),
            'landing_branch_saturation_fraction':float((np.abs(data.get('landing_bounded',np.zeros((n,16))))>.95).mean()),
            'sequence_branch_saturation_fraction':float((np.abs(data.get('sequence_bounded',np.zeros((n,16))))>.95).mean()),
            'old_branch_saturation_fraction':float((np.abs(data['residual_bounded'])>.95).mean())}


def acceptance(pairs):
    checks={}
    checks['all_cases_complete']=all(p['candidate_status']=='cleared_stairs' for p in pairs)
    checks['complete_entry_and_middle_samples']=all(p['candidate']['entry'] is not None and p['candidate']['middle'] is not None for p in pairs)
    if not checks['complete_entry_and_middle_samples']:
        return {**checks,'passed':False}
    def values(which,key):
        return np.array([p[which][key] for p in pairs],float)
    def slips(which,window):
        return np.array([p[which][window]['sum_slip_distance_per_forward_m'] for p in pairs],float)
    baseline_timing=values('baseline','entry_support_timing_mse')
    candidate_timing=values('candidate','entry_support_timing_mse')
    checks['each_run_time_le_baseline_110pct']=all(p['candidate_duration_s']<=p['baseline_duration_s']*1.10 for p in pairs)
    checks['front_coverage_no_case_regression']=bool(np.all(values('candidate','front_tread_1_to_4_coverage')>=values('baseline','front_tread_1_to_4_coverage')))
    checks['all_front_treads_covered_every_case']=bool(np.all(values('candidate','front_tread_1_to_4_coverage')==1.))
    checks['mean_entry_slip_reduced_10pct']=bool(slips('candidate','entry').mean()<=slips('baseline','entry').mean()*.90)
    checks['mean_support_timing_error_reduced_5pct']=bool(candidate_timing.mean()<=baseline_timing.mean()*.95)
    checks['each_case_entry_slip_le_115pct']=bool(np.all(slips('candidate','entry')<=slips('baseline','entry')*1.15))
    checks['each_case_middle_slip_le_115pct']=bool(np.all(slips('candidate','middle')<=slips('baseline','middle')*1.15))
    checks['no_new_nonwheel_or_self_contacts']=all(
        p['candidate'][window][key]<=p['baseline'][window][key]
        for p in pairs for window in ('entry','middle') for key in ('nonwheel_contact_frames','self_contact_frames'))
    checks['passed']=all(checks.values())
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,default=ROOT/'docs/test_docs/07_上楼步态校正/reference_20261020')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--speeds',type=float,nargs='+',default=[.5,.7])
    parser.add_argument('--spawn-x-offsets',type=float,nargs='+',default=[-.1,0.,.1])
    parser.add_argument('--require-pass',action='store_true')
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new output directory to preserve evaluation evidence')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    actor=d.load_actor(args.candidate)
    if not getattr(actor,'uses_ascent_entry',False):
        parser.error('Candidate must use the v1 ascent-entry architecture')
    if getattr(actor, 'uses_landing', False):
        expected = actor.m20_training_manifest.get('frozen_stage2_state_sha256')
        if not expected or actor.source_state_digest() != expected:
            raise ValueError('Complete stage2 source policy changed or has no frozen-state digest')
    if getattr(actor, 'uses_merged_ascent', False):
        d.training_module('algorithms.initialization.ascent_unified_init').validate_source(args.candidate)
    if getattr(actor, 'uses_sequence', False):
        expected = actor.m20_training_manifest.get('frozen_stage3_state_sha256')
        if not expected or actor.sequence_source_digest() != expected:
            raise ValueError('Complete stage3 source policy changed or has no frozen-state digest')
    manifest=json.loads((args.baseline_dir/'manifest.json').read_text())
    if actor.m20_training_manifest.get('frozen_full_baseline_sha256')!=manifest['checkpoint_sha256']:
        raise ValueError('Candidate and paired baseline provenance differ')
    baseline_state=torch.load(manifest['checkpoint'],map_location='cpu',weights_only=True)['actor_state_dict']
    for name,value in baseline_state.items():
        if not torch.equal(actor.state_dict()[name],value):
            raise ValueError('Frozen baseline parameter changed: '+name)
    cfg=SimpleNamespace(output_dir=args.output_dir,model=ROOT/'deep_robotics_model/M20/mjcf/M20.xml',
        terrain_xml=ROOT/'deploy/deploy_mujoco/terrains/stairs_ascent.xml',stair_height=None,tread_depth=None,
        stair_count=None,settle=6.,duration=32.)
    pairs=[]
    for speed in args.speeds:
        reference=json.loads((args.baseline_dir/f'reference_vx{speed:.2f}.json').read_text())
        for offset in args.spawn_x_offsets:
            baseline_case=next(c for c in manifest['cases'] if c['speed_m_s']==speed and c['spawn_x_offset_m']==offset)
            case=run_case(actor,cfg,speed,offset)
            baseline=summarize(load_data(args.baseline_dir/baseline_case['case']/'physics.npz'),baseline_case['case'],reference)
            candidate=summarize(load_data(args.output_dir/case['case']/'physics.npz'),case['case'],reference)
            pair={'case':case['case'],'speed_m_s':speed,'spawn_x_offset_m':offset,
                  'baseline_status':baseline_case['status'],'candidate_status':case['status'],
                  'baseline_duration_s':baseline_case['duration_s'],'candidate_duration_s':case['duration_s'],
                  'baseline':baseline,'candidate':candidate}
            pairs.append(pair)
            (args.output_dir/'paired_progress.json').write_text(json.dumps(pairs,indent=2)+'\n')
    checks=acceptance(pairs)
    if getattr(actor, 'uses_sequence', False) or getattr(actor, 'uses_merged_ascent', False):
        checks.update(front_same_tread_sequence_full_flight=all(
            p['candidate']['sequence']['front_same_tread_sequential_full_flight'] == 1. for p in pairs),
            no_front_early_advance=all(p['candidate']['sequence']['front_early_advance_count'] == 0 for p in pairs),
            front_then_rear_platform=all(p['candidate']['sequence']['front_then_rear_platform'] for p in pairs),
            platform_level_and_forward=all(p['candidate']['sequence']['platform_level_forward_035m_stable_020s'] for p in pairs))
        checks['passed'] = all(value for key, value in checks.items() if key != 'passed')
    result={'candidate':str(args.candidate.resolve()),'candidate_sha256':hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
            'baseline_checkpoint_sha256':manifest['checkpoint_sha256'],'checks':checks,'pairs':pairs,
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__).resolve(),ROOT/'scripts/tools/collect_m20_gait_reference.py',
                 ROOT/'source/rl_training/rl_training/models/ascent_entry_actor.py',
                 ROOT/'source/rl_training/rl_training/models/ascent_landing_actor.py',
                 ROOT/'source/rl_training/rl_training/models/ascent_sequence_actor.py',
                 ROOT/'deploy/deploy_mujoco/deploy_mujoco.py')},
            'acceptance_scope':'default 15/30cm staircase and controlled starting offsets; not hardware certification'}
    (args.output_dir/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# v1 上楼改进配对评估','','候选：`'+str(args.candidate)+'`。',
           '',f"验收通过：**{checks['passed']}**。所有阈值见evaluation.json；失败保留数据，不替换baseline。",'',
           '|速度|X偏移|完成|前四阶双前轮覆盖率 baseline→候选|入阶归一化滑移 baseline→候选|支撑时序MSE baseline→候选|',
           '|---|---|---|---|---|---|']
    for p in pairs:
        b,c=p['baseline'],p['candidate']
        def fmt(x):return '无样本' if x is None else f'{x:.4f}'
        bs=b['entry']['sum_slip_distance_per_forward_m'] if b['entry'] else None
        cs=c['entry']['sum_slip_distance_per_forward_m'] if c['entry'] else None
        lines.append(f"|{p['speed_m_s']}|{p['spawn_x_offset_m']:+.2f}|{p['candidate_status']}|{b['front_tread_1_to_4_coverage']:.3f}→{c['front_tread_1_to_4_coverage']:.3f}|{fmt(bs)}→{fmt(cs)}|{fmt(b['entry_support_timing_mse'])}→{fmt(c['entry_support_timing_mse'])}|")
    lines+=['','归一化滑移为四轮踏面接触切向速度积分之和/机身前进距离；包含落地瞬态。覆盖率要求每只前轮在每一级出现≥40ms稳定支撑。',
            '时序MSE基于实测FL落地重新对齐参考支撑概率；覆盖率与通行时间另行验收，避免停步或漏阶掩盖时序误差。','',
            '## 检查项','']+[f'- {key}: {value}' for key,value in checks.items()]
    (args.output_dir/'评估报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(checks,indent=2),flush=True)
    if args.require_pass and not checks['passed']:
        raise SystemExit(1)


if __name__=='__main__':main()
