#!/usr/bin/env python3
"""Paired raw-contact/path evaluation for either stair direction.

Passing the path checks alone never certifies ideal gait. Full per-wheel
coverage, both axle pairs and confirmed swing turns are measured separately.
"""
import argparse, hashlib, json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from collect_m20_gait_reference import ROOT,d,run_case
from analyze_m20_gait_reference import stable_events,LEGS
from evaluate_ascent_entry_v1 import sequence_summary


def summarize(folder, case):
    with np.load(folder/case['case']/'physics.npz') as z:data={key:z[key] for key in z.files}
    active=data['time']>=case['settle_s'];pos=data['root_pos'];first=case['terrain']['terrain_parameters']['start_x'][0]
    stairs=active&(pos[:,0]>=first)&(pos[:,0]<=first+20*case['terrain']['terrain_parameters']['tread_depth'][0])
    policy=np.load(folder/case['case']/'policy.npz')
    first_move=np.flatnonzero(active)[0];origin=pos[first_move,:2];heading=data['rpy'][first_move,2]
    normal=np.array([-np.sin(heading),np.cos(heading)]);error=(pos[:,:2]-origin)@normal
    result=dict(case=case['case'],speed=case['speed_m_s'],offset=case['spawn_x_offset_m'],status=case['status'],
        max_world_lateral_m=float(np.abs(pos[active,1]).max()),final_world_lateral_m=float(pos[-1,1]),
        max_path_error_m=float(np.abs(error[active]).max()),final_path_error_m=float(error[-1]),
        mean_abs_yaw_deg=float(np.rad2deg(np.abs(data['rpy'][stairs,2])).mean()) if stairs.any() else None,
        normalized_contact_slip=float(np.sum(data['slip'][active]*data['support'][active])*.005/max(pos[-1,0]-pos[first_move,0],.05)),
        max_effective_vy=float(np.abs(data['effective_command'][active,1]).max()),
        original_raw_vy_zero=bool((data['requested_command'][active,1]==0.).all()),
        platform_final_roll_pitch_deg=np.rad2deg(data['rpy'][-1,:2]).tolist())
    if 'descent_gate' in data:result['descent_gate_mean']=float(data['descent_gate'][active].mean())
    if 'path_control' in data:result['path_features_recorded']=True
    if case['terrain']['terrain']=='ascent':
        events,_=stable_events(data,case['case']);times=np.full((4,20),np.nan)
        for e in events:
            if 1<=e['tread_id']<=20:times[LEGS.index(e['wheel']),e['tread_id']-1]=e['touchdown_time_s']
        core=d.training_module('tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_alternating')
        book=core.SwingAlternation(1,'cpu')
        for i in range(len(data['time'])):
            book.update(torch.from_numpy(data['support'][i:i+1]),torch.from_numpy(data['wheel_pos'][i:i+1,:,2]),
                torch.from_numpy(data['wheel_velocity'][i:i+1,:,2]),
                torch.from_numpy(data['wheel_pos'][i:i+1,:,0].reshape(1,2,2).max(-1)>=first-.35),
                torch.tensor([bool(data['requested_command'][i,0]>.1)]),.005)
        result.update(sequence=sequence_summary(data,events),swing_failed=bool(book.failed[0]),
            all_wheel_stable_coverage=float(np.isfinite(times).mean()),
            rear_same_tread_sequential=float((np.isfinite(times[2:]).all(0)&(np.abs(times[2]-times[3])>=.04-1.e-6)).mean()))
    policy.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--reuse-source-report',type=Path,help='Reuse matching archived source rollouts; checkpoint hash and geometry must match')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--offsets',type=float,nargs='+',default=[-.1,0.,.1])
    parser.add_argument('--speeds',type=float,nargs='+',default=[.5,.7])
    parser.add_argument('--height',type=float,default=.15)
    parser.add_argument('--depth',type=float,default=.30)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.output_dir.exists():parser.error('Choose a new report directory')
    args.output_dir.mkdir(parents=True);rows=[]
    metadata=dict(source=str(args.source.resolve()),candidate=str(args.candidate.resolve()),
        checkpoint_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('source',args.source),('candidate',args.candidate)]},
        height_m=args.height,depth_m=args.depth,steps=20,settle_s=6.,recording_exit_distance_m=1.6,
        scope='Deterministic CPU MuJoCo screening, not PhysX or hardware acceptance')
    reused = None
    if args.reuse_source_report:
        reused=json.loads(args.reuse_source_report.read_text())
        if (reused['checkpoint_sha256']['source'] != metadata['checkpoint_sha256']['source']
                or reused['height_m'] != args.height or reused['depth_m'] != args.depth
                or reused['steps'] != 20 or reused['settle_s'] != 6.
                or reused['recording_exit_distance_m'] != metadata['recording_exit_distance_m']):
            raise ValueError('Archived source checkpoint or scenario differs')
        metadata['reused_source_report']=str(args.reuse_source_report.resolve())
        metadata['reused_source_report_sha256']=hashlib.sha256(args.reuse_source_report.read_bytes()).hexdigest()
    for label,checkpoint in [('source',args.source),('candidate',args.candidate)]:
        if label=='source' and reused is not None:
            for terrain in ('ascent','descent'):
                for speed in args.speeds:
                    for offset in args.offsets:
                        matches=[row for row in reused['rows'] if row['model']=='source' and row['terrain']==terrain
                                 and row['speed']==speed and row['offset']==offset]
                        if len(matches)!=1:raise ValueError('Archived source case missing/duplicate')
                        rows.append(matches[0])
            continue
        actor=d.load_actor(checkpoint)
        for terrain in ('ascent','descent'):
            folder=args.output_dir/label/terrain;folder.mkdir(parents=True)
            cfg=NS(output_dir=folder,model=ROOT/'deep_robotics_model/M20/mjcf/M20.xml',
                terrain_xml=ROOT/f'deploy/deploy_mujoco/terrains/stairs_{terrain}.xml',stair_height=args.height,
                tread_depth=args.depth,stair_count=20,settle=6.,duration=36.,exit_distance=1.6)
            for speed in args.speeds:
                for offset in args.offsets:
                    case=run_case(actor,cfg,speed,offset);row=summarize(folder,case)
                    row.update(model=label,terrain=terrain);rows.append(row)
                    (args.output_dir/'evaluation.json').write_text(json.dumps(dict(**metadata,rows=rows),indent=2)+'\n')
                    print(json.dumps(row),flush=True)
    pairs=[]
    for row in [r for r in rows if r['model']=='candidate']:
        old=next(r for r in rows if r['model']=='source' and all(r[key]==row[key] for key in ('terrain','speed','offset')))
        pairs.append(dict(terrain=row['terrain'],speed=row['speed'],offset=row['offset'],
            source_max_lateral_m=old['max_world_lateral_m'],candidate_max_lateral_m=row['max_world_lateral_m'],
            source_max_path_error_m=old['max_path_error_m'],candidate_max_path_error_m=row['max_path_error_m']))
    (args.output_dir/'evaluation.json').write_text(json.dumps(dict(**metadata,rows=rows,pairs=pairs),indent=2)+'\n')


if __name__=='__main__':main()
