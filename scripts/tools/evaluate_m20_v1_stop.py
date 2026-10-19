#!/usr/bin/env python3
"""v1 stop refinement acceptance: paired control, side-slip, progress and restarts.

Baseline-only mode records the starting behavior. It does not claim any new
policy has been trained. Yaw offsets are finite tests, not a success-rate proof.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('m20_stop_evaluator',ROOT/'scripts/tools/evaluate_m20_refinement.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
BASELINE=ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-15_11-59-41_refine_stop_heading_posture_v1/model_5100.pt'
STOP_CASES={
 'flat_zero':('flat',[(6.,0.)]),
 'flat_move_stop':('flat',[(6.,0.),(3.,.5),(6.,0.),(3.,.5),(6.,0.)]),
 'flat_reverse_stop':('flat',[(6.,0.),(3.,.5),(6.,0.),(3.,-.5),(6.,0.)]),
 'ascent_hold':('ascent',[(6.,0.),(9.,.7),(6.,0.)]),
 'descent_hold':('descent',[(6.,0.),(14.,.7),(6.,0.)]),
 'ascent_mid_stair_pause':('ascent',[(6.,0.),(3.5,.7),(6.,0.),(6.,.7),(6.,0.)]),
 'descent_mid_stair_pause':('descent',[(6.,0.),(5.,.7),(6.,0.),(10.,.7),(6.,0.)]),
 'ascent_reverse_stop':('ascent',[(6.,0.),(3.5,.7),(6.,0.),(2.,-.4),(6.,0.),(8.,.7),(6.,0.)]),
 'descent_reverse_stop':('descent',[(6.,0.),(5.,.7),(6.,0.),(2.,-.4),(6.,0.),(10.,.7),(6.,0.)]),
}


def acceptance(base, candidate, name):
    """Gait style is measured but is not this stop-only experiment's target."""
    m=candidate['metrics'];b=base['metrics'];g=candidate['gates']
    checks={k:v for k,v in g.items() if k.startswith('zero_')}
    checks.update(complete_profile=m['complete_profile'],no_fall=not m['fell'],
                  no_lateral_edge_contamination=m['near_edge_samples']==0)
    if 'mid_stair_pause' in name:
        checks['stair_stop_actually_exercised']=any(z['stair_samples_after_brake']>=100 for z in m['zero_phases'])
    if 'reverse_stop' in name and STOP_CASES[name][0]!='flat':
        for direction,label in ((1,'forward'),(-1,'reverse')):
            checks[f'{label}_stair_stop_actually_exercised']=any(
                direction*z.get('previous_command',[0,0,0])[0]>.1 and z['stair_samples_after_brake']>=100
                for z in m['zero_phases'])
    phases=m['zero_phases']
    checks['each_stop_has_one_second_quiet']=bool(phases) and all(z['max_quiet_duration_s']>=1.-1.e-6 for z in phases)
    checks['position_drift_le_003m']=bool(phases) and all(z['max_position_drift_m'] is not None and z['max_position_drift_m']<=.03 for z in phases)
    if m['moving_route_error_m']['samples']:
        checks['moving_max_lateral_not_worse_002m']=m['moving_route_error_m']['max']<=b['moving_route_error_m']['max']+.02
        checks['moving_mean_lateral_not_worse_001m']=m['moving_route_error_m']['mean']<=b['moving_route_error_m']['mean']+.01
        checks['moving_max_heading_not_worse_2deg']=m['moving_max_abs_yaw_deg']<=b['moving_max_abs_yaw_deg']+2.
        checks['moving_progress_at_least_90pct']=m['moving_route_progress_m']['max']>=max(0.,b['moving_route_progress_m']['max']*.9)
        checks['moving_end_x_not_behind_025m']=m['moving_end_x_m']>=b['moving_end_x_m']-.25
        checks['each_motion_and_restart_progress_preserved']=(len(m['moving_segments'])==len(b['moving_segments']) and
            all(c['end_progress_m']>=max(0.,old['end_progress_m']*.9-.02)
                for old,c in zip(b['moving_segments'],m['moving_segments'])))
    if STOP_CASES[name][0]!='flat':
        checks['crossed_finish']=m['crossed_finish']
    for k in ('rear_front_wheel_contact_physics_substeps','rear_wheel_front_leg_contact_physics_substeps'):
        checks[k+'_not_worse']=m['physics_contact_substeps'][k]<=b['physics_contact_substeps'][k]
    checks['nonwheel_contact_not_worse']=m['nonwheel_ground_contact_samples']<=b['nonwheel_ground_contact_samples']
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=BASELINE)
    p.add_argument('--candidate',type=Path,help='Omit to record baseline only')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',action='append',choices=tuple(STOP_CASES))
    p.add_argument('--yaw-offset-deg',action='append',type=float,help='Repeat; default 0,-3,+3 degrees')
    p.add_argument('--steps',type=int,help='Smoke cap only; incomplete profiles cannot pass')
    p.add_argument('--require-pass',action='store_true')
    a=p.parse_args()
    if a.require_pass and a.candidate is None:p.error('--require-pass requires --candidate')
    if a.steps is not None and a.steps<=0:p.error('--steps must be positive')
    e.CASES.update(STOP_CASES)
    torch.set_num_threads(1)
    report={'purpose':'v1 stop-only acceptance','trained_candidate_present':a.candidate is not None,
            'controller':'legacy_v1 for both actors','deploy_sha256':e.d.DEPLOY_SCRIPT_SHA256,
            'runtime':dict(python=platform.python_version(),torch=str(torch.__version__),mujoco=e.d.mujoco.__version__),
            'limitations':['Finite default-stance/yaw-offset trials do not establish behavior for all starting poses.',
                           'Reward/config reconstruction is v1-compatible, not a historical source replay.',
                           'Two-centimetre maximum and one-centimetre mean lateral margins are engineering tolerances, not confidence bounds.',
                           'Only flat and fixed ascending/descending stairs are replayed; rough terrain still needs validation.'],
            'runs':{}}
    runs=[('baseline',a.baseline)]+([('candidate',a.candidate)] if a.candidate else [])
    for label,path in runs:
        actor=e.d.load_actor(path)
        report['runs'][label]={'checkpoint':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':{}}
        for yaw in a.yaw_offset_deg if a.yaw_offset_deg is not None else [0.,-3.,3.]:
            cli=SimpleNamespace(model=ROOT/'deep_robotics_model/M20/mjcf/M20.xml',
                terrain_dir=ROOT/'deploy/deploy_mujoco/terrains',steps=a.steps,
                disable_heading_hold=False,legacy_heading_hold=True,spawn_yaw_offset_deg=yaw)
            for name in a.case or STOP_CASES:
                key=f'{name}/yaw{yaw:+g}'
                print(f'[STOP EVAL] {label}/{key}',flush=True)
                value=e.run_case(actor,name,cli)
                if label=='candidate':
                    value['stop_only_acceptance']=acceptance(report['runs']['baseline']['cases'][key],value,name)
                report['runs'][label]['cases'][key]=value
                a.output.parent.mkdir(parents=True,exist_ok=True)
                a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if a.candidate:
        report['candidate_pass']=all(all(c['stop_only_acceptance'].values()) for c in report['runs']['candidate']['cases'].values())
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(a.output),'candidate_pass':report.get('candidate_pass')},indent=2))
    if a.require_pass and not report['candidate_pass']:raise SystemExit(2)

if __name__=='__main__':main()
