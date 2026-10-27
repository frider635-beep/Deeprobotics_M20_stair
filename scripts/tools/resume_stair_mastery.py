#!/usr/bin/env python3
"""Resume Stage9 v8 -> v9 graded course evidence; never deserialize policies here.

Stop the old training process and wait for its saved-checkpoint message first.
Use --print-only to inspect the resolved command without starting training.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
LOGS=ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher_v1_stage2'
V5='m20_stair_path_fixed247_20261026_v5_progressive_course'
V6='m20_stair_path_fixed247_20261027_v6_mastered_prefix_floor'
V8='m20_stair_path_fixed247_20261027_v8_soft_pairing'
V9='m20_stair_path_fixed247_20261027_v9_course_evidence'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load-run',default='2026-10-27_13-52-28_ascent_entry_v1_stage9_v8')
    parser.add_argument('--checkpoint',help='Exact filename; default: greatest saved iteration number')
    parser.add_argument('--total-iterations',type=int,default=40000)
    parser.add_argument('--sampling-stage',type=int,choices=range(11),default=1)
    parser.add_argument('--run-name',default='ascent_entry_v1_stage9_v9')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--ascent-learning-rate','--ascent_learning_rate',type=float,default=1.5e-5,
                        help='Fixed PPO learning rate for resume (default: 1.5e-5).')
    parser.add_argument('--print-only',action='store_true')
    args=parser.parse_args()
    if not 0.<args.ascent_learning_rate<1.:
        parser.error('--ascent-learning-rate must be finite and in (0,1)')
    if Path(args.load_run).name!=args.load_run:parser.error('--load-run must be a run directory name')
    run=LOGS/args.load_run
    manifest_path=run/'params/launch_manifest.json'
    if not manifest_path.is_file():parser.error('Missing run manifest: '+str(manifest_path))
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('m20_profile')!='ascent_entry_v1_stage9':parser.error('Source must be Stage9')
    version=manifest.get('refinement_version')
    if version not in (V8,V9):parser.error('v9 accepts the current v8 or native v9 source only')
    checkpoints=[]
    for path in run.glob('model_*.pt'):
        match=re.fullmatch(r'model_(\d+)\.pt',path.name)
        if match and path.is_file() and path.stat().st_size>0:
            checkpoints.append((int(match[1]),path.name))
    if args.checkpoint:
        checkpoints=[p for p in checkpoints if p[1]==args.checkpoint]
    if not checkpoints:parser.error('No matching nonempty checkpoint file exists')
    iteration,checkpoint=max(checkpoints)
    remaining=args.total_iterations-iteration
    if remaining<=0:parser.error('--total-iterations must exceed the saved iteration')
    source_args=manifest['cli_args']
    command=[sys.executable,'-u','scripts/reinforcement_learning/rsl_rl/train.py',
        '--task','Rough-Deeprobotics-M20-StairTeacher-v1',
        '--m20_profile','ascent_entry_v1_stage9','--resume',
        '--load_run',re.escape(args.load_run)+'$','--checkpoint',re.escape(checkpoint)+'$']
    if version==V8:command+=['--stair_ascent_migrate_v8']
    command+=['--stair_sampling_mode','manual','--stair_sampling_stage',str(args.sampling_stage),
        '--run_name',args.run_name,'--num_envs',str(source_args['num_envs']),
        '--max_iterations',str(remaining),'--seed',str(source_args['seed']),
        '--ascent_learning_rate',str(args.ascent_learning_rate),
        '--device',args.device,'--headless']
    print(f'Source: {run/checkpoint}\nSaved iteration: {iteration}; additional iterations: {remaining}; total target: {args.total_iterations}',flush=True)
    print(f'Fixed PPO learning rate: {args.ascent_learning_rate:g}',flush=True)
    print(shlex.join(command),flush=True)
    if args.print_only:return
    environment=os.environ.copy();environment.setdefault('OMP_NUM_THREADS','4')
    raise SystemExit(subprocess.call(command,cwd=ROOT,env=environment))


if __name__=='__main__':main()
