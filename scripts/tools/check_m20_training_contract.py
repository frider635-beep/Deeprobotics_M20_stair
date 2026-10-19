#!/usr/bin/env python3
"""Validate the control contract and optional initialization checkpoint without Isaac."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('m20_preflight',ROOT/'source/rl_training/rl_training/m20_preflight.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path);p.add_argument('--for-resume',action='store_true');a=p.parse_args()
 tuning=m.source_constants(ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py',{'TUNING'})['TUNING']
 result={'control':m.validate_control_contract(tuning,ROOT/'deploy/deploy_mujoco/deploy_mujoco.py')}
 if a.checkpoint:result['checkpoint']=m.validate_actor_checkpoint(torch.load(a.checkpoint,map_location='cpu',weights_only=True),require_version=a.for_resume)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
