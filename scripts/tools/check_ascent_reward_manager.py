#!/usr/bin/env python3
"""Exercise installed Isaac reward-manager code on CPU with a small scene adapter.

Loads the actual manager and RotationGaitSymmetry class bodies without starting
Kit/PhysX. Covers class construction, compute, gating and partial episode reset.
"""
import argparse
import ast
import copy
import inspect
import json
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from abc import ABC, abstractmethod
from collections.abc import Sequence
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'deploy/deploy_mujoco'))
import deploy_mujoco as deployment


def load_classes(path, names, scope):
    tree = ast.parse(path.read_text())
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    nodes += [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(nodes) == len(names) + 1
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), scope)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isaaclab-root', type=Path, default=Path('/home/arcus/IsaacLab'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    managers = args.isaaclab_root / 'source/isaaclab/isaaclab/managers'

    class TermCfg(NS): pass
    class Entity:
        def __init__(self, name): self.name = name; self.calls = 0
        def resolve(self, scene): self.calls += 1; assert self.name in scene.sensors

    scope = dict(ABC=ABC, abstractmethod=abstractmethod, Sequence=Sequence, torch=torch, copy=copy, inspect=inspect,
                 ManagerTermBaseCfg=TermCfg, RewardTermCfg=TermCfg, SceneEntityCfg=Entity,
                 logger=logging.getLogger(__name__))
    load_classes(managers / 'manager_base.py', {'ManagerTermBase', 'ManagerBase'}, scope)
    load_classes(managers / 'reward_manager.py', {'RewardManager'}, scope)
    base = scope['ManagerTermBase']
    # The adapter imports only this base. Restore the process module table later.
    manager_module = ModuleType('isaaclab.managers'); manager_module.ManagerTermBase = base
    previous = sys.modules.get('isaaclab.managers')
    sys.modules['isaaclab.managers'] = manager_module
    try:
        mdp = deployment.training_module('tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_alternating_mdp')
        scope.update(RewTerm=TermCfg, get_gait_level_tensor=lambda env: torch.ones(env.num_envs))
        load_classes(ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py',
                     {'RotationGaitSymmetry'}, scope)
        symmetry = scope['RotationGaitSymmetry']
        sensor = NS(data=NS(current_contact_time=torch.ones(4, 4)),
                    find_bodies=lambda name: ([['fl', 'hr', 'fr', 'hl'].index(name)], [name]))
        scene = NS(sensors={'contact_forces': sensor}, terrain=NS(terrain_types=torch.tensor([0, 0, 8, 9])))
        env = NS(num_envs=4, device='cpu', step_dt=.02, max_episode_length_s=24.,
                 sim=NS(is_playing=lambda: True), scene=scene,
                 command_manager=NS(get_command=lambda _: torch.tensor([[.5, 0., 0.]]*4)),
                 _ascent_entry_observer=NS(positive_gate=torch.tensor([False, True, False, False])))
        params = dict(command_name='base_velocity', sensor_cfg=Entity('contact_forces'),
                      group_a_body_names=['fl', 'hr'], group_b_body_names=['fr', 'hl'],
                      target_duty=.5, std=.5, lin_vel_threshold=2., ang_vel_threshold=-1.)
        # Reproduce the reported failure with the previous function-only wrapper.
        broken = scope['RewardManager'](NS(symmetry=TermCfg(func=mdp.positive_reward, weight=2.,
            params=dict(source_func=symmetry, source_params=copy.deepcopy(params), require_progress=False))), env)
        try:
            broken.compute(dt=.02)
        except TypeError as exc:
            assert 'command_name' in str(exc)
        else:
            raise AssertionError('Original class-call failure was not reproduced')
        terms = NS(symmetry=TermCfg(func=symmetry, weight=2., params=params))
        mdp.gate_positive_rewards(terms)
        assert mdp.audit_positive_rewards(terms) == ['symmetry']
        wrapper = terms.symmetry.func
        mdp.gate_positive_rewards(terms)
        assert terms.symmetry.func is wrapper, 'Class wrapper was wrapped twice'
        manager = scope['RewardManager'](terms, env)
        assert len(manager._class_term_cfgs) == 1
        configured = manager.get_term_cfg('symmetry')
        assert mdp.audit_positive_rewards(NS(symmetry=configured)) == ['symmetry']
        instance = configured.func
        assert isinstance(instance, base) and isinstance(instance.source, symmetry)
        assert instance.source.cfg.params['sensor_cfg'].calls == 1
        for index in (1, 2):
            reward = manager.compute(dt=.02).clone()
            assert reward[0] == 0. and (reward[1:] > 0.).all() and torch.isfinite(reward).all()
            assert instance.source.buffer_idx == index
        manager.reset(env_ids=torch.tensor([1]))
        assert instance.source.contact_buffer[1].sum() == 0.
        assert (instance.source.contact_buffer[[0, 2, 3]].sum(dim=(1, 2)) > 0.).all()
        assert instance.source.cfg.params['sensor_cfg'].calls == 1
        result = dict(original_failure_reproduced=True, installed_isaac_manager_code=True,
                      actual_rotation_gait_symmetry=True, initialized_once=True,
                      positive_gate_and_flat_preserved=True, class_registered_for_reset=True,
                      partial_reset_forwarded=True, wrapping_idempotent=True,
                      nested_parameters_resolved_once=True, physx_started=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        if previous is None: sys.modules.pop('isaaclab.managers', None)
        else: sys.modules['isaaclab.managers'] = previous


if __name__ == '__main__': main()
