#!/usr/bin/env python3
"""CPU counterexamples for teacher stop-aware curriculum and action history.

Production function bodies are compiled with simulator dependencies supplied.
These checks validate attribution/state transitions, not learned locomotion.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from check_stair_rewards import Entity, Scene, load_functions, quat_apply, yaw_quat


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py"


def production_namespace():
    spec = importlib.util.spec_from_file_location(
        "m20_test_diagnostics", ROOT / "source/rl_training/rl_training/teacher_diagnostics.py")
    diagnostics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostics)
    pair_spec = importlib.util.spec_from_file_location(
        "m20_test_pair_diagnostics", SOURCE.parent / "descent_front_pair.py")
    pair = importlib.util.module_from_spec(pair_spec)
    pair_spec.loader.exec_module(pair)
    ns = dict(torch=torch, np=np, SceneEntityCfg=Entity, quat_apply=quat_apply, yaw_quat=yaw_quat,
              quat_apply_inverse=lambda q, v: quat_apply(q * q.new_tensor([1., -1., -1., -1.]), v),
              export_teacher_metrics=diagnostics.export_teacher_metrics,
              descent_front_pair_metrics=pair.descent_front_pair_metrics,
              update_gait_level_from_terrain_mean=lambda value: None,
              get_gait_level_tensor=lambda env: torch.ones(env.num_envs))
    load_functions(SOURCE, ns)
    return ns


def check_curriculum(ns):
    budgets = torch.tensor([4., 4., 0., .5, 4., 4., 8., 4., 100.])
    net_requests = torch.tensor([4., 4., 0., .5, 0., 4., 8., 4., 100.])
    distances = torch.tensor([2.5, 2.5, 10., .4, 0., .2, 1.5, 1.5, 5.])
    n = budgets.numel()
    command = torch.zeros(n, 3)
    command[1, 0] = .7  # Same exposure and motion as row 0, different final command.
    term = SimpleNamespace(episode_commanded_translation_m=budgets,
                           episode_commanded_translation_w=torch.stack((net_requests, torch.zeros(n)), -1))
    captured = {}

    def update(env_ids, up, down):
        captured.update(up=up.clone(), down=down.clone())

    terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_generator=SimpleNamespace(size=(8., 8.))),
                              terrain_levels=torch.ones(n, dtype=torch.long), update_env_origins=update)
    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.stack((distances, torch.zeros(n), torch.zeros(n)), -1))))
    scene.terrain = terrain
    scene.env_origins = torch.zeros(n, 3)
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(
        get_term=lambda name: term, get_command=lambda name: command))
    ns["stair_distance_terrain_levels"](env, torch.arange(n), min_commanded_translation_m=.1)
    assert captured["up"].tolist() == [True, True, False, False, False, False, False, False, False]
    assert captured["down"].tolist() == [False, False, False, False, False, True, True, False, False]
    assert captured["up"][0] == captured["up"][1]
    assert captured["down"][0] == captured["down"][1]
    return {"zero_and_moving_episode_end_equivalent": True,
            "no_command_no_promotion": True,
            "reversal_practice_keeps_level": True,
            "insufficient_opportunity_keeps_level": True,
            "weak_progress_demotes_but_neutral_band_keeps_level": True,
            "demotion_threshold_capped_by_tile": True}


def command_class(ns):
    tree = ast.parse(SOURCE.read_text())
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "StairTeacherVelocityCommand")
    methods = [node for node in source_class.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_update_metrics", "reset")]

    class Parent:
        def _update_metrics(self):
            pass

        def reset(self, env_ids):
            return {}

    ns = dict(ns, Parent=Parent)
    wrapper = ast.ClassDef(name="CommandUnderTest", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                           keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(SOURCE), "exec"), ns)
    return ns["CommandUnderTest"]


def check_command_integrals(ns):
    command = command_class(ns)()
    command.num_envs, command.device = 2, "cpu"
    robot = SimpleNamespace(data=SimpleNamespace(root_quat_w=torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]])))
    command.robot = robot
    command._env = SimpleNamespace(step_dt=.02)
    command.vel_command_b = torch.tensor([[.5, 0, 0], [.5, 0, 0]])
    command.episode_commanded_translation_m = torch.zeros(2)
    command.episode_commanded_translation_w = torch.zeros(2, 2)
    command._metric_step_counter = torch.ones(2)
    command._entry_pause_pending = torch.zeros(2, dtype=torch.bool)
    command._entry_pause_decided_token = torch.full((2,), -1, dtype=torch.long)
    command._entry_pause_until_step = torch.zeros(2, dtype=torch.long)
    command._entry_pause_timer_expired = torch.zeros(2, dtype=torch.bool)
    for name in ("_stair_zero_transition_env", "_flat_zero_transition_env", "_stair_exit_zero_env"):
        setattr(command, name, torch.zeros(2, dtype=torch.bool))
    command.metrics = {key: torch.zeros(2) for key in (
        "stair_zero_transition_fraction", "flat_zero_transition_fraction", "stair_exit_zero_fraction",
        "descent_entry_pause_time_fraction")}
    command._update_metrics()
    robot.data.root_quat_w[0] = torch.tensor([0., 0, 0, 1.])  # Turned 180 degrees.
    command._update_metrics()
    assert torch.allclose(command.episode_commanded_translation_m, torch.tensor([.02, .02]))
    assert torch.allclose(command.episode_commanded_translation_w, torch.tensor([[0., 0], [.02, 0]]), atol=1e-7)
    command.reset(torch.tensor([0]))
    assert command.episode_commanded_translation_m[0] == 0
    assert command.episode_commanded_translation_w[0].abs().sum() == 0
    assert command.episode_commanded_translation_m[1] > 0
    assert command.episode_commanded_translation_w[1, 0] > 0
    return {"requested_path_and_world_net_integrals_distinct": True,
            "command_reset_is_per_environment": True}


def check_zero_history(ns):
    env = SimpleNamespace(num_envs=1, device="cpu", common_step_counter=0,
                          episode_length_buf=torch.zeros(1, dtype=torch.long),
                          action_manager=SimpleNamespace(action=torch.zeros(1, 1), prev_action=torch.zeros(1, 1)))
    cost = ns["teacher_action_smooth_l2"]
    values = []
    for age, value in enumerate((1., 0., 1., 0.), start=1):
        env.common_step_counter += 1
        env.episode_length_buf.fill_(age)
        env.action_manager.prev_action = env.action_manager.action.clone()
        env.action_manager.action.fill_(value)
        values.append(float(cost(env)))
        assert float(cost(env)) == values[-1]  # A second consumer cannot advance history.
    assert values == [0., 0., 4., 4.]
    env.common_step_counter += 1
    env.episode_length_buf.zero_()
    assert float(cost(env)) == 0
    late_env = SimpleNamespace(num_envs=1, device="cpu", common_step_counter=100,
                              episode_length_buf=torch.tensor([500]),
                              action_manager=SimpleNamespace(action=torch.ones(1, 1), prev_action=torch.zeros(1, 1)))
    assert float(cost(late_env)) == 0
    late_env.common_step_counter += 1
    late_env.episode_length_buf += 1
    assert float(cost(late_env)) == 0
    return {"alternating_zero_nonzero_targets_are_penalized": values,
            "repeated_consumers_preserve_history": True,
            "reset_incomplete_history_not_penalized": True,
            "random_initial_episode_age_does_not_fake_history": True}


if __name__ == "__main__":
    namespace = production_namespace()
    print(json.dumps(check_curriculum(namespace) | check_command_integrals(namespace)
                     | check_zero_history(namespace), indent=2))
