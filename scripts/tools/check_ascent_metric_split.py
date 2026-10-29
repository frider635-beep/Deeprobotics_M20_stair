#!/usr/bin/env python3
"""CPU checks for the main/practice ascent metric split and reward audit."""
from types import SimpleNamespace as NS
import ast
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import inspect
import json
from pathlib import Path
import pickle

import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'metric_split_loader', ROOT / 'source/rl_training/rl_training/utils/module_loader.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
BASE = 'tasks.manager_based.locomotion.velocity.mdp.ascent.'
adaptive_mdp = loader.load_module(BASE + 'ascent_adaptive_mdp')
alternating = loader.load_module(BASE + 'ascent_alternating_mdp')
audit = loader.load_module(BASE + 'ascent_reward_audit')
pairing = loader.load_module(BASE + 'ascent_pairing')


def metric_split_check():
    count, treads = 4, 4
    # Use the real ledger: targets is a shared [1, 1, treads] broadcast table,
    # while covered/pair_paid/best_dwell have one row per environment.
    ledger = pairing.RearEntryPairLedger(count, 'cpu', targets=treads)
    assert ledger.targets.shape == (1, 1, treads)
    ledger.covered[:] = torch.tensor([
        [[1, 1, 1, 1]] * 4,
        [[1, 0, 0, 0]] * 4,
        [[1, 1, 1, 0]] * 4,
        [[1, 1, 1, 1]] * 4,
    ], dtype=torch.bool)
    ledger.pair_paid[:] = torch.tensor([
        [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]
    ], dtype=torch.bool)
    ledger.best_dwell.fill_(1)
    env = NS(
        num_envs=count, device=torch.device("cpu"),
        cfg=NS(ascent_adaptive_ascent_column_ids=(0, 1, 2, 3),
               ascent_sequence_step_count=4,
               ascent_practice_only_columns=(0, 2), ascent_adaptive_training=True),
        scene=NS(terrain=NS(terrain_types=torch.tensor([0, 1, 2, 3]))),
    )
    observer = NS(
        sequence=ledger,
        episode_goals=torch.tensor([4, 2, 3, 4]),
        prefix_ready=torch.ones(count, dtype=torch.bool),
        top_ready=torch.ones(count, dtype=torch.bool),
        stats={},
    )
    for name in ('pairs', 'ordered', 'missing', 'early', 'failure', 'landing_credit',
                 'FL2_support_frames', 'FR1_support_frames'):
        observer.stats['adaptive_' + name] = torch.arange(1, count + 1, dtype=torch.float32)
    for name in ('FL2_longest_support_s', 'FR1_longest_support_s',
                 'adaptive_height_total', 'adaptive_opportunity'):
        observer.stats['adaptive_' + name.removeprefix('adaptive_')] = torch.ones(count)
    env._ascent_entry_observer = observer
    env._ascent_adaptive_curriculum = NS(goals=lambda: observer.episode_goals)
    result = adaptive_mdp.metrics(env, None)
    assert result['sample_count'].item() == 2
    assert result['practice_sample_count'].item() == 2
    assert torch.isclose(result['front_current_prefix_coverage'], torch.tensor(5 / 6))
    assert result['practice_front_current_prefix_coverage'] == 1.
    assert torch.isclose(result['front_same_tread_sequential_current_prefix'], torch.tensor(5 / 6))
    assert result['practice_front_same_tread_sequential_current_prefix'] == 1.
    # Partial resets use global, non-contiguous environment IDs. Either group
    # can be empty, including an initial reset with no ascent environments.
    for ids in (torch.tensor([3, 1]), torch.tensor([2, 0]), torch.tensor([], dtype=torch.long)):
        subset = adaptive_mdp.metrics(env, ids)
        practice = sum(int(index) in (0, 2) for index in ids)
        assert subset['sample_count'] == len(ids) - practice
        assert subset['practice_sample_count'] == practice
        assert all(torch.isfinite(value) for value in subset.values())
    # An evaluation pools all ascent columns into the main metrics.
    env.cfg.ascent_adaptive_training = False
    evaluation = adaptive_mdp.metrics(env, None)
    assert evaluation['sample_count'] == count and evaluation['practice_sample_count'] == 0
    assert torch.isclose(evaluation['front_current_prefix_coverage'], torch.tensor(12 / 13))
    return 'passed: real shared-target ledger, exact main/practice metrics, partial/empty resets, evaluation'


def reward_audit_check():
    rewards = NS(**{name: NS(weight=1.0, func=source_reward, params={'scale': 2.0})
                    for name in audit.CONFLICTING_REWARDS})
    # Match the real Stage4 -> Stage6/7 -> Stage9 order, including reapplication.
    # Default profiles must keep the historical globally-disabled weights.
    historical = NS(**{name: NS(weight=1.0, func=source_reward, params={'scale': 2.0})
                       for name in audit.CONFLICTING_REWARDS})
    historical.joint_mirror.weight = -.03
    audit.disable_conflicting_rewards(historical)
    alternating.gate_positive_rewards(historical)
    assert all(term.weight == 0.0 for term in vars(historical).values())
    assert all(item['weight'] == 0.0 for item in audit.audit_conflicting_rewards(historical).values())
    # Exercise the explicit ascent-only masking option separately.
    audit.disable_conflicting_rewards(rewards, preserve_descent=True)
    alternating.gate_positive_rewards(rewards)
    audit.audit_conflicting_rewards(rewards)
    audit.disable_conflicting_rewards(rewards, preserve_descent=True)
    alternating.gate_positive_rewards(rewards)
    report = audit.audit_conflicting_rewards(rewards)
    assert all(item['ascent_masked'] or item['weight'] == 0.0 for item in report.values())
    assert set(alternating.audit_positive_rewards(rewards)) == audit.DESCENT_PRESERVED_REWARDS
    env = NS(num_envs=3, scene=NS(terrain=NS(terrain_types=torch.tensor([0, 9, 8]))),
             cfg=NS(ascent_adaptive_ascent_column_ids=(0,)),
             _ascent_entry_observer=NS(positive_gate=torch.zeros(3, dtype=torch.bool)),
             command_manager=NS(get_command=lambda _: torch.tensor([[.5, 0., 0.]] * 3)))
    for name in audit.DESCENT_PRESERVED_REWARDS:
        term = getattr(rewards, name)
        assert term.weight == 1.0
        assert term.func is alternating.positive_reward
        assert term.params['source_func'] is audit.ascent_disabled_reward
        assert term.params['source_params']['source_func'] is source_reward
        # Mask and gate both preserve descent/flat output, even with a false gate.
        assert term.func(env, **term.params).tolist() == [0., 4., 6.]
        assert pickle.loads(pickle.dumps(term.params['source_func'])) is audit.ascent_disabled_reward
        assert set(inspect.signature(audit.ascent_disabled_reward).parameters) == {
            'env', *term.params['source_params']}
        # Removing just the mask must still fail the conflict audit.
        original = term.params['source_func']
        term.params['source_func'] = source_reward
        try:
            audit.audit_conflicting_rewards(rewards)
        except ValueError:
            pass
        else:
            raise AssertionError('Unmasked positive-gated shared reward escaped audit: ' + name)
        term.params['source_func'] = original
    return 'passed: Stage4/6/7/9 mask+gate order, repeat application, serialization, descent/flat parity'


def source_reward(env, scale):
    return scale * torch.arange(1, env.num_envs + 1, dtype=torch.float32)


def native_reward_contract_check():
    """Run the actual Stage9 profile body with CPU-only config containers.

    Seed unchanged terms from the real launch JSON and replay the inherited
    baseline weights for conflicting terms, rather than pass saved weights
    straight back into the validator. No checkpoint or simulator is opened.
    """
    package = ROOT / 'source/rl_training/rl_training'
    profiles = package / 'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles'
    run = ROOT / 'logs/rsl_rl/deeprobotics_m20_stair_teacher_v1_stage2/2026-10-28_17-00-37_ascent_entry_v1_stage11_short_tread_rear'
    prior = json.loads((run / 'params/launch_manifest.json').read_text())
    baseline = json.loads((profiles / 'data/v1_stop_baseline.json').read_text())
    layout = loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_layout')
    objectives = loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_objectives')
    evidence = loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_evidence')
    actor = loader.load_module('models.stair_path_actor')

    def term_cfg(**kwargs):
        return NS(params={}, **kwargs) if 'params' not in kwargs else NS(**kwargs)

    weights = dict(prior['reward_weights'])
    for name in audit.CONFLICTING_REWARDS:
        if name in baseline['reward_weights']:
            weights[name] = baseline['reward_weights'][name]
    env = NS(rewards=NS(**{name: term_cfg(weight=weight, func=source_reward)
                          for name, weight in weights.items()}),
             scene=NS(terrain=NS(terrain_generator=NS(sub_terrains={
                 name: NS() for name in layout.GENERAL_TERRAINS}))),
             commands=NS(base_velocity=NS()), events=NS(randomize_reset_base=NS()),
             terminations=NS(), curriculum=NS())
    scope = dict(
        PROFILE='ascent_entry_v1_stage9', STAGE8_PROFILE='ascent_entry_v1_stage8_fixed',
        VERSION=pairing.VERSION, previous=NS(apply_profile=lambda *a, **k: None,
                                           configure_agent=lambda agent: {}),
        deepcopy=deepcopy, asdict=asdict, layout=layout, pairing=pairing,
        alternating=alternating, disable_conflicting_rewards=audit.disable_conflicting_rewards,
        audit_conflicting_rewards=audit.audit_conflicting_rewards,
        RewardTermCfg=term_cfg, CurriculumTermCfg=term_cfg, TerminationTermCfg=term_cfg,
        TerrainGeneratorCfg=NS, PathStairsCfg=NS, MeshPlaneTerrainCfg=NS,
        PathVelocityCommand=object, reset_path_root=source_reward,
        narrow_ascent_riser_contact_cost=source_reward, stair_teacher=NS(TUNING={}),
        HEIGHTS=loader.load_module(BASE + 'ascent_adaptive').HEIGHTS,
        PATH_DEPTHS=tuple(depth for depth, _ in layout.ASCENT_TASKS) + (layout.REFERENCE_TASK[0],)
                    + (.30,) * layout.FLAT_COPIES
                    + tuple(depth for depth, _ in layout.DESCENT_TASKS) + (.30,) * len(layout.GENERAL_COLUMNS),
        TERRAIN_COLUMNS=layout.TERRAIN_COLUMNS, ASCENT_REFERENCE_COLUMN=layout.ASCENT_REFERENCE_COLUMN,
        PathConfig=loader.load_module('control.stair_path').PathConfig,
        PATH_VERSION=loader.load_module('control.stair_path').PATH_VERSION,
        path=loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_mdp'),
        progressive=loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_course'),
        random_grid_terrain=source_reward,
        REWARD_VERSION=objectives.REWARD_VERSION, REWARD_WEIGHTS=objectives.REWARD_WEIGHTS,
        EVIDENCE_SETTINGS=evidence.TRANSITION_SETTINGS,
        HEIGHT_PRACTICE_SETTINGS=loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_height_practice').SETTINGS,
        ReadinessConfig=loader.load_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_readiness').ReadinessConfig,
        DESCENT_LIMITS=actor.DESCENT_LIMITS, DESCENT_WARMUP=actor.DESCENT_WARMUP,
        DESCENT_RAMP=actor.DESCENT_RAMP,
    )
    path = profiles / 'stair_path_profile.py'
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ('apply_profile', 'configure_agent')]
    for node in functions:
        # The only in-body import obtains the simulator's stair tuning dict.
        node.body = [stmt for stmt in node.body if not isinstance(stmt, ast.ImportFrom)]
        for block in ast.walk(node):
            if isinstance(block, ast.ImportFrom):
                assert block.names[0].name == 'stair_teacher'
                block.module = None
        # Replace that import with the explicitly supplied tuning container.
        class NoSimulatorImport(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return ast.copy_location(ast.Pass(), node)
        node = NoSimulatorImport().visit(node)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), scope)
    target = scope['apply_profile'](env, NS(algorithm=NS(), num_steps_per_env=24))
    resolved = {name: term.weight for name, term in vars(env.rewards).items() if term is not None}
    differences = {name: (prior['reward_weights'].get(name), resolved.get(name))
                   for name in set(prior['reward_weights']) | set(resolved)
                   if prior['reward_weights'].get(name) != resolved.get(name)}
    expected_changes={
        'ascent_rear_paired':(prior['reward_weights']['ascent_rear_paired'],1.5),
        'ascent_rear_landing_credit':(None,.12)}
    assert differences == expected_changes,differences
    assert env.stair_ascent_landing_quality_v12
    assert env.rewards.stair_up_task_crossing.params['require_clean_landing']
    validator = loader.load_module('algorithms.initialization.stair_path_init').validate_ascent_pairing_resume
    validator(prior,target,resolved,migrate=True,source='v11')
    native=dict(prior,refinement_version=pairing.VERSION,ascent_pairing=deepcopy(pairing.SETTINGS),reward_weights=resolved)
    # Retain strict rejection for every reward, including the four terms
    # that accidentally acquired baseline weights in the broken profile.
    for name in resolved:
        changed = dict(resolved)
        changed[name] += 1.
        try:
            validator(native, target, changed)
        except ValueError:
            pass
        else:
            raise AssertionError('Native reward drift escaped validation: ' + name)
    return f'passed: actual Stage9 v12 profile, only 2 approved rear reward weight changes; all {len(resolved)} native weights protected'


if __name__ == '__main__':
    print(metric_split_check())
    print(reward_audit_check())
    print(native_reward_contract_check())
