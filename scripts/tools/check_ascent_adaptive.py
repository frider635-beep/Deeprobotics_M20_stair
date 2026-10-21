#!/usr/bin/env python3
"""CPU counterexamples, terrain consistency and real PPO integration for stage5."""
import argparse
import ast
import copy
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
import trimesh
from tensordict import TensorDict
from check_ascent_entry_stage2 import ROOT, deployment, local_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture-checkpoint', type=Path, required=True)
    parser.add_argument('--demonstration', type=Path, nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(42)
    actor_module = deployment.ascent_adaptive_module()
    core = deployment.training_module('ascent_adaptive')
    init = deployment.training_module('ascent_adaptive_init')
    mdp = deployment.training_module('ascent_adaptive_mdp')
    provenance = init.validate_source(args.source_checkpoint)
    saved = torch.load(args.source_checkpoint, map_location='cpu', weights_only=True)

    def ledger(): return core.StrictSequenceLedger(1, 'cpu', targets=4)
    def tick(book, levels, support=None, positions=None, goals=2):
        lev = torch.tensor([levels], dtype=torch.long)
        support = lev > 0 if support is None else torch.tensor([support], dtype=torch.bool)
        if positions is None: positions = (lev.float()-1.)*.30+.12
        else: positions = torch.tensor([positions])
        return book.update_strict(lev, support, positions, torch.zeros(1), torch.tensor([True]), .005, torch.tensor([goals]))

    for lead in [0, 1]:
        book = ledger(); levels = [0]*4
        for step in [1, 2]:
            levels[lead] = step
            for _ in range(8): event = tick(book, levels)
            assert not book.failed.any()
            levels[1-lead] = step
            pairs = 0.
            for _ in range(10): pairs += tick(book, levels)['paired'].item()
            assert pairs == 1. and not book.failed.any()
        for step in [1, 2]:
            levels[2:] = [step, step]
            for _ in range(8): tick(book, levels)
        assert book.covered[:, :, :2].all() and book.pair_paid[:, :2].all()
        for _ in range(20):
            event = tick(book, levels)
            assert event['ordered'].sum() == event['paired'].sum() == event['failure'].sum() == 0.

    # Early preparation is enabled although the mate has not yet supported 1.
    book = ledger()
    for _ in range(8): tick(book, [1, 0, 0, 0])
    target, enabled, pair = book.front_targets(torch.tensor([True]))
    assert target.tolist() == [[2, 1]] and enabled.all() and pair.item() == 1
    # Actually landing on 2 before the mate's 1 must fail, not earn a bonus.
    failed = 0.
    for _ in range(8):
        event = tick(book, [2, 0, 0, 0]); failed += event['failure'].item()
    assert failed == 1. and book.failed.item()
    for _ in range(20):
        event = tick(book, [3, 3, 1, 1])
        assert event['ordered'].sum() == event['paired'].sum() == event['dwell'].sum() == event['failure'].sum() == 0.
    book = ledger()
    failures = 0.
    for _ in range(10): failures += tick(book, [1, 1, 0, 0])['failure'].item()
    assert failures == 1. and book.failed.item(), 'Simultaneous fronts wrongly accepted'
    book = ledger()
    event = tick(book, [0]*4, positions=[.7, -.4, -.4, -.4])
    assert book.failed.item() and event['failure'].item() == 1.
    assert event['ordered'].sum() == event['paired'].sum() == 0.
    book.reset(slice(None)); assert not book.failed.any()

    # Progress before/in the landing patch is escrowed, never independently paid.
    target = torch.ones(1, 2, dtype=torch.long); inside = torch.ones(1, 2, dtype=torch.bool)
    new = torch.zeros(1, 4, 4, dtype=torch.bool)
    assert book.landing_credit(target, torch.full((1, 2), 4.), inside, new).item() == 0.
    assert book.landing_credit(target, torch.full((1, 2), 2.), inside, new).item() == 0.
    new[:, :2, 0] = True
    assert book.landing_credit(target, torch.full((1, 2), 2.), inside, new).item() == 2.
    assert book.landing_credit(target, torch.full((1, 2), 2.), inside, new).item() == 0.
    book.failed[:] = True
    assert book.landing_credit(target, torch.zeros(1, 2), inside, new).item() == 0.

    # Batched target height/depth, and correct gradient direction inside tread.
    positions = torch.tensor([[[.12, .1, .19], [.12, -.1, .19]],
                              [[.17, .1, .29], [.17, -.1, .29]]], requires_grad=True)
    guide = core.landing_guidance(positions, torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.bool), torch.zeros(2), torch.zeros(2),
        torch.tensor([.25, .35]), torch.tensor([.10, .20]))
    assert torch.allclose(guide['height'], torch.zeros(2), atol=1.e-6)
    high = positions.detach().clone(); high[..., 2] += .13; high.requires_grad_()
    high_guide = core.landing_guidance(high, torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.bool), torch.zeros(2), torch.zeros(2), torch.tensor([.25, .35]), torch.tensor([.10, .20]))
    assert (torch.autograd.grad(high_guide['height'].sum(), high)[0][..., 2] > 0).all()

    state = core.ContactCurriculum(2, 'cpu'); ids = torch.arange(2); refs = torch.tensor([False, True])
    for skill in range(3):
        for _ in range(8):
            up, _ = state.update(ids, torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.long), refs)
            assert not up.any()
        assert (state.skill == skill+1).all()
    for _ in range(8): up, _ = state.update(ids, torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.long), refs)
    assert up.tolist() == [True, False]
    assert state.skill.tolist() == [1, 3]
    for _ in range(3): _, down = state.update(ids, torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.long), refs)
    assert down.tolist() == [True, False]
    old = state.state_dict(); state.update(ids, torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.long), refs)
    assert all(torch.equal(v, state.state_dict()[k]) for k, v in old.items())
    restored = core.ContactCurriculum(2, 'cpu'); restored.load_state_dict(state.state_dict())
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in state.state_dict().items())
    # Geometry comes from current rows and columns, including after promotion.
    geometry_env = NS(num_envs=2, device='cpu', cfg=NS(ascent_adaptive_geometry=True,
        ascent_adaptive_height_grid=[[.10]*10, [.20]*10],
        ascent_adaptive_depth_grid=[[.25]*10, [.35]*10]),
        scene=NS(terrain=NS(terrain_levels=torch.tensor([0, 1]), terrain_types=torch.tensor([0, 5]))))
    heights, depths = core.geometry(geometry_env)
    assert torch.allclose(heights, torch.tensor([.10, .20])) and torch.allclose(depths, torch.tensor([.25, .35]))
    geometry_env.scene.terrain.terrain_levels[:] = torch.tensor([1, 0])
    assert torch.allclose(core.geometry(geometry_env)[0], torch.tensor([.20, .10]))
    assert mdp.critic_state(geometry_env).shape == (2, 12)
    assert torch.isfinite(mdp.critic_state(geometry_env)).all()
    # Execute actual mesh functions without importing Kit/Isaac registration.
    files = ROOT/'source/rl_training/rl_training'
    scope = dict(np=np, trimesh=trimesh, HEIGHTS=core.HEIGHTS, depth_for_row=core.depth_for_row)
    for filename, function in [('tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_entry_profile.py', 'straight_stairs'), ('tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_adaptive_profile.py', 'adaptive_stairs')]:
        tree = ast.parse((files/filename).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(files/filename), 'exec'), scope)
    class Config(NS):
        def replace(self, **kwargs): return Config(**(vars(self) | kwargs))
    for row, height in enumerate(core.HEIGHTS):
        for column, depth in enumerate(core.DEPTHS[:8]):
            cfg = Config(size=(12., 5.), tread_depth=depth, step_height=.15, step_count=18,
                         reference=column in core.REFERENCE_COLUMNS)
            expected_height = .15 if cfg.reference else height
            for fraction in [.001, .999]:
                meshes, origin = scope['adaptive_stairs']((row+fraction)/len(core.HEIGHTS), cfg)
                assert np.isclose(meshes[1].bounds[1, 2], expected_height)
                assert np.isclose(meshes[18].bounds[1, 2], expected_height*18)
                expected_depth = .30 if cfg.reference else core.depth_for_row(row, depth)
                assert np.isclose(meshes[1].extents[0], expected_depth)
                assert meshes[-1].extents[0] > 0.

    obs = TensorDict({'policy': torch.randn(32, 244), 'motion_control': torch.zeros(32, 3)}, batch_size=[32])
    obs['motion_control'][:, 0] = .7
    profile = torch.tensor([.7]*10+[.55]*3+[.4]*4)
    obs['policy'][:, 57:] = profile[None, None].expand(32, 11, 17).reshape(32, 187)
    def make_actor(batch):
        return actor_module.AscentAdaptiveMLPModel(batch, {'actor': ['policy', 'motion_control']}, 'actor', 16,
            hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False,
            distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'log'})
    actor = make_actor(obs); init.load_initialized_actor(actor, saved)
    source = deployment.load_actor(args.source_checkpoint)
    with torch.no_grad(): assert torch.equal(actor(obs), source(obs))
    assert all(k.startswith('sequence_mlp.') for k, v in actor.named_parameters() if v.requires_grad)
    actor(obs, stochastic_output=True)
    assert (actor.distribution.std > actor.distribution.log_std_param.exp()).any()
    zero = obs.clone(); zero['motion_control'].zero_()
    actor(zero, stochastic_output=True)
    assert torch.allclose(actor.distribution.std, actor.distribution.log_std_param.exp().expand_as(actor.distribution.std))
    fixture = copy.deepcopy(saved['m20_training_manifest'])
    fixture.update(m20_profile='ascent_entry_v1_stage5', refinement_version='m20_ascent_adaptive_20261021_v1_stage5',
        frozen_stage3_state_sha256=actor.sequence_source_digest(), frozen_stage2_state_sha256=actor.source_state_digest(),
        control_profile='legacy_v1', test_fixture_only=True)
    fixture['actor_model_config'].update(class_name='rl_training.models.ascent_adaptive_actor:AscentAdaptiveMLPModel')
    fixture['critic_input_groups'] = dict(critic=285, motion_control=3, ascent_control=12)
    fixture_critic = dict(saved['critic_state_dict'])
    fixture_critic['mlp.0.weight'] = torch.cat((fixture_critic['mlp.0.weight'], torch.zeros(512, 12)), -1)
    args.fixture_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(actor_state_dict=actor.state_dict(), critic_state_dict=fixture_critic, m20_training_manifest=fixture), args.fixture_checkpoint)
    native = deployment.load_actor(args.fixture_checkpoint)
    with torch.no_grad(): assert torch.equal(native(obs), actor(obs))

    # Actual PPO losses/log-probabilities with conditional Gaussian exploration.
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    guarded = local_module('stair_guarded_ppo')
    batch = obs.clone(); batch['critic'] = torch.randn(32, 285); batch['ascent_control'] = torch.randn(32, 12)
    learner = make_actor(batch); init.load_initialized_actor(learner, saved)
    critic = MLPModel(batch, {'critic': ['critic', 'motion_control', 'ascent_control']}, 'critic', 1,
        hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False)
    init.load_initialized_critic(critic, saved['critic_state_dict'])
    original_critic = MLPModel(batch, {'critic': ['critic', 'motion_control']}, 'critic', 1,
        hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False)
    original_critic.load_state_dict(saved['critic_state_dict'], strict=True)
    with torch.no_grad(): assert torch.allclose(critic(batch), original_critic(batch), atol=1.e-5)
    storage = RolloutStorage('rl', 32, 4, batch, [16], device='cpu')
    algorithm = guarded.StairGuardedPPO(learner, critic, storage, device='cpu', num_learning_epochs=1,
        num_mini_batches=2, schedule='fixed', learning_rate=3.e-5, critic_warmup_iterations=1,
        anchor_coefficient=.1, anchor_stair_weight=0., motion_regularization_coefficient=.01)
    terrain = NS(terrain_levels=torch.zeros(32, dtype=torch.long), terrain_types=torch.arange(32)%8,
                 terrain_origins=torch.randn(6, 10, 3), env_origins=torch.zeros(32, 3))
    fake_env = NS(num_envs=32, device='cpu', scene=NS(terrain=terrain),
                  cfg=NS(ascent_adaptive_training=True, ascent_adaptive_height_grid=[[.15]*10]*6), reset_count=0)
    def reset(): fake_env.reset_count += 1
    fake_env.reset = reset
    init.attach_curriculum_checkpoint(algorithm, fake_env)
    demos = deployment.training_module('ascent_adaptive_demo')
    demos.attach_demonstrations(algorithm, args.demonstration)
    demonstration_authority = demos.validate_demo_authority(algorithm)
    algorithm.set_reference(learner.state_dict()); ppo = []
    for iteration in range(3):
        before = learner.sequence_mlp[-1].weight.detach().clone()
        for _ in range(4):
            with torch.inference_mode():
                actions = algorithm.act(batch)
                algorithm.process_env_step(batch, torch.randn(32)-.02*actions.square().mean(-1), torch.zeros(32, dtype=torch.long), {})
        with torch.inference_mode(): algorithm.compute_returns(batch)
        losses = algorithm.update()
        assert all(math.isfinite(float(v)) for v in losses.values())
        changed = not torch.equal(before, learner.sequence_mlp[-1].weight)
        assert changed == (iteration > 0)
        assert learner.sequence_source_digest() == fixture['frozen_stage3_state_sha256']
        ppo.append(dict(iteration=iteration, sequence_updated=changed, source_frozen=True, finite=True))
        if args.demonstration and iteration > 0: assert losses['contact_demonstration'] > 0.
    state = mdp.curriculum_state(fake_env)
    state.skill[0] = 2; terrain.terrain_levels[0] = 2
    snapshot = algorithm.save()
    state.skill.zero_(); terrain.terrain_levels.zero_()
    # Resume into a fresh actor/algorithm, as the real training entry point
    # does; forward-pass autograd caches do not belong in a checkpoint.
    resumed_actor = make_actor(batch)
    resumed = guarded.StairGuardedPPO(resumed_actor, critic, storage, device='cpu', num_learning_epochs=1,
        num_mini_batches=2, schedule='fixed', learning_rate=3.e-5, critic_warmup_iterations=1,
        anchor_coefficient=.1, anchor_stair_weight=0., motion_regularization_coefficient=.01)
    init.attach_curriculum_checkpoint(resumed, fake_env)
    demos.attach_demonstrations(resumed)
    with tempfile.TemporaryDirectory(prefix='m20_adaptive_resume_') as folder:
        path = Path(folder)/'checkpoint.pt'; torch.save(snapshot, path)
        resumed.load(torch.load(path, map_location='cpu', weights_only=True))
    assert state.skill[0] == 2 and terrain.terrain_levels[0] == 2 and fake_env.reset_count == 1
    assert torch.equal(terrain.env_origins, terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
    if args.demonstration:
        assert resumed._contact_demo is not None and snapshot['ascent_contact_demonstrations'] is not None
    assert all(torch.equal(v, resumed.actor.state_dict()[k]) for k, v in learner.state_dict().items())
    result = dict(source=provenance, strict_sequence_either_lead=True,
        early_preparation_without_early_landing=True, miss_not_resolved_as_success=True,
        no_positive_sequence_credit_after_failure=True, simultaneous_not_success=True,
        landing_credit_requires_stable_contact=True, exact_mesh_metadata_96_checks=True,
        height_depth_batch_and_landing_gradient=True, curriculum_requires_contacts=True,
        critic_privileged_12d_and_zero_padded_initial_value=True,
        reference_geometry_retained=True, curriculum_state_roundtrip=True, actual_algorithm_checkpoint_hooks=True,
        contact_demonstrations_inside_real_ppo=bool(args.demonstration),
        contact_demonstration_authority=demonstration_authority,
        warm_start_and_native_loader_exact=True, conditional_exploration_finite=True,
        cpu_ppo_synthetic_integration=ppo, gpu_test_run=False,
        scope='CPU logic/mesh/real PPO on synthetic rewards; not Isaac startup or learned gait acceptance.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
