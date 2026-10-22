#!/usr/bin/env python3
"""Copy Stage7 network weights exactly; attach fixed-input path-control metadata."""
import argparse, copy, hashlib, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output; checkpoints are never overwritten')
    torch.set_num_threads(1)
    init = d.training_module('algorithms.initialization.stair_path_init')
    provenance = init.validate_source(args.source)
    source = torch.load(args.source, map_location='cpu', weights_only=True)
    prior = source['m20_training_manifest']
    if prior['m20_profile'] != 'ascent_entry_v1_stage7': parser.error('--source must be Stage7')
    state = copy.deepcopy(source['actor_state_dict'])
    model = d.training_module('models.stair_path_actor')
    path = d.training_module('control.stair_path')
    critic = copy.deepcopy(source['critic_state_dict'])
    assert state['unified_mlp.0.weight'].shape == (256,247)
    assert critic['mlp.0.weight'].shape == (512,300)
    manifest = copy.deepcopy(prior)
    manifest.update(m20_profile=model.PROFILE, refinement_version=model.VERSION,
        control_profile='legacy_v1_path_v1', frozen_motion_state_sha256=init.frozen_digest(state),
        actor_model_config=dict(class_name='rl_training.models.stair_path_actor:StairPathMLPModel',
            unified_hidden_dims=[256,128], residual_hidden_dims=[128,64]),
        actor_input_groups={'policy':244,'motion_control':3},
        critic_input_groups={'critic':285,'motion_control':3,'ascent_control':12},
        stair_path_version=path.PATH_VERSION, stair_path_config=path.PathConfig().to_dict(),
        curriculum_terrain_columns=prior.get('curriculum_terrain_columns',10),
        descent_limits_action_units=model.DESCENT_LIMITS,
        path_migration=dict(source_checkpoint=str(args.source.resolve()), source_sha256=provenance['sha256'],
            source_profile=prior['m20_profile'], unified_input_columns_added=0,
            network_state_copied_exactly=True, descent_authority_schedule=[30,800],
            mlp_count_before=3, mlp_count_after=3, optimizer='fresh_on_training_start'))
    result = dict(actor_state_dict=state,critic_state_dict=critic,m20_training_manifest=manifest)
    for key in ('ascent_adaptive_curriculum','ascent_adaptive_terrain_levels'):
        if key in source: result[key] = copy.deepcopy(source[key])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+'.partial')
    torch.save(result, temporary); temporary.replace(args.output)
    init.validate_source(args.output)
    # Both actor and critic tensors are exact copies, with identical state keys.
    for group in ('actor_state_dict', 'critic_state_dict'):
        assert set(result[group]) == set(source[group])
        assert all(torch.equal(value,source[group][key]) for key,value in result[group].items())
    actor = d.load_actor(args.output); old_actor = d.load_actor(args.source)
    torch.manual_seed(42); n = 128
    obs = {'policy':torch.randn(n,244), 'motion_control':torch.tensor([[.5,0.,0.]]*n)}
    with torch.no_grad():
        expected, actual = old_actor(obs), actor(obs)
    maximum = float((actual-expected).abs().max())
    if maximum > 2.e-5: raise RuntimeError(f'Migration action error {maximum}')
    manifest['path_migration']['max_action_error_random_observations'] = maximum
    result['m20_training_manifest'] = manifest
    torch.save(result, temporary); temporary.replace(args.output)
    print(json.dumps(dict(output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        migration=manifest['path_migration']), indent=2))


if __name__ == '__main__': main()
