"""Source locations and legacy module names for manifests and old checkpoints.

New code imports canonical packages directly. The basename map is shared only
by compatibility loading and source provenance tools; it never copies code.
"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]

LEGACY_MODULES = {
    'ascent_adaptive': 'tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_adaptive',
    'ascent_adaptive_actor': 'models.ascent_adaptive_actor',
    'ascent_adaptive_demo': 'algorithms.initialization.ascent_adaptive_demo',
    'ascent_adaptive_init': 'algorithms.initialization.ascent_adaptive_init',
    'ascent_adaptive_mdp': 'tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_adaptive_mdp',
    'ascent_adaptive_profile': 'tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.ascent_adaptive_profile',
    'ascent_entry_actor': 'models.ascent_entry_actor',
    'ascent_entry_env': 'tasks.manager_based.locomotion.velocity.envs.ascent_entry_env',
    'ascent_entry_mdp': 'tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_entry_mdp',
    'ascent_entry_profile': 'tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.ascent_entry_profile',
    'ascent_entry_stage2': 'tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_entry_stage2',
    'ascent_entry_stage2_init': 'algorithms.initialization.ascent_entry_stage2_init',
    'ascent_entry_stage2_profile': 'tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.ascent_entry_stage2_profile',
    'ascent_landing_actor': 'models.ascent_landing_actor',
    'ascent_landing_init': 'algorithms.initialization.ascent_landing_init',
    'ascent_landing_mdp': 'tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_landing_mdp',
    'ascent_landing_profile': 'tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.ascent_landing_profile',
    'ascent_reward_audit': 'tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_reward_audit',
    'ascent_sequence': 'tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_sequence',
    'ascent_sequence_actor': 'models.ascent_sequence_actor',
    'ascent_sequence_init': 'algorithms.initialization.ascent_sequence_init',
    'ascent_sequence_mdp': 'tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_sequence_mdp',
    'ascent_sequence_profile': 'tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.ascent_sequence_profile',
    'fast_logging': 'utils.fast_logging',
    'm20_preflight': 'utils.m20_preflight',
    'metric_aggregation': 'utils.metric_aggregation',
    'motion_residual_actor': 'models.motion_residual_actor',
    'stair_guarded_ppo': 'algorithms.stair_guarded_ppo',
    'stop_aware_actor': 'models.stop_aware_actor',
    'stop_branch': 'models.stop_branch',
    'stop_coverage': 'tasks.manager_based.locomotion.velocity.mdp.stop.stop_coverage',
    'teacher_diagnostics': 'utils.teacher_diagnostics',
    'v1_stop_profile': 'tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.v1_stop_profile',
}


def source_path(name):
    """Resolve a historical basename or a canonical relative source path."""
    path = Path(name)
    if len(path.parts) == 1:
        module = LEGACY_MODULES.get(path.stem)
        if module:
            path = Path(*module.split('.')).with_suffix('.py')
        elif path.name in ('ascent_reference.json', 'v1_stop_baseline.json'):
            path = Path('tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/data') / path.name
    return PACKAGE_ROOT / path
