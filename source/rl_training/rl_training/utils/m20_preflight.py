"""M20 launch checks without Isaac/Kit startup or simulator imports."""
from __future__ import annotations
import ast
import hashlib
import math
from pathlib import Path

import numpy as np

REFINEMENT_VERSION = 'm20_engineering_v5_20261016'


def source_constants(path: Path, names: set[str]) -> dict:
    """Evaluate selected constant expressions from this trusted checkout only."""
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                result[name] = eval(compile(ast.Expression(node.value), str(path), 'eval'),
                                    {'np': np, 'math': math, **result})
    missing = names - result.keys()
    if missing:
        raise ValueError(f'{path}: missing contract constants: {sorted(missing)}')
    return result


def validate_control_contract(tuning: dict, deploy_path: Path) -> dict:
    mapping = {'translation_threshold': 'COMMAND_THRESHOLD', 'yaw_threshold': 'COMMAND_THRESHOLD',
               'wheel_radius': 'WHEEL_RADIUS', 'heading_hold_filter_tau': 'HEADING_HOLD_FILTER_TAU',
               'heading_hold_exit_deadband': 'HEADING_HOLD_EXIT_DEADBAND',
               'heading_hold_rate_enter': 'HEADING_HOLD_RATE_ENTER', 'heading_hold_rate_exit': 'HEADING_HOLD_RATE_EXIT',
               'heading_hold_slew_flat': 'HEADING_HOLD_SLEW_FLAT', 'heading_hold_slew_stairs': 'HEADING_HOLD_SLEW_STAIRS',
               'zero_hold_brake_grace_s': 'STAIR_STOP_BRAKE_GRACE',
               'stair_stop_position_kp': 'STAIR_STOP_KP', 'stair_stop_velocity_kd': 'STAIR_STOP_KD',
               'stair_stop_position_deadband': 'STAIR_STOP_POSITION_DEADBAND',
               'stair_stop_velocity_deadband': 'STAIR_STOP_SPEED_DEADBAND',
               'stair_stop_max_speed': 'STAIR_STOP_MAX_SPEED', 'stair_stop_slew': 'STAIR_STOP_SLEW'}
    for direction in ('ascent', 'descent'):
        for field in ('kp', 'kd', 'max_lateral_speed', 'lateral_deadband', 'speed_deadband', 'lateral_slew'):
            mapping[f'{direction}_path_{field}'] = f'{direction.upper()}_PATH_{field.upper()}'
    names = set(mapping.values()) | {'HEADING_HOLD_FLAT', 'HEADING_HOLD_STAIRS', 'ACTOR_OBS_DIM', 'ACTION_DIM', 'TELEMETRY_VERSION'}
    constants = source_constants(deploy_path, names)
    mismatches = []
    for key, constant in mapping.items():
        if key not in tuning or not math.isclose(float(tuning[key]), float(constants[constant]), rel_tol=1.e-6, abs_tol=1.e-8):
            mismatches.append(f'{key}: training={tuning.get(key)!r}, deploy={constants[constant]!r}')
    for prefix, name in (('', 'HEADING_HOLD_FLAT'), ('stair_', 'HEADING_HOLD_STAIRS')):
        expected = [tuning[prefix+'heading_hold_'+key] for key in ('kp', 'kd', 'max_yaw_rate', 'deadband')]
        if not np.allclose(expected, constants[name], rtol=1.e-6, atol=1.e-8):
            mismatches.append(f'{name}: training={expected}, deploy={constants[name]}')
    if constants['ACTOR_OBS_DIM'] != 244 or constants['ACTION_DIM'] != 16:
        mismatches.append('M20 actor must remain 244D -> 16D')
    if constants['TELEMETRY_VERSION'] != REFINEMENT_VERSION:
        mismatches.append(f'deployment version must be {REFINEMENT_VERSION}')
    if tuning['stair_stop_max_speed'] > tuning['translation_threshold']:
        mismatches.append('stop recovery exceeds the PPO preservation/standing threshold')
    if mismatches:
        raise ValueError('M20 control contract mismatch:\n'+'\n'.join(mismatches))
    return {'version': REFINEMENT_VERSION, 'checked_parameters': len(mapping)+8,
            'deploy_sha256': hashlib.sha256(deploy_path.read_bytes()).hexdigest()}


def validate_actor_checkpoint(saved: dict, *, require_version: bool = False) -> dict:
    import torch
    state = saved.get('actor_state_dict')
    if not isinstance(state, dict):
        raise ValueError('M20 checkpoint has no actor_state_dict')
    for key, shape in (('mlp.0.weight', (512,244)), ('mlp.6.weight', (16,128))):
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f'M20 checkpoint actor layout mismatch at {key}')
    if any(not torch.isfinite(value).all() for value in state.values() if isinstance(value,torch.Tensor)):
        raise ValueError('M20 checkpoint contains non-finite actor parameters')
    manifest = saved.get('m20_training_manifest') or {}
    version = manifest.get('refinement_version')
    if require_version and version != REFINEMENT_VERSION:
        raise ValueError(f'Full resume uses {version!r}, current rewards/control use {REFINEMENT_VERSION!r}; '
                         'use --init_actor_from for a fresh critic/optimizer after changing training semantics')
    return {'checkpoint_version': version, 'runtime_version': REFINEMENT_VERSION,
            'version_match': version == REFINEMENT_VERSION}
