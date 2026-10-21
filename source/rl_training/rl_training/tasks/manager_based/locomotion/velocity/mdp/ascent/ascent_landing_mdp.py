"""Dense pre-contact landing guidance, keeping the stage2 200Hz tread ledger."""
from __future__ import annotations

import torch
from .ascent_entry_mdp import smooth01, context as original_context


def landing_guidance(wheel_pos, wheel_vz, target, unresolved, first_edge, floor, active,
                     depth=.30, rise=.15):
    """Spatial lift/clear/descend waypoints for the two front wheels' first two treads.

    Negative costs only: no per-frame support bonus, and no reward for waiting.
    Contact confirmation and once-only credit remain the stage2 ledger's job.
    """
    edge = first_edge[:, None]+(target-1.)*depth
    offset = wheel_pos[..., 0]-edge
    opportunity = smooth01((offset+.30)/.20)
    opportunity *= ((offset < depth+.09) & unresolved & (target <= 2) & (target >= 1)).float()
    opportunity *= active[:, None]
    landing = smooth01((offset-.01)/.07)
    top = floor[:, None]+target*rise
    desired_bottom = top+.04*(1.-landing)
    height_error = wheel_pos[..., 2]-.09-desired_bottom
    # 4cm extra clearance before crossing the edge, then descend to support
    # only after the wheel center is safely over the tread.
    height = (height_error/.10).square().clamp_max(4.)*opportunity
    x_error = (.07-offset).clamp_min(0.)+(offset-(depth-.07)).clamp_min(0.)
    horizontal = (x_error/.25).square().clamp_max(4.)*opportunity
    desired_vz = (-3.*height_error).clamp(-.45, .35)
    velocity = ((wheel_vz-desired_vz)/.60).square().clamp_max(4.)*opportunity
    return {'height': height.mean(-1), 'horizontal': horizontal.mean(-1),
            'velocity': velocity.mean(-1), 'opportunity': opportunity,
            'height_error': height_error, 'desired_bottom': desired_bottom, 'desired_vz': desired_vz}


def guidance(env):
    observer = env._ascent_entry_observer
    if getattr(observer, 'landing_context_key', -1) == env.common_step_counter:
        return observer.landing_cached
    original = original_context(env)
    unresolved = ~(observer.ledger.covered[:, :2] | observer.ledger.missed[:, :2])
    next_index = unresolved.long().argmax(-1)
    target = next_index+1
    data = env.scene['robot'].data
    result = landing_guidance(data.body_pos_w[:, observer.wheels[:2]],
        data.body_com_lin_vel_w[:, observer.wheels[:2], 2], target, unresolved.any(-1),
        env.scene.env_origins[:, 0]+1.25, observer.first_floor, original['active'],
        env.cfg.ascent_entry_depth, env.cfg.ascent_entry_height)
    observer.landing_context_key, observer.landing_cached = env.common_step_counter, result
    return result


def height_cost(env): return guidance(env)['height']
def horizontal_cost(env): return guidance(env)['horizontal']
def velocity_cost(env): return guidance(env)['velocity']


def phase_cost(env):
    original = original_context(env)
    observer = env._ascent_entry_observer
    # Reference timing came from continuous climbing, not a clean first-step
    # demonstration. Introduce it smoothly after both front wheels pass tread2.
    x = env.scene['robot'].data.body_pos_w[:, observer.wheels[:2], 0].amin(-1)
    progress = (x-env.scene.env_origins[:, 0]-1.25)/env.cfg.ascent_entry_depth
    return original['phase_cost']*smooth01((progress-1.)/1.5)


def metrics(env, env_ids):
    observer = getattr(env, '_ascent_entry_observer', None)
    if observer is None or not hasattr(observer, 'landing_cached'):
        return {}
    info = observer.landing_cached
    ids = slice(None) if env_ids is None else env_ids
    weight = info['opportunity'][ids]
    return {'target_height_error_abs_m': (info['height_error'][ids].abs()*weight).sum()/weight.sum().clamp_min(1.),
            'opportunity_samples': (weight > 0).float().sum()}
