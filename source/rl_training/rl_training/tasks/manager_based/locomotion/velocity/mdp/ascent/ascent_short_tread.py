"""Ascent-only short-tread riser preparation; no actor or simulator dependency."""
import torch
from .ascent_entry_mdp import smooth01

SETTINGS = dict(version='ascent_short_tread_20261028_v1', depths_m=[.20,.25],
    axle='rear_only', clearance_margin_m=.04, wheel_radius_m=.09,
    preparation='legal_swing_only_continuous_pseudo_huber_no_20cm_deficit_plateau',
    low_approach='all_unresolved_rear_wheels_positive_world_x_velocity_near_riser',
    approach_window_m=[-.18,.07], clearance_reward_weight=-.30,
    constraints='unchanged_coverage_pairing_simultaneous_swing_and_tread3_anchor')

# Keep the historical sum, but retain the worse rear wheel so one wheel's
# successful lift cannot hide the other wheel's clearance deficit.
WORST_REAR_WHEEL_MULTIPLIER = .75


def aggregate_rear_cost(cost):
    """Aggregate two rear-wheel costs while retaining the worse wheel."""
    if cost.ndim != 2 or cost.shape[-1] != 2:
        raise ValueError('rear cost must have shape [batch, 2]')
    return cost.sum(-1) + WORST_REAR_WHEEL_MULTIPLIER * cost.amax(-1)


def rear_riser_cost(offset, bottom, top, depth, enabled, permission, velocity_x):
    """Prefer legal lift before crossing; never reward waiting or lifting high.

    A waiting follower has no lift objective. It still pays for moving into a
    riser below clearance. Wheel COM velocity excludes pure in-place spin,
    whose contact force/scrubbing remains covered by the existing face cost.
    """
    short = (depth[:,None] <= .25+1.e-6) & enabled
    blend = smooth01((offset-.01)/.06)
    near = smooth01((offset+.35)/.25)*(offset < depth[:,None]+.09)
    deficit = (top+.04-bottom).clamp_min(0.)
    # Unlike clipped squared error, this keeps distinguishing a low wheel
    # from one that has started to clear even on 18--25 cm risers.
    preparation = 4.*(torch.sqrt(1.+(deficit/.10).square())-1.)
    preparation *= (1.-blend)*near*permission*short
    face_near = smooth01((offset+.18)/.08)*(1.-blend)
    low = ((top+.02-bottom).clamp_min(0.)/.10).clamp_max(3.)
    inward = 2.*(torch.sqrt(1.+(velocity_x.clamp_min(0.)/.30).square())-1.)
    approach = inward*low*face_near*short
    diagnostic = dict(near=short & (offset >= -.18) & (offset < .07),
        deficit=deficit, permitted=permission & short,
        low_approach=(velocity_x > .05) & (bottom < top+.02)
                     & (offset >= -.18) & (offset < .07) & short)
    return preparation+approach, diagnostic
