"""Explicitly remove inherited cross-tread alternation from the paired profile.

The resolved reward table can be checked before launching Kit. Historical paired
profiles disabled these terms globally, including joint_mirror and shared stair
terms. Retain their zero weights on resume. Ascent-only masking of active shared
terms is an explicit option; physical safety, task crossing and slip are retained.
"""

import torch

CONFLICTING_REWARDS = {
    'stair_front_wheel_touchdown': 'front_event is an opposite-side next-riser transition, not generic contact',
    'stair_front_alternation': 'opposite front wheel on the next physical tread',
    'stair_up_step_completion': 'one alternating front landing per physical tread',
    'stair_up_rear_step_completion': 'one alternating rear landing per physical tread',
    'stair_up_order_violation_cost': 'old alternation state treats paired same-tread support as a violation',
    'stair_up_same_tread_cost': 'penalizes both wheels of an axle supporting the same tread',
    'stair_up_same_tread_dwell_cost': 'penalizes time spent in paired same-tread support',
    'stair_rear_style_refund': 'withdraws rear style credit after the second rear wheel catches up on the same tread',
    'stair_rear_transition_recovery': 'prefers returning from paired rear support to single-wheel alternation',
    'stair_front_single_support': 'persistent single-front-support preference',
    'stair_diagonal_support_symmetry': 'forces diagonal support independent of paired-tread progress',
    'stair_front_wheel_lift': 'lift target selects the old expected alternating front side',
    'stair_rear_wheel_lift': 'preparation target selects the old expected alternating rear side',
    'joint_mirror': 'couples diagonal front/rear joint poses independent of local support geometry',
    'ascent_entry_phase': 'old FL-clock phase template is replaced by side-neutral paired events and cadence',
}

# These inherited terms also contain descent behavior. Active descent behavior
# may be retained explicitly, but must not be restored from baseline weights
# when the source paired profile recorded these terms as globally disabled.
DESCENT_PRESERVED_REWARDS = {
    'stair_front_wheel_touchdown',
    'stair_front_single_support',
    'stair_diagonal_support_symmetry',
}


def ascent_disabled_reward(env, source_func, source_params):
    """Serializable ascent-only mask, compatible with the positive reward gate."""
    from .ascent_alternating_mdp import resolve_reward_params
    from . import ascent_adaptive as adaptive

    resolve_reward_params(env, source_params)
    value = source_func(env, **source_params)
    return torch.where(adaptive.ascent_mask(env), torch.zeros_like(value), value)


def _ascent_disabled(func, params):
    # Stage6/7 wrap the Stage4 mask in the common positive reward gate.
    # Follow only that known gate: arbitrary source_func parameters do not
    # prove that a callable actually applies the ascent mask.
    from .ascent_alternating_mdp import is_positive_reward_gate

    while is_positive_reward_gate(func):
        func, params = params.get('source_func'), params.get('source_params', {})
    return func is ascent_disabled_reward


def audit_conflicting_rewards(rewards):
    result = {}
    for name, reason in CONFLICTING_REWARDS.items():
        term = rewards.get(name) if isinstance(rewards, dict) else getattr(rewards, name, None)
        weight = None if term is None else term.weight
        masked = bool(term is not None and _ascent_disabled(
            getattr(term, 'func', None), getattr(term, 'params', {})))
        if name in DESCENT_PRESERVED_REWARDS:
            if term is not None and weight not in (None, 0, 0.) and not masked:
                raise ValueError(f'Paired ascent profile did not mask shared reward {name}')
        elif weight not in (None, 0, 0.):
            raise ValueError(f'Paired ascent profile re-enabled conflicting reward {name}: {weight}')
        result[name] = {'weight': weight, 'ascent_masked': masked, 'reason': reason}
    return result


def disable_conflicting_rewards(rewards, preserve_descent=False):
    for name in CONFLICTING_REWARDS:
        term = rewards.get(name) if isinstance(rewards, dict) else getattr(rewards, name, None)
        if term is None:
            continue
        if preserve_descent and name in DESCENT_PRESERVED_REWARDS:
            if not _ascent_disabled(term.func, term.params):
                term.params = dict(source_func=term.func, source_params=term.params)
                term.func = ascent_disabled_reward
        else:
            term.weight = 0.
    return audit_conflicting_rewards(rewards)
