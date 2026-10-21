"""Explicitly remove inherited cross-tread alternation from the paired profile.

Pure Python so the resolved weight table can be checked before launching Kit.
Physical safety, forward task crossing and per-wheel slip are retained.
"""

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


def audit_conflicting_rewards(rewards):
    result = {}
    for name, reason in CONFLICTING_REWARDS.items():
        term = rewards.get(name) if isinstance(rewards, dict) else getattr(rewards, name, None)
        weight = None if term is None else term.weight
        if weight not in (None, 0, 0.):
            raise ValueError(f'Paired ascent profile re-enabled conflicting reward {name}: {weight}')
        result[name] = {'weight': weight, 'reason': reason}
    return result


def disable_conflicting_rewards(rewards):
    for name in CONFLICTING_REWARDS:
        term = rewards.get(name) if isinstance(rewards, dict) else getattr(rewards, name, None)
        if term is not None: term.weight = 0.
    return audit_conflicting_rewards(rewards)
