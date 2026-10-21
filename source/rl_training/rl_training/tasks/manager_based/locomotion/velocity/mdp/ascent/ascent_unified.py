"""Stage7 bridge curriculum, paired landing guidance and failure attribution."""
import torch
from .ascent_alternating import BalancedContactCurriculum
from .ascent_adaptive import StrictSequenceLedger, HEIGHTS, landing_guidance

PREFIXES = (2, 3, 4, 6, 18)
DEADLINES = (8., 10., 12., 16., 23.5)


class BridgeCurriculum(BalancedContactCurriculum):
    prefixes = PREFIXES
    deadlines = DEADLINES

    def __init__(self, count, device):
        super().__init__(count, device)
        group = torch.arange(count, device=device).remainder(10)
        # Pinned practice has the same strict success criteria. Only the minimum
        # task length differs, so difficult transitions keep receiving samples.
        self.minimum_skill = torch.where(group < 2, 2, torch.where(group < 5, 1, 0))
        self.skill.copy_(self.minimum_skill)

    def goals(self):
        return self.skill.new_tensor(self.prefixes)[self.skill]

    def update(self, ids, success, eligible, rows, reference, training=True):
        up = torch.zeros_like(success); down = torch.zeros_like(success)
        if not training: return up, down
        bucket = self.speed_bucket[ids]; slot = self.cursor[ids, bucket]
        self.window[ids, bucket, slot] = torch.where(eligible, success.long(), self.window[ids, bucket, slot])
        self.cursor[ids, bucket] = (slot+eligible.long()) % 8
        self.trials[ids, bucket] = (self.trials[ids, bucket]+eligible.long()).clamp_max(8)
        self.completed[ids] += eligible.long()
        good = eligible & success; bad = eligible & ~success
        self.good[ids] = torch.where(good, self.good[ids]+1, torch.where(bad, 0, self.good[ids]))
        self.bad[ids] = torch.where(bad, self.bad[ids]+1, torch.where(good, 0, self.bad[ids]))
        promote = good & (self.trials[ids] >= 8).all(-1) & (self.window[ids].sum(-1) >= 6).all(-1)
        demote = eligible & (self.bad[ids] >= 3)
        old = self.skill[ids].clone()
        up = promote & ((old == 4) | (self.geometry_probe[ids] & (old == 3) & (rows < 2)))
        up &= ~reference & (rows < len(HEIGHTS)-1)
        down = demote & ~reference & (rows > 0)
        skill = torch.where(promote, (old+1).clamp_max(4), old)
        skill = torch.where(up, 2, skill)
        skill = torch.where(demote & (rows == 0), skill-1, skill)
        skill = torch.maximum(skill, self.minimum_skill[ids])
        changed = (skill != old) | up | down
        self.skill[ids] = skill
        reset_ids = ids[changed]
        self.window[reset_ids] = 0; self.cursor[reset_ids] = 0; self.trials[reset_ids] = 0
        self.good[ids] = torch.where(promote | demote, 0, self.good[ids])
        self.bad[ids] = torch.where(promote | demote, 0, self.bad[ids])
        return up, down

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected): raise ValueError('Stage7 curriculum state keys mismatch')
        for key, value in state.items():
            if value.shape != expected[key].shape or value.dtype != torch.long or (value < 0).any():
                raise ValueError('Invalid stage7 curriculum '+key)
        if ((state['skill'] < self.minimum_skill.to(state['skill'].device)).any() or (state['skill'] > 4).any()
                or (state['speed_bucket'] > 1).any() or (state['window'] > 1).any()
                or (state['cursor'] >= 8).any() or (state['trials'] > 8).any()):
            raise ValueError('Stage7 curriculum state out of range')
        for key, value in state.items(): getattr(self, key).copy_(value.to(self.skill.device))


class PairedSequenceLedger(StrictSequenceLedger):
    """Strict per-wheel coverage and same-tread sequential support for both axles."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rear_pair_paid = torch.zeros_like(self.pair_paid)
        self.failure_reason = torch.zeros_like(self.time, dtype=torch.long)

    def reset(self, ids):
        super().reset(ids)
        self.rear_pair_paid[ids] = False; self.failure_reason[ids] = 0

    def update_strict(self, level, support, wheel_x, first, active, dt, goals, period=None):
        old = self.covered.clone(); was_failed = self.failed.clone()
        result = super().update_strict(level, support, wheel_x, first, active, dt, goals, period)
        required = self.targets <= goals[:, None, None]
        new = self.covered & ~old & required
        mate_previous = torch.cat((torch.ones_like(old[:, 2:, :1]), old[:, [3, 2], :-1]), -1)
        rear_early = (new[:, 2:] & ~mate_previous).any((1, 2))
        difference = (self.landing_time[:, 2]-self.landing_time[:, 3]).abs()
        rear_both = self.covered[:, 2:].all(1)
        self.rear_pair_paid |= rear_both & (difference >= .04-1.e-6)
        rear_simultaneous = (rear_both & ~self.rear_pair_paid & required[:, 0]).any(-1)
        front_previous = torch.cat((torch.ones_like(old[:, :2, :1]), old[:, [1, 0], :-1]), -1)
        early = (new[:, :2] & ~front_previous).any((1, 2)) | rear_early
        missed = (self.missed & required).any((1, 2))
        paired_bad = ((self.covered[:, :2].all(1) & ~self.pair_paid & required[:, 0]).any(-1)
                      | rear_simultaneous)
        failure = active & ~was_failed & (missed | early | paired_bad)
        self.failure_reason = torch.where(failure, torch.where(missed, 1, torch.where(early, 2, 3)), self.failure_reason)
        self.failed |= failure
        valid = active & ~self.failed
        result['valid'] &= valid; result['new_mask'] &= valid[:, None, None]
        for key in ('ordered', 'dwell'): result[key] *= valid[:, None]
        result['paired'] *= valid; result['failure'] = torch.maximum(result['failure'], failure.float())
        result['early_advance'] = torch.maximum(result['early_advance'], rear_early.float())
        return result


def paired_guidance(ledger, position, velocity_z, first, floor, depth, height, goals, active):
    """Land on each wheel's next tread; hold a leading wheel until its mate lands.

    No front/rear tread separation is imposed. Negative approach costs remain
    available before contact success, with all positive credit still gated.
    """
    count = ledger.covered.shape[-1]
    own_target = (~ledger.covered).long().argmax(-1)+1
    pair_covered = ledger.covered.reshape(-1, 2, 2, count).all(2)
    pair_target = (~pair_covered).long().argmax(-1)+1
    pair_target = pair_target.repeat_interleave(2, -1)
    own_done = ledger.covered.gather(-1, (pair_target-1)[..., None]).squeeze(-1)
    pending_pair = (~pair_covered).any(-1).repeat_interleave(2, -1)
    hold = own_done & pending_pair & (pair_target <= goals[:, None])
    targets = torch.where(hold, pair_target, own_target)
    enabled = active[:, None] & ~ledger.failed[:, None] & (~ledger.covered).any(-1) & (targets <= goals[:, None])
    enabled |= hold & active[:, None] & ~ledger.failed[:, None]
    guide = landing_guidance(position, velocity_z, targets, enabled, first, floor, depth, height)
    edge = first[:, None]+(targets-1)*depth[:, None]
    dx = position[..., 0]-edge
    dz = position[..., 2]-.09-(floor[:, None]+targets*height[:, None])
    # Once its own tread has been supported, a leading wheel should stay near
    # that tread rather than preparing an early move to the next one.
    hold_cost = ((dz/.08).square()+((dx-(depth[:, None]-.08)).clamp_min(0.)/.08).square()
                 +(velocity_z.clamp_min(0.)/.3).square()).clamp_max(9.)
    hold_cost = (hold_cost*hold*enabled).sum(-1)/hold.sum(-1).clamp_min(1)
    return guide, hold_cost
