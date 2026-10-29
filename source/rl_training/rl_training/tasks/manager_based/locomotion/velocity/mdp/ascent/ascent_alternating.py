"""Confirmed swing alternation, speed-balanced courses and dense ascent costs.

Front (FL/FR) and rear (HL/HR) have separate clocks. A side is a wheel/leg,
not a tread: alternating sides never authorizes skipping a physical tread.
"""
import torch
from .ascent_adaptive import ContactCurriculum, PREFIXES, HEIGHTS


class SwingAlternation:
    """Count one real swing per stable support, filtering short contact dropouts.

The first side is free. Same-side repeats or simultaneous starts latch failure
until reset. Front/rear swings may overlap; no diagonal or axle-gap rule exists.
"""
    def __init__(self, count, device):
        self.clock = torch.zeros(count, device=device)
        self.contact_age = torch.zeros(count, 4, device=device)
        self.air_age = torch.zeros_like(self.contact_age)
        self.support_z = torch.zeros_like(self.contact_age)
        self.armed = torch.zeros(count, 4, device=device, dtype=torch.bool)
        self.seen_support = torch.zeros_like(self.armed)
        self.engaged = torch.zeros(count, 2, device=device, dtype=torch.bool)
        self.last_side = torch.full((count, 2), -1, device=device, dtype=torch.long)
        self.last_start = torch.full((count, 2), -100., device=device)
        # A fixed environment-index tie breaker can imprint a persistent
        # left/right bias on the first swing. Choose the tie side per episode.
        ids = torch.arange(count, device=device)
        pair = ids // 2
        base = torch.randint(0, 2, (int(pair[-1].item()) + 1, 1), device=device)
        self.tie_side = ((base[pair] + (ids % 2)[:, None]) % 2).expand(-1, 2).clone()
        self.failed = torch.zeros(count, device=device, dtype=torch.bool)
        self.starts = torch.zeros(count, 2, device=device)
        self.violations = torch.zeros_like(self.starts)

    def reset(self, ids):
        for name in ('clock', 'contact_age', 'air_age', 'support_z', 'armed',
                     'seen_support', 'engaged', 'failed', 'starts', 'violations'):
            getattr(self, name)[ids] = 0
        self.last_side[ids] = -1
        self.last_start[ids] = -100.
        pair = ids // 2
        unique, inverse = torch.unique(pair, sorted=True, return_inverse=True)
        base = torch.randint(0, 2, (len(unique), 1), device=self.tie_side.device)
        self.tie_side[ids] = ((base[inverse] + (ids % 2)[:, None]) % 2).expand(-1, 2)

    def update(self, support, wheel_z, wheel_vz, pair_near, active, dt, repeat_grace=None, soft_repeat=None):
        self.clock += dt
        self.engaged |= pair_near & active[:, None]
        monitor = self.engaged.repeat_interleave(2, -1) & active[:, None]
        self.contact_age = torch.where(support & monitor, self.contact_age+dt, 0.)
        stable = self.contact_age >= .04-1.e-6
        self.armed |= stable
        self.seen_support |= stable
        self.support_z = torch.where(stable, wheel_z, self.support_z)
        self.air_age = torch.where(~support & monitor, self.air_age+dt, 0.)
        rise = wheel_z-self.support_z
        real_lift = (rise >= .02) | ((rise >= .006) & (wheel_vz >= .12))
        lift = monitor & self.armed & (self.air_age >= .02-1.e-6) & real_lift
        self.armed &= ~lift
        events = lift.reshape(-1, 2, 2)
        count = events.sum(-1)
        side = events.long().argmax(-1)
        repeated = (count == 1) & (side == self.last_side)
        if repeat_grace is not None:
            repeated &= ~repeat_grace
        bad = ((count > 1) | repeated) & ~self.failed[:, None]
        soft = torch.zeros_like(bad) if soft_repeat is None else (repeated & soft_repeat & ~self.failed[:,None])
        hard = bad & ~soft
        self.violations += bad
        self.starts += count
        self.last_side = torch.where(count > 0, side, self.last_side)
        self.last_start = torch.where(count > 0, self.clock[:, None], self.last_start)
        failed_now = hard.any(-1) & ~self.failed
        self.failed |= hard.any(-1)
        ready = (~self.engaged | self.seen_support.reshape(-1, 2, 2).all(-1)).all(-1)
        return dict(lift=lift, violation=bad, failed_now=failed_now,soft_violation=soft,
                    valid=~self.failed & ready, front_valid=~self.failed & (~self.engaged[:, 0] | self.seen_support[:, :2].all(-1)),
                    rear_valid=~self.failed & (~self.engaged[:, 1] | self.seen_support[:, 2:].all(-1)))


class BalancedContactCurriculum(ContactCurriculum):
    """Advance only when BOTH speeds succeed on >=6 of their last 8 episodes."""
    def __init__(self, count, device):
        super().__init__(count, device)
        self.speed_bucket = torch.zeros(count, device=device, dtype=torch.long)
        self.window = torch.zeros(count, 2, 8, device=device, dtype=torch.long)
        self.cursor = torch.zeros(count, 2, device=device, dtype=torch.long)
        self.trials = torch.zeros_like(self.cursor)
        # Fixed IDs reconstruct the same small geometry-practice group on
        # same-num_envs resume. Reference columns remain fixed geometrically.
        self.geometry_probe = torch.arange(count, device=device).remainder(4) == 0

    def update(self, ids, success, eligible, rows, reference, training=True):
        up = torch.zeros_like(success); down = torch.zeros_like(success)
        if not training:
            return up, down
        bucket = self.speed_bucket[ids]
        slot = self.cursor[ids, bucket]
        previous = self.window[ids, bucket, slot]
        self.window[ids, bucket, slot] = torch.where(eligible, success.long(), previous)
        self.cursor[ids, bucket] = (slot+eligible.long()) % 8
        self.trials[ids, bucket] = (self.trials[ids, bucket]+eligible.long()).clamp_max(8)
        self.completed[ids] += eligible.long()
        good = eligible & success; bad = eligible & ~success
        self.good[ids] = torch.where(good, self.good[ids]+1, torch.where(bad, 0, self.good[ids]))
        self.bad[ids] = torch.where(bad, self.bad[ids]+1, torch.where(good, 0, self.bad[ids]))
        promote = good & (self.trials[ids] >= 8).all(-1) & (self.window[ids].sum(-1) >= 6).all(-1)
        demote = eligible & (self.bad[ids] >= 3)
        old = self.skill[ids].clone()
        # A quarter of non-reference environments may expand nearby geometry
        # after a verified six-tread task. Limit this early path to rows 0->1->2
        # (10->12->15 cm); higher rows still require the complete flight.
        probe_ready = self.geometry_probe[ids] & (old == 2) & (rows < 2)
        full_ready = old == len(PREFIXES)-1
        up = promote & (probe_ready | full_ready) & ~reference & (rows < len(HEIGHTS)-1)
        down = demote & ~reference & (rows > 0)
        skill = torch.where(promote & (old < len(PREFIXES)-1), old+1, old)
        skill = torch.where(up, 1, skill)
        skill = torch.where(demote & (rows == 0), (skill-1).clamp_min(0), skill)
        changed = (skill != old) | up | down
        self.skill[ids] = skill
        reset_ids = ids[changed]
        self.window[reset_ids] = 0; self.cursor[reset_ids] = 0; self.trials[reset_ids] = 0
        self.good[ids] = torch.where(promote | demote, 0, self.good[ids])
        self.bad[ids] = torch.where(promote | demote, 0, self.bad[ids])
        return up, down

    def state_dict(self):
        result = super().state_dict()
        result.update({k: getattr(self, k).detach().cpu().clone()
                       for k in ('speed_bucket', 'window', 'cursor', 'trials')})
        return result

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError('Stage6 resume requires both-speed curriculum state')
        for key, value in state.items():
            if value.shape != expected[key].shape or value.dtype != torch.long or (value < 0).any():
                raise ValueError('Invalid stage6 curriculum '+key)
        if ((state['skill'] >= len(PREFIXES)).any() or (state['speed_bucket'] > 1).any() or
                (state['window'] > 1).any() or (state['cursor'] >= 8).any() or (state['trials'] > 8).any()):
            raise ValueError('Stage6 curriculum state out of range')
        for key, value in state.items():
            getattr(self, key).copy_(value.to(self.skill.device))


def dense_costs(command_x, velocity_x, lateral_error, heading_error, slip, support,
                height, depth, progress, age, tangent=None):
    """Dimensionless costs, with no positive reward for waiting or wheel spin."""
    target = command_x.clamp_min(.1)
    grace = ((age-.5)/.5).clamp(0., 1.)
    stall = ((.8*target-velocity_x).clamp_min(0.)/(.8*target)).square().clamp_max(4.)*grace
    transition = 1.+.5*((progress >= 2.) & (progress <= 4.5))
    # Separate lateral drift from rolling-direction contact motion. A single
    # tangent norm hides the distinction and makes the policy trade speed for
    # a lower normalized slip signal.
    if tangent is None:
        lateral_slip = slip
        longitudinal_slip = torch.zeros_like(slip)
    else:
        lateral_slip = tangent[..., 1].abs()
        longitudinal_slip = tangent[..., 0].abs()
    lateral_cost = (torch.sqrt(1.+(lateral_slip/.15).square())-1.).clamp_max(8.)*support
    longitudinal_cost = (torch.sqrt(1.+(longitudinal_slip/.20).square())-1.).clamp_max(8.)*support
    slip_cost = lateral_cost + .35*longitudinal_cost
    # Do not triple the slip penalty merely because the robot slows at a
    # riser. Slow forward motion is already covered by the stall term; keep
    # the slip scale in a bounded command-relative band.
    denominator = (velocity_x.clamp_min(.15)/target).clamp(.6, 1.4)
    front = (.5*slip_cost[:, :2].mean(-1)+.5*slip_cost[:, :2].amax(-1))/denominator
    rear = (.5*slip_cost[:, 2:].mean(-1)+.5*slip_cost[:, 2:].amax(-1))/denominator
    lateral = ((lateral_error.abs()-.03).clamp_min(0.)/.12).square().clamp_max(9.)
    heading = ((heading_error.abs()-.025).clamp_min(0.)/.15).square().clamp_max(9.)
    period = depth/target*(1.+.4*(height/.15-1.)).clamp(.8, 1.4)
    front_lateral = (.5*lateral_cost[:, :2].mean(-1)+.5*lateral_cost[:, :2].amax(-1))/denominator
    rear_lateral = (.5*lateral_cost[:, 2:].mean(-1)+.5*lateral_cost[:, 2:].amax(-1))/denominator
    front_longitudinal = (.5*longitudinal_cost[:, :2].mean(-1)+.5*longitudinal_cost[:, :2].amax(-1))/denominator
    rear_longitudinal = (.5*longitudinal_cost[:, 2:].mean(-1)+.5*longitudinal_cost[:, 2:].amax(-1))/denominator
    return dict(stall=stall, front_slip=front*transition, rear_slip=rear*transition,
                front_lateral_slip=front_lateral*transition, rear_lateral_slip=rear_lateral*transition,
                front_longitudinal_slip=front_longitudinal*transition,
                rear_longitudinal_slip=rear_longitudinal*transition,
                lateral=lateral, heading=heading, period=period.clamp(.25, 1.2))


class CadenceTracker:
    """One-tread per-wheel intervals, from the second landing onward."""
    def __init__(self, count, device):
        self.previous_level = torch.zeros(count, 4, device=device, dtype=torch.long)
        self.previous_time = torch.full((count, 4), -100., device=device)
        self.ema = torch.zeros(count, 4, device=device)

    def reset(self, ids):
        self.previous_level[ids] = 0; self.previous_time[ids] = -100.; self.ema[ids] = 0.

    def update(self, level, landing_time, now, period, active, goals=None):
        new = (level > self.previous_level) & active[:, None]
        interval = landing_time-self.previous_time
        valid = new & (self.previous_level > 0) & (level == self.previous_level+1)
        target = period[:, None]
        absolute = ((interval/target-1.).abs()-.25).clamp_min(0.).square().clamp_max(4.)
        regular = ((interval-self.ema).abs()/target-.20).clamp_min(0.).square().clamp_max(4.)
        regular *= self.ema > 0.
        event_cost = ((absolute+.5*regular)*valid*(1.+.5*(level == 4))).sum(-1)
        next_ema = torch.where(self.ema > 0., .8*self.ema+.2*interval, interval)
        self.ema = torch.where(valid, next_ema, self.ema)
        self.previous_time = torch.where(new, landing_time, self.previous_time)
        self.previous_level = torch.where(new, level, self.previous_level)
        overdue = ((now[:, None]-self.previous_time)/target-1.6).clamp_min(0.).square().clamp_max(4.)
        overdue *= active[:, None] & (self.previous_level > 0)
        if goals is not None:
            overdue *= self.previous_level < goals[:, None]
        return event_cost, overdue.mean(-1)
