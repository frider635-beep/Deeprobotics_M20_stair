"""Stage5 pure-Torch geometry, strict entry sequencing and contact curriculum.

All state in this module is training-side. No tread ledger enters the actor.
"""
import torch
from .ascent_sequence import SequenceLedger
from .ascent_entry_mdp import smooth01

HEIGHTS = (.10, .12, .15, .18, .20, .25)
DEPTHS = (.30, .30, .30, .25, .30, .35, .30, .35, .30, .30)
REFERENCE_COLUMNS = (4, 6)
PREFIXES = (2, 4, 6, 18)
DEADLINES = (8., 12., 16., 23.5)


def depth_for_row(row, nominal):
    return .30+(nominal-.30)*min(1., max(0, row)/2.)


def geometry(env):
    """Exact geometry indexed by the same row/column as the generated mesh."""
    cfg = env.cfg
    if not getattr(cfg, 'ascent_adaptive_geometry', False):
        return (torch.full((env.num_envs,), float(cfg.ascent_entry_height), device=env.device),
                torch.full((env.num_envs,), float(cfg.ascent_entry_depth), device=env.device))
    if not hasattr(env, '_ascent_adaptive_geometry_tables'):
        env._ascent_adaptive_geometry_tables = tuple(torch.tensor(v, device=env.device)
            for v in (cfg.ascent_adaptive_height_grid, cfg.ascent_adaptive_depth_grid))
    terrain = env.scene.terrain
    rows, columns = terrain.terrain_levels, terrain.terrain_types
    return tuple(table[rows, columns] for table in env._ascent_adaptive_geometry_tables)


class ContactCurriculum:
    """Promote only after repeated real sequential contacts, never distance alone."""
    def __init__(self, count, device):
        self.skill = torch.zeros(count, device=device, dtype=torch.long)
        self.good = torch.zeros_like(self.skill)
        self.bad = torch.zeros_like(self.skill)
        self.completed = torch.zeros_like(self.skill)

    def goals(self):
        return self.skill.new_tensor(PREFIXES)[self.skill]

    def update(self, ids, success, eligible, rows, reference, training=True):
        up = torch.zeros_like(success); down = torch.zeros_like(success)
        if not training:
            return up, down
        good = eligible & success; bad = eligible & ~success
        self.completed[ids] += eligible.long()
        self.good[ids] = torch.where(good, self.good[ids]+1, torch.where(bad, 0, self.good[ids]))
        self.bad[ids] = torch.where(bad, self.bad[ids]+1, torch.where(good, 0, self.bad[ids]))
        promote = self.good[ids] >= 8
        demote = self.bad[ids] >= 3
        old_skill = self.skill[ids].clone()
        # Full courses must be solved before promoting geometry. References
        # stay on 15/30 cm regardless of their nominal row.
        up = promote & (old_skill == len(PREFIXES)-1) & ~reference & (rows < len(HEIGHTS)-1)
        down = demote & ~reference & (rows > 0)
        self.skill[ids] = torch.where(promote & (old_skill < len(PREFIXES)-1), old_skill+1, old_skill)
        # At a newly promoted geometry, revisit the first four treads before
        # expanding to six and the full flight. The old geometry was already
        # solved in full; a new row need not be learned only through long falls.
        self.skill[ids] = torch.where(up, 1, self.skill[ids])
        # Keep the full task when reducing geometry, but shorten the task if
        # it is already failing on the easiest row.
        self.skill[ids] = torch.where(demote & (rows == 0) & ~reference,
                                     (self.skill[ids]-1).clamp_min(0), self.skill[ids])
        changed = promote | demote
        self.good[ids] = torch.where(changed, 0, self.good[ids])
        self.bad[ids] = torch.where(changed, 0, self.bad[ids])
        return up, down

    def state_dict(self):
        return {k: getattr(self, k).detach().cpu().clone() for k in ('skill', 'good', 'bad', 'completed')}

    def load_state_dict(self, state):
        for key in ('skill', 'good', 'bad', 'completed'):
            value = state[key]
            if value.shape != self.skill.shape or value.dtype != torch.long or (value < 0).any():
                raise ValueError('Stage5 curriculum shape/dtype/value mismatch: '+key)
        if (state['skill'] >= len(PREFIXES)).any():
            raise ValueError('Stage5 curriculum skill out of range')
        for key, value in state.items():
            if key in ('skill', 'good', 'bad', 'completed'):
                getattr(self, key).copy_(value.to(self.skill.device))


class StrictSequenceLedger(SequenceLedger):
    def __init__(self, count, device, depth=.30, targets=18):
        super().__init__(count, device, depth, targets)
        self.failed = torch.zeros(count, device=device, dtype=torch.bool)
        self.pending_landing = torch.zeros(count, 2, targets, device=device)

    def reset(self, ids):
        super().reset(ids)
        self.failed[ids] = False; self.pending_landing[ids] = 0.

    def update_strict(self, level, support, wheel_x, first, active, dt, goals, period=None):
        alive = active & ~self.failed
        prior_pairs = self.pair_paid.clone()
        old_covered = self.covered.clone()
        result = super().update(level, support, wheel_x, first, alive, dt, period)
        required = self.targets <= goals[:, None, None]
        new = result['new_mask'] & required
        mate_previous = torch.cat((torch.ones_like(old_covered[:, :2, :1]), old_covered[:, [1, 0], :-1]), -1)
        early = (new[:, :2] & ~mate_previous).any((1, 2))
        simultaneous = self.covered[:, :2].all(1) & ~self.pair_paid
        failed_now = alive & ((self.missed & required).any((1, 2)) | early |
                              (simultaneous & required[:, 0]).any(-1))
        self.failed |= failed_now
        valid = alive & ~self.failed
        result['ordered'] = new.float().sum(-1)*valid[:, None]
        result['paired'] = ((self.pair_paid & ~prior_pairs) & required[:, 0]).float().sum(-1)*valid
        result['dwell'] *= valid[:, None] & (level <= goals[:, None])
        result['early_advance'] = early.float()
        result['failure'] = failed_now.float()
        result['valid'] = valid
        result['new_mask'] = new & valid[:, None, None]
        # A miss is not a successfully completed pair. Its successor cannot
        # collect positive rewards before the failure termination is consumed.
        return result

    def front_targets(self, active):
        unresolved = ~self.covered[:, :2]
        target = unresolved.long().argmax(-1)+1
        enabled = unresolved.any(-1) & active[:, None] & ~self.failed[:, None]
        pair_target = (~self.covered[:, :2].all(1)).long().argmax(-1)+1
        return target, enabled, pair_target

    def landing_credit(self, target, error, inside, new):
        rows = torch.arange(len(target), device=target.device)[:, None]
        wheels = torch.arange(2, device=target.device)[None]
        index = target-1
        previous = self.best_error[rows, wheels, index, 1]
        initial = torch.where(torch.isfinite(previous), previous, error)
        valid = inside & ~self.failed[:, None]
        improvement = (initial-error).clamp(0., 1.)*valid
        self.best_error[rows, wheels, index, 1] = torch.where(valid, torch.minimum(initial, error), previous)
        self.pending_landing[rows, wheels, index] = (self.pending_landing[rows, wheels, index]+improvement).clamp_max(2.)
        paid = (self.pending_landing*new[:, :2]).sum((1, 2))
        self.pending_landing *= ~new[:, :2]
        self.pending_landing[self.failed] = 0.
        return paid


def landing_guidance(position, velocity_z, target, enabled, first, floor, depth, rise):
    """Prepare immediately; reward progress only after real target support.

    Dense landing costs require a joint horizontal/vertical landing window.
    The preparation cost keeps riser clearance without paying for lifting high.
    """
    depth, rise = depth[:, None], rise[:, None]
    edge = first[:, None]+(target-1)*depth
    offset = position[..., 0]-edge
    bottom = position[..., 2]-.09
    top = floor[:, None]+target*rise
    near = smooth01((offset+.35)/.25)*(offset < depth+.09)*enabled
    blend = smooth01((offset-.01)/.06)
    desired = top+.04*(1.-blend)
    error_z = bottom-desired
    land = blend*near
    dx = (.06-offset).clamp_min(0.)+(offset-(depth-.06)).clamp_min(0.)
    height = 2.*(torch.sqrt(1.+(error_z/.08).square())-1.)
    horizontal = 2.*(torch.sqrt(1.+(dx/.15).square())-1.)
    desired_vz = (-3.*error_z).clamp(-.40, .30)
    speed = ((velocity_z-desired_vz)/.5).square().clamp_max(4.)
    prepare = ((desired-bottom).clamp_min(0.)/.08).square().clamp_max(4.)*(1.-blend)*near
    inside = (offset >= .06) & (offset <= depth-.06) & enabled
    return dict(prepare=prepare.mean(-1), height=(height*land).mean(-1),
                horizontal=(horizontal*near).mean(-1), velocity=(speed*land).mean(-1),
                error=(height+.5*horizontal).clamp_max(8.), inside=inside,
                height_error=error_z, opportunity=land)
