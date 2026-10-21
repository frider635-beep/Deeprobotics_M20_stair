"""Pure-Torch event accounting and spatial targets for a full straight flight.

Either front wheel may lead. All constraints are soft; no fixed axle tread gap,
no contact labels in the actor, and no repeated credit for waiting/retreating.
"""
import torch
from .ascent_entry_mdp import smooth01
from .ascent_entry_stage2 import TreadLedger


class SequenceLedger(TreadLedger):
    def __init__(self, count, device, depth=.30, targets=18):
        super().__init__(count, device, depth, targets)
        self.time = torch.zeros(count, device=device)
        self.landing_time = torch.full_like(self.best_dwell, -100.)
        self.pair_paid = torch.zeros(count, targets, device=device, dtype=torch.bool)
        self.pair_done = torch.zeros_like(self.pair_paid)
        self.best_error = torch.full((count, 2, targets, 2), float('inf'), device=device)

    def reset(self, ids):
        super().reset(ids)
        self.time[ids] = 0.; self.landing_time[ids] = -100.
        self.pair_paid[ids] = False; self.pair_done[ids] = False
        self.best_error[ids] = float('inf')

    def update(self, level, support, wheel_x, first_edge, active, dt, period=None):
        old = self.covered.clone()
        result = super().update(level, support, wheel_x, first_edge, active, dt)
        self.time += dt
        new = self.covered & ~old
        # Backdate to first contact so different contact-confirmation times
        # cannot manufacture a sequential landing out of simultaneous contact.
        touch = self.time[:, None]-self.duration+dt
        self.landing_time = torch.where(new, touch[:, :, None], self.landing_time)
        both = self.covered[:, :2].all(1)
        newly_paired = both & ~self.pair_done
        difference = (self.landing_time[:, 0]-self.landing_time[:, 1]).abs()
        sequential = newly_paired & (difference >= .04-1.e-6)
        self.pair_done |= both
        self.pair_paid |= sequential
        result['paired'] = sequential.float().sum(-1)
        result['simultaneous'] = (newly_paired & ~sequential).float().sum(-1)
        # A front wheel landing on k+1 before its mate has supported k is a
        # soft early-advance violation. Do not impose any front/rear axle gap.
        mate_previous = torch.cat((torch.ones_like(old[:, :2, :1]), old[:, [1, 0], :-1]), -1)
        result['early_advance'] = (new[:, :2] & ~mate_previous).float().sum((1, 2))
        result['cadence'] = torch.zeros_like(self.time)
        if period is not None:
            interval = self.landing_time[:, :, 1:]-self.landing_time[:, :, :-1]
            valid = new[:, :, 1:] & self.covered[:, :, :-1]
            # A broad timing hint, introduced gradually; landing is primary.
            error = ((interval/period[:, None, None]-1.).abs()-.20).clamp_min(0.)/.5
            blend = smooth01((self.targets[:, :, 1:].float()-1.)/3.)
            result['cadence'] = (error.square().clamp_max(4.)*valid*blend).sum((1, 2))
        result['new_mask'] = new
        return result

    def approach_credit(self, target, error, opportunity, after_edge):
        """Bounded best-so-far improvement, once per wheel/tread/lift-or-land phase."""
        rows = torch.arange(len(target), device=target.device)[:, None]
        wheels = torch.arange(2, device=target.device)[None]
        index = (target-1).clamp(0, self.targets.shape[-1]-1)
        phase = after_edge.long()
        previous = self.best_error[rows, wheels, index, phase]
        valid = opportunity > 0.
        initial = torch.where(torch.isfinite(previous), previous, error)
        improvement = (initial-error).clamp(0., 1.)*valid
        best = torch.where(valid, torch.minimum(initial, error), previous)
        self.best_error[rows, wheels, index, phase] = best
        return improvement.sum(-1)


def front_guidance(position, velocity_z, target, enabled, first_edge, floor, depth=.30, rise=.15):
    edge = first_edge[:, None]+(target-1)*depth
    offset = position[..., 0]-edge
    opportunity = smooth01((offset+.35)/.25)*(offset < depth+.09)*enabled
    descent = smooth01((offset-.02)/.08)
    desired_bottom = floor[:, None]+target*rise+.04*(1.-descent)
    dz = position[..., 2]-.09-desired_bottom
    dx = (.07-offset).clamp_min(0.)+(offset-(depth-.06)).clamp_min(0.)
    # Pseudo-Huber retains a gradient for large errors instead of saturating
    # the entire landing opportunity at a constant squared-cost ceiling.
    height = 2.*(torch.sqrt(1.+(dz/.10).square())-1.)
    horizontal = 2.*(torch.sqrt(1.+(dx/.20).square())-1.)
    desired_vz = (-3.*dz).clamp(-.45, .35)
    speed = ((velocity_z-desired_vz)/.60).square().clamp_max(4.)
    return {'height': (height*opportunity).mean(-1),
            'horizontal': (horizontal*opportunity).mean(-1),
            'velocity': (speed*opportunity).mean(-1),
            'error': (height+.25*horizontal).clamp_max(8.),
            'opportunity': opportunity, 'height_error': dz, 'after_edge': offset > .04}


def support_margin_cost(wheel_xy, support, com_xy, velocity_xy):
    """Soft projected-COM/velocity-lookahead heuristic; not a ZMP guarantee.

    Four-wheel hull edges are found without CPU hull construction. Two-contact
    support uses a segment with finite width. Pitch is deliberately absent.
    """
    projected = com_xy+.10*velocity_xy
    count = support.sum(-1)
    centroid = (wheel_xy*support[:, :, None]).sum(1)/count.clamp_min(1.)[:, None]
    distances = []
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        origin = wheel_xy[:, i]; edge = wheel_xy[:, j]-origin
        length = torch.linalg.vector_norm(edge, dim=-1).clamp_min(.01)
        def cross(vector): return edge[:, 0]*vector[..., 1]-edge[:, 1]*vector[..., 0]
        center_side = cross(centroid-origin)
        sign = torch.where(center_side >= 0., 1., -1.)
        relative = wheel_xy-origin[:, None]
        side = (edge[:, None, 0]*relative[..., 1]-edge[:, None, 1]*relative[..., 0])*sign[:, None]
        hull = ((side >= -1.e-5) | ~support).all(-1) & support[:, i] & support[:, j]
        distance = cross(projected-origin)*sign/length
        # A two-wheel support line becomes a finite capsule, not an infinite
        # line that would incorrectly accept COM far beyond both contacts.
        fraction = ((projected-origin)*edge).sum(-1)/length.square()
        nearest = origin+fraction.clamp(0., 1.)[:, None]*edge
        capsule = .06-torch.linalg.vector_norm(projected-nearest, dim=-1)
        distance = torch.where(count == 2, capsule, distance)
        distances.append(torch.where(hull, distance, torch.full_like(distance, 10.)))
    margin = torch.stack(distances, -1).amin(-1)
    cost = ((-.04-margin).clamp_min(0.)/.12).square().clamp_max(4.)
    return torch.where(count >= 2, cost, torch.full_like(cost, 4.)), margin


def terrain_pitch_target(position, first_edge, depth=.30, rise=.15, count=18):
    # Smooth the height changes over each tread so the platform target does
    # not jump to level when a single rear wheel crosses an edge.
    coordinate = (position[..., 0]-first_edge[:, None])/depth+.5
    index = torch.floor(coordinate)
    height = (index+smooth01(coordinate-index)).clamp(0., count)*rise
    dx = (position[:, :2, 0].mean(-1)-position[:, 2:, 0].mean(-1)).clamp_min(.25)
    return torch.atan2(height[:, :2].mean(-1)-height[:, 2:].mean(-1), dx)


class TopTransition:
    def __init__(self, count, device):
        self.front_paid = torch.zeros(count, dtype=torch.bool, device=device)
        self.rear_paid = torch.zeros_like(self.front_paid)
        self.front_only_seen = torch.zeros_like(self.front_paid)
        self.best_forward = torch.zeros(count, device=device)
        self.origin = torch.zeros_like(self.best_forward)
        self.settled = torch.zeros_like(self.best_forward)

    def reset(self, ids):
        self.front_paid[ids] = False; self.rear_paid[ids] = False
        self.front_only_seen[ids] = False
        self.best_forward[ids] = 0.; self.origin[ids] = 0.; self.settled[ids] = 0.

    def update(self, on_top, root_x, level_error, speed, enabled, dt):
        front = on_top[:, :2].all(-1) & enabled
        rear = on_top[:, 2:].all(-1) & front & enabled
        front_event = front & ~self.front_paid
        self.front_only_seen |= front & ~on_top[:, 2:].all(-1)
        rear_event = rear & self.front_paid & self.front_only_seen & ~self.rear_paid
        self.front_paid |= front
        self.origin = torch.where(rear_event, root_x, self.origin)
        self.rear_paid |= rear_event
        # Progress can only be credited with all wheels currently on top,
        # near-level body, and a forward command. Histories do not suffice.
        valid = rear & self.rear_paid & (level_error < .15) & (speed > .1)
        forward = (root_x-self.origin).clamp(0., .60)
        new_best = torch.where(valid, torch.maximum(self.best_forward, forward), self.best_forward)
        credit = new_best-self.best_forward; self.best_forward = new_best
        self.settled = torch.where(valid, self.settled+dt, 0.)
        return {'front': front_event.float(), 'rear': rear_event.float(), 'forward': credit,
                'ready': valid & (forward >= .35) & (self.settled >= .20)}
