"""Batched path feedback; identical equations in Isaac and MuJoCo.

The reference is the straight requested route latched at motion onset. It does
not require a privileged terrain id, a tread ledger or a known stair centre.
"""
from dataclasses import dataclass, asdict
import torch

PATH_DIM = 6
PATH_VERSION = 'm20_stair_path_6d_20261022'


@dataclass(frozen=True)
class PathConfig:
    up_kp_slow: float = .60
    up_kp_fast: float = .65
    up_kd_slow: float = 0.
    up_kd_fast: float = .50
    up_limit_slow: float = .12
    up_limit_fast: float = .15
    up_deadband_slow: float = .08
    up_deadband_fast: float = .025
    down_kp: float = .90
    down_kd: float = .55
    down_limit: float = .18
    down_deadband: float = .025
    speed_deadband: float = .025
    lateral_slew: float = .25
    exit_memory_s: float = 1.0

    def to_dict(self):
        return asdict(self)


def stair_edges(scan):
    """The same central 5 scan rows and rising threshold used by Stage7."""
    profile = scan.reshape(-1, 11, 17)[:, 3:8].median(dim=1).values
    difference = profile[:, :-1] - profile[:, 1:]
    up = ((difference-.045)/.055).clamp(0., 1.).amax(-1)
    down = ((-difference-.045)/.055).clamp(0., 1.).amax(-1)
    return up, down


def straight_gate(command):
    gate = ((command[:, 0]-.1)/.2).clamp(0., 1.)
    gate = gate*(1.-command[:, 1].abs()/.15).clamp(0., 1.)
    return gate*(1.-command[:, 2].abs()/.2).clamp(0., 1.)


def deadband(value, threshold):
    return value.sign()*(value.abs()-threshold).clamp_min(0.)


class StairPathController:
    def __init__(self, count, device, config=None):
        self.config = PathConfig(**config) if isinstance(config, dict) else config or PathConfig()
        self.origin = torch.zeros(count, 2, device=device)
        self.heading = torch.zeros(count, device=device)
        self.straight = torch.zeros(count, device=device, dtype=torch.bool)
        self.direction = torch.zeros(count, device=device)
        self.age = torch.full((count,), 1.e6, device=device)
        self.correction = torch.zeros(count, device=device)
        self.features = torch.zeros(count, PATH_DIM, device=device)
        self.active = torch.zeros_like(self.straight)
        self.saturated = torch.zeros_like(self.straight)
        self.error = torch.zeros(count, device=device)
        self.speed = torch.zeros_like(self.error)

    def reset(self, ids):
        self.straight[ids] = False
        self.direction[ids] = 0.
        self.age[ids] = 1.e6
        self.correction[ids] = 0.
        self.features[ids] = 0.
        self.active[ids] = False
        self.saturated[ids] = False

    def measure(self, position, velocity, yaw):
        normal = torch.stack((-torch.sin(self.heading), torch.cos(self.heading)), -1)
        error = ((position-self.origin)*normal).sum(-1)
        speed = (velocity*normal).sum(-1)
        heading_error = torch.atan2(torch.sin(self.heading-yaw), torch.cos(self.heading-yaw))
        return error, speed, heading_error, normal

    def update(self, position, velocity, yaw, command, scan, dt):
        cfg = self.config
        straight = (command[:, 0] > .1) & (command[:, 1:].abs().amax(-1) <= .1)
        entering = straight & ~self.straight
        self.origin = torch.where(entering[:, None], position, self.origin)
        self.heading = torch.where(entering, yaw, self.heading)
        self.direction = torch.where(entering | ~straight, 0., self.direction)
        self.age = torch.where(entering | ~straight, 1.e6, self.age)
        up, down = stair_edges(scan)
        seen = straight & (torch.maximum(up, down) > .05)
        observed_direction = torch.where(up >= down, 1., -1.)
        self.direction = torch.where(seen, observed_direction, self.direction)
        self.age = torch.where(seen, 0., self.age+dt)
        active = straight & (self.age <= cfg.exit_memory_s)
        error, speed, heading_error, _ = self.measure(position, velocity, yaw)
        blend = ((command[:, 0]-.5)/.2).clamp(0., 1.)
        interpolate = lambda slow, fast: slow+(fast-slow)*blend
        descending = self.direction < 0.
        kp = torch.where(descending, cfg.down_kp, interpolate(cfg.up_kp_slow, cfg.up_kp_fast))
        kd = torch.where(descending, cfg.down_kd, interpolate(cfg.up_kd_slow, cfg.up_kd_fast))
        limit = torch.where(descending, cfg.down_limit, interpolate(cfg.up_limit_slow, cfg.up_limit_fast))
        band = torch.where(descending, cfg.down_deadband, interpolate(cfg.up_deadband_slow, cfg.up_deadband_fast))
        desired = -kp*deadband(error, band)-kd*deadband(speed, cfg.speed_deadband)
        saturated = active & (desired.abs() >= limit)
        desired = desired.clamp(-limit, limit)*torch.cos(heading_error)
        desired = torch.where(active, desired, 0.)
        before = torch.where(entering | ~straight, 0., self.correction)
        correction = before+(desired-before).clamp(-cfg.lateral_slew*dt, cfg.lateral_slew*dt)
        # Decay smoothly on platform exit; deliberate manual requests/stop take
        # immediate ownership of vy, including after a reset or reversal.
        correction = torch.where(straight, correction, 0.)
        context = active | (straight & (correction.abs() > 1.e-6))
        features = torch.stack(((error/.4).clamp(-3., 3.), (speed/.3).clamp(-3., 3.),
            torch.sin(heading_error), up, down, context.float()), -1)*straight[:, None]
        self.features, self.correction = features, correction
        self.active, self.saturated, self.error, self.speed = active, saturated, error, speed
        self.straight = straight
        return correction, features
