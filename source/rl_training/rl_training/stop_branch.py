"""Shared IsaacLab/MuJoCo zero-request reference and trainable holding branch.

The 244D motion actor is frozen. The stop branch additionally observes 24D
reference errors. A latched physical pose and wheel/position feedback provide
the initial holding action; PPO learns bounded balance corrections.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.models import MLPModel

STOP_DIM = 24
DEFAULT_LEGS = [0., -.3, .6, 0., -.3, .6, 0., .3, -.6, 0., .3, -.6]
LEG_SCALE = [.125, .25, .25]*4


class StopReference:
    """Latch stop errors and planar hold references; clear on restart/reset."""
    def __init__(self, dt=.02):
        self.dt = dt
        self.state = None

    def update(self, command, joint_pos, xy, yaw, linear_velocity_b, reset=None, support=None, joint_velocity=None, planar_support=None, angular_velocity=None):
        zero = command.abs().amax(-1) <= 1.e-4
        if reset is None:
            reset = torch.zeros_like(zero)
        if self.state is None or self.state['was_zero'].shape != zero.shape:
            self.state = dict(was_zero=torch.zeros_like(zero), held=torch.zeros_like(zero), legs_held=torch.zeros_like(zero),
                              braking=torch.zeros_like(zero),
                              stable_ticks=torch.zeros_like(yaw), raw_age=torch.zeros_like(yaw),
                              xy_latched=torch.zeros_like(zero),
                              age=torch.zeros_like(yaw), leg_age=torch.zeros_like(yaw),
                              joints=joint_pos.clone(), xy=xy.clone(), yaw=yaw.clone())
        s = self.state
        entering = zero & (~s['was_zero'] | reset)
        s['held'] &= zero & ~reset & ~entering
        s['legs_held'] &= zero & ~reset & ~entering
        s['braking'] &= zero & ~reset & ~entering
        safe = torch.ones_like(zero) if support is None else support
        slow = linear_velocity_b[:, :2].norm(dim=-1) < .08
        slow_legs = torch.ones_like(zero) if joint_velocity is None else joint_velocity[:, :12].abs().amax(-1) < 1.
        slow_rotation = torch.ones_like(zero) if angular_velocity is None else angular_velocity.norm(dim=-1) < .3
        candidate = zero & safe & slow & slow_legs & slow_rotation
        s['stable_ticks'] = torch.where(candidate & ~entering & ~reset, s['stable_ticks']+1., 0.)
        s['braking'] |= zero & (s['stable_ticks'] >= 3)
        capture = zero & ~s['held'] & s['braking'] & safe & slow
        refresh = ~s['held'] | capture
        s['joints'][:, 12:] = torch.where(refresh[:, None], joint_pos[:, 12:], s['joints'][:, 12:])
        s['raw_age'] = torch.where(entering | ~zero, 0., s['raw_age']+self.dt)
        s['xy_latched'] &= zero & ~entering & ~reset
        latch_xy = zero & (s['raw_age'] >= .4) & ~s['xy_latched']
        s['xy'] = torch.where((entering | latch_xy | ~zero)[:, None], xy, s['xy'])
        s['xy_latched'] |= latch_xy
        s['yaw'] = torch.where(entering | ~zero, yaw, s['yaw'])
        s['age'] = torch.where(entering | ~zero | ~s['braking'], 0., s['age']+self.dt)
        s['held'] |= capture
        planar = torch.ones_like(zero) if planar_support is None else planar_support
        leg_capture = zero & s['held'] & safe & planar & slow & slow_legs & ~s['legs_held']
        refresh_legs = ~s['legs_held'] | leg_capture
        s['joints'][:, :12] = torch.where(refresh_legs[:, None], joint_pos[:, :12], s['joints'][:, :12])
        s['leg_age'] = torch.where(leg_capture | ~s['legs_held'], 0., s['leg_age']+self.dt)
        s['legs_held'] |= leg_capture
        s['was_zero'] = zero.clone()
        delta = s['xy']-xy
        c, sn = torch.cos(yaw), torch.sin(yaw)
        error_b = torch.stack((c*delta[:, 0]+sn*delta[:, 1], -sn*delta[:, 0]+c*delta[:, 1]), -1)
        yaw_error = torch.atan2(torch.sin(s['yaw']-yaw), torch.cos(s['yaw']-yaw))
        wheel_error = s['joints'][:, 12:]-joint_pos[:, 12:]
        leg_reference = (s['joints'][:, :12]-joint_pos.new_tensor(DEFAULT_LEGS))/joint_pos.new_tensor(LEG_SCALE)
        return torch.cat((zero.float()[:, None], (((s['age']+.02)/.2).clamp(0., 1.)*s['braking'])[:, None],
                          wheel_error.clamp(-1., 1.), error_b.clamp(-.5, .5),
                          linear_velocity_b[:, :2].clamp(-2., 2.), yaw_error[:, None],
                          leg_reference, (((s['leg_age']+.02)/.2).clamp(0., 1.)*s['legs_held'])[:, None]), -1)


def stable_support(wheels, force, xy, velocity_world):
    """Require the velocity-shifted base projection inside loaded wheel support."""
    loaded = (force[..., 2] > 5.) & (force[..., 2] > .7*force.norm(dim=-1))
    point = xy+velocity_world[:, :2]/4.5
    p = wheels[..., :2]-point[:, None, :]
    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    def inside(poly):
        edge = torch.roll(poly, -1, dims=-2)-poly
        side = cross(edge, -poly)
        margin = .005*edge.norm(dim=-1)
        return (side >= margin).all(-1) | (side <= -margin).all(-1)
    quad = inside(p[:, [0, 1, 3, 2]]) & loaded.all(-1)
    triangles = p[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]]
    tri_loaded = loaded[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].all(-1)
    return quad | (inside(triangles) & tri_loaded).any(-1)


def holding_action(policy, control):
    """Pose hold plus wheel angular spring and world-position damping.

    Positive wheel angular velocity drives body -X in the M20 contract. Wheel
    targets may remain nonzero to generate static supporting force on stairs.
    """
    desired_vx = (1.*control[:, 6]-.2*control[:, 8]).clamp(-.2, .2)
    # The old policy angular observation is deliberately noisy during training.
    # Wheel servo damping supplies rate damping; use the clean latched yaw error.
    desired_yaw = (1.2*control[:, 10]).clamp(-.3, .3)
    lever = policy.new_tensor([-.28, .28, -.28, .28])
    desired_wheel_linear = desired_vx[:, None]+desired_yaw[:, None]*lever
    angular_target = (2.*control[:, 2:6]-desired_wheel_linear/.09).clamp(-8., 8.)
    return torch.cat((control[:, 11:23], angular_target/5.), -1)


class StopBranchMLPModel(MLPModel):
    frozen_motion = True
    uses_stop_control = True

    def __init__(self, obs, obs_groups, obs_set, output_dim, stop_std_factor=.1,
                 stop_hidden_dims=(128, 64), **kwargs):
        groups = copy.deepcopy(obs_groups)
        groups[obs_set] = ['policy']
        super().__init__(obs, groups, obs_set, output_dim, **kwargs)
        if self.obs_dim != 244 or output_dim != 16 or obs['stop_control'].shape[-1] != STOP_DIM:
            raise ValueError('StopBranch requires policy244 + stop_control24 -> action16')
        if not 0 < stop_std_factor <= 1:
            raise ValueError('Invalid stop sampling factor')
        self.stop_std_factor = stop_std_factor
        self.stop_hidden_dims = tuple(stop_hidden_dims)
        self.mlp.requires_grad_(False)
        self.obs_normalizer.requires_grad_(False)
        self.distribution.requires_grad_(False)
        dims = (244+STOP_DIM, *self.stop_hidden_dims, 16)
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ELU())
        self.stop_mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.stop_mlp[-1].weight)
        nn.init.zeros_(self.stop_mlp[-1].bias)
        self._rollout_stats = None

    def load_motion_state(self, state):
        expected = set(self.state_dict())-set(k for k in self.state_dict() if k.startswith('stop_mlp.'))
        if set(state) != expected:
            raise ValueError('Motion actor initialization does not match the frozen v1 architecture')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith('stop_mlp.') for k in result.missing_keys):
            raise ValueError('Unexpected motion initialization keys')

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            raise ValueError('StopBranch is a nonrecurrent single-device policy')
        control = obs['stop_control']
        self._last_control = control.detach()
        zero = control[:, 0] > .5
        motion = super().forward(obs, stochastic_output=False)
        physical_scale = motion.new_tensor(LEG_SCALE+[5.]*4)
        residual_limits = motion.new_tensor([.15, .6, .8]*4+[10.]*4)
        bounded = torch.tanh(self.stop_mlp(torch.cat((obs['policy'], control), -1)))
        residual = bounded*residual_limits/physical_scale
        self._last_stop_regularization = (bounded.square().mean(-1)*zero).sum()/zero.sum().clamp_min(1)
        target = holding_action(obs['policy'], control)+residual
        blend = control[:, 1:2]
        leg_blend = control[:, 23:24]
        stop = torch.cat(((1.-leg_blend)*motion[:, :12]+leg_blend*target[:, :12]+(1.-leg_blend)*residual[:, :12],
                          (1.-blend)*motion[:, 12:]+blend*target[:, 12:]+(1.-blend)*residual[:, 12:]), -1)
        mean = torch.where(zero[:, None], stop, motion)
        if not stochastic_output:
            return mean
        self.distribution.update(mean)
        sigma_scale = torch.where(zero, self.stop_std_factor, 1.)[:, None]
        self.distribution._distribution = Normal(mean, self.distribution.std*sigma_scale)
        return self.distribution.sample()

    def stop_regularization(self):
        return self._last_stop_regularization

    @torch.no_grad()
    def record_rollout_sample(self, actions):
        control = self._last_control
        zero = control[:, 0] > .5
        mean = self.output_distribution_params[0]
        physical = mean.new_tensor(LEG_SCALE+[.45]*4)
        noise = ((actions-mean)*physical).square()
        values = torch.stack((zero.sum(), (zero & (control[:, 1] >= .99)).sum(),
                              (zero & (control[:, 23] >= .99)).sum(),
                              (noise[:, :12].mean(-1)*zero).sum(), (noise[:, 12:].mean(-1)*zero).sum())).float()
        self._rollout_stats = values if self._rollout_stats is None else self._rollout_stats+values

    def take_rollout_metrics(self):
        result = {'stop_std_factor': self.stop_std_factor}
        if self._rollout_stats is not None:
            count, wheels, legs, leg_noise, wheel_noise = self._rollout_stats.cpu().tolist()
            if count:
                result.update(stop_wheel_hold_fraction=wheels/count, stop_leg_hold_fraction=legs/count,
                              stop_sample_noise_rms_rad=(leg_noise/count)**.5,
                              stop_sample_noise_rms_m_s=(wheel_noise/count)**.5)
        self._rollout_stats = None
        return result

    def as_jit(self):
        raise NotImplementedError('StopBranch requires policy + stop_control; use the checkpoint MuJoCo loader')

    def as_onnx(self, verbose=False):
        raise NotImplementedError('StopBranch requires explicit stop_control inputs; no legacy-only export')
