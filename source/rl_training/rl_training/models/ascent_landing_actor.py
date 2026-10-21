"""Freeze the complete stage2 policy; add a zero-output entrance correction.

Same policy244 + raw_command3 inputs. Extra authority is restricted to a
scan-derived entrance gate, never a privileged tread counter or phase input.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal
from .ascent_entry_actor import AscentEntryMLPModel, ascent_gate
from ..algorithms.initialization.ascent_landing_init import frozen_source_digest


LANDING_LIMITS = [.16, 1.92, 1.92]*2 + [.12, .48, .48]*2 + [.12]*4


def entrance_gate(policy, command):
    profile = policy[:, 57:244].reshape(-1, 11, 17)[:, 3:8].median(dim=1).values
    rear_edges = ((profile[:, :8]-profile[:, 1:9]-.045)/.055).clamp(0., 1.).sum(-1)
    early = (2.-rear_edges).clamp(0., 1.)
    early = early.square()*(3.-2.*early)
    return ascent_gate(policy, command)*early


class AscentLandingMLPModel(AscentEntryMLPModel):
    uses_landing = True

    def __init__(self, *args, landing_hidden_dims=(128, 64), landing_limits=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)
        dims = (247, *landing_hidden_dims, 16)
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ELU())
        self.landing_mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.landing_mlp[-1].weight)
        nn.init.zeros_(self.landing_mlp[-1].bias)
        limits = LANDING_LIMITS if landing_limits is None else landing_limits
        if len(limits) != 16 or any(not 0. < value <= 2. for value in limits):
            raise ValueError('Invalid entrance landing correction limits')
        self.register_buffer('landing_limits', torch.tensor(limits, dtype=torch.float32))
        self._landing_rollout_sum = None

    def load_stage2_checkpoint(self, state):
        expected = {key for key in self.state_dict()
                    if not key.startswith('landing_mlp.') and key != 'landing_limits'}
        if set(state) != expected:
            raise ValueError('Stage3 must inherit the complete stage2 actor including entry_mlp')
        self.load_state_dict(state, strict=False)

    def source_state_digest(self):
        return frozen_source_digest(self.state_dict())

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        inherited = super().forward(obs, masks=masks, hidden_state=hidden_state, stochastic_output=False)
        policy, command = obs['policy'], obs['motion_control']
        bounded = torch.tanh(self.landing_mlp(torch.cat((policy, command), -1)))
        gate = entrance_gate(policy, command)
        correction = bounded*self.landing_limits*gate[:, None]
        mean = inherited+correction
        self._landing_bounded, self._landing_gate = bounded.detach(), gate.detach()
        self._last_motion_regularization = (bounded.square().mean(-1)*gate).sum()/gate.sum().clamp_min(1.)
        self._landing_metrics = torch.stack((gate.mean(), (bounded.abs() > .95).float().mean(),
                                            correction.square().mean().sqrt())).detach()
        if stochastic_output:
            self.distribution.update(mean)
            self.distribution._distribution = Normal(mean, self.distribution.std)
            return self.distribution.sample()
        return mean

    @torch.no_grad()
    def record_rollout_sample(self, actions):
        super().record_rollout_sample(actions)
        self._landing_rollout_sum = (self._landing_metrics.clone() if self._landing_rollout_sum is None
                                    else self._landing_rollout_sum+self._landing_metrics)
        self._landing_rollout_count = getattr(self, '_landing_rollout_count', 0)+1

    def take_rollout_metrics(self):
        metrics = super().take_rollout_metrics()
        if self._landing_rollout_sum is not None:
            values = (self._landing_rollout_sum/max(self._landing_rollout_count, 1)).cpu().tolist()
            metrics.update(zip(('landing_gate_mean', 'landing_saturation_fraction', 'landing_correction_rms'), values))
            self._landing_rollout_sum, self._landing_rollout_count = None, 0
        return metrics
