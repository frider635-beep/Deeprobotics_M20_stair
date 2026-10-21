"""Ascent-only correction around the complete, frozen teacher_baseline_ actor.

Same policy244 + raw_command3 contract. No privileged phase or contact sensor
enters the actor. Legacy mlp/residual_mlp keys retain their original meanings.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.models import MLPModel


def ascent_gate(policy, command):
    """Continuous look-ahead gate; scan order matches 11y x 17x Isaac xy grid."""
    profile = policy[:, 57:244].reshape(-1, 11, 17)[:, 3:8].median(dim=1).values
    rise = profile[:, 8:-1] - profile[:, 9:]
    strength = ((rise-.045)/.055).clamp(0., 1.)
    proximity = policy.new_tensor([1., 1., 1., 1., 1., .9, .65, .25])
    geometry = (strength*proximity).amax(-1)
    forward = ((command[:, 0]-.1)/.2).clamp(0., 1.)
    straight = (1.-command[:, 1].abs()/.15).clamp(0., 1.) * (1.-command[:, 2].abs()/.2).clamp(0., 1.)
    return geometry * forward * straight


class AscentEntryMLPModel(MLPModel):
    frozen_motion = True
    uses_motion_control = True
    uses_ascent_entry = True

    def __init__(self, obs, obs_groups, obs_set, output_dim, residual_hidden_dims=(128, 64),
                 entry_hidden_dims=(128, 64), entry_limits=None, **kwargs):
        groups = copy.deepcopy(obs_groups)
        groups[obs_set] = ["policy"]
        super().__init__(obs, groups, obs_set, output_dim, **kwargs)
        if self.obs_dim != 244 or output_dim != 16 or obs['motion_control'].shape[-1] != 3:
            raise ValueError("AscentEntry requires policy244 + raw_command3 -> action16")
        def network(hidden):
            dims = (247, *hidden, 16)
            layers = []
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims)-2:
                    layers.append(nn.ELU())
            return nn.Sequential(*layers)
        self.residual_mlp = network(residual_hidden_dims)
        self.requires_grad_(False)
        self.entry_mlp = network(entry_hidden_dims)
        nn.init.zeros_(self.entry_mlp[-1].weight)
        nn.init.zeros_(self.entry_mlp[-1].bias)
        limits = entry_limits or [.12, .24, .24]*4 + [.12]*4
        if len(limits) != 16 or any(not 0 < v <= 1. for v in limits):
            raise ValueError("Invalid ascent correction limits")
        self.register_buffer('entry_limits', torch.tensor(limits, dtype=torch.float32))
        self._last_motion_regularization = None
        self._entry_metrics = None

    def load_motion_checkpoint(self, state):
        expected = set(self.state_dict()) - {k for k in self.state_dict() if k.startswith('entry_mlp.') or k == 'entry_limits'}
        if set(state) != expected:
            raise ValueError("Expected the complete baseline MLP AND its trained motion residual")
        self.load_state_dict(state, strict=False)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            raise ValueError("AscentEntry is nonrecurrent")
        policy, command = obs['policy'], obs['motion_control']
        latent = torch.cat((policy, command), -1)
        base = super().forward(obs, stochastic_output=False)
        old_bounded = torch.tanh(self.residual_mlp(latent))
        moving = command.abs().amax(-1) > 1.e-4
        legacy_limits = base.new_tensor([.18, .35, .35]*4 + [.16]*4)
        baseline = base + old_bounded*legacy_limits*moving[:, None]
        gate = ascent_gate(policy, command)
        bounded = torch.tanh(self.entry_mlp(latent))
        correction = bounded*self.entry_limits*gate[:, None]
        mean = baseline + correction
        self._last_motion_regularization = (bounded.square().mean(-1)*gate).sum()/gate.sum().clamp_min(1.)
        self._last_metrics = torch.stack((gate.mean(), (bounded.abs()>.95).float().mean(),
                                          correction.square().mean().sqrt(), (old_bounded.abs()>.95).float().mean())).detach()
        if stochastic_output:
            self.distribution.update(mean)
            self.distribution._distribution = Normal(mean, self.distribution.std)
            return self.distribution.sample()
        return mean

    def motion_regularization(self):
        return self._last_motion_regularization

    @torch.no_grad()
    def record_rollout_sample(self, actions):
        self._entry_metrics = self._last_metrics if self._entry_metrics is None else self._entry_metrics+self._last_metrics
        self._entry_count = getattr(self, '_entry_count', 0)+1

    def take_rollout_metrics(self):
        if self._entry_metrics is None:
            return {}
        values = (self._entry_metrics/max(self._entry_count, 1)).cpu().tolist()
        self._entry_metrics, self._entry_count = None, 0
        return dict(zip(('ascent_gate_mean', 'entry_saturation_fraction', 'entry_correction_rms',
                         'legacy_residual_saturation_fraction'), values))

    def as_jit(self):
        raise NotImplementedError("AscentEntry export requires policy + raw motion_control inputs")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("AscentEntry export requires policy + raw motion_control inputs")
