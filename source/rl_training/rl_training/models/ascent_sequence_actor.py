"""Frozen full stage3 policy plus a zero-initialized ascent/exit correction."""
import torch
from torch import nn
from torch.distributions import Normal
from .ascent_landing_actor import AscentLandingMLPModel, entrance_gate
from ..algorithms.initialization.ascent_sequence_init import frozen_source_digest

SEQUENCE_LIMITS = [.16, 1.92, 1.92]*2 + [.12, .48, .48]*2 + [.12]*4


def sequence_gate(policy, command):
    # Include rear scan edges: the correction must remain available when the
    # front wheels have reached the platform but the rear wheels are climbing.
    profile = policy[:, 57:244].reshape(-1, 11, 17)[:, 3:8].median(dim=1).values
    rising = ((profile[:, :-1]-profile[:, 1:]-.045)/.055).clamp(0., 1.).amax(-1)
    straight = ((command[:, 0]-.1)/.2).clamp(0., 1.)
    straight *= (1.-command[:, 1].abs()/.15).clamp(0., 1.)
    straight *= (1.-command[:, 2].abs()/.2).clamp(0., 1.)
    # Full authority at entry, one fifth elsewhere. No tread/phase labels
    # enter the actor; entirely flat scan eventually switches the branch off.
    return rising*straight*(.2+.8*entrance_gate(policy, command))


class AscentSequenceMLPModel(AscentLandingMLPModel):
    uses_sequence = True

    def __init__(self, *args, sequence_hidden_dims=(128, 64), sequence_limits=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)
        dims = (247, *sequence_hidden_dims, 16)
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ELU())
        self.sequence_mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.sequence_mlp[-1].weight)
        nn.init.zeros_(self.sequence_mlp[-1].bias)
        values = SEQUENCE_LIMITS if sequence_limits is None else sequence_limits
        if len(values) != 16 or any(not 0. < v <= 2. for v in values):
            raise ValueError('Invalid sequence correction limits')
        self.register_buffer('sequence_limits', torch.tensor(values, dtype=torch.float32))
        self._sequence_rollout_sum = None

    def load_stage3_checkpoint(self, state):
        expected = {k for k in self.state_dict() if not k.startswith('sequence_mlp.') and k != 'sequence_limits'}
        if set(state) != expected:
            raise ValueError('Stage4 must inherit the complete stage3 actor')
        self.load_state_dict(state, strict=False)

    def sequence_source_digest(self):
        return frozen_source_digest(self.state_dict())

    def source_state_digest(self):
        from ..algorithms.initialization.ascent_landing_init import frozen_source_digest as stage2_digest
        return stage2_digest({k: v for k, v in self.state_dict().items()
                              if not k.startswith('sequence_mlp.') and k != 'sequence_limits'})

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        inherited = super().forward(obs, masks=masks, hidden_state=hidden_state, stochastic_output=False)
        gate = sequence_gate(obs['policy'], obs['motion_control'])
        bounded = torch.tanh(self.sequence_mlp(torch.cat((obs['policy'], obs['motion_control']), -1)))
        correction = bounded*self.sequence_limits*gate[:, None]
        mean = inherited+correction
        self._sequence_gate, self._sequence_bounded = gate.detach(), bounded.detach()
        self._last_motion_regularization = (bounded.square().mean(-1)*gate).sum()/gate.sum().clamp_min(1.)
        self._sequence_metrics = torch.stack((gate.mean(), (bounded.abs() > .95).float().mean(),
                                              correction.square().mean().sqrt())).detach()
        if stochastic_output:
            self.distribution.update(mean)
            self.distribution._distribution = Normal(mean, self.distribution.std)
            return self.distribution.sample()
        return mean

    @torch.no_grad()
    def record_rollout_sample(self, actions):
        super().record_rollout_sample(actions)
        self._sequence_rollout_sum = (self._sequence_metrics.clone() if self._sequence_rollout_sum is None
                                     else self._sequence_rollout_sum+self._sequence_metrics)
        self._sequence_rollout_count = getattr(self, '_sequence_rollout_count', 0)+1

    def take_rollout_metrics(self):
        result = super().take_rollout_metrics()
        if self._sequence_rollout_sum is not None:
            values = (self._sequence_rollout_sum/max(self._sequence_rollout_count, 1)).cpu().tolist()
            result.update(zip(('sequence_gate_mean', 'sequence_saturation_fraction', 'sequence_correction_rms'), values))
            self._sequence_rollout_sum, self._sequence_rollout_count = None, 0
        return result
