"""Frozen original motion baseline plus one trainable full-ascent correction."""
import hashlib
import torch
from torch import nn
from torch.distributions import Normal
from .motion_residual_actor import MotionResidualMLPModel

PROFILE = 'ascent_entry_v1_stage7'
VERSION = 'm20_ascent_unified_20261021_v1_stage7'
# Sum of the three old correction envelopes. A uniform gate contains all three
# old gates; migration can fit their sum without clipping the target actions.
UNIFIED_LIMITS = [.44, 4.08, 4.08]*2+[.36, 1.20, 1.20]*2+[.36]*4
EXPLORATION = [.012, .06, .06]*2+[.008, .025, .025]*2+[.03]*4


def unified_gate(policy, command):
    profile = policy[:, 57:244].reshape(-1, 11, 17)[:, 3:8].median(dim=1).values
    rising = ((profile[:, :-1]-profile[:, 1:]-.045)/.055).clamp(0., 1.).amax(-1)
    straight = ((command[:, 0]-.1)/.2).clamp(0., 1.)
    straight *= (1.-command[:, 1].abs()/.15).clamp(0., 1.)
    straight *= (1.-command[:, 2].abs()/.2).clamp(0., 1.)
    return rising*straight


def baseline_digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        if key.startswith('unified_mlp.') or key == 'unified_limits': continue
        h.update(key.encode()); h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class AscentUnifiedMLPModel(MotionResidualMLPModel):
    uses_ascent_entry = True
    uses_merged_ascent = True

    def __init__(self, *args, unified_hidden_dims=(256, 128), unified_limits=None,
                 adaptive_exploration=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)
        dims = (247, *unified_hidden_dims, 16)
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2: layers.append(nn.ELU())
        self.unified_mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.unified_mlp[-1].weight); nn.init.zeros_(self.unified_mlp[-1].bias)
        limits = UNIFIED_LIMITS if unified_limits is None else unified_limits
        exploration = EXPLORATION if adaptive_exploration is None else adaptive_exploration
        if len(limits) != 16 or any(not 0. < x <= 5. for x in limits):
            raise ValueError('Invalid unified ascent envelope')
        if len(exploration) != 16 or any(not 0. <= x <= .2 for x in exploration):
            raise ValueError('Invalid unified ascent exploration')
        self.register_buffer('unified_limits', torch.tensor(limits, dtype=torch.float32))
        self.register_buffer('_adaptive_exploration', torch.tensor(exploration), persistent=False)
        self._unified_metrics = None

    def load_motion_checkpoint(self, state):
        base = {k:v for k,v in state.items() if not k.startswith(('entry_mlp.', 'landing_mlp.', 'sequence_mlp.'))
                and k not in ('entry_limits', 'landing_limits', 'sequence_limits')}
        result = self.load_state_dict(base, strict=False)
        if result.unexpected_keys or any(not (k.startswith('unified_mlp.') or k == 'unified_limits') for k in result.missing_keys):
            raise ValueError('Unified initialization requires the complete frozen motion baseline')

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        baseline = super().forward(obs, masks, hidden_state, stochastic_output=False)
        gate = unified_gate(obs['policy'], obs['motion_control'])
        bounded = torch.tanh(self.unified_mlp(torch.cat((obs['policy'], obs['motion_control']), -1)))
        correction = gate[:, None]*self.unified_limits*bounded
        mean = baseline+correction
        self._unified_gate, self._unified_bounded = gate.detach(), bounded.detach()
        self._last_motion_regularization = (bounded.square().mean(-1)*gate).sum()/gate.sum().clamp_min(1.)
        self._step_unified_metrics = torch.stack((gate.mean(), (bounded.abs() > .95).float().mean(),
                                                 correction.square().mean().sqrt())).detach()
        if stochastic_output:
            self.distribution.update(mean)
            std = torch.sqrt(self.distribution.std.square()+(gate[:, None]*self._adaptive_exploration).square())
            self.distribution._distribution = Normal(mean, std)
            return self.distribution.sample()
        return mean

    @torch.no_grad()
    def record_rollout_sample(self, actions):
        super().record_rollout_sample(actions)
        self._unified_metrics = (self._step_unified_metrics.clone() if self._unified_metrics is None
                                 else self._unified_metrics+self._step_unified_metrics)
        self._unified_count = getattr(self, '_unified_count', 0)+1

    def take_rollout_metrics(self):
        result = super().take_rollout_metrics()
        if self._unified_metrics is not None:
            values = (self._unified_metrics/self._unified_count).cpu().tolist()
            result.update(zip(('unified_gate_mean', 'unified_saturation_fraction', 'unified_correction_rms'), values))
            self._unified_metrics = None; self._unified_count = 0
        return result
