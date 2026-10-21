"""Continue the stage4 correction with modest scan-gated training exploration."""
import torch
from torch.distributions import Normal
from .ascent_sequence_actor import AscentSequenceMLPModel


class AscentAdaptiveMLPModel(AscentSequenceMLPModel):
    uses_adaptive_ascent = True

    def __init__(self, *args, adaptive_exploration=None, **kwargs):
        super().__init__(*args, **kwargs)
        values = ([.015, .12, .12]*2+[.01, .04, .04]*2+[.04]*4
                  if adaptive_exploration is None else adaptive_exploration)
        if len(values) != 16 or any(not 0. <= v <= .2 for v in values):
            raise ValueError('Invalid stage5 exploration')
        # Configuration, not a trained/frozen source parameter. State keys and
        # deterministic inference remain exactly compatible with stage4.
        self.register_buffer('_adaptive_exploration', torch.tensor(values), persistent=False)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        mean = super().forward(obs, masks=masks, hidden_state=hidden_state, stochastic_output=False)
        if not stochastic_output:
            return mean
        self.distribution.update(mean)
        sigma = torch.sqrt(self.distribution.std.square()+
                           (self._sequence_gate[:, None]*self._adaptive_exploration).square())
        self.distribution._distribution = Normal(mean, sigma)
        return self.distribution.sample()
