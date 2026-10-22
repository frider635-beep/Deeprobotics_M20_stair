"""Stage8: one conditioned correction MLP for ascent AND descent.

The two inherited motion MLPs remain frozen. Six columns extend the existing
unified MLP input. Descent uses its SAME 16 outputs, with a bounded trainable
output gain initialized to zero; no extra policy network or output head.
"""
import torch
from torch import nn
from torch.distributions import Normal
from ..ascent_unified_actor import AscentUnifiedMLPModel, UNIFIED_LIMITS
from ..motion_residual_actor import MotionResidualMLPModel
from ...control.stair_path import PATH_DIM, stair_edges, straight_gate

PROFILE = 'ascent_entry_v1_stage8'
VERSION = 'm20_stair_path_20261022_v1_stage8'
DESCENT_LIMITS = [.18, .60, .60]*2+[.18, .45, .45]*2+[.18]*4
DESCENT_EXPLORATION = [.008, .035, .035]*4+[.025]*4


class StairPathMLPModel(AscentUnifiedMLPModel):
    uses_path_control = True

    def __init__(self, *args, descent_limits=None, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.unified_mlp[0]
        first = nn.Linear(old.in_features+PATH_DIM, old.out_features)
        with torch.no_grad():
            first.weight.zero_(); first.bias.copy_(old.bias)
            first.weight[:, :247].copy_(old.weight)
        self.unified_mlp[0] = first
        self.descent_gain = nn.Parameter(torch.zeros(16))
        limits = DESCENT_LIMITS if descent_limits is None else descent_limits
        if len(limits) != 16 or any(not 0. < value <= 1. for value in limits):
            raise ValueError('Invalid descent correction envelope')
        self.register_buffer('descent_limits', torch.tensor(limits, dtype=torch.float32))
        self.register_buffer('_descent_exploration', torch.tensor(DESCENT_EXPLORATION), persistent=False)
        self._path_metrics = None

    def load_stage7_checkpoint(self, state):
        adapted = dict(state)
        first = self.unified_mlp[0].weight.detach().clone().zero_()
        if state['unified_mlp.0.weight'].shape != (256, 247):
            raise ValueError('Stage8 migration requires the complete Stage7 unified actor')
        first[:, :247] = state['unified_mlp.0.weight'].to(first.device)
        adapted['unified_mlp.0.weight'] = first
        adapted['descent_gain'] = self.descent_gain.detach().clone().zero_()
        adapted['descent_limits'] = self.descent_limits.detach().clone()
        self.load_state_dict(adapted, strict=True)

    @staticmethod
    def preservation_weights(observations):
        up, down = stair_edges(observations['policy'][:, 57:244])
        weight = torch.where((up <= .05) & (down > .05), .02, 1.)
        # The critic-only geometry weights a TRAINING loss; it never enters
        # the deployed actor. Do not anchor the known bad narrow-tread actions
        # as strongly as the successful default gait.
        if 'ascent_control' in observations:
            state = observations['ascent_control']
            ascent = state[:, 0] > 0.
            depth = state[:, 1]*.35
            narrow_weight = torch.where(depth < .225, .05, .15)
            weight = torch.where(ascent & (depth < .275), narrow_weight, weight)
        return weight

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        baseline = MotionResidualMLPModel.forward(self, obs, masks, hidden_state, stochastic_output=False)
        up, down = stair_edges(obs['policy'][:, 57:244])
        straight = straight_gate(obs['motion_control'])
        up_gate = up*straight
        # Do not mix a false descending edge into a scan dominated by ascent.
        down_gate = down*straight*(up <= .05)
        features = torch.cat((obs['policy'], obs['motion_control'], obs['path_control']), -1)
        bounded = torch.tanh(self.unified_mlp(features))
        scale = (up_gate[:, None]*self.unified_limits
                 +down_gate[:, None]*self.descent_limits*torch.tanh(self.descent_gain))
        correction = scale*bounded
        mean = baseline+correction
        self._unified_gate, self._unified_bounded = up_gate.detach(), bounded.detach()
        self._descent_gate = down_gate.detach()
        gate = torch.maximum(up_gate, down_gate)
        self._last_motion_regularization = (correction.square().mean(-1)*gate).sum()/gate.sum().clamp_min(1.)
        self._step_unified_metrics = torch.stack((up_gate.mean(), (bounded.abs() > .95).float().mean(),
                                                 correction.square().mean().sqrt())).detach()
        self._step_path_metrics = torch.stack((down_gate.mean(), torch.tanh(self.descent_gain).abs().mean(),
            (correction.square().mean(-1)*down_gate).sum().sqrt()/down_gate.sum().clamp_min(1.).sqrt())).detach()
        if stochastic_output:
            self.distribution.update(mean)
            extra = up_gate[:, None]*self._adaptive_exploration+down_gate[:, None]*self._descent_exploration
            std = torch.sqrt(self.distribution.std.square()+extra.square())
            self.distribution._distribution = Normal(mean, std)
            return self.distribution.sample()
        return mean

    @torch.no_grad()
    def record_rollout_sample(self, actions):
        super().record_rollout_sample(actions)
        self._path_metrics = (self._step_path_metrics.clone() if self._path_metrics is None
                              else self._path_metrics+self._step_path_metrics)
        self._path_count = getattr(self, '_path_count', 0)+1

    def take_rollout_metrics(self):
        result = super().take_rollout_metrics()
        if self._path_metrics is not None:
            result.update(zip(('descent_gate_mean', 'descent_gain_abs_mean', 'descent_correction_rms'),
                              (self._path_metrics/self._path_count).cpu().tolist()))
            self._path_metrics = None; self._path_count = 0
        return result
