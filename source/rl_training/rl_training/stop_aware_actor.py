"""Training-only stop exploration; deterministic actor stays 244D -> 16D.

RSL-RL 5.0.1's Gaussian properties/log_prob all read _distribution. Updating
that distribution before sampling also makes PPO recomputation use the same
conditional std. Stop metadata is stored in rollouts, never fed to the MLP.
"""
from __future__ import annotations

import math
import torch
from torch.distributions import Normal
from rsl_rl.models import MLPModel


class StopAwareMLPModel(MLPModel):
    def __init__(self, *args, stop_std_initial=0.5, stop_std_final=0.1,
                 stop_std_anneal_updates=500, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < stop_std_final <= stop_std_initial <= 1 or stop_std_anneal_updates < 1:
            raise ValueError("Invalid stop exploration schedule")
        if self.distribution is None or not hasattr(self.distribution, "log_std_param"):
            raise ValueError("StopAwareMLPModel requires a log Gaussian distribution")
        self.stop_std_initial = stop_std_initial
        self.stop_std_final = stop_std_final
        self.stop_std_anneal_updates = stop_std_anneal_updates
        self.stop_std_factor = stop_std_initial
        self._history = None
        self._rollout_stats = None

    def set_update_iteration(self, iteration):
        progress = min(max(float(iteration) / self.stop_std_anneal_updates, 0.), 1.)
        self.stop_std_factor = self.stop_std_initial + progress * (self.stop_std_final - self.stop_std_initial)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        mean = super().forward(obs, masks=masks, hidden_state=hidden_state, stochastic_output=False)
        if not stochastic_output:
            return mean
        if "stop_metadata" not in obs:
            raise ValueError("Stochastic stop policy requires raw stop_metadata in the rollout")
        stop = obs["stop_metadata"].squeeze(-1) > .5
        if stop.shape != mean.shape[:-1]:
            raise ValueError("Stop metadata must match the action batch")
        self.distribution.update(mean)
        scale = torch.where(stop, self.stop_std_factor, 1.).unsqueeze(-1)
        self.distribution._distribution = Normal(mean, self.distribution.std * scale)
        self._last_stop = stop
        return self.distribution.sample()

    @torch.no_grad()
    def record_rollout_sample(self, actions):
        """Separate mean-target changes from sampled exploration noise."""
        mean = self.output_distribution_params[0].detach()
        stop = self._last_stop
        scale = mean.new_tensor([.125, .25, .25] * 4 + [.45] * 4)
        targets, sampled = mean * scale, actions * scale
        h = self._history
        if h is None or h["previous"].shape != targets.shape:
            h = dict(previous=targets.clone(), older=targets.clone(),
                     count=torch.zeros_like(stop, dtype=torch.long))
            self._history = h
        second = targets - 2 * h["previous"] + h["older"]
        valid = stop & (h["count"] >= 2)
        if self._rollout_stats is None:
            self._rollout_stats = {k: mean.new_zeros(()) for k in
                ("stop_count", "history_count", "leg_noise", "wheel_noise", "leg_second", "wheel_second")}
        s = self._rollout_stats
        s["stop_count"] += stop.sum()
        s["history_count"] += valid.sum()
        for key, values, mask in (
            ("leg_noise", (sampled[:, :12] - targets[:, :12]).square().mean(-1), stop),
            ("wheel_noise", (sampled[:, 12:] - targets[:, 12:]).square().mean(-1), stop),
            ("leg_second", second[:, :12].square().mean(-1), valid),
            ("wheel_second", second[:, 12:].square().mean(-1), valid),
        ):
            s[key] += (values * mask).sum()
        h.update(older=h["previous"].clone(), previous=targets.clone(),
                 count=torch.where(stop, h["count"] + 1, 0))

    def reset(self, dones=None):
        super().reset(dones)
        if self._history is not None and dones is not None:
            self._history["count"][dones.bool().reshape(-1)] = 0

    def take_rollout_metrics(self):
        result = {"stop_std_factor": self.stop_std_factor}
        s = self._rollout_stats
        if s is not None:
            for key, name, denominator in (
                ("leg_noise", "stop_sample_noise_rms_rad", "stop_count"),
                ("wheel_noise", "stop_sample_noise_rms_m_s", "stop_count"),
                ("leg_second", "stop_mean_leg_second_difference_rms_rad", "history_count"),
                ("wheel_second", "stop_mean_wheel_second_difference_rms_m_s", "history_count"),
            ):
                if s[denominator].item() > 0:
                    result[name] = math.sqrt((s[key] / s[denominator]).item())
        self._rollout_stats = None
        return result
