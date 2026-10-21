"""Isaac manager lifecycle adapter for stateful Stage6 positive reward terms."""
import copy

from isaaclab.managers import ManagerTermBase

from .ascent_alternating_mdp import positive_reward, resolve_reward_params


class AlternatingPositiveReward(ManagerTermBase):
    """Keep construction and episode reset of the original class-based reward."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        source_cfg = copy.copy(cfg)
        source_cfg.func = cfg.params['source_func']
        source_cfg.params = cfg.params['source_params']
        resolve_reward_params(env, source_cfg.params)
        self.source = source_cfg.func(cfg=source_cfg, env=env)

    def __call__(self, env, source_func, source_params, require_progress=False):
        return positive_reward(env, self.source, source_params, require_progress)

    def reset(self, env_ids=None):
        self.source.reset(env_ids=env_ids)
