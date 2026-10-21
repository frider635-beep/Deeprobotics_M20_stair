"""Registered v1 environment installs its physics observer for train and play."""
from isaaclab.envs import ManagerBasedRLEnv
from ..mdp.ascent.ascent_entry_mdp import install_physics_observer


class AscentEntryRLEnv(ManagerBasedRLEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if getattr(self.cfg, 'm20_refinement_profile', None) == 'ascent_entry_v1_stage7':
            from ..mdp.ascent.ascent_unified_mdp import install_physics_observer as install_unified
            install_unified(self)
        elif getattr(self.cfg, 'm20_refinement_profile', None) == 'ascent_entry_v1_stage6':
            from ..mdp.ascent.ascent_alternating_mdp import install_physics_observer as install_alternating
            install_alternating(self)
        elif getattr(self.cfg, 'm20_refinement_profile', None) == 'ascent_entry_v1_stage5':
            from ..mdp.ascent.ascent_adaptive_mdp import install_physics_observer as install_adaptive
            install_adaptive(self)
        elif getattr(self.cfg, 'm20_refinement_profile', None) == 'ascent_entry_v1_stage4':
            from ..mdp.ascent.ascent_sequence_mdp import install_physics_observer as install_sequence
            install_sequence(self)
        elif getattr(self.cfg, 'm20_refinement_profile', None) in ('ascent_entry_v1_stage2', 'ascent_entry_v1_stage3'):
            from ..mdp.ascent.ascent_entry_stage2 import install_physics_observer as install_stage2
            install_stage2(self)
        else:
            install_physics_observer(self)
