"""Registered v1 ascent-entry task, isolated from the existing v0 defaults."""
from isaaclab.utils import configclass
from .stair_teacher_env_cfg import DeeproboticsM20StairTeacherEnvCfg
from .agents.stair_teacher_ppo_cfg import DeeproboticsM20StairTeacherPPORunnerCfg


@configclass
class DeeproboticsM20StairTeacherV1EnvCfg(DeeproboticsM20StairTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.ascent_entry_profile import apply_profile
        apply_profile(self,DeeproboticsM20StairTeacherPPORunnerCfg())
