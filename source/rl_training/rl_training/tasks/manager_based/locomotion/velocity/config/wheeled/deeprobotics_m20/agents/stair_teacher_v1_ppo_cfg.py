"""PPO defaults for the independently registered v1 task."""
from copy import deepcopy
from isaaclab.utils import configclass
from .stair_teacher_ppo_cfg import DeeproboticsM20StairTeacherPPORunnerCfg


@configclass
class DeeproboticsM20StairTeacherV1PPORunnerCfg(DeeproboticsM20StairTeacherPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm=deepcopy(DeeproboticsM20StairTeacherPPORunnerCfg().algorithm)
        self.experiment_name='deeprobotics_m20_stair_teacher_v1'
