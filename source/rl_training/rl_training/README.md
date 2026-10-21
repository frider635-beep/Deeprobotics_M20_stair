# rl_training 工程结构

整理日期：2026-10-21。包根目录只保留 `__init__.py` 和本说明；代码按任务、网络、算法和工具分层。

```text
rl_training/
├── __init__.py                 # 兼容旧模块路径，然后注册 Gym 任务
├── algorithms/
│   ├── stair_guarded_ppo.py    # PPO 扩展
│   └── initialization/        # checkpoint 校验、初始化、示范训练挂接
├── models/                    # 策略网络与停止控制分支
├── utils/                     # 日志、指标、诊断、预检、兼容和 CPU 加载工具
├── actuator/                  # 执行器
├── assets/                    # 机器人资源配置
└── tasks/manager_based/locomotion/velocity/
    ├── velocity_env_cfg.py
    ├── envs/                  # 自定义运行环境 AscentEntryRLEnv
    ├── mdp/
    │   ├── stair_teacher.py   # 共用教师逻辑
    │   ├── ascent/            # 上楼奖励、逐阶账本、课程、物理观测器
    │   └── stop/              # 停止课程覆盖统计
    └── config/wheeled/deeprobotics_m20/
        ├── __init__.py        # M20 任务注册
        ├── *_env_cfg.py       # 环境配置
        ├── agents/            # PPO/runner 配置
        └── profiles/
            ├── *_profile.py  # 各阶段环境与训练参数覆盖
            └── data/         # ascent_reference.json、v1_stop_baseline.json
```

新包均有 `__init__.py`。包入口保持轻量；配置放在 `config`，任务奖励与课程放在 `mdp`，网络放在 `models`，通用训练扩展放在 `algorithms`。

## 导入与入口

新代码使用规范路径，例如：

```python
from rl_training.models.ascent_adaptive_actor import AscentAdaptiveMLPModel
from rl_training.algorithms.stair_guarded_ppo import StairGuardedPPO
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_adaptive_mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles import ascent_adaptive_profile
```

训练、play、Gym 注册和配置内的网络类路径均已更新。现有任务 ID、profile 名称及 `scripts/train_ascent_stage5.sh` 命令保持不变。

`utils/legacy_imports.py` 为历史 checkpoint、YAML 和脚本提供惰性模块别名，例如旧的 `rl_training.ascent_sequence_actor` 指向 `rl_training.models.ascent_sequence_actor`；不复制实现，也不在根目录留下转发文件。新代码不要使用旧路径。

MuJoCo 部署和 CPU 检查通过 `utils/module_loader.py` 加载同一份代码。该入口使用独立命名空间跳过 Isaac/Kit 任务注册，不改变正常训练的注册流程。工具可通过 `deploy_mujoco.training_module(name)` 和 `training_source(name)` 访问模块与源码路径。

`utils/module_layout.py` 集中维护历史名称与新路径，以及仓库和 JSON 资源的位置。训练源码哈希、采集来源记录均使用迁移后的真实路径。既有实验记录和历史源码快照保留原样。

安装配置使用 `find_packages()` 收集子包，并明确打包两份 JSON 数据；已有源码 editable 安装可继续使用。

## 迁移对照

| 原根目录文件 | 新位置 |
| --- | --- |
| `ascent_adaptive.py` | [tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_adaptive.py](tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_adaptive.py) |
| `ascent_adaptive_actor.py` | [models/ascent_adaptive_actor.py](models/ascent_adaptive_actor.py) |
| `ascent_adaptive_demo.py` | [algorithms/initialization/ascent_adaptive_demo.py](algorithms/initialization/ascent_adaptive_demo.py) |
| `ascent_adaptive_init.py` | [algorithms/initialization/ascent_adaptive_init.py](algorithms/initialization/ascent_adaptive_init.py) |
| `ascent_adaptive_mdp.py` | [tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_adaptive_mdp.py](tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_adaptive_mdp.py) |
| `ascent_adaptive_profile.py` | [tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_adaptive_profile.py](tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_adaptive_profile.py) |
| `ascent_entry_actor.py` | [models/ascent_entry_actor.py](models/ascent_entry_actor.py) |
| `ascent_entry_env.py` | [tasks/manager_based/locomotion/velocity/envs/ascent_entry_env.py](tasks/manager_based/locomotion/velocity/envs/ascent_entry_env.py) |
| `ascent_entry_mdp.py` | [tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_entry_mdp.py](tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_entry_mdp.py) |
| `ascent_entry_profile.py` | [tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_entry_profile.py](tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_entry_profile.py) |
| `ascent_entry_stage2.py` | [tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_entry_stage2.py](tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_entry_stage2.py) |
| `ascent_entry_stage2_init.py` | [algorithms/initialization/ascent_entry_stage2_init.py](algorithms/initialization/ascent_entry_stage2_init.py) |
| `ascent_entry_stage2_profile.py` | [tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_entry_stage2_profile.py](tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_entry_stage2_profile.py) |
| `ascent_landing_actor.py` | [models/ascent_landing_actor.py](models/ascent_landing_actor.py) |
| `ascent_landing_init.py` | [algorithms/initialization/ascent_landing_init.py](algorithms/initialization/ascent_landing_init.py) |
| `ascent_landing_mdp.py` | [tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_landing_mdp.py](tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_landing_mdp.py) |
| `ascent_landing_profile.py` | [tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_landing_profile.py](tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_landing_profile.py) |
| `ascent_reward_audit.py` | [tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_reward_audit.py](tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_reward_audit.py) |
| `ascent_sequence.py` | [tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_sequence.py](tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_sequence.py) |
| `ascent_sequence_actor.py` | [models/ascent_sequence_actor.py](models/ascent_sequence_actor.py) |
| `ascent_sequence_init.py` | [algorithms/initialization/ascent_sequence_init.py](algorithms/initialization/ascent_sequence_init.py) |
| `ascent_sequence_mdp.py` | [tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_sequence_mdp.py](tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_sequence_mdp.py) |
| `ascent_sequence_profile.py` | [tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_sequence_profile.py](tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_sequence_profile.py) |
| `fast_logging.py` | [utils/fast_logging.py](utils/fast_logging.py) |
| `m20_preflight.py` | [utils/m20_preflight.py](utils/m20_preflight.py) |
| `metric_aggregation.py` | [utils/metric_aggregation.py](utils/metric_aggregation.py) |
| `motion_residual_actor.py` | [models/motion_residual_actor.py](models/motion_residual_actor.py) |
| `stair_guarded_ppo.py` | [algorithms/stair_guarded_ppo.py](algorithms/stair_guarded_ppo.py) |
| `stop_aware_actor.py` | [models/stop_aware_actor.py](models/stop_aware_actor.py) |
| `stop_branch.py` | [models/stop_branch.py](models/stop_branch.py) |
| `stop_coverage.py` | [tasks/manager_based/locomotion/velocity/mdp/stop/stop_coverage.py](tasks/manager_based/locomotion/velocity/mdp/stop/stop_coverage.py) |
| `teacher_diagnostics.py` | [utils/teacher_diagnostics.py](utils/teacher_diagnostics.py) |
| `v1_stop_profile.py` | [tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/v1_stop_profile.py](tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/v1_stop_profile.py) |
| `ascent_reference.json` | [tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/data/ascent_reference.json](tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/data/ascent_reference.json) |
| `v1_stop_baseline.json` | [tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/data/v1_stop_baseline.json](tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/data/v1_stop_baseline.json) |

## 验证

```bash
python scripts/tools/check_training_package_layout.py
bash scripts/train_ascent_stage5.sh --check-only
```

以上命令在仓库根目录、配置好依赖的 Python 环境中执行。目录检查覆盖真实模块导入、历史别名对象一致性、内部导入路径与 JSON 读取；CPU 预检覆盖 Stage5 初始化与示范数据。

本次还通过 Stage2、Stage4、Stage5 现有 CPU 回归，包含实际 PPO 更新与 checkpoint 保存/恢复；未启动 Isaac/PhysX 或正式训练。目录迁移本身不更改奖励、课程参数、策略权重键或 checkpoint 文件。
