# rl_training 工程结构

整理日期：2026-10-21。包根目录只保留 `__init__.py` 和本说明；代码按任务、网络、算法和工具分层。

```text
rl_training/
├── __init__.py                 # 兼容旧模块路径，然后注册 Gym 任务
├── algorithms/
│   ├── stair_guarded_ppo.py    # PPO 扩展
│   └── initialization/        # checkpoint 校验、初始化、示范训练挂接
├── control/                   # 训练和部署共用的路径反馈（纯 Torch）
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
    │   ├── path/              # 上下楼路径观测、代价与下楼课程
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

## Stage6 连续左右交替

`ascent_entry_v1_stage6` 的奖励与课程位于 `mdp/ascent/ascent_alternating*.py`，配置位于 `config/wheeled/deeprobotics_m20/profiles/ascent_alternating_profile.py`，初始化校验位于 `algorithms/initialization/ascent_alternating_init.py`。正奖励统一受前、后腿各自连续左右交替前提约束；原Stage5 profile保留。

### Stage7 统一上楼修正

`models/ascent_unified_actor.py`保留冻结的原始`mlp`与`residual_mlp`，用一个247→256→128→16的`unified_mlp`替代三个上楼修正分支。`ascent_unified_profile.py`注册独立profile `ascent_entry_v1_stage7`；`mdp/ascent/ascent_unified.py`实现2/3/4/6/18阶课程与双轴同阶记账，`ascent_unified_mdp.py`提供成对落脚引导和失败诊断。旧模型需经`merge_ascent_branches.py`离线迁移，不能直接按Stage7恢复。


### Stage8 上下楼路径修正

`ascent_entry_v1_stage8_fixed` 保持Stage7 Actor247/Critic300及全部网络state键名、形状、参数总数。`models/stair_path_actor.py`复用现有统一MLP的16个输出，以前30轮关闭、后800轮线性增加的非训练门控开放下楼；保留参考策略的下楼权限恒为零。下楼限幅及探索尺度是非持久控制常量，不添加可训练参数。只训练原统一MLP的98,448个参数和Critic。

`control/stair_path.py`共用训练与部署的路径反馈，位置与法向速度通过已有有效vy进入policy244，原始command3继续单独作为门控输入。内部6维诊断状态不进入Actor或Critic。`mdp/path/stair_path_mdp.py`负责双方向路径代价、横移失败、到平台稳定前行与下楼课程；所有正向上楼奖励仍经过连续交替门控，逐轮不漏阶和同阶配对记账继续保留。通过路径边界失败的环境不能作为上楼课程成功。

`profiles/stair_path_profile.py`生成13列：8上楼、1平地、4下楼，上下楼均含20/25/30/35cm踏深。每列踏深在所有阶高保持固定，20cm上楼首次从10cm阶高、4阶前缀开始。`algorithms/initialization/stair_path_init.py`严格加载原model11400，不扩展权重，迁移10→13列课程进度并重置结果窗口。`prepare_stair_path.py`只提供可选的相同权重加控制元数据的评估文件，正式训练直接使用原model11400。

旧扩维版本仅保留于`models/legacy/stair_path_expanded_actor.py`供MuJoCo历史回放，不作为当前训练方案。CPU真实PPO、GPU短测与MuJoCo复测证据见本地 `docs/test_docs/07_上楼步态校正/v1_20261022/stage8_fixed/改进与验证.md`。

后续改进优先处理奖励、控制反馈、门控、课程和数据分布，保护已形成的步态；固定输入方案有明确对照瓶颈证据后才考虑改变网络结构或输入输出维度。

### 固定输入版本的奖励修订v2

`mdp/path/stair_path_objectives.py`集中实现可独立验证的目标：有效XY命令跟踪、平滑的受控回线速度目标、按承重轮分别计算的横向/纵向接触滑移，以及入口与第3→4阶强调的max合并（上限1.5）。只在当前路径profile启用；原Stage7及更早profile保持原行为。

路径位置/速度/滑移权重为−1.5/−0.25/−0.4，旧Stage2前后轮滑移和旧下楼横移权重为0；原连续上楼前后轮滑移−0.65/−0.55保留。下楼新增0.5倍纵向滑移分量（0.20m/s归一化），横向滑移按0.15m/s归一化；上楼已有完整切向滑移惩罚，通用项不再重复纵向。新日志包括ascent/descent的`longitudinal_contact_slip_m_s`及`return_velocity_error_m_s`。

训练版本附加`_reward_v2`且记录奖励语义与代码SHA；网络版本和state结构不变。原固定输入v1仍能初始化权重，完整恢复要求相同奖励版本。工具`check_stair_path_objectives.py`验证正常滚动、纵向空转、单轮打滑、腾空轮排除、镜像/旋转一致和回线方向，真实训练启动另以已验证的model11400做隔离短测；不重复测试用户指定的model12100。
