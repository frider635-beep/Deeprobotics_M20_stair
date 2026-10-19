# v0 教师训练源码说明与整理记录

整理日期：2026-10-19。本文覆盖 `source/rl_training/rl_training`。当前工程称为 **v0 教师训练基线**；已有任务 ID、profile 名称、Python 导入路径和 checkpoint 内部版本继续沿用。

本轮检查了全部 45 个 Python 文件、1 个基准 JSON、任务注册、训练入口、自定义模型类路径和 MuJoCo 文件加载。删除 1 个确定无业务用途的 UI 示例，保留 44 个 Python 文件和基准 JSON；仅修改包入口以移除示例导入。清理了 11 个 `__pycache__` 目录中的 45 个生成缓存文件。没有调整奖励、网络、PPO、观测、动作、地形或训练超参数。

## 1. 整理结果与保留原则

| 处理 | 文件或内容 | 依据 |
| --- | --- | --- |
| 删除 | `ui_extension_example.py` | Isaac 扩展模板，仅包含 `x**x` 演示函数和 `My Window` 的 Add/Reset 计数按钮；唯一业务目录引用来自包入口，未参与任务注册、策略、奖励或部署 |
| 修改 | [__init__.py](__init__.py) | 删除 UI 示例的星号导入，更新模块说明；继续通过 `.tasks` 注册任务。包导入不再主动导出示例窗口类和演示函数 |
| 清理 | `__pycache__/` 与 `.pyc` | Python 运行时生成缓存，可自动重建；不属于训练源码 |
| 原样保留 | 其余 43 个 Python 文件、[v1_stop_baseline.json](v1_stop_baseline.json) | 当前任务使用、共享依赖、旧配置/模型兼容，或尚不能确定无用 |

没有按“当前训练命令未启用”删除历史分支，也没有把运行模块移动到新目录。RSL-RL 配置中存在 `rl_training.xxx:ClassName` 动态类路径，MuJoCo 也按源码路径加载模型；保存的 YAML/Pickle 还可能引用兼容模块，移动文件可能破坏加载。

删除前的源码包及逐文件 SHA256 清单保存在仓库根目录的 `.scratch/04_change_backups/source_package_cleanup_v0_20261019/`。`package_before.tar.xz` 是源码归档（已移除可重建缓存），`manifest.json` 记录原始文件和处理依据，`verification.json` 记录本轮检查结果。该备份遵循现有规则，仅本地保留。

## 2. 目录与执行关系

```text
rl_training/
├── __init__.py                  # 包入口：触发任务注册
├── assets/                      # M20 模型、执行器等资产配置
├── actuator/                    # 自定义执行器，保留备用接口
├── tasks/
│   └── manager_based/locomotion/velocity/
│       ├── velocity_env_cfg.py  # 通用场景、命令、动作、观测、奖励、课程
│       ├── mdp/                # 具体奖励、状态、命令、课程、事件、观测
│       └── config/wheeled/deeprobotics_m20/
│           ├── __init__.py      # 注册 5 个 M20 任务
│           ├── *_env_cfg.py     # 各任务环境配置
│           └── agents/         # 对应 PPO/runner 配置
├── v1_stop_profile.py           # 显式配置覆盖及基准身份校验
├── v1_stop_baseline.json        # v1 参数、奖励权重与基准哈希
├── motion_residual_actor.py     # 当前运动修正分支
├── stop_aware_actor.py          # 历史停车探索分支
├── stop_branch.py               # 历史停车网络及共享停车参考
├── stair_guarded_ppo.py         # 教师 PPO 与保留/正则约束
├── stop_coverage.py             # 停车暴露和重启诊断
├── teacher_diagnostics.py       # 诊断选择和批量标量导出
├── metric_aggregation.py        # 条件指标跨 episode 汇总
├── fast_logging.py              # 批量 TensorBoard 写入
└── m20_preflight.py             # 旧版离线契约检查，保留待适配
```

从仓库根目录启动的 [train.py](../../../scripts/reinforcement_learning/rsl_rl/train.py) 将本地 `source/rl_training` 加入导入路径，加载任务，取得环境和 PPO 配置，再按 `--m20_profile` 应用覆盖。创建环境后，入口选择 actor 类和观测组，由 `StairGuardedPPO` 更新策略，诊断经过聚合后写入 TensorBoard。

[deploy_mujoco.py](../../../deploy/deploy_mujoco/deploy_mujoco.py) 根据 checkpoint/manifest 选择普通 actor、停车分支或运动残差分支，按文件路径加载共享源码。部署不是训练目录的独立副本，因此 `stop_branch.py`、`motion_residual_actor.py` 也属于部署依赖。

## 3. 包根目录模块

| 模块 | 职责 | 使用状态与保留原因 |
| --- | --- | --- |
| [v1_stop_profile.py](v1_stop_profile.py) | 恢复记录的 v1 参数与奖励，应用各 profile 差异，校验初始 actor SHA256 | 训练入口直接调用；当前 `idle_stop_motion_v1` 仍依赖它 |
| [v1_stop_baseline.json](v1_stop_baseline.json) | 保存 v1 的参数、奖励权重、策略噪声和来源哈希 | 所有记录在该 profile 模块中的方案共用；文件名旧但仍生效 |
| [motion_residual_actor.py](motion_residual_actor.py) | 冻结基准 actor、归一化及分布参数，仅学习有界运动残差 | 当前 `idle_stop_motion_v1` actor；部署也直接加载 |
| [stair_guarded_ppo.py](stair_guarded_ppo.py) | 参考策略约束、critic 预热、分支正则、有限值保护与辅助状态保存/恢复 | 教师 PPO 配置通过字符串类路径调用；不是普通可删检查脚本 |
| [metric_aggregation.py](metric_aggregation.py) | 按真实样本计数合并条件均值、RMS、最小值 | 训练日志入口直接调用，避免无样本回合稀释指标 |
| [teacher_diagnostics.py](teacher_diagnostics.py) | 选择核心/详细诊断、批量转换标量 | `mdp/stair_teacher.py` 直接调用，减少逐字段设备同步 |
| [fast_logging.py](fast_logging.py) | 汇总 episode 数据、写入 TensorBoard，缩短逐字段日志开销 | 训练入口对 `v1_stop_v5`、`idle_stop_motion_v1` 启用快路径；其他情况使用原 logger |
| [stop_branch.py](stop_branch.py) | 24 维停车控制分支、停车参考与支撑判断工具 | v5 模型训练/部署需要；教师源码也直接导入共享工具，不能随旧 profile 一并删除 |
| [stop_aware_actor.py](stop_aware_actor.py) | 根据原始停车标签调整探索标准差并记录动作诊断 | `v1_stop_v2/v3/v3_target_control/v4` 动态选择，保留历史兼容 |
| [stop_coverage.py](stop_coverage.py) | 独立于指令的地形身份、停车暴露、持续停车与重启统计 | 教师源码直接导入，旧停车分支按条件调用；不是奖励修正器 |
| [m20_preflight.py](m20_preflight.py) | 离线提取旧控制常量、校验普通 actor 布局与历史版本 | 保留的 `check_m20_engineering.py` 仍调用；当前 `train.py` 没有把它作为启动门禁 |

`m20_preflight.py` 的常量提取假设沿用旧部署布局，不能把它的结果直接当作当前所有 profile 的完整契约验证；尤其不能用其普通 actor 校验替代自定义分支检查。本轮未为适配旧检查修改生产控制或训练逻辑。

## 4. 任务配置：5 个注册入口均保留

注册位置：[deeprobotics_m20/__init__.py](tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py)。下表配置文件均位于同一目录，PPO 文件位于其 `agents/`。

| Gym 任务 ID | 环境配置 | PPO/runner 配置 | 保留原因 |
| --- | --- | --- | --- |
| `Flat-Deeprobotics-M20-v0` | `flat_env_cfg.py` | `rsl_rl_ppo_cfg.py` | 平地任务仍注册，与 Rough 共用基础实现 |
| `Rough-Deeprobotics-M20-v0` | `rough_env_cfg.py` | `rsl_rl_ppo_cfg.py` | M20 动作/关节/奖励的基础配置，教师和预训练也继承 |
| `Rough-Deeprobotics-M20-PreTeacher-v0` | `pre_teacher_env_cfg.py` | `pre_teacher_ppo_cfg.py` | 与教师输入布局匹配的轮式预训练，仍是可选注册任务 |
| `Rough-Deeprobotics-M20-StairTeacher-v0` | `stair_teacher_env_cfg.py` | `stair_teacher_ppo_cfg.py` | 当前主要教师任务 |
| `Rough-Deeprobotics-M20-StairStudent-v0` | `stair_student_env_cfg.py` | `stair_student_ppo_cfg.py` | 实验性感知学生入口仍注册；不等于已完成学生训练或部署验收 |

通用 [velocity_env_cfg.py](tasks/manager_based/locomotion/velocity/velocity_env_cfg.py) 提供场景、观测、命令、事件、奖励、课程和终止条件。教师配置是在 Rough 和这些通用配置上扩展，而不是可以脱离它们运行的单文件任务。

`__init__.py` 一共保留 12 个：包入口、任务递归发现、MDP 导出、任务注册、资产元数据、执行器导出，以及各层包边界。部分文件只有许可证或少量导入，仍不作为“空文件”删除。

## 5. MDP 模块

以下文件位于 [tasks/manager_based/locomotion/velocity/mdp](tasks/manager_based/locomotion/velocity/mdp)。

| 模块 | 内容与调用关系 | 处理 |
| --- | --- | --- |
| `stair_teacher.py` | 教师核心：`TUNING`、命令整形/航向保持、扫描几何、上下楼逐阶状态、停车状态、奖励和诊断。环境配置及 profile 引用大量函数 | 原样保留，不在整理中拆分或修改函数 |
| `rewards.py` | 通用运动跟踪、姿态、动作平滑、关节和接触奖励；由 `mdp/__init__.py` 导出，基础/教师配置复用 | 原样保留 |
| `commands.py` | 阈值速度命令与离散命令控制器；由 MDP 导出并被配置引用 | 原样保留，不删除未被当前命令选中的类 |
| `curriculums.py` | 地形、步态和命令课程 | 原样保留，基础配置仍使用 |
| `events.py` | 惯性、质心等随机化和姿态终止辅助 | 原样保留，基础配置仍使用 |
| `observations.py` | 排除轮关节的位置观测、步态相位等 | 原样保留，基础配置仍使用 |
| `descent_front_pair.py` | 前三级下楼双前轮进度差、着地延时、横滚惩罚及诊断；共享逐阶账本 | 原样保留；教师直接导入，但某个 profile 下是否有奖励梯度由最终奖励配置决定 |
| `stair_ascent.py` | 从 `stair_teacher.py` 重导出上楼状态函数 | 保留旧 YAML/Pickle 调用路径和工具依赖；不是一套重复奖励实现 |
| `gait_refinement.py` | 从 `stair_teacher.py` 重导出转向、折腿、步态质量函数 | 同样是兼容层，保留旧调用路径和工具依赖 |
| `stair_student.py` | 感知学生的高度/有效性/置信度通道、状态复位和蒸馏辅助函数 | 学生配置仍直接引用，保留实验接口 |

MDP 的 `__init__.py` 还导入 Isaac Lab 原生 MDP；配置中的 `mdp.xxx` 既可能来自本地，也可能来自上游。仅在单个源码文件中搜索不到定义，不能据此删除配置调用。

## 6. 资产与执行器

| 模块 | 职责与判断 |
| --- | --- |
| [assets/__init__.py](assets/__init__.py) | 定位扩展目录、模型目录并读取 `extension.toml` 元数据；由资产配置依赖 |
| [assets/deeprobotics.py](assets/deeprobotics.py) | 定义 `DEEPROBOTICS_M20_CFG`、机器人 USD 与执行器参数；Rough 配置直接使用 |
| [actuator/PDActuator.py](actuator/PDActuator.py) | 自定义 `RandomPDActuator` 及配置，含增益/强度/偏置随机化和转矩速度限制 |
| [actuator/__init__.py](actuator/__init__.py) | 导出上述执行器类 |

当前 `assets/deeprobotics.py` 使用 Isaac Lab 的 `DelayedPDActuatorCfg`，没有切换到 `RandomPDActuatorCfg`。自定义执行器暂未接入这条默认资产路径，但保留为用途尚不能排除的接口，未为了让它“用起来”修改电机模型。

## 7. 当前 profile 与历史分支的区别

| `--m20_profile` | actor 与附加输入 | 状态 |
| --- | --- | --- |
| `idle_stop_motion_v1` | 基准 policy 244 维 + 原始命令 `motion_control` 3 维；运动残差分支输入 247 维、输出 16 维 | 当前以 `pre_train/teacher_baseline_idle_stop.pt` 为基准的运动修正方案 |
| `v1_stop` | 普通 244→16 actor | 历史 v1 停车修正基线 |
| `v1_stop_v2/v3/v3_target_control/v4` | `StopAwareMLPModel`；策略仍读 244 维，停车标签供训练使用，v4 另有教师/支撑信息 | 历史探索/教师方案，兼容入口保留 |
| `v1_stop_v5` | `StopBranchMLPModel`；policy 244 + `stop_control` 24 维 | 历史停车分支；共享代码仍有其他依赖 |

不指定上述 profile 时使用教师默认配置，不自动等价于 `idle_stop_motion_v1`。**文件被导入、奖励函数存在、奖励被启用是三件不同的事**：最终行为要结合环境继承、`apply_profile()`、训练 CLI 覆盖和本次运行保存的配置判断。

当前运动残差在原始命令绝对值最大值超过 `1e-4` 时生效；零命令时关闭残差，并冻结基准 actor 参数。这保证相同观测下零命令 actor 输出的保留，不保证运动后到达的状态、实际滑动或碰撞结果自动不变，仍需固定工况回归。

`idle_stop_motion_v1` 关闭 `Curriculum/zero_stop` 回调，也不启用旧停车 v2–v5 开关；看到这组指标缺失不能判断停车改善或恶化。应使用当前 [评估工具说明](../../../scripts/tools/README.md) 中的部署/停车回归流程。普通单输入导出器或旧 `play.py` 不能代替新增分支的加载验证。

## 8. 本轮核验与边界

- 对保留的全部 Python 文件进行 AST 解析，并与删除前 SHA256 对照；除包入口外，43 个 Python 文件和基准 JSON 字节一致。
- 检查保留源码中不再存在对 UI 示例的导入、字符串加载或示例类调用。
- 在 `isaaclab` Python 环境使用真实 Gymnasium 注册表检查 5 个任务的注册内容，并核对环境/PPO 入口文件及类定义存在。测试隔离任务递归发现，不启动 Isaac Sim 或创建物理环境。
- 在同一环境导入包和 4 个自定义 actor/PPO 类；仅隔离需要 Isaac Sim 的任务模块，模型类使用真实 PyTorch/RSL-RL。检查 profile 和基准 JSON 可读取。
- 校验说明文档链接与 `git diff --check`。

上述核验针对整理引起的依赖和入口风险，没有重新训练、运行完整 Isaac Sim 任务或宣称策略效果改善。旧检查器的历史失败没有被当作删生产代码的依据。

后续奖励或网络改动应单独记录。技术路线仍保留在 [docs/TECHNICAL_IMPROVEMENT_ROADMAP_CN.md](../../../docs/TECHNICAL_IMPROVEMENT_ROADMAP_CN.md)，本说明放在源码包内，随工程同步，便于克隆后直接查看模块用途和保留理由。
