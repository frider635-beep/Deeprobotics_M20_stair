# M20 楼梯与多地形运动训练

[English](README.md) · 更新日期：**2026-10-19**

当前教师训练工程记为 **v0 基线**，保留已有任务、profile 和 checkpoint 的兼容版本标记。常用程序见[工具清单](scripts/tools/README.md)。

本工程基于 [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training) 与 Isaac Lab，包含 M20 轮式预训练、特权楼梯教师训练、MuJoCo 部署、遥测、回归评测和实验性感知学生接口。

当前微调方向是以 `pre_train/teacher_baseline_idle_stop.pt` 为基准，保留用户已观察到的停车能力，学习运动阶段的横移与下楼前几级碰撞修正。相关代码已实现；现有记录**尚不能证明新训练的策略已解决这些问题**。

## 任务与网络接口

| 任务 ID | 用途 |
| --- | --- |
| `Flat-Deeprobotics-M20-v0` | 平地速度跟踪 |
| `Rough-Deeprobotics-M20-v0` | 粗糙地形速度跟踪 |
| `Rough-Deeprobotics-M20-PreTeacher-v0` | 轮式预训练 |
| `Rough-Deeprobotics-M20-StairTeacher-v0` | 带特权信息的楼梯与混合地形教师 |
| `Rough-Deeprobotics-M20-StairStudent-v0` | 实验性感知接口，尚无完整蒸馏流程 |

| 教师配置 | Actor 输入 | Critic 输入 | 策略结构 |
| --- | --- | --- | --- |
| 普通教师；`v1_stop` 至 `v1_stop_v4` | 244 | 285 | 共享 `244→512→256→128→16` MLP；部分配置另存训练辅助组 |
| `v1_stop_v5` | 244＋24 | 285＋24 | 冻结基准，新增 `268→128→64→16` 零指令分支 |
| `idle_stop_motion_v1` | 244＋3 | 285＋3 | 冻结停车基准，新增 `247→128→64→16` 运动残差分支 |

教师原输入为 57 维本体/命令/历史量加 187 点高度扫描，仍输出 12 个腿位置动作和 4 个轮速度动作。学生 actor/critic 为 618 维，另有 244 维教师查询组。维数相同不代表命令语义和控制器版本相同。

## 环境与模型

以下命令从仓库根目录执行，使用相互兼容的 Isaac Sim / Isaac Lab 环境。本机 Conda 环境名为 `isaaclab`，历史记录曾使用 `m20_wzh`。2026-10-19 本机元数据：Python 3.11.14、Isaac Sim 5.1.0.0、`isaaclab` 包 0.54.3、RSL-RL 5.0.1、PyTorch 2.7.0+cu128、MuJoCo 3.3.7。包版本不等同于 Isaac Lab 上游发行标签；这是复现快照。

```bash
conda activate isaaclab
git submodule update --init --recursive
python -m pip install -e source/rl_training
python scripts/tools/list_envs.py
```

模型子模块提供 Isaac Lab 使用的 `deep_robotics_model/M20/usd/M20.usd` 和 MuJoCo 使用的 `deep_robotics_model/M20/mjcf/M20.xml`。教师/预训练的 USD 路径从仓库解析，`RL_TRAINING_M20_USD_PATH` 可覆盖位置。训练入口优先加载本仓库源码。必要时用 `python scripts/tools/setup_conda_runtime.py` 检查 C++ ABI 并安装激活钩子，随后重新激活环境。

## 训练

轮式预训练：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Rough-Deeprobotics-M20-PreTeacher-v0 --max_iterations 20000 --device cuda:0 --headless
```

将普通预训练 actor 迁移到楼梯任务，替换示例运行目录：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Rough-Deeprobotics-M20-StairTeacher-v0 --init_actor_from logs/rsl_rl/deeprobotics_m20_pre_teacher/RUN/model_19999.pt --run_name stair_teacher --max_iterations 3000 --device cuda:0 --headless
```

当前停车基准微调：

```bash
conda run --no-capture-output -n isaaclab python -u scripts/reinforcement_learning/rsl_rl/train.py --task Rough-Deeprobotics-M20-StairTeacher-v0 --m20_profile idle_stop_motion_v1 --init_actor_from pre_train/teacher_baseline_idle_stop.pt --run_name idle_stop_motion_v1_stage1 --num_envs 4096 --max_iterations 1000 --seed 42 --device cuda:0 --headless
```

基准权重被 Git 忽略，需要另外准备。要求的 SHA256：`0e6d0c93560b5a3f7ce9219c7b6a0cea84bd4a2d3f5dba33c02d79f7d4eb5f34`。该配置冻结基准 MLP 和高斯分布参数，残差输出从零初始化，仅在原始命令任一分量绝对值大于 `1e-4` 时加入有限修正。相同观测下，零指令仍使用基准映射；运动修正可能改变停车开始时的状态。

配置强化上楼跟踪，新增下楼横向位置/速度约束，开启温和的临界距离和接近速度代价。这些是候选训练设计，不是效果已验证。前 20 轮预热 critic，每 100 轮保存；先检查 100/300/500 轮，再决定是否延长首轮 1000 轮预算。轮次指 PPO 更新，不是 episode 数。

| 参数 | 含义 |
| --- | --- |
| `--init_actor_from PATH` | 初始化策略；默认重建 critic、优化器和迭代计数 |
| `--init_critic_from PATH` | 同时初始化形状兼容的 critic，须配合 actor 初始化 |
| `--resume --load_run RUN --checkpoint FILE` | 恢复兼容的完整状态；`--max_iterations` 为追加轮数 |
| `--m20_profile NAME` | 创建环境前选择配置；不传时使用环境默认值 |
| `--preserve_actor_from PATH` | 适用共享策略配置的能力参照；`idle_stop_motion_v1` 不传 |
| `--descent_guide_iters N --descent_guide_decay_iters M` | 可选前轮先行引导及衰减，仅适用于 `idle_stop_motion_v1`；默认 `N=0`，不开启 |
| `--num_envs N --seed N --device cuda:0` | 环境数量、种子和设备 |

actor 初始化与 `--resume` 不能同用。配置按文件内容哈希核验基准；`v1_stop_v5` 要求原 v1/model5100，不接受新的停车 baseline。运行中的进程不会热加载奖励/网络修改。日志位于 `logs/rsl_rl/deeprobotics_m20_stair_teacher/<时间戳_运行名>/`；通过 `params/launch_manifest.json` 和检查点 manifest 核对源码/基准哈希、奖励和观测组。

## 部署与回放

先用与运动配置一致的 `legacy_v1` 控制器回放停车基准：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py --model deep_robotics_model/M20/mjcf/M20.xml --checkpoint pre_train/teacher_baseline_idle_stop.pt --terrain-xml deploy/deploy_mujoco/terrains/stairs_ascent.xml --legacy-heading-hold --viewer --telemetry --telemetry-every 5 --output .scratch/idle_stop_baseline/ascent.npz
```

随后替换为训练候选检查点。新配置通过 manifest 自动选择 `legacy_v1`，无 manifest 的旧模型需要显式参数才能保持本次对照一致。先确认基准在相同控制器下的表现，再把差异归因于训练。

窗口按键：`Z` 起立、`C` 策略、`W/S` 前后、`A/D` 横移、`Q/E` 转向、`H` 扫描、`X` 趴下、`R` 阻尼、`Esc` 退出。X11 直接读取按住状态，松开/失焦后移动命令归零；终端输入使用连发/超时机制。停车时检查遥测中的原始零命令。

下楼使用 `stairs_descent.xml`，平地明确使用 **`--terrain flat`**，不传地形默认上楼。`--viewer --autoplay` 执行自动指令，策略/物理周期为 0.02/0.005 秒。批量回放示例：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py --model deep_robotics_model/M20/mjcf/M20.xml --checkpoint pre_train/teacher_baseline_idle_stop.pt --terrain flat --legacy-heading-hold --command-profile deploy/deploy_mujoco/profiles/zero_hold.json --telemetry --output .scratch/idle_stop_baseline/flat.npz
```

地形参数见[地形说明](deploy/deploy_mujoco/terrains/README_CN.md)。遥测包含命令、控制器状态、机体/轮子运动、路线误差、步态和求解接触；接触子步数量不是独立碰撞事件数。

**普通无新增分支的检查点**可用 Isaac 回放：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task Rough-Deeprobotics-M20-StairTeacher-v0 --checkpoint logs/rsl_rl/deeprobotics_m20_stair_teacher/RUN/model_500.pt --num_envs 1 --keyboard
```

当前 `play.py` 不会根据 manifest 重建自定义分支。`v1_stop_v5` 和 `idle_stop_motion_v1` 使用更新后的 MuJoCo loader；两者尚不支持旧式单输入 JIT/ONNX 导出，只导出冻结主干会遗漏已学习修正。`deploy/deploy_real/` 为预留目录，尚无完整真机部署。

## 评估与监控

显式传入新基准，否则评估器默认使用历史 model5100：

```bash
python scripts/tools/evaluate_m20_v1_stop.py --baseline pre_train/teacher_baseline_idle_stop.pt --candidate logs/rsl_rl/deeprobotics_m20_stair_teacher/RUN/model_300.pt --output .scratch/idle_stop_motion_v1/model300_evaluation.json --require-pass
```

两个模型使用同一 legacy 控制器，默认九类工况、0°/−3°/+3° 初始偏航，检查停车、进度、横移和接触回归。现有容差用于判断“不退化”，通过不等于横移或前三阶碰撞显著减少，还需单独比较实测量，重复首阶/中段/反向停车及重启试验。带 `--steps` 的截断运行不算验收；`--require-pass` 失败时返回非零退出码并保留报告。

```bash
tensorboard --logdir logs/rsl_rl/deeprobotics_m20_stair_teacher
```

| 指标 | 解释 |
| --- | --- |
| `Loss/motion_residual_rms`、`Loss/motion_command_fraction`、`Loss/raw_zero_command_fraction` | 运动残差大小与命令曝光 |
| `Curriculum/ascent_direction/*`、`Curriculum/descent_direction/*` | 方向/路线统计；航向小不等于没有横滑 |
| `Curriculum/gait_quality/*`、`Curriculum/step_completion/*` | 姿态、间距、逐阶事件；比例须结合样本量 |
| `Episode_Reward/*` | 带权奖励贡献，不是独立质量评分 |
| `Perf/collection_time`、`Perf/learning_time`、`Perf/logging_time` | 采样、PPO、日志耗时；墙钟时间还含其他开销 |
| `Curriculum/zero_stop/*` | 停车配置使用；`idle_stop_motion_v1` 关闭此回调，停车须外部验证 |

固定地形、命令、控制器、初态与软件版本后比较。教师损失/动作噪声下降、总奖励升高，或因拒绝前进而横移变小，都不能单独证明成功。

需要详细统计时在训练命令后追加 `env.teacher_detailed_diagnostics=true`；该项只调整诊断，不改变奖励或任务难度。

## 代码与开发记录

- [训练源码说明与整理记录](source/rl_training/rl_training/README.md)：模块职责、任务/profile 依赖、兼容文件及删除依据。
- [stair_teacher.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py)：几何、逐阶账本、奖励、命令与诊断；运行时软包络参数维护在 `TUNING`。
- [stair_teacher_env_cfg.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py)：观测、实体和默认奖励；[v1_stop_profile.py](source/rl_training/rl_training/v1_stop_profile.py) 应用配置覆盖。
- [stop_branch.py](source/rl_training/rl_training/stop_branch.py)、[motion_residual_actor.py](source/rl_training/rl_training/motion_residual_actor.py)、[stair_guarded_ppo.py](source/rl_training/rl_training/stair_guarded_ppo.py)：自定义网络与 PPO 扩展。
- [deploy_mujoco.py](deploy/deploy_mujoco/deploy_mujoco.py) 与[评估工具](scripts/tools)：部署及回归。
- [技术改进路线：截至 2026-10-19](docs/TECHNICAL_IMPROVEMENT_ROADMAP_CN.md)：时间线、改动、证据、退化与待完成事项。

`docs/` 仅发布 `TECHNICAL_IMPROVEMENT_ROADMAP_CN.md`。其他文档、`.scratch/`、`logs/`、`outputs/` 以及权重、缓存继续仅本地保留。路线文档指向历史报告和本地备份的链接在新克隆中不可用；基准模型和结果需另行准备。当前源码和运行 manifest 优先于历史命令与奖励表。
