# M20 楼梯与多地形运动训练

[English](README.md) · 更新日期：**2026-10-20**

当前教师训练工程记为 **v0 基线**，保留已有任务、profile 和 checkpoint 的兼容版本标记。常用程序见[工具清单](scripts/tools/README.md)。

本工程基于 [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training) 与 Isaac Lab，包含 M20 轮式预训练、特权楼梯教师训练、MuJoCo 部署、遥测、回归评测和实验性感知学生接口。

当前微调方向是以 `pre_train/teacher_baseline_idle_stop.pt` 为基准，保留用户已观察到的停车能力，学习运动阶段的横移与下楼前几级碰撞修正。相关代码已实现；现有记录**尚不能证明新训练的策略已解决这些问题**。

2026-10-20另注册独立上楼步态任务 **`Rough-Deeprobotics-M20-StairTeacher-v1`**，自动使用`ascent_entry_v1`，完整冻结`pre_train/teacher_baseline_.pt`的基础网络与运动残差，仅训练新增上楼修正分支。v1包含入阶相位/落点引导、200 Hz接触滑移约束和60/20/20入口采样；v0任务及下文历史流程保留。实现、证据与验收边界见[v1实施说明](docs/test_docs/07_上楼步态校正/v1_20261020/实施说明.md)。候选训练完成不等于步态验收通过。

```bash
OMP_NUM_THREADS=4 python -u scripts/reinforcement_learning/rsl_rl/train.py --task Rough-Deeprobotics-M20-StairTeacher-v1 --init_actor_from pre_train/teacher_baseline_.pt --init_critic_from pre_train/teacher_baseline_.pt --run_name ascent_entry_v1 --num_envs 512 --max_iterations 1000 --seed 42 --device cuda:0 --headless
```

2000轮复核后，第二阶段通过同一v1任务的显式`--m20_profile ascent_entry_v1_stage2`启用，继承完整v1 Actor及可选Critic权重、重建优化器和计数，强化前四阶逐轮稳定支撑、前后轮独立滑移与困难入口采样。Actor 244+3、Critic 285+3及修正限幅保持原样；原`ascent_entry_v1`默认行为保留。新命令及验证证据见[第二阶段调整说明](docs/test_docs/07_上楼步态校正/v1_20261020/stage2_20261020/调整说明.md)。

第二阶段800轮复核后，新增显式`--m20_profile ascent_entry_v1_stage3`：冻结完整第二阶段Actor（包括其`entry_mlp`），增加零输出初始化的入口落阶分支`247→128→64→16`，提前引导前两阶落轮并扩大入口修正范围。观测仍为Actor 244+3、Critic 285+3，**Actor结构新增一个分支**。推荐从第二阶段`model_600.pt`重新初始化，先训练400轮；日志继续写入`deeprobotics_m20_stair_teacher_v1_stage2/`。训练命令、CPU验证及几何分析局限见[第三阶段调整说明](docs/test_docs/07_上楼步态校正/v1_20261020/stage3_20261020/调整说明.md)。尚未验证新训练的步态效果。

验证约定：按用户要求，不主动运行GPU短测；使用CPU、静态检查和已有数据分析，正式GPU训练由用户在终端自行启动。

第三阶段400轮复核后新增`--m20_profile ascent_entry_v1_stage4`：冻结完整第三阶段Actor并增加零输出顺序修正分支，奖励同阶两前轮先后支撑、四轮完整逐阶推进、支撑裕度和到顶后放平前行，不固定前后轮阶差。新增入口速度课程，保留已有减滑约束；CPU反例、PPO集成、观察器离线重放与零分支MuJoCo保持检查通过，尚未运行第四阶段GPU训练或验证训练效果。初始化模型、手动命令及验证边界见[第四阶段调整说明](docs/test_docs/07_上楼步态校正/v1_20261020/stage4_20261020/调整说明.md)。

检查点目录约定：后续v1改进训练、进度复核和候选选择默认使用`logs/rsl_rl/deeprobotics_m20_stair_teacher_v1_stage2/`下对应运行的检查点；`deeprobotics_m20_stair_teacher_v1/`保留为历史对照。选择模型时核对运行配置与checkpoint元数据。

## 任务与网络接口

| 任务 ID | 用途 |
| --- | --- |
| `Flat-Deeprobotics-M20-v0` | 平地速度跟踪 |
| `Rough-Deeprobotics-M20-v0` | 粗糙地形速度跟踪 |
| `Rough-Deeprobotics-M20-PreTeacher-v0` | 轮式预训练 |
| `Rough-Deeprobotics-M20-StairTeacher-v0` | 带特权信息的楼梯与混合地形教师 |
| `Rough-Deeprobotics-M20-StairTeacher-v1` | 独立上楼入阶步态修正，完整继承带运动残差的baseline |
| `Rough-Deeprobotics-M20-StairStudent-v0` | 实验性感知接口，尚无完整蒸馏流程 |

| 教师配置 | Actor 输入 | Critic 输入 | 策略结构 |
| --- | --- | --- | --- |
| 普通教师；`v1_stop` 至 `v1_stop_v4` | 244 | 285 | 共享 `244→512→256→128→16` MLP；部分配置另存训练辅助组 |
| `v1_stop_v5` | 244＋24 | 285＋24 | 冻结基准，新增 `268→128→64→16` 零指令分支 |
| `idle_stop_motion_v1` | 244＋3 | 285＋3 | 冻结停车基准，新增 `247→128→64→16` 运动残差分支 |
| v1任务；`ascent_entry_v1` | 244＋3 | 285＋3 | 冻结完整运动策略，新增 `247→128→64→16` 上楼修正分支 |
| v1任务；`ascent_entry_v1_stage2` | 244＋3 | 285＋3 | 同原v1结构，更新其上楼修正分支 |
| v1任务；`ascent_entry_v1_stage3` | 244＋3 | 285＋3 | 冻结完整第二阶段Actor，另增 `247→128→64→16` 入口落阶分支 |

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

## v1 上楼第五阶段（2026-10-21）

`ascent_entry_v1_stage5` 先建立双前轮从第一阶开始的同阶先后支撑，再将任务扩展到整段楼梯和不同阶高、踏深。采用真实接触失败终止、提前准备、接触后结算的落地信用，以及经过 CPU MuJoCo 验证的初期示例；课程从 10/30 cm 扩展到 10–25 cm 阶高、25–35 cm 踏深，另保留固定 15/30 cm 参考列。

从第三阶段 `2026-10-20_17-01-23_ascent_entry_v1_stage3/model_1100.pt` 新建实验。Actor 仍为 244+3 维，Critic 增加训练专用状态后为300维；不能跨 profile 直接 resume 第四阶段。

```bash
conda activate isaaclab
bash scripts/train_ascent_stage5.sh
```

脚本先做 CPU 预检查；`bash scripts/train_ascent_stage5.sh --check-only` 仅检查、不启动训练。默认4096环境、2000轮；用 `M20_STAGE5_ENVS`、`M20_STAGE5_ITERS` 调整。示例文件在本地 `docs/test_docs/07_上楼步态校正/v1_20261021/stage5_20261021/verified_contact_demonstrations`，未随 Git 发布，新克隆需另行准备。

本次完成CPU动力学示例、真实PPO更新、断点恢复与isaaclab Python环境预检查；没有做GPU/Isaac端到端启动测试，没有开始第五阶段正式训练。实现和局部示例通过不代表完整上楼步态已学会。详细证据见本地[改进与验证](docs/test_docs/07_上楼步态校正/v1_20261021/stage5_20261021/改进与验证.md)和[初始化对比](docs/test_docs/07_上楼步态校正/v1_20261021/stage5_20261021/初始化检查点对比.md)。

## v1 第八阶段：固定输入的上下楼路径修正（2026-10-22）

当前使用 `--m20_profile ascent_entry_v1_stage8_fixed`。保留Stage7 Actor的247维输入、统一MLP的247→256→128→16结构和Critic的300维输入；所有网络参数键名、形状及总数保持不变。启动链路已用原始model_11400.pt验证；按用户后续要求，正式训练直接加载同一次Stage7训练的最新model_12100.pt，使用新优化器，不重复本轮验证。旧的253/306维Stage8方案已撤回，不再作为训练入口；旧文件仍可通过MuJoCo兼容加载器回放。

依次改进路径反馈、下楼门控、奖励及多尺寸课程。横向位置/速度反馈经现有有效vy通道进入策略，原始指令独立用于门控，保留legacy航向控制。下楼复用统一MLP的16个输出，以固定限幅和非训练参数的时间调度开放权限：前30轮为零，之后800轮线性增加。保留上楼输出约束，20/25cm窄踏面适当降低旧动作保留权重。上下楼路径偏差、横向速度和接触横滑均有代价，偏离路径后不能计为课程成功。上楼连续左右交替、逐轮不漏阶、同阶先后和到顶放平要求继续保留，不固定前后轮相隔级数。

13列地形包含8列上楼、1列平地、4列下楼，**上下楼均覆盖20/25/30/35cm踏深**，阶高课程为10/12/15/18/20/25cm。新增20cm上楼从10cm阶高、4阶任务开始，课程降级不会把20cm踏深改宽；25cm上楼列最低阶高15cm。下楼各踏深按0.5/0.7m/s两档速度分别累计成功，均达到最近8次至少6次成功才升级，连续3次失败降级。

在仓库根目录、现有isaaclab环境执行：

```bash
OMP_NUM_THREADS=4 python -u scripts/reinforcement_learning/rsl_rl/train.py --task Rough-Deeprobotics-M20-StairTeacher-v1 --m20_profile ascent_entry_v1_stage8_fixed --init_actor_from logs/rsl_rl/deeprobotics_m20_stair_teacher_v1_stage2/2026-10-21_18-29-16_ascent_entry_v1_stage7/model_12100.pt --init_critic_from logs/rsl_rl/deeprobotics_m20_stair_teacher_v1_stage2/2026-10-21_18-29-16_ascent_entry_v1_stage7/model_12100.pt --run_name ascent_entry_v1_stage8_fixed_from12100_reward_v2 --num_envs 4096 --max_iterations 12000 --seed 42 --device cuda:0 --headless
```

建议先训练500–1000轮复测默认楼梯及20/25cm踏深，同时查看上下楼横移、左右偏航、落阶覆盖、同阶先后、左右交替和平台过渡。固定输入版本的实现及实测证据见本地 `docs/test_docs/07_上楼步态校正/v1_20261022/stage8_fixed/改进与验证.md`；初始化与短测不代表正式续训后的完整步态验收。

当前奖励修订为 `m20_stair_path_rewards_20261022_v2`：路径反馈激活时XY跟踪采用有效指令，离线时允许受控回线、近线时抑制横向速度；路径位置权重−1.5，横向速度−0.25，接触滑移−0.4。关闭旧Stage2前后轮滑移项及旧下楼横移项，入口强调合入保留的连续上楼滑移项；下楼接触滑移增加纵向分量，上楼纵向仍由连续前后轮滑移项负责。航向、前进/停滞奖励及步态约束保持原配置。新增分方向纵向滑移与回线速度误差日志，网络及20/25/30/35cm课程保持原配置。修订后的实现与验证见本地 `docs/test_docs/07_上楼步态校正/v1_20261022/stage8_reward_v2/改进与验证.md`。

奖励修订会记录在manifest中。旧奖励版本完整`--resume`会被拒绝，需使用Actor/Critic初始化和新优化器；正式启动仍按用户要求直接加载原Stage7的model12100，不重复验证该检查点。

## 代码与开发记录

- [训练源码说明与整理记录](source/rl_training/rl_training/README.md)：模块职责、任务/profile 依赖、兼容文件及删除依据。
- [stair_teacher.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py)：几何、逐阶账本、奖励、命令与诊断；运行时软包络参数维护在 `TUNING`。
- [stair_teacher_env_cfg.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py)：观测、实体和默认奖励；[v1_stop_profile.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/v1_stop_profile.py) 应用配置覆盖。
- [stop_branch.py](source/rl_training/rl_training/models/stop_branch.py)、[motion_residual_actor.py](source/rl_training/rl_training/models/motion_residual_actor.py)、[stair_guarded_ppo.py](source/rl_training/rl_training/algorithms/stair_guarded_ppo.py)：自定义网络与 PPO 扩展。
- [deploy_mujoco.py](deploy/deploy_mujoco/deploy_mujoco.py) 与[评估工具](scripts/tools)：部署及回归。
- [技术改进路线：截至 2026-10-19](docs/TECHNICAL_IMPROVEMENT_ROADMAP_CN.md)：时间线、改动、证据、退化与待完成事项。

`docs/` 仅发布 `TECHNICAL_IMPROVEMENT_ROADMAP_CN.md`。其他文档、`.scratch/`、`logs/`、`outputs/` 以及权重、缓存继续仅本地保留。路线文档指向历史报告和本地备份的链接在新克隆中不可用；基准模型和结果需另行准备。当前源码和运行 manifest 优先于历史命令与奖励表。
