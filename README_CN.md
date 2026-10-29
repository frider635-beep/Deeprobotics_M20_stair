# M20 楼梯与多地形运动训练

[English](README.md) · 更新日期：**2026-10-20**

2026-10-27 当前 Stage9 为 **v10 上楼后轴入楼过渡**：仅后轴前两阶允许调整先手，第三阶首次稳定落脚后固定先手，之后顺序违规由奖励惩罚和严格步态指标约束，不再独立卡住地形课程；前轴原规则、物理无漏阶/同阶依次支撑保持。保留两速度物理完成和前轴课程证据、已学进度；下楼奖励/课程/控制与网络结构保持。采样手动stage1，学习率 `1.5e-5`。当前v9按一次Ctrl+C等待保存退出后，在 `isaaclab` 环境执行 `OMP_NUM_THREADS=4 python scripts/tools/resume_stair_mastery.py`，从v9最新保存文件迁入v10；加 `--print-only`仅预览。后续原生恢复指定 `--load-run <实际v10目录名>`。规则、迁移和验证见 [Stage9 v10 调整与续训](docs/test_docs/07_上楼步态校正/v1_20261023/stage9_v10/调整与续训.md)。下文为历史记录。

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

窗口按键：`Z` 起立、`C` 策略、数字小键盘 `8/2` 前后、`4/6` 左右横移、`7/9` 左右转向、`H` 扫描、`X` 趴下、`R` / `Backspace` 重置到初始卧地状态（需重新按 `Z` 起立，再按 `C` 进入策略）、`Esc` 退出。控制按键优先处理，不再同时触发 MuJoCo 默认快捷功能；移动使用独立数字小键盘，非主键盘数字行。X11 直接读取按住状态，松开/失焦后移动命令归零；终端输入使用连发/超时机制。停车时检查遥测中的原始零命令。

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

## v1 第八阶段历史配置：固定输入的上下楼路径修正（2026-10-22）

历史配置 `ascent_entry_v1_stage8_fixed` 保留Stage7 Actor的247维输入、统一MLP的247→256→128→16结构和Critic的300维输入；所有网络参数键名、形状及总数保持不变。该配置采用13列固定宽度楼梯课程，保留用于复现实验与旧检查点回放；窄路线和混合地形训练请使用下方Stage9。旧的253/306维Stage8方案已撤回；旧文件仍可通过MuJoCo兼容加载器回放。

依次改进路径反馈、下楼门控和奖励。横向位置/速度反馈经现有有效vy通道进入策略，原始指令独立用于门控，保留legacy航向控制。下楼复用统一MLP的16个输出，以固定限幅和非训练参数的时间调度开放权限：前30轮为零，之后800轮线性增加。保留上楼输出约束，20/25cm窄踏面适当降低旧动作保留权重。上下楼路径偏差、横向速度和接触横滑均有代价，偏离路径后不能计为课程成功。上楼连续左右交替、逐轮不漏阶、同阶先后和到顶放平要求继续保留，不固定前后轮相隔级数。

13列地形包含8列上楼、1列平地、4列下楼，**上下楼均覆盖20/25/30/35cm踏深**，阶高课程为10/12/15/18/20/25cm。新增20cm上楼从10cm阶高、4阶任务开始，课程降级不会把20cm踏深改宽；25cm上楼列最低阶高15cm。下楼各踏深按0.5/0.7m/s两档速度分别累计成功，均达到最近8次至少6次成功才升级，连续3次失败降级。

建议用短程训练先复测默认楼梯及20/25cm踏深，同时查看上下楼横移、左右偏航、落阶覆盖、同阶先后、左右交替和平台过渡。固定输入版本的实现及实测证据见本地 `docs/test_docs/07_上楼步态校正/v1_20261022/stage8_fixed/改进与验证.md`；初始化与短测不代表正式续训后的完整步态验收。

截至14900轮的现有日志使用 `m20_stair_path_fixed247_20261022_v1_stage8_reward_v2`，实际从Stage7 `model_12300.pt`初始化，是固定宽楼梯课程的历史对照。v2奖励细节与验证见本地 `docs/test_docs/07_上楼步态校正/v1_20261022/stage8_reward_v2/改进与验证.md`。

Stage8的旧13列布局与Stage9的28列布局分别保留，避免旧配置回放或恢复时误用新课程。

## v1 第九阶段：窄楼梯与混合地形保留（2026-10-23）

Stage9 针对1.0/1.2m楼梯路线的横向漂移、踏面不足和25cm踏深下的滑移/立板碰撞增加路径约束与窄踏面接触代价。地形列固定为28列：0–7是20/25/30/35cm踏深与1.0/1.2m宽度组合的上楼，8是5m参考上楼，9–11是平地，12–19是对应的下楼任务，20–27是箱体、粗糙地面和双向坡面。上下楼高度行仍为10/12/15/18/20/25cm；降级仅降低高度行，列上的踏深和宽度不变。平地与非楼梯地形保留原命令、奖励及基于行走距离的地形课程，路径反馈只作用于楼梯列。

Stage9 v4（2026-10-26）取消3k/6k/1w等固定时间切换，默认由用户手动选择配比阶段；`--stair_sampling_mode auto`可选用趋势判稳与低成功率回退。容易组为30/35cm踏深、1.2m宽及5m参考楼梯；中等组为25cm/1.2m和30/35cm/1.0m；困难组为20cm/1.0m、20cm/1.2m和25cm/1.0m。表中比例仅指上楼环境内部，各组内三列均匀分配。

| 阶段参数 | 容易组目标 | 中等组目标 | 困难组目标 |
| --- | --- | --- | --- |
| 0 | 70% | 20% | 10% |
| 1 | 65% | 22.5% | 12.5% |
| 2 | 60% | 25% | 15% |
| 3 | 55% | 27.5% | 17.5% |
| 4 | 50% | 30% | 20% |
| 5 | 45% | 32.5% | 22.5% |
| 6 | 40% | 35% | 25% |
| 7 | 37.5% | 35% | 27.5% |
| 8 | 35% | 35% | 30% |
| 9 | 34.17% | 34.17% | 31.67% |
| 10 | 33.33% | 33.33% | 33.33% |

每次修改阶段后，在1000轮内从保存时的目标进度连续过渡到新配比；上调、下调、过渡中断后改方向均保留连续性。手动模式保持所选目标，达到3k/6k/1w或预算末尾都不会自行切换。高度和接触前缀仍独立按回合表现升降，降低阶高时保留踏深和宽度，20cm任务一直覆盖。40000轮是总预算，不是必须在预算内完成所有配比的要求。

初始化时按整数配额分配环境；随后下楼和平地/箱体/粗糙地面/坡面环境固定任务列，上楼仅在episode重置时从超额列向缺额列迁移必要的环境。满足当前配额后，重复重置不会换列；不会在阶段切换时让所有环境重新抽签。未换几何环境的高度行和接触课程进度保留；只有实际换了上楼几何的环境回到最低高度行及其规定的最短练习前缀，并清空连续计数和两档速度成功窗口。地形网格、各列踏深/宽度和奖励权重保持原配置。10584环境下，按当前逐列整数配额实际分配上楼5022个，下楼与非楼梯环境合计5562个，接近旧8192环境Stage9对应任务的绝对覆盖量。

日志在`Curriculum/stair_path/`下记录`sampling_stage`、`sampling_transition_fraction`、`sampling_budget_fraction`；`sampling_target_easy_fraction`/`middle`/`hard`和`sampling_actual_easy_fraction`/`middle`/`hard`分别显示目标与实际的上楼组内比例。`sampling_pending_ascent_fraction`表示尚需换列的上楼份额，`sampling_changed_fraction`表示本次重置批次实际换列比例。实际比例会等待相应环境结束回合后逐渐追上目标；`sampling_actual_ascent_fraction`、`sampling_actual_descent_fraction`和`sampling_actual_non_stair_fraction`可检查三个固定环境池。

Actor/Critic维度、参数形状和Stage7统一分支结构均不变。冻结的基础运动分支保留其原策略输出，但最终动作仍叠加可训练统一修正分支；因此冻结降低了遗忘风险，却不能单独证明粗糙地面、箱体、坡面上的组合策略无回归。Stage9继续采样这些地形并记录分类型地形等级，验收时要同时看各非楼梯类别行走表现和楼梯课程，不以楼梯成功率替代全地形回归。

在仓库根目录、现有isaaclab环境，从原v3实验的2000轮分支续训。这是显式的采样逻辑迁移，保留Actor/Critic、优化器、参考策略、环境分配、前缀/高度和双速度历史；不采用仅初始化权重的方式。`--stair_sampling_migrate_v3`仅接受首次切换前且配额仍为70/20/10的v3 checkpoint，不接受后期3k/6k/7200状态。课程版本为`m20_stair_path_fixed247_20261026_v4_readiness_sampling`：

```bash
OMP_NUM_THREADS=4 python -u scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v1 --m20_profile ascent_entry_v1_stage9 \
  --resume --load_run 2026-10-23_19-15-28_ascent_entry_v1_stage9_v3 --checkpoint model_2000.pt \
  --stair_sampling_migrate_v3 --stair_sampling_mode manual --stair_sampling_stage 0 \
  --run_name ascent_entry_v1_stage9_v4_manual_from2000 \
  --num_envs 10584 --max_iterations 38000 --seed 42 --device cuda:0 --headless
```

后续判断近期约600–1000轮的主要曲线基本平稳、不再明显提升时，按一次Ctrl+C，程序在本次PPO更新结束后保存并打印checkpoint路径，再退出；连续第二次Ctrl+C为强制退出，只能使用之前已保存的文件。下一次`--resume`指向**新v4分支**的实际`--load_run`和刚保存的`--checkpoint model_N.pt`，去掉`--stair_sampling_migrate_v3`，把`--stair_sampling_stage`加一；表现明显恶化时减一。`--max_iterations`是本次追加训练轮数，若仍希望到40000轮，应设为`40000-N`。不要每次都重新指向原v3的model_2000。正式长训练仍由用户启动。

自动模式为可选项，不要求高绝对成功率：在配比实际稳定后至少学习1000轮，用最近三个200轮窗口、相同几何/原阶高/原前缀/速度的足量回合统计拟合趋势；8项指标中至少5项趋稳、成功率也不再明显提升且无明显退化，连续两次复核才加一档。TensorBoard的0.9仅是显示平滑，不直接作用到200轮大窗口。若低于20%成功率且没有明显进步的任务组占上楼人口至少20%，或同条件成功率低于35%且没有明显进步，连续两次复核则退一档；被退回的档位需相关成功率至少改善5个百分点才自动重试，避免来回振荡。初始档不再降低几何配比，仍可通过原逐环境课程缩短前缀/降低阶高。样本不足会等待，不以缺数据当作零值或平台。具体阈值和诊断在`stair_path_readiness.py`与manifest中记录；这些判据已做CPU逻辑检查，训练效果尚待观察。

28列、6个高度行共168个地形单元；10584环境下简单平均每列约378个、每列每高度行约63个环境，实际占用由采样配额和各环境独立高度课程决定。检查上述采样日志，以及 `general_flat_terrain_level_mean`、`general_boxes_terrain_level_mean`、`general_random_rough_terrain_level_mean`、两个坡面地形等级和各楼梯列的成功/缺阶/路径误差、横向速度与接触滑移。CPU检查命令为`python scripts/tools/check_stair_path_sampling.py`和`python scripts/tools/check_stair_path_readiness.py`，不加载策略checkpoint、不执行完整恢复或仿真。详细切换步骤见[操作说明](docs/test_docs/07_上楼步态校正/v1_20261023/stage9_v4/阶段切换与续训.md)。

## 代码与开发记录

- [训练源码说明与整理记录](source/rl_training/rl_training/README.md)：模块职责、任务/profile 依赖、兼容文件及删除依据。
- [stair_teacher.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py)：几何、逐阶账本、奖励、命令与诊断；运行时软包络参数维护在 `TUNING`。
- [stair_teacher_env_cfg.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py)：观测、实体和默认奖励；[v1_stop_profile.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/v1_stop_profile.py) 应用配置覆盖。
- [stop_branch.py](source/rl_training/rl_training/models/stop_branch.py)、[motion_residual_actor.py](source/rl_training/rl_training/models/motion_residual_actor.py)、[stair_guarded_ppo.py](source/rl_training/rl_training/algorithms/stair_guarded_ppo.py)：自定义网络与 PPO 扩展。
- [deploy_mujoco.py](deploy/deploy_mujoco/deploy_mujoco.py) 与[评估工具](scripts/tools)：部署及回归。
- [技术改进路线：截至 2026-10-19](docs/TECHNICAL_IMPROVEMENT_ROADMAP_CN.md)：时间线、改动、证据、退化与待完成事项。

`docs/` 仅发布 `TECHNICAL_IMPROVEMENT_ROADMAP_CN.md`。其他文档、`.scratch/`、`logs/`、`outputs/` 以及权重、缓存继续仅本地保留。路线文档指向历史报告和本地备份的链接在新克隆中不可用；基准模型和结果需另行准备。当前源码和运行 manifest 优先于历史命令与奖励表。
