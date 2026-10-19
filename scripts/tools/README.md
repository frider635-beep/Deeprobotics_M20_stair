# v0 教师训练工具清单

整理日期：2026-10-19。当前教师训练工程记为 **v0 基线**；这是工程整理标签，不重写已有任务 ID、profile、控制器或 checkpoint 内部版本。

训练入口仍是 [train.py](../reinforcement_learning/rsl_rl/train.py)，部署入口仍是 [deploy_mujoco.py](../../deploy/deploy_mujoco/deploy_mujoco.py)。本轮从 34 个 Python 文件中删除 7 个明确不用于当前教师流程、且无保留代码依赖的专项脚本，保留 27 个。仍在使用和用途不确定的代码保留原内容；不是每个保留检查都要在日常训练时运行。

## 当前使用

| 程序 | 用途与边界 |
| --- | --- |
| [list_envs.py](list_envs.py) | 在 Isaac Lab 环境列出已注册任务 |
| [setup_conda_runtime.py](setup_conda_runtime.py) | 需要时修正 Conda C++ 运行库激活配置；不是训练步骤 |
| [analyze_stair_run.py](analyze_stair_run.py) | 读取 TensorBoard、导出训练指标摘要与图表 |
| [compare_runs.py](compare_runs.py) | 对比两个运行目录的实际配置 |
| [evaluate_m20_refinement.py](evaluate_m20_refinement.py) | 使用当前 MuJoCo loader 的固定工况综合评估 |
| [evaluate_m20_v1_stop.py](evaluate_m20_v1_stop.py) | 当前仍使用的停车/直行回归；名称保留兼容，须显式指定新 baseline |

## 通用检查与接线

| 程序 | 用途与边界 |
| --- | --- |
| [check_m20_evaluation_cli.py](check_m20_evaluation_cli.py) | 不加载策略、不进行物理回放地检查评估 CLI、报告与失败退出 |
| [check_m20_heading_hold.py](check_m20_heading_hold.py) | 当前部署控制器的模式切换、滤波和遥测检查，不证明策略质量 |
| [check_mujoco_terrains.py](check_mujoco_terrains.py) | 地形射线/参数和普通基准 actor 的短接线检查；显式传本仓库 `--model`，不直接使用分支检查点 |
| [smoke_stair_rewards.py](smoke_stair_rewards.py) | Isaac Lab 默认教师奖励接线短测；只适用普通基准检查点，不自动构建新增分支 |

当前模型含自定义分支时，优先通过两个 `evaluate_m20_*` 入口和更新后的 MuJoCo loader 验证，不能用普通 MLP 冒烟或旧导出器代替。

## 常用命令

从仓库根目录、在 `isaaclab` 环境执行。替换候选运行目录：

```bash
python scripts/tools/evaluate_m20_v1_stop.py --baseline pre_train/teacher_baseline_idle_stop.pt --candidate logs/rsl_rl/deeprobotics_m20_stair_teacher/RUN/model_300.pt --output .scratch/teacher_v0/model300_evaluation.json --require-pass
```

```bash
python scripts/tools/check_m20_evaluation_cli.py
```

## 共享夹具与用途尚不能排除的程序

以下 17 个程序保留原位置。它们包含共享函数、仍对应当前奖励实现的检查，或普通基准评估/导出功能；未将它们全部认定为日常必用。版本名只表示建立时的用途，不单凭文件名删除。

| 类别 | 保留程序 |
| --- | --- |
| 基础奖励/步态与共享夹具 | `check_stair_rewards.py`、`check_gait_safety.py`、`check_gait_refinement.py`、`check_gait_v3.py`、`check_stair_gait_fix.py`、`check_stair_preservation.py` |
| 方向/入口与共享状态 | `check_ascent_direction.py`、`check_descent_entry.py`、`check_m20_curriculum.py`、`check_m20_entry_pause.py` |
| 综合控制/安全检查及夹具 | `check_m20_refinement.py`、`check_m20_safety_v2.py`、`check_m20_stop_v2.py`、`check_m20_engineering.py` |
| 仍被历史验证程序复用的停车夹具 | `check_m20_v1_stop.py` |
| 普通基准网络评估/导出 | `evaluate_stair_gait.py`、`export_onnx_fast.py`；不能直接替代自定义分支的评估和导出 |

例如 `check_stair_rewards.py`、`check_gait_safety.py`、`check_m20_curriculum.py`、`check_m20_stop_v2.py`、`check_stair_preservation.py` 为其他保留脚本提供函数和夹具，直接删除会造成导入错误。其他未完全确认失去用途的检查也保留。

此前核验中，`check_stair_rewards.py --assert-fixed` 的半径状态夹具断言未通过，尚未作生产算法归因，因此保留待查；未为通过旧检查而修改训练逻辑。评估 CLI 和航向/接触检查曾通过，不代表所有历史检查通过或策略效果合格。

## 本轮已删除的 7 个文件

| 文件 | 判断依据 |
| --- | --- |
| `check_m20_v1_stop_v2.py` | 只检查旧停车 v2 的轮目标与降噪方案，当前运动残差不使用 |
| `check_m20_v1_stop_v3.py` | 旧停车 v3/轮目标对照；导入它的 v4/v5 专项同批删除，无其他代码引用 |
| `check_m20_v1_stop_v4.py` | 只检查 model900 腿教师/双参考，当前配置不使用 |
| `check_m20_v1_stop_v5.py` | 只检查 24 维停车分支，当前教师使用 3 维命令的运动残差 |
| `check_m20_descent_hold_v4.py` | 旧 v4 的路线与停车 XY 指令接管专项，当前 legacy_v1 配置不启用 |
| `check_m20_training_contract.py` | 旧静态常量/版本检查器不匹配当前布局，也未被当前训练调用 |
| `smoke_stair_student.py` | 只运行 Student 618 维接口，不属于当前教师训练流程 |

已检查 `scripts`、`source`、`deploy` 与 `docs` 中剩余 Python 的导入和文件名加载引用，没有指向这 7 个文件的代码引用。历史文档中的旧执行命令是版本记录，不再保证脚本仍在当前工具目录。

本轮删除前的备份与理由：[tools_prune_v0_20261019](../../.scratch/04_change_backups/tools_prune_v0_20261019/manifest.json)。7 个文件的原始字节保存在同目录 `removed.tar.xz` 归档，当前工具目录确已删除。前一轮完整备份仍在 [tools_before](../../.scratch/04_change_backups/tools_cleanup_v0_20261019/tools_before.tar.xz)。保留的 27 个 Python 文件内容未改，训练、奖励、网络及部署实现未改。

旧工具与历史文档中的版本名用于复现，不代表当前工程发布版本。整理后的历史报告见 [docs/test_docs](../../docs/test_docs/README.md)，技术路线位置不变。
