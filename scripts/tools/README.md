# v0 教师训练工具清单

整理日期：2026-10-19。当前教师训练工程记为 **v0 基线**；这是工程整理标签，不重写已有任务 ID、profile、控制器或 checkpoint 内部版本。

2026-10-20 新增上楼参考节奏工具（不包含在下文10月19日的27个保留脚本统计中）：

| 程序 | 用途与边界 |
| --- | --- |
| [collect_m20_gait_reference.py](collect_m20_gait_reference.py) | 沿用部署actor/控制器，采集50 Hz策略和200 Hz接触、滑移、状态；支持运动残差检查点，不训练 |
| [analyze_m20_gait_reference.py](analyze_m20_gait_reference.py) | 提取稳定落地、逐阶事件、相位与分速度节奏；严格干净筛选和仅节奏候选分别记录 |
| [check_m20_gait_reference.py](check_m20_gait_reference.py) | 验证纯滚动/滑移、事件归因和采集对部署回放无扰动；不证明策略或参考动作无碰撞 |
| [check_ascent_entry_v1.py](check_ascent_entry_v1.py) | v1完整继承、冻结、门控、限幅和滑移解析反例 |
| [check_ascent_entry_stage2.py](check_ascent_entry_stage2.py) | 第二阶段逐轮落阶记账、短接触/漏阶反例、完整v1权重继承及原生部署一致性 |
| [check_ascent_landing.py](check_ascent_landing.py) | 第三阶段完整源策略冻结、入口门控、落轮引导、原生加载和CPU PPO预热/更新检查；不验证Isaac物理环境或步态效果 |
| [check_ascent_sequence.py](check_ascent_sequence.py) | 第四阶段完整源冻结、同阶前轮先后、灵活前后阶差、支撑裕度及到顶过渡反例和CPU PPO检查 |
| [check_ascent_sequence_observer.py](check_ascent_sequence_observer.py) | 归档轨迹经CPU场景适配器重放新观察器，检查物理/策略缓存、奖励有限及reset；不验证PhysX或精确滑移一致性 |
| [check_ascent_reward_audit.py](check_ascent_reward_audit.py) | 核对旧跨阶交替/同阶惩罚全部停用，测试禁止重新启用、同阶目标计分优于跨阶漏阶及任务通行奖励独立性 |
| [analyze_ascent_entry_reachability.py](analyze_ascent_entry_reachability.py) | 归档漏阶位置的固定机身PD目标几何分析；不等同于动力学可达性 |
| [analyze_ascent_landing_preparation.py](analyze_ascent_landing_preparation.py) | 实际关节姿态附近的支撑约束局部IK，记录不可行残差及穿透；不保证训练能学会 |
| [smoke_ascent_entry_v1.py](smoke_ascent_entry_v1.py) | 新注册v1任务的真实Isaac接线验证，断言接触/落地奖励生效 |
| [evaluate_ascent_entry_v1.py](evaluate_ascent_entry_v1.py) | v1候选与六组baseline逐工况配对；失败保留报告，不自动替换模型 |

第二阶段仍使用v1任务，须显式指定`--m20_profile ascent_entry_v1_stage2`，从完整v1检查点初始化；`smoke_ascent_entry_v1.py`可由检查点元数据自动选择第二阶段，或用`--profile ascent_entry_v1_stage2`检查旧Actor在新环境中的接线。详见[第二阶段说明](../../docs/test_docs/07_上楼步态校正/v1_20261020/stage2_20261020/调整说明.md)。

第三阶段使用`--m20_profile ascent_entry_v1_stage3`，冻结完整第二阶段Actor并训练新增入口分支。按用户要求不主动运行GPU短测；本次仅执行CPU检查、CPU回放及归档分析，正式训练由用户启动。日志根目录仍为`logs/rsl_rl/deeprobotics_m20_stair_teacher_v1_stage2/`。详见[第三阶段说明](../../docs/test_docs/07_上楼步态校正/v1_20261020/stage3_20261020/调整说明.md)。

第四阶段使用`--m20_profile ascent_entry_v1_stage4`，冻结完整第三阶段Actor，加入同阶先后落前轮、全程逐阶支撑、灵活前后跟随和到顶过渡。回放评估使用默认速度，不采用训练入口速度课程。详见[第四阶段说明](../../docs/test_docs/07_上楼步态校正/v1_20261020/stage4_20261020/调整说明.md)；本次未运行GPU训练启动测试或正式训练。

默认检查点为用户指定的 `pre_train/teacher_baseline_.pt`。可复现的2026-10-20结果位于
`docs/test_docs/07_上楼步态校正/reference_20261020/分析报告.md`；原始数据与JSON/NPZ节奏模板同目录。

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

## 上楼第五阶段工具（2026-10-21）

|工具|用途|
|---|---|
|`preflight_ascent_stage5.py`|CPU检查来源、Actor247/Critic300及示例动作包络，不启动Isaac|
|`check_ascent_adaptive.py`|顺序/漏阶/落地信用反例、网格几何、实际PPO与课程/示例断点恢复|
|`check_ascent_sequence_observer.py --profile stage5`|新观察器的归档接触重放；可用 `--comparison-json` 检查不同尺寸|
|`search_ascent_contact_preparation.py`|CPU有限时域接触准备搜索；仅接触验证成功才输出 `verified_demonstration.npz`|
|`compare_ascent_warmstarts.py`|比较检查点在不同尺寸下的通行、接触、横偏；`--extended` 增加上限与未见尺寸组合|
|`smoke_ascent_entry_v1.py --profile ascent_entry_v1_stage5`|手动运行的Isaac接触/奖励短测，本次没有执行|

仓库根目录 `scripts/train_ascent_stage5.sh` 先运行CPU预检查，再启动独立第五阶段；`--check-only` 不启动训练。完整训练仍由用户手动启动。示例和检查报告保存在本地 `docs/test_docs/07_上楼步态校正/v1_20261021/stage5_20261021`，不随源码发布。局部接触示例不是整段上楼策略验收。

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


### 训练包目录与兼容性检查

`python scripts/tools/check_training_package_layout.py` 验证迁移后的模块位置、旧导入别名与规范导入的对象一致性、内部导入路径及 JSON 资源。该检查跳过 Isaac 任务发现，不启动仿真或训练。目录和逐文件迁移表见 [训练包说明](../../source/rl_training/rl_training/README.md)。
