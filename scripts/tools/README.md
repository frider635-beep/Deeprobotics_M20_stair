# v0 教师训练工具清单

2026-10-27 Stage9 v9：新增 `check_stair_path_evidence.py` 检查逐轴质量晋级、升高与后轴第2阶条件、近期通行窗口抑制偶发退级、必要降高/已学下限、v8状态迁移、真实课程调用与左右镜像相位；`check_stair_ascent_pairing.py`继续检查理想交替与物理硬约束、错序信用、奖励迁移和下楼对照（`--baseline-root <改动前rl_training包目录>`）。全部仅CPU，不加载策略。`resume_stair_mastery.py --print-only`预览从当前v8最新断点迁入v9；去掉预览参数才启动。后续恢复须指定 `--load-run <实际v9目录名>`。完整规则和命令见 [v9 调整与续训](../../docs/test_docs/07_上楼步态校正/v1_20261023/stage9_v9/调整与续训.md)。

保留的v6课程检查：`check_stair_path_mastery.py`验证已学6阶保护、12→10→8→6同高度退级、持续失败降高、换几何清理、状态保存和v5迁移；规则见 [v6 记录](../../docs/test_docs/07_上楼步态校正/v1_20261023/stage9_v6/调整与续训.md)。

2026-10-26 Stage9 v5 当前工具：`check_stair_path_course.py` 检查渐进课程、v4 状态映射与原回合标签；`check_ppo_diagnostics.py` 使用合成 Actor/Critic 检查实际 PPO 更新，确认非零保护梯度的诊断不改变更新结果；`analyze_stair_optimizer.py --run <运行目录> --last 600 --output <报告.json>` 读取 TensorBoard 的梯度、KL 与参数更新量，输出启发式学习率判断。这些工具不加载策略 checkpoint。续训命令和指标定义见 [v5 说明](../../docs/test_docs/07_上楼步态校正/v1_20261023/stage9_v5/改进与续训.md)。下文 v3→v4 迁移记录为历史流程，当前 v4→v5 使用 `--stair_course_migrate_v4`。

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

### Stage6 左右交替奖励检查

`check_ascent_alternating.py --source-checkpoint <stage5 checkpoint> --output <report.json> --fixture-checkpoint <test_fixture.pt>` 覆盖连续起摆、奖励门控、速度课程、实际CPU PPO更新与checkpoint恢复。`check_ascent_sequence_observer.py --profile stage6`可重放既有数据，`--cases`可筛选工况。`preflight_ascent_stage5.py --profile stage6 --checkpoint <checkpoint>`只做CPU初始化预检，无需旧示范。

`check_ascent_reward_manager.py --output <json>`：使用本机 IsaacLab 的奖励管理器和 `RotationGaitSymmetry` 实际类定义，在 CPU 场景适配器上复现旧报错并验证 Stage6 类奖励的初始化、正奖励门控及部分环境 reset；不启动 Kit/PhysX。

### Stage7 合并与验证

`merge_ascent_branches.py --source <旧checkpoint> --data <采集目录...> --output <新checkpoint>`将三个旧上楼分支的合成动作蒸馏到单个MLP；按完整零偏移工况留出验证，不启动PPO。`--warm-start`可对迁移模型用更多状态继续拟合。它是动作函数迁移，不是理想步态示范认证。

`check_ascent_unified.py --checkpoint <Stage7 checkpoint> --output <json>`检查真实单分支合并、基础权重、3阶桥接与固定4阶组、双速度升级、前后轮同阶约束、领先轮等待配对、CPU PPO更新和课程恢复。训练中新增`Curriculum/ascent_unified/`下的失败分类、逐轮第3/4阶覆盖及周期诊断。


## Stage8 上下楼路径修正（2026-10-22）

- `prepare_stair_path.py --source SOURCE_STAGE7.pt --output NEW_STAGE8.pt`：在同一统一修正网络中增加6维路径输入，零初始化新增输入列与下楼输出增益，核对基础权重、维度和迁移输出。不覆盖已有checkpoint。
- `check_stair_path.py --checkpoint NEW_STAGE8.pt --source SOURCE_STAGE7.pt --output RESULT.json`：检查实际MLP数量、迁移函数、下楼梯度、真实CPU PPO、冻结基础、保存/恢复、训练/部署批量/单例反馈一致性、MuJoCo缓存、下楼驻足奖励防护和双速度课程，含10列源课程到13列新课程的迁移与断点布局校验。
- `evaluate_stair_path.py --source SOURCE_STAGE7.pt --candidate NEW_STAGE8.pt --output-dir NEW_DIRECTORY`：上/下楼、两速度、三起点的接触与路径对照。默认20阶15/30cm，平台记录延后1.2m；上楼单独统计四轮落稳、同阶先后、连续交替、平台联合通过。`--height`/`--depth`改变尺寸；`--reuse-source-report`可复用checkpoint哈希、几何、速度与起点均匹配的原始源策略工况，报告记录来源哈希。

`collect_m20_gait_reference.py`现在也可采集下楼，使用checkpoint指定控制器，Stage8额外记录`path_control`和`descent_gate`。旧`analyze_m20_gait_reference.py`的递增台阶事件定义用于上楼，下楼路径分析优先用`evaluate_stair_path.py`，不要将上楼落阶逻辑直接套用到下楼。

当前路径工具采用固定网络版本：`prepare_stair_path.py`逐张量原样复制Actor/Critic，仅附加控制元数据；`check_stair_path.py`验证相同键名、形状、参数总数和真实PPO/断点恢复；`evaluate_stair_path.py`分别记录横移与理想步态，含上下楼20cm踏深。正式训练无需运行prepare，直接加载原model_11400.pt。

`check_stair_path_objectives.py --output <json>`验证当前奖励修订v2的行为反例：有效/原始命令选择、受控回线与远离路线、正常滚动与纵向空转、单轮打滑/腾空排除、左右镜像/参考路线旋转、入口强调不重复放大。该工具不加载model12100，不启动训练。

`check_stair_path_sampling.py`在CPU检查Stage9 v4：默认手动保持阶段、不按时间表切换、1000轮连续过渡、20cm始终覆盖、上楼重置时的最小必要换列、固定下楼/非楼梯任务池和未换几何的进度保留。`check_stair_path_readiness.py`使用合成回合数据检查可选趋势判稳、明显提升/退化时保持、低成功率降级及重试滞回、调度器状态序列化、中途反向切换连续性、v3初始配额迁移元数据和Ctrl+C在完整更新后保存的时机。均不加载策略checkpoint、不执行完整恢复或仿真。

`check_stair_path_performance.py`使用合成CPU场景检查真实Stage9观测器、稀疏/显式重置与同一物理tick重复调用。可用`--baseline-root <旧rl_training包目录>`逐tick比较优化前后的全部观测器张量，并统计非零索引操作次数；`--num-envs 10584`覆盖正式环境数量。CPU计时不代表PhysX/GPU训练加速比例；不加载策略checkpoint、不启动仿真或PPO。
