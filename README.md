# M20 stair and multi-terrain locomotion

[中文说明](README_CN.md) · Updated **2026-10-19**

Stage9 v10 (2026-10-27) changes ascent rear entry only: treads1–2 allow a leader change, tread3 fixes the rear leader, and subsequent rear order errors use rewards and strict gait metrics rather than terrain promotion gates. Front order, descent objectives/control and network shapes are retained. After stopping the current v9 run and waiting for its checkpoint, use `OMP_NUM_THREADS=4 python scripts/tools/resume_stair_mastery.py` in the activated IsaacLab environment. See [v10 rules and resume](docs/test_docs/07_上楼步态校正/v1_20261023/stage9_v10/调整与续训.md).

Current teacher engineering baseline: **v0**. This label preserves existing task/profile/checkpoint compatibility identifiers. See the [maintained tool list](scripts/tools/README.md).

This M20 workspace builds on [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training) and Isaac Lab. It includes wheeled pretraining, privileged stair-teacher training, MuJoCo deployment, telemetry, regression evaluation and an experimental perception-student interface.

The current refinement direction uses `pre_train/teacher_baseline_idle_stop.pt`: preserve its reported stopping behavior while learning corrections for lateral drift and collisions during the first descent steps. The code exists; the available records do **not** establish that a newly trained policy has solved these problems.

## Tasks and network contracts

| Task ID | Purpose |
| --- | --- |
| `Flat-Deeprobotics-M20-v0` | Flat-ground velocity tracking |
| `Rough-Deeprobotics-M20-v0` | Rough-terrain velocity tracking |
| `Rough-Deeprobotics-M20-PreTeacher-v0` | Wheeled pretraining |
| `Rough-Deeprobotics-M20-StairTeacher-v0` | Privileged stair/mixed-terrain teacher |
| `Rough-Deeprobotics-M20-StairStudent-v0` | Experimental perception interface; no complete distillation pipeline |

| Teacher configuration | Actor input | Critic input | Policy structure |
| --- | --- | --- | --- |
| Standard teacher; `v1_stop` through `v1_stop_v4` | 244 | 285 | Shared `244→512→256→128→16` MLP; some profiles store additional training-only groups |
| `v1_stop_v5` | 244 + 24 | 285 + 24 | Frozen baseline plus `268→128→64→16` zero-command branch |
| `idle_stop_motion_v1` | 244 + 3 | 285 + 3 | Frozen idle-stop baseline plus `247→128→64→16` commanded-motion residual |

The original teacher input contains 57 proprioceptive/command/history values and 187 height samples. Outputs remain 12 leg-position actions and 4 wheel-velocity actions. The student has 618 actor/critic inputs and a separate 244-value teacher query group. Equal dimensions do not imply identical command semantics or controller versions.

## Environment and assets

Run from the repository root in a compatible Isaac Sim / Isaac Lab environment. The local Conda environment is `isaaclab`; older records use `m20_wzh`. Local metadata on 2026-10-19: Python 3.11.14, Isaac Sim 5.1.0.0, `isaaclab` package 0.54.3, RSL-RL 5.0.1, PyTorch 2.7.0+cu128 and MuJoCo 3.3.7. The package version is not an upstream Isaac Lab release-tag claim; this is a reproducibility snapshot.

```bash
conda activate isaaclab
git submodule update --init --recursive
python -m pip install -e source/rl_training
python scripts/tools/list_envs.py
```

The asset submodule provides `deep_robotics_model/M20/usd/M20.usd` for Isaac Lab and `deep_robotics_model/M20/mjcf/M20.xml` for MuJoCo. Teacher/preteacher USD paths resolve from the repository; `RL_TRAINING_M20_USD_PATH` overrides the location. Training entry points prioritize this checkout's source. If needed, `python scripts/tools/setup_conda_runtime.py` checks the C++ ABI before installing activation hooks; reactivate the environment afterward.

## Train

Wheeled pretraining:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Rough-Deeprobotics-M20-PreTeacher-v0 --max_iterations 20000 --device cuda:0 --headless
```

Transfer a conventional preteacher actor into stair training; replace the example run path:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Rough-Deeprobotics-M20-StairTeacher-v0 --init_actor_from logs/rsl_rl/deeprobotics_m20_pre_teacher/RUN/model_19999.pt --run_name stair_teacher --max_iterations 3000 --device cuda:0 --headless
```

Current idle-stop-baseline refinement:

```bash
conda run --no-capture-output -n isaaclab python -u scripts/reinforcement_learning/rsl_rl/train.py --task Rough-Deeprobotics-M20-StairTeacher-v0 --m20_profile idle_stop_motion_v1 --init_actor_from pre_train/teacher_baseline_idle_stop.pt --run_name idle_stop_motion_v1_stage1 --num_envs 4096 --max_iterations 1000 --seed 42 --device cuda:0 --headless
```

Supply the baseline separately: weights are ignored by Git. Required SHA256: `0e6d0c93560b5a3f7ce9219c7b6a0cea84bd4a2d3f5dba33c02d79f7d4eb5f34`. This profile freezes the baseline MLP and Gaussian-distribution parameters, zero-initializes the residual output, and applies bounded corrections only when a raw command component exceeds `1e-4`. For the same observation, zero-command outputs retain the baseline mapping; changed preceding motion can still change the stopping state.

The profile strengthens ascent tracking, adds descent lateral-position/velocity shaping, and enables moderate close-distance/closing-speed costs. These are candidate choices, not validated improvements. Warm up the critic for 20 iterations, save every 100, and inspect candidates at 100/300/500 before extending the initial 1000-iteration budget. Iterations are PPO updates, not episodes.

| Option | Meaning |
| --- | --- |
| `--init_actor_from PATH` | Initialize policy; reset critic, optimizer and iteration count by default |
| `--init_critic_from PATH` | Also initialize a shape-compatible critic; requires actor initialization |
| `--resume --load_run RUN --checkpoint FILE` | Restore compatible full state; `--max_iterations` adds iterations |
| `--m20_profile NAME` | Select a profile before environment creation; omission uses environment defaults |
| `--preserve_actor_from PATH` | Reference for supported shared-policy profiles; omit for `idle_stop_motion_v1` |
| `--descent_guide_iters N --descent_guide_decay_iters M` | Optional front-lead guidance/decay for `idle_stop_motion_v1` only; disabled by default (`N=0`) |
| `--num_envs N --seed N --device cuda:0` | Environment count, seed and device |

Actor initialization and `--resume` are mutually exclusive. Profiles check baseline content hashes: `v1_stop_v5` expects original v1/model5100, not the idle-stop baseline. Running processes do not hot-load reward/network edits. Logs are in `logs/rsl_rl/deeprobotics_m20_stair_teacher/<timestamp_run>/`; inspect `params/launch_manifest.json` and checkpoint manifests for source/baseline hashes, rewards and observation groups.

## Deploy and replay

First replay the idle-stop baseline under the motion profile's `legacy_v1` controller:

```bash
python deploy/deploy_mujoco/deploy_mujoco.py --model deep_robotics_model/M20/mjcf/M20.xml --checkpoint pre_train/teacher_baseline_idle_stop.pt --terrain-xml deploy/deploy_mujoco/terrains/stairs_ascent.xml --legacy-heading-hold --viewer --telemetry --telemetry-every 5 --output .scratch/idle_stop_baseline/ascent.npz
```

Replace the checkpoint with a trained candidate. New profile manifests select `legacy_v1` automatically; old checkpoints without manifests need the explicit flag for this comparison. Verify baseline behavior under the same controller before attributing differences to training.

Viewer keys: `Z` stand, `C` policy, keypad `8/2` forward/reverse, `4/6` left/right, `7/9` turn left/right, `H` scan, `X` prone, `R` / `Backspace` reset to the initial prone state (press `Z` again to stand, then `C` for policy), `Esc` exit. Teleop keys are consumed before native viewer shortcuts; movement uses the separate numeric keypad, not the top number row. X11 reads held keys directly; release/focus loss clears movement. Terminal input uses repeat/timeouts. Confirm raw-zero commands in telemetry.

Use `stairs_descent.xml` for descent and **`--terrain flat`** for flat ground. Omitting terrain arguments defaults to ascent. `--viewer --autoplay` executes automatic commands. Policy/physics periods are 0.02/0.005 seconds. Batch example:

```bash
python deploy/deploy_mujoco/deploy_mujoco.py --model deep_robotics_model/M20/mjcf/M20.xml --checkpoint pre_train/teacher_baseline_idle_stop.pt --terrain flat --legacy-heading-hold --command-profile deploy/deploy_mujoco/profiles/zero_hold.json --telemetry --output .scratch/idle_stop_baseline/flat.npz
```

See [terrain parameters](deploy/deploy_mujoco/terrains/README_CN.md). Telemetry records commands, controller state, body/wheel motion, route error, gait and solved contacts. Contact-substep counts are not independent collision-event counts.

Isaac playback for **conventional non-branch checkpoints**:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task Rough-Deeprobotics-M20-StairTeacher-v0 --checkpoint logs/rsl_rl/deeprobotics_m20_stair_teacher/RUN/model_500.pt --num_envs 1 --keyboard
```

Current `play.py` does not reconstruct custom branch profiles from manifests. Use the updated MuJoCo loader for `v1_stop_v5` and `idle_stop_motion_v1`. Neither supports legacy single-input JIT/ONNX export; exporting only the frozen MLP would omit learned corrections. `deploy/deploy_real/` is a placeholder, not a completed hardware deployment path.

## Evaluate and monitor

Explicitly select the new baseline; the evaluator otherwise defaults to historical model5100:

```bash
python scripts/tools/evaluate_m20_v1_stop.py --baseline pre_train/teacher_baseline_idle_stop.pt --candidate logs/rsl_rl/deeprobotics_m20_stair_teacher/RUN/model_300.pt --output .scratch/idle_stop_motion_v1/model300_evaluation.json --require-pass
```

Both models use the same legacy controller. Defaults cover nine cases and yaw offsets 0°, −3°, +3°, checking stopping, progress, lateral deviation and contact regression. Existing tolerances test non-regression; passing alone does not establish meaningful reductions in drift or first-three-step collisions. Inspect those measurements separately and repeat first-step/mid-stair/reverse-stop/restart trials. Truncated `--steps` runs are not acceptance tests. `--require-pass` returns nonzero on failure and retains the report.

```bash
tensorboard --logdir logs/rsl_rl/deeprobotics_m20_stair_teacher
```

| Signals | Interpretation |
| --- | --- |
| `Loss/motion_residual_rms`, `Loss/motion_command_fraction`, `Loss/raw_zero_command_fraction` | Residual activity and command exposure |
| `Curriculum/ascent_direction/*`, `Curriculum/descent_direction/*` | Direction/route statistics; small heading error does not exclude lateral slip |
| `Curriculum/gait_quality/*`, `Curriculum/step_completion/*` | Posture, clearance, step events; inspect sample counts alongside ratios |
| `Episode_Reward/*` | Weighted reward contributions, not independent quality scores |
| `Perf/collection_time`, `Perf/learning_time`, `Perf/logging_time` | Sampling, PPO and logging cost; wall time includes other overhead |
| `Curriculum/zero_stop/*` | Stop profiles only; this callback is disabled in `idle_stop_motion_v1`, so evaluate stopping externally |

Hold terrain, commands, controller, initial state and software versions fixed. Lower teacher loss/action noise, higher reward, or smaller drift caused by refusing to move does not establish success.

Append `env.teacher_detailed_diagnostics=true` when detailed statistics are needed; it changes diagnostics rather than rewards or task difficulty.

## Source and development records

- [Training source guide and cleanup record](source/rl_training/rl_training/README.md): module roles, task/profile dependencies, compatibility files and removal reasons (Chinese).
- [stair_teacher.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py): geometry, step ledgers, rewards, command handling and diagnostics; `TUNING` stores runtime envelope parameters.
- [stair_teacher_env_cfg.py](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py): observations, entities and default rewards; [v1_stop_profile.py](source/rl_training/rl_training/v1_stop_profile.py) applies profile overrides.
- [stop_branch.py](source/rl_training/rl_training/stop_branch.py), [motion_residual_actor.py](source/rl_training/rl_training/motion_residual_actor.py), [stair_guarded_ppo.py](source/rl_training/rl_training/stair_guarded_ppo.py): custom networks and PPO extensions.
- [deploy_mujoco.py](deploy/deploy_mujoco/deploy_mujoco.py) and [evaluation tools](scripts/tools): deployment and regression.
- [Technical improvement roadmap through 2026-10-19](docs/TECHNICAL_IMPROVEMENT_ROADMAP_CN.md): chronology, changes, evidence, regressions and remaining work.

Only `docs/TECHNICAL_IMPROVEMENT_ROADMAP_CN.md` is published from `docs/`. Other documents, `.scratch/`, `logs/`, `outputs/`, checkpoints and caches remain local. The roadmap's links to historical reports and local backups will not resolve in a fresh clone; supply baselines/results separately. Current source and run manifests take precedence over historical commands and reward tables.
