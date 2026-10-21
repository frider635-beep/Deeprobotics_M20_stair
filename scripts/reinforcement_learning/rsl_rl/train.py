# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import hashlib
import json
from contextlib import redirect_stdout
from io import StringIO
import os
import statistics
import sys
import time
from pathlib import Path
from types import MethodType

# Use this checkout even when m20_wzh also has an older rl_training installation.
_LOCAL_SOURCE = Path(__file__).resolve().parents[3] / "source" / "rl_training"
sys.path.insert(0, str(_LOCAL_SOURCE))

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

ASCENT_PROFILES = ('ascent_entry_v1', 'ascent_entry_v1_stage2', 'ascent_entry_v1_stage3', 'ascent_entry_v1_stage4', 'ascent_entry_v1_stage5')

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--descent_guide_iters", type=int, default=0,
                    help="Keep the optional front-lead descent shaping at full strength for this many PPO iterations.")
parser.add_argument("--descent_guide_decay_iters", type=int, default=3000,
                    help="Linearly decay front-lead descent shaping to zero over this many PPO iterations.")
parser.add_argument(
    "--init_actor_from", type=str, default=None,
    help="Initialize only the actor from an RSL-RL checkpoint; critic, optimizer, and iteration start fresh.",
)
parser.add_argument("--init_critic_from", type=str, default=None,
                    help="Initialize critic weights with --init_actor_from, keeping a fresh optimizer.")
parser.add_argument("--preserve_actor_from", type=str, default=None,
                    help="Freeze a task-capable reference for StairGuardedPPO (also persisted in checkpoints).")
parser.add_argument("--stop_teacher_from", type=str, default=None,
                    help="v1_stop_v4 frozen model900 leg teacher; defaults to its initial actor.")
parser.add_argument("--m20_profile", choices=("v1_stop", "v1_stop_v2", "v1_stop_v3", "v1_stop_v3_target_control", "v1_stop_v4", "v1_stop_v5", "idle_stop_motion_v1", *ASCENT_PROFILES), default=None,
                    help="Select an explicit M20 actor and its isolated refinement profile.")
parser.add_argument('--ascent_learning_rate',type=float,default=None,
                    help='Optional positive learning rate for the v1 ascent correction; recorded in manifest.')
parser.add_argument('--ascent_demo', type=str, nargs='+', default=None,
                    help='Stage5 contact-verified NPZ demonstrations; stored in checkpoints and decayed over 800 updates.')
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task == 'Rough-Deeprobotics-M20-StairTeacher-v1':
    if args_cli.m20_profile not in (None, *ASCENT_PROFILES):
        parser.error('The v1 task requires an ascent_entry_v1 profile')
    args_cli.m20_profile=args_cli.m20_profile or 'ascent_entry_v1'
elif args_cli.m20_profile in ASCENT_PROFILES:
    parser.error('Ascent profiles require --task Rough-Deeprobotics-M20-StairTeacher-v1')
if args_cli.ascent_learning_rate is not None:
    if args_cli.m20_profile not in ASCENT_PROFILES or not 0.<args_cli.ascent_learning_rate<1.:
        parser.error('--ascent_learning_rate requires v1 and a finite value in (0,1)')
if args_cli.ascent_demo and (args_cli.m20_profile != 'ascent_entry_v1_stage5' or args_cli.resume):
    parser.error('--ascent_demo requires fresh stage5 initialization; full resume restores embedded demonstrations')
if args_cli.init_actor_from and args_cli.resume:
    parser.error("--init_actor_from and --resume cannot be used together")
if args_cli.init_critic_from and (not args_cli.init_actor_from or args_cli.resume):
    parser.error("--init_critic_from requires --init_actor_from and cannot be used with --resume")
if args_cli.stop_teacher_from and (args_cli.m20_profile != "v1_stop_v4" or args_cli.resume):
    parser.error("--stop_teacher_from requires a fresh v1_stop_v4 run; full resume restores its saved teacher")
if args_cli.m20_profile == "idle_stop_motion_v1" and args_cli.preserve_actor_from:
    parser.error("idle_stop_motion_v1 freezes its initialized baseline; omit --preserve_actor_from")
if args_cli.m20_profile in ("v1_stop_v4", "v1_stop_v5") and args_cli.resume and args_cli.preserve_actor_from:
    parser.error("Full stop-profile resume restores its frozen policy; omit --preserve_actor_from")
if args_cli.descent_guide_iters < 0 or args_cli.descent_guide_decay_iters < 0:
    parser.error("Descent guide iteration counts must be non-negative")
if args_cli.descent_guide_iters > 0 and args_cli.m20_profile != "idle_stop_motion_v1":
    parser.error("--descent_guide_iters requires --m20_profile idle_stop_motion_v1")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# suppress noisy omni.usd warnings (e.g. unresolved visual prim references)
import carb
carb.logging.acquire_logging().set_level_threshold_for_source(
    "omni.usd", carb.logging.LogSettingBehavior.OVERRIDE, carb.logging.LEVEL_ERROR
)

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "5.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    cmd = [sys.executable, "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401
from rl_training.utils.metric_aggregation import pool_episode_metrics

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _descent_guide_scale(iteration, guided_iterations, decay_iterations):
    if iteration < guided_iterations:
        return 1.0
    if decay_iterations <= 0:
        return 0.0
    return max(0.0, 1.0 - (iteration - guided_iterations) / decay_iterations)


def use_compact_console_logging(logger, fast=False, descent_guide_schedule=None):
    """Keep RSL-RL's full TensorBoard logging while shortening its console output."""
    original_log = logger.log

    def compact_log(self, *args, **kwargs):
        # Logger.log writes every scalar to TensorBoard before printing its full table.
        # Capture only that table; the writer and episode buffers remain untouched.
        log_start = time.perf_counter()
        self.ep_extras[:] = pool_episode_metrics(self.ep_extras)
        if fast and self.logger_type == 'tensorboard' and not self.cfg['algorithm']['rnd_cfg']:
            from rl_training.utils.fast_logging import write_iteration
            write_iteration(self, **kwargs)
        else:
            with redirect_stdout(StringIO()):
                original_log(*args, **kwargs)
        log_elapsed = time.perf_counter()-log_start
        if fast and self.writer is not None:
            self.writer.add_scalar('Perf/logging_time', log_elapsed, kwargs['it'])

        iteration = kwargs["it"]
        if descent_guide_schedule and self.writer is not None:
            guide_scale = _descent_guide_scale(
                iteration, descent_guide_schedule["guided_iterations"],
                descent_guide_schedule["decay_iterations"])
            self.writer.add_scalar("Curriculum/descent_entry_guide_scale", guide_scale, iteration)
            for name, weight in descent_guide_schedule["base_weights"].items():
                self.writer.add_scalar(f"Curriculum/descent_entry_guide_weight/{name}",
                                       weight * guide_scale, iteration)
        start_iteration = kwargs["start_it"]
        total_iterations = kwargs["total_it"]
        collect_time = kwargs["collect_time"]
        learn_time = kwargs["learn_time"]
        loss_dict = kwargs["loss_dict"]
        action_std = kwargs["action_std"]

        samples = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        iteration_time = collect_time + learn_time
        fps = int(samples / iteration_time) if iteration_time > 0 else 0
        mean_reward = statistics.mean(self.rewbuffer) if self.rewbuffer else None
        mean_episode_length = statistics.mean(self.lenbuffer) if self.lenbuffer else None
        completed = iteration + 1 - start_iteration
        remaining = total_iterations - iteration - 1
        eta_seconds = self.tot_time / completed * remaining if completed > 0 else 0.0

        def metric(value, precision=3):
            return f"{value:.{precision}f}" if value is not None else "n/a"

        print(
            f"[train] {iteration + 1}/{total_iterations}"
            f" | reward={metric(mean_reward, 2)}"
            f" | ep_len={metric(mean_episode_length, 0)}"
            f" | fps={fps}"
            f" | action_std={metric(action_std.mean().item(), 3)}"
            f" | value_loss={metric(loss_dict.get('value'), 4)}"
            f" | policy_loss={metric(loss_dict.get('surrogate'), 4)}"
            f" | iter={iteration_time:.2f}s"
            f" | ETA={eta_seconds / 3600:.1f}h"
        )

    logger.log = MethodType(compact_log, logger)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    profile_manifest = None
    if args_cli.m20_profile:
        if args_cli.task not in ("Rough-Deeprobotics-M20-StairTeacher-v0", "Rough-Deeprobotics-M20-StairTeacher-v1"):
            raise ValueError("--m20_profile requires the M20 stair teacher task")
        from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.v1_stop_profile import apply_profile, validate_initialization
        if not agent_cfg.resume:
            validate_initialization(args_cli.init_actor_from, args_cli.preserve_actor_from,
                                    args_cli.m20_profile, args_cli.stop_teacher_from)
        profile_manifest = apply_profile(env_cfg, agent_cfg, args_cli.m20_profile)
        if args_cli.ascent_learning_rate is not None:
            agent_cfg.algorithm.learning_rate=args_cli.ascent_learning_rate
        print(f"[INFO] M20 profile: {profile_manifest}")
    descent_guide_schedule = None
    if args_cli.descent_guide_iters > 0:
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_teacher
        # Conservative shaping: a paired touchdown plus the order event are
        # meaningful, while the dense potential stays small relative to the
        # existing whole-step completion reward (+3).
        guide_weights = {
            "entry_front_lead_weight": 0.5,
            "entry_front_progress_weight": 0.5,
            "entry_sequence_weight": 0.5,
        }
        stair_teacher.TUNING.update(guide_weights)
        env_cfg.rewards.stair_descent_entry_front_lead_reward.weight = guide_weights["entry_front_lead_weight"]
        env_cfg.rewards.stair_descent_entry_front_progress_reward.weight = guide_weights["entry_front_progress_weight"]
        env_cfg.rewards.stair_descent_entry_sequence_reward.weight = guide_weights["entry_sequence_weight"]
        descent_guide_schedule = {
            "guided_iterations": args_cli.descent_guide_iters,
            "decay_iterations": args_cli.descent_guide_decay_iters,
            "base_weights": guide_weights,
        }
        print(f"[INFO] Descent front-lead guide schedule: {descent_guide_schedule}")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.m20_profile in ('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
        from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_reward_audit import audit_conflicting_rewards
        profile_manifest['resolved_conflicting_reward_audit'] = audit_conflicting_rewards(env_cfg.rewards)
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    runner_class_name = getattr(agent_cfg, "class_name", "OnPolicyRunner")
    is_custom_runner = runner_class_name not in ("OnPolicyRunner", "DistillationRunner")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if hasattr(env_cfg, "teacher_detailed_diagnostics"):
        diagnostics_mode = "detailed" if env_cfg.teacher_detailed_diagnostics else "core"
        print(f"[INFO] M20 teacher diagnostics={diagnostics_mode}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.m20_profile in ASCENT_PROFILES:
        from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_entry_mdp import install_physics_observer
        install_physics_observer(env.unwrapped)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    _algo_class_name = getattr(getattr(agent_cfg, "algorithm", None), "class_name", None)
    if agent_cfg.resume or _algo_class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # convert config to dict and create runner
    train_cfg = agent_cfg.to_dict()
    if version.parse(installed_version) >= version.parse("5.0.0"):
        train_cfg = cli_args.convert_rsl_rl_cfg_dict(train_cfg)
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles.v1_stop_profile import PROFILES, uses_stop_metadata
    if uses_stop_metadata(args_cli.m20_profile):
        train_cfg["actor"].update(class_name="rl_training.models.stop_aware_actor:StopAwareMLPModel",
                                  stop_std_initial=0.5, stop_std_final=0.1, stop_std_anneal_updates=500)
        if args_cli.m20_profile == "v1_stop_v4":
            # Return to v1's full exploration rather than collapsing the stop
            # distribution; isolate this profile's support/teacher correction.
            train_cfg["actor"].update(stop_std_initial=1.0, stop_std_final=1.0)
        train_cfg["obs_groups"].update(actor=["policy"], critic=["critic"])
        if args_cli.m20_profile == 'v1_stop_v5':
            for key in ('stop_std_initial', 'stop_std_final', 'stop_std_anneal_updates'):
                train_cfg['actor'].pop(key, None)
            train_cfg['actor'].update(class_name='rl_training.models.stop_branch:StopBranchMLPModel',
                                      stop_std_factor=.1, stop_hidden_dims=[128, 64])
            train_cfg['obs_groups'].update(actor=['policy', 'stop_control'], critic=['critic', 'stop_control'])
    if args_cli.m20_profile in ('idle_stop_motion_v1', *ASCENT_PROFILES):
        train_cfg['actor'].update(class_name='rl_training.models.motion_residual_actor:MotionResidualMLPModel',
                                  residual_hidden_dims=[128, 64])
        train_cfg['obs_groups'].update(actor=['policy', 'motion_control'],
                                       critic=['critic', 'motion_control'])
        if args_cli.m20_profile in ASCENT_PROFILES:
            train_cfg['actor'].update(class_name='rl_training.models.ascent_entry_actor:AscentEntryMLPModel',
                                      entry_hidden_dims=[128,64])
            if args_cli.m20_profile in ('ascent_entry_v1_stage3', 'ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
                train_cfg['actor'].update(class_name='rl_training.models.ascent_landing_actor:AscentLandingMLPModel',
                                          landing_hidden_dims=[128, 64])
            if args_cli.m20_profile in ('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
                train_cfg['actor'].update(class_name='rl_training.models.ascent_sequence_actor:AscentSequenceMLPModel',
                                          sequence_hidden_dims=[128, 64])
    if args_cli.m20_profile == 'ascent_entry_v1_stage5':
        train_cfg['actor'].update(class_name='rl_training.models.ascent_adaptive_actor:AscentAdaptiveMLPModel',
                                  adaptive_exploration=profile_manifest['adaptive_exploration_action_std'])
        train_cfg['obs_groups']['critic'] = ['critic', 'motion_control', 'ascent_control']
    effective_actor_config = copy.deepcopy(train_cfg.get("actor"))
    runner_class = resolve_callable(runner_class_name) if is_custom_runner else OnPolicyRunner
    runner = runner_class(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    if args_cli.m20_profile == 'ascent_entry_v1_stage5':
        from rl_training.algorithms.initialization.ascent_adaptive_init import attach_curriculum_checkpoint
        attach_curriculum_checkpoint(runner.alg, env.unwrapped)
        from rl_training.algorithms.initialization.ascent_adaptive_demo import attach_demonstrations
        attach_demonstrations(runner.alg, args_cli.ascent_demo)
    use_compact_console_logging(
        runner.logger, fast=args_cli.m20_profile in ('v1_stop_v5', 'idle_stop_motion_v1', *ASCENT_PROFILES),
        descent_guide_schedule=descent_guide_schedule)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        prior = torch.load(resume_path, map_location="cpu", weights_only=True).get("m20_training_manifest") or {}
        if prior.get("m20_profile") in PROFILES and not profile_manifest:
            raise ValueError("This checkpoint requires its matching --m20_profile when resuming")
        if profile_manifest:
            if prior.get("refinement_version") != profile_manifest["version"]:
                raise ValueError("v1_stop resume requires the same profile; use actor-only v1/model5100 initialization")
        # load previously trained model
        runner.load(resume_path)
        if args_cli.m20_profile in ('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
            env.unwrapped._ascent_sequence_step_offset = int(runner.current_learning_iteration)*agent_cfg.num_steps_per_env
    if descent_guide_schedule:
        env.unwrapped._m20_descent_entry_guide_schedule = {
            **descent_guide_schedule,
            "steps_per_iteration": int(agent_cfg.num_steps_per_env),
            "iteration_offset": int(getattr(runner, "current_learning_iteration", 0)),
        }
    if args_cli.init_actor_from:
        init_path = Path(args_cli.init_actor_from).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"Actor initialization checkpoint does not exist: {init_path}")
        saved = torch.load(init_path, map_location="cpu", weights_only=True)
        actor_state = saved.get("actor_state_dict")
        if not isinstance(actor_state, dict):
            raise ValueError(f"Checkpoint has no actor_state_dict: {init_path}")
        if args_cli.m20_profile == 'v1_stop_v5':
            runner.alg.actor.load_motion_state(actor_state)
        elif args_cli.m20_profile == 'ascent_entry_v1_stage5':
            from rl_training.algorithms.initialization.ascent_adaptive_init import load_initialized_actor
            load_initialized_actor(runner.alg.actor, saved)
        elif args_cli.m20_profile == 'ascent_entry_v1_stage4':
            runner.alg.actor.load_stage3_checkpoint(actor_state)
        elif args_cli.m20_profile == 'ascent_entry_v1_stage3':
            runner.alg.actor.load_stage2_checkpoint(actor_state)
        elif args_cli.m20_profile == 'ascent_entry_v1_stage2':
            runner.alg.actor.load_state_dict(actor_state, strict=True)
            if not all(torch.equal(value, runner.alg.actor.state_dict()[key].detach().cpu())
                       for key, value in actor_state.items()):
                raise RuntimeError('Stage2 full actor warm start changed a source weight')
        elif args_cli.m20_profile == 'ascent_entry_v1':
            runner.alg.actor.load_motion_checkpoint(actor_state)
        elif args_cli.m20_profile == 'idle_stop_motion_v1':
            runner.alg.actor.load_baseline_state(actor_state)
        else:
            runner.alg.actor.load_state_dict(actor_state, strict=True)
        print(f"[INFO]: Initialized actor only from: {init_path}")

    if args_cli.init_critic_from:
        init_path = Path(args_cli.init_critic_from).expanduser().resolve()
        saved = torch.load(init_path, map_location="cpu", weights_only=True)
        if args_cli.m20_profile == 'ascent_entry_v1_stage5':
            from rl_training.algorithms.initialization.ascent_adaptive_init import load_initialized_critic
            load_initialized_critic(runner.alg.critic, saved['critic_state_dict'])
        else:
            runner.alg.critic.load_state_dict(saved["critic_state_dict"], strict=True)
        print(f"[INFO]: Initialized critic weights (fresh optimizer) from: {init_path}")
    reference_path = args_cli.preserve_actor_from or args_cli.init_actor_from
    if args_cli.m20_profile in ('v1_stop_v5', 'idle_stop_motion_v1'):
        branch = 'stop' if args_cli.m20_profile == 'v1_stop_v5' else 'commanded-motion residual'
        print(f'[INFO]: baseline MLP and sampling distribution are frozen; PPO updates only the {branch} and critic')
    elif hasattr(runner.alg, "set_reference") and reference_path:
        reference = torch.load(Path(reference_path).expanduser().resolve(), map_location="cpu", weights_only=True)
        if args_cli.m20_profile in ASCENT_PROFILES:
            # Complete inherited policy plus the zero-initialized new branch;
            # a plain motion checkpoint has no entry_mlp keys.
            runner.alg.set_reference(runner.alg.actor.state_dict())
        else:
            runner.alg.set_reference(reference["actor_state_dict"])
        print(f"[INFO]: Frozen task preservation actor from: {reference_path}")
    elif args_cli.preserve_actor_from:
        raise ValueError("--preserve_actor_from requires the StairGuardedPPO algorithm")
    if args_cli.m20_profile == 'ascent_entry_v1_stage5':
        from rl_training.algorithms.initialization.ascent_adaptive_demo import validate_demo_authority
        demo_authority = validate_demo_authority(runner.alg)
    stop_teacher_path = None
    if args_cli.m20_profile == "v1_stop_v4" and not agent_cfg.resume:
        stop_teacher_path = Path(args_cli.stop_teacher_from or args_cli.init_actor_from).expanduser().resolve()
        teacher_checkpoint = torch.load(stop_teacher_path, map_location="cpu", weights_only=True)
        runner.alg.set_stop_teacher(teacher_checkpoint["actor_state_dict"])
        print(f"[INFO]: Frozen quiet-leg stop teacher from: {stop_teacher_path}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # Record the actual imported tuning and file versions at launch. Editing
    # the checkout during an existing training process does not update it.
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_teacher
    source_root = _LOCAL_SOURCE / "rl_training"
    tracked = [source_root / "tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py",
               source_root / "tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py",
               source_root / "tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/stair_teacher_ppo_cfg.py",
               source_root / "algorithms/stair_guarded_ppo.py", source_root / "utils/metric_aggregation.py",
               source_root / "utils/teacher_diagnostics.py",
               source_root / "utils/module_layout.py", source_root / "utils/legacy_imports.py",
               source_root / "tasks/manager_based/locomotion/velocity/mdp/descent_front_pair.py"]
    if profile_manifest:
        tracked.extend((source_root / "tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/v1_stop_profile.py", source_root / "tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/data/v1_stop_baseline.json"))
        if uses_stop_metadata(args_cli.m20_profile):
            tracked.append(source_root / "models/stop_aware_actor.py")
        if args_cli.m20_profile in ("v1_stop_v3", "v1_stop_v3_target_control", "v1_stop_v4", "v1_stop_v5"):
            tracked.append(source_root / "tasks/manager_based/locomotion/velocity/mdp/stop/stop_coverage.py")
        if args_cli.m20_profile == 'v1_stop_v5':
            tracked.append(source_root/'models/stop_branch.py')
            tracked.append(source_root/'utils/fast_logging.py')
        if args_cli.m20_profile == 'idle_stop_motion_v1':
            tracked.extend((source_root/'models/motion_residual_actor.py', source_root/'utils/fast_logging.py'))
        if args_cli.m20_profile in ASCENT_PROFILES:
            tracked.extend(source_root/name for name in ('models/ascent_entry_actor.py','tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_entry_mdp.py',
                           'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_entry_profile.py','tasks/manager_based/locomotion/velocity/envs/ascent_entry_env.py','tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/data/ascent_reference.json','utils/fast_logging.py',
                           'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_teacher_v1_env_cfg.py',
                           'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/stair_teacher_v1_ppo_cfg.py',
                           'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py'))
            if args_cli.m20_profile in ('ascent_entry_v1_stage2', 'ascent_entry_v1_stage3', 'ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
                tracked.extend(source_root/name for name in ('tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_entry_stage2.py',
                    'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_entry_stage2_profile.py', 'algorithms/initialization/ascent_entry_stage2_init.py'))
            if args_cli.m20_profile in ('ascent_entry_v1_stage3', 'ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
                tracked.extend(source_root/name for name in ('models/ascent_landing_actor.py', 'tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_landing_mdp.py',
                    'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_landing_profile.py', 'algorithms/initialization/ascent_landing_init.py'))
            if args_cli.m20_profile in ('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
                tracked.extend(source_root/name for name in ('tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_sequence.py', 'models/ascent_sequence_actor.py',
                    'tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_sequence_mdp.py', 'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_sequence_profile.py', 'algorithms/initialization/ascent_sequence_init.py', 'tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_reward_audit.py'))
    if args_cli.m20_profile == 'ascent_entry_v1_stage5':
        tracked.extend(source_root/name for name in ('tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_adaptive.py', 'tasks/manager_based/locomotion/velocity/mdp/ascent/ascent_adaptive_mdp.py',
            'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles/ascent_adaptive_profile.py', 'models/ascent_adaptive_actor.py', 'algorithms/initialization/ascent_adaptive_init.py', 'algorithms/initialization/ascent_adaptive_demo.py'))
    manifest = {"refinement_version": "m20_descent_hold_v4_20261016",
                "cli_args": vars(args_cli), "stair_tuning": stair_teacher.TUNING,
                "source_sha256": {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in tracked}}
    if profile_manifest:
        manifest.update(refinement_version=profile_manifest["version"],
                        m20_profile=profile_manifest["profile"],
                        control_profile=profile_manifest["control_profile"],
                        baseline_actor_sha256=profile_manifest["baseline_actor_sha256"],
                        baseline_config_sha256=profile_manifest["baseline_config_sha256"])
        manifest["reward_weights"] = {name: term.weight for name, term in vars(env_cfg.rewards).items()
                                      if term is not None and hasattr(term, "weight")}
        manifest["actor_model_config"] = effective_actor_config
        if descent_guide_schedule:
            manifest["descent_entry_guide_schedule"] = descent_guide_schedule
        if args_cli.m20_profile == 'v1_stop_v5':
            manifest['stop_control_version'] = 'm20_stop_branch_24d_20261017'
            manifest['actor_input_groups'] = {'policy': 244, 'stop_control': 24}
            manifest['critic_input_groups'] = {'critic': 285, 'stop_control': 24}
        if args_cli.m20_profile in ('idle_stop_motion_v1', *ASCENT_PROFILES):
            manifest['motion_control_version'] = 'm20_raw_command_gate_20261017'
            manifest['actor_input_groups'] = {'policy': 244, 'motion_control': 3}
            manifest['critic_input_groups'] = {'critic': 285, 'motion_control': 3}
            if args_cli.m20_profile in ASCENT_PROFILES:
                manifest['ascent_entry_profile'] = profile_manifest
                manifest['frozen_full_baseline_sha256'] = profile_manifest['baseline_actor_sha256']
                manifest['new_branch_limits_action_units'] = runner.alg.actor.entry_limits.detach().cpu().tolist()
                if args_cli.m20_profile == 'ascent_entry_v1_stage2':
                    manifest['actor_initialization_mode'] = 'complete_v1_weights_fresh_optimizer_and_iteration'
                    manifest['critic_warmup_iterations'] = agent_cfg.algorithm.critic_warmup_iterations
                elif args_cli.m20_profile in ('ascent_entry_v1_stage3', 'ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
                    manifest['actor_initialization_mode'] = 'frozen_complete_stage2_plus_zero_landing_branch'
                    manifest['critic_warmup_iterations'] = agent_cfg.algorithm.critic_warmup_iterations
                    manifest['landing_limits_action_units'] = runner.alg.actor.landing_limits.detach().cpu().tolist()
                    manifest['frozen_stage2_state_sha256'] = runner.alg.actor.source_state_digest()
                    if args_cli.init_actor_from:
                        manifest['frozen_stage2_source_sha256'] = hashlib.sha256(Path(args_cli.init_actor_from).expanduser().read_bytes()).hexdigest()
                    else:
                        manifest['frozen_stage2_source_sha256'] = prior.get('frozen_stage2_source_sha256')
                    if args_cli.m20_profile in ('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
                        source_manifest = (torch.load(Path(args_cli.init_actor_from).expanduser(), map_location='cpu',
                                                      weights_only=True)['m20_training_manifest']
                                           if args_cli.init_actor_from else prior)
                        manifest['frozen_stage2_source_sha256'] = source_manifest.get('frozen_stage2_source_sha256')
                        manifest['actor_initialization_mode'] = 'complete_stage3_frozen_plus_zero_sequence_branch'
                        manifest['sequence_limits_action_units'] = runner.alg.actor.sequence_limits.detach().cpu().tolist()
                        manifest['frozen_stage3_state_sha256'] = runner.alg.actor.sequence_source_digest()
                        manifest['frozen_stage3_source_sha256'] = (
                            hashlib.sha256(Path(args_cli.init_actor_from).expanduser().read_bytes()).hexdigest()
                            if args_cli.init_actor_from else prior.get('frozen_stage3_source_sha256'))
                        if args_cli.m20_profile == 'ascent_entry_v1_stage5' and source_manifest.get('m20_profile') in ('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
                            manifest['frozen_stage3_source_sha256'] = source_manifest.get('frozen_stage3_source_sha256')
        if args_cli.m20_profile == 'ascent_entry_v1_stage5':
            manifest['actor_initialization_mode'] = 'validated_warm_start_fresh_optimizer_contact_and_geometry_curriculum'
            manifest['adaptive_exploration_action_std'] = profile_manifest['adaptive_exploration_action_std']
            manifest['contact_demonstrations'] = runner.alg._contact_demo_manifest
            manifest['contact_demonstration_schedule'] = dict(coefficient=.5, decay_updates=800)
            manifest['contact_demonstration_authority'] = demo_authority
            manifest['critic_input_groups']['ascent_control'] = 12
        if stop_teacher_path is not None:
            manifest["stop_teacher_sha256"] = hashlib.sha256(stop_teacher_path.read_bytes()).hexdigest()
        elif args_cli.m20_profile == "v1_stop_v4":
            manifest["stop_teacher_sha256"] = prior.get("stop_teacher_sha256")
    if reference_path and args_cli.m20_profile != 'idle_stop_motion_v1':
        manifest["preservation_actor_sha256"] = hashlib.sha256(Path(reference_path).expanduser().read_bytes()).hexdigest()
    elif args_cli.m20_profile in ("v1_stop_v4", "v1_stop_v5", "idle_stop_motion_v1") and agent_cfg.resume:
        manifest["preservation_actor_sha256"] = prior.get("preservation_actor_sha256")
    if hasattr(env_cfg, "teacher_detailed_diagnostics"):
        manifest["teacher_detailed_diagnostics"] = bool(env_cfg.teacher_detailed_diagnostics)
    if args_cli.init_actor_from:
        manifest["initial_actor_sha256"] = hashlib.sha256(Path(args_cli.init_actor_from).expanduser().read_bytes()).hexdigest()
    elif args_cli.m20_profile in ('ascent_entry_v1_stage2', 'ascent_entry_v1_stage3', 'ascent_entry_v1_stage4', 'ascent_entry_v1_stage5') and agent_cfg.resume:
        manifest['initial_actor_sha256'] = prior.get('initial_actor_sha256')
        manifest['preservation_actor_sha256'] = prior.get('preservation_actor_sha256')
    manifest["train_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # JSON primitives remain compatible with torch.load(weights_only=True),
    # even when a checkpoint is copied away from its original params folder.
    manifest = json.loads(json.dumps(manifest, default=str))
    if hasattr(runner.alg, "set_reference"):
        runner.alg.m20_training_manifest = manifest
    Path(log_dir, "params", "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=args_cli.m20_profile != 'ascent_entry_v1_stage5')
    if args_cli.m20_profile in ASCENT_PROFILES:
        initial_path=Path(args_cli.init_actor_from) if args_cli.init_actor_from else None
        current=runner.alg.actor.state_dict()
        initial=(torch.load(initial_path,map_location='cpu',weights_only=True)['actor_state_dict']
                 if initial_path else None)
        source_entry_weight = initial.get('entry_mlp.4.weight') if initial is not None else None
        if args_cli.m20_profile == 'ascent_entry_v1_stage2' and initial is not None:
            initial = {key: value for key, value in initial.items() if not key.startswith('entry_mlp.')}
        if args_cli.m20_profile == 'ascent_entry_v1_stage5' and initial is not None:
            initial = {k: v for k, v in initial.items() if not k.startswith('sequence_mlp.')}
        if initial is None:
            reference=runner.alg.reference_actor.state_dict()
            trained_prefix = ('sequence_mlp.' if args_cli.m20_profile in ('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5') else
                              'landing_mlp.' if args_cli.m20_profile == 'ascent_entry_v1_stage3' else 'entry_mlp.')
            initial={key:value.detach().cpu() for key,value in reference.items()
                     if not key.startswith(trained_prefix)}
        frozen_exact=all(torch.equal(current[key].detach().cpu(),value) for key,value in initial.items())
        if frozen_exact is False or not all(torch.isfinite(v).all() for v in current.values()):
            raise RuntimeError('v1 frozen policy preservation or finite-state audit failed')
        audit={'task':args_cli.task,'iterations':agent_cfg.max_iterations,
               'guard_updates':runner.alg.guard_updates,'frozen_full_baseline_exact':frozen_exact,
               'entry_output_weight_norm':float(runner.alg.actor.entry_mlp[-1].weight.norm()),
               'physics_observer_samples':int(env.unwrapped._sim_step_counter),
               'filtered_support_samples':int(env.unwrapped._ascent_entry_observer.lifetime_support_samples.item()),
               'new_tread_landings':int(env.unwrapped._ascent_entry_observer.lifetime_landings.item()),
               'completed':True,'behavioral_acceptance':'requires paired MuJoCo evaluation'}
        if source_entry_weight is not None:
            audit['entry_output_weight_change_norm'] = float(
                (runner.alg.actor.entry_mlp[-1].weight.detach().cpu()-source_entry_weight).norm())
        if args_cli.m20_profile in ('ascent_entry_v1_stage2', 'ascent_entry_v1_stage3', 'ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
            observer = env.unwrapped._ascent_entry_observer
            audit.update(stage2_ordered_treads=int(observer.lifetime_ordered.item()),
                         stage2_missing_treads=int(observer.lifetime_missing.item()),
                         fresh_optimizer=not agent_cfg.resume)
        if args_cli.m20_profile == 'ascent_entry_v1_stage3':
            audit.update(frozen_complete_stage2_exact=frozen_exact,
                         landing_output_weight_norm=float(runner.alg.actor.landing_mlp[-1].weight.norm()))
        if args_cli.m20_profile in ('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5'):
            observer = env.unwrapped._ascent_entry_observer
            audit.update(frozen_complete_stage3_exact=frozen_exact,
                         sequence_output_weight_norm=float(runner.alg.actor.sequence_mlp[-1].weight.norm()))
        Path(log_dir,'training_complete.json').write_text(json.dumps(audit,indent=2)+'\n')
        print('[INFO] V1_TRAINING_COMPLETE '+json.dumps(audit),flush=True)
        if runner.logger.writer is not None:
            runner.logger.writer.flush()
            runner.logger.writer.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    if args_cli.headless and args_cli.m20_profile in ASCENT_PROFILES:
        # This installed Kit build faults in both close() and post_quit().
        # All checkpoints, audit, writer and env are finalized above. Match
        # immediate headless shutdown without invoking the failing Kit teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    simulation_app.close()
