#!/usr/bin/env python3
"""MuJoCo sim-to-sim deployment of the M20 stair teacher actor.

With --viewer, the robot starts prone: Z stands, C enters the teacher policy,
W/S/A/D/Q/E set velocity commands, H toggles the height scan, X lies down,
R damps, and Esc quits.
Without --viewer, the script runs a fixed-requested-command batch rollout with
the same heading hold used by interactive policy control.
The checkpoint was trained with asymmetric PPO: actor=57D proprioception +
187D ground height scan, critic=actor information plus privileged signals.
Only actor_state_dict is loaded here. This script reproduces the actor input
and the saved action contract; it never reads critic-only observations.

The observation/action constants below come from the checkpoint run's
params/env.yaml (2026-09-29_19-34-19), not from mutable current config.
Heading hold parameters match the current M20 stair teacher TUNING values.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
import select
import sys
import termios
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from keyboard_input import X11WindowKeys
from terrain_loader import PRESETS, STAIR_OVERRIDES, Terrain, attach_terrain, select_terrain


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-09-29_19-34-19/model_39300.pt"
)
JOINT_NAMES = (
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
)
DEFAULT_JOINT_POS = np.array(
    [0.0, -0.3, 0.6, 0.0, -0.3, 0.6, 0.0, 0.3, -0.6, 0.0, 0.3, -0.6, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
ACTION_SCALE = np.array([0.125, 0.25, 0.25] * 4 + [5.0] * 4, dtype=np.float64)
LEG_EFFORT_LIMIT = 76.4
WHEEL_EFFORT_LIMIT = 21.6
WHEEL_ARMATURE = 0.00243216
POLICY_DT = 0.02
PHYSICS_DT = 0.005
SCAN_OFFSET = 0.5
SCAN_HEIGHT = 20.0
WHEEL_RADIUS = 0.09
ACTOR_OBS_DIM = 244
ACTION_DIM = 16
# Match the float32 command tensor and threshold comparison used by training.
COMMAND_THRESHOLD = float(np.float32(0.10))
HEADING_HOLD_FLAT = (0.8, 0.4, 0.25, math.radians(0.5))
HEADING_HOLD_STAIRS = (1.6, 0.7, 0.50, math.radians(0.5))
HEADING_HOLD_FILTER_TAU = 0.04
HEADING_HOLD_EXIT_DEADBAND = math.radians(0.25)
HEADING_HOLD_RATE_ENTER = 0.03
HEADING_HOLD_RATE_EXIT = 0.015
HEADING_HOLD_SLEW_FLAT = 1.5
HEADING_HOLD_SLEW_STAIRS = 3.0
ASCENT_PATH_KP = 0.35
ASCENT_PATH_KD = 0.4
ASCENT_PATH_MAX_LATERAL_SPEED = 0.12
ASCENT_PATH_LATERAL_DEADBAND = 0.03
ASCENT_PATH_SPEED_DEADBAND = 0.03
ASCENT_PATH_LATERAL_SLEW = 0.3
DESCENT_PATH_MAX_LATERAL_SPEED = 0.10
STAIR_STOP_BRAKE_GRACE = 0.4
STAIR_STOP_KP = 0.5
STAIR_STOP_KD = 0.5
STAIR_STOP_POSITION_DEADBAND = 0.02
STAIR_STOP_SPEED_DEADBAND = 0.015
STAIR_STOP_MAX_SPEED = 0.10
STAIR_STOP_SLEW = 0.30
TELEMETRY_VERSION = "m20_descent_hold_v4_20261016"
DEPLOY_SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PRONE_JOINT_POS = np.array(
    [-0.438, -1.16, 2.76, 0.438, -1.16, 2.76,
     -0.438, 1.16, -2.76, 0.438, 1.16, -2.76, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
PRE_STAND_JOINT_POS = np.array(
    [0.0, -0.6, 1.0, 0.0, -0.6, 1.0,
     0.0, 0.6, -1.0, 0.0, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
STAND_PHASE_DURATION = 2.5
TELEOP_KEY_TIMEOUT = 0.5
TELEOP_VELOCITIES = {"w": (0.7, 0.0, 0.0), "s": (-0.7, 0.0, 0.0),
                     "a": (0.0, 0.5, 0.0), "d": (0.0, -0.5, 0.0),
                     "q": (0.0, 0.0, 0.6), "e": (0.0, 0.0, -0.6)}


def scan_grid() -> np.ndarray:
    """Match Isaac Lab GridPatternCfg(size=(1.6, 1.0), resolution=0.1, ordering='xy')."""
    x = np.arange(-0.8, 0.8 + 1.0e-9, 0.1, dtype=np.float64)
    y = np.arange(-0.5, 0.5 + 1.0e-9, 0.1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    if points.shape != (187, 2):
        raise RuntimeError(f"Unexpected height grid shape: {points.shape}")
    return points


def load_actor(checkpoint: Path) -> MLPModel:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain actor_state_dict")
    first = state.get("mlp.0.weight")
    last = state.get("mlp.6.weight")
    if first is None or last is None or tuple(first.shape) != (512, ACTOR_OBS_DIM) or tuple(last.shape) != (ACTION_DIM, 128):
        raise ValueError("Checkpoint actor architecture does not match this 244D -> 16D contract")
    obs = TensorDict({"policy": torch.zeros((1, ACTOR_OBS_DIM), dtype=torch.float32)}, batch_size=[1])
    manifest = saved.get("m20_training_manifest") or {}
    actor_type = MLPModel
    extra = {}
    if manifest.get("m20_profile") == "v1_stop_v5":
        actor_type = stop_branch_module().StopBranchMLPModel
        obs["stop_control"] = torch.zeros(1, 24)
        config = manifest.get("actor_model_config", {})
        extra = dict(stop_std_factor=config.get("stop_std_factor", .1),
                     stop_hidden_dims=config.get("stop_hidden_dims", [128, 64]))
    elif manifest.get('m20_profile') == 'ascent_entry_v1_stage7':
        actor_type = training_module('models.ascent_unified_actor').AscentUnifiedMLPModel
        obs['motion_control'] = torch.zeros(1,3)
        config = manifest.get('actor_model_config', {})
        extra = dict(residual_hidden_dims=[128,64],unified_hidden_dims=config.get('unified_hidden_dims',[256,128]),
            unified_limits=manifest.get('unified_limits_action_units'),
            adaptive_exploration=manifest.get('adaptive_exploration_action_std'))
    elif manifest.get("m20_profile") in ("ascent_entry_v1", "ascent_entry_v1_stage2", "ascent_entry_v1_stage3", "ascent_entry_v1_stage4", "ascent_entry_v1_stage5", "ascent_entry_v1_stage6"):
        actor_type = (ascent_adaptive_module().AscentAdaptiveMLPModel if manifest.get('m20_profile') in ('ascent_entry_v1_stage5', 'ascent_entry_v1_stage6') else
                      ascent_sequence_module().AscentSequenceMLPModel if manifest.get('m20_profile') == 'ascent_entry_v1_stage4' else
                      ascent_landing_module().AscentLandingMLPModel if manifest.get('m20_profile') == 'ascent_entry_v1_stage3'
                      else ascent_entry_module().AscentEntryMLPModel)
        obs["motion_control"] = torch.zeros(1,3)
        config = manifest.get("actor_model_config",{})
        extra = dict(residual_hidden_dims=config.get('residual_hidden_dims',[128,64]),
                     entry_hidden_dims=config.get('entry_hidden_dims',[128,64]),
                     entry_limits=manifest.get('new_branch_limits_action_units'))
        if manifest.get('m20_profile') in ('ascent_entry_v1_stage3', 'ascent_entry_v1_stage4', 'ascent_entry_v1_stage5', 'ascent_entry_v1_stage6'):
            extra.update(landing_hidden_dims=config.get('landing_hidden_dims', [128, 64]),
                         landing_limits=manifest.get('landing_limits_action_units'))
        if manifest.get('m20_profile') in ('ascent_entry_v1_stage4', 'ascent_entry_v1_stage5', 'ascent_entry_v1_stage6'):
            extra.update(sequence_hidden_dims=config.get('sequence_hidden_dims', [128, 64]),
                         sequence_limits=manifest.get('sequence_limits_action_units'))
        if manifest.get('m20_profile') in ('ascent_entry_v1_stage5', 'ascent_entry_v1_stage6'):
            extra['adaptive_exploration'] = config.get('adaptive_exploration', manifest.get('adaptive_exploration_action_std'))
    elif manifest.get("m20_profile") == "idle_stop_motion_v1":
        actor_type = motion_residual_module().MotionResidualMLPModel
        obs["motion_control"] = torch.zeros(1, 3)
        config = manifest.get("actor_model_config", {})
        extra = dict(residual_hidden_dims=config.get("residual_hidden_dims", [128, 64]))
    actor = actor_type(
        obs,
        {"actor": ["policy"]},
        "actor",
        ACTION_DIM,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
        **extra,
    )
    actor.load_state_dict(state, strict=True)
    actor.eval()
    actor.m20_training_manifest = saved.get("m20_training_manifest") or {}
    return actor


def _training_loader():
    """Share implementation packages with training without starting IsaacLab."""
    import importlib.util
    name = 'm20_training_module_loader'
    if name not in sys.modules:
        path = REPO_ROOT / 'source/rl_training/rl_training/utils/module_loader.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def training_module(name):
    return _training_loader().load_module(name)


def training_source(name):
    return _training_loader().source_path(name)


def stop_branch_module():
    return training_module('stop_branch')


def motion_residual_module():
    return training_module('motion_residual_actor')


def ascent_entry_module():
    return training_module('ascent_entry_actor')


def ascent_landing_module():
    return training_module('ascent_landing_actor')


def ascent_sequence_module():
    return training_module('ascent_sequence_actor')


def ascent_adaptive_module():
    return training_module('ascent_adaptive_actor')


def actor_inputs(contract, data, effective_command, previous_action, actor, requested_command=None):
    observation = contract.observe(data, effective_command, previous_action)
    values = {"policy": torch.from_numpy(observation).unsqueeze(0)}
    if getattr(actor, "uses_stop_control", False):
        if requested_command is None:
            raise ValueError("StopBranch requires the raw requested command")
        if getattr(contract, "_stop_reference", None) is None:
            contract._stop_reference = stop_branch_module().StopReference(POLICY_DT)
        tick = (float(data.time), tuple(float(x) for x in requested_command))
        if getattr(contract, "_stop_control_tick", None) != tick:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(contract.model, data, mujoco.mjtObj.mjOBJ_BODY, contract.base_id, velocity, 1)
            rotation = data.xmat[contract.base_id].reshape(3, 3)
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(1, -1)
            force, _, _, _ = wheel_ground_contact_wrenches(contract.model, data, contract.wheel_body_ids,
                                                         data.xpos[contract.base_id])
            support = stop_branch_module().stable_support(
                torch.as_tensor(data.xpos[contract.wheel_body_ids], dtype=torch.float32)[None],
                torch.as_tensor(force, dtype=torch.float32)[None],
                tensor(data.xpos[contract.base_id, :2]), tensor((rotation@velocity[3:])[:2]))
            contract._last_stop_control = contract._stop_reference.update(
                tensor(requested_command), tensor(data.qpos[contract.qpos_ids]),
                tensor(data.xpos[contract.base_id, :2]), torch.tensor([yaw], dtype=torch.float32),
                tensor(velocity[3:5]), support=support, joint_velocity=tensor(data.qvel[contract.qvel_ids]),
                planar_support=torch.tensor([np.ptp(data.xpos[contract.wheel_body_ids, 2]) <= .03]),
                angular_velocity=tensor(velocity[:3]))
            contract._stop_control_tick = tick
        values['stop_control'] = contract._last_stop_control
    if getattr(actor, "uses_motion_control", False):
        if requested_command is None:
            raise ValueError("MotionResidual requires the raw requested command")
        values['motion_control'] = torch.as_tensor(
            np.asarray(requested_command), dtype=torch.float32).reshape(1, 3)
    return TensorDict(values, batch_size=[1])


def use_checkpoint_control_profile(args, actor) -> bool:
    """A stop-only checkpoint retains the v1 motion command convention."""
    manifest = getattr(actor, "m20_training_manifest", {})
    return bool(getattr(args, "legacy_heading_hold", False) or manifest.get("control_profile") == "legacy_v1")


def load_model(model_path: Path, args: argparse.Namespace) -> mujoco.MjModel:
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo M20 model not found: {model_path}")
    spec = mujoco.MjSpec.from_file(str(model_path))
    spec.option.timestep = PHYSICS_DT
    args.terrain_config = attach_terrain(spec, select_terrain(args),
        {name: getattr(args, name, None) for name in STAIR_OVERRIDES})
    model = spec.compile()
    if model.nu != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} M20 motor actuators, got {model.nu}")
    # The training actuator adds wheel-side armature absent from this MJCF.
    for name in JOINT_NAMES[12:]:
        model.dof_armature[model.joint(name).dofadr[0]] = WHEEL_ARMATURE
    return model


class M20Contract:
    """Resolve each Isaac joint by name, independent of MuJoCo XML storage order."""

    def __init__(self, model: mujoco.MjModel, terrain: Terrain):
        self.model = model
        self.terrain = terrain
        self.base_id = model.body("base_link").id
        self.root_joint = model.joint("floating_base")
        if self.root_joint.type != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("M20 root joint must be a free joint")
        self.qpos_ids = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES], dtype=np.int32)
        self.qvel_ids = np.array([model.joint(name).dofadr[0] for name in JOINT_NAMES], dtype=np.int32)
        self.actuator_ids = np.array([model.actuator(name).id for name in JOINT_NAMES], dtype=np.int32)
        self.wheel_actuator_ids = np.array(
            [model.actuator(f"{name}_wheel_joint").id for name in ("fl", "fr", "hl", "hr")],
            dtype=np.int32,
        )
        self.wheel_body_ids = np.array(
            [model.body(name).id for name in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")],
            dtype=np.int32,
        )
        self.hip_body_ids = np.array([model.body(f"{leg}_hipy").id for leg in ("fl", "fr", "hl", "hr")])
        self.front_knee_body_ids = np.array([model.body(f"{leg}_knee").id for leg in ("fl", "fr")])
        if len(set(self.qpos_ids.tolist())) != ACTION_DIM or len(set(self.actuator_ids.tolist())) != ACTION_DIM:
            raise ValueError("Duplicate or missing M20 joint/actuator mapping")
        self.grid = scan_grid()
        self.last_scan_hits = np.empty((len(self.grid), 3), dtype=np.float64)
        self.terrain_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        terrain_geoms = [
            i for i in range(model.ngeom)
            if int(model.geom_group[i]) == 0 and int(model.geom_bodyid[i]) == 0
        ]
        if not terrain_geoms:
            raise ValueError("Model has no static group-0 terrain geometry for the height scan")

    def reset(self, data: mujoco.MjData, prone: bool = False) -> None:
        self._stop_reference = None
        self._stop_control_tick = None
        mujoco.mj_resetData(self.model, data)
        mujoco.mj_forward(self.model, data)
        x, y = self.terrain.spawn_xy
        root_z = (0.2 if prone else 0.58) + self.ground_height(data, x, y)
        yaw = self.terrain.spawn_yaw
        root_qpos = int(self.root_joint.qposadr[0])
        data.qpos[root_qpos : root_qpos + 7] = [x, y, root_z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        data.qpos[self.qpos_ids] = PRONE_JOINT_POS if prone else DEFAULT_JOINT_POS
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, data)

    def ground_height(self, data: mujoco.MjData, x: float, y: float) -> float:
        """Query actual collision terrain, including ramps and native XML geoms."""
        origin = np.array([x, y, self.terrain.ray_top], dtype=np.float64)
        geom_id = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(self.model, data, origin, np.array([0., 0., -1.]),
                                self.terrain_group, True, -1, geom_id)
        if distance < 0 or geom_id[0] < 0 or self.model.geom_bodyid[geom_id[0]] != 0:
            raise RuntimeError(f"No static collision terrain at {(x, y)}")
        return float(origin[2] - distance)

    def crossing_summary(self, data: mujoco.MjData, fell: bool) -> dict:
        """Geometric end-line crossing, not proof of traversing every obstacle."""
        wheels = data.xpos[self.wheel_body_ids]
        crossed = bool(np.all(wheels[:, 0] >= self.terrain.goal_x + WHEEL_RADIUS))
        inside = bool(np.all(np.abs(wheels[:, 1]) <= self.terrain.course_width / 2))
        cleared = crossed and inside and not fell
        return {"crossed_finish": crossed, "inside_course": inside,
                "cleared_terrain": cleared,
                "cleared_stairs": cleared if self.terrain.kind in {"ascent", "descent"} else False,
                "clearance_criterion": "all wheels beyond +X goal, inside width at final frame, no fall; bypass not checked"}

    def height_scan(self, data: mujoco.MjData) -> np.ndarray:
        """Raycast static terrain using Isaac's yaw-aligned grid and sensor pose."""
        rotation = data.xmat[self.base_id].reshape(3, 3)
        base_pos = data.xpos[self.base_id]
        ray_z = max(base_pos[2] + SCAN_HEIGHT, self.terrain.ray_top)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        world_xy = np.empty_like(self.grid)
        world_xy[:, 0] = base_pos[0] + cos_yaw * self.grid[:, 0] - sin_yaw * self.grid[:, 1]
        world_xy[:, 1] = base_pos[1] + sin_yaw * self.grid[:, 0] + cos_yaw * self.grid[:, 1]
        ray_origin = np.array([0.0, 0.0, ray_z], dtype=np.float64)
        ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geom_id = np.zeros(1, dtype=np.int32)
        scan = np.empty(len(self.grid), dtype=np.float32)
        for i, xy in enumerate(world_xy):
            ray_origin[:2] = xy
            distance = mujoco.mj_ray(
                self.model, data, ray_origin, ray_direction, self.terrain_group, True, -1, geom_id
            )
            if distance < 0 or geom_id[0] < 0 or self.model.geom_bodyid[geom_id[0]] != 0:
                raise RuntimeError(f"No static terrain ray hit at scan index {i}, world_xy={xy.tolist()}")
            hit_z = ray_z - distance
            self.last_scan_hits[i] = (xy[0], xy[1], hit_z)
            scan[i] = np.clip(base_pos[2] - hit_z - SCAN_OFFSET, -1.0, 1.0)
        return scan

    def observe(self, data: mujoco.MjData, command: np.ndarray, previous_action: np.ndarray) -> np.ndarray:
        body_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, body_velocity, 1
        )
        rotation = data.xmat[self.base_id].reshape(3, 3)
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        joint_pos = data.qpos[self.qpos_ids] - DEFAULT_JOINT_POS
        joint_pos[12:] = 0.0  # training's joint_pos_rel_without_wheel
        joint_vel = data.qvel[self.qvel_ids]
        obs = np.concatenate(
            (
                np.clip(body_velocity[:3], -100.0, 100.0) * 0.25,
                np.clip(projected_gravity, -100.0, 100.0),
                command,
                np.clip(joint_pos, -100.0, 100.0),
                np.clip(joint_vel, -100.0, 100.0) * 0.05,
                np.clip(previous_action, -100.0, 100.0),
                self.height_scan(data),
            )
        ).astype(np.float32)
        if obs.shape != (ACTOR_OBS_DIM,) or not np.isfinite(obs).all():
            raise RuntimeError(f"Invalid actor observation: shape={obs.shape}, finite={np.isfinite(obs).all()}")
        return obs

    def apply_pose_pd(self, data: mujoco.MjData, target_pos: np.ndarray) -> None:
        """Hold a transition pose with the saved joint limits and wheel damping."""
        joint_pos = data.qpos[self.qpos_ids]
        joint_vel = data.qvel[self.qvel_ids]
        effort = np.zeros(ACTION_DIM, dtype=np.float64)
        effort[:12] = np.clip(
            80.0 * (target_pos[:12] - joint_pos[:12]) - 2.0 * joint_vel[:12],
            -LEG_EFFORT_LIMIT, LEG_EFFORT_LIMIT,
        )
        effort[12:] = np.clip(-0.6 * joint_vel[12:], -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT)
        data.ctrl[self.actuator_ids] = effort

    def apply_damping(self, data: mujoco.MjData) -> None:
        joint_vel = data.qvel[self.qvel_ids]
        effort = np.zeros(ACTION_DIM, dtype=np.float64)
        effort[:12] = np.clip(-2.0 * joint_vel[:12], -LEG_EFFORT_LIMIT, LEG_EFFORT_LIMIT)
        effort[12:] = np.clip(-0.6 * joint_vel[12:], -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT)
        data.ctrl[self.actuator_ids] = effort

    def apply_pd(self, data: mujoco.MjData, action: np.ndarray) -> None:
        action = np.clip(action, -100.0, 100.0)
        joint_pos = data.qpos[self.qpos_ids]
        joint_vel = data.qvel[self.qvel_ids]
        target_pos = DEFAULT_JOINT_POS[:12] + action[:12] * ACTION_SCALE[:12]
        target_vel = action[12:] * ACTION_SCALE[12:]
        effort = np.empty(ACTION_DIM, dtype=np.float64)
        effort[:12] = np.clip(80.0 * (target_pos - joint_pos[:12]) - 2.0 * joint_vel[:12],
                              -LEG_EFFORT_LIMIT, LEG_EFFORT_LIMIT)
        effort[12:] = np.clip(0.6 * (target_vel - joint_vel[12:]),
                              -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT)
        data.ctrl[self.actuator_ids] = effort


def smoothstep(fraction: float) -> float:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return fraction * fraction * (3.0 - 2.0 * fraction)


class TeleopKeyboard:
    """Share MuJoCo window and terminal key presses with the physics loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: deque[str] = deque()
        self._last_motion_press: dict[str, float] = {}
        self._window_keys = X11WindowKeys.open()

    def on_keycode(self, keycode: int) -> None:
        with self._lock:
            if self._window_keys is not None:
                self._window_keys.note_viewer_event()
                # MuJoCo invokes this callback on PRESS, not GLFW_REPEAT/RELEASE.
                # Motion keys therefore come from physical state, not a timeout.
                if 65 <= keycode <= 90 and chr(keycode).lower() in TELEOP_VELOCITIES:
                    return
        if keycode == 256:  # GLFW escape
            self.on_char("escape")
        elif 65 <= keycode <= 90:
            self.on_char(chr(keycode).lower())

    def on_char(self, key: str) -> None:
        key = "escape" if key == "\x1b" else key.lower()
        with self._lock:
            if key in TELEOP_VELOCITIES:
                self._last_motion_press[key] = time.monotonic()
            elif key in {"z", "c", "x", "r", "h", "escape"}:
                self._events.append(key)

    def take_events(self) -> list[str]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def clear_motion(self) -> None:
        with self._lock:
            self._last_motion_press.clear()

    def close(self) -> None:
        with self._lock:
            if self._window_keys is not None:
                self._window_keys.close()
                self._window_keys = None
            self._last_motion_press.clear()

    def velocity_command(self) -> np.ndarray:
        now = time.monotonic()
        with self._lock:
            self._last_motion_press = {
                key: seen for key, seen in self._last_motion_press.items()
                if now - seen <= TELEOP_KEY_TIMEOUT
            }
            pressed = set(self._last_motion_press)
            if self._window_keys is not None:
                pressed.update(self._window_keys.pressed())
        command = np.zeros(3, dtype=np.float32)
        for key in pressed:
            command += TELEOP_VELOCITIES[key]
        return np.clip(command, [-0.7, -0.5, -0.6], [0.7, 0.5, 0.6]).astype(np.float32)


class TerminalInput:
    """Read the same keys from a focused terminal when stdin is a TTY."""

    def __init__(self, keyboard: TeleopKeyboard) -> None:
        self.keyboard = keyboard
        self.fd = 0
        self.saved_attrs = None
        self.saved_flags = None

    def start(self) -> bool:
        if not os.isatty(self.fd):
            return False
        self.saved_attrs = termios.tcgetattr(self.fd)
        self.saved_flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        attrs = termios.tcgetattr(self.fd)
        attrs[3] &= ~(termios.ECHO | termios.ICANON)
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.saved_flags | os.O_NONBLOCK)
        return True

    def poll(self) -> None:
        if self.saved_attrs is None:
            return
        while select.select([self.fd], [], [], 0.0)[0]:
            try:
                keys = os.read(self.fd, 64)
            except BlockingIOError:
                break
            if not keys:
                break
            for key in keys:
                self.keyboard.on_char(chr(key))

    def close(self) -> None:
        if self.saved_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.saved_attrs)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, self.saved_flags)
            self.saved_attrs = None


def launch_viewer(model: mujoco.MjModel, data: mujoco.MjData, key_callback=None):
    from mujoco import viewer as mj_viewer
    existing_threads = set(threading.enumerate())
    viewer = mj_viewer.launch_passive(model, data, key_callback=key_callback)
    viewer_threads = [thread for thread in threading.enumerate() if thread not in existing_threads]
    return viewer, viewer_threads


def close_viewer(viewer, viewer_threads: list[threading.Thread]) -> None:
    if viewer is None:
        return
    viewer.close()
    # MuJoCo 3.10 close() requests exit but does not join its GL thread.
    for thread in viewer_threads:
        thread.join(timeout=5.0)
    if any(thread.is_alive() for thread in viewer_threads):
        raise RuntimeError("MuJoCo viewer thread did not stop after close()")


def draw_height_scan(viewer, contract: M20Contract, visible: bool) -> None:
    """Draw the actor's cached ray hits in the passive viewer's user scene."""
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        if not visible:
            return
        hits = contract.last_scan_hits
        if scene.maxgeom < len(hits):
            raise RuntimeError(f"Viewer supports {scene.maxgeom} user geoms, need {len(hits)}")
        center_z = hits[len(hits) // 2, 2]
        size = np.array([0.018, 0.0, 0.0], dtype=np.float64)
        orientation = np.eye(3, dtype=np.float64).ravel()
        for i, hit in enumerate(hits):
            delta = float(hit[2] - center_z)
            strength = min(abs(delta) / 0.25, 1.0)
            if delta > 0.02:
                rgb = (0.2 + 0.8 * strength, 0.85 - 0.7 * strength, 0.2)
            elif delta < -0.02:
                rgb = (0.2, 0.85 - 0.55 * strength, 0.2 + 0.8 * strength)
            else:
                rgb = (0.2, 0.9, 0.25)
            rgba = np.array([*rgb, 0.9], dtype=np.float32)
            pos = hit.copy()
            pos[2] += 0.025
            mujoco.mjv_initGeom(scene.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                                size, pos, orientation, rgba)
        scene.ngeom = len(hits)


def wheel_ground_contact_wrenches(model: mujoco.MjModel, data: mujoco.MjData,
                                  wheel_body_ids: np.ndarray,
                                  base_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Aggregate wheel-ground force and yaw moment; retain each contact wrench."""
    forces = np.zeros((len(wheel_body_ids), 3), dtype=np.float64)
    contact_counts = np.zeros(len(wheel_body_ids), dtype=np.int32)
    yaw_moments = np.zeros(len(wheel_body_ids), dtype=np.float64)
    details = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1, body2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
        if body1 != 0 and body2 != 0:
            continue
        wheel_body = body2 if body1 == 0 else body1
        match = np.flatnonzero(wheel_body_ids == wheel_body)
        if not match.size:
            continue
        local = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_id, local)
        world_force = contact.frame.reshape(3, 3).T @ local[:3]
        world_contact_torque = contact.frame.reshape(3, 3).T @ local[3:]
        sign = 1.0 if body2 == wheel_body else -1.0
        index = int(match[0])
        force_on_wheel = sign * world_force
        torque_on_wheel = sign * world_contact_torque
        contact_pos = np.asarray(contact.pos, dtype=np.float64).copy()
        relative_pos = contact_pos - base_pos
        yaw_moment = float(np.cross(relative_pos, force_on_wheel)[2] + torque_on_wheel[2])
        forces[index] += force_on_wheel
        contact_counts[index] += 1
        yaw_moments[index] += yaw_moment
        details.append({
            "wheel_index": index,
            "contact_id": int(contact_id),
            "position_world_m": contact_pos.tolist(),
            "position_from_base_m": relative_pos.tolist(),
            "force_on_wheel_world_N": force_on_wheel.tolist(),
            "contact_torque_on_wheel_world_Nm": torque_on_wheel.tolist(),
            "yaw_moment_about_base_Nm": yaw_moment,
        })
    return forces, contact_counts, yaw_moments, details


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def horizontal_heading_rate(rotation: np.ndarray, angular_velocity_body: np.ndarray) -> float:
    """Derivative of atan2(R[1,0], R[0,0]), with a near-vertical fallback."""
    forward = rotation[:, 0]
    horizontal_squared = float(np.dot(forward[:2], forward[:2]))
    if horizontal_squared < 0.25:
        return float(angular_velocity_body[2])
    omega = rotation @ angular_velocity_body
    return float(omega[2] - forward[2] * np.dot(forward[:2], omega[:2]) / horizontal_squared)


class HeadingHoldController:
    """Match the teacher's straight-travel and zero-command yaw correction."""

    def __init__(self, contract: M20Contract, enabled: bool = True, legacy: bool = False):
        self.contract = contract
        self.enabled = enabled
        self.legacy = legacy
        self.reset()

    def reset(self) -> None:
        self.mode = "manual"
        self.target_yaw = 0.0
        self.target_xy = np.zeros(2, dtype=np.float64)
        self.straight_direction = 0.0
        self.filtered_yaw_rate = 0.0
        self.world_heading_rate_active = False
        self.correction = 0.0
        self.lateral_correction = 0.0
        self.path_active = False
        self.stop_correction = np.zeros(2, dtype=np.float64)
        self.stop_start_time = 0.0
        self.stop_position_latched = False
        self.active = False
        self._last_time = None
        self._cache_key = None
        self._cached_result = None
        self._state_before_update = None

    def update(self, data: mujoco.MjData, requested: np.ndarray) -> tuple[np.ndarray, dict]:
        rotation = data.xmat[self.contract.base_id].reshape(3, 3)
        yaw = float(math.atan2(rotation[1, 0], rotation[0, 0]))
        body_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.contract.model, data, mujoco.mjtObj.mjOBJ_BODY,
            self.contract.base_id, body_velocity, 1,
        )
        body_yaw_rate = float(body_velocity[2])
        world_heading_rate = horizontal_heading_rate(rotation, body_velocity[:3])
        command_values = np.asarray(requested, dtype=np.float32)
        straight = abs(float(command_values[0])) > COMMAND_THRESHOLD and abs(float(command_values[1])) <= COMMAND_THRESHOLD \
            and abs(float(command_values[2])) <= COMMAND_THRESHOLD
        stationary = float(np.linalg.norm(command_values[:2])) <= COMMAND_THRESHOLD and abs(float(command_values[2])) <= COMMAND_THRESHOLD
        mode = "straight" if straight else "zero" if stationary else "manual"
        terrain = self.contract.terrain
        on_stairs = False
        if terrain.kind in {"ascent", "descent"}:
            start = terrain.parameters["start_x"][0]
            on_stairs = start - 0.70 <= float(data.xpos[self.contract.base_id, 0]) <= terrain.goal_x + 0.50
        world_heading_rate_active = (
            not self.legacy and mode == "straight" and float(command_values[0]) > COMMAND_THRESHOLD
            and terrain.kind == "ascent" and on_stairs
            and float(np.dot(rotation[:2, 0], rotation[:2, 0])) >= 0.25
        )
        path_active = (not self.legacy and mode == "straight" and float(command_values[0]) > COMMAND_THRESHOLD
                       and terrain.kind in {"ascent", "descent"} and on_stairs
                       and float(np.dot(rotation[:2, 0], rotation[:2, 0])) >= 0.25)
        yaw_rate = world_heading_rate if world_heading_rate_active else body_yaw_rate
        now = float(data.time)
        base_xy = np.asarray(data.xpos[self.contract.base_id, :2], dtype=np.float64)
        world_velocity = rotation @ body_velocity[3:]
        cache_key = (now, yaw, yaw_rate, tuple(float(x) for x in requested),
                     tuple(float(x) for x in base_xy), tuple(float(x) for x in world_velocity))
        if cache_key == self._cache_key:
            effective, info = self._cached_result
            return effective.copy(), dict(info)
        if self._last_time is not None and now < self._last_time:
            self.reset()
        # Re-observing a state must not advance either filter a second time.
        # A changed request at the same simulation time is evaluated from the
        # state preceding that tick, which also makes reset/pose edits safe.
        if self._last_time == now and self._state_before_update is not None:
            (self.mode, self.target_yaw, self.target_xy, self.straight_direction,
             self.filtered_yaw_rate, self.world_heading_rate_active,
             self.correction, self.lateral_correction, self.active, previous_time,
             self.path_active, self.stop_correction, self.stop_start_time,
             self.stop_position_latched) = self._state_before_update
            self.target_xy = self.target_xy.copy()
            self.stop_correction = self.stop_correction.copy()
        else:
            previous_time = self._last_time
            self._state_before_update = (
                self.mode, self.target_yaw, self.target_xy.copy(), self.straight_direction,
                self.filtered_yaw_rate, self.world_heading_rate_active,
                self.correction, self.lateral_correction, self.active, previous_time,
                self.path_active, self.stop_correction.copy(), self.stop_start_time,
                self.stop_position_latched,
            )
        dt = POLICY_DT if previous_time is None else max(0.0, min(now - previous_time, 5 * POLICY_DT))
        entering = mode != self.mode
        direction = float(np.sign(command_values[0])) if straight else 0.0
        reversed_direction = mode == "straight" and self.mode == "straight" and direction != self.straight_direction
        frame_changed = world_heading_rate_active != self.world_heading_rate_active
        path_entering = path_active and not self.path_active
        if frame_changed:
            self.filtered_yaw_rate = yaw_rate
        self.world_heading_rate_active = world_heading_rate_active
        if (entering and mode != "manual") or (reversed_direction and not self.legacy):
            self.target_yaw = yaw
            self.target_xy = base_xy.copy()
            self.filtered_yaw_rate = yaw_rate
            if self.mode == "manual":
                self.correction = 0.0
            self.active = False
        if entering and mode == "zero":
            self.stop_start_time = now
            self.stop_position_latched = False
        if mode != "zero":
            self.stop_position_latched = False
        if mode == "zero" and not self.stop_position_latched and now - self.stop_start_time >= STAIR_STOP_BRAKE_GRACE - 1.e-8:
            self.target_xy = base_xy.copy()
            self.stop_position_latched = True
        self.mode = mode
        self.straight_direction = direction
        effective = requested.copy()
        error = 0.0
        limited = False
        slew_limited = False
        desired_correction = 0.0
        lateral_error = 0.0
        lateral_speed = 0.0
        if mode != "manual":
            kp, kd, limit, deadband = HEADING_HOLD_STAIRS if on_stairs else HEADING_HOLD_FLAT
            error = wrap_angle(self.target_yaw - yaw)
            if self.legacy:
                self.filtered_yaw_rate = yaw_rate
                self.active = abs(error) > deadband or abs(yaw_rate) > HEADING_HOLD_RATE_ENTER
            else:
                alpha = 1.0 - math.exp(-dt / HEADING_HOLD_FILTER_TAU)
                self.filtered_yaw_rate += alpha * (yaw_rate - self.filtered_yaw_rate)
                if self.active:
                    self.active = abs(error) > HEADING_HOLD_EXIT_DEADBAND or abs(self.filtered_yaw_rate) > HEADING_HOLD_RATE_EXIT
                else:
                    self.active = abs(error) > deadband or abs(self.filtered_yaw_rate) > HEADING_HOLD_RATE_ENTER
            if self.active:
                unrestricted = kp * error - kd * self.filtered_yaw_rate
                desired_correction = float(np.clip(unrestricted, -limit, limit))
                limited = abs(unrestricted) > limit
            if self.legacy:
                self.correction = desired_correction
            else:
                slew = HEADING_HOLD_SLEW_STAIRS if on_stairs else HEADING_HOLD_SLEW_FLAT
                delta = desired_correction - self.correction
                slew_limited = abs(delta) > slew * dt
                self.correction += float(np.clip(delta, -slew * dt, slew * dt))
            effective[2] = self.correction
        else:
            self.filtered_yaw_rate = yaw_rate
            self.correction = 0.0
            self.active = False
        if path_active:
            normal = np.array([-math.sin(self.target_yaw), math.cos(self.target_yaw)])
            lateral_error = float(np.dot(base_xy - self.target_xy, normal))
            lateral_speed = float(np.dot(world_velocity[:2], normal))
            position_excess = math.copysign(max(abs(lateral_error) - ASCENT_PATH_LATERAL_DEADBAND, 0.0), lateral_error)
            speed_excess = math.copysign(max(abs(lateral_speed) - ASCENT_PATH_SPEED_DEADBAND, 0.0), lateral_speed)
            lateral_limit = DESCENT_PATH_MAX_LATERAL_SPEED if terrain.kind == "descent" else ASCENT_PATH_MAX_LATERAL_SPEED
            desired_lateral = float(np.clip(
                -ASCENT_PATH_KP * position_excess - ASCENT_PATH_KD * speed_excess,
                -lateral_limit, lateral_limit,
            )) * math.cos(yaw - self.target_yaw)
            if path_entering or entering or reversed_direction:
                self.lateral_correction = 0.0
            self.lateral_correction += float(np.clip(
                desired_lateral - self.lateral_correction,
                -ASCENT_PATH_LATERAL_SLEW * dt, ASCENT_PATH_LATERAL_SLEW * dt,
            ))
            effective[1] = self.lateral_correction
        else:
            self.lateral_correction = 0.0
        self.path_active = path_active
        stop_feedback = not self.legacy and mode == "zero" and on_stairs and self.stop_position_latched
        stop_error = base_xy - self.target_xy
        if stop_feedback:
            position_excess = np.sign(stop_error) * np.maximum(np.abs(stop_error) - STAIR_STOP_POSITION_DEADBAND, 0.)
            velocity_excess = np.sign(world_velocity[:2]) * np.maximum(np.abs(world_velocity[:2]) - STAIR_STOP_SPEED_DEADBAND, 0.)
            desired_world = -STAIR_STOP_KP * position_excess - STAIR_STOP_KD * velocity_excess
            desired_world *= STAIR_STOP_MAX_SPEED / max(np.linalg.norm(desired_world), STAIR_STOP_MAX_SPEED)
            desired_body = np.array([math.cos(yaw) * desired_world[0] + math.sin(yaw) * desired_world[1],
                                     -math.sin(yaw) * desired_world[0] + math.cos(yaw) * desired_world[1]])
            if entering:
                self.stop_correction.fill(0.)
            delta_xy = desired_body - self.stop_correction
            limit_xy = STAIR_STOP_SLEW * dt
            self.stop_correction += delta_xy * limit_xy / max(np.linalg.norm(delta_xy), limit_xy, 1.e-12)
            effective[:2] = self.stop_correction
        else:
            self.stop_correction = np.zeros(2, dtype=np.float64)
        if not self.enabled:
            effective = requested.copy()
        info = {
            "mode": mode if self.enabled else "disabled", "target_yaw_deg": math.degrees(self.target_yaw),
            "error_deg": math.degrees(error), "on_stairs": on_stairs, "limited": limited,
            "active": self.active and self.enabled, "slew_limited": slew_limited and self.enabled,
            "filtered_yaw_rate_rad_s": self.filtered_yaw_rate,
            "body_yaw_rate_rad_s": body_yaw_rate,
            "world_heading_rate_rad_s": world_heading_rate,
            "world_heading_rate_active": self.world_heading_rate_active,
            "path_target_x_world_m": float(self.target_xy[0]),
            "path_target_y_world_m": float(self.target_xy[1]),
            "path_lateral_error_m": lateral_error,
            "path_lateral_speed_m_s": lateral_speed,
            "path_lateral_correction_m_s": self.lateral_correction,
            "path_active": self.path_active and self.enabled,
            "stop_feedback_active": stop_feedback and self.enabled,
            "stop_position_error_x_m": float(stop_error[0]),
            "stop_position_error_y_m": float(stop_error[1]),
            "stop_correction_vx_m_s": float(self.stop_correction[0]),
            "stop_correction_vy_m_s": float(self.stop_correction[1]),
            "desired_correction_rad_s": desired_correction,
            "controller_version": "legacy_v1" if self.legacy else "filtered_v4",
        }
        self._last_time = now
        self._cache_key = cache_key
        self._cached_result = (effective.copy(), dict(info))
        return effective, info


class StairTelemetry:
    """Read-only live stair yaw, wheel contact, and diagonal torque monitor."""

    LEG_NAMES = ("FL", "FR", "HL", "HR")

    def __init__(self, contract: M20Contract):
        self.contract = contract
        self.level_before = None
        self.yaw_at_previous_level = None
        self.yaw_at_policy_start = None
        self.transitions: deque[dict] = deque(maxlen=5)
        self.initial_level = None
        self.descent_drop_count = 0
        self.previous_action = None
        self.previous_action_delta = None
        self.previous_action_time = None
        self._physics_window = self._empty_physics_window()
        self._front_wheels = set(int(x) for x in contract.wheel_body_ids[:2])
        self._rear_wheels = set(int(x) for x in contract.wheel_body_ids[2:])
        self._front_leg_bodies = {
            body for body in range(1, contract.model.nbody)
            if (mujoco.mj_id2name(contract.model, mujoco.mjtObj.mjOBJ_BODY, body) or "").startswith(("fl_", "fr_"))
        } - self._front_wheels

    @staticmethod
    def _empty_physics_window() -> dict:
        return {"physics_substeps": 0, "self_contact_physics_substeps": 0,
                "rear_front_wheel_contact_physics_substeps": 0,
                "rear_wheel_front_leg_contact_physics_substeps": 0,
                "rear_front_peak_contact_force_N": 0.0,
                "rear_front_contact_impulse_Ns": 0.0}

    def observe_physics_step(self, data: mujoco.MjData) -> None:
        """Aggregate solved forces after every physics step, before contacts vanish.

        Only robot self contacts with >1 N resultant force count as a hit.
        Counts describe substeps, rather than distinct collision events.
        """
        window = self._physics_window
        window["physics_substeps"] += 1
        flags = [False, False, False]
        impulse_force = 0.0
        for contact_id, contact in enumerate(data.contact):
            first, second = [int(self.contract.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if first == 0 or second == 0:
                continue
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.contract.model, data, contact_id, wrench)
            force = float(np.linalg.norm(wrench[:3]))
            if force <= 1.0:
                continue
            flags[0] = True
            rear_front_wheel = (first in self._rear_wheels and second in self._front_wheels) or (second in self._rear_wheels and first in self._front_wheels)
            rear_front_leg = (first in self._rear_wheels and second in self._front_leg_bodies) or (second in self._rear_wheels and first in self._front_leg_bodies)
            flags[1] |= rear_front_wheel
            flags[2] |= rear_front_leg
            if rear_front_wheel or rear_front_leg:
                window["rear_front_peak_contact_force_N"] = max(window["rear_front_peak_contact_force_N"], force)
                impulse_force += force
        for key, hit in zip(("self_contact_physics_substeps", "rear_front_wheel_contact_physics_substeps",
                             "rear_wheel_front_leg_contact_physics_substeps"), flags):
            window[key] += int(hit)
        window["rear_front_contact_impulse_Ns"] += impulse_force * float(self.contract.model.opt.timestep)

    def sample(self, data: mujoco.MjData, action: np.ndarray | None = None) -> dict:
        contract = self.contract
        action_stats = {}
        if action is not None:
            action = np.asarray(action, dtype=np.float64)
            gap = float(data.time) - self.previous_action_time if self.previous_action_time is not None else float("inf")
            continuous = 0 < gap <= 1.5 * POLICY_DT
            delta = action - self.previous_action if continuous else np.zeros_like(action)
            second_delta = delta - self.previous_action_delta if continuous else np.zeros_like(action)
            for group, indices in (("leg", slice(0, 12)), ("wheel", slice(12, 16))):
                action_stats[f"{group}_action_delta_rms"] = float(np.sqrt(np.mean(delta[indices]**2)))
                action_stats[f"{group}_action_second_difference_rms"] = float(np.sqrt(np.mean(second_delta[indices]**2)))
                physical_scale = ACTION_SCALE[indices] * (WHEEL_RADIUS if group == "wheel" else 1.0)
                unit = "m_s" if group == "wheel" else "rad"
                action_stats[f"{group}_target_delta_rms_{unit}"] = float(np.sqrt(np.mean((delta[indices] * physical_scale)**2)))
                action_stats[f"{group}_target_second_difference_rms_{unit}"] = float(np.sqrt(np.mean((second_delta[indices] * physical_scale)**2)))
            self.previous_action = action.copy()
            self.previous_action_delta = delta.copy()
            self.previous_action_time = float(data.time)
        physics_window = self._physics_window
        self._physics_window = self._empty_physics_window()
        transition_event = None
        rotation = data.xmat[contract.base_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        if self.yaw_at_policy_start is None:
            self.yaw_at_policy_start = yaw
            self.yaw_at_previous_level = yaw
        wheel_ground = np.asarray([
            contract.ground_height(data, *data.xpos[body_id, :2])
            for body_id in contract.wheel_body_ids
        ])
        step_height = contract.terrain.parameters.get("step_height", [0.15])[0]
        levels = np.rint(wheel_ground / step_height).astype(int)
        level = int(levels.max() if contract.terrain.kind == "ascent" else levels.min())
        if self.initial_level is None:
            self.initial_level = level
        if contract.terrain.kind == "descent":
            self.descent_drop_count = max(self.descent_drop_count, self.initial_level - level)
        if self.level_before is None:
            self.level_before = level
        elif level != self.level_before:
            direction = 1 if contract.terrain.kind == "ascent" else -1
            if (level - self.level_before) * direction > 0:
                yaw_before = float(self.yaw_at_previous_level)
                transition_event = {
                    "from": self.level_before,
                    "to": level,
                    "yaw_before_deg": math.degrees(yaw_before),
                    "yaw_after_deg": math.degrees(yaw),
                    "delta_yaw_deg": math.degrees(wrap_angle(yaw - yaw_before)),
                }
                self.transitions.append(transition_event)
                self.yaw_at_previous_level = yaw
            self.level_before = level
        angular_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(contract.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                 contract.base_id, angular_velocity, 1)
        base_pos = data.xpos[contract.base_id].copy()
        forces, contact_counts, contact_yaw_moments, contact_details = wheel_ground_contact_wrenches(
            contract.model, data, contract.wheel_body_ids, base_pos
        )
        torques = data.actuator_force[contract.actuator_ids]
        leg_torques = torques[:12].reshape(4, 3)
        wheel_torques = data.actuator_force[contract.wheel_actuator_ids].copy()
        rotation = data.xmat[contract.base_id].reshape(3, 3)
        wheel_pos = data.xpos[contract.wheel_body_ids]
        hip_delta_body = (wheel_pos - data.xpos[contract.hip_body_ids]) @ rotation
        knees = data.xpos[contract.front_knee_body_ids]
        segment = wheel_pos[:2] - knees
        projection = np.sum((wheel_pos[2:] - knees) * segment, axis=1) / np.maximum(np.sum(segment**2, axis=1), 1.0e-9)
        nearest = knees + np.clip(projection, 0, 1)[:, None] * segment
        world_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(contract.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                contract.base_id, world_velocity, 0)
        self_contacts = wheel_pair_contacts = rear_front_leg_contacts = nonwheel_ground = 0
        front_wheels, rear_wheels = set(contract.wheel_body_ids[:2]), set(contract.wheel_body_ids[2:])
        for contact in data.contact:
            bodies = [int(contract.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if 0 in bodies:
                nonwheel_ground += int(next((b for b in bodies if b != 0), 0) not in set(contract.wheel_body_ids) | {0})
                continue
            self_contacts += 1
            wheel_pair_contacts += int(any(b in front_wheels for b in bodies) and any(b in rear_wheels for b in bodies))
            for rear, front in (bodies, bodies[::-1]):
                if rear in rear_wheels:
                    name = mujoco.mj_id2name(contract.model, mujoco.mjtObj.mjOBJ_BODY, front) or ""
                    rear_front_leg_contacts += int(name.startswith(("fl_", "fr_")) and front not in front_wheels)
        lateral_margin = contract.terrain.course_width / 2 - float(np.max(np.abs(wheel_pos[:, 1]))) - WHEEL_RADIUS
        stop_control = getattr(contract, '_last_stop_control', None)
        stop_info = {} if stop_control is None else {
            'stop_branch_active': bool(stop_control[0, 0] > .5),
            'stop_branch_wheel_hold_blend': float(stop_control[0, 1]),
            'stop_branch_leg_hold_blend': float(stop_control[0, 23]),
            'stop_branch_position_error_x_m': float(stop_control[0, 6]),
            'stop_branch_position_error_y_m': float(stop_control[0, 7]),
            'stop_branch_control_observation': stop_control[0].numpy().copy(),
        }
        return {
            **stop_info,
            "yaw_deg": math.degrees(yaw),
            "yaw_from_policy_start_deg": math.degrees(wrap_angle(yaw - self.yaw_at_policy_start)),
            "yaw_rate_rad_s": float(angular_velocity[2]),
            "body_yaw_rate_rad_s": float(angular_velocity[2]),
            "world_heading_rate_rad_s": horizontal_heading_rate(rotation, angular_velocity[:3]),
            "roll_deg": math.degrees(math.atan2(rotation[2, 1], rotation[2, 2])),
            "pitch_deg": math.degrees(math.atan2(-rotation[2, 0], np.linalg.norm(rotation[2, 1:]))),
            "level": level,
            "wheel_levels": levels,
            "wheel_ground_m": wheel_ground,
            "wheel_forces": forces,
            "wheel_contact_counts": contact_counts,
            "wheel_contact_yaw_moments": contact_yaw_moments,
            "wheel_contact_details": contact_details,
            "wheel_motor_torques": wheel_torques,
            "leg_torques": leg_torques,
            "joint_positions": data.qpos[contract.qpos_ids].copy(),
            "joint_velocities": data.qvel[contract.qvel_ids].copy(),
            "policy_action": None if action is None else action.copy(),
            **action_stats,
            **physics_window,
            "wheel_linear_speeds": -data.qvel[contract.qvel_ids[12:]].copy() * WHEEL_RADIUS,
            "wheel_positions_world_m": wheel_pos.copy(),
            "base_speed_m_s": float(np.linalg.norm(world_velocity[3:5])),
            "base_velocity_world_m_s": world_velocity[3:].copy(),
            "hip_wheel_forward": hip_delta_body[:, 0],
            "hip_wheel_length": np.linalg.norm(hip_delta_body, axis=1),
            "wheel_pair_distances": np.linalg.norm(wheel_pos[:2] - wheel_pos[2:], axis=1),
            "rear_wheel_front_shank_distances": np.linalg.norm(wheel_pos[2:] - nearest, axis=1),
            "lateral_edge_margin_m": lateral_margin,
            "near_lateral_edge": lateral_margin < 0.10,
            "nonwheel_ground_contacts": nonwheel_ground,
            "self_contacts": self_contacts,
            "rear_front_wheel_contacts": wheel_pair_contacts,
            "rear_wheel_front_leg_contacts": rear_front_leg_contacts,
            "descent_drop_count": self.descent_drop_count,
            "transition_event": transition_event,
            "recent_transitions": list(self.transitions),
        }


def render_stair_telemetry(sample: dict, data: mujoco.MjData, contract: M20Contract,
                           command: np.ndarray, mode: str) -> None:
    """Render a live terminal dashboard next to the MuJoCo viewer window."""
    level_label = "上升层级" if contract.terrain.kind == "ascent" else "下降层级"
    yaw = sample["yaw_deg"]
    baseline_delta = sample["yaw_from_policy_start_deg"]
    angular_speed = sample["yaw_rate_rad_s"]
    lines = [
        "M20 MuJoCo 实时监视 | 正偏航=向左，负偏航=向右",
        f"模式={mode}  t={data.time:7.2f}s  策略cmd=[{command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}]  原始角速度={sample['requested_command'][2]:+.2f}",
        f"航向保持={sample['heading_hold']['mode']}  目标={sample['heading_hold']['target_yaw_deg']:+.2f}°  修正前误差={sample['heading_hold']['error_deg']:+.2f}°",
        f"偏航={yaw:+7.2f}°  相对进入策略时={baseline_delta:+7.2f}°  yaw_rate={angular_speed:+.3f} rad/s  {level_label}=L{sample['level']}",
        "四轮接触力 [Fx, Fy, Fz] N（负值按世界坐标方向显示）:",
    ]
    for name, force, level in zip(StairTelemetry.LEG_NAMES, sample["wheel_forces"], sample["wheel_levels"]):
        norm = float(np.linalg.norm(force))
        lines.append(f"  {name} L{level}: [{force[0]:+7.1f}, {force[1]:+7.1f}, {force[2]:+7.1f}]  |F|={norm:7.1f}")
    wheel_torques = sample["wheel_motor_torques"]
    contact_moments = sample["wheel_contact_yaw_moments"]
    lines.append("轮电机力矩 / 轮地接触偏航力矩（相对机身基座，+为左转）:")
    for index, name in enumerate(StairTelemetry.LEG_NAMES):
        lines.append(
            f"  {name}: motor={wheel_torques[index]:+7.2f} Nm  "
            f"ground_Mz={contact_moments[index]:+8.2f} Nm  "
            f"contacts={sample['wheel_contact_counts'][index]}"
        )
    lines.append(f"  四轮地面接触偏航力矩合计: {contact_moments.sum():+8.2f} Nm")
    lines.append("对角组腿部关节力矩 [hipx, hipy, knee] Nm:")
    for group_name, leg_indices in (("FR+HL", (1, 2)), ("FL+HR", (0, 3))):
        parts = []
        group_values = []
        for leg_index in leg_indices:
            values = sample["leg_torques"][leg_index]
            group_values.extend(values.tolist())
            parts.append(f"{StairTelemetry.LEG_NAMES[leg_index]}=[{values[0]:+6.1f},{values[1]:+6.1f},{values[2]:+6.1f}]")
        lines.append(f"  {group_name}: " + "  ".join(parts) + f"  sum|τ|={np.abs(group_values).sum():.1f}")
    lines.append("最近台阶偏航变化（前一级进入时 → 当前级进入时）:")
    if sample["recent_transitions"]:
        for transition in sample["recent_transitions"]:
            lines.append(
                f"  L{transition['from']}→L{transition['to']}: "
                f"{transition['yaw_before_deg']:+.2f}° → {transition['yaw_after_deg']:+.2f}° "
                f"(Δ {transition['delta_yaw_deg']:+.2f}°)"
            )
    else:
        lines.append("  尚未检测到新的踏阶层级")
    if sys.stdout.isatty():
        print("\033[2J\033[H" + "\n".join(lines), end="\n", flush=True)
    else:
        print("\n".join(lines), flush=True)


def export_stair_telemetry(samples: list[dict], args: argparse.Namespace) -> tuple[Path, Path, Path] | None:
    """Write per-step summary, per-contact detail, and a concise Markdown report."""
    if not samples:
        return None
    if args.output is not None:
        stem = args.output.with_suffix("")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(".scratch") / f"mujoco_stair_{stamp}"
    csv_path = stem.with_name(stem.name + "_telemetry.csv")
    md_path = stem.with_name(stem.name + "_telemetry.md")
    contacts_path = stem.with_name(stem.name + "_contacts.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    leg_names = ("fl", "fr", "hl", "hr")
    wheel_names = ("FL", "FR", "HL", "HR")
    rows = []
    contact_rows = []
    transitions = []
    for sample_index, sample in enumerate(samples):
        leg_torques = np.asarray(sample["leg_torques"], dtype=float)
        wheel_motor_torques = np.asarray(sample["wheel_motor_torques"], dtype=float)
        wheel_contact_yaw_moments = np.asarray(sample["wheel_contact_yaw_moments"], dtype=float)
        wheel_contact_counts = np.asarray(sample["wheel_contact_counts"], dtype=int)
        row = {
            "sample_index": sample_index,
            "policy_step": sample["policy_step"],
            "time_s": sample["time_s"],
            "command_vx_m_s": sample["command"][0],
            "command_vy_m_s": sample["command"][1],
            "command_yaw_rad_s": sample["command"][2],
            "effective_command_vx_m_s": sample["effective_command"][0],
            "effective_command_vy_m_s": sample["effective_command"][1],
            "effective_command_yaw_rad_s": sample["effective_command"][2],
            "heading_target_deg": sample["heading_hold"]["target_yaw_deg"],
            "heading_error_at_command_deg": sample["heading_hold"]["error_deg"],
            "heading_hold_mode": sample["heading_hold"]["mode"],
            "heading_hold_on_stairs": sample["heading_hold"]["on_stairs"],
            "heading_hold_limited": sample["heading_hold"]["limited"],
            "heading_hold_active": sample["heading_hold"].get("active", False),
            "heading_hold_slew_limited": sample["heading_hold"].get("slew_limited", False),
            "heading_filtered_yaw_rate_rad_s": sample["heading_hold"].get("filtered_yaw_rate_rad_s", float("nan")),
            "heading_body_yaw_rate_at_command_rad_s": sample["heading_hold"].get("body_yaw_rate_rad_s", float("nan")),
            "heading_world_heading_rate_at_command_rad_s": sample["heading_hold"].get("world_heading_rate_rad_s", float("nan")),
            "heading_world_heading_rate_active": sample["heading_hold"].get("world_heading_rate_active", False),
            "path_target_x_world_m": sample["heading_hold"].get("path_target_x_world_m", float("nan")),
            "path_target_y_world_m": sample["heading_hold"].get("path_target_y_world_m", float("nan")),
            "path_lateral_error_m": sample["heading_hold"].get("path_lateral_error_m", float("nan")),
            "path_lateral_speed_m_s": sample["heading_hold"].get("path_lateral_speed_m_s", float("nan")),
            "path_lateral_correction_m_s": sample["heading_hold"].get("path_lateral_correction_m_s", float("nan")),
            "path_feedback_active": sample["heading_hold"].get("path_active", False),
            "stop_feedback_active": sample["heading_hold"].get("stop_feedback_active", False),
            "stop_position_error_x_m": sample["heading_hold"].get("stop_position_error_x_m", float("nan")),
            "stop_position_error_y_m": sample["heading_hold"].get("stop_position_error_y_m", float("nan")),
            "stop_correction_vx_m_s": sample["heading_hold"].get("stop_correction_vx_m_s", 0.0),
            "stop_correction_vy_m_s": sample["heading_hold"].get("stop_correction_vy_m_s", 0.0),
            "heading_desired_correction_rad_s": sample["heading_hold"].get("desired_correction_rad_s", float("nan")),
            "heading_controller_version": sample["heading_hold"].get("controller_version", "unknown"),
            "yaw_deg": sample["yaw_deg"],
            "yaw_from_policy_start_deg": sample["yaw_from_policy_start_deg"],
            "yaw_rate_rad_s": sample["yaw_rate_rad_s"],
            "body_yaw_rate_rad_s": sample["body_yaw_rate_rad_s"],
            "world_heading_rate_rad_s": sample["world_heading_rate_rad_s"],
            "roll_deg": sample["roll_deg"],
            "pitch_deg": sample["pitch_deg"],
            "stair_level": sample["level"],
            "base_x_m": sample["base_xyz"][0],
            "base_y_m": sample["base_xyz"][1],
            "base_z_m": sample["base_xyz"][2],
            "telemetry_version": TELEMETRY_VERSION,
            "stop_branch_active": sample.get('stop_branch_active', False),
            "stop_branch_wheel_hold_blend": sample.get('stop_branch_wheel_hold_blend', 0.),
            "stop_branch_leg_hold_blend": sample.get('stop_branch_leg_hold_blend', 0.),
            "stop_branch_position_error_x_m": sample.get('stop_branch_position_error_x_m', 0.),
            "stop_branch_position_error_y_m": sample.get('stop_branch_position_error_y_m', 0.),
            "base_speed_m_s": sample["base_speed_m_s"],
            "base_velocity_x_world_m_s": sample["base_velocity_world_m_s"][0],
            "base_velocity_y_world_m_s": sample["base_velocity_world_m_s"][1],
            "base_velocity_z_world_m_s": sample["base_velocity_world_m_s"][2],
            "wheel_speed_max_m_s": float(np.abs(sample["wheel_linear_speeds"]).max()),
            "lateral_edge_margin_m": sample["lateral_edge_margin_m"],
            "near_lateral_edge": sample["near_lateral_edge"],
            "nonwheel_ground_contacts": sample["nonwheel_ground_contacts"],
            "self_contacts": sample["self_contacts"],
            "rear_front_wheel_contacts": sample["rear_front_wheel_contacts"],
            "rear_wheel_front_leg_contacts": sample["rear_wheel_front_leg_contacts"],
            "descent_drop_count": sample["descent_drop_count"],
        }
        for key in StairTelemetry._empty_physics_window():
            row[key] = sample.get(key, 0)
        for group in ("leg", "wheel"):
            for statistic in ("action_delta_rms", "action_second_difference_rms"):
                key = f"{group}_{statistic}"
                row[key] = sample.get(key, float("nan"))
            for statistic in ("delta", "second_difference"):
                key = f"{group}_target_{statistic}_rms_{'m_s' if group == 'wheel' else 'rad'}"
                row[key] = sample.get(key, float("nan"))
        for index, wheel in enumerate(wheel_names):
            row[f"{wheel}_level"] = int(sample["wheel_levels"][index])
            row[f"{wheel}_ground_m"] = float(sample["wheel_ground_m"][index])
            for axis, coordinate in zip("xyz", sample["wheel_positions_world_m"][index]):
                row[f"{wheel}_center_{axis}_world_m"] = float(coordinate)
            force = sample["wheel_forces"][index]
            row[f"{wheel}_force_x_N"] = float(force[0])
            row[f"{wheel}_force_y_N"] = float(force[1])
            row[f"{wheel}_force_z_N"] = float(force[2])
            row[f"{wheel}_force_norm_N"] = float(np.linalg.norm(force))
            row[f"{wheel}_motor_torque_Nm"] = float(wheel_motor_torques[index])
            row[f"{wheel}_ground_contact_count"] = int(wheel_contact_counts[index])
            row[f"{wheel}_ground_yaw_moment_about_base_Nm"] = float(wheel_contact_yaw_moments[index])
            row[f"{wheel}_wheel_linear_speed_m_s"] = float(sample["wheel_linear_speeds"][index])
            row[f"{wheel}_hip_wheel_forward_m"] = float(sample["hip_wheel_forward"][index])
            row[f"{wheel}_hip_wheel_length_m"] = float(sample["hip_wheel_length"][index])
            policy_action = sample.get("policy_action")
            row[f"{wheel}_target_wheel_speed_rad_s"] = float(policy_action[12 + index] * ACTION_SCALE[12 + index]) if policy_action is not None else float("nan")
        for index, side in enumerate(("left", "right")):
            row[f"{side}_wheel_center_distance_m"] = float(sample["wheel_pair_distances"][index])
            row[f"{side}_rear_wheel_front_shank_distance_m"] = float(sample["rear_wheel_front_shank_distances"][index])
        for index, leg in enumerate(leg_names):
            for joint_index, joint in enumerate(("hipx", "hipy", "knee")):
                row[f"{leg}_{joint}_torque_Nm"] = float(leg_torques[index, joint_index])
                row[f"{leg}_{joint}_angle_rad"] = float(sample["joint_positions"][3 * index + joint_index])
                row[f"{leg}_{joint}_velocity_rad_s"] = float(sample["joint_velocities"][3 * index + joint_index])
        for group, indices in (("fr_hl", (1, 2)), ("fl_hr", (0, 3))):
            row[f"{group}_sum_abs_torque_Nm"] = float(
                np.abs(leg_torques[list(indices)]).sum()
            )
        event = sample.get("transition_event")
        row["transition_from_level"] = event["from"] if event else ""
        row["transition_to_level"] = event["to"] if event else ""
        row["transition_yaw_before_deg"] = event["yaw_before_deg"] if event else ""
        row["transition_yaw_after_deg"] = event["yaw_after_deg"] if event else ""
        row["transition_delta_yaw_deg"] = event["delta_yaw_deg"] if event else ""
        if event:
            transitions.append({"time_s": sample["time_s"], **event})
        for detail in sample["wheel_contact_details"]:
            wheel = wheel_names[detail["wheel_index"]]
            row_detail = {
                "sample_index": sample_index,
                "policy_step": sample["policy_step"],
                "time_s": sample["time_s"],
                "wheel": wheel,
                "contact_id": detail["contact_id"],
                "contact_x_world_m": detail["position_world_m"][0],
                "contact_y_world_m": detail["position_world_m"][1],
                "contact_z_world_m": detail["position_world_m"][2],
                "contact_x_from_base_m": detail["position_from_base_m"][0],
                "contact_y_from_base_m": detail["position_from_base_m"][1],
                "contact_z_from_base_m": detail["position_from_base_m"][2],
                "force_x_on_wheel_N": detail["force_on_wheel_world_N"][0],
                "force_y_on_wheel_N": detail["force_on_wheel_world_N"][1],
                "force_z_on_wheel_N": detail["force_on_wheel_world_N"][2],
                "contact_torque_x_on_wheel_Nm": detail["contact_torque_on_wheel_world_Nm"][0],
                "contact_torque_y_on_wheel_Nm": detail["contact_torque_on_wheel_world_Nm"][1],
                "contact_torque_z_on_wheel_Nm": detail["contact_torque_on_wheel_world_Nm"][2],
                "ground_yaw_moment_about_base_Nm": detail["yaw_moment_about_base_Nm"],
            }
            contact_rows.append(row_detail)
        rows.append(row)

    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    contact_fieldnames = [
        "sample_index", "policy_step", "time_s", "wheel", "contact_id",
        "contact_x_world_m", "contact_y_world_m", "contact_z_world_m",
        "contact_x_from_base_m", "contact_y_from_base_m", "contact_z_from_base_m",
        "force_x_on_wheel_N", "force_y_on_wheel_N", "force_z_on_wheel_N",
        "contact_torque_x_on_wheel_Nm", "contact_torque_y_on_wheel_Nm",
        "contact_torque_z_on_wheel_Nm", "ground_yaw_moment_about_base_Nm",
    ]
    with contacts_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=contact_fieldnames)
        writer.writeheader()
        writer.writerows(contact_rows)

    yaw_offsets = np.asarray([row["yaw_from_policy_start_deg"] for row in rows], dtype=float)
    duration = float(rows[-1]["time_s"] - rows[0]["time_s"])
    contact_summary = []
    for wheel in wheel_names:
        fz = np.asarray([row[f"{wheel}_force_z_N"] for row in rows], dtype=float)
        contact_samples = int(np.count_nonzero(fz > 5.0))
        peak_index = int(np.argmax(fz))
        contact_summary.append((wheel, contact_samples, float(fz.max()), rows[peak_index]["time_s"]))
    wheel_drive_summary = []
    ground_yaw_summary = []
    for wheel in wheel_names:
        drive = np.asarray([row[f"{wheel}_motor_torque_Nm"] for row in rows], dtype=float)
        yaw_moment = np.asarray([
            row[f"{wheel}_ground_yaw_moment_about_base_Nm"] for row in rows
        ], dtype=float)
        wheel_drive_summary.append((wheel, float(drive.mean()), float(np.abs(drive).mean()),
                                   float(np.abs(drive).max())))
        ground_yaw_summary.append((wheel, float(yaw_moment.mean()), float(np.abs(yaw_moment).mean()),
                                   float(np.abs(yaw_moment).max())))
    ground_yaw_total = np.asarray([
        sum(row[f"{wheel}_ground_yaw_moment_about_base_Nm"] for wheel in wheel_names)
        for row in rows
    ], dtype=float)
    group_summary = []
    for group in ("fr_hl", "fl_hr"):
        values = np.asarray([row[f"{group}_sum_abs_torque_Nm"] for row in rows], dtype=float)
        group_summary.append((group.upper().replace("_", "+"), float(values.mean()), float(values.max())))

    level_groups: dict[int, list[dict]] = {}
    for row in rows:
        level_groups.setdefault(int(row["stair_level"]), []).append(row)
    lines = [
        "# MuJoCo 楼梯偏航与接触力矩记录",
        "",
        f"- Checkpoint：`{args.checkpoint.resolve()}`",
        f"- Checkpoint SHA256：`{hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}`",
        f"- 部署脚本 SHA256（模块载入时）：`{DEPLOY_SCRIPT_SHA256}`",
        f"- 遥测版本：`{TELEMETRY_VERSION}`；航向修正启用：`{not args.disable_heading_hold}`。",
        f"- 航向参数（kp, kd, limit, deadband）：flat={HEADING_HOLD_FLAT}，stairs={HEADING_HOLD_STAIRS}。",
        f"- 航向控制器：{'legacy_v1' if getattr(args, 'legacy_heading_hold', False) else 'filtered_v5'}；滤波 tau={HEADING_HOLD_FILTER_TAU}s，退出死区={math.degrees(HEADING_HOLD_EXIT_DEADBAND)}°，角速度进入/退出={HEADING_HOLD_RATE_ENTER}/{HEADING_HOLD_RATE_EXIT}rad/s，修正变化率 flat/stairs={HEADING_HOLD_SLEW_FLAT}/{HEADING_HOLD_SLEW_STAIRS}rad/s²。",
        f"- 机器人模型：`{args.model.resolve()}`",
        f"- 地形：`{args.terrain_config.path}`（{args.terrain_config.kind}）",
        f"- 策略有效采样：{len(rows)} 帧，约 {duration:.2f} 秒；采样周期约 {POLICY_DT:.3f} 秒。",
        f"- 偏航角采用世界坐标；正值为向左，负值为向右。相对角按首次进入策略控制时的航向计算。",
        "- `command_*` 为键盘原始请求；`effective_command_yaw_rad_s` 是策略实际收到的角速度命令。航向误差在策略步起点计算，偏航角在步末采样。",
        "- filtered_v5 在上下楼自动直行时启用路线横向反馈；上楼航向阻尼使用世界水平航向导数，其余模式保留机身Z角速度约定。",
        f"- 上楼自动直行增加路线横向保持：kp={ASCENT_PATH_KP}，kd={ASCENT_PATH_KD}，最大横向速度修正={ASCENT_PATH_MAX_LATERAL_SPEED}m/s，位置/速度死区={ASCENT_PATH_LATERAL_DEADBAND}m/{ASCENT_PATH_SPEED_DEADBAND}m/s，变化率={ASCENT_PATH_LATERAL_SLEW}m/s²。它只修正 vy，保留原始 vx；无积分及固定方向偏置。",
        f"- 下楼采用相同路线反馈增益，横向修正限幅为{DESCENT_PATH_MAX_LATERAL_SPEED}m/s；楼梯零指令制动{STAIR_STOP_BRAKE_GRACE}s后锁存位置，恢复速度限幅{STAIR_STOP_MAX_SPEED}m/s，kp/kd={STAIR_STOP_KP}/{STAIR_STOP_KD}。",
        "- 新增机身 roll/pitch、世界平移速度和四个轮心世界坐标，用于区分航向转动、横向滑移及左右前轮落阶时差。",
        "- `*_action_delta_rms` 和 `*_action_second_difference_rms` 是归一化策略动作的一阶、二阶差分 RMS；不除以采样周期。策略控制中断后重新计数。",
        "- `leg_target_*_rms_rad` 是应用各关节动作缩放后的腿部位置目标差分；`wheel_target_*_rms_m_s` 是应用轮速动作缩放和轮半径后的轮缘速度目标差分，与训练静止抖动统计采用相同物理单位。",
        "- `*_contact_physics_substeps` 在每个物理子步统计求解接触力大于1 N的接触；是接触子步数，不是独立碰撞事件次数。`rear_front_contact_impulse_Ns` 是后轮与前轮/前腿接触合力模长的累计冲量。",
        f"- 全回放后轮碰前轮接触子步：{sum(row['rear_front_wheel_contact_physics_substeps'] for row in rows)}；后轮碰前腿接触子步：{sum(row['rear_wheel_front_leg_contact_physics_substeps'] for row in rows)}；峰值接触力：{max(row['rear_front_peak_contact_force_N'] for row in rows):.3f} N。",
        "",
        "## 偏航概况",
        "",
        f"- 最终相对偏航：**{yaw_offsets[-1]:+.2f}°**；最大右偏：**{min(0.0, float(yaw_offsets.min())):.2f}°**；最大左偏：**{max(0.0, float(yaw_offsets.max())):+.2f}°**。",
        f"- 观测到的踏阶层级跃迁：**{len(transitions)} 次**。每行的 Δ 是从前一级首次达到该层级时到当前层级首次达到时的偏航变化。",
        "",
        "| 时间 (s) | 层级 | 前级偏航 (°) | 当前偏航 (°) | 变化 Δ (°) |",
        "|---:|:---:|---:|---:|---:|",
    ]
    if transitions:
        for event in transitions:
            lines.append(
                f"| {event['time_s']:.2f} | L{event['from']}→L{event['to']} | "
                f"{event['yaw_before_deg']:+.2f} | {event['yaw_after_deg']:+.2f} | {event['delta_yaw_deg']:+.2f} |"
            )
    else:
        lines.append("| — | 未检测到层级跃迁 | — | — | — |")
    lines.extend([
        "",
        "## 轮地接触力",
        "",
        "接触样本按该轮法向合力 `Fz > 5 N` 计数；最大值时间便于与偏航/踏阶跃迁对齐。",
        "",
        "| 车轮 | 接触样本数 | 最大 Fz (N) | 峰值时间 (s) |",
        "|:---:|---:|---:|---:|",
    ])
    for wheel, count, peak, peak_time in contact_summary:
        lines.append(f"| {wheel} | {count} | {peak:.1f} | {peak_time:.2f} |")
    lines.extend([
        "",
        "## 对角腿力矩",
        "",
        "组值是两条腿 hipx、hipy、knee 力矩绝对值之和；CSV 同时保留每个关节的带符号原始力矩。",
        "",
        "| 对角腿组 | 平均绝对力矩和 (Nm) | 最大绝对力矩和 (Nm) |",
        "|:---:|---:|---:|",
    ])
    for group, mean, maximum in group_summary:
        lines.append(f"| {group} | {mean:.1f} | {maximum:.1f} |")
    lines.extend([
        "",
        "## 轮电机力矩与轮地偏航力矩",
        "",
        "轮电机力矩是 MuJoCo 执行器在轮关节上的实际力矩，保留关节坐标符号。地面接触偏航力矩由各接触点的接触力臂矩和接触偶矩求和，参考点是 base_link 原点；世界 +Z 为正（左转）。",
        "",
        "| 车轮 | 平均电机力矩 (Nm) | 平均绝对电机力矩 (Nm) | 最大绝对电机力矩 (Nm) | 平均接触偏航力矩 (Nm) | 平均绝对接触偏航力矩 (Nm) | 最大绝对接触偏航力矩 (Nm) |",
        "|:---:|---:|---:|---:|---:|---:|---:|",
    ])
    for drive, yaw_moment in zip(wheel_drive_summary, ground_yaw_summary):
        lines.append(
            f"| {drive[0]} | {drive[1]:+.2f} | {drive[2]:.2f} | {drive[3]:.2f} | "
            f"{yaw_moment[1]:+.2f} | {yaw_moment[2]:.2f} | {yaw_moment[3]:.2f} |"
        )
    lines.extend([
        f"\n- 四轮接触偏航力矩总和：平均 **{ground_yaw_total.mean():+.2f} Nm**，平均绝对值 **{np.abs(ground_yaw_total).mean():.2f} Nm**，最大绝对值 **{np.abs(ground_yaw_total).max():.2f} Nm**。",
        "- 逐接触点数据另存为 `" + contacts_path.name + "`；可据此核对接触位置、接触力和每个接触点对 base_link 的偏航力矩贡献。",
        "",
        "## 按踏阶层级汇总",
        "",
        "| 层级 | 样本数 | 平均相对偏航 (°) | 平均 FR+HL 绝对力矩和 (Nm) | 平均 FL+HR 绝对力矩和 (Nm) |",
        "|---:|---:|---:|---:|---:|",
    ])
    for level, level_rows in sorted(level_groups.items()):
        mean_yaw = np.mean([r["yaw_from_policy_start_deg"] for r in level_rows])
        fr_hl = np.mean([r["fr_hl_sum_abs_torque_Nm"] for r in level_rows])
        fl_hr = np.mean([r["fl_hr_sum_abs_torque_Nm"] for r in level_rows])
        lines.append(f"| L{level} | {len(level_rows)} | {mean_yaw:+.2f} | {fr_hl:.1f} | {fl_hr:.1f} |")
    lines.extend([
        "",
        f"逐帧汇总：[`{csv_path.name}`]({csv_path.name})。每个策略步记录四轮接触力、轮电机力矩、接触计数/偏航力矩以及 12 个腿部关节力矩。",
        f"逐接触明细：[`{contacts_path.name}`]({contacts_path.name})。",
        "",
    ])
    lines.extend(refinement_summary_lines(rows, args.terrain_config.kind))
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path, contacts_path


def refinement_summary_lines(rows: list[dict], terrain_kind: str) -> list[str]:
    """Report command blocks and contact-contaminated samples explicitly."""
    lines = ["## 按原始命令阶段验收", "",
             "每段零指令剔除最初 0.4 s 制动期；遇到侧边余量不足 0.10 m 或非轮部件碰地后，停止该段纯静止统计。侧边标记是几何代理，不是已确认碰撞。",
             "成功样本要求机身速度<0.02 m/s、最大轮速<0.03 m/s、|偏航角速度|<0.03 rad/s、航向误差<2°。比例是时间样本比例，不是整次回放成功率。", "",
             "| 时间/s | 原始 vx,vy,wz | 有效帧/总帧 | 航向变化/° | 平均机身速度/(m/s) | 稳定样本比例 | 保持期累计路程/m |",
             "|---|---|---:|---:|---:|---:|---:|"]
    blocks = []
    for row in rows:
        command = tuple(round(float(row[k]), 4) for k in ("command_vx_m_s", "command_vy_m_s", "command_yaw_rad_s"))
        if not blocks or blocks[-1][0] != command:
            blocks.append((command, []))
        blocks[-1][1].append(row)
    for command, block in blocks:
        zero = np.linalg.norm(command[:2]) <= 0.10 and abs(command[2]) <= 0.10
        clean = []
        for row in block:
            if row["near_lateral_edge"] or row["nonwheel_ground_contacts"] > 0:
                break
            if not zero or row["time_s"] - block[0]["time_s"] >= 0.4:
                clean.append(row)
        if clean:
            delta = math.degrees(wrap_angle(math.radians(clean[-1]["yaw_deg"] - clean[0]["yaw_deg"])))
            speed = np.mean([r["base_speed_m_s"] for r in clean])
            xy = np.array([[r["base_x_m"], r["base_y_m"]] for r in clean])
            path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
            stable = np.mean([r["base_speed_m_s"] < 0.02 and r["wheel_speed_max_m_s"] < 0.03
                              and abs(r["yaw_rate_rad_s"]) < 0.03 and abs(r["heading_error_at_command_deg"]) < 2.0 for r in clean])
            values = f"{delta:+.2f} | {speed:.4f} | {stable:.1%} | {path:.4f}" if zero else f"{delta:+.2f} | {speed:.4f} | — | —"
        else:
            values = "— | — | — | —"
        lines.append(f"| {block[0]['time_s']:.2f}–{block[-1]['time_s']:.2f} | {command} | {len(clean)}/{len(block)} | {values} |")
    wheel_contacts = sum(r["rear_front_wheel_contacts"] > 0 for r in rows)
    shank_contacts = sum(r["rear_wheel_front_leg_contacts"] > 0 for r in rows)
    lines.extend(["", f"- 后轮碰前轮样本帧：{wheel_contacts}；后轮碰前腿非轮部件样本帧：{shank_contacts}。这是50 Hz采样帧数，不能替代物理子步接触事件次数。", ""])
    if terrain_kind == "descent":
        lines.extend(["## 前三级下楼前腿姿态", "",
                      "级数按前轮首次查询到更低踏面的累计最大层差统计，与训练的整车完成计数不同。支撑期使用轮地接触且 Fz>5 N；髋轮前向距离为机身坐标，长度为三维距离。", "",
                      "| 下落级数 | 前腿 | 阶段 | 帧数 | 前向距离均值/m | 前向距离P95/m | 髋轮长度均值/m | 膝角绝对值均值/rad |",
                      "|---:|---|---|---:|---:|---:|---:|---:|"])
        for level in (1, 2, 3):
            for wheel, leg in (("FL", "fl"), ("FR", "fr")):
                for support in (True, False):
                    selected = [r for r in rows if r["descent_drop_count"] == level
                                and r["command_vx_m_s"] > 0.10 and not r["near_lateral_edge"]
                                and ((r[f"{wheel}_force_z_N"] > 5 and r[f"{wheel}_ground_contact_count"] > 0) == support)]
                    if selected:
                        forward = [r[f"{wheel}_hip_wheel_forward_m"] for r in selected]
                        length = np.mean([r[f"{wheel}_hip_wheel_length_m"] for r in selected])
                        knee = np.mean([abs(r[f"{leg}_knee_angle_rad"]) for r in selected])
                        lines.append(f"| {level} | {wheel} | {'支撑' if support else '摆动'} | {len(selected)} | {np.mean(forward):.4f} | {np.percentile(forward,95):.4f} | {length:.4f} | {knee:.3f} |")
    return lines + [""]


def interactive_rollout(args: argparse.Namespace) -> dict:
    """Start prone, stand with Z, then hand control to the teacher with C."""
    torch.set_num_threads(1)
    actor = load_actor(args.checkpoint)
    args.legacy_heading_hold = use_checkpoint_control_profile(args, actor)
    model = load_model(args.model, args)
    contract = M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data, prone=True)
    print("[TERRAIN] " + json.dumps(args.terrain_config.summary()), flush=True)
    print_deployment_contract(args)
    keys = TeleopKeyboard()
    terminal = TerminalInput(keys)
    viewer, viewer_threads = launch_viewer(model, data, key_callback=keys.on_keycode)
    try:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = contract.base_id
            viewer.cam.distance = 3.2
            viewer.cam.elevation = -18.0
        terminal_active = terminal.start()
        print("[KEYS] Z stand | C teacher policy | X lie down | R damping | H height scan | Esc quit", flush=True)
        print("[HEIGHTMAP] 187 points: green=local ground, red=higher, blue=lower", flush=True)
        print("[KEYS] W/S forward/back | A/D left/right | Q/E turn left/right", flush=True)
        print(f"[KEYS] Input: MuJoCo window{' or focused terminal' if terminal_active else ''}", flush=True)
        print(f"[KEYS] Window motion: {'hold/release state (X11)' if keys._window_keys is not None else 'repeat events with timeout; use focused terminal if window only reports PRESS'}", flush=True)
        if args.telemetry:
            print("[TELEMETRY] live terminal dashboard enabled", flush=True)
        print("[MODE] prone", flush=True)
    except BaseException:
        terminal.close()
        try:
            close_viewer(viewer, viewer_threads)
        finally:
            keys.close()
        raise

    mode = "prone"
    stand_start = 0.0
    stand_initial = None
    lie_start = 0.0
    lie_initial = None
    policy_start = 0.0
    pending_policy = False
    show_height_scan = True
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    heading_hold = HeadingHoldController(contract, enabled=not args.disable_heading_hold,
                                         legacy=getattr(args, "legacy_heading_hold", False))
    telemetry = StairTelemetry(contract)
    telemetry_samples = []
    records = {"time": [], "root_pos": [], "command": [], "requested_command": [],
               "mode": [], "action": []} if args.output else None
    transitions = [{"time": 0.0, "mode": "prone"}]
    start_x = float(data.xpos[contract.base_id, 0])
    max_x = start_x
    fell_ever = False
    substeps = round(POLICY_DT / PHYSICS_DT)
    step = 0
    steps_done = 0

    def enter(new_mode: str) -> None:
        nonlocal mode
        mode = new_mode
        transitions.append({"time": round(float(data.time), 3), "mode": new_mode})
        print(f"[MODE] {new_mode} t={data.time:.2f}s", flush=True)

    try:
        while viewer.is_running() and (args.steps is None or step < args.steps):
            tick_start = time.perf_counter()
            terminal.poll()
            quit_requested = False
            for key in keys.take_events():
                if key == "escape":
                    quit_requested = True
                elif key == "h":
                    show_height_scan = not show_height_scan
                    print(f"[HEIGHTMAP] {'visible' if show_height_scan else 'hidden'}", flush=True)
                elif key == "r":
                    pending_policy = False
                    previous_action.fill(0)
                    heading_hold.reset()
                    keys.clear_motion()
                    enter("damping")
                elif key == "z":
                    if mode in {"prone", "damping", "lying"}:
                        stand_start = float(data.time)
                        stand_initial = data.qpos[contract.qpos_ids].copy()
                        pending_policy = False
                        enter("standing")
                elif key == "c":
                    if mode == "standing":
                        pending_policy = True
                        print("[MODE] policy queued until standing finishes", flush=True)
                    elif mode == "ready":
                        policy_start = float(data.time)
                        previous_action.fill(0)
                        heading_hold.reset()
                        enter("policy")
                    elif mode != "policy":
                        print("[MODE] press Z and wait for ready before C", flush=True)
                elif key == "x" and mode not in {"prone", "lying"}:
                    lie_start = float(data.time)
                    lie_initial = data.qpos[contract.qpos_ids].copy()
                    pending_policy = False
                    previous_action.fill(0)
                    heading_hold.reset()
                    keys.clear_motion()
                    enter("lying")
            if quit_requested:
                break

            if mode == "standing" and data.time - stand_start >= 2.0 * STAND_PHASE_DURATION:
                position = data.xpos[contract.base_id]
                ground = contract.ground_height(data, float(position[0]), float(position[1]))
                upright = data.xmat[contract.base_id].reshape(3, 3)[2, 2] > 0.8
                if position[2] >= ground + 0.42 and upright:
                    enter("ready")
                    if pending_policy:
                        policy_start = float(data.time)
                        previous_action.fill(0)
                        heading_hold.reset()
                        enter("policy")
                elif step % args.print_every == 0:
                    print("[MODE] holding stand pose until body is upright", flush=True)
            if mode == "lying" and data.time - lie_start >= STAND_PHASE_DURATION:
                enter("prone")

            requested_command = keys.velocity_command() if mode == "policy" else np.zeros(3, dtype=np.float32)
            command = requested_command.copy()
            heading_info = None
            if mode == "policy":
                command, heading_info = heading_hold.update(data, requested_command)
            applied_action = np.zeros(ACTION_DIM, dtype=np.float32)
            if mode == "policy":
                with torch.inference_mode():
                    tensor_obs = actor_inputs(contract, data, command, previous_action, actor, requested_command)
                    raw_action = actor(tensor_obs).squeeze(0).cpu().numpy()
                if raw_action.shape != (ACTION_DIM,) or not np.isfinite(raw_action).all():
                    raise RuntimeError(f"Invalid actor action at step {step}: {raw_action}")
                blend = min(1.0, (data.time - policy_start) / 0.3)
                applied_action = np.clip(raw_action * blend, -100.0, 100.0).astype(np.float32)
                previous_action = applied_action.copy()
            elif mode == "standing":
                elapsed = data.time - stand_start
                if elapsed < STAND_PHASE_DURATION:
                    blend = smoothstep(elapsed / STAND_PHASE_DURATION)
                    target = stand_initial + blend * (PRE_STAND_JOINT_POS - stand_initial)
                else:
                    blend = smoothstep((elapsed - STAND_PHASE_DURATION) / STAND_PHASE_DURATION)
                    target = PRE_STAND_JOINT_POS + blend * (DEFAULT_JOINT_POS - PRE_STAND_JOINT_POS)
            elif mode == "lying":
                blend = smoothstep((data.time - lie_start) / STAND_PHASE_DURATION)
                target = lie_initial + blend * (PRONE_JOINT_POS - lie_initial)
            elif mode == "ready":
                target = DEFAULT_JOINT_POS
            if mode != "policy" and show_height_scan:
                contract.height_scan(data)

            if records is not None:
                records["time"].append(float(data.time))
                records["root_pos"].append(data.xpos[contract.base_id].copy())
                records["command"].append(command.copy())
                records["requested_command"].append(requested_command.copy())
                records["mode"].append(mode)
                records["action"].append(applied_action.copy())
            for _ in range(substeps):
                if mode == "policy":
                    contract.apply_pd(data, applied_action)
                elif mode in {"standing", "ready", "lying"}:
                    contract.apply_pose_pd(data, target)
                else:
                    contract.apply_damping(data)
                mujoco.mj_step(model, data)
                if args.telemetry and mode == "policy":
                    telemetry.observe_physics_step(data)
            steps_done += 1
            draw_height_scan(viewer, contract, show_height_scan)
            viewer.sync()
            time.sleep(max(0.0, POLICY_DT - (time.perf_counter() - tick_start)))
            position = data.xpos[contract.base_id]
            max_x = max(max_x, float(position[0]))
            if mode == "policy":
                ground = contract.ground_height(data, float(position[0]), float(position[1]))
                upright = data.xmat[contract.base_id].reshape(3, 3)[2, 2] > 0.35
                fell = position[2] < ground + 0.22 or not upright
                fell_ever |= fell
                if fell and args.stop_on_fall:
                    print("[MODE] policy fall detected; stopping", flush=True)
                    break
            if args.telemetry and mode == "policy":
                sample = telemetry.sample(data, applied_action)
                sample["heading_hold"] = heading_info
                sample["requested_command"] = requested_command.copy()
                telemetry_samples.append({
                    **sample,
                    "policy_step": step,
                    "time_s": float(data.time),
                    "command": requested_command.astype(float).tolist(),
                    "effective_command": command.astype(float).tolist(),
                    "heading_hold": heading_info,
                    "base_xyz": data.xpos[contract.base_id].astype(float).tolist(),
                })
                if step % args.telemetry_every == 0:
                    render_stair_telemetry(sample, data, contract, command, mode)
            elif step % args.print_every == 0:
                print(
                    f"step={step:04d} mode={mode} t={data.time:.2f}s "
                    f"x={position[0]:.3f} z={position[2]:.3f} cmd={command.tolist()}",
                    flush=True,
                )
            step += 1
    except KeyboardInterrupt:
        print("[MODE] interrupted by user; closing viewer and exporting collected telemetry", flush=True)
    finally:
        try:
            terminal.close()
        finally:
            try:
                close_viewer(viewer, viewer_threads)
            finally:
                keys.close()

    telemetry_paths = export_stair_telemetry(telemetry_samples, args) if args.telemetry else None
    if telemetry_paths is not None:
        print(f"[TELEMETRY] CSV: {telemetry_paths[0].resolve()}", flush=True)
        print(f"[TELEMETRY] Markdown summary: {telemetry_paths[1].resolve()}", flush=True)
        print(f"[TELEMETRY] Contact details CSV: {telemetry_paths[2].resolve()}", flush=True)
    elif args.telemetry:
        print("[TELEMETRY] no policy-controlled samples; no telemetry files written", flush=True)

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model": str(args.model.resolve()),
        **args.terrain_config.summary(),
        **contract.crossing_summary(data, fell_ever),
        "steps_executed": steps_done,
        "final_mode": mode,
        "transitions": transitions,
        "max_x": max_x,
        "final_x": float(data.xpos[contract.base_id, 0]),
        "final_z": float(data.xpos[contract.base_id, 2]),
        "fell_in_policy": bool(fell_ever),
        "critic_used": False,
    }
    if telemetry_paths is not None:
        result["telemetry_csv"] = str(telemetry_paths[0].resolve())
        result["telemetry_markdown"] = str(telemetry_paths[1].resolve())
        result["telemetry_contacts_csv"] = str(telemetry_paths[2].resolve())
    if records is not None:
        for key in records:
            records[key] = np.asarray(records[key])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **records, summary=json.dumps(result))
        result["trajectory"] = str(args.output.resolve())
    print(json.dumps(result, indent=2), flush=True)
    return result


def rollout(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    actor = load_actor(args.checkpoint)
    args.legacy_heading_hold = use_checkpoint_control_profile(args, actor)
    model = load_model(args.model, args)
    contract = M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data)
    print("[TERRAIN] " + json.dumps(args.terrain_config.summary()), flush=True)
    print_deployment_contract(args)
    command = np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
    heading_hold = HeadingHoldController(contract, enabled=not args.disable_heading_hold,
                                         legacy=getattr(args, "legacy_heading_hold", False))
    telemetry = StairTelemetry(contract)
    telemetry_samples = []
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    substeps = round(POLICY_DT / PHYSICS_DT)
    if not math.isclose(substeps * PHYSICS_DT, POLICY_DT):
        raise RuntimeError("Policy period is not an integer number of physics steps")
    records = {"time": [], "root_pos": [], "root_quat": [], "requested_command": [],
               "command": [], "observation": [], "action": []}
    if getattr(actor, 'uses_stop_control', False):
        records['stop_control_observation'] = []
    start_x = float(data.xpos[contract.base_id, 0])
    fell = False
    fell_ever = False
    max_x = start_x
    viewer = None
    viewer_threads = []
    if args.viewer:
        viewer, viewer_threads = launch_viewer(model, data)

    try:
        for step in range(args.steps if args.steps is not None else 1000):
            tic = time.perf_counter()
            if args.command_segments:
                command = command_profile_at(args.command_segments, step * POLICY_DT)
            effective_command, heading_info = heading_hold.update(data, command)
            with torch.inference_mode():
                tensor_obs = actor_inputs(contract, data, effective_command, previous_action, actor, command)
                obs = tensor_obs['policy'].squeeze(0).numpy()
                action = actor(tensor_obs).squeeze(0).cpu().numpy()
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise RuntimeError(f"Invalid actor action at step {step}: {action}")
            action = np.clip(action, -100.0, 100.0)
            records["time"].append(float(data.time))
            records["root_pos"].append(data.xpos[contract.base_id].copy())
            records["root_quat"].append(data.xquat[contract.base_id].copy())
            records["requested_command"].append(command.copy())
            records["command"].append(effective_command.copy())
            records["observation"].append(obs)
            records["action"].append(action.copy())
            if 'stop_control_observation' in records:
                records['stop_control_observation'].append(tensor_obs['stop_control'][0].numpy().copy())
            for _ in range(substeps):
                contract.apply_pd(data, action)
                mujoco.mj_step(model, data)
                if args.telemetry:
                    telemetry.observe_physics_step(data)
            if args.telemetry:
                sample = telemetry.sample(data, action)
                sample.update(time_s=float(data.time), policy_step=step,
                              command=command.copy(), effective_command=effective_command.copy(),
                              heading_hold=heading_info, base_xyz=data.xpos[contract.base_id].copy())
                telemetry_samples.append(sample)
            previous_action = action.astype(np.float32)
            if viewer is not None:
                draw_height_scan(viewer, contract, True)
                viewer.sync()
                time.sleep(max(0.0, POLICY_DT - (time.perf_counter() - tic)))
            position = data.xpos[contract.base_id]
            gravity_z = float(data.xmat[contract.base_id].reshape(3, 3)[:, 2] @ np.array([0.0, 0.0, -1.0]))
            local_ground = contract.ground_height(data, float(position[0]), float(position[1]))
            fell = bool(position[2] < local_ground + 0.22 or gravity_z > -0.35)
            fell_ever |= fell
            max_x = max(max_x, float(position[0]))
            if step % args.print_every == 0 or fell:
                print(
                    f"step={step:04d} t={data.time:.2f}s x={position[0]:.3f} "
                    f"z={position[2]:.3f} ground={local_ground:.3f} fall={fell}",
                    flush=True,
                )
            if fell and args.stop_on_fall:
                break
            if viewer is not None and not viewer.is_running():
                break
    finally:
        close_viewer(viewer, viewer_threads)

    for key in records:
        records[key] = np.asarray(records[key])
    final_pos = data.xpos[contract.base_id].copy()
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model": str(args.model.resolve()),
        **args.terrain_config.summary(),
        **contract.crossing_summary(data, fell_ever),
        "steps_executed": len(records["time"]),
        "physics_enabled": True,
        "actor_input_dim": ACTOR_OBS_DIM,
        "critic_used": False,
        "start_x": start_x,
        "final_x": float(final_pos[0]),
        "final_z": float(final_pos[2]),
        "max_x": max_x,
        "forward_distance": float(final_pos[0] - start_x),
        "fell": fell_ever,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **records, summary=json.dumps(result))
        result["trajectory"] = str(args.output.resolve())
    telemetry_paths = export_stair_telemetry(telemetry_samples, args) if args.telemetry else None
    if telemetry_paths is not None:
        result.update(telemetry_csv=str(telemetry_paths[0].resolve()),
                      telemetry_markdown=str(telemetry_paths[1].resolve()),
                      telemetry_contacts_csv=str(telemetry_paths[2].resolve()))
    print(json.dumps(result, indent=2), flush=True)
    return result


def print_deployment_contract(args: argparse.Namespace) -> None:
    print(f"[CONTROL] {TELEMETRY_VERSION} heading_hold={not args.disable_heading_hold} "
          f"controller={'legacy_v1' if getattr(args, 'legacy_heading_hold', False) else 'filtered_v5'} "
          f"flat={HEADING_HOLD_FLAT} stairs={HEADING_HOLD_STAIRS}", flush=True)


def command_profile_at(segments: list[dict], elapsed: float) -> np.ndarray:
    end = 0.0
    for segment in segments:
        end += segment["duration_s"]
        if elapsed < end - 1.0e-9:
            return np.asarray(segment["command"], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", type=Path, required=True, help="M20 MuJoCo XML (pin the tested model version)")
    terrain_choice = parser.add_mutually_exclusive_group()
    terrain_choice.add_argument("--terrain", choices=tuple(PRESETS),
                               help="Built-in terrain XML preset; defaults to ascent")
    terrain_choice.add_argument("--terrain-xml", type=Path,
                               help="Parameterized/native terrain XML, independent of --model")
    parser.add_argument("--stair-height", type=float, help="Override staircase XML step_height (metres)")
    parser.add_argument("--tread-depth", type=float, help="Override staircase XML tread_depth (metres)")
    parser.add_argument("--stair-count", type=int, help="Override staircase XML step_count")
    parser.add_argument("--stair-start", type=float, help="Override staircase XML start_x (metres)")
    parser.add_argument("--command-x", type=float, default=0.5)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--command-profile", type=Path,
                        help="JSON list of {duration_s, command:[vx,vy,wz]} for batch/autoplay replay")
    parser.add_argument("--disable-heading-hold", action="store_true",
                        help="Send raw commands for a controlled comparison; default enables heading hold")
    parser.add_argument("--legacy-heading-hold", action="store_true",
                        help="Use the previous unfiltered heading controller for an otherwise identical A/B replay")
    parser.add_argument("--steps", type=int, help="Limit to this many 50 Hz steps; viewer teleop runs until Esc by default")
    parser.add_argument("--print-every", type=int, default=10,
                        help="Print status every N policy steps (default: 10 = 0.2 simulated seconds)")
    parser.add_argument("--telemetry", action="store_true",
                        help="Show live stair telemetry and export a Markdown summary plus per-step CSV on exit")
    parser.add_argument("--telemetry-every", type=int, default=5,
                        help="Refresh the live telemetry dashboard every N policy steps (default: 5 = 0.1 simulated seconds)")
    parser.add_argument("--stop-on-fall", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="Open interactive keyboard teleop")
    parser.add_argument("--autoplay", action="store_true", help="Run fixed requested commands with heading hold in the viewer")
    parser.add_argument("--output", type=Path, help="Optional compressed trajectory .npz")
    args = parser.parse_args()
    args.command_segments = []
    if args.command_profile is not None:
        if args.viewer and not args.autoplay:
            parser.error("--command-profile with --viewer requires --autoplay")
        try:
            segments = json.loads(args.command_profile.read_text())
            if not isinstance(segments, list) or not segments:
                raise ValueError("profile must be a nonempty list")
            for segment in segments:
                duration = float(segment["duration_s"])
                command = np.asarray(segment["command"], dtype=float)
                if not math.isfinite(duration) or duration <= 0 or command.shape != (3,) or not np.isfinite(command).all():
                    raise ValueError("each segment needs positive duration_s and three finite command values")
                args.command_segments.append({"duration_s": duration, "command": command.tolist()})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            parser.error(f"invalid --command-profile: {exc}")
        if args.steps is None:
            args.steps = math.ceil(sum(s["duration_s"] for s in args.command_segments) / POLICY_DT)
    if ((args.steps is not None and args.steps <= 0)
        or any(value is not None and value <= 0 for value in
               (args.stair_count, args.stair_height, args.tread_depth))):
        parser.error("steps (when given), stair-count, stair-height, and tread-depth must be positive")
    if args.print_every <= 0 or args.telemetry_every <= 0:
        parser.error("print-every and telemetry-every must be positive")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.viewer and not cli_args.autoplay:
        interactive_rollout(cli_args)
    else:
        rollout(cli_args)
