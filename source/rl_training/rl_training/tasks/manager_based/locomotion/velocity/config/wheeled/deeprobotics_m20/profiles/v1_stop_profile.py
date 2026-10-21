"""Explicit v1-compatible motion baseline with only stop-state refinement.

The saved v1 manifest establishes numeric provenance, not a reconstruction of
the unavailable historical source. Select this profile before constructing envs.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

PROFILE = "v1_stop"
VERSION = "m20_v1_stop_20261017"
VERSION_V2 = "m20_v1_stop_v2_20261017"
VERSION_V3 = "m20_v1_stop_v3_20261017"
VERSION_V4 = "m20_v1_stop_v4_20261017"
VERSION_V5 = "m20_v1_stop_v5_20261017"
VERSION_V6 = "m20_idle_stop_motion_v1_20261017"
QUIET_ACTOR_SHA256 = "fa2f9eb31145db46db45b337fe538fc7aa9009629b244134538d03f0c7f84587"
IDLE_STOP_ACTOR_SHA256 = "0e6d0c93560b5a3f7ce9219c7b6a0cea84bd4a2d3f5dba33c02d79f7d4eb5f34"
PROFILE_VERSIONS = {PROFILE: VERSION, "v1_stop_v2": VERSION_V2,
                    "v1_stop_v3": VERSION_V3,
                    "v1_stop_v3_target_control": VERSION_V3 + "_target_control",
                    "v1_stop_v4": VERSION_V4, "v1_stop_v5": VERSION_V5,
                    "idle_stop_motion_v1": VERSION_V6,
                    "ascent_entry_v1": "m20_ascent_entry_20261020_v1",
                    "ascent_entry_v1_stage2": "m20_ascent_entry_20261020_v1_stage2",
                    "ascent_entry_v1_stage3": "m20_ascent_landing_20261020_v1_stage3",
                    "ascent_entry_v1_stage4": "m20_ascent_sequence_20261020_v1_stage4",
                    "ascent_entry_v1_stage5": "m20_ascent_adaptive_20261021_v1_stage5",
                    "ascent_entry_v1_stage6": "m20_ascent_alternating_20261021_v1_stage6_geom_v2",
                    "ascent_entry_v1_stage7": "m20_ascent_unified_20261021_v1_stage7"}
PROFILES = tuple(PROFILE_VERSIONS)


def uses_stop_metadata(profile):
    return profile in PROFILES and profile not in (PROFILE, "idle_stop_motion_v1", "ascent_entry_v1", "ascent_entry_v1_stage2", "ascent_entry_v1_stage3", "ascent_entry_v1_stage4", "ascent_entry_v1_stage5", "ascent_entry_v1_stage6", "ascent_entry_v1_stage7")
BASELINE_PATH = Path(__file__).parent / "data" / "v1_stop_baseline.json"


def baseline():
    return json.loads(BASELINE_PATH.read_text())


def configure_tuning(tuning, profile=PROFILE):
    """Restore recorded v1 values; leave later implementations behind gates."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown stop profile: {profile}")
    tuning.update(baseline()["tuning"])
    tuning.update(v1_motion_baseline=True, ascent_heading_tolerance=0.0,
                  ascent_straight_motion_weight=0.0, descent_front_pair_weight=0.0,
                  descent_entry_pause_probability=0.0, interleg_critical_gain=0.0,
                  interleg_braking_mix=0.0)
    tuning.update(zero_stop_v2=uses_stop_metadata(profile),
                  zero_stop_v3=profile in ("v1_stop_v3", "v1_stop_v3_target_control", "v1_stop_v4", "v1_stop_v5"),
                  zero_stop_v4=profile in ("v1_stop_v4", "v1_stop_v5"),
                  zero_stop_v5=profile == "v1_stop_v5",
                  zero_stop_fast_stats=profile == "v1_stop_v5",
                  zero_stop_max_terrain_level=3)
    if profile == "idle_stop_motion_v1":
        # Retain bounded inter-leg protection and strengthen ascent path
        # tracking for the idle-stop motion refinement.
        tuning.update(interleg_critical_gain=.5, interleg_braking_mix=.25,
                      ascent_direction_weight=-2.0,
                      ascent_straight_motion_weight=-2.0)


def apply_profile(env_cfg, agent_cfg, profile=PROFILE):
    if profile == 'ascent_entry_v1_stage7':
        from .ascent_unified_profile import apply_profile as unified_profile
        return unified_profile(env_cfg, agent_cfg)
    if profile == 'ascent_entry_v1_stage6':
        from .ascent_alternating_profile import apply_profile as alternating_profile
        return alternating_profile(env_cfg, agent_cfg)
    if profile == 'ascent_entry_v1_stage5':
        from .ascent_adaptive_profile import apply_profile as adaptive_profile
        return adaptive_profile(env_cfg, agent_cfg)
    if profile == 'ascent_entry_v1_stage4':
        from .ascent_sequence_profile import apply_profile as sequence_profile
        return sequence_profile(env_cfg, agent_cfg)
    if profile == 'ascent_entry_v1_stage3':
        from .ascent_landing_profile import apply_profile as landing_profile
        return landing_profile(env_cfg, agent_cfg)
    if profile == 'ascent_entry_v1_stage2':
        from .ascent_entry_stage2_profile import apply_profile as stage2_profile
        return stage2_profile(env_cfg, agent_cfg)
    if profile == 'ascent_entry_v1':
        from .ascent_entry_profile import apply_profile as ascent_profile
        return ascent_profile(env_cfg,agent_cfg)
    from isaaclab.managers import CurriculumTermCfg, RewardTermCfg, SceneEntityCfg
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_teacher as teacher
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards

    saved = baseline()
    configure_tuning(teacher.TUNING, profile)
    # No later movement shaping remains merely because its weight is nonzero
    # in the current config. Retain v1 rewards, including its safety margins.
    for name, term in vars(env_cfg.rewards).items():
        if term is not None and hasattr(term, "weight"):
            if name not in saved["reward_weights"]:
                setattr(env_cfg.rewards, name, None)
            else:
                term.weight = saved["reward_weights"][name]
    for name, weight in saved["reward_weights"].items():
        if weight and getattr(env_cfg.rewards, name, None) is None:
            raise ValueError(f"v1 baseline reward missing: {name}")
    if profile == "idle_stop_motion_v1":
        # This profile has its own bounded, raw-command-gated motion residual.
        # Keep zero-command stop rewards out of its policy objective.
        from isaaclab.managers import RewardTermCfg
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_teacher as teacher
        env_cfg.rewards.stair_ascent_direction_cost.weight = -2.0
        env_cfg.rewards.stair_ascent_heading_cost.weight = -3.5
        env_cfg.rewards.stair_ascent_yaw_tracking_cost.weight = -2.0
        # This term was absent from the recorded v1 reward table, so the
        # generic restoration above removes it. Re-add it explicitly here.
        env_cfg.rewards.stair_ascent_straight_motion_cost = RewardTermCfg(
            func=teacher.stair_ascent_straight_motion_cost, weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"})
        # These shaping terms are intentionally outside the recorded v1
        # reward table. Keep zero-weight terms in the profile so the staged
        # training CLI can activate them without dereferencing None.
        legs = ("fl", "fr", "hl", "hr")
        guide_entities = {
            "asset_cfg": SceneEntityCfg("robot", body_names=[f"{leg}_wheel" for leg in legs], preserve_order=True),
            "hip_cfg": SceneEntityCfg("robot", body_names=[f"{leg}_hipy" for leg in legs], preserve_order=True),
            "knee_cfg": SceneEntityCfg("robot", joint_names=[f"{leg}_knee_joint" for leg in legs], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"{leg}_wheel" for leg in legs], preserve_order=True),
        }
        for name, func in (
            ("stair_descent_entry_front_lead_reward", teacher.stair_descent_entry_front_lead_reward),
            ("stair_descent_entry_front_progress_reward", teacher.stair_descent_entry_front_progress_reward),
            ("stair_descent_entry_sequence_reward", teacher.stair_descent_entry_sequence_reward),
        ):
            setattr(env_cfg.rewards, name, RewardTermCfg(func=func, weight=0.0, params=guide_entities.copy()))
        env_cfg.rewards.stair_descent_lateral_cost = RewardTermCfg(
            func=teacher.stair_descent_lateral_cost, weight=-0.75,
            params={"command_name": "base_velocity"})
        for name in ("zero_stop_velocity", "zero_stop_position", "zero_stop_chatter"):
            setattr(env_cfg.rewards, name, None)
        env_cfg.scene.terrain.max_init_terrain_level = 3
    env_cfg.rewards.action_smooth_l2.func = rewards.action_smooth_l2
    env_cfg.rewards.stand_still.func = teacher.zero_stop_flat_stand_still
    # Replace old stop penalties instead of stacking several generations.
    for name in ("zero_command_hold_cost", "zero_command_flat_leg_motion_cost",
                 "zero_command_flat_support_cost", "zero_command_wheel_action_cost"):
        setattr(env_cfg.rewards, name, None)
    legs = [f"{leg}_{joint}_joint" for leg in ("fl", "fr", "hl", "hr")
            for joint in ("hipx", "hipy", "knee")]
    wheels = [f"{leg}_wheel_joint" for leg in ("fl", "fr", "hl", "hr")]
    entities = dict(asset_cfg=SceneEntityCfg("robot"),
                    leg_joint_cfg=SceneEntityCfg("robot", joint_names=legs, preserve_order=True),
                    wheel_joint_cfg=SceneEntityCfg("robot", joint_names=wheels, preserve_order=True))
    if profile != "idle_stop_motion_v1":
        for name, func, weight in (
            ("zero_stop_velocity", teacher.zero_stop_velocity_cost, -1.0),
            ("zero_stop_position", teacher.zero_stop_position_cost, -0.5),
            ("zero_stop_chatter", teacher.zero_stop_chatter_cost, -0.1),
        ):
            setattr(env_cfg.rewards, name, RewardTermCfg(func=func, weight=weight, params=entities.copy()))
    if uses_stop_metadata(profile):
        from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
        from isaaclab.utils import configclass

        @configclass
        class StopMetadataCfg(ObservationGroupCfg):
            raw_zero = ObservationTermCfg(func=teacher.zero_stop_metadata)
            concatenate_terms = True
            enable_corruption = False

        # Training metadata is a separate group, not another policy input.
        env_cfg.observations.stop_metadata = StopMetadataCfg()
        if profile == "v1_stop_v5":
            @configclass
            class StopControlCfg(ObservationGroupCfg):
                control = ObservationTermCfg(func=teacher.zero_stop_control_observation)
                concatenate_terms = True
                enable_corruption = False

            env_cfg.observations.stop_control = StopControlCfg()
            env_cfg.observations.stop_metadata = None  # raw-zero is already in stop_control[:, 0]
            env_cfg.scene.terrain.max_init_terrain_level = 3
            env_cfg.rewards.zero_stop_pose_hold = RewardTermCfg(
                func=teacher.zero_stop_pose_hold_cost, weight=-.3, params=entities.copy())
            env_cfg.rewards.zero_stop_settled = RewardTermCfg(
                func=teacher.zero_stop_settled_reward, weight=.5, params=entities.copy())
        if profile == "v1_stop_v4":
            @configclass
            class StopSupportCfg(ObservationGroupCfg):
                support = ObservationTermCfg(func=teacher.zero_stop_support_metadata)
                concatenate_terms = True
                enable_corruption = False

            # Privileged labels/clean teacher inputs are stored for PPO only.
            # The exported actor remains 244D, with unchanged command semantics.
            env_cfg.observations.stop_support = StopSupportCfg()
            env_cfg.observations.stop_teacher = deepcopy(env_cfg.observations.policy)
            env_cfg.observations.stop_teacher.enable_corruption = False
        if profile in ("v1_stop_v2", "v1_stop_v3_target_control"):
            env_cfg.rewards.zero_stop_wheel_target = RewardTermCfg(
                func=teacher.zero_stop_wheel_target_cost, weight=-0.75, params=entities.copy())
        else:
            env_cfg.rewards.zero_stop_wheel_target = None
    if profile == "idle_stop_motion_v1":
        from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
        from isaaclab.utils import configclass

        @configclass
        class MotionControlCfg(ObservationGroupCfg):
            raw_command = ObservationTermCfg(func=teacher.raw_velocity_command_observation)
            concatenate_terms = True
            enable_corruption = False

        env_cfg.observations.motion_control = MotionControlCfg()
    env_cfg.curriculum.zero_command_hold = None
    env_cfg.curriculum.zero_stop = CurriculumTermCfg(func=teacher.zero_stop_metrics, params={})
    if profile == "idle_stop_motion_v1":
        env_cfg.curriculum.zero_stop = None
    for name, noise in saved["policy_noise"].items():
        actual = getattr(env_cfg.observations.policy, name).noise
        actual.n_min, actual.n_max = noise["n_min"], noise["n_max"]
    env_cfg.m20_refinement_profile = profile
    settings = agent_cfg.algorithm
    settings.learning_rate, settings.entropy_coef = 1.e-5, 1.e-4
    settings.anchor_coefficient = 0.1
    settings.anchor_stair_weight, settings.anchor_stop_weight = 1.0, 0.02
    settings.anchor_stop_command_epsilon = 1.e-4
    settings.critic_warmup_iterations, settings.reference_std_multiplier = 50, 1.02
    if profile == "v1_stop_v4":
        settings.stop_teacher_coefficient = 0.1
    if profile == "v1_stop_v5":
        settings.critic_warmup_iterations = 20
        settings.learning_rate = 1.e-4
        settings.stop_regularization_coefficient = .01
    if profile == "idle_stop_motion_v1":
        settings.critic_warmup_iterations = 20
        settings.learning_rate = 1.e-4
        settings.motion_regularization_coefficient = .02
    # Keep early candidates for behavioral validation rather than only the last.
    agent_cfg.save_interval = 100
    return {"profile": profile, "version": PROFILE_VERSIONS[profile],
            "control_profile": "legacy_v1",
            "baseline_actor_sha256": (IDLE_STOP_ACTOR_SHA256 if profile == "idle_stop_motion_v1"
                                      else saved["actor_sha256"]),
            "baseline_config_sha256": hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()}


def validate_initialization(init_actor, preserve_actor, profile=PROFILE, stop_teacher=None):
    if profile == 'ascent_entry_v1_stage7':
        from rl_training.algorithms.initialization.ascent_unified_init import validate_source
        if not init_actor: raise ValueError('Stage7 requires a consolidated checkpoint')
        validate_source(init_actor, preserve_actor)
        return
    if profile == 'ascent_entry_v1_stage6':
        from rl_training.algorithms.initialization.ascent_alternating_init import validate_source
        if not init_actor: raise ValueError('Stage6 requires a validated warm start')
        validate_source(init_actor, preserve_actor)
        return
    if profile == 'ascent_entry_v1_stage5':
        from rl_training.algorithms.initialization.ascent_adaptive_init import validate_source
        if not init_actor: raise ValueError('Stage5 requires a validated warm start')
        validate_source(init_actor, preserve_actor)
        return
    if profile == 'ascent_entry_v1_stage4':
        from rl_training.algorithms.initialization.ascent_sequence_init import validate_source
        if not init_actor:
            raise ValueError('Stage4 requires a complete stage3 checkpoint')
        validate_source(init_actor, preserve_actor)
        return
    if profile == 'ascent_entry_v1_stage3':
        from rl_training.algorithms.initialization.ascent_landing_init import validate_source
        if not init_actor:
            raise ValueError('Stage3 requires a complete stage2 checkpoint')
        validate_source(init_actor, preserve_actor)
        return
    if profile == 'ascent_entry_v1_stage2':
        from rl_training.algorithms.initialization.ascent_entry_stage2_init import validate_checkpoint
        if not init_actor:
            raise ValueError('Stage2 requires --init_actor_from a complete v1 ascent checkpoint')
        validate_checkpoint(init_actor)
        if preserve_actor and Path(preserve_actor).resolve() != Path(init_actor).resolve():
            raise ValueError('Stage2 preserves its complete initialized v1 actor')
        return
    if profile == 'ascent_entry_v1':
        expected = 'db341fd21914d7b5846ea110992332d66b41a224b38ae4d032d5d665f30f2512'
        if not init_actor or hashlib.sha256(Path(init_actor).expanduser().read_bytes()).hexdigest()!=expected:
            raise ValueError('ascent_entry_v1 must inherit the recorded pre_train/teacher_baseline_.pt including its motion residual')
        if preserve_actor and Path(preserve_actor).resolve()!=Path(init_actor).resolve():
            raise ValueError('ascent_entry_v1 preserves the complete initialization checkpoint')
        return
    expected = baseline()["actor_sha256"]
    expected_init = (QUIET_ACTOR_SHA256 if profile == "v1_stop_v4" else
                     IDLE_STOP_ACTOR_SHA256 if profile == "idle_stop_motion_v1" else expected)
    actors = [("initial actor", init_actor, expected_init)]
    if profile != "idle_stop_motion_v1":
        actors.append(("preservation actor", preserve_actor or init_actor, expected))
    if profile == "v1_stop_v4":
        actors.append(("quiet stop teacher", stop_teacher or init_actor, QUIET_ACTOR_SHA256))
    for label, path, digest in actors:
        if not path or hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest() != digest:
            name = ("v1_stop/model900.pt" if digest == QUIET_ACTOR_SHA256 else
                    "pre_train/teacher_baseline_idle_stop.pt" if digest == IDLE_STOP_ACTOR_SHA256 else
                    "v1/model5100.pt")
            raise ValueError(f"M20 stop profile: {label} must be the recorded {name}")
