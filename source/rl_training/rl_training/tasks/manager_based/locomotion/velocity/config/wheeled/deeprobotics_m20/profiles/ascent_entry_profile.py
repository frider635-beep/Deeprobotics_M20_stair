"""Isolated ascent refinement profile; original profiles keep their semantics."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
import trimesh
from isaaclab.managers import RewardTermCfg, CurriculumTermCfg, TerminationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg, TerrainImporter
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import reset_root_state_uniform
import isaaclab.sim as sim_utils
from rl_training.tasks.manager_based.locomotion.velocity.mdp.ascent import ascent_entry_mdp as entry
from rl_training.tasks.manager_based.locomotion.velocity.mdp.stair_teacher import StairTeacherVelocityCommand
from rl_training.tasks.manager_based.locomotion.velocity.mdp.path import stair_path_layout as path_layout

PROFILE = 'ascent_entry_v1'
VERSION = 'm20_ascent_entry_20261020_v1'
BASELINE_SHA256 = 'db341fd21914d7b5846ea110992332d66b41a224b38ae4d032d5d665f30f2512'
CONFLICTING_ASCENT_TERMS = ('stair_front_single_support','stair_diagonal_support_symmetry','stair_front_alternation',
    'stair_up_step_completion','stair_up_rear_step_completion','stair_up_order_violation_cost',
    'stair_up_same_tread_cost','stair_up_same_tread_dwell_cost','stair_rear_style_refund',
    'stair_rear_transition_recovery','stair_front_wheel_lift','stair_rear_wheel_lift')


class ContactTerrainImporter(TerrainImporter):
    """Fixed kinematic ground enables PhysX's filtered contact-point API.

    PhysX 5.1 cannot filter contact patches against bare static colliders.
    The terrain is never moved; geometry and infinite-support behavior remain.
    """
    def import_mesh(self,name,mesh):
        super().import_mesh(name,mesh)
        sim_utils.define_rigid_body_properties(self.cfg.prim_path+'/'+name,
            sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True))


def straight_stairs(difficulty, cfg):
    length, width = cfg.size
    h, depth, count, start = cfg.step_height, cfg.tread_depth, cfg.step_count, 3.
    def box(x0,x1,z):
        return trimesh.creation.box((x1-x0,width,z+.1),
                trimesh.transformations.translation_matrix(((x0+x1)/2,width/2,(z-.1)/2)))
    meshes = [box(0,length,0.)]
    for i in range(count):
        meshes.append(box(start+i*depth,start+(i+1)*depth,(i+1)*h))
    meshes.append(box(start+count*depth,length,count*h))
    return meshes,np.array([start-1.25,width/2,0.])


@configclass
class StraightStairsCfg(SubTerrainBaseCfg):
    function = straight_stairs
    step_height: float = .15
    tread_depth: float = .30
    step_count: int = 18


class EntryVelocityCommand(StairTeacherVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if not hasattr(self,'entry_speeds'):
            self.entry_speeds = torch.zeros(self.num_envs,device=self.device)
        ids = torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        self.entry_speeds[ids] = torch.where(torch.rand(len(ids),device=self.device)<.5,.5,.7)
        self._apply_ascent_command()

    def _apply_ascent_command(self):
        mask = self._env.scene.terrain.terrain_types<8
        self.vel_command_b[mask,0] = self.entry_speeds[mask]
        self.vel_command_b[mask,1:] = 0.
        self.is_standing_env[mask] = False
        self.is_heading_env[mask] = False

    def _update_command(self):
        super()._update_command()
        if hasattr(self,'entry_speeds'):
            self._apply_ascent_command()


def reset_entry_root(env, env_ids, pose_range, velocity_range, asset_cfg=SceneEntityCfg('robot')):
    reset_root_state_uniform(env,env_ids,pose_range,velocity_range,asset_cfg)
    observer=getattr(env,'_ascent_entry_observer',None)
    if observer is not None:
        observer.reset(env_ids)
    ids = torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    ids = ids[env.scene.terrain.terrain_types[ids]<8]
    asset = env.scene[asset_cfg.name]
    pose = asset.data.default_root_state[ids,:7].clone()
    pose[:,:3] += env.scene.env_origins[ids]
    pose[:,0] += torch.empty(len(ids),device=env.device).uniform_(-.15,.15)
    pose[:,1] += torch.empty(len(ids),device=env.device).uniform_(-.03,.03)
    yaw = torch.empty(len(ids),device=env.device).uniform_(-.025,.025)
    pose[:,3:] = 0.
    pose[:,3], pose[:,6] = torch.cos(yaw/2),torch.sin(yaw/2)
    asset.write_root_pose_to_sim(pose,env_ids=ids)
    velocity = torch.zeros(len(ids),6,device=env.device)
    velocity[:,:2].uniform_(-.02,.02)
    asset.write_root_velocity_to_sim(velocity,env_ids=ids)


def apply_profile(env_cfg, agent_cfg):
    if getattr(env_cfg,'m20_refinement_profile',None)==PROFILE:
        return configure_agent(agent_cfg)
    source_terrains = env_cfg.scene.terrain.terrain_generator.sub_terrains
    env_cfg._m20_general_terrain_sources = path_layout.preserve_general_terrains(source_terrains)
    from .v1_stop_profile import apply_profile as original_profile
    original_profile(env_cfg,agent_cfg,'idle_stop_motion_v1')
    env_cfg.ascent_entry_height, env_cfg.ascent_entry_depth = .15,.30
    env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(10.,5.),num_rows=2,num_cols=10,curriculum=True,use_cache=False,
        sub_terrains={'entry_stairs':StraightStairsCfg(proportion=.6),
                      'continuous_stairs':StraightStairsCfg(proportion=.2),
                      'flat':MeshPlaneTerrainCfg(proportion=.1),
                      'pyramid_stairs_inv':env_cfg.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs_inv'].replace(proportion=.1)})
    env_cfg.scene.terrain.max_init_terrain_level=0
    env_cfg.scene.terrain.class_type=ContactTerrainImporter
    env_cfg.scene.terrain.visual_material=None
    env_cfg.scene.sky_light=None
    for leg in ('fl','fr','hl','hr'):
        setattr(env_cfg.scene,f'entry_{leg}_contact',ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/Robot/'+leg+'_wheel',filter_prim_paths_expr=['/World/ground/terrain'],
            track_contact_points=True,track_air_time=True,history_length=1,update_period=env_cfg.sim.dt,
            max_contact_data_count_per_prim=16))
    env_cfg.commands.base_velocity.class_type=EntryVelocityCommand
    env_cfg.commands.base_velocity.debug_vis=False
    env_cfg.commands.base_velocity.resampling_time_range=(8.,12.)
    env_cfg.events.randomize_reset_base.func=reset_entry_root
    env_cfg.events.randomize_reset_base.params['pose_range'].update(yaw=(-.05,.05),roll=(-.02,.02),pitch=(-.02,.02))
    # New target uses both front wheels on each tread. Remove ascent style
    # terms which explicitly reward skipping/alternating across physical treads.
    for name in CONFLICTING_ASCENT_TERMS:
        term=getattr(env_cfg.rewards,name,None)
        if term is not None:
            term.weight=0.
    for name, func, weight in (
        ('ascent_entry_phase',entry.phase_cost,-.20),
        ('ascent_entry_landing',entry.landing_cost,-.25),
        ('ascent_entry_new_tread',entry.landing_bonus,.35),
        ('ascent_entry_clearance',entry.clearance_cost,-.30),
        ('ascent_wheel_slip',entry.wheel_slip_cost,-.65),
        ('ascent_riser_contact',entry.riser_contact_cost,-.15),
        ('ascent_entry_continuity',entry.entry_continuity_cost,-.10)):
        setattr(env_cfg.rewards,name,RewardTermCfg(func=func,weight=weight))
    env_cfg.curriculum.terrain_levels=None
    for name in vars(env_cfg.curriculum):
        if name.startswith('terrain_level_'):
            setattr(env_cfg.curriculum,name,None)
    env_cfg.curriculum.ascent_entry=CurriculumTermCfg(func=entry.metrics)
    env_cfg.terminations.ascent_entry_finished=TerminationTermCfg(func=entry.entry_finished,time_out=True)
    env_cfg.terminations.terrain_out_of_bounds.params['distance_buffer']=.25
    env_cfg.episode_length_s=16.
    env_cfg.m20_refinement_profile=PROFILE
    return configure_agent(agent_cfg)


def configure_agent(agent_cfg):
    agent_cfg.algorithm.learning_rate=5.e-5
    agent_cfg.algorithm.critic_warmup_iterations=20
    agent_cfg.algorithm.anchor_coefficient=.10
    agent_cfg.algorithm.anchor_stair_weight=.15
    agent_cfg.algorithm.motion_regularization_coefficient=.02
    agent_cfg.save_interval=100
    return dict(profile=PROFILE,version=VERSION,control_profile='legacy_v1',baseline_actor_sha256=BASELINE_SHA256,
                baseline_config_sha256=hashlib.sha256((Path(__file__).parent / 'data' / 'ascent_reference.json').read_bytes()).hexdigest(),
                entry_mix={'entry':.6,'continuous':.2,'flat':.1,'descent':.1},
                incompatible_ascent_style_terms_disabled=list(CONFLICTING_ASCENT_TERMS))
