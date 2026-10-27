"""Physical-contact ascent objectives. Only enabled by ascent_entry_v1.

Four single-wheel filtered sensors provide the mean contact patch and summed
normal force. Mixed corner contacts are an approximation, explicitly distinct
from MuJoCo's per-contact validation. A 200 Hz observer retains short impacts.
"""
from __future__ import annotations
import json
import math
from .......utils.module_layout import source_path
from types import MethodType
import torch


def smooth01(value):
    value = value.clamp(0., 1.)
    return value.square()*(3.-2.*value)


def period_error(interval,expected):
    """A whole-cycle delay must not disappear through phase modulo aliasing."""
    error=((interval-expected).abs()-.1*expected).clamp_min(0.)/(.2*expected).clamp_min(.02)
    return error.square().clamp_max(4.)


def contact_patch_velocity(com_pos, com_velocity, angular_velocity, point, force, return_tangent=False):
    valid = torch.isfinite(point).all(-1) & torch.isfinite(force).all(-1)
    point = torch.where(valid[..., None], point, com_pos)
    norm = torch.linalg.vector_norm(force, dim=-1)
    normal = force/norm.clamp_min(1.e-6)[..., None]
    velocity = com_velocity + torch.linalg.cross(angular_velocity, point-com_pos, dim=-1)
    tangent = velocity - (velocity*normal).sum(-1, keepdim=True)*normal
    active = valid & (norm > 5.)
    support = active & (normal[..., 2] > .7) & (force[..., 2] > 5.)
    slip = torch.linalg.vector_norm(tangent, dim=-1)*active
    result = (slip, support, active & ~support, normal)
    return (*result, tangent) if return_tangent else result


class EntryPhysicsObserver:
    def __init__(self, env):
        self.env = env
        robot = env.scene['robot']
        self.wheels = robot.find_bodies([f'{leg}_wheel' for leg in ('fl','fr','hl','hr')], preserve_order=True)[0]
        n, device = env.num_envs, env.device
        self.previous_lengths = torch.zeros(n, device=device, dtype=torch.long)
        self._episode_reset_tick = -1
        self._episode_reset_ids = torch.empty(0, device=device, dtype=torch.long)
        self.highest_z = env.scene.env_origins[:, 2:3].repeat(1,4).clone()
        self.stable_time = torch.zeros(n,4,device=device)
        self.previous_top_z = self.highest_z.clone()
        self.last_landing_time = torch.full((n,4), -100., device=device)
        self.last_landing_x = torch.zeros(n,4,device=device)
        self.action_prev = torch.zeros(n,16,device=device)
        self.action_delta_prev = self.action_prev.clone()
        self.action_age = torch.zeros(n,device=device,dtype=torch.long)
        self.first_floor = env.scene.env_origins[:,2].clone()
        self.window_key, self.last_physics_tick = -1, -1
        self.window = {}
        self.stats = {k: torch.zeros(n,device=device) for k in ('physics_samples','slip_sum','face_sum','new_landings')}
        reference = json.loads(source_path('ascent_reference.json').read_text())
        self.probabilities = torch.tensor([x['support_probability'] for x in reference['references']],device=device)
        self.periods = torch.tensor([x['period_s'] for x in reference['references']],device=device)
        self.phases = torch.tensor([x['touchdown_phase'] for x in reference['references']],device=device)
        self.context_key = -1
        self.lifetime_support_samples=torch.zeros((),device=device)
        self.lifetime_landings=torch.zeros((),device=device)

    def reset(self,ids):
        self.highest_z[ids]=self.env.scene.env_origins[ids,2:3]
        self.previous_top_z[ids]=self.highest_z[ids]
        self.first_floor[ids]=self.env.scene.env_origins[ids,2]
        self.stable_time[ids]=0.
        self.last_landing_time[ids]=-100.
        self.last_landing_x[ids]=0.
        self.previous_lengths[ids]=0
        self.action_age[ids]=0
        self.action_prev[ids]=0.
        self.action_delta_prev[ids]=0.
        for value in self.stats.values(): value[ids]=0.
        for value in self.window.values(): value[ids]=0.

    def episode_reset_ids(self):
        """Share one sparse reset selection across the observer inheritance chain.

        Boolean indexing every history tensor launches repeated nonzero scans
        on CUDA, including on substeps with no resets. Select once before the
        base observer updates previous_lengths; consumers skip empty batches.
        """
        if self._episode_reset_tick != self.env._sim_step_counter:
            self._episode_reset_ids = (self.env.episode_length_buf < self.previous_lengths).nonzero().flatten()
            self._episode_reset_tick = self.env._sim_step_counter
        return self._episode_reset_ids

    def observe(self):
        env = self.env
        if env._sim_step_counter==0:
            return
        if self.last_physics_tick == env._sim_step_counter:
            return
        self.last_physics_tick = env._sim_step_counter
        reset = self.episode_reset_ids()
        self.previous_lengths.copy_(env.episode_length_buf)
        if len(reset):
            self.highest_z[reset] = env.scene.env_origins[reset, 2:3]
            self.first_floor[reset] = env.scene.env_origins[reset,2]
            self.stable_time[reset] = 0.
            self.last_landing_time[reset] = -100.
            self.action_age[reset] = 0
            for value in self.stats.values():
                value[reset] = 0.
        if self.window_key != env.common_step_counter:
            self.window_key = env.common_step_counter
            z = torch.zeros(env.num_envs,4,device=env.device)
            self.window = {k: z.clone() for k in ('slip','face','touchdown','landing_inset_cost','skip','support','period')}
            self.substeps = 0
        points, forces = [], []
        for leg in ('fl','fr','hl','hr'):
            sensor = env.scene.sensors[f'entry_{leg}_contact'].data
            force = sensor.force_matrix_w[:,0]
            index = torch.linalg.vector_norm(force, dim=-1).argmax(-1)
            row = torch.arange(env.num_envs,device=env.device)
            points.append(sensor.contact_pos_w[:,0][row,index])
            forces.append(force[row,index])
        point, force = torch.stack(points,1), torch.stack(forces,1)
        robot = env.scene['robot'].data
        slip, support, face, normal, self.contact_tangent = contact_patch_velocity(robot.body_com_pos_w[:,self.wheels],
                    robot.body_com_lin_vel_w[:,self.wheels], robot.body_ang_vel_w[:,self.wheels], point, force, return_tangent=True)
        self.contact_slip, self.contact_support = slip, support
        point = torch.nan_to_num(point)
        # Bounded pseudo-Huber: retain initial contact slip without letting a
        # single impact dominate all task progress rewards.
        self.window['slip'] += (torch.sqrt(1.+(slip/.2).square())-1.).clamp_max(8.)*support
        self.window['face'] += face*((slip/.3).square().clamp_max(4.) +
                                     (torch.linalg.vector_norm(force[...,:2],dim=-1)/150.).clamp_max(2.))
        self.window['support'] += support
        same_height = (point[...,2]-self.previous_top_z).abs() < .03
        self.stable_time = torch.where(support, torch.where(same_height,self.stable_time,0.)+env.physics_dt, 0.)
        self.previous_top_z = torch.where(support,point[...,2],self.previous_top_z)
        landed = support & (self.stable_time >= .04-1.e-6) & (point[...,2] > self.highest_z+.06)
        now = env._sim_step_counter*env.physics_dt
        command=env.command_manager.get_command('base_velocity')
        blend=((command[:,0]-.5)/.2).clamp(0.,1.)
        target_period=torch.lerp(self.periods[0],self.periods[1],blend)
        actual_interval=now-self.stable_time[:,0]+env.physics_dt-self.last_landing_time[:,0]
        self.window['period'][:,0] += period_error(actual_interval,target_period)*landed[:,0]*(self.last_landing_time[:,0]>=0.)
        self.last_landing_time = torch.where(landed, now-self.stable_time+env.physics_dt,self.last_landing_time)
        self.last_landing_x = torch.where(landed,point[...,0],self.last_landing_x)
        rise = float(getattr(env.cfg,'ascent_entry_height',.15))
        depth = float(getattr(env.cfg,'ascent_entry_depth',.30))
        if getattr(env.cfg, 'ascent_adaptive_geometry', False):
            from .ascent_adaptive import geometry
            rise, depth = (v[:, None] for v in geometry(env))
        start_x = env.scene.env_origins[:,0]+1.25
        level = torch.round((point[...,2]-self.first_floor[:,None])/rise).clamp_min(1.)
        edge_x = start_x[:,None]+(level-1.)*depth
        inset = point[...,0]-edge_x
        inset_error = ((.07-inset).clamp_min(0.) + (inset-(depth-.05)).clamp_min(0.))/.1
        self.window['landing_inset_cost'] += inset_error.square().clamp_max(4.)*landed
        self.window['skip'] += (((point[...,2]-self.highest_z)/rise-1.25).clamp_min(0.)).clamp_max(3.)*landed
        self.highest_z = torch.where(landed,point[...,2],self.highest_z)
        self.window['touchdown'] += landed
        self.lifetime_support_samples += support.float().sum()
        self.lifetime_landings += landed.float().sum()
        if env._sim_step_counter==100 and self.lifetime_support_samples.item()==0:
            raise RuntimeError('No filtered wheel/terrain support after 100 physics steps; check collider filter path')
        self.stats['physics_samples'] += 1
        self.stats['slip_sum'] += (slip*support).sum(-1)
        self.stats['face_sum'] += face.float().sum(-1)
        self.stats['new_landings'] += landed.float().sum(-1)
        self.substeps += 1
        self.support = support


def install_physics_observer(env):
    if hasattr(env, '_ascent_entry_observer'):
        return
    observer = EntryPhysicsObserver(env)
    env._ascent_entry_observer = observer
    original = env.scene.update
    def update(scene, dt):
        original(dt)
        observer.observe()
    env.scene.update = MethodType(update,env.scene)


def context(env):
    observer = getattr(env,'_ascent_entry_observer',None)
    if observer is None:
        raise RuntimeError('ascent_entry_v1 requires install_physics_observer before stepping')
    if observer.context_key == env.common_step_counter:
        return observer.cached
    robot = env.scene['robot'].data
    command = env.command_manager.get_command('base_velocity')
    from .ascent_adaptive import ascent_mask
    course = ascent_mask(env)
    forward = (command[:,0]>.1)&(command[:,1].abs()<.1)&(command[:,2].abs()<.1)
    active = course & forward
    depth, rise = float(env.cfg.ascent_entry_depth), float(env.cfg.ascent_entry_height)
    if getattr(env.cfg, 'ascent_adaptive_geometry', False):
        from .ascent_adaptive import geometry
        rise, depth = geometry(env)
    start_x = env.scene.env_origins[:,0]+1.25
    wheel_pos = robot.body_pos_w[:,observer.wheels]
    progress = (wheel_pos[:,:2,0].amax(-1)-start_x)/depth
    entry = 1.-smooth01((progress-2.)/3.)
    # Avoid any reward or branch switch at the third/fourth stair boundary.
    entry *= active
    reference_blend = ((command[:,0]-.5)/.2).clamp(0.,1.)
    period = torch.lerp(observer.periods[0],observer.periods[1],reference_blend)
    since_fl = env._sim_step_counter*env.physics_dt-observer.last_landing_time[:,0]
    valid_phase = (since_fl >= 0.) & (since_fl < 1.5*period)
    phase = torch.remainder(since_fl/period,1.)
    f = phase*200
    lo, hi = f.long().clamp(0,200), (f.long()+1).clamp(0,200)
    tables = torch.lerp(observer.probabilities[0][None],observer.probabilities[1][None],reference_blend[:,None,None])
    row = torch.arange(env.num_envs,device=env.device)
    expected = torch.lerp(tables[row,lo],tables[row,hi],(f-lo)[:,None])
    phases = torch.lerp(observer.phases[0][None],observer.phases[1][None],reference_blend[:,None])
    event_phase = torch.remainder((observer.last_landing_time-observer.last_landing_time[:,:1])/period[:,None],1.)
    difference = torch.remainder(event_phase-phases+.5,1.)-.5
    phase_event = (difference/.20).square().clamp_max(4.)*observer.window['touchdown']
    phase_cost = (phase_event.sum(-1)+observer.window['period'][:,0])/env.step_dt + .15*(expected-observer.support.float()).square().mean(-1)
    phase_cost *= valid_phase
    wheel_depth = depth[:, None] if torch.is_tensor(depth) else depth
    wheel_rise = rise[:, None] if torch.is_tensor(rise) else rise
    next_level = (torch.floor((wheel_pos[...,0]-start_x[:,None])/wheel_depth)+2.).clamp(1.,18.)
    next_edge = start_x[:,None]+(next_level-1.)*wheel_depth
    distance = next_edge-wheel_pos[...,0]
    bottom = wheel_pos[...,2]-.09
    next_top = observer.first_floor[:,None]+next_level*wheel_rise
    near = (distance<.12)&(distance>-.015)&~observer.support
    clearance_cost = (((next_top+.015-bottom)/.10).clamp(0.,2.).square()*near).sum(-1)
    action = env.action_manager.action
    delta = action-observer.action_prev
    second = delta-observer.action_delta_prev
    valid_action = observer.action_age>=2
    continuity = (second[:,:12].square().mean(-1)+.1*second[:,12:].square().mean(-1)).clamp_max(4.)*valid_action
    observer.action_prev.copy_(action)
    observer.action_delta_prev.copy_(delta)
    observer.action_age += 1
    w = observer.window
    result = dict(entry=entry, active=active.float(), phase=phase,
                  phase_cost=phase_cost*entry,
                  landing_cost=(w['landing_inset_cost']+w['skip']).sum(-1)*entry/env.step_dt,
                  landing_bonus=w['touchdown'].sum(-1)*entry/env.step_dt,
                  clearance_cost=clearance_cost*entry,
                  slip_cost=w['slip'].mean(-1)/max(observer.substeps,1)*active*(.5+.5*entry),
                  face_cost=w['face'].mean(-1)/max(observer.substeps,1)*active*(.5+.5*entry),
                  continuity_cost=continuity*entry)
    observer.context_key, observer.cached = env.common_step_counter,result
    return result


def phase_cost(env): return context(env)['phase_cost']
def landing_cost(env): return context(env)['landing_cost']
def landing_bonus(env): return context(env)['landing_bonus']
def clearance_cost(env): return context(env)['clearance_cost']
def wheel_slip_cost(env): return context(env)['slip_cost']
def riser_contact_cost(env): return context(env)['face_cost']
def entry_continuity_cost(env): return context(env)['continuity_cost']


def metrics(env, env_ids):
    observer = getattr(env,'_ascent_entry_observer',None)
    if observer is None:
        return {}
    ids = env_ids if env_ids is not None else slice(None)
    stats = observer.stats
    n = stats['physics_samples'][ids].sum().clamp_min(1.)
    return {'physics_samples':stats['physics_samples'][ids].sum(),
            'support_slip_sum_per_physics_sample':stats['slip_sum'][ids].sum()/n,
            'face_contacts_per_physics_sample':stats['face_sum'][ids].sum()/n,
            'new_landings':stats['new_landings'][ids].sum()}


def entry_finished(env):
    robot = env.scene['robot'].data
    types = env.scene.terrain.terrain_types
    # Entry courses recycle after the fifth-to-sixth transition, preserving
    # its entire window instead of terminating immediately at tread three.
    progress = robot.root_pos_w[:,0]-env.scene.env_origins[:,0]
    depth=float(env.cfg.ascent_entry_depth)
    return ((types<6)&(progress>1.25+6.*depth)) | ((types>=6)&(types<8)&(progress>1.25+18.*depth+.5))
