"""Stage5 contact-first observer, geometry curriculum and bounded prefix tasks."""
from types import MethodType
import torch
from .ascent_entry_mdp import smooth01
from .ascent_entry_stage2 import Stage2PhysicsObserver
from .ascent_sequence import TopTransition, support_margin_cost
from .ascent_sequence_mdp import orientation
from . import ascent_adaptive as adaptive
from .ascent_adaptive import StrictSequenceLedger, ContactCurriculum, geometry, landing_guidance, REFERENCE_COLUMNS, DEADLINES


def curriculum_state(env):
    if not hasattr(env, '_ascent_adaptive_curriculum'):
        state = ContactCurriculum(env.num_envs, env.device)
        if not getattr(env.cfg, 'ascent_adaptive_training', True):
            state.skill.fill_(3)
        env._ascent_adaptive_curriculum = state
    return env._ascent_adaptive_curriculum


class AdaptivePhysicsObserver(Stage2PhysicsObserver):
    def __init__(self, env):
        super().__init__(env)
        self.sequence = getattr(self, 'ledger_type', StrictSequenceLedger)(env.num_envs, env.device, targets=env.cfg.ascent_sequence_step_count)
        self.top = TopTransition(env.num_envs, env.device)
        self.top_ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.top_posture = torch.zeros(env.num_envs, device=env.device)
        self.top_blend = torch.zeros_like(self.top_posture)
        self.prefix_ready = torch.zeros_like(self.top_ready)
        self.previous_vx = torch.zeros_like(self.top_posture)
        self.speed_age = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.active_time = torch.zeros_like(self.top_posture)
        self.action_history = torch.zeros(env.num_envs, 16, device=env.device)
        self.action_delta = torch.zeros_like(self.action_history)
        self.adaptive_action_age = torch.zeros_like(self.speed_age)
        self.adaptive_context_key = -1
        self.adaptive_action_step = -1
        self.adaptive_smoothness = torch.zeros_like(self.top_posture)
        curriculum_state(env)
        for key in ('pairs', 'ordered', 'missing', 'early', 'failure', 'landing_credit',
                    'height_total', 'opportunity', 'top_front', 'top_rear',
                    'FL2_support_frames', 'FR1_support_frames', 'FL2_longest_support_s', 'FR1_longest_support_s'):
            self.stats['adaptive_'+key] = torch.zeros_like(self.top_posture)

    def reset(self, ids):
        super().reset(ids)
        self.sequence.reset(ids); self.top.reset(ids)
        for name in ('top_ready', 'top_posture', 'top_blend', 'prefix_ready', 'previous_vx',
                     'speed_age', 'active_time', 'action_history', 'action_delta', 'adaptive_action_age', 'adaptive_smoothness'):
            getattr(self, name)[ids] = 0
        self.adaptive_context_key = -1

    def observe(self):
        env = self.env
        if env._sim_step_counter == 0 or self.last_physics_tick == env._sim_step_counter:
            return
        resets = env.episode_length_buf < self.previous_lengths
        self.sequence.reset(resets); self.top.reset(resets)
        self.speed_age[resets] = 0; self.active_time[resets] = 0.; self.adaptive_action_age[resets] = 0
        super().observe()
        data = env.scene['robot'].data
        command = env.command_manager.get_command('base_velocity')
        active = adaptive.ascent_mask(env) & (command[:, 0] > .1)
        active &= command[:, 1:].abs().amax(-1) < .1
        height, depth = geometry(env)
        position = data.body_pos_w[:, self.wheels]
        level = torch.round((self.previous_top_z-self.first_floor[:, None])/height[:, None]).long()
        first = env.scene.env_origins[:, 0]+1.25
        goals = curriculum_state(env).goals()
        self.sequence.depth = depth[:, None, None]
        blend = ((command[:, 0]-.5)/.2).clamp(0., 1.)
        period = torch.lerp(self.periods[0], self.periods[1], blend)*(depth/.30)
        period *= (.5/command[:, 0].clamp_min(.3)).clamp(1., 1.67)
        events = self.sequence.update_strict(level, self.support, position[..., 0], first,
                                             active, env.physics_dt, goals, period)
        self.active_time += active*env.physics_dt
        deadline = self.active_time.new_tensor(getattr(curriculum_state(env), 'deadlines', DEADLINES))[curriculum_state(env).skill]
        expired = active & ~self.sequence.failed & (self.active_time > deadline)
        self.sequence.failed |= expired
        events['failure'] = torch.maximum(events['failure'], expired.float())
        events['valid'] &= ~expired
        for name in ('ordered', 'dwell'): events[name] *= (~expired)[:, None]
        events['paired'] *= ~expired
        events['new_mask'] &= (~expired)[:, None, None]
        target, enabled, pair_target = self.sequence.front_targets(active)
        enabled &= target <= goals[:, None]
        guide = landing_guidance(position[:, :2], data.body_com_lin_vel_w[:, self.wheels[:2], 2],
                                 target, enabled, first, self.first_floor, depth, height)
        credit = self.sequence.landing_credit(target, guide['error'], guide['inside'], events['new_mask'])
        valid = events['valid']
        required = self.sequence.targets <= goals[:, None, None]
        pair_done = (self.sequence.pair_paid | ~required[:, 0]).all(-1)
        all_wheels = (self.sequence.covered | ~required).all((1, 2))
        self.prefix_ready = active & ~self.sequence.failed & pair_done & all_wheels
        com = getattr(data, 'root_com_pos_w', data.root_pos_w)
        balance, _ = support_margin_cost(position[..., :2], self.support, com[:, :2], data.root_lin_vel_w[:, :2])
        stable = self.support & (self.sequence.duration >= .04-1.e-6)
        balance += .25*((~self.support[:, :2]).all(-1) & (stable.sum(-1) < 3))
        # Per-environment geometry is also used through the platform transition.
        coordinate = (position[..., 0]-first[:, None])/depth[:, None]+.5
        index = torch.floor(coordinate)
        ground_z = (index+smooth01(coordinate-index)).clamp(0., env.cfg.ascent_sequence_step_count)*height[:, None]
        dx = (position[:, :2, 0].mean(-1)-position[:, 2:, 0].mean(-1)).clamp_min(.25)
        slope = torch.atan2(ground_z[:, :2].mean(-1)-ground_z[:, 2:].mean(-1), dx)
        q = data.root_quat_w
        body_angle = torch.asin((2.*(q[:, 1]*q[:, 3]-q[:, 0]*q[:, 2])).clamp(-1., 1.))
        roll_sine = data.projected_gravity_b[:, 1]
        top_edge = first+(env.cfg.ascent_sequence_step_count-1)*depth
        self.top_blend = smooth01((position[:, :2, 0].amin(-1)-top_edge+.25)/.35)*active
        self.top_posture = (((body_angle-slope)/.25).square()+(.5*roll_sine/.20).square()).clamp_max(8.)
        on_top = stable & (level == env.cfg.ascent_sequence_step_count) & (position[..., 0] > top_edge[:, None]+.07)
        full_task = goals == env.cfg.ascent_sequence_step_count
        top = self.top.update(on_top, data.root_pos_w[:, 0], body_angle.abs()+roll_sine.abs(),
                              data.root_lin_vel_w[:, 0], valid & full_task, env.physics_dt)
        self.top_ready = top['ready'] & self.prefix_ready
        vx = data.root_lin_vel_w[:, 0]
        acceleration = ((vx-self.previous_vx)/env.physics_dt).clamp_min(0.)
        surge = ((vx-command[:, 0]-.15).clamp_min(0.)/.25).square().clamp_max(4.)
        surge += .1*((acceleration-2.).clamp_min(0.)/3.).square().clamp_max(4.)*(self.speed_age > 0)
        self.previous_vx.copy_(vx); self.speed_age += 1
        if self.adaptive_action_step != env.common_step_counter:
            action = env.action_manager.action
            delta = action-self.action_history
            second = delta-self.action_delta
            self.adaptive_smoothness = (second[:, :12].square().mean(-1)+.1*second[:, 12:].square().mean(-1)).clamp_max(4.)
            self.adaptive_smoothness *= self.adaptive_action_age >= 2
            self.action_history.copy_(action); self.action_delta.copy_(delta)
            self.adaptive_action_age += 1; self.adaptive_action_step = env.common_step_counter
        weights = position.new_tensor([1.5, 1.5, 1., 1.])
        values = dict(ordered=(events['ordered']*weights).sum(-1),
            missing=(events['missing']*weights).sum(-1), dwell=(events['dwell']*weights).sum(-1),
            paired=events['paired'], failure=events['failure'], early=events['early_advance'],
            cadence=events['cadence']*valid, landing_credit=credit,
            prepare=guide['prepare'], height=guide['height'], horizontal=guide['horizontal'],
            velocity=guide['velocity'], balance=balance*active, surge=surge*active,
            smoothness=self.adaptive_smoothness*active,
            top_front=top['front'], top_rear=top['rear'], top_forward=top['forward'])
        for key, value in values.items():
            name = 'adaptive_'+key
            if name not in self.window: self.window[name] = torch.zeros_like(value)
            self.window[name] += value
        for name, value in dict(pairs=events['paired'], ordered=events['ordered'].sum(-1),
            missing=events['missing'].sum(-1), early=events['early_advance'], failure=events['failure'],
            landing_credit=credit, height_total=(guide['height_error'].abs()*guide['opportunity']).sum(-1),
            opportunity=guide['opportunity'].sum(-1), top_front=top['front'], top_rear=top['rear']).items():
            self.stats['adaptive_'+name] += value
        for name, wheel, tread in [('FL2', 0, 2), ('FR1', 1, 1)]:
            supporting = self.support[:, wheel] & (level[:, wheel] == tread) & active
            self.stats['adaptive_'+name+'_support_frames'] += supporting.float()
            longest = 'adaptive_'+name+'_longest_support_s'
            self.stats[longest] = torch.maximum(self.stats[longest], self.sequence.duration[:, wheel]*supporting)


def install_physics_observer(env):
    if hasattr(env, '_ascent_entry_observer'):
        if not isinstance(env._ascent_entry_observer, AdaptivePhysicsObserver):
            raise RuntimeError('Stage5 requires its observer from the first physics step')
        return
    env._ascent_entry_observer = AdaptivePhysicsObserver(env)
    original = env.scene.update
    def update(scene, dt):
        original(dt); env._ascent_entry_observer.observe()
    env.scene.update = MethodType(update, env.scene)


def context(env):
    observer = env._ascent_entry_observer
    if observer.adaptive_context_key == env.common_step_counter:
        return observer.adaptive_cached
    events = ('ordered', 'missing', 'dwell', 'paired', 'failure', 'early', 'cadence',
              'landing_credit', 'top_front', 'top_rear', 'top_forward')
    names = (*events, 'prepare', 'height', 'horizontal', 'velocity', 'balance', 'surge', 'smoothness')
    zero = torch.zeros(env.num_envs, device=env.device)
    result = {k: observer.window.get('adaptive_'+k, zero)/(env.step_dt if k in events else max(observer.substeps, 1)) for k in names}
    observer.adaptive_context_key, observer.adaptive_cached = env.common_step_counter, result
    return result


def reward(env, name): return context(env)[name]


def critic_state(env):
    """Expose task/geometry state only to the training critic (12 values)."""
    height, depth = geometry(env)
    state = curriculum_state(env)
    prefix = state.goals().float()/18.
    observer = getattr(env, '_ascent_entry_observer', None)
    history = torch.zeros(env.num_envs, 8, device=env.device)
    if observer is not None:
        history[:, :4] = observer.sequence.covered.float().sum(-1)/18.
        history[:, 4] = observer.sequence.pair_paid.float().sum(-1)/18.
        history[:, 5] = observer.sequence.failed.float()
        history[:, 6] = observer.active_time/24.
        history[:, 7] = observer.prefix_ready.float()
    skill_scale = max(len(getattr(state, 'prefixes', (2, 4, 6, 18)))-1, 1)
    result = torch.cat((torch.stack((height/.25, depth/.35, prefix, state.skill.float()/skill_scale), -1), history), -1)
    return result*adaptive.ascent_mask(env)[:, None]


def failed(env):
    observer = getattr(env, '_ascent_entry_observer', None)
    if observer is None: return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return observer.sequence.failed & adaptive.ascent_mask(env)


def finished(env):
    observer = getattr(env, '_ascent_entry_observer', None)
    if observer is None: return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    # Short prefixes are real terminals, not timeouts that bootstrap an
    # imaginary continuation. Full courses retain the complete platform exit.
    goals = curriculum_state(env).goals()
    return observer.prefix_ready & ((goals < env.cfg.ascent_sequence_step_count) | observer.top_ready)


def curriculum(env, env_ids):
    state = curriculum_state(env)
    observer = getattr(env, '_ascent_entry_observer', None)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    terrain = env.scene.terrain
    ids = ids[adaptive.ascent_mask(env)[ids]]
    if observer is not None and len(ids):
        reference = torch.zeros(len(ids), device=env.device, dtype=torch.bool)
        for column in adaptive.reference_columns(env): reference |= terrain.terrain_types[ids] == column
        success = finished(env)[ids]
        # The environment computes curriculum immediately before the reset
        # event. Reward/contact histories still belong to the old geometry.
        eligible = observer.stats['physics_samples'][ids] > 0
        up, down = state.update(ids, success, eligible, terrain.terrain_levels[ids], reference,
                                getattr(env.cfg, 'ascent_adaptive_training', True))
        terrain.update_env_origins(ids, up, down)
    height, depth = geometry(env)
    select = adaptive.ascent_mask(env)
    mean = lambda x: x[select].float().mean() if select.any() else torch.zeros((), device=env.device)
    return dict(skill_mean=mean(state.skill), terrain_level_mean=mean(terrain.terrain_levels),
                step_height_m=mean(height), tread_depth_m=mean(depth))


def metrics(env, env_ids):
    observer = getattr(env, '_ascent_entry_observer', None)
    if observer is None: return {}
    ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is not None: ids = ids[env_ids]
    ids = ids[adaptive.ascent_mask(env)[ids]]
    count = max(len(ids), 1); ledger = observer.sequence
    result = {key: observer.stats['adaptive_'+key][ids].sum()/count
              for key in ('pairs', 'ordered', 'missing', 'early', 'failure', 'landing_credit')}
    for wheel, index, tread in [('FL2', 0, 1), ('FR1', 1, 0)]:
        result[wheel+'_covered'] = ledger.covered[ids, index, tread].float().sum()/count
        result[wheel+'_best_dwell_s'] = ledger.best_dwell[ids, index, tread].sum()*.04/count
        result[wheel+'_support_frames'] = observer.stats['adaptive_'+wheel+'_support_frames'][ids].sum()/count
        result[wheel+'_longest_support_ms'] = observer.stats['adaptive_'+wheel+'_longest_support_s'][ids].sum()*1000./count
    goals = curriculum_state(env).goals()[ids]
    required = ledger.targets <= goals[:, None, None]
    denominator = goals.sum().clamp_min(1.)
    result['front_current_prefix_coverage'] = (ledger.covered[ids, :2]*required).float().sum()/(2.*denominator)
    result['front_same_tread_sequential_current_prefix'] = (ledger.pair_paid[ids]*required[:, 0]).float().sum()/denominator
    result['prefix_success_fraction'] = finished(env)[ids].float().sum()/count
    result['front_first_four_coverage'] = ledger.covered[ids, :2, :4].float().sum()/(count*8)
    result['front_same_tread_sequential_first_four'] = ledger.pair_paid[ids, :4].float().sum()/(count*4)
    full_four = goals >= 4
    result['__weighted_total__/front_first_four_coverage_opportunity'] = ledger.covered[ids[full_four], :2, :4].float().sum()
    result['__weighted_count__/front_first_four_coverage_opportunity'] = full_four.sum()*8.
    result['__weighted_total__/target_height_error_abs_m'] = observer.stats['adaptive_height_total'][ids].sum()
    result['__weighted_count__/target_height_error_abs_m'] = observer.stats['adaptive_opportunity'][ids].sum()
    return result
