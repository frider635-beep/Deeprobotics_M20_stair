"""Stage6 observer and one common alternation prerequisite for positive rewards."""
from types import MethodType
import copy
import inspect
import torch
from . import ascent_adaptive_mdp as adaptive
from . import ascent_adaptive as terrain_adaptive
from .ascent_adaptive import geometry, REFERENCE_COLUMNS
from .......utils.teacher_diagnostics import export_teacher_metrics
from .ascent_alternating import SwingAlternation, BalancedContactCurriculum, CadenceTracker, dense_costs

EXTRA_NAMES = ('progress', 'stall', 'front_slip', 'rear_slip', 'lateral', 'heading', 'cadence', 'overdue', 'turn')


class AlternatingPhysicsObserver(adaptive.AdaptivePhysicsObserver):
    def __init__(self, env):
        env._ascent_adaptive_curriculum = getattr(self, 'curriculum_type', BalancedContactCurriculum)(env.num_envs, env.device)
        if not getattr(env.cfg, 'ascent_adaptive_training', True):
            env._ascent_adaptive_curriculum.skill.fill_(len(getattr(env._ascent_adaptive_curriculum, 'prefixes', (2, 4, 6, 18)))-1)
        super().__init__(env)
        self.turns = SwingAlternation(env.num_envs, env.device)
        self.cadence = CadenceTracker(env.num_envs, env.device)
        self.best_x = torch.full((env.num_envs,), -torch.inf, device=env.device)
        self.positive_gate = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.episode_goals = adaptive.curriculum_state(env).goals().clone()
        self.refined_context_key = -1
        self.combined_context_key = -1
        self.reward_gate_key = -1
        for key in ('front_swing_starts', 'rear_swing_starts', 'front_repeat_or_simultaneous',
                    'rear_repeat_or_simultaneous', 'front_slip_cost', 'rear_slip_cost',
                    'front_lateral_slip_cost', 'rear_lateral_slip_cost',
                    'front_longitudinal_slip_cost', 'rear_longitudinal_slip_cost', 'lateral_max'):
            self.stats['alternating_'+key] = torch.zeros(env.num_envs, device=env.device)

    def reset(self, ids):
        super().reset(ids)
        self.turns.reset(ids); self.cadence.reset(ids)
        self.best_x[ids] = -torch.inf; self.positive_gate[ids] = True
        self.episode_goals[ids] = adaptive.curriculum_state(self.env).goals()[ids]
        self.refined_context_key = -1
        self.combined_context_key = -1
        self.reward_gate_key = -1

    def observe(self):
        env = self.env
        if env._sim_step_counter == 0 or self.last_physics_tick == env._sim_step_counter:
            return
        resets = self.episode_reset_ids()
        if len(resets):
            self.turns.reset(resets); self.cadence.reset(resets)
            self.best_x[resets] = -torch.inf
        super().observe()
        data = env.scene['robot'].data
        command = env.command_manager.get_command('base_velocity')
        active = terrain_adaptive.ascent_mask(env) & (command[:, 0] > .1)
        active &= command[:, 1:].abs().amax(-1) < .1
        position = data.body_pos_w[:, self.wheels]
        first = env.scene.env_origins[:, 0]+1.25
        pair_x = position[..., 0].reshape(-1, 2, 2).amax(-1)
        near = pair_x >= first[:, None]-.35
        turn_options = {}
        if getattr(self.sequence, 'rear_entry_transition', False):
            grace = torch.zeros_like(near)
            grace[:,1] = self.sequence.first_side[:,1] < 0
            turn_options['repeat_grace'] = grace
            soft = torch.zeros_like(near);soft[:,1] = True
            turn_options['soft_repeat'] = soft
        turn = self.turns.update(self.support, position[..., 2],
            data.body_com_lin_vel_w[:, self.wheels, 2], near, active & ~self.sequence.failed, env.physics_dt, **turn_options)
        self.sequence.failed |= self.turns.failed & active
        if getattr(self.sequence,'rear_entry_transition',False):
            # Rear order affects gait acceptance/reward, never the physical
            # completion label used by terrain demotion and success windows.
            rear_bad = turn['soft_violation'][:,1] & active
            self.sequence.style_failed |= rear_bad
            self.sequence.rear_swing_style_failed |= rear_bad
        self.positive_gate = turn['valid'] & ~self.sequence.failed
        self.prefix_ready &= self.positive_gate
        self.top_ready &= self.positive_gate
        # Clear earlier physics-substep credits too if this policy step later
        # violates alternation. The generic wrapper also gates other positives.
        for key in ('ordered', 'dwell', 'paired', 'landing_credit', 'top_front', 'top_rear', 'top_forward'):
            self.window['adaptive_'+key] *= self.positive_gate
        height, depth = geometry(env)
        progress = (pair_x[:, 0]-first)/depth
        q = data.root_quat_w
        yaw = torch.atan2(2.*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                          1.-2.*(q[:, 2].square()+q[:, 3].square()))
        costs = dense_costs(command[:, 0], data.root_lin_vel_w[:, 0],
            data.root_pos_w[:, 1]-env.scene.env_origins[:, 1], yaw,
            self.contact_slip, self.contact_support, height, depth, progress, self.active_time,
            tangent=self.contact_tangent)
        if getattr(env.cfg, 'stair_path_merge_entry_slip', False):
            from ..path.stair_path_objectives import merged_entry_slip_multiplier
            multiplier = merged_entry_slip_multiplier(progress)
            costs['front_slip'] *= multiplier
            costs['rear_slip'] *= multiplier
        covered = self.sequence.covered.long().sum(-1)
        index = (covered-1).clamp_min(0)[..., None]
        landing_time = self.sequence.landing_time.gather(-1, index).squeeze(-1)
        cadence, overdue = self.cadence.update(covered, landing_time,
            self.turns.clock, costs['period'], active & self.positive_gate, goals=adaptive.curriculum_state(env).goals())
        x = data.root_pos_w[:, 0]
        best = torch.where(torch.isfinite(self.best_x), self.best_x, x)
        forward = torch.minimum((x-best).clamp_min(0.), env.physics_dt*1.5*command[:, 0].clamp_min(.1))
        self.best_x = torch.maximum(best, x)
        values = dict(progress=forward/env.physics_dt/command[:, 0].clamp_min(.1)*self.positive_gate,
            stall=costs['stall'], front_slip=costs['front_slip'], rear_slip=costs['rear_slip'],
            lateral=costs['lateral'], heading=costs['heading'], cadence=cadence,
            overdue=overdue, turn=(turn['failed_now'] | turn['soft_violation'].any(-1)).float())
        for name, value in values.items():
            key = 'alternating_'+name
            if key not in self.window: self.window[key] = torch.zeros_like(value)
            self.window[key] += value*active
        self.stats['alternating_front_swing_starts'] += turn['lift'][:, :2].sum(-1)
        self.stats['alternating_rear_swing_starts'] += turn['lift'][:, 2:].sum(-1)
        for i, pair in enumerate(('front', 'rear')):
            self.stats['alternating_'+pair+'_repeat_or_simultaneous'] += turn['violation'][:, i]
            self.stats['alternating_'+pair+'_slip_cost'] += costs[pair+'_slip']*active
            self.stats['alternating_'+pair+'_lateral_slip_cost'] += costs[pair+'_lateral_slip']*active
            self.stats['alternating_'+pair+'_longitudinal_slip_cost'] += costs[pair+'_longitudinal_slip']*active
        self.stats['alternating_lateral_max'] = torch.maximum(self.stats['alternating_lateral_max'],
            (data.root_pos_w[:, 1]-env.scene.env_origins[:, 1]).abs()*active)


def install_physics_observer(env):
    if hasattr(env, '_ascent_entry_observer'):
        if not isinstance(env._ascent_entry_observer, AlternatingPhysicsObserver):
            raise RuntimeError('Stage6 requires its alternation observer before stepping')
        return
    env._ascent_entry_observer = AlternatingPhysicsObserver(env)
    original = env.scene.update
    def update(scene, dt):
        original(dt); env._ascent_entry_observer.observe()
    env.scene.update = MethodType(update, env.scene)


def context(env):
    observer = env._ascent_entry_observer
    if getattr(observer, 'combined_context_key', -1) != env.common_step_counter:
        result = dict(adaptive.context(env))
        result.update({'alternating_'+name: reward(env, name) for name in EXTRA_NAMES})
        observer.combined_context_key, observer.combined_cached = env.common_step_counter, result
    return observer.combined_cached


def reward(env, name):
    observer = env._ascent_entry_observer
    if observer.refined_context_key != env.common_step_counter:
        result = {}
        for key in EXTRA_NAMES:
            value = observer.window.get('alternating_'+key, torch.zeros(env.num_envs, device=env.device))
            # cadence and turn are impulses. Other costs are physics-time means.
            result[key] = value/(env.step_dt if key in ('cadence', 'turn') else max(observer.substeps, 1))
        observer.refined_context_key = env.common_step_counter
        observer.refined_cached = result
    return observer.refined_cached[name]


def resolve_reward_params(env, source_params):
    """Resolve the original term's nested scene parameters once per environment."""
    cache = getattr(env, '_alternating_resolved_reward_params', None)
    if cache is None:
        cache = env._alternating_resolved_reward_params = set()
    if id(source_params) not in cache:
        for value in source_params.values():
            if hasattr(value, 'resolve'):
                value.resolve(env.scene)
        cache.add(id(source_params))


def positive_reward(env, source_func, source_params, require_progress=False, require_clean_landing=False):
    """Preserve original parameter resolution and gate every positive ascent term.

    Isaac resolves SceneEntityCfg only at the top level of term.params. Resolve
    nested original parameters once, when the simulated scene is available.
    The wrapper and its original callable have stable, YAML-serializable names.
    """
    resolve_reward_params(env, source_params)
    value = source_func(env, **source_params)
    observer = env._ascent_entry_observer
    step = getattr(env, 'common_step_counter', None)
    if step is None or getattr(observer, 'reward_gate_key', -1) != step:
        command = env.command_manager.get_command('base_velocity')
        ascent = terrain_adaptive.ascent_mask(env) & (command[:, 0] > .1)
        ascent &= command[:, 1:].abs().amax(-1) < .1
        observer.reward_allowed = ~ascent | observer.positive_gate
        # Compute once per policy tick, shared by both tracking terms.
        if hasattr(observer, 'active_time'):
            speed = env.scene['robot'].data.root_lin_vel_w[:, 0]
            moving = (speed/(.6*command[:, 0].clamp_min(.1))).clamp(0., 1.)
            progress_required = ascent
            age = observer.active_time
            if hasattr(observer, "path_time"):
                descent = terrain_adaptive.descent_mask(env) & (command[:, 0] > .1)
                descent &= command[:, 1:].abs().amax(-1) < .1
                progress_required = ascent | descent
                age = torch.where(descent, observer.path_time, age)
                observer.reward_allowed &= ~observer.path_failed
            grace = (age/.5).clamp(0., 1.)
            observer.moving_factor = torch.where(progress_required, 1.-grace+grace*moving, torch.ones_like(moving))
        observer.reward_gate_key = step
    value = torch.where(observer.reward_allowed | (value <= 0.), value, torch.zeros_like(value))
    quality=getattr(observer,'landing_quality',None)
    if require_clean_landing and quality is not None:
        ascent=terrain_adaptive.ascent_mask(env)
        clean=~quality.scrubbed.any((1,2))
        value=torch.where(~ascent | clean | (value <= 0.),value,torch.zeros_like(value))
    if require_progress:
        # A stationary robot must not collect the old positive heading reward.
        value = torch.where(value > 0., value*observer.moving_factor, value)
    return value


def is_positive_reward_gate(func):
    if func is positive_reward:
        return True
    if inspect.isclass(func) or (callable(func) and not inspect.isroutine(func)):
        from .ascent_alternating_reward import AlternatingPositiveReward
        return func is AlternatingPositiveReward or isinstance(func, AlternatingPositiveReward)
    return False


def gate_positive_rewards(rewards):
    """Wrap resolved positive-weight terms, including tracking and forward motion."""
    result = []
    for name, term in vars(rewards).items():
        if term is None or not hasattr(term, 'weight') or term.weight <= 0.:
            continue
        if not is_positive_reward_gate(term.func):
            wrapper = positive_reward
            if inspect.isclass(term.func):
                from .ascent_alternating_reward import AlternatingPositiveReward
                wrapper = AlternatingPositiveReward
            term.params = dict(source_func=term.func, source_params=copy.deepcopy(term.params),
                               require_progress=name in ('track_lin_vel_xy_exp', 'track_ang_vel_z_exp'))
            term.func = wrapper
        result.append(name)
    return sorted(result)


def audit_positive_rewards(rewards):
    names = []
    for name, term in vars(rewards).items():
        if term is not None and hasattr(term, 'weight') and term.weight > 0.:
            if not is_positive_reward_gate(term.func):
                raise ValueError('Ungated stage6 positive reward: '+name)
            if name in ('track_lin_vel_xy_exp', 'track_ang_vel_z_exp') and not term.params.get('require_progress'):
                raise ValueError('Stage6 tracking reward permits idle collection: '+name)
            names.append(name)
    return sorted(names)


def curriculum(env, env_ids):
    state = adaptive.curriculum_state(env)
    state.speed_bucket.copy_((env.command_manager.get_command('base_velocity')[:, 0] > .6).long())
    result = adaptive.curriculum(env, env_ids)
    ids = torch.arange(env.num_envs, device=env.device)
    ids = ids[terrain_adaptive.ascent_mask(env)]
    for bucket, label in ((0, 'vx05'), (1, 'vx07')):
        counts = state.trials[ids, bucket].sum().clamp_min(1)
        result[label+'_window_success'] = state.window[ids, bucket].sum()/counts
    reference = torch.zeros(len(ids), device=env.device, dtype=torch.bool)
    for column in terrain_adaptive.reference_columns(env): reference |= env.scene.terrain.terrain_types[ids] == column
    probe = state.geometry_probe[ids] & ~reference
    for mask, label in ((probe, 'geometry_probe'), (~probe, 'full_flight_group')):
        count = mask.sum().clamp_min(1)
        result[label+'_fraction'] = mask.float().sum()/max(len(ids), 1)
        result[label+'_terrain_level_mean'] = (env.scene.terrain.terrain_levels[ids]*mask).sum()/count
    if hasattr(state,'mastered'):
        rows=env.scene.terrain.terrain_levels[ids]
        learned=state.mastered[ids,rows]
        protected=state.protected_skill(rows,ids)
        result['protected_prefix_mean']=state.skill.new_tensor(state.prefixes)[protected].float().mean()
        result['certified_environment_fraction']=(learned>=0).float().mean()
        result['certified6_environment_fraction']=(learned>=state.prefixes.index(6)).float().mean()
    if hasattr(state,'quality_window'):
        passes=state.window[ids].sum(-1)
        quality=(state.quality_window[ids]*state.window[ids,:,:,None]).sum(2)/passes.clamp_min(1)[:,:,None]
        for axis,label in enumerate(('front','rear')):
            result[label+'_pair_quality_window_mean']=quality[:,:,axis].mean()
        floor=state.protected_skill(env.scene.terrain.terrain_levels[ids],ids)
        result['at_protected_floor_fraction']=(state.skill[ids] <= floor).float().mean()
    return export_teacher_metrics(result, result.keys(), detailed=True)


def metrics(env, env_ids):
    observer = env._ascent_entry_observer
    ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is not None: ids = ids[env_ids]
    ids = ids[terrain_adaptive.ascent_mask(env)[ids]]
    count = max(len(ids), 1)
    result = {key: observer.stats['alternating_'+key][ids].sum()/count
              for key in ('front_swing_starts', 'rear_swing_starts', 'front_repeat_or_simultaneous', 'rear_repeat_or_simultaneous',
                          'front_lateral_slip_cost', 'rear_lateral_slip_cost',
                          'front_longitudinal_slip_cost', 'rear_longitudinal_slip_cost')}
    result['positive_gate_fraction'] = observer.positive_gate[ids].float().sum()/count
    # Episode maxima are retained, alongside the inherited mean path diagnostic.
    result['lateral_episode_max_mean_m'] = observer.stats['alternating_lateral_max'][ids].sum()/count
    state = adaptive.curriculum_state(env)
    command = env.command_manager.get_command('base_velocity')
    # Curriculum promotion happens before this diagnostic term in the manager.
    # Classify the completed episode by its original prefix, not the new skill.
    goals = observer.episode_goals
    success = observer.prefix_ready & ((goals < env.cfg.ascent_sequence_step_count) | observer.top_ready)
    for bucket, label in ((False, 'vx05'), (True, 'vx07')):
        select = ids[(command[ids, 0] > .6) == bucket]
        result['__weighted_total__/'+label+'_prefix_success'] = success[select].float().sum()
        result['__weighted_count__/'+label+'_prefix_success'] = torch.as_tensor(len(select), device=env.device).float()
        for steps in getattr(state, 'prefixes', (2, 4, 6, 18)):
            select_skill = select[goals[select] == steps]
            result['__weighted_total__/'+label+'_prefix'+str(steps)+'_success'] = success[select_skill].float().sum()
            result['__weighted_count__/'+label+'_prefix'+str(steps)+'_success'] = torch.as_tensor(len(select_skill), device=env.device).float()
    return export_teacher_metrics(result, result.keys(), detailed=True)
