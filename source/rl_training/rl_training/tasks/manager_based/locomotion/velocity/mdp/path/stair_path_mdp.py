"""Shared stair-path feedback and measured per-direction training objectives."""
from types import MethodType
import torch
from .......control.stair_path import StairPathController
from .......utils.teacher_diagnostics import export_teacher_metrics
from ..ascent.ascent_unified_mdp import UnifiedPhysicsObserver
from ..ascent import ascent_adaptive_mdp as ascent_mdp
from ..ascent import ascent_adaptive as terrain_adaptive
from .stair_path_objectives import contact_slip_cost, return_velocity_cost
from . import stair_path_layout as layout
from .stair_path_layout import ASCENT_TASKS, DESCENT_TASKS, DESCENT_START, task_label


def controller(env):
    if not hasattr(env, '_stair_path_controller'):
        env._stair_path_controller = StairPathController(env.num_envs, env.device, env.cfg.stair_path_config)
        env._stair_path_command_step = -1
    return env._stair_path_controller


def apply_path_command(env, effective, raw, command_name='base_velocity'):
    control = controller(env)
    # Every observation/reward consumer in a tick sees the same filter update.
    if env._stair_path_command_step != env.common_step_counter:
        data = env.scene['robot'].data
        q = data.root_quat_w
        yaw = torch.atan2(2.*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                          1.-2.*(q[:, 2].square()+q[:, 3].square()))
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.envs.mdp import height_scan
        scan = height_scan(env, sensor_cfg=SceneEntityCfg('height_scanner')).clamp(-1., 1.)
        control.update(data.root_pos_w[:, :2], data.root_lin_vel_w[:, :2], yaw, raw, scan, env.step_dt)
        env._stair_path_command_step = env.common_step_counter
    result = effective.clone()
    # Manual lateral/turn commands retain ownership; a platform exit decays
    # smoothly only while the original straight request is still held.
    stair_course = terrain_adaptive.ascent_mask(env) | terrain_adaptive.descent_mask(env)
    result[:, 1] = torch.where(control.straight & stair_course, control.correction, raw[:, 1])
    return result


def path_state(env):
    from ..stair_teacher import heading_hold_velocity_commands
    heading_hold_velocity_commands(env)
    return controller(env).features


class PathPhysicsObserver(UnifiedPhysicsObserver):
    def __init__(self, env):
        super().__init__(env)
        self.path_max = torch.zeros(env.num_envs, device=env.device)
        self.path_signed = torch.zeros_like(self.path_max)
        self.path_samples = torch.zeros_like(self.path_max)
        self.path_vy_total = torch.zeros_like(self.path_max)
        self.path_slip_total = torch.zeros_like(self.path_max)
        self.path_longitudinal_slip_total = torch.zeros_like(self.path_max)
        self.path_return_error_total = torch.zeros_like(self.path_max)
        self.path_saturated = torch.zeros_like(self.path_max)
        self.down_ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.down_success = torch.zeros_like(self.down_ready)
        self.down_exit_time = torch.zeros_like(self.path_max)
        self.down_stable_time = torch.zeros_like(self.path_max)
        self.path_failed = torch.zeros_like(self.down_ready)
        self.path_time = torch.zeros_like(self.path_max)

    def reset(self, ids):
        super().reset(ids)
        for name in ('path_max','path_signed','path_samples','path_vy_total','path_slip_total',
                     'path_longitudinal_slip_total','path_return_error_total',
                     'path_saturated','down_ready','down_success','path_failed','path_time','down_exit_time','down_stable_time'):
            getattr(self, name)[ids] = 0
        if hasattr(self.env, '_stair_path_controller'):
            self.env._stair_path_controller.reset(ids)
            self.env._stair_path_command_step = -1

    def observe(self):
        env = self.env
        if env._sim_step_counter == 0 or self.last_physics_tick == env._sim_step_counter:
            return
        super().observe()
        ctrl = controller(env); data = env.scene['robot'].data
        command = env.command_manager.get_command('base_velocity')
        stairs = terrain_adaptive.ascent_mask(env) | terrain_adaptive.descent_mask(env)
        straight = (command[:, 0] > .1) & (command[:, 1:].abs().amax(-1) <= .1)
        active = stairs & straight
        q = data.root_quat_w
        yaw = torch.atan2(2.*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                          1.-2.*(q[:, 2].square()+q[:, 3].square()))
        error, speed, _, normal = ctrl.measure(data.root_pos_w[:, :2], data.root_lin_vel_w[:, :2], yaw)
        descending = active & terrain_adaptive.descent_mask(env)
        slip_cost, lateral_slip, longitudinal_slip = contact_slip_cost(
            self.contact_tangent, self.contact_support, normal, descending)
        relative_yaw = yaw-ctrl.heading
        desired_side = command[:, 0]*torch.sin(relative_yaw)+ctrl.correction*torch.cos(relative_yaw)
        velocity_cost, return_target = return_velocity_cost(error, speed, desired_side)
        self.path_time += active*env.physics_dt
        self.path_max = torch.maximum(self.path_max, error.abs()*active)
        self.path_signed = torch.where(active, error, self.path_signed)
        self.path_samples += active
        self.path_vy_total += speed.abs()*active
        self.path_slip_total += lateral_slip*active
        self.path_longitudinal_slip_total += longitudinal_slip*active
        self.path_return_error_total += (speed-return_target).abs()*active
        self.path_saturated += ctrl.saturated*active
        self.path_failed |= active & (error.abs() > .40) & (self.path_time > .5)
        # A successful tread prefix cannot override a failed path corridor.
        self.prefix_ready &= ~self.path_failed
        self.top_ready &= ~self.path_failed
        _, depth = terrain_adaptive.geometry(env)
        end = env.scene.env_origins[:, 0]+1.25+env.cfg.ascent_sequence_step_count*depth
        wheel_pos = data.body_pos_w[:, self.wheels]
        self.down_ready |= descending & (wheel_pos[..., 0].amin(-1) > end+.40)
        level = data.projected_gravity_b[:, :2].abs().sum(-1) < .20
        stable = (self.stable_time >= .04-1.e-6).all(-1)
        self.down_exit_time += self.down_ready*env.physics_dt
        joint = self.down_ready & level & stable & (data.root_lin_vel_w[:, 0] > .1)
        self.down_stable_time = torch.where(joint, self.down_stable_time+env.physics_dt, 0.)
        self.down_success = (self.down_stable_time >= .20-1.e-6) & (self.path_max <= .20) & ~self.path_failed
        values = dict(position=(((error.abs()-.025).clamp_min(0.)/.12).square().clamp_max(9.))*active,
            velocity=velocity_cost*active,
            slip=slip_cost*active,
            down_stall=((command[:, 0]-data.root_lin_vel_w[:, 0]-.10).clamp_min(0.)/.25).square().clamp_max(4.)*descending*(self.path_time > .5),
            down_complete=self.down_success.float())
        for key, value in values.items():
            name = 'path_'+key
            if name not in self.window: self.window[name] = torch.zeros_like(value)
            self.window[name] += value


def install_physics_observer(env):
    if hasattr(env, '_ascent_entry_observer'):
        if not isinstance(env._ascent_entry_observer, PathPhysicsObserver):
            raise RuntimeError('Stair path profiles need the shared path observer from the first physics step')
        return
    env._ascent_entry_observer = PathPhysicsObserver(env)
    original = env.scene.update
    def update(scene, dt):
        original(dt); env._ascent_entry_observer.observe()
    env.scene.update = MethodType(update, env.scene)


def reward(env, name):
    observer = env._ascent_entry_observer
    value = observer.window.get('path_'+name, torch.zeros(env.num_envs, device=env.device))
    divisor = max(observer.substeps, 1)*(env.step_dt if name == "down_complete" else 1.)
    return value/divisor


def failed(env):
    observer = getattr(env, '_ascent_entry_observer', None)
    return (observer.path_failed if observer is not None else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))


def descent_finished(env):
    observer = getattr(env, '_ascent_entry_observer', None)
    return ((observer.down_success | (observer.down_ready & (observer.down_exit_time > 1.)))
            if observer is not None else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))


class DescentCurriculum:
    def __init__(self, count, device):
        self.window = torch.zeros(count, 2, 8, device=device, dtype=torch.long)
        self.trials = torch.zeros(count, 2, device=device, dtype=torch.long)
        self.cursor = torch.zeros_like(self.trials)
        self.bad = torch.zeros(count, device=device, dtype=torch.long)

    def update(self, ids, success, speed):
        bucket = (speed > .6).long(); slot = self.cursor[ids, bucket]
        self.window[ids, bucket, slot] = success.long()
        self.cursor[ids, bucket] = (slot+1)%8
        self.trials[ids, bucket] = (self.trials[ids, bucket]+1).clamp_max(8)
        self.bad[ids] = torch.where(success, 0, self.bad[ids]+1)
        up = success & (self.trials[ids] >= 8).all(-1) & (self.window[ids].sum(-1) >= 6).all(-1)
        down = self.bad[ids] >= 3
        changed = ids[up | down]
        self.window[changed] = 0; self.trials[changed] = 0; self.cursor[changed] = 0; self.bad[changed] = 0
        return up, down

    def state_dict(self):
        return {name: getattr(self, name).detach().cpu().clone() for name in ('window','trials','cursor','bad')}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected): raise ValueError('Descent curriculum keys mismatch')
        for key, value in state.items():
            maximum = dict(window=1, trials=8, cursor=7, bad=2)[key]
            if value.shape != expected[key].shape or value.dtype != torch.long or (value < 0).any() or (value > maximum).any():
                raise ValueError('Invalid descent curriculum '+key)
            getattr(self, key).copy_(value.to(self.bad.device))


def descent_course(env):
    if not hasattr(env, '_stair_path_descent_curriculum'):
        env._stair_path_descent_curriculum = DescentCurriculum(env.num_envs, env.device)
    return env._stair_path_descent_curriculum


def curriculum(env, env_ids):
    course = descent_course(env); terrain = env.scene.terrain
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[terrain_adaptive.descent_mask(env)[ids]]
    observer = getattr(env, '_ascent_entry_observer', None)
    if observer is not None and len(ids) and getattr(env.cfg, 'ascent_adaptive_training', True):
        ids = ids[observer.path_samples[ids] > 0]
        up, down = course.update(ids, observer.down_success[ids], env.command_manager.get_command('base_velocity')[ids, 0])
        # TerrainImporter normally randomizes levels on promotion past the
        # hardest row. A solved staircase should remain on that hardest row.
        up &= terrain.terrain_levels[ids] < terrain.terrain_origins.shape[0]-1
        terrain.update_env_origins(ids, up, down)
    selected = terrain_adaptive.descent_mask(env)
    result = dict(terrain_level_mean=terrain.terrain_levels[selected].float().mean() if selected.any() else torch.zeros((), device=env.device))
    for offset, (depth, width) in enumerate(DESCENT_TASKS):
        column = DESCENT_START+offset
        label = task_label('descent', depth, width)
        chosen = terrain.terrain_types == column
        result['__weighted_total__/'+label+'_terrain_level_mean'] = terrain.terrain_levels[chosen].float().sum()
        result['__weighted_count__/'+label+'_terrain_level_mean'] = chosen.float().sum()
    return export_teacher_metrics(result, result.keys(), detailed=True)


def general_terrain_curriculum(env, env_ids, min_commanded_translation_m=.1):
    terrain = env.scene.terrain
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(
        env_ids, device=env.device, dtype=torch.long)
    general = terrain_adaptive.general_mask(env) | terrain_adaptive.flat_mask(env)
    ids = ids[general[ids]]
    if len(ids):
        from .. import stair_teacher
        return stair_teacher.stair_distance_terrain_levels(
            env, ids, min_commanded_translation_m=min_commanded_translation_m)
    return terrain.terrain_levels.float().mean()


def _sampling_stage(env):
    steps_per_iteration = int(getattr(env.cfg, 'stair_path_sampling_steps_per_iteration', 24))
    offset = int(getattr(env, '_stair_path_sampling_iteration_offset', 0))
    iteration = offset + int(env.common_step_counter) // max(steps_per_iteration, 1)
    return (0 if iteration < 600 else 1 if iteration < 2000 else 2), iteration


def _resample_reset_terrains(env, ids, stage):
    """Assign environments at phase boundaries, then keep each task stable within a phase."""
    if not len(ids):
        return
    terrain = env.scene.terrain
    if not hasattr(env, '_stair_path_sampling_stage'):
        env._stair_path_sampling_stage = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    ids = ids[env._stair_path_sampling_stage[ids] != stage]
    if not len(ids):
        return
    probabilities = torch.tensor(layout.staged_sampling_probabilities(stage), device=env.device)
    old_types = terrain.terrain_types[ids].clone()
    new_types = torch.multinomial(probabilities.expand(len(ids), -1).contiguous(), 1).squeeze(-1)
    changed = new_types != old_types
    env._stair_path_sampling_stage[ids] = stage
    if not changed.any():
        return
    changed_ids = ids[changed]
    old_changed = old_types[changed]
    new_changed = new_types[changed]
    terrain.terrain_types[changed_ids] = new_changed
    old_stair = ((old_changed < layout.ASCENT_COLUMNS)
                 | ((old_changed >= layout.DESCENT_START)
                    & (old_changed < layout.DESCENT_START + len(layout.DESCENT_COLUMNS))))
    new_stair = ((new_changed < layout.ASCENT_COLUMNS)
                 | ((new_changed >= layout.DESCENT_START)
                    & (new_changed < layout.DESCENT_START + len(layout.DESCENT_COLUMNS))))
    terrain.terrain_levels[changed_ids[old_stair | new_stair]] = 0
    terrain.env_origins[changed_ids] = terrain.terrain_origins[
        terrain.terrain_levels[changed_ids], new_changed]

    # Contact-prefix progress belongs to a geometry. Do not carry an easy
    # column's full-flight target onto a newly sampled narrow staircase.
    ascent_changed = (old_changed < layout.ASCENT_COLUMNS) | (new_changed < layout.ASCENT_COLUMNS)
    if ascent_changed.any():
        reset_ids = changed_ids[ascent_changed]
        state = ascent_mdp.curriculum_state(env)
        for name in ('skill', 'good', 'bad', 'completed'):
            getattr(state, name)[reset_ids] = 0

    descent_changed = ((old_changed >= layout.DESCENT_START) & (old_changed < layout.DESCENT_START + len(layout.DESCENT_COLUMNS)))
    descent_changed |= ((new_changed >= layout.DESCENT_START) & (new_changed < layout.DESCENT_START + len(layout.DESCENT_COLUMNS)))
    if descent_changed.any() and hasattr(env, '_stair_path_descent_curriculum'):
        reset_ids = changed_ids[descent_changed]
        state = env._stair_path_descent_curriculum
        for name in ('window', 'trials', 'cursor', 'bad'):
            getattr(state, name)[reset_ids] = 0


def metrics(env, env_ids):
    observer = getattr(env, '_ascent_entry_observer', None)
    if observer is None: return {}
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    terrain = env.scene.terrain; command = env.command_manager.get_command('base_velocity')
    result = {}
    for label, select in (('ascent', terrain_adaptive.ascent_mask(env)),
                          ('descent', terrain_adaptive.descent_mask(env))):
        picked = ids[select[ids] & (observer.path_samples[ids] > 0)]
        count = len(picked); samples = observer.path_samples[picked].sum()
        for name, total, denominator in (
            ('max_error_m', observer.path_max[picked].sum(), count),
            ('final_signed_error_m', observer.path_signed[picked].sum(), count),
            ('abs_vy_m_s', observer.path_vy_total[picked].sum(), samples),
            ('lateral_contact_slip_m_s', observer.path_slip_total[picked].sum(), samples),
            ('longitudinal_contact_slip_m_s', observer.path_longitudinal_slip_total[picked].sum(), samples),
            ('return_velocity_error_m_s', observer.path_return_error_total[picked].sum(), samples),
            ('feedback_saturated_fraction', observer.path_saturated[picked].sum(), samples),
            ('corridor_failure', observer.path_failed[picked].float().sum(), count)):
            result['__weighted_total__/'+label+'_'+name] = total
            result['__weighted_count__/'+label+'_'+name] = torch.as_tensor(denominator, device=env.device).float()
        if label == 'descent':
            for fast, suffix in ((False, 'vx05'), (True, 'vx07')):
                group = picked[(command[picked, 0] > .6) == fast]
                result['__weighted_total__/descent_'+suffix+'_success'] = observer.down_success[group].float().sum()
                result['__weighted_count__/descent_'+suffix+'_success'] = torch.as_tensor(len(group), device=env.device).float()
    groups = [(column, task_label('ascent', depth, width), True)
              for column, (depth, width) in enumerate(ASCENT_TASKS)]
    groups.append((len(ASCENT_TASKS), 'ascent_reference', True))
    groups.extend((DESCENT_START+offset, task_label('descent', depth, width), False)
                  for offset, (depth, width) in enumerate(DESCENT_TASKS))
    for column,label,is_ascent in groups:
        picked = ids[(terrain.terrain_types[ids] == column) & (observer.path_samples[ids] > 0)]
        success = ascent_mdp.finished(env)[picked] if is_ascent else observer.down_success[picked]
        missing = observer.stats['adaptive_missing'][picked].sum() if is_ascent else torch.zeros((), device=env.device)
        for name, total in (('success',success.float().sum()),
                            ('max_error_m',observer.path_max[picked].sum()),
                            ('missing_treads',missing)):
            result['__weighted_total__/'+label+'_'+name] = total
            result['__weighted_count__/'+label+'_'+name] = torch.as_tensor(len(picked), device=env.device).float()
    for terrain_name, columns in layout.GENERAL_COLUMN_GROUPS.items():
        selected = torch.zeros_like(terrain.terrain_types, dtype=torch.bool)
        for column in columns:
            selected |= terrain.terrain_types == column
        chosen = ids[selected[ids]]
        result['__weighted_total__/general_'+terrain_name+'_terrain_level_mean'] = terrain.terrain_levels[chosen].float().sum()
        result['__weighted_count__/general_'+terrain_name+'_terrain_level_mean'] = torch.as_tensor(
            len(chosen), device=env.device).float()
    flat = torch.zeros_like(terrain.terrain_types, dtype=torch.bool)
    for column in layout.FLAT_COLUMNS:
        flat |= terrain.terrain_types == column
    chosen = ids[flat[ids]]
    result['__weighted_total__/general_flat_terrain_level_mean'] = terrain.terrain_levels[chosen].float().sum()
    result['__weighted_count__/general_flat_terrain_level_mean'] = torch.as_tensor(len(chosen), device=env.device).float()
    if getattr(env.cfg, 'm20_refinement_profile', None) == 'ascent_entry_v1_stage9':
        stage, iteration = _sampling_stage(env)
        probabilities = layout.staged_sampling_probabilities(stage)
        result['sampling_stage'] = torch.as_tensor(stage, device=env.device, dtype=torch.float32)
        result['sampling_iteration'] = torch.as_tensor(iteration, device=env.device, dtype=torch.float32)
        result['sampling_ascent_fraction'] = torch.as_tensor(sum(probabilities[:layout.ASCENT_COLUMNS]), device=env.device)
        result['sampling_descent_fraction'] = torch.as_tensor(
            sum(probabilities[layout.DESCENT_START:layout.DESCENT_START+len(layout.DESCENT_COLUMNS)]), device=env.device)
        result['sampling_non_stair_fraction'] = torch.as_tensor(
            sum(probabilities[column] for column in (*layout.FLAT_COLUMNS, *layout.GENERAL_COLUMNS)), device=env.device)
        _resample_reset_terrains(env, ids, stage)
    return export_teacher_metrics(result, result.keys(), detailed=True)
