"""Stage4 physics observer: paired fronts, adaptive support, and platform exit."""
from types import MethodType
import torch
from .ascent_entry_mdp import smooth01
from .ascent_adaptive import ascent_column_count, ascent_mask
from .ascent_entry_stage2 import Stage2PhysicsObserver
from .ascent_sequence import SequenceLedger, TopTransition, front_guidance, support_margin_cost, terrain_pitch_target


class SequencePhysicsObserver(Stage2PhysicsObserver):
    def __init__(self, env):
        super().__init__(env)
        self.sequence = SequenceLedger(env.num_envs, env.device, env.cfg.ascent_entry_depth,
                                       env.cfg.ascent_sequence_step_count)
        self.top = TopTransition(env.num_envs, env.device)
        self.sequence_context_key = -1
        self.top_ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.top_posture = torch.zeros(env.num_envs, device=env.device)
        self.top_blend = torch.zeros_like(self.top_posture)
        self.previous_vx = torch.zeros(env.num_envs, device=env.device)
        self.speed_age = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.sequence_action_prev = torch.zeros(env.num_envs, 16, device=env.device)
        self.sequence_action_delta = torch.zeros_like(self.sequence_action_prev)
        self.sequence_action_age = torch.zeros_like(self.speed_age)
        self.sequence_action_step = -1
        self.sequence_action_smooth = torch.zeros(env.num_envs, device=env.device)
        for name in ('seq_pairs', 'seq_ordered', 'seq_missing', 'seq_early', 'seq_simultaneous',
                     'seq_height_total', 'seq_opportunity', 'seq_top_front', 'seq_top_rear'):
            self.stats[name] = torch.zeros(env.num_envs, device=env.device)

    def reset(self, ids):
        super().reset(ids)
        self.sequence.reset(ids); self.top.reset(ids)
        self.previous_vx[ids] = 0.; self.speed_age[ids] = 0
        self.top_ready[ids] = False; self.top_posture[ids] = 0.; self.top_blend[ids] = 0.
        self.sequence_action_prev[ids] = 0.; self.sequence_action_delta[ids] = 0.
        self.sequence_action_age[ids] = 0; self.sequence_action_smooth[ids] = 0.
        self.sequence_context_key = -1

    def observe(self):
        env = self.env
        if env._sim_step_counter == 0 or self.last_physics_tick == env._sim_step_counter:
            return
        resets = env.episode_length_buf < self.previous_lengths
        self.sequence.reset(resets); self.top.reset(resets); self.speed_age[resets] = 0
        self.sequence_action_age[resets] = 0
        super().observe()
        data = env.scene['robot'].data
        command = env.command_manager.get_command('base_velocity')
        active = ascent_mask(env) & (command[:, 0] > .1)
        active &= command[:, 1:].abs().amax(-1) < .1
        position = data.body_pos_w[:, self.wheels]
        level = torch.round((self.previous_top_z-self.first_floor[:, None])/env.cfg.ascent_entry_height).long()
        first = env.scene.env_origins[:, 0]+1.25
        blend = ((command[:, 0]-.5)/.2).clamp(0., 1.)
        period = torch.lerp(self.periods[0], self.periods[1], blend)
        period *= (.5/command[:, 0].clamp_min(.3)).clamp(1., 1.67)
        events = self.sequence.update(level, self.support, position[..., 0], first, active, env.physics_dt, period)
        unresolved = ~(self.sequence.covered[:, :2] | self.sequence.missed[:, :2])
        target = unresolved.long().argmax(-1)+1
        previous = (target-2).clamp_min(0)
        mate_resolved = (self.sequence.covered[:, [1, 0]] | self.sequence.missed[:, [1, 0]])
        ready = (target == 1) | mate_resolved.gather(-1, previous[:, :, None]).squeeze(-1)
        enabled = unresolved.any(-1) & ready & active[:, None]
        guide = front_guidance(position[:, :2], data.body_com_lin_vel_w[:, self.wheels[:2], 2],
                               target, enabled, first, self.first_floor,
                               env.cfg.ascent_entry_depth, env.cfg.ascent_entry_height)
        credit = self.sequence.approach_credit(target, guide['error'], guide['opportunity'], guide['after_edge'])
        com = getattr(data, 'root_com_pos_w', data.root_pos_w)
        balance, margin = support_margin_cost(position[..., :2], self.support,
                                              com[:, :2], data.root_lin_vel_w[:, :2])
        # At least two stable supporting wheels are encouraged during transfer;
        # no condition compares front and rear physical tread indices.
        stable = self.support & (self.sequence.duration >= .04-1.e-6)
        transfer = ((~self.support[:, :2]).all(-1) & (stable.sum(-1) < 3)).float()
        balance = (balance+.25*transfer)*active
        depth, rise, count = env.cfg.ascent_entry_depth, env.cfg.ascent_entry_height, env.cfg.ascent_sequence_step_count
        slope = terrain_pitch_target(position, first, depth, rise, count)
        q = data.root_quat_w
        forward_z = 2.*(q[:, 1]*q[:, 3]-q[:, 0]*q[:, 2])
        body_angle = torch.asin(forward_z.clamp(-1., 1.))
        roll_sine = data.projected_gravity_b[:, 1]
        top_edge = first+(count-1)*depth
        top_blend = smooth01((position[:, :2, 0].amin(-1)-top_edge+.25)/.35)*active
        self.top_posture = (((body_angle-slope)/.25).square()+(.5*roll_sine/.20).square()).clamp_max(8.)
        self.top_blend = top_blend
        on_top = stable & (level == count) & (position[..., 0] > top_edge[:, None]+.07)
        top = self.top.update(on_top, data.root_pos_w[:, 0], body_angle.abs()+roll_sine.abs(),
                              data.root_lin_vel_w[:, 0], active, env.physics_dt)
        self.top_ready = top['ready']
        vx = data.root_lin_vel_w[:, 0]
        acceleration = ((vx-self.previous_vx)/env.physics_dt).clamp_min(0.)
        surge = ((vx-command[:, 0]-.15).clamp_min(0.)/.25).square().clamp_max(4.)
        surge += .10*((acceleration-2.).clamp_min(0.)/3.).square().clamp_max(4.)*(self.speed_age > 0)
        self.previous_vx.copy_(vx); self.speed_age += 1
        if self.sequence_action_step != env.common_step_counter:
            action = env.action_manager.action
            delta = action-self.sequence_action_prev
            second = delta-self.sequence_action_delta
            self.sequence_action_smooth = (second[:, :12].square().mean(-1)+
                                            .1*second[:, 12:].square().mean(-1)).clamp_max(4.)
            self.sequence_action_smooth *= self.sequence_action_age >= 2
            self.sequence_action_prev.copy_(action); self.sequence_action_delta.copy_(delta)
            self.sequence_action_age += 1; self.sequence_action_step = env.common_step_counter
        weights = position.new_tensor([1.5, 1.5, 1., 1.])
        values = {'ordered': (events['ordered']*weights).sum(-1),
                  'missing': (events['missing']*weights).sum(-1),
                  'dwell': (events['dwell']*weights).sum(-1),
                  'paired': events['paired'], 'simultaneous': events['simultaneous'],
                  'early': events['early_advance'], 'cadence': events['cadence'],
                  'approach': credit, 'height': guide['height'], 'horizontal': guide['horizontal'],
                  'velocity': guide['velocity'], 'balance': balance, 'surge': surge*active,
                  'smoothness': self.sequence_action_smooth*active,
                  'top_front': top['front'], 'top_rear': top['rear'], 'top_forward': top['forward']}
        for key, value in values.items():
            name = 'seq_'+key
            if name not in self.window: self.window[name] = torch.zeros_like(value)
            self.window[name] += value
        self.stats['seq_pairs'] += events['paired']; self.stats['seq_early'] += events['early_advance']
        self.stats['seq_simultaneous'] += events['simultaneous']
        self.stats['seq_ordered'] += events['ordered'].sum(-1); self.stats['seq_missing'] += events['missing'].sum(-1)
        self.stats['seq_height_total'] += (guide['height_error'].abs()*guide['opportunity']).sum(-1)
        self.stats['seq_opportunity'] += guide['opportunity'].sum(-1)
        self.stats['seq_top_front'] += top['front']; self.stats['seq_top_rear'] += top['rear']


def install_physics_observer(env):
    if hasattr(env, '_ascent_entry_observer'):
        if not isinstance(env._ascent_entry_observer, SequencePhysicsObserver):
            raise RuntimeError('Stage4 requires its own observer from the first physics step')
        return
    env._ascent_entry_observer = SequencePhysicsObserver(env)
    original = env.scene.update
    def update(scene, dt):
        original(dt); env._ascent_entry_observer.observe()
    env.scene.update = MethodType(update, env.scene)


def context(env):
    observer = env._ascent_entry_observer
    if observer.sequence_context_key == env.common_step_counter:
        return observer.sequence_cached
    events = ('ordered', 'missing', 'dwell', 'paired', 'simultaneous', 'early', 'cadence',
              'approach', 'top_front', 'top_rear', 'top_forward')
    names = (*events, 'height', 'horizontal', 'velocity', 'balance', 'surge', 'smoothness')
    zero = torch.zeros(env.num_envs, device=env.device)
    result = {k: observer.window.get('seq_'+k, zero)/(env.step_dt if k in events else max(observer.substeps, 1))
              for k in names}
    observer.sequence_context_key, observer.sequence_cached = env.common_step_counter, result
    return result


def ordered(env): return context(env)['ordered']
def missing(env): return context(env)['missing']
def dwell(env): return context(env)['dwell']
def paired(env): return context(env)['paired']
def simultaneous(env): return context(env)['simultaneous']
def early(env): return context(env)['early']
def cadence(env): return context(env)['cadence']
def approach(env): return context(env)['approach']
def height(env): return context(env)['height']
def horizontal(env): return context(env)['horizontal']
def velocity(env): return context(env)['velocity']
def balance(env): return context(env)['balance']
def surge(env): return context(env)['surge']
def smoothness(env): return context(env)['smoothness']
def top_front(env): return context(env)['top_front']
def top_rear(env): return context(env)['top_rear']
def top_forward(env): return context(env)['top_forward']


def orientation(env, asset_cfg, sensor_cfg, command_name='base_velocity', flat_weight=50., stair_weight=5.):
    from ..stair_teacher import stair_terrain_aware_orientation_cost
    native = stair_terrain_aware_orientation_cost(env, asset_cfg, sensor_cfg, command_name, flat_weight, stair_weight)
    observer = getattr(env, '_ascent_entry_observer', None)
    if observer is None or not hasattr(observer, 'top_posture'): return native
    # When only the front axle is on the platform, still permit the pitch
    # implied by the wheels' terrain heights. Flatten only as rear wheels rise.
    return torch.lerp(native, observer.top_posture, observer.top_blend)


def finished(env):
    observer = getattr(env, '_ascent_entry_observer', None)
    if observer is None or not hasattr(observer, 'top_ready'):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    types = env.scene.terrain.terrain_types
    progress = env.scene['robot'].data.root_pos_w[:, 0]-env.scene.env_origins[:, 0]
    entry = (types < 6) & (progress > 1.25+6.*env.cfg.ascent_entry_depth)
    return entry | ((types >= 6) & (types < ascent_column_count(env)) & observer.top_ready)


def metrics(env, env_ids):
    observer = getattr(env, '_ascent_entry_observer', None)
    if observer is None or not hasattr(observer, 'sequence'): return {}
    ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is not None: ids = ids[env_ids]
    ids = ids[ascent_mask(env)[ids]]
    count = max(len(ids), 1); ledger = observer.sequence; stats = observer.stats
    result = {k: stats[k][ids].sum()/count for k in ('seq_pairs', 'seq_ordered', 'seq_missing', 'seq_early',
                                                  'seq_simultaneous', 'seq_top_front', 'seq_top_rear')}
    result['front_first_four_coverage'] = ledger.covered[ids, :2, :4].float().sum()/(count*8)
    result['front_same_tread_sequential_first_four'] = ledger.pair_paid[ids, :4].float().sum()/(count*4)
    # Only episodes with a real opportunity contribute; emit no artificial
    # zero-error measurement when the most recent state is already past entry.
    result['__weighted_total__/target_height_error_abs_m'] = stats['seq_height_total'][ids].sum()
    result['__weighted_count__/target_height_error_abs_m'] = stats['seq_opportunity'][ids].sum()
    return result
