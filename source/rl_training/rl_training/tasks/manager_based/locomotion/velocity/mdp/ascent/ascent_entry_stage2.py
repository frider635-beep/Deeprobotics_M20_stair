"""Stage-two objectives; the original v1 observer/rewards remain unchanged.

Only reward state is added here. No ledger, contact or phase enters the actor.
"""
from __future__ import annotations

import torch
from types import MethodType
from .ascent_entry_mdp import EntryPhysicsObserver, contact_patch_velocity, context as entry_context, smooth01


class TreadLedger:
    """One credit or one miss per wheel/tread, with bounded contact-gap bridging.

    Missing lower treads resolve when a higher tread is stably reached or the
    wheel has passed beyond their reachable edge. A missed target cannot keep
    paying a penalty forever or invite reversing to collect old rewards.
    """
    def __init__(self, count, device, depth=.30, targets=4):
        self.depth = depth
        self.targets = torch.arange(1, targets+1, device=device)[None, None, :]
        shape = (count, 4, targets)
        self.covered = torch.zeros(shape, dtype=torch.bool, device=device)
        self.missed = torch.zeros_like(self.covered)
        self.best_dwell = torch.zeros(shape, device=device)
        self.level = torch.full((count, 4), -1, dtype=torch.long, device=device)
        self.duration = torch.zeros(count, 4, device=device)
        self.gap = torch.zeros_like(self.duration)

    def reset(self, ids):
        self.covered[ids] = False
        self.missed[ids] = False
        self.best_dwell[ids] = 0.
        self.level[ids] = -1
        self.duration[ids] = 0.
        self.gap[ids] = 0.

    def update(self, level, support, wheel_x, first_edge, active, dt):
        enabled = active[:, None]
        support = support & enabled
        continuing = (level == self.level) & (self.gap <= .01+1.e-7)
        duration = torch.where(continuing, self.duration+self.gap+dt, dt)
        self.duration = torch.where(support, duration, self.duration)
        self.gap = torch.where(support, 0., self.gap+dt)
        self.duration = torch.where((self.gap > .01+1.e-7) | ~enabled, 0., self.duration)
        self.level = torch.where(support, level, self.level)
        qualified = support & (self.duration >= .04-1.e-7)
        unresolved = ~(self.covered | self.missed)
        passed = wheel_x[:, :, None] > first_edge[:, None, None]+self.targets*self.depth+.09
        jumped = qualified[:, :, None] & (level[:, :, None] > self.targets)
        missed = unresolved & enabled[:, :, None] & (passed | jumped)
        self.missed |= missed
        unresolved = ~(self.covered | self.missed)
        next_index = unresolved.long().argmax(-1)
        next_target = self.targets.expand_as(unresolved).gather(-1, next_index[:, :, None])
        eligible = unresolved & (self.targets == next_target)
        touching_target = support[:, :, None] & (level[:, :, None] == self.targets)
        dwell = (self.duration/.04).clamp(0., 1.)[:, :, None]
        new_best = torch.maximum(self.best_dwell, dwell*eligible*touching_target)
        dwell_credit = (new_best-self.best_dwell).sum(-1)
        self.best_dwell.copy_(new_best)
        covered = eligible & touching_target & qualified[:, :, None]
        self.covered |= covered
        return {'ordered': covered.float().sum(-1), 'missing': missed.float().sum(-1),
                'dwell': dwell_credit}


def pair_slip_cost(per_wheel):
    """Equal group weights plus a worst-wheel term; no front/rear averaging."""
    return .5*per_wheel.mean(-1)+.5*per_wheel.amax(-1)


class Stage2PhysicsObserver(EntryPhysicsObserver):
    def __init__(self, env):
        super().__init__(env)
        self.ledger = TreadLedger(env.num_envs, env.device, float(env.cfg.ascent_entry_depth))
        for key in ('stage2_front_slip_sum', 'stage2_rear_slip_sum', 'stage2_covered', 'stage2_missing'):
            self.stats[key] = torch.zeros(env.num_envs, device=env.device)
        self.lifetime_ordered = torch.zeros((), device=env.device)
        self.lifetime_missing = torch.zeros((), device=env.device)
        self.stage2_context_key = -1

    def reset(self, ids):
        super().reset(ids)
        self.ledger.reset(ids)

    def observe(self):
        env = self.env
        if env._sim_step_counter == 0 or self.last_physics_tick == env._sim_step_counter:
            return
        reset = env.episode_length_buf < self.previous_lengths
        self.ledger.reset(reset)
        super().observe()
        command = env.command_manager.get_command('base_velocity')
        active = (env.scene.terrain.terrain_types < 8) & (command[:, 0] > .1)
        active &= (command[:, 1:].abs().amax(-1) < .1)
        data = env.scene['robot'].data
        rise = env.cfg.ascent_entry_height
        if getattr(env.cfg, 'ascent_adaptive_geometry', False):
            from .ascent_adaptive import geometry
            height, depth = geometry(env)
            rise = height[:, None]
            self.ledger.depth = depth[:, None, None]
        level = torch.round((self.previous_top_z-self.first_floor[:, None])/rise).long()
        events = self.ledger.update(level, self.support, data.body_pos_w[:, self.wheels, 0],
                                   env.scene.env_origins[:, 0]+1.25, active, env.physics_dt)
        for key, value in events.items():
            window_key = 'stage2_'+key
            if window_key not in self.window:
                self.window[window_key] = torch.zeros_like(value)
            self.window[window_key] += value
        self.lifetime_ordered += events['ordered'].sum()
        self.lifetime_missing += events['missing'].sum()
        self.stats['stage2_covered'] += events['ordered'].sum(-1)
        self.stats['stage2_missing'] += events['missing'].sum(-1)
        # Sensor data are cached at this same physics tick. Reconstruct raw
        # per-wheel tangent speed for diagnostics independently of the reward
        # pseudo-Huber transform in the inherited observer.
        points, forces = [], []
        for leg in ('fl', 'fr', 'hl', 'hr'):
            sensor = env.scene.sensors[f'entry_{leg}_contact'].data
            force = sensor.force_matrix_w[:, 0]
            index = torch.linalg.vector_norm(force, dim=-1).argmax(-1)
            row = torch.arange(env.num_envs, device=env.device)
            points.append(sensor.contact_pos_w[:, 0][row, index])
            forces.append(force[row, index])
        slip, support, _, _ = contact_patch_velocity(data.body_com_pos_w[:, self.wheels],
            data.body_com_lin_vel_w[:, self.wheels], data.body_ang_vel_w[:, self.wheels],
            torch.stack(points, 1), torch.stack(forces, 1))
        # Reuse the same physical contact calculation in later observers.
        self.contact_slip, self.contact_support = slip, support
        supported = slip*support*active[:, None]
        self.stats['stage2_front_slip_sum'] += supported[:, :2].sum(-1)
        self.stats['stage2_rear_slip_sum'] += supported[:, 2:].sum(-1)


def install_physics_observer(env):
    if hasattr(env, '_ascent_entry_observer'):
        if not isinstance(env._ascent_entry_observer, Stage2PhysicsObserver):
            raise RuntimeError('Stage2 must install its observer before the v1 observer')
        return
    env._ascent_entry_observer = Stage2PhysicsObserver(env)
    original = env.scene.update
    def update(scene, dt):
        original(dt)
        env._ascent_entry_observer.observe()
    env.scene.update = MethodType(update, env.scene)


def context(env):
    observer = env._ascent_entry_observer
    if not isinstance(observer, Stage2PhysicsObserver):
        raise RuntimeError('Stage2 reward requires Stage2PhysicsObserver')
    if observer.stage2_context_key == env.common_step_counter:
        return observer.stage2_cached
    original = entry_context(env)
    w = observer.window
    data = env.scene['robot'].data
    front_x = data.body_pos_w[:, observer.wheels[:2], 0].amax(-1)
    depth = env.cfg.ascent_entry_depth
    if getattr(env.cfg, 'ascent_adaptive_geometry', False):
        from .ascent_adaptive import geometry
        _, depth = geometry(env)
    progress = (front_x-env.scene.env_origins[:, 0]-1.25)/depth
    bump = smooth01((progress-.8)/.8)*(1.-smooth01((progress-4.5)/1.))
    gain = original['active']*(.5+.5*original['entry'])*(1.+.5*bump)
    cost = w['slip']/max(observer.substeps, 1)
    wheel_weights = cost.new_tensor([1.5, 1.5, 1., 1.])
    # Ledger events have their own per-wheel first-four-tread window; they
    # must not disappear because the other front wheel has advanced farther.
    result = {name: (w['stage2_'+name]*wheel_weights).sum(-1)/env.step_dt
              for name in ('ordered', 'missing', 'dwell')}
    result.update(front_slip=pair_slip_cost(cost[:, :2])*gain,
                  rear_slip=pair_slip_cost(cost[:, 2:])*gain,
                  landing=w['landing_inset_cost'].sum(-1)*original['entry']/env.step_dt,
                  stall=((.15-data.root_lin_vel_w[:, 0])/.15).clamp(0., 2.).square()
                        *original['entry']*(progress > -1.).float())
    observer.stage2_context_key, observer.stage2_cached = env.common_step_counter, result
    return result


def ordered_bonus(env): return context(env)['ordered']
def missing_cost(env): return context(env)['missing']
def dwell_bonus(env): return context(env)['dwell']
def front_slip_cost(env): return context(env)['front_slip']
def rear_slip_cost(env): return context(env)['rear_slip']
def landing_cost(env): return context(env)['landing']
def stall_cost(env): return context(env)['stall']


def metrics(env, env_ids):
    observer = env._ascent_entry_observer
    ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is not None:
        ids = ids[env_ids]
    ids = ids[env.scene.terrain.terrain_types[ids] < 8]
    samples = observer.stats['physics_samples'][ids].sum().clamp_min(1.)
    count = max(len(ids), 1)
    return {
        'front_first_four_coverage': observer.ledger.covered[ids, :2].float().sum()/(count*8),
        'rear_first_four_coverage': observer.ledger.covered[ids, 2:].float().sum()/(count*8),
        'front_missing_first_four': observer.ledger.missed[ids, :2].float().sum()/count,
        'rear_missing_first_four': observer.ledger.missed[ids, 2:].float().sum()/count,
        'front_tangent_slip_sum_per_physics_sample': observer.stats['stage2_front_slip_sum'][ids].sum()/samples,
        'rear_tangent_slip_sum_per_physics_sample': observer.stats['stage2_rear_slip_sum'][ids].sum()/samples,
    }
