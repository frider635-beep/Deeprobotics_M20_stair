"""Unified ascent observer: retain strict turns, teach paired landing before miss."""
from types import MethodType
import torch
from . import ascent_alternating_mdp as previous
from . import ascent_adaptive_mdp as adaptive
from . import ascent_adaptive as terrain_adaptive
from .ascent_adaptive import geometry
from .ascent_unified import BridgeCurriculum, PairedSequenceLedger, paired_guidance
from .......utils.teacher_diagnostics import export_teacher_metrics

NAMES = ('landing_height', 'landing_horizontal', 'landing_velocity', 'hold_mate', 'overspeed')


class UnifiedPhysicsObserver(previous.AlternatingPhysicsObserver):
    curriculum_type = BridgeCurriculum
    ledger_type = PairedSequenceLedger

    def __init__(self, env):
        super().__init__(env)
        self.failure_reason = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.unified_context_key = -1
        self.pairing_diagnostics = None
        if getattr(env.cfg, 'stair_ascent_pairing_v9', False):
            from .ascent_pairing import PairingDiagnostics
            self.pairing_diagnostics = PairingDiagnostics(env.num_envs,env.device,env.cfg.ascent_sequence_step_count)

    def reset(self, ids):
        super().reset(ids)
        self.failure_reason[ids] = 0; self.unified_context_key = -1
        if self.pairing_diagnostics is not None:
            self.pairing_diagnostics.reset(ids)

    def observe(self):
        env = self.env
        if env._sim_step_counter == 0 or self.last_physics_tick == env._sim_step_counter: return
        if self.pairing_diagnostics is not None:
            resets = self.episode_reset_ids()
            if len(resets): self.pairing_diagnostics.reset(resets)
        super().observe()
        data = env.scene['robot'].data
        command = env.command_manager.get_command('base_velocity')
        active = terrain_adaptive.ascent_mask(env) & (command[:, 0] > .1)
        active &= command[:, 1:].abs().amax(-1) < .1
        height, depth = geometry(env)
        position = data.body_pos_w[:, self.wheels]
        goals = adaptive.curriculum_state(env).goals()
        if self.pairing_diagnostics is None:
            guide, hold = paired_guidance(self.sequence, position, data.body_com_lin_vel_w[:, self.wheels, 2],
                env.scene.env_origins[:, 0]+1.25, self.first_floor, depth, height, goals, active)
        else:
            from .ascent_pairing import pairing_guidance
            first = env.scene.env_origins[:,0]+1.25
            guide,hold,clearance,state = pairing_guidance(self.sequence,self.turns,self.support,
                self.stable_time,position,data.body_com_lin_vel_w[:,self.wheels,2],
                first,self.first_floor,depth,height,goals,active)
            self.pairing_diagnostics.observe(state,self.sequence,position,first,depth,
                self.active_time,env.physics_dt)
        overspeed = ((data.root_lin_vel_w[:, 0]-command[:, 0]-.10).clamp_min(0.)/.20).square().clamp_max(9.)
        # A dedicated terminal corridor limits catastrophic path drift while the
        # dense whole-episode lateral cost remains active well inside it.
        path_failed = active & ((data.root_pos_w[:, 1]-env.scene.env_origins[:, 1]).abs() > .40)
        path_failed &= self.active_time > .5
        self.failure_reason = torch.where(self.failure_reason > 0, self.failure_reason, self.sequence.failure_reason)
        self.failure_reason = torch.where((self.failure_reason == 0) & self.turns.failed, 4, self.failure_reason)
        course = adaptive.curriculum_state(env)
        deadline = self.active_time.new_tensor(course.deadlines)[course.skill]
        self.failure_reason = torch.where((self.failure_reason == 0) & (self.active_time > deadline) & active, 5, self.failure_reason)
        self.failure_reason = torch.where((self.failure_reason == 0) & path_failed, 6, self.failure_reason)
        self.sequence.failed |= path_failed
        self.positive_gate &= ~path_failed
        self.prefix_ready &= self.positive_gate; self.top_ready &= self.positive_gate
        for key in ('ordered', 'dwell', 'paired', 'landing_credit', 'top_front', 'top_rear', 'top_forward'):
            self.window['adaptive_'+key] *= self.positive_gate
        self.window['alternating_progress'] *= self.positive_gate
        if self.pairing_diagnostics is not None:
            for key,value in dict(clearance=clearance,rear_paired=self.sequence.rear_paired_event,landing_order=self.sequence.style_event,rear_phase=state['rear_phase']).items():
                name = 'ascent_pairing_'+key
                if name not in self.window: self.window[name] = torch.zeros_like(value)
                self.window[name] += value*active
            self.window['ascent_pairing_rear_paired'] *= self.positive_gate
            # Local gait credit is blocked through a reversed pair and through
            # the policy tick containing its violation. Good later pairs can
            # learn again; heading, lateral and slip terms retain their gates.
            style_gate = self.sequence.style_reward_allowed & (self.window['ascent_pairing_landing_order'] == 0)
            for key in ('ordered','dwell','paired','landing_credit'):
                self.window['adaptive_'+key] *= style_gate
            self.window['alternating_progress'] *= style_gate
            self.window['ascent_pairing_rear_paired'] *= style_gate
            for key in ('top_front','top_rear','top_forward'):
                self.window['adaptive_'+key] *= ~self.sequence.style_failed
        values = dict(landing_height=guide['height'], landing_horizontal=guide['horizontal'],
                      landing_velocity=guide['velocity'], hold_mate=hold, overspeed=overspeed)
        for key,value in values.items():
            name = 'unified_'+key
            if name not in self.window: self.window[name] = torch.zeros_like(value)
            self.window[name] += value*active


def install_physics_observer(env):
    if hasattr(env, '_ascent_entry_observer'):
        if not isinstance(env._ascent_entry_observer, UnifiedPhysicsObserver):
            raise RuntimeError('Stage7 requires its own observer')
        return
    env._ascent_entry_observer = UnifiedPhysicsObserver(env)
    original = env.scene.update
    def update(scene, dt):
        original(dt); env._ascent_entry_observer.observe()
    env.scene.update = MethodType(update, env.scene)


def reward(env, name):
    observer = env._ascent_entry_observer
    if observer.unified_context_key != env.common_step_counter:
        observer.unified_cached = {key:observer.window.get('unified_'+key, torch.zeros(env.num_envs,device=env.device))/max(observer.substeps,1) for key in NAMES}
        observer.unified_context_key = env.common_step_counter
    return observer.unified_cached[name]


def metrics(env, env_ids):
    observer = env._ascent_entry_observer
    ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is not None: ids = ids[env_ids]
    ids = ids[terrain_adaptive.ascent_mask(env)[ids]]
    count = max(len(ids), 1)
    result = {}
    for code,label in enumerate(('missing_tread','early_advance','unpaired_landing','swing_turn','deadline','lateral_corridor'), start=1):
        result['failure_'+label] = (observer.failure_reason[ids] == code).float().sum()/count
    for j,leg in enumerate(('FL','FR','HL','HR')):
        for tread in (3,4):
            selected = ids[observer.episode_goals[ids] >= tread]
            result['__weighted_total__/'+leg+'_tread'+str(tread)+'_covered'] = observer.sequence.covered[selected,j,tread-1].float().sum()
            result['__weighted_count__/'+leg+'_tread'+str(tread)+'_covered'] = torch.as_tensor(len(selected),device=env.device).float()
    for j,leg in enumerate(('FL','FR','HL','HR')):
        times = observer.sequence.landing_time[ids,j]
        covered = observer.sequence.covered[ids,j]
        valid = covered[:,2] & covered[:,3]
        result['__weighted_total__/'+leg+'_interval3_to4_s'] = ((times[:,3]-times[:,2])*valid).sum()
        result['__weighted_count__/'+leg+'_interval3_to4_s'] = valid.sum().float()
        valid_middle = covered[:,4:-1] & covered[:,5:]
        intervals = times[:,5:]-times[:,4:-1]
        result['__weighted_total__/'+leg+'_middle_interval_s'] = (intervals*valid_middle).sum()
        result['__weighted_count__/'+leg+'_middle_interval_s'] = valid_middle.sum().float()
    if observer.pairing_diagnostics is not None:
        physical = adaptive.finished(env)[ids]
        result['strict_prefix_success_fraction'] = (physical & ~observer.sequence.style_failed[ids]).float().sum()/count
        result['landing_order_violation_fraction'] = observer.sequence.style_failed[ids].float().sum()/count
    state = adaptive.curriculum_state(env)
    if observer.pairing_diagnostics is not None:
        from .ascent_pairing import DIAGNOSTIC_FIELDS
        values = observer.pairing_diagnostics.values(observer,ids).sum(0)
        for j,name in enumerate(DIAGNOSTIC_FIELDS):
            result['__weighted_total__/pairing_'+name] = values[j]
            result['__weighted_count__/pairing_'+name] = torch.as_tensor(len(ids),device=env.device).float()
        for label,numerator,denominator in (
                ('rear_opportunity_coverage','rear_covered_attempts','rear_attempts'),
                ('rear_mature_opportunity_coverage','rear_covered_mature_attempts','rear_mature_attempts'),
                ('legacy_target_disagreement_fraction','legacy_spatial_target_disagreement_s','target_opportunity_s'),
                *((f'rear_tread{tread}_{label}',f'rear_tread{tread}_{num}',f'rear_tread{tread}_{den}')
                  for tread in range(1,19) for label,num,den in (
                      ('opportunity_coverage','covered_attempts','attempts'),
                      ('mature_opportunity_coverage','covered_mature_attempts','mature_attempts')))):
            result['__weighted_total__/pairing_'+label] = values[DIAGNOSTIC_FIELDS.index(numerator)]
            result['__weighted_count__/pairing_'+label] = values[DIAGNOSTIC_FIELDS.index(denominator)]
    for length in state.prefixes:
        result['prefix'+str(length)+'_environment_fraction'] = (state.goals()[ids] == length).float().sum()/count
    return export_teacher_metrics(result, result.keys(), detailed=True)
