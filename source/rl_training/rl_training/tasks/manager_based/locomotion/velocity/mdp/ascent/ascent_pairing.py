"""Stage9 v9 ascent-only pairing quality and rear same-tread phase guidance.

Neither actor inputs nor descent objectives consume this training-side state.
Each axle independently follows L1,R1,L2,R2,... or its mirror for a full episode.
"""
import torch
from .ascent_unified import PairedSequenceLedger
from .ascent_adaptive import landing_guidance
from .ascent_entry_mdp import smooth01

PREVIOUS_VERSION = 'm20_stair_path_fixed247_20261027_v8_soft_pairing'
VERSION = 'm20_stair_path_fixed247_20261027_v9_course_evidence'
SOURCE_VERSION = 'm20_stair_path_fixed247_20261027_v6_mastered_prefix_floor'
REWARD_VERSION = 'ascent_pairing_20261027_v2'
REAR_PAIR_WEIGHT = .75
LANDING_ORDER_WEIGHT = -.5
SETTINGS = dict(version=REWARD_VERSION, scope='ascent_only',
    target='unresolved_same_tread_per_axle_until_both_stably_supported',
    alternation='continuous_landing_and_swing_alternation_independent_per_axle',
    clearance='continuous_current_tread_landing_with_legal_preparation_bridge',
    landing_order=dict(mode='soft_no_termination',weight=LANDING_ORDER_WEIGHT,
        reward_gate='incorrect_tread_gait_credit_only_recover_next_pair',
        leader='first_stable_landing_side_fixed_per_axle_for_episode',
        promotion='physical_success_and_clean_episode',demotion='physical_failures_only'),
    narrow_depths_m=[.20,.25], clearance_weight=-.30,
    rear_pair_weight=REAR_PAIR_WEIGHT, front_pair_weight=1.5,
    landing_credit='existing_front_only_no_additional_dense_positive',
    unchanged='descent_rewards_path_control_lateral_heading_slip_network_learning_rate')

# Exact v8 contract for explicit migration; old running jobs retain this version.
from copy import deepcopy
V8_SETTINGS = deepcopy(SETTINGS)
REWARD_VERSION = 'ascent_pairing_20261027_v3'
REAR_PHASE_WEIGHT = -.15
SETTINGS['version'] = REWARD_VERSION
SETTINGS['landing_order']['promotion'] = 'physical_success_with_graded_per_axle_pair_quality'
SETTINGS['landing_order']['demotion'] = 'physical_streak_and_low_recent_physical_success'
SETTINGS['rear_phase'] = dict(weight=REAR_PHASE_WEIGHT,start_tread=2,
    waiting='follower_on_previous_tread_until_original_leader_stably_lands_current_tread',
    ideal_order='same_episode_first_side_per_axle',dense_cost_cap=4.,
    existing_landing_target='waiting_rear_follower_uses_previous_tread_all_others_current')


class AlternatingPairLedger(PairedSequenceLedger):
    """Keep physical failures strict; teach landing order without ending rollout.

    The first leader is fixed independently for each axle. A reversed pair
    loses gait credit, but a correct next pair can earn credit again. Physical
    coverage remains available to the curriculum; strict success is separate.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = len(self.time)
        self.last_landing_side = torch.full((n,2), -1, dtype=torch.long, device=self.time.device)
        self.first_side = torch.full_like(self.last_landing_side,-1)
        self.style_bad_treads = torch.zeros(n,2,self.covered.shape[-1],dtype=torch.bool,device=self.time.device)
        self.style_failed = torch.zeros_like(self.failed)
        self.style_event = torch.zeros_like(self.time)
        self.style_reward_allowed = torch.ones_like(self.failed)
        self.first_style_tread = torch.zeros(n,2,dtype=torch.long,device=self.time.device)
        self.rear_paired_event = torch.zeros_like(self.time)
        self.first_fault_wheels = torch.zeros(n,4,dtype=torch.bool,device=self.time.device)
        self.first_fault_treads = torch.zeros(n,4,dtype=torch.long,device=self.time.device)
        self.first_fault_passed = torch.zeros(n,4,dtype=torch.bool,device=self.time.device)
        self.first_fault_jumped = torch.zeros_like(self.first_fault_passed)

    def reset(self, ids):
        super().reset(ids)
        self.last_landing_side[ids] = -1; self.first_side[ids] = -1
        self.style_reward_allowed[ids] = True
        for name in ('style_bad_treads','style_failed','style_event','first_style_tread',
                     'rear_paired_event','first_fault_wheels','first_fault_treads',
                     'first_fault_passed','first_fault_jumped'):
            getattr(self,name)[ids] = 0

    def update_strict(self, level, support, wheel_x, first, active, dt, goals, period=None):
        prior_rear = self.rear_pair_paid.clone()
        prior_front = self.pair_paid.clone()
        old_covered = self.covered.clone()
        old_missed = self.missed.clone()
        was_failed = self.failed.clone()
        result = super().update_strict(level,support,wheel_x,first,active,dt,goals,period)
        valid = active & ~self.failed
        physical_new = result['new_mask'].reshape(-1,2,2,self.covered.shape[-1])
        new_side = physical_new.any(-1)
        count = new_side.sum(-1)
        side = new_side.long().argmax(-1)
        first_landing = (self.first_side < 0) & (count == 1) & valid[:,None]
        self.first_side = torch.where(first_landing,side,self.first_side)
        self.last_landing_side = torch.where((count == 1) & valid[:,None],side,self.last_landing_side)
        # Require the same leading side on every tread. This also permits a
        # return to the ideal order after a bad pair, without resetting history.
        previous = old_covered.reshape_as(physical_new)
        leader = self.first_side.clamp_min(0)[:,:,None,None].expand(-1,-1,1,self.covered.shape[-1])
        leader_done = previous.gather(2,leader).squeeze(2)
        expected = torch.where(leader_done,1-self.first_side[:,:,None],self.first_side[:,:,None])
        sides = torch.arange(2,device=self.time.device)[None,None,:,None]
        wrong = (physical_new & (sides != expected[:,:,None])).any(2) & valid[:,None,None]
        newly_bad = wrong & ~self.style_bad_treads
        self.style_bad_treads |= wrong
        self.style_failed |= newly_bad.any((1,2))
        self.style_event = newly_bad.float().sum((1,2))
        first_bad = newly_bad.long().argmax(-1)+1
        self.first_style_tread = torch.where((self.first_style_tread == 0) & newly_bad.any(-1),first_bad,self.first_style_tread)
        clean = ~self.style_bad_treads
        result['new_mask'] &= clean.repeat_interleave(2,1)
        result['ordered'] = result['new_mask'].float().sum(-1)
        index = (level-1).clamp(0,self.covered.shape[-1]-1)[:,:,None]
        clean_level = clean.repeat_interleave(2,1).gather(-1,index).squeeze(-1)
        result['dwell'] *= clean_level
        required = self.targets[:,0] <= goals[:,None]
        result['paired'] = ((self.pair_paid & ~prior_front) & required & clean[:,0]).float().sum(-1)*valid
        self.rear_paired_event = ((self.rear_pair_paid & ~prior_rear) & required & clean[:,1]).float().sum(-1)*valid
        # Front landing improvements for a bad pair can never be paid later.
        self.pending_landing *= clean[:,0,None]
        pending = ~self.covered.reshape_as(physical_new).all(2) & required[:,None]
        self.style_reward_allowed = ~(pending & self.style_bad_treads).any((1,2))
        # Capture the first physical miss, rather than infer its cause later
        # from the earliest uncovered target. Multiple wheels may fail together.
        missed = self.missed & ~old_missed & required[:,None] & (~was_failed & active)[:,None,None]
        fault = missed.any(-1)
        tread = missed.long().argmax(-1)+1
        self.first_fault_wheels |= fault
        self.first_fault_treads = torch.where(fault,tread,self.first_fault_treads)
        passed = wheel_x[:,:,None] > first[:,None,None]+self.targets*self.depth+.09
        jumped = (level[:,:,None] > self.targets) & support[:,:,None] & (self.duration >= .04-1.e-7)[:,:,None]
        self.first_fault_passed |= (missed & passed).any(-1)
        self.first_fault_jumped |= (missed & jumped).any(-1)
        return result


def paired_targets(ledger, goals, active):
    """One unresolved target per axle; a landed leader holds that exact tread."""
    required = ledger.targets[:,0] <= goals[:,None]
    pending = ~ledger.covered.reshape(-1,2,2,ledger.covered.shape[-1]).all(2) & required[:,None]
    target = (pending.long().argmax(-1)+1).repeat_interleave(2,-1)
    enabled = pending.any(-1).repeat_interleave(2,-1) & active[:,None] & ~ledger.failed[:,None]
    done = ledger.covered.gather(-1,(target-1)[:,:,None]).squeeze(-1)
    result = dict(target=target, enabled=enabled, hold=done & enabled)
    if hasattr(ledger,'first_side'):
        leader = ledger.first_side
        lead_index = leader.clamp_min(0)[:,:,None]
        leader_done = done.reshape(-1,2,2).gather(-1,lead_index).squeeze(-1)
        result['expected_side'] = torch.where(leader_done,1-leader,leader)
    return result


def swing_permissions(state, turns, support, stable_time, position, first, depth):
    """Prepare only a legal side, retaining guidance throughout its real swing.

    The first side is free in the validator. Guidance prefers the nearer wheel;
    exact ties alternate across environments/axles, without an FL-first rule.
    """
    n = len(position)
    sides = torch.arange(2,device=position.device)[None,None]
    last = turns.last_side[:,:,None]
    hold = state['hold'].reshape(n,2,2)
    enabled = state['enabled'].reshape(n,2,2)
    armed = turns.armed.reshape(n,2,2)
    ongoing = (last == sides) & ~armed & turns.seen_support.reshape(n,2,2) & ~hold & enabled
    x = position[...,0].reshape(n,2,2)
    tie_side = (torch.arange(n,device=x.device)[:,None]+torch.arange(2,device=x.device))%2
    first_side = torch.where((x[:,:,0]-x[:,:,1]).abs() > 1.e-5,x.argmax(-1),tie_side)
    next_side = torch.where(turns.last_side < 0,first_side,1-turns.last_side)
    mate_stable = ((stable_time >= .04-1.e-6) & support).reshape(n,2,2).flip(-1)
    start = (sides == next_side[:,:,None]) & ~ongoing.any(-1)[:,:,None]
    # Keep legal preparation active during the <20ms airborne confirmation
    # interval. The stable mate and armed/seen-support guards still apply.
    start &= armed & mate_stable & turns.seen_support.reshape(n,2,2) & ~hold & enabled
    if 'expected_side' in state:
        expected = state['expected_side'][:,:,None]
        start &= (expected < 0) | (sides == expected)
    return (start | ongoing).reshape(n,4)


def rear_phase_state(ledger,state,position,first,floor,depth,height,active):
    """Hold the rear follower before its next edge until its leader lands.

    Previous-pair coverage is required. Neither a missing first tread nor a
    chosen front/rear offset can trigger this. No left/right coordinate target
    or direct force is added; all costs remain bounded training rewards.
    """
    target=state['target'][:,2:]
    n=len(target);side=torch.arange(2,device=target.device)[None]
    leader=ledger.first_side[:,1]
    previous=(target-2).clamp_min(0)[:,:,None]
    both_previous=ledger.covered[:,2:].gather(-1,previous).squeeze(-1).all(-1)
    current=(target-1)[:,:,None]
    done=ledger.covered[:,2:].gather(-1,current).squeeze(-1)
    leader_done=done.gather(1,leader.clamp_min(0)[:,None]).squeeze(-1)
    waiting=(side != leader[:,None]) & (leader >= 0)[:,None] & (target >= 2)
    waiting &= both_previous[:,None] & ~leader_done[:,None] & state['enabled'][:,2:]
    waiting &= active[:,None]
    edge=first[:,None]+(target-1)*depth[:,None]
    near=smooth01((position[:,2:,0]-edge+.35)/.25)
    previous_top=floor[:,None]+(target-1)*height[:,None]
    lift=((position[:,2:,2]-.09-previous_top-.015).clamp_min(0.)/.08).square()
    advance=((position[:,2:,0]-(edge-.02)).clamp_min(0.)/.10).square()
    cost=((lift+advance).clamp_max(4.)*waiting*near).sum(-1)
    mask=torch.zeros(n,4,dtype=torch.bool,device=target.device);mask[:,2:]=waiting
    return mask,cost


def pairing_guidance(ledger, turns, support, stable_time, position, velocity_z,
                     first, floor, depth, height, goals, active):
    state = paired_targets(ledger,goals,active)
    permission = swing_permissions(state,turns,support,stable_time,position,first,depth)
    target, enabled, hold = state['target'],state['enabled'],state['hold']
    rear_wait,rear_phase = rear_phase_state(ledger,state,position,first,floor,depth,height,active)
    # Only the rear follower awaiting the current leader uses the PREVIOUS
    # tread. Legal swings keep continuous current-target guidance.
    guide_target = torch.where(rear_wait,target-1,target)
    guide = landing_guidance(position,velocity_z,guide_target,enabled,
                             first,floor,depth,height)
    edge = first[:,None]+(target-1)*depth[:,None]
    offset = position[...,0]-edge
    bottom = position[...,2]-.09
    top = floor[:,None]+target*height[:,None]
    blend = smooth01((offset-.01)/.06)
    near = smooth01((offset+.35)/.25)*(offset < depth[:,None]+.09)
    prepare = ((top+.04-bottom).clamp_min(0.)/.10).square().clamp_max(4.)
    prepare *= (1.-blend)*near*permission*enabled*(depth[:,None] <= .25+1.e-6)
    hold_cost = (((bottom-top)/.08).square()
        +((offset-(depth[:,None]-.08)).clamp_min(0.)/.08).square()
        +(velocity_z.clamp_min(0.)/.3).square()).clamp_max(9.)
    hold_cost = (hold_cost*hold).sum(-1)/hold.sum(-1).clamp_min(1)
    state.update(offset=offset,permission=permission,rear_wait=rear_wait,rear_phase=rear_phase)
    return guide,hold_cost,prepare.sum(-1),state


DIAGNOSTIC_FIELDS = ('pairing_attempts','pairing_mature_attempts','pairing_covered_attempts',
    'pairing_covered_mature_attempts','rear_attempts','rear_mature_attempts',
    'rear_covered_attempts','rear_covered_mature_attempts','rear_truncated_attempts',
    'rear_no_opportunity_at_failure','legacy_spatial_target_disagreement_s','target_opportunity_s',
    'front_pair_events','rear_pair_events','first_fault_passed_wheels','first_fault_jumped_wheels',
    *(f'first_fault_{wheel}_count' for wheel in ('FL','FR','HL','HR')),
    *(f'first_fault_tread{tread}_count' for tread in range(1,19)),
    'physical_success','strict_success','landing_order_violation_fraction',
    'front_landing_order_violation_fraction','rear_landing_order_violation_fraction',
    'front_ordered_pair_quality','rear_ordered_pair_quality','rear_second_ordered_pair',
    *(f'first_style_tread{tread}_count' for tread in range(1,19)),
    *(f'rear_tread{tread}_{kind}' for tread in range(1,19)
      for kind in ('attempts','mature_attempts','covered_attempts','covered_mature_attempts')))


class PairingDiagnostics:
    def __init__(self,count,device,targets=18):
        self.started = torch.full((count,4,targets),float('inf'),device=device)
        self.mature = torch.zeros_like(self.started,dtype=torch.bool)
        self.disagreement = torch.zeros(count,device=device)
        self.opportunity_frames = torch.zeros_like(self.disagreement)

    def reset(self,ids):
        self.started[ids] = float('inf'); self.mature[ids] = False
        self.disagreement[ids] = 0; self.opportunity_frames[ids] = 0

    def observe(self,state,ledger,position,first,depth,time,dt):
        index = (state['target']-1)[:,:,None]
        previous = self.started.gather(-1,index).squeeze(-1)
        # Do not select on swing permission: a blocked wheel's failed attempt
        # remains in the denominator. Keep shorter-lived attempts separately.
        opportunity = state['enabled'] & ~state['hold'] & (state['offset'] >= -.35) & (state['offset'] < depth[:,None]+.09)
        start = torch.where(opportunity & ~torch.isfinite(previous),time[:,None],previous)
        self.started.scatter_(-1,index,start[:,:,None])
        matured = self.mature.gather(-1,index).squeeze(-1)
        matured |= state['enabled'] & torch.isfinite(start) & (time[:,None]-start >= .04-1.e-6)
        self.mature.scatter_(-1,index,matured[:,:,None])
        spatial = (torch.floor((position[...,0]-first[:,None])/depth[:,None])+2).clamp(1,18)
        self.disagreement += ((spatial != state['target']) & opportunity).sum(-1)*dt
        self.opportunity_frames += opportunity.sum(-1)*dt

    def values(self,observer,ids):
        ledger = observer.sequence
        required = ledger.targets <= observer.episode_goals[:,None,None]
        attempts = torch.isfinite(self.started) & required
        mature = self.mature & attempts
        covered = ledger.covered & attempts
        rear = attempts[:,2:]; rear_mature = mature[:,2:]; rear_covered = covered[:,2:]
        failure = (observer.failure_reason > 0)
        values = (attempts.sum((1,2)),mature.sum((1,2)),covered.sum((1,2)),
            (covered & mature).sum((1,2)),rear.sum((1,2)),rear_mature.sum((1,2)),
            rear_covered.sum((1,2)),(rear_covered & rear_mature).sum((1,2)),
            (rear & ~rear_mature).sum((1,2)),
            ((~rear.any(-1)) & failure[:,None]).sum(-1),self.disagreement,self.opportunity_frames,
            (ledger.pair_paid & required[:,0]).sum(-1),(ledger.rear_pair_paid & required[:,0]).sum(-1),
            ledger.first_fault_passed.sum(-1),ledger.first_fault_jumped.sum(-1),
            *ledger.first_fault_wheels.unbind(-1),
            *((ledger.first_fault_treads[:,:,None] == torch.arange(1,19,device=attempts.device)).sum(1)).unbind(-1))
        physical = observer.prefix_ready & ((observer.episode_goals < ledger.covered.shape[-1]) | observer.top_ready)
        from ..path.stair_path_evidence import pair_quality
        quality,second=pair_quality(ledger,observer.episode_goals)
        extra = (physical,physical & ~ledger.style_failed,ledger.style_failed,
            ledger.style_bad_treads[:,0].any(-1),ledger.style_bad_treads[:,1].any(-1),
            quality[:,0],quality[:,1],second,
            *((ledger.first_style_tread[:,:,None] == torch.arange(1,19,device=attempts.device)).sum(1)).unbind(-1),
            *(value for tread in range(18) for value in (
                rear[:,:,tread].sum(-1),rear_mature[:,:,tread].sum(-1),rear_covered[:,:,tread].sum(-1),
                (rear_covered[:,:,tread] & rear_mature[:,:,tread]).sum(-1))))
        return torch.stack((*values,*extra),-1)[ids].float()


def reward(env,name):
    """New terms are exactly zero outside ascent, including all descent slots."""
    from .ascent_adaptive import ascent_mask
    observer = env._ascent_entry_observer
    command = env.command_manager.get_command('base_velocity')
    active = ascent_mask(env) & (command[:,0] > .1) & (command[:,1:].abs().amax(-1) < .1)
    value = observer.window.get('ascent_pairing_'+name,torch.zeros(env.num_envs,device=env.device))
    if name == 'rear_paired':
        active &= observer.positive_gate & ~observer.sequence.failed
        if hasattr(observer,'path_failed'):
            active &= ~observer.path_failed
    divisor = env.step_dt if name in ('rear_paired','landing_order') else max(observer.substeps,1)
    return value/divisor*active
