"""Physical terrain evidence, legacy v9 quality, and v10 rear gait diagnostics."""
import torch
from copy import deepcopy
from .stair_path_course import ProgressiveCourse, PREFIXES, EVENTS
from ..ascent.ascent_adaptive import HEIGHTS

VERSION = 'm20_course_evidence_20261027_v1'
SETTINGS = dict(version=VERSION, physical_passes_per_speed=6, window_per_speed=8,
    prefix_pair_quality=[.50,.60,.65,.70,.75,.80,.85,.85,.90],
    height_pair_quality=.90, height_rear_second_quality=.50,
    demotion_min_trials=8, demotion_max_physical_success=.50, demotion_window='current_speed',
    consecutive_failures=3, height_failure_counts=[6,12],
    quality_denominator='required_pairs_per_axle_then_mean_over_physical_successes_each_speed',
    strict_episode_success='final_acceptance_only', mastery='physical_ability_independent_of_style',
    prefix_consolidation_episodes=16)
TRANSITION_VERSION = 'm20_course_evidence_20261027_v2_rear_tail'
TRANSITION_SETTINGS = deepcopy(SETTINGS)
TRANSITION_SETTINGS.pop('height_rear_second_quality')
TRANSITION_SETTINGS.update(version=TRANSITION_VERSION,
    rear_quality='treads3_to_goal_only_or_physical_pairs_for_entry_only_tasks',
    rear_tail_quality='all_required_rear_pairs_from_tread3_ordered_and_complete',
    strict_episode_success='front_all_treads_rear_from_tread3_physical_all_treads',
    promotion_pair_quality_axes=[0],
    rear_quality_role='reward_and_strict_gait_acceptance_only_not_terrain_promotion',
    rear_swing_order='penalty_only_after_entry_not_physical_failure')


def pair_quality(ledger, goals):
    required = ledger.targets[:,0] <= goals[:,None]
    paired = torch.stack((ledger.pair_paid,ledger.rear_pair_paid),1)
    clean = paired & ~ledger.style_bad_treads & required[:,None]
    quality = clean.float().sum(-1)/goals[:,None].clamp_min(1)
    if getattr(ledger, 'rear_entry_transition', False):
        tail = required & (ledger.targets[:,0] >= 3)
        count = tail.sum(-1)
        tail_quality = (clean[:,1] & tail).sum(-1)/count.clamp_min(1)
        quality[:,1] = torch.where(count > 0,tail_quality,quality[:,1])
        tail_clean = ((clean[:,1] | ~tail).all(-1) & (goals >= 4)).float()
        tail_clean *= ~ledger.rear_swing_style_failed
        return quality,tail_clean
    return quality,clean[:,1,1].float()


class EvidenceCourse(ProgressiveCourse):
    """Protect physical mastery; gate upward moves with separate gait quality."""
    evidence_fields = ('quality_window','rear_second_window')
    settings = SETTINGS
    evidence_version = VERSION
    height_quality_key = 'height_rear_second_quality'
    rear_entry_transition = False
    def __init__(self,count,device):
        super().__init__(count,device)
        self.quality_window = torch.zeros(count,2,8,2,device=device)
        setattr(self,self.evidence_fields[1],torch.zeros(count,2,8,device=device))
        # Per-finished-episode diagnostic flags. These are deliberately not
        # persistent evidence, and are only read on completed ascent slots.
        self.evidence_last = torch.zeros(count,6,device=device)

    def update(self,ids,success,eligible,rows,reference,training=True,
               promotion_success=None,pair_quality=None,rear_second_quality=None,rear_tail_quality=None):
        up = torch.zeros_like(success);down = torch.zeros_like(success)
        self.last_events[ids] = 0;self.evidence_last[ids] = 0
        if not training:return up,down
        if self.rear_entry_transition:
            rear_second_quality = rear_tail_quality
        if pair_quality is None or rear_second_quality is None:
            raise ValueError('EvidenceCourse needs pair quality and its versioned rear evidence')
        bucket = self.speed_bucket[ids];slot = self.cursor[ids,bucket]
        old = self.skill[ids].clone()
        self.window[ids,bucket,slot] = torch.where(eligible,success.long(),self.window[ids,bucket,slot])
        self.quality_window[ids,bucket,slot] = torch.where(eligible[:,None],pair_quality,self.quality_window[ids,bucket,slot])
        anchor_window = getattr(self,self.evidence_fields[1])
        anchor_window[ids,bucket,slot] = torch.where(eligible,rear_second_quality,anchor_window[ids,bucket,slot])
        self.cursor[ids,bucket] = (slot+eligible.long())%8
        self.trials[ids,bucket] = (self.trials[ids,bucket]+eligible.long()).clamp_max(8)
        self.completed[ids] += eligible.long()
        good = eligible & success;bad = eligible & ~success
        self.good[ids] = torch.where(good,self.good[ids]+1,torch.where(bad,0,self.good[ids]))
        self.bad[ids] = torch.where(bad,self.bad[ids]+1,torch.where(good,0,self.bad[ids])).clamp_max(self.mastered_height_failure_episodes)
        held = self.promotion_hold[ids] > 0
        self.promotion_hold[ids] = (self.promotion_hold[ids]-eligible.long()).clamp_min(0)
        self.height_age[ids] += eligible.long()
        passes = self.window[ids].sum(-1)
        enough = (self.trials[ids] >= 8).all(-1)
        physical_ready = good & enough & (passes >= 6).all(-1)
        quality = (self.quality_window[ids]*self.window[ids,:,:,None]).sum(2)/passes.clamp_min(1)[:,:,None]
        second = (anchor_window[ids]*self.window[ids]).sum(-1)/passes.clamp_min(1)
        threshold = self.skill.new_tensor(self.settings['prefix_pair_quality'],dtype=torch.float32)[old]
        promotion_quality = quality[:,:,self.settings.get('promotion_pair_quality_axes',[0,1])]
        style_ready = (promotion_quality+1.e-6 >= threshold[:,None,None]).all((1,2))
        height_style_ready = (promotion_quality+1.e-6 >= self.settings['height_pair_quality']).all((1,2))
        if self.height_quality_key is not None:
            height_style_ready &= (second+1.e-6 >= self.settings[self.height_quality_key]).all(-1)
        # Physical mastery is retained independently of whether ideal gait
        # quality yet allows a longer task; no physical pass window is lost.
        certified = physical_ready & (old > self.mastered[ids,rows])
        self._remember(ids,rows,old,physical_ready)
        can_height = ((old == len(PREFIXES)-1)
            | (self.geometry_probe[ids] & (old == PREFIXES.index(6)) & (rows < 2)))
        can_height &= ~reference & (rows < len(HEIGHTS)-1)
        promote = physical_ready & style_ready & ~held
        up = promote & can_height & height_style_ready
        # A six-tread probe whose height quality is not ready may still learn
        # eight treads at its current height instead of being frozen entirely.
        floor = self.protected_skill(rows,ids)
        at_floor = old <= floor
        trial_count = self.trials[ids,bucket]
        physical_rate = passes.gather(1,bucket[:,None]).squeeze(-1)/trial_count.clamp_min(1)
        poor = (trial_count >= SETTINGS['demotion_min_trials']) & (physical_rate <= SETTINGS['demotion_max_physical_success'])
        streak = eligible & (self.bad[ids] >= self.prefix_failure_episodes)
        retreat = streak & poor & ~at_floor
        learned_floor = self.mastered[ids,rows] >= self.minimum_skill[ids]
        height_threshold = torch.where(learned_floor,self.mastered_height_failure_episodes,self.unmastered_height_failure_episodes)
        down = eligible & at_floor & poor & (self.bad[ids] >= height_threshold) & ~reference & (rows > 0)
        skill = torch.where(promote,(old+1).clamp_max(len(PREFIXES)-1),old)
        skill = torch.where(retreat,torch.maximum(old-1,floor),skill)
        self._remember(ids,(rows-1).clamp_min(0),floor,down & learned_floor)
        lower_floor = self.protected_skill((rows-1).clamp_min(0),ids)
        skill = torch.where(down,torch.maximum(floor,lower_floor),skill)
        next_floor = self.protected_skill((rows+1).clamp_max(len(HEIGHTS)-1),ids)
        skill = torch.where(up,torch.maximum(next_floor,torch.full_like(old,PREFIXES.index(4))),skill)
        skill = torch.maximum(skill,self.minimum_skill[ids])
        returned = down & self.new_height[ids].bool()
        ready = physical_ready & style_ready
        self.last_events[ids] = torch.stack(((skill > old) & ~up,(skill < old) & ~up,
            up,down,returned,ready & held,eligible,good,certified,bad & at_floor),-1).long()
        self.event_counts[ids] += self.last_events[ids]
        self.evidence_last[ids] = torch.stack((physical_ready,physical_ready & ~style_ready,
            physical_ready & can_height & ~height_style_ready,streak & ~poor,
            pair_quality[:,0],pair_quality[:,1]),-1).float()
        changed = (skill != old) | up | down
        reset_ids = ids[changed]
        self.skill[ids] = skill
        self._clear_evidence(reset_ids)
        acted = changed
        self.good[ids] = torch.where(acted,0,self.good[ids])
        self.bad[ids] = torch.where(acted,0,self.bad[ids])
        height_changed = up | down
        self.height_age[ids[height_changed]] = 0
        self.new_height[ids[height_changed]] = up[height_changed].long()
        self.new_height[ids[(skill > old) & ~up]] = 0
        self.promotion_hold[ids[height_changed]] = self.consolidation_episodes
        self.promotion_hold[ids[(skill > old) & ~height_changed]] = SETTINGS['prefix_consolidation_episodes']
        return up,down

    def _clear_evidence(self,ids):
        for name in ('window','cursor','trials',*self.evidence_fields):getattr(self,name)[ids] = 0

    def reset_geometry(self,ids):
        super().reset_geometry(ids)
        self._clear_evidence(ids);self.evidence_last[ids] = 0

    def state_dict(self):
        state = super().state_dict()
        state.update({name:getattr(self,name).detach().cpu().clone() for name in self.evidence_fields})
        state['evidence_version'] = self.evidence_version
        return state

    def _validate(self,saved,legacy=False):
        if legacy:raise ValueError('v9 migration requires v8 evidence source')
        base = {k:v for k,v in saved.items() if k not in (*self.evidence_fields,'evidence_version')}
        ProgressiveCourse(len(self.skill),self.skill.device)._validate(base)
        if set(saved) != set(self.state_dict()) or saved.get('evidence_version') != self.evidence_version:
            raise ValueError('Invalid v9 evidence keys or version')
        for name in self.evidence_fields:
            value=saved[name];expected=getattr(self,name)
            if (not isinstance(value,torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype
                    or not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any()):
                raise ValueError('Invalid v9 evidence '+name)

    def migrate_v8(self,saved):
        ProgressiveCourse(len(self.skill),self.skill.device)._validate(saved)
        result = dict(saved)
        # v8 windows contain strict successes; interpreting them as physical
        # outcomes would trigger false retreats. Fresh physical evidence only.
        for name in ('window','cursor','trials','good'):
            result[name] = torch.zeros_like(saved[name])
        for name in self.evidence_fields:result[name] = torch.zeros_like(getattr(self,name),device='cpu')
        result['evidence_version'] = VERSION
        self._validate(result)
        return result


class RearEntryEvidenceCourse(EvidenceCourse):
    """Retain physical/front terrain gates; rear tail quality is diagnostic only."""
    evidence_fields = ('quality_window','rear_tail_window')
    settings = TRANSITION_SETTINGS
    evidence_version = TRANSITION_VERSION
    height_quality_key = None
    rear_entry_transition = True

    def migrate_v9(self,saved):
        EvidenceCourse(len(self.skill),self.skill.device)._validate(saved)
        result = {k:v for k,v in saved.items() if k != 'rear_second_window'}
        # Physical outcomes and front evidence have identical semantics. Keep
        # them so changing rear gait rules does not restart terrain evidence.
        # Rear slots start at zero and refill; they are diagnostic only.
        result['quality_window'] = saved['quality_window'].clone()
        result['quality_window'][...,1] = 0
        result['rear_tail_window'] = torch.zeros_like(self.rear_tail_window,device='cpu')
        result['evidence_version'] = self.evidence_version
        self._validate(result)
        return result
