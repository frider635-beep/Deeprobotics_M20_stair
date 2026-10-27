"""Stage9 progressive contact courses; no policy or simulator dependencies."""
import torch
from ..ascent.ascent_unified import BridgeCurriculum
from ..ascent.ascent_adaptive import HEIGHTS

LEGACY_VERSION = 'm20_stair_path_fixed247_20261026_v5_progressive_course'
VERSION = 'm20_stair_path_fixed247_20261027_v6_mastered_prefix_floor'
PREFIXES = (2, 3, 4, 6, 8, 10, 12, 15, 18)
DEADLINES = (8., 10., 12., 16., 18., 20., 22., 23.5, 23.5)
EVENTS = ('prefix_up', 'prefix_down', 'height_up', 'height_down',
          'new_height_return', 'promotion_held', 'eligible', 'success',
          'mastery_confirmed', 'protected_floor_failure')


class ProgressiveCourse(BridgeCurriculum):
    prefixes = PREFIXES
    deadlines = DEADLINES
    # All durations below are eligible episodes, not PPO iterations.
    consolidation_episodes = 24
    prefix_failure_episodes = 3
    mastered_height_failure_episodes = 12
    unmastered_height_failure_episodes = 6

    def __init__(self, count, device):
        super().__init__(count, device)
        self.height_age = torch.zeros_like(self.skill)
        self.promotion_hold = torch.zeros_like(self.skill)
        self.new_height = torch.zeros_like(self.skill)
        # Highest certified prefix INDEX at each height of this environment's
        # geometry. -1 means no evidence; this is not the minimum practice task.
        self.mastered = torch.full((count,len(HEIGHTS)),-1,device=device,dtype=torch.long)
        self.event_counts = torch.zeros(count, len(EVENTS), device=device, dtype=torch.long)
        self.last_events = torch.zeros_like(self.event_counts)

    def protected_skill(self, rows, ids=None):
        learned = self.mastered.gather(1,rows[:,None]).squeeze(1) if ids is None else self.mastered[ids,rows]
        minimum = self.minimum_skill if ids is None else self.minimum_skill[ids]
        return torch.maximum(learned,minimum)

    def _remember(self, ids, rows, skills, verified):
        # A pass on a higher stair also protects the same prefix at lower
        # heights, on the SAME width/depth. Never infer mastery at higher rows.
        lower = torch.arange(len(HEIGHTS),device=rows.device)[None,:] <= rows[:,None]
        known = self.mastered[ids]
        self.mastered[ids] = torch.where(verified[:,None] & lower,
            torch.maximum(known,skills[:,None]),known)

    def critic_skill(self):
        # Preserve the old critic feature at every legacy task, including18.
        return self.skill.new_tensor((0., .25, .5, .75, .80, .85, .90, .95, 1.),
                                     dtype=torch.float32)[self.skill]

    def update(self, ids, success, eligible, rows, reference, training=True, promotion_success=None):
        up = torch.zeros_like(success); down = torch.zeros_like(success)
        self.last_events[ids] = 0
        if not training:
            return up, down
        strict = success if promotion_success is None else success & promotion_success
        bucket = self.speed_bucket[ids]; slot = self.cursor[ids, bucket]
        old = self.skill[ids].clone()
        self.window[ids, bucket, slot] = torch.where(eligible, strict.long(), self.window[ids, bucket, slot])
        self.cursor[ids, bucket] = (slot+eligible.long()) % 8
        self.trials[ids, bucket] = (self.trials[ids, bucket]+eligible.long()).clamp_max(8)
        self.completed[ids] += eligible.long()
        good = eligible & success; bad = eligible & ~success
        self.good[ids] = torch.where(good, self.good[ids]+1, torch.where(bad, 0, self.good[ids]))
        self.bad[ids] = torch.where(bad, self.bad[ids]+1, torch.where(good, 0, self.bad[ids]))
        self.bad[ids] = self.bad[ids].clamp_max(self.mastered_height_failure_episodes)
        held = self.promotion_hold[ids] > 0
        self.promotion_hold[ids] = (self.promotion_hold[ids]-eligible.long()).clamp_min(0)
        self.height_age[ids] += eligible.long()
        ready = good & strict & (self.trials[ids] >= 8).all(-1) & (self.window[ids].sum(-1) >= 6).all(-1)
        certified = ready & (old > self.mastered[ids,rows])
        self._remember(ids,rows,old,ready)
        promote = ready & ~held
        floor = self.protected_skill(rows,ids)
        at_floor = old <= floor
        retreat = eligible & (self.bad[ids] >= self.prefix_failure_episodes) & ~at_floor
        learned_floor = self.mastered[ids,rows] >= self.minimum_skill[ids]
        height_threshold = torch.where(learned_floor,self.mastered_height_failure_episodes,
                                       self.unmastered_height_failure_episodes)
        up = promote & ((old == len(PREFIXES)-1)
                        | (self.geometry_probe[ids] & (old == PREFIXES.index(6)) & (rows < 2)))
        up &= ~reference & (rows < len(HEIGHTS)-1)
        # Long-task failure is handled by shortening the task first, at ALL
        # heights. The learned floor is permanent for this geometry.
        down = eligible & at_floor & (self.bad[ids] >= height_threshold) & ~reference & (rows > 0)
        skill = torch.where(promote, (old+1).clamp_max(len(PREFIXES)-1), old)
        skill = torch.where(retreat,torch.maximum(old-1,floor),skill)
        # On height fallback keep the already learned prefix, including when
        # the lower row had no separate historical record yet.
        self._remember(ids,(rows-1).clamp_min(0),floor,down & learned_floor)
        lower_floor = self.protected_skill((rows-1).clamp_min(0),ids)
        skill = torch.where(down,torch.maximum(floor,lower_floor),skill)
        next_floor = self.protected_skill((rows+1).clamp_max(len(HEIGHTS)-1),ids)
        skill = torch.where(up,torch.maximum(next_floor,torch.full_like(old,PREFIXES.index(4))),skill)
        skill = torch.maximum(skill, self.minimum_skill[ids])
        returned = down & self.new_height[ids].bool()
        events = torch.stack(((skill > old) & ~up, (skill < old) & ~up,
            up, down, returned, ready & held, eligible, good,certified,bad & at_floor), -1).long()
        self.last_events[ids] = events
        self.event_counts[ids] += events
        changed = (skill != old) | up | down
        reset_ids = ids[changed]
        self.skill[ids] = skill
        for name in ('window','cursor','trials'):
            getattr(self,name)[reset_ids] = 0
        # Failed floor retries MUST accumulate beyond3; resetting every3
        # would make sustained-failure height fallback impossible.
        reset_streak = promote | retreat | up | down
        self.good[ids] = torch.where(reset_streak, 0, self.good[ids])
        self.bad[ids] = torch.where(reset_streak, 0, self.bad[ids])
        height_changed = up | down
        self.height_age[ids[height_changed]] = 0
        self.new_height[ids[height_changed]] = up[height_changed].long()
        # Keep a newly raised height under observation until a prefix has
        # actually been mastered there; downgrade is always available.
        self.new_height[ids[promote & ~up]] = 0
        self.promotion_hold[ids[height_changed]] = self.consolidation_episodes
        return up, down

    def fresh_promotion_evidence(self, saved):
        """Clone state; preserve mastery/geometry/physical failures, clear old pass windows."""
        self._validate(saved)
        result = dict(saved)
        for name in ('window','cursor','trials','good'):
            result[name] = torch.zeros_like(saved[name])
        return result

    def reset_geometry(self, ids):
        for name in ('height_age','promotion_hold','new_height'):
            getattr(self,name)[ids] = 0
        self.last_events[ids] = 0
        self.mastered[ids] = -1

    def state_dict(self):
        state = super().state_dict()
        state.update({name:getattr(self,name).detach().cpu().clone()
                      for name in ('height_age','promotion_hold','new_height','event_counts','mastered')})
        state.update(curriculum_version=VERSION, prefixes=list(PREFIXES))
        return state

    def migrate_bridge(self, saved, rows=None):
        # Validate with the actual old implementation before mapping indices.
        old = BridgeCurriculum(len(self.skill), self.skill.device)
        old.load_state_dict(saved)
        result = dict(saved)
        result['skill'] = torch.tensor((0,1,2,3,8),device=saved['skill'].device)[saved['skill']]
        for name in ('height_age','promotion_hold','new_height'):
            result[name] = torch.zeros_like(getattr(self,name),device='cpu')
        result['event_counts'] = torch.zeros(len(self.skill),8,dtype=torch.long)
        result.update(curriculum_version=LEGACY_VERSION,prefixes=list(PREFIXES))
        return self.migrate_v5(result,rows)

    def migrate_v5(self, saved, rows=None):
        self._validate(saved,legacy=True)
        rows = torch.zeros_like(self.skill) if rows is None else torch.as_tensor(rows,device=self.skill.device)
        if rows.shape != self.skill.shape or rows.dtype != torch.long or (rows < 0).any() or (rows >= len(HEIGHTS)).any():
            raise ValueError('Invalid migration terrain rows')
        result = dict(saved)
        skill = saved['skill'].to(self.skill.device)
        # A v5 long target proves at least6 was passed on this geometry, at
        # this or a higher height: every upward height move restarted at4.
        # Do not infer8/10/12 merely from the target: v4 jumped6->18 and v5
        # could have shortened that target without ever passing those tasks.
        inferred = torch.where(skill >= PREFIXES.index(8),PREFIXES.index(6),
                               torch.where(skill == PREFIXES.index(6),PREFIXES.index(4),-1))
        window = saved['window'].to(self.skill.device)
        trials = saved['trials'].to(self.skill.device)
        qualified = (trials >= 8).all(-1) & (window.sum(-1) >= 6).all(-1)
        inferred = torch.where(qualified,torch.maximum(inferred,skill),inferred)
        lower = torch.arange(len(HEIGHTS),device=self.skill.device)[None,:] <= rows[:,None]
        result['mastered'] = torch.where(lower,inferred[:,None],-1).cpu()
        counts = torch.zeros_like(self.event_counts,device='cpu')
        counts[:,:8] = saved['event_counts'].cpu()
        result.update(event_counts=counts,curriculum_version=VERSION)
        return result

    def _validate(self, saved, legacy=False):
        expected = self.state_dict()
        if legacy:
            expected.pop('mastered')
            expected['event_counts'] = expected['event_counts'][:,:8]
        if (set(saved) != set(expected) or saved.get('curriculum_version') != (LEGACY_VERSION if legacy else VERSION)
                or saved.get('prefixes') != list(PREFIXES)):
            raise ValueError('Course requires native v6 state or explicit v4/v5 migration')
        for name,value in expected.items():
            if not isinstance(value,torch.Tensor):
                continue
            candidate = saved[name]
            if (not isinstance(candidate,torch.Tensor) or candidate.shape != value.shape
                    or candidate.dtype != torch.long or (candidate < (-1 if name=='mastered' else 0)).any()):
                raise ValueError('Invalid progressive course '+name)
        skill = saved['skill']
        if ((skill < self.minimum_skill.to(skill.device)).any() or (skill >= len(PREFIXES)).any()
                or (saved['speed_bucket'] > 1).any() or (saved['window'] > 1).any()
                or (saved['cursor'] >= 8).any() or (saved['trials'] > 8).any()
                or (saved['new_height'] > 1).any() or (saved['promotion_hold'] > self.consolidation_episodes).any()
                or (saved['bad'] > (2 if legacy else self.mastered_height_failure_episodes)).any()):
            raise ValueError('Progressive course state out of range')
        if not legacy and ((saved['mastered'] >= len(PREFIXES)).any()
                or (saved['mastered'][:,:-1] < saved['mastered'][:,1:]).any()):
            raise ValueError('Invalid mastered prefix floor history')

    def load_state_dict(self, saved):
        self._validate(saved)
        expected = self.state_dict()
        for name,value in expected.items():
            if isinstance(value,torch.Tensor):
                getattr(self,name).copy_(saved[name].to(self.skill.device))
        self.last_events.zero_()
