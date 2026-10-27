"""Episode-stratified sampling readiness; pure Torch, no simulator or policy.

The old iteration schedule is kept only to identify legacy migration sources.
Neither the total iteration budget nor a timeout can authorize a transition.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch

from . import stair_path_layout as layout
from .stair_path_course import PREFIXES

METRICS = ('success', 'missing', 'swing', 'paired', 'coverage', 'lateral', 'heading', 'slip')
SHAPE = (9, 6, len(PREFIXES), 2)


@dataclass(frozen=True)
class ReadinessConfig:
    block_iterations: int = 200
    history_blocks: int = 3
    min_hold_iterations: int = 1000
    transition_iterations: int = 1000
    consecutive_checks: int = 2
    min_stratum_episodes: int = 20
    min_comparable_fraction: float = .50
    min_geometry_fraction: float = 2/3
    low_success_warning: float = .25
    plateau_fraction: float = .625
    # Absolute changes per 1000 iterations; success/coverage use fractions.
    slope_limits: tuple = (.030, .025, .025, .020, .020, .010, math.pi/90, .010)
    frontier_drop_tolerance: float = .02
    # A weak group is allowed to learn at a small share; a large, persistently
    # unsuccessful population causes a one-step retreat rather than more load.
    demotion_group_success: float = .20
    demotion_population_fraction: float = .20
    demotion_overall_success: float = .35
    demotion_checks: int = 2
    retry_success_gain: float = .05


class ReadinessSampler:
    def __init__(self, device='cpu', start_iteration=0, config=None, mode='manual'):
        self.config = config or ReadinessConfig()
        self.device = device
        self.stage = 0
        self.mode = mode
        self.transition_start = None
        self.transition_from = None
        self.transition_origin_shares = None
        self.stable_since = int(start_iteration)
        self.block_start = int(start_iteration)
        self.last_iteration = int(start_iteration)
        self.population_pending = 0.
        self.population_long_fraction = 0.
        self.checks = 0
        self.bad_checks = 0
        self.retry_lock = None
        self.history = []
        self.totals = torch.zeros(math.prod(SHAPE), 1+len(METRICS), device=device, dtype=torch.float64)
        self.status = 'collecting'
        self.reasons = ['insufficient_history']
        self.diagnostics = {}

    def schedule(self, iteration):
        blend = (min(max((iteration-self.transition_start)/self.config.transition_iterations, 0.), 1.)
                 if self.transition_start is not None else 1.)
        return self.stage, blend

    def probabilities(self, iteration):
        stage, blend = self.schedule(iteration)
        old = self.transition_origin_shares or layout.READINESS_GROUP_SHARES[stage]
        new = layout.READINESS_GROUP_SHARES[stage]
        return layout.probabilities_for_shares(tuple(a+(b-a)*blend for a,b in zip(old,new)))

    def observe(self, columns, rows, goals, speeds, values):
        """Add finished episodes with their ORIGINAL task labels before reset."""
        if self.mode != 'auto' or self.transition_start is not None or not len(columns):
            return
        prefixes = torch.tensor(PREFIXES, device=goals.device)
        valid = (goals[:,None] == prefixes).any(-1)
        valid &= (columns >= 0) & (columns < SHAPE[0]) & (rows >= 0) & (rows < SHAPE[1])
        # Invalid data must not silently become a small, apparently stable sample.
        if not valid.all() or values.shape != (len(columns), len(METRICS)) or not torch.isfinite(values).all():
            raise ValueError('Invalid readiness episode observations')
        prefix = (goals[:,None] == prefixes).long().argmax(-1)
        indices = (((columns*SHAPE[1]+rows)*SHAPE[2]+prefix)*2+(speeds > .6).long())
        packed = torch.cat((torch.ones(len(columns), 1, device=values.device), values), -1)
        self.totals.index_add_(0, indices, packed.to(dtype=self.totals.dtype, device=self.totals.device))

    def advance(self, iteration):
        iteration = int(iteration)
        if iteration < self.last_iteration:
            raise ValueError('Readiness iteration moved backwards')
        self.last_iteration = iteration
        cfg = self.config
        if self.transition_start is not None:
            if iteration-self.transition_start < cfg.transition_iterations or self.population_pending > 1.e-6:
                self.status = 'transitioning'
                return
            self.transition_start = None
            self.transition_from = None
            self.transition_origin_shares = None
            self.stable_since = iteration
            self.block_start = iteration
            self.history.clear()
            self.totals.zero_()
            self.checks = 0
            self.bad_checks = 0
            self.status = 'collecting'
        if self.mode == 'manual':
            self.status = 'manual'
            self.reasons = []
            return
        if iteration-self.block_start < cfg.block_iterations:
            return
        # A discontinuity cannot fabricate multiple evidence windows.
        if iteration-self.block_start > 2*cfg.block_iterations:
            self.history.clear()
            self.totals.zero_()
            self.block_start = iteration
            self.checks = 0
            self.status = 'collecting'
            self.reasons = ['observation_gap']
            return
        self.history.append(dict(end=iteration, totals=self.totals.detach().cpu().clone().reshape(*SHAPE, -1),
                                 long_fraction=self.population_long_fraction))
        self.history = self.history[-cfg.history_blocks:]
        self.totals.zero_()
        self.block_start = iteration
        if len(self.history) < cfg.history_blocks:
            self.reasons = ['insufficient_history']
            return
        evidence, plateau, reasons = self._assess()
        bad = evidence and (self.diagnostics.get('low_success_population',0.) >= cfg.demotion_population_fraction-1.e-9
                            or (self.diagnostics.get('success',1.) < cfg.demotion_overall_success
                                and self.diagnostics.get('success_slope_per_1000',1.) <= cfg.slope_limits[0]))
        self.bad_checks = self.bad_checks+1 if bad else 0
        if bad:
            self.checks = 0
            self.status = 'low_success_hold'
            self.reasons = ['low_success_population']
            if self.bad_checks >= cfg.demotion_checks and self.stage > 0:
                groups = [name for name in ('easy','middle','hard') if self.diagnostics.get(name+'_demotion',0.)]
                self.retry_lock = dict(target_stage=self.stage, baselines={
                    name:self.diagnostics[name+'_success'] for name in groups} if groups else {'overall':self.diagnostics['success']})
                self._transition(self.stage-1,iteration)
            return
        if self.stage == len(layout.READINESS_GROUP_SHARES)-1:
            self.status = 'final'
            return
        if self.retry_lock and self.stage+1 >= self.retry_lock['target_stage']:
            recovered = all(self.diagnostics.get('success' if name == 'overall' else name+'_success',0.) >=
                            value+cfg.retry_success_gain for name,value in self.retry_lock['baselines'].items())
            if not recovered:
                plateau = False
                reasons.append('retry_cooldown')
            else:
                self.retry_lock = None
        held = iteration-self.stable_since >= cfg.min_hold_iterations
        if not held:
            reasons.append('minimum_hold')
        ready = evidence and plateau and held
        self.checks = self.checks+1 if ready else 0
        self.reasons = reasons
        self.status = ('confirming' if ready else 'learning' if evidence else 'collecting')
        if self.checks >= cfg.consecutive_checks:
            self._transition(self.stage+1,iteration)

    def _transition(self, target, iteration):
        stage, blend = self.schedule(iteration)
        origin = self.transition_origin_shares or layout.READINESS_GROUP_SHARES[stage]
        dest = layout.READINESS_GROUP_SHARES[stage]
        self.transition_origin_shares = tuple(a+(b-a)*blend for a,b in zip(origin,dest))
        self.transition_from = self.stage
        self.stage = target
        self.transition_start = iteration
        self.stable_since = None
        self.history.clear()
        self.totals.zero_()
        self.checks = self.bad_checks = 0
        self.status = 'transitioning'
        self.reasons = []

    def request(self, mode=None, stage=None, iteration=None):
        """Explicit launch-time mode/stage override; no environment reset here."""
        mode = ('manual' if stage is not None else self.mode) if mode is None else mode
        if mode not in ('manual','auto') or (stage is not None and (type(stage) is not int
                or not 0 <= stage < len(layout.READINESS_GROUP_SHARES) or mode != 'manual')):
            raise ValueError('Invalid sampling mode/stage request')
        iteration = self.last_iteration if iteration is None else int(iteration)
        if self.mode != mode:
            self.mode = mode
            self.history.clear()
            self.totals.zero_()
            self.checks = self.bad_checks = 0
            self.block_start = iteration
            if self.transition_start is None: self.stable_since = iteration
        if stage is not None and stage != self.stage:
            self._transition(stage,iteration)

    def _assess(self):
        cfg = self.config
        raw = torch.stack([b['totals'] for b in self.history])
        count = raw[...,0]
        common = (count >= cfg.min_stratum_episodes).all(0)
        # Compare only identical geometry/height/prefix/speed strata. Both
        # speeds and every geometry must retain sufficient comparable coverage.
        coverage = (count*common).sum((2,3))/count.sum((2,3)).clamp_min(1)
        usable = (coverage >= cfg.min_comparable_fraction).all((0,2)) & common.any((1,2)).all(-1)
        if int(usable.sum()) < math.ceil(SHAPE[0]*cfg.min_geometry_fraction) or not all(
                usable[list(columns)].any() for columns in (
                    layout.ASCENT_EASY_COLUMNS,layout.ASCENT_MIDDLE_COLUMNS,layout.ASCENT_HARD_COLUMNS)):
            self.diagnostics = {'comparable_fraction_min':float(coverage.min())}
            return False, False, ['insufficient_comparable_samples']
        weight = count[0]*common
        weight = weight/weight.sum((1,2),keepdim=True).clamp_min(1)
        # Geometry and speed each have fixed, equal importance regardless of
        # short failures producing many more reset episodes than long courses.
        rates = raw[...,1:]/count.clamp_min(1)[...,None]
        per_geometry = (rates*weight[None,...,None]).sum((2,3)).mean(2)
        scores = per_geometry[:,usable].mean(1)
        latest = scores[-1]
        diagnostics = dict(zip(METRICS, latest.tolist()))
        diagnostics['comparable_fraction_min'] = float(coverage[:,usable].min())
        diagnostics['comparable_geometry_fraction'] = float(usable.double().mean())
        diagnostics['low_success_warning'] = float(latest[0] < cfg.low_success_warning)
        reasons = []
        x = torch.tensor([b['end'] for b in self.history],dtype=torch.float64)/1000
        x -= x.mean()
        low_population = 0.
        for group_index,(name,columns) in enumerate((
                ('easy',layout.ASCENT_EASY_COLUMNS),
                ('middle',layout.ASCENT_MIDDLE_COLUMNS),
                ('hard',layout.ASCENT_HARD_COLUMNS))):
            selected = [c for c in columns if usable[c]]
            curve = per_geometry[:,selected,0].mean(1)
            value = float(curve[-1])
            slope = float((x*(curve-curve.mean())).sum()/x.square().sum().clamp_min(1.e-9))
            diagnostics[name+'_success'] = value
            diagnostics[name+'_success_slope_per_1000'] = slope
            weak = value < cfg.demotion_group_success and slope <= cfg.slope_limits[0]
            diagnostics[name+'_demotion'] = float(weak)
            if weak: low_population += layout.READINESS_GROUP_SHARES[self.stage][group_index]
        diagnostics['low_success_population'] = low_population
        # Long-course exposure is diagnostic, not an absolute promotion floor.
        diagnostics['long_fraction'] = self.history[-1]['long_fraction']
        # Regressing to shorter/lower tasks is not a successful plateau.
        pref = torch.tensor(PREFIXES,dtype=torch.float64)[None,None,:,None]/18
        height = torch.arange(SHAPE[1],dtype=torch.float64)[None,:,None,None]/(SHAPE[1]-1)
        frontier = ((count*(pref+height)/2).sum((2,3))/count.sum((2,3)).clamp_min(1))[:,usable].mean((1,2))
        if float(frontier[-1]-frontier[0]) < -cfg.frontier_drop_tolerance:
            reasons.append('curriculum_retreat')
        slopes = (x[:,None]*(scores-scores.mean(0))).sum(0)/x.square().sum().clamp_min(1.e-9)
        limits = torch.tensor(cfg.slope_limits,dtype=torch.float64)
        fraction = float((slopes.abs() <= limits).double().mean())
        # Five of eight signals suffice, but a clearly improving success curve
        # should keep receiving practice even if unrelated costs are flat.
        plateau = fraction >= cfg.plateau_fraction and bool(slopes[0].abs() <= limits[0]) and not reasons
        # A materially worsening signal blocks progression even if others flatten.
        directions = torch.tensor([1,-1,-1,1,1,-1,-1,-1],dtype=torch.float64)
        if (slopes*directions < -2*limits).any():
            plateau = False
            reasons.append('regression')
        if not plateau: reasons.append('not_plateaued')
        diagnostics.update(plateau_fraction=fraction, frontier=float(frontier[-1]))
        diagnostics.update({name+'_slope_per_1000':float(value) for name,value in zip(METRICS,slopes)})
        self.diagnostics = diagnostics
        return True,plateau,reasons

    def telemetry(self):
        result = dict(ready_checks=self.checks, history_blocks=len(self.history),
                      manual_mode=float(self.mode == 'manual'),
                      stalled=float(self.diagnostics.get('low_success_warning',0.) and self.diagnostics.get('plateau_fraction',0.) >= self.config.plateau_fraction),
                      transitioning=float(self.transition_start is not None),bad_checks=self.bad_checks,
                      retry_blocked=float(self.retry_lock is not None),
                      transition_direction=(self.stage-self.transition_from if self.transition_from is not None else 0))
        for reason in ('insufficient_history','insufficient_comparable_samples','minimum_hold','not_plateaued',
                       'regression','curriculum_retreat','low_success_population','retry_cooldown'):
            result['blocked_'+reason] = float(reason in self.reasons)
        result.update(self.diagnostics)
        return result

    def state_dict(self):
        names = ('mode','stage','transition_start','transition_from','transition_origin_shares','stable_since','block_start','last_iteration','checks','bad_checks','retry_lock',
                 'population_pending','population_long_fraction','status','reasons','diagnostics','history')
        state = {name:deepcopy(getattr(self,name)) for name in names}
        state.update(version=layout.SAMPLING_VERSION,config=asdict(self.config),prefixes=list(PREFIXES),totals=self.totals.detach().cpu().clone())
        return state

    def reset_evidence(self):
        """A different inner course needs fresh trends, but keeps quotas/ramp."""
        self.totals.zero_()
        self.history=[];self.checks=0;self.bad_checks=0;self.diagnostics={}
        self.status='collecting';self.reasons=['insufficient_history']
        self.block_start=self.last_iteration
        if self.transition_start is None:
            self.stable_since=self.last_iteration

    def load_state_dict(self, saved, migrate_v4=False):
        if saved.get('prefixes') != list(PREFIXES):
            if not migrate_v4 or 'prefixes' in saved:
                raise ValueError('Readiness prefix schema mismatch; v4 migration must be explicit')
            # Old evidence has only five prefix bins. Keep stage/ramp/retry
            # state, but require new comparable windows under the new course.
            legacy = saved.get('totals')
            if not isinstance(legacy,torch.Tensor) or legacy.shape != (9*6*5*2,9):
                raise ValueError('Invalid v4 readiness buffer')
            saved = deepcopy(saved)
            saved.update(prefixes=list(PREFIXES),totals=torch.zeros_like(self.totals,device='cpu'),
                history=[],checks=0,bad_checks=0,diagnostics={},status='collecting',
                reasons=['insufficient_history'],block_start=saved['last_iteration'])
            if saved.get('transition_start') is None:
                saved['stable_since'] = saved['last_iteration']
        if saved.get('version') != layout.SAMPLING_VERSION or saved.get('config') != asdict(self.config):
            raise ValueError('Readiness sampling version/config mismatch')
        if type(saved.get('stage')) is not int or not 0 <= saved['stage'] < len(layout.READINESS_GROUP_SHARES):
            raise ValueError('Invalid readiness stage')
        if saved.get('mode') not in ('manual','auto'):
            raise ValueError('Invalid sampling mode')
        origin = saved.get('transition_origin_shares')
        if origin is not None and (len(origin) != 3 or min(origin) <= 0 or abs(sum(origin)-1) > 1.e-9):
            raise ValueError('Invalid sampling transition origin')
        if (saved.get('transition_start') is None) != (origin is None):
            raise ValueError('Missing sampling transition origin')
        for key in ('block_start','last_iteration','checks'):
            if type(saved.get(key)) is not int or saved[key] < 0: raise ValueError('Invalid readiness '+key)
        for key in ('transition_start','stable_since'):
            if saved.get(key) is not None and (type(saved[key]) is not int or saved[key] < 0):
                raise ValueError('Invalid readiness '+key)
        if (saved.get('stable_since') is None) != (saved.get('transition_start') is not None):
            raise ValueError('Invalid readiness transition state')
        tensors = [saved.get('totals')]+[b['totals'].reshape_as(self.totals) for b in saved.get('history',[])]
        if len(saved.get('history',[])) > self.config.history_blocks or any(
                not isinstance(t,torch.Tensor) or t.shape != self.totals.shape or not torch.isfinite(t).all() or (t < 0).any()
                for t in tensors):
            raise ValueError('Invalid readiness evidence buffers')
        for name,value in saved.items():
            if name not in ('version','config','prefixes','totals'): setattr(self,name,deepcopy(value))
        self.totals.copy_(saved['totals'].to(self.device))


def validate_legacy_migration(saved, terrain_types, allowed):
    """Only an explicit, pre-first-switch v3 checkpoint can seed phase zero."""
    iteration = saved.get('iteration')
    if (not allowed or saved.get('version') != layout.LEGACY_SAMPLING_VERSION
            or saved.get('initialized') is not True or type(iteration) is not int
            or not 0 <= iteration < layout.SAMPLING_BOUNDARIES[0]):
        raise ValueError('v3 -> v4 migration requires --stair_sampling_migrate_v3 and a pre-3000 checkpoint')
    expected = layout.integer_sample_counts(layout.staged_sampling_probabilities(0),len(terrain_types))
    if torch.bincount(terrain_types,minlength=layout.TERRAIN_COLUMNS).cpu().tolist() != expected:
        raise ValueError('v3 phase-zero terrain quotas do not match; refusing to reset learned courses')
    return iteration
