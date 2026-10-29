"""Low-frequency matched-course diagnostics, independent of policy training."""
import json
from pathlib import Path
import torch
from . import stair_path_layout as layout
from .stair_path_course import PREFIXES, EVENTS, VERSION
from ..ascent.ascent_pairing import DIAGNOSTIC_FIELDS, VERSION as PAIRING_VERSION

FIELDS = ('episodes','success',*EVENTS[:6], 'rear_pair_coverage','four_wheel_coverage',
          'max_path_error_m','heading_abs_rad','forward_coordinate_max_m','missing_failure','swing_failure',
          *EVENTS[8:],*DIAGNOSTIC_FIELDS,
          'physical_ready','prefix_quality_blocked','height_quality_blocked','retreat_prevented_by_window',
          'episode_front_pair_quality','episode_rear_pair_quality')
SHAPE = (9,6,len(PREFIXES),2)


class CourseDiagnostics:
    def __init__(self, device):
        self.totals = torch.zeros(9*6*len(PREFIXES)*2,len(FIELDS),dtype=torch.float64,device=device)
        self.first_uncovered = torch.zeros(18,device=device,dtype=torch.float64)
        self.practice_first_uncovered = torch.zeros_like(self.first_uncovered)
        self.first_iteration = None
        self.version = VERSION

    def observe(self, env, ids, iteration):
        if not len(ids):
            return
        observer = env._ascent_entry_observer
        course = env._ascent_adaptive_curriculum
        goals = observer.episode_goals[ids]
        prefix = (goals[:,None] == goals.new_tensor(PREFIXES)).long().argmax(-1)
        rows = observer.episode_terrain_rows[ids]
        columns = env.scene.terrain.terrain_types[ids]
        fast = (env.command_manager.get_command('base_velocity')[ids,0] > .6).long()
        indices = (((columns*6+rows)*len(PREFIXES)+prefix)*2+fast)
        required = observer.sequence.targets <= goals[:,None,None]
        success = observer.prefix_ready[ids] & ((goals < 18) | observer.top_ready[ids])
        rear = (observer.sequence.rear_pair_paid[ids]*required[:,0]).sum(-1)/goals
        coverage = (observer.sequence.covered[ids]*required).sum((1,2))/(4*goals)
        distance = observer.path_forward_max[ids]
        values = torch.cat((torch.ones(len(ids),1,device=goals.device),success[:,None],
            course.last_events[ids,:6],rear[:,None],coverage[:,None],observer.path_max[ids,None],
            (observer.path_heading_total[ids]/observer.path_samples[ids].clamp_min(1))[:,None],
            distance[:,None],(observer.failure_reason[ids] == 1)[:,None],
            (observer.failure_reason[ids] == 4)[:,None],course.last_events[ids,8:]),-1)
        diagnostics = getattr(observer,'pairing_diagnostics',None)
        extra = (diagnostics.values(observer,ids) if diagnostics is not None else
                 torch.zeros(len(ids),len(DIAGNOSTIC_FIELDS),device=goals.device))
        if diagnostics is not None: self.version = PAIRING_VERSION
        evidence=(course.evidence_last[ids] if hasattr(course,'evidence_last') else torch.zeros(len(ids),6,device=goals.device))
        self.totals.index_add_(0,indices,torch.cat((values,extra,evidence),-1).to(self.totals.dtype))
        # Earliest required tread not covered by all four wheels at failure.
        # This is a diagnostic location proxy, not an asserted contact cause.
        covered = observer.sequence.covered[ids].all(1)
        uncovered = (~covered & required[:,0]).long().argmax(-1)
        failed = ~success & (~covered & required[:,0]).any(-1)
        practice = torch.isin(columns, ids.new_tensor(layout.NARROW_ASCENT_COLUMNS))
        self.first_uncovered.index_add_(0,uncovered,
            (failed & ~practice).to(self.first_uncovered.dtype))
        self.practice_first_uncovered.index_add_(0,uncovered,
            (failed & practice).to(self.practice_first_uncovered.dtype))
        if self.first_iteration is None:
            self.first_iteration = int(iteration)

    def write(self, writer, log_dir, iteration, stage, force=False):
        if not force and (iteration+1)%100:
            return
        if self.first_iteration is None or writer is None:
            return
        # One small host transfer every100 iterations, not per reset or field.
        packed = self.totals.cpu()
        table = packed.reshape(*SHAPE,len(FIELDS))
        main_columns = [c for c in range(9) if c not in layout.NARROW_ASCENT_COLUMNS]
        practice_columns = list(layout.NARROW_ASCENT_COLUMNS)
        total = table[main_columns].sum((0,1,2,3))
        count = float(total[0])
        if not count:
            return
        for name in FIELDS[1:]:
            writer.add_scalar('Course/'+name,float(total[FIELDS.index(name)])/count,iteration)
        # Keep exploratory narrow-stair failures visible without mixing them
        # into the main course's displayed progress and mechanical diagnostics.
        for label,columns in (('main',main_columns), ('practice',practice_columns)):
            group = table[columns].sum((0,1,2,3))
            episodes = float(group[0])
            if not episodes:continue
            for name in FIELDS[1:]:
                writer.add_scalar(f'Course/{label}/{name}',float(group[FIELDS.index(name)])/episodes,iteration)
            for wheel in ('FL','FR','HL','HR'):
                for tread in range(1,5):
                    stem=f'quality_{wheel}_tread{tread}_'
                    for name,num,den in (
                            ('clean_landing_rate','clean_landings','attempts'),
                            ('scrub_rate','scrubbed_attempts','attempts'),
                            ('peak_clearance_mean_m','peak_clearance_m','attempts'),
                            ('permission_fraction','permitted_s','near_s'),
                            ('face_impulse_per_attempt_ns','filtered_face_impulse_ns','attempts'),
                            ('support_force_mean_n','support_force_n_s','support_time_s'),
                            ('support_slip_mean_m_s','patch_slip_m','support_time_s')):
                        denominator=float(group[FIELDS.index(stem+den)])
                        if denominator > 0:
                            writer.add_scalar(f'Course/{label}/{stem}{name}',
                                float(group[FIELDS.index(stem+num)])/denominator,iteration)
        for label,num,den in (
                ('rear_opportunity_coverage','rear_covered_attempts','rear_attempts'),
                ('rear_mature_opportunity_coverage','rear_covered_mature_attempts','rear_mature_attempts'),
                ('rear_tail_ordered_pair_quality','rear_tail_ordered_pairs','rear_tail_required_pairs'),
                ('rear_tail_strict_success_rate','rear_tail_strict_success','rear_tail_evaluated'),
                ('legacy_target_disagreement_fraction','legacy_spatial_target_disagreement_s','target_opportunity_s'),
                ('short_rear_clearance_deficit_mean_m','short_rear_deficit_m_wheel_s','short_rear_near_wheel_s'),
                ('short_rear_legal_preparation_fraction','short_rear_permitted_wheel_s','short_rear_near_wheel_s'),
                ('short_rear_low_approach_fraction','short_rear_low_approach_wheel_s','short_rear_near_wheel_s'),
                *((f'rear_tread{tread}_{label}',f'rear_tread{tread}_{num}',f'rear_tread{tread}_{den}')
                  for tread in range(1,19) for label,num,den in (
                      ('opportunity_coverage','covered_attempts','attempts'),
                      ('mature_opportunity_coverage','covered_mature_attempts','mature_attempts')))):
            denominator = float(total[FIELDS.index(den)])
            if denominator > 0:
                writer.add_scalar('Course/'+label,float(total[FIELDS.index(num)])/denominator,iteration)
        for column in main_columns:
            cell = table[column].sum((0,1,2))
            if cell[0] > 0:
                writer.add_scalar(f'Course/geometry{column}/success',float(cell[1]/cell[0]),iteration)
                writer.add_scalar(f'Course/geometry{column}/episodes',float(cell[0]),iteration)
        for bucket,label in ((table[main_columns].sum((0,2,3)),'height_row'),
                             (table[main_columns].sum((0,1,3)),'prefix')):
            for index,cell in enumerate(bucket):
                if cell[0] > 0:
                    value = PREFIXES[index] if label == 'prefix' else index
                    writer.add_scalar(f'Course/{label}{value}/success',float(cell[1]/cell[0]),iteration)
                    writer.add_scalar(f'Course/{label}{value}/episodes',float(cell[0]),iteration)
        locations = self.first_uncovered.cpu()
        location_denominator = max(float(locations.sum()),1.)
        for index,value in enumerate(locations):
            writer.add_scalar(f'Course/failure_first_uncovered_tread{index+1}',float(value)/location_denominator,iteration)
        practice_locations = self.practice_first_uncovered.cpu()
        practice_denominator = max(float(practice_locations.sum()),1.)
        for index,value in enumerate(practice_locations):
            writer.add_scalar(f'Course/practice/failure_first_uncovered_tread{index+1}',
                              float(value)/practice_denominator,iteration)
        indices = (table[...,0] > 0).nonzero()
        cells = [dict(column=c,row=r,prefix=PREFIXES[p],speed=(.5,.7)[v],
                      totals=table[c,r,p,v].tolist()) for c,r,p,v in indices.tolist()]
        record = dict(version=self.version,iteration_start=self.first_iteration,iteration_end=int(iteration),
                      sampling_stage=int(stage),fields=FIELDS,cells=cells,
                      failure_first_uncovered_counts=locations.tolist(),
                      practice_failure_first_uncovered_counts=practice_locations.tolist(),
                      distance_definition='maximum_forward_coordinate_from_original_terrain_origin_not_travel_length')
        path = Path(log_dir)/'course_strata.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(record,separators=(',',':'))+'\n')
        self.totals.zero_();self.first_uncovered.zero_();self.practice_first_uncovered.zero_();self.first_iteration=None
