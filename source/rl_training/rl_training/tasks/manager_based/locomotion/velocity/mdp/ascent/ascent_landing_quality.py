"""Training-only landing credit; never modifies the physical/course ledger."""
import torch
from .ascent_entry_mdp import smooth01

SETTINGS = dict(version='ascent_landing_quality_20261029_v1', scope='all_ascent_wheels',
    lift_margin_m=.03, desired_lift_margin_m=.04, crossing_margin_m=.01,
    inset_m=.04, support_duration_s=.04, min_support_force_n=15.,
    full_support_force_n=40., max_patch_slip_m_s=.10, max_vertical_speed_m_s=.12,
    height_tolerance_m=.03, meaningful_face_force_n=10.,
    reward='once_per_wheel_tread_after_clean_crossing_and_loaded_low_slip_support',
    scrub='latch_per_wheel_tread_no_later_refund', physical_course='unchanged',
    rear_ordered_multiplier=1.75, rear_landing_credit_weight=.12,
    rear_pair_weight=1.5, forward_guidance='short_tread_legal_lift_before_forward')

METRIC_NAMES = ('attempts', 'near_s', 'permitted_s', 'peak_clearance_m',
    'filtered_face_impulse_ns', 'scrubbed_attempts', 'clean_landings',
    'support_time_s', 'support_force_n_s', 'patch_slip_m')
DIAGNOSTIC_FIELDS = tuple(f'quality_{wheel}_tread{tread}_{name}'
    for wheel in ('FL','FR','HL','HR') for tread in range(1,5) for name in METRIC_NAMES)


def forward_factor(bottom, top, offset, permission):
    """Restore horizontal landing guidance smoothly after legal clearance.

    The landing blend restores full guidance after crossing, so descending
    onto the tread cannot turn the horizontal objective off again.
    """
    crossed = smooth01((offset-.01)/.06)
    clearance = smooth01((bottom-top)/.04)
    return crossed+(1.-crossed)*clearance*permission


class LandingQuality:
    def __init__(self, count, device, targets=18):
        shape=(count,4,targets)
        for name in ('attempted','lifted','crossed','scrubbed','paid'):
            setattr(self,name,torch.zeros(shape,dtype=torch.bool,device=device))
        for name in ('duration','score_integral','score','pending_credit','near_s',
                     'permitted_s','face_impulse','support_s','force_integral','slip_m'):
            setattr(self,name,torch.zeros(shape,device=device))
        self.peak=torch.full(shape,-.5,device=device)
        self.best_error=torch.full(shape,float('inf'),device=device)
        self.pair_paid=torch.zeros(count,2,targets,dtype=torch.bool,device=device)
        self.qualified=torch.zeros_like(self.paid)

    def reset(self,ids):
        for name,value in vars(self).items():
            value[ids] = -.5 if name=='peak' else float('inf') if name=='best_error' else 0

    @staticmethod
    def _read(value,index):
        return value.gather(-1,index).squeeze(-1)

    def observe(self,state,permission,position,velocity_z,support,face,force,slip,
                level,first,floor,depth,height,active,dt,contact_point=None):
        """Called before physical coverage can advance the paired target."""
        target=state['target'];index=(target-1)[:,:,None]
        offset=position[...,0]-first[:,None]-(target-1)*depth[:,None]
        bottom=position[...,2]-.09
        top=floor[:,None]+target*height[:,None]
        enabled=state['enabled'] & ~state['transition_hold'] & active[:,None]
        near=enabled & (offset >= -.18) & (offset < depth[:,None]+.09)
        permission=permission & ~state['hold'] & ~state.get('rear_wait',torch.zeros_like(permission))
        self.attempted.scatter_(-1,index,(self._read(self.attempted,index)|near)[:,:,None])
        self.peak.scatter_(-1,index,torch.where(near,
            torch.maximum(self._read(self.peak,index),bottom-top),self._read(self.peak,index))[:,:,None])
        for name,increment in (('near_s',near*dt),('permitted_s',(near & permission)*dt)):
            value=getattr(self,name);value.scatter_add_(-1,index,increment[:,:,None])
        raised=near & permission & ~support & (bottom >= top+SETTINGS['lift_margin_m'])
        lifted=self._read(self.lifted,index)|raised
        self.lifted.scatter_(-1,index,lifted[:,:,None])
        crossing=near & permission & lifted & (offset >= .01) & (offset <= depth[:,None]-.04)
        crossing &= bottom >= top+SETTINGS['crossing_margin_m']
        self.crossed.scatter_(-1,index,(self._read(self.crossed,index)|crossing)[:,:,None])

        # Face contacts are attributed to the unresolved riser, not inferred
        # from a later missing-tread label. Tiny corner-force noise is ignored.
        face_force=torch.linalg.vector_norm(torch.nan_to_num(force[...,:2]),dim=-1)
        hit=near & face & (face_force >= SETTINGS['meaningful_face_force_n'])
        if contact_point is not None:
            # A wheel on a short tread can also touch the NEXT riser. Do not
            # attribute that face to the current landing's approach history.
            edge=first[:,None]+(target-1)*depth[:,None]
            hit &= (contact_point[...,0]-edge).abs() <= .04
            hit &= (contact_point[...,2] >= top-height[:,None]-.03) & (contact_point[...,2] <= top+.03)
        sliding=near & support & (level == target) & (force[...,2] >= SETTINGS['min_support_force_n'])
        sliding &= slip > SETTINGS['max_patch_slip_m_s']
        bad=(hit | sliding) & ~self._read(self.paid,index)
        self.scrubbed.scatter_(-1,index,(self._read(self.scrubbed,index)|bad)[:,:,None])
        self.face_impulse.scatter_add_(-1,index,(face_force*hit*dt)[:,:,None])

        # Retain the same pending, bounded landing-improvement mechanism for
        # each wheel. It is paid only after quality qualification below.
        blend=smooth01((offset-.01)/.06)
        dz=bottom-top-.04*(1.-blend)
        dx=(.06-offset).clamp_min(0.)+(offset-(depth[:,None]-.06)).clamp_min(0.)
        error=(2.*(torch.sqrt(1.+(dz/.08).square())-1.)+
               torch.sqrt(1.+(dx/.15).square())-1.).clamp_max(8.)
        previous=self._read(self.best_error,index)
        improvement=(torch.where(torch.isfinite(previous),previous,error)-error).clamp(0.,1.)
        improving=near & ~self._read(self.scrubbed,index) & (permission | state['hold'])
        self.pending_credit.scatter_(-1,index,(self._read(self.pending_credit,index)+
            improvement*improving).clamp_max(2.)[:,:,None])
        self.best_error.scatter_(-1,index,torch.where(improving,
            torch.minimum(previous,error),previous)[:,:,None])

        # Contact may qualify after physical coverage has advanced the target.
        # Evaluate its actual tread independently; no extra physical hold or
        # delay of the next legal swing is imposed.
        contact_index=(level-1).clamp(0,self.paid.shape[-1]-1)[:,:,None]
        contact_top=floor[:,None]+level*height[:,None]
        contact_offset=position[...,0]-first[:,None]-(level-1)*depth[:,None]
        valid_level=(level >= 1) & (level <= self.paid.shape[-1]) & active[:,None]
        on_tread=valid_level & support & (contact_offset >= SETTINGS['inset_m'])
        on_tread &= contact_offset <= depth[:,None]-SETTINGS['inset_m']
        fz=torch.nan_to_num(force[...,2]).clamp_min(0.)
        good=on_tread & ~face & (fz >= SETTINGS['min_support_force_n'])
        good &= (slip <= SETTINGS['max_patch_slip_m_s']) & (velocity_z.abs() <= SETTINGS['max_vertical_speed_m_s'])
        good &= (bottom-contact_top).abs() <= SETTINGS['height_tolerance_m']
        good &= self._read(self.crossed,contact_index) & ~self._read(self.scrubbed,contact_index)
        # Zero inactive/old tread timers; only consecutive good support counts.
        self.duration *= torch.nn.functional.one_hot(contact_index.squeeze(-1),self.paid.shape[-1])
        self.score_integral *= self.duration > 0
        duration=torch.where(good,self._read(self.duration,contact_index)+dt,0.)
        quality=(.5+.5*smooth01((fz-SETTINGS['min_support_force_n'])/
            (SETTINGS['full_support_force_n']-SETTINGS['min_support_force_n'])))
        quality *= 1.-.5*(slip/SETTINGS['max_patch_slip_m_s']).clamp(0.,1.)
        integral=torch.where(good,self._read(self.score_integral,contact_index)+quality*dt,0.)
        self.duration.scatter_(-1,contact_index,duration[:,:,None])
        self.score_integral.scatter_(-1,contact_index,integral[:,:,None])
        qualified=good & (duration >= SETTINGS['support_duration_s']-1.e-7)
        self.qualified.zero_()
        self.qualified.scatter_(-1,contact_index,qualified[:,:,None])
        self.score.scatter_(-1,contact_index,torch.where(qualified,
            integral/duration.clamp_min(dt),self._read(self.score,contact_index))[:,:,None])
        for name,increment in (('support_s',on_tread*dt),('force_integral',fz*on_tread*dt),
                               ('slip_m',slip*on_tread*dt)):
            getattr(self,name).scatter_add_(-1,contact_index,increment[:,:,None])

    def credit(self,ledger,valid,goals):
        required=ledger.targets <= goals[:,None,None]
        clean=~ledger.style_bad_treads.repeat_interleave(2,1)
        new=self.qualified & ledger.covered & ~self.paid & ~self.scrubbed & clean & required
        new &= valid[:,None,None] & ~ledger.failed[:,None,None]
        self.paid |= new
        weighted=new*self.score
        credit=(weighted*self.pending_credit).sum(-1)
        self.pending_credit *= ~new
        both=self.paid.reshape(-1,2,2,self.paid.shape[-1]).all(2)
        physical=torch.stack((ledger.pair_paid,ledger.rear_pair_paid),1)
        pair=both & physical & ~self.pair_paid & ~ledger.style_bad_treads
        pair &= valid[:,None,None] & ~ledger.failed[:,None,None] & required[:,:2]
        self.pair_paid |= pair
        pair_score=self.score.reshape(-1,2,2,self.score.shape[-1]).amin(2)
        return dict(ordered=weighted.sum(-1),dwell=weighted.sum(-1),new_mask=new,
            front_credit=credit[:,:2].sum(-1),rear_credit=credit[:,2:].sum(-1),
            front_paired=(pair[:,0]*pair_score[:,0]).sum(-1),
            rear_paired=(pair[:,1]*pair_score[:,1]).sum(-1))

    def diagnostics(self,ids):
        t=min(4,self.paid.shape[-1])
        values=(self.attempted,self.near_s,self.permitted_s,
            torch.where(self.attempted,self.peak,torch.zeros_like(self.peak)),
            self.face_impulse,self.scrubbed,self.paid,self.support_s,self.force_integral,self.slip_m)
        packed=torch.stack([v[ids,:,:t] for v in values],-1)
        if t < 4:
            packed=torch.nn.functional.pad(packed,(0,0,0,4-t))
        return packed.reshape(len(ids),-1).float()
