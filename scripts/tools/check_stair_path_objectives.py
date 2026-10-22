#!/usr/bin/env python3
"""Semantic counterexamples for reward alignment and rolling/contact slip."""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    obj = d.training_module('tasks.manager_based.locomotion.velocity.mdp.path.stair_path_objectives')
    contact = d.training_module('tasks.manager_based.locomotion.velocity.mdp.ascent.ascent_entry_mdp')
    raw = torch.tensor([[.5,0.,0.]]*4)
    effective = raw.clone(); effective[:,1] = -.12
    feedback = torch.tensor([True,True,False,False])
    actual = torch.tensor([[.5,-.12],[.5,0.],[.5,-.12],[.5,0.]])
    reward = obj.velocity_tracking_reward(raw,effective,actual,feedback,2**-.5)
    assert reward[0]>reward[1] and reward[3]>reward[2]
    assert not raw[:,1:].any(), 'Feedback modified the raw gait command'
    mirrored = obj.velocity_tracking_reward(raw,-effective,-actual,feedback,2**-.5)
    assert torch.allclose(mirrored[:2],reward[:2])
    error = torch.tensor([.2,.2,.2,.2,0.,-.2])
    speed = torch.tensor([-.12,0.,.12,-.40,-.12,.12])
    requested = torch.tensor([-.12,-.12,-.12,-.12,-.12,.12])
    costs,target = obj.return_velocity_cost(error,speed,requested)
    assert costs[0]<costs[1]<costs[2] and costs[3]>costs[0]
    assert costs[4]>0 and target[4]==0 and costs[5]==costs[0]
    _,wrong = obj.return_velocity_cost(torch.tensor([.2]),torch.tensor([.1]),torch.tensor([.1]))
    assert wrong.item()==0., 'Wrong-way filter transient became a target'
    _,limited = obj.return_velocity_cost(torch.tensor([.2]),torch.tensor([-.18]),torch.tensor([-1.]))
    assert torch.allclose(limited,torch.tensor([-.18])), 'Return target exceeded the route correction limit'
    # Actual wheel kinematics: omega x contact radius cancels COM velocity in rolling.
    radius = .09
    position = torch.zeros(3,4,3);position[...,2] = radius
    point = position.clone();point[...,2] = 0.
    force = torch.zeros_like(point);force[...,2] = 100.
    linear = torch.zeros_like(position);linear[0,:,0] = .6
    angular = torch.zeros_like(position);angular[:2,:,1] = .6/radius
    angular[2,0,1] = .6/radius # only one wheel spinning, others supported/quiet
    _,support,_,_,tangent = contact.contact_patch_velocity(position,linear,angular,point,force,return_tangent=True)
    normal = torch.tensor([[0.,1.]]*3)
    cost,lateral,longitudinal = obj.contact_slip_cost(tangent,support,normal,torch.ones(3,dtype=torch.bool))
    assert cost[0]<1.e-10 and longitudinal[0]<1.e-6
    assert cost[1]>cost[2]>0. and lateral.max()==0.
    assert torch.allclose(cost[2]*4,cost[1])
    no_load = torch.zeros_like(support)
    air,_,_ = obj.contact_slip_cost(tangent,no_load,normal,torch.ones(3,dtype=torch.bool))
    assert air.max()==0.
    # Ascent full-tangent terms own longitudinal slip; new route cost does not duplicate it.
    up_cost,_,_ = obj.contact_slip_cost(tangent,support,normal,torch.zeros(3,dtype=torch.bool))
    assert up_cost.max()==0.
    lateral_tangent = torch.zeros_like(tangent);lateral_tangent[...,1]=.15
    sideways,_,_ = obj.contact_slip_cost(lateral_tangent,support,normal,torch.zeros(3,dtype=torch.bool))
    assert (sideways>0.).all()
    mirror,_,_ = obj.contact_slip_cost(-lateral_tangent,support,normal,torch.zeros(3,dtype=torch.bool))
    assert torch.equal(sideways,mirror)
    angle = .7;rotation = torch.tensor([[torch.cos(torch.tensor(angle)),-torch.sin(torch.tensor(angle))],
                                       [torch.sin(torch.tensor(angle)),torch.cos(torch.tensor(angle))]])
    rotated = torch.zeros_like(tangent);rotated[...,:2]=tangent[...,:2]@rotation.T
    rotated_cost,_,_ = obj.contact_slip_cost(rotated,support,normal@rotation.T,torch.ones(3,dtype=torch.bool))
    assert torch.allclose(cost,rotated_cost,atol=1.e-6)
    progress = torch.linspace(-1.,7.,1000)
    old = 1.+.5*((progress>=2.)&(progress<=4.5))
    merged = old*obj.merged_entry_slip_multiplier(progress)
    assert merged.max()<=1.5 and (merged>=old).all()
    assert (merged[(progress>1.)&(progress<1.5)]>1.).all()
    result = dict(reward_version=obj.REWARD_VERSION, weights=obj.REWARD_WEIGHTS,
        effective_command_tracking_preferred=True,raw_commands_untouched=True,
        inward_return_preferred_over_stall_and_outward=True,near_route_damping=True,
        wrong_way_target_rejected=True,normal_rolling_not_penalized=True,
        descent_longitudinal_spin_penalized=True,single_wheel_spin_detected=True,
        airborne_spin_excluded=True,ascent_longitudinal_not_double_counted=True,
        mirror_and_route_rotation_invariant=True,entry_emphasis_merged_capped_1p5=True,
        examples=dict(tracking_rewards=reward.tolist(),return_costs=costs.tolist(),
                      rolling_all_spin_one_spin_costs=cost.tolist()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
