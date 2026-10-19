#!/usr/bin/env python3
"""CPU counterexamples for clearance severity and sample-aware gait metrics.

Executes production reward functions with the established Isaac import stubs.
These checks verify safety signals; they do not establish collision-free gait.
"""

import json
from types import SimpleNamespace

import torch

from check_gait_safety import load
from check_stair_rewards import MDP
from check_m20_curriculum import production_namespace


def check():
    ns = load(MDP / "stair_teacher.py")
    diagnostics = production_namespace()
    for key in ("descent_front_pair_metrics", "export_teacher_metrics"):
        ns[key] = diagnostics[key]
    tuning = ns["TUNING"]
    barrier = ns["interleg_clearance_cost"]
    wheel_distances = torch.tensor([.35, .28, .27, .25, .24, .22, .18], requires_grad=True)
    costs = barrier(wheel_distances, tuning["entry_wheel_spacing_min"],
                    tuning["interleg_wheel_critical"], tuning["entry_wheel_spacing_scale"])
    assert torch.all(costs[:2] == 0)
    assert torch.all(torch.diff(costs[1:]) > 0)
    costs.sum().backward()
    assert torch.all(wheel_distances.grad[2:] < 0)
    assert abs(float(wheel_distances.grad[-1])) > 10*abs(float(wheel_distances.grad[2]))
    old = 2*(torch.sqrt(1+((.28-.18)/.05)**2*torch.ones(1))-1)
    assert costs[-1] > 10*old

    risk = ns["descent_transfer_risk"]
    moving = torch.ones(1, 2, dtype=torch.bool)
    wheel = torch.tensor([[.30, .50]])
    shank = torch.tensor([[.25, .35]])
    still, _ = risk(wheel, shank, moving)
    closing, unsafe = risk(wheel, shank, moving, torch.tensor([[.25, 0.]]))
    faster, _ = risk(wheel, shank, moving, torch.tensor([[.50, 0.]]))
    separating, _ = risk(wheel, shank, moving, torch.tensor([[-.5, 0.]]))
    assert still == 0 and separating == 0 and faster > closing > 0 and unsafe
    # One unsafe side retains 75% of the equivalent two-sided risk, so the
    # safe side cannot halve the collision warning.
    both, _ = risk(wheel[:, :1].expand(1, 2), shank[:, :1].expand(1, 2), moving,
                   torch.full((1, 2), .25))
    assert torch.isclose(closing, .75*both)
    inactive, _ = risk(wheel, shank, torch.zeros_like(moving), torch.full((1, 2), .5))
    assert inactive == 0

    # Knee swings toward a fixed rear wheel while both axles remain fixed.
    # Axle closing is zero, but the front-calf approach must still be seen.
    knee = torch.tensor([[[0., 0., 1.], [0., 0., 1.]]])
    knee_velocity = torch.tensor([[[.2, 0., 0.], [.2, 0., 0.]]])
    env = SimpleNamespace(_m20_front_knee_body_ids=[0, 1], scene={
        "robot": SimpleNamespace(data=SimpleNamespace(body_pos_w=knee, body_lin_vel_w=knee_velocity))})
    wheels = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [.2, 0., .5], [.2, 0., .5]]])
    pair_closing, calf_closing = ns["_same_side_interleg_closing_speeds"](env,wheels,torch.zeros_like(wheels))
    assert torch.allclose(pair_closing, torch.zeros_like(pair_closing))
    assert torch.allclose(calf_closing, torch.full_like(calf_closing, .1))
    _, before = ns["_same_side_interleg_distances"](env,wheels)
    env.scene["robot"].data.body_pos_w = knee + .001*knee_velocity
    _, after = ns["_same_side_interleg_distances"](env,wheels)
    assert torch.allclose((before-after)/.001, calf_closing, atol=1.0e-4)

    # Two completed episodes: one visited drop three and one did not. The
    # sufficient statistics must describe only the observed posture samples.
    env = SimpleNamespace(num_envs=2, device="cpu", episode_length_buf=torch.full((2,), 3))
    ns["_legacy_gait_quality_metrics"] = lambda env, ids: {}
    zero = torch.zeros(2, 2)
    mask = torch.zeros(2, 2, dtype=torch.bool)
    ns["_update_pose_stats"](env,torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),
        zero,mask,mask,zero,zero,zero,zero,mask,zero,torch.zeros(2),
        torch.zeros(2,dtype=torch.bool),torch.zeros(2,4,dtype=torch.bool),torch.ones(2,4,dtype=torch.bool))
    support = torch.tensor([[True, True], [False, False]])
    forward = torch.tensor([[.10, .14], [0., 0.]])
    ns["_update_descent_front_posture_stats"](env,torch.tensor([2,0]),support,mask,
        torch.ones(2,2)*.4,forward,zero,zero,zero,zero)
    metric = "descent_step3_front_support_forward_reach_mean_m"
    seen = ns["gait_quality_metrics"](env,torch.tensor([0]))
    empty = ns["gait_quality_metrics"](env,torch.tensor([1]))
    total = seen[f"__weighted_total__/{metric}"] + empty[f"__weighted_total__/{metric}"]
    count = seen[f"__weighted_count__/{metric}"] + empty[f"__weighted_count__/{metric}"]
    assert torch.isclose(torch.as_tensor(total/count), torch.tensor(.12))
    assert seen["descent_step3_front_support_valid"] == 1
    assert empty["descent_step3_front_support_valid"] == 0
    assert all(torch.isfinite(value).all() if isinstance(value,torch.Tensor) else True for value in empty.values())
    return {"wheel_clearance_m":wheel_distances.detach().tolist(),
            "wheel_clearance_cost":costs.detach().tolist(),
            "single_side_closing_cost":float(closing),
            "faster_closing_cost":float(faster),
            "moving_calf_detected_without_axle_closing":True,
            "third_drop_zero_sample_dilution_prevented":True}


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(check(),indent=2))
