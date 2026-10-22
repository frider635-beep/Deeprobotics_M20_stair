"""Pure tensor objectives for route correction and physical contact slip."""
import torch

REWARD_VERSION = 'm20_stair_path_rewards_20261022_v2'
REWARD_WEIGHTS = dict(position=-1.5, velocity=-.25, slip=-.40,
                      down_stall=-.40, down_complete=2.)


def velocity_tracking_reward(raw, effective, measured_body_xy, feedback, std):
    """Follow exactly the command seen by the actor while route feedback owns it."""
    target = torch.where(feedback[:, None], effective[:, :2], raw[:, :2])
    return torch.exp(-(target-measured_body_xy).square().sum(-1)/std**2)


def return_velocity_cost(error, velocity, requested_return_velocity):
    """Allow bounded return motion away from the line, damp motion near the line.

    Wrong-way filter transients never become a rewarded return target. The
    smooth band also limits the incentive to cross the line at high speed.
    """
    blend = ((error.abs()-.025)/.05).clamp(0., 1.)
    requested_return_velocity = requested_return_velocity.clamp(-.18, .18)
    target = torch.where(error*requested_return_velocity < 0.,
                         requested_return_velocity, 0.)*blend
    return ((velocity-target)/.15).square().clamp_max(9.), target


def contact_slip_cost(tangent, support, normal, descending):
    """Lateral support slip on both directions, longitudinal slip on descent.

    Ascent's retained front/rear terms already cover the complete tangent.
    Work per wheel before averaging, so one spinning wheel is not cancelled
    by stationary mates. Airborne wheels contribute neither cost nor counts.
    """
    clean = torch.nan_to_num(tangent[..., :2])
    forward = torch.stack((normal[:, 1], -normal[:, 0]), -1)
    lateral = (clean*normal[:, None]).sum(-1).abs()
    longitudinal = (clean*forward[:, None]).sum(-1).abs()
    denominator = support.sum(-1).clamp_min(1)
    average = lambda value: torch.where(support, value, 0.).sum(-1)/denominator
    per_wheel = (lateral/.15).square().clamp_max(9.)
    per_wheel += .5*(longitudinal/.20).square().clamp_max(9.)*descending[:, None]
    return average(per_wheel), average(lateral), average(longitudinal)


def merged_entry_slip_multiplier(progress):
    """Merge the old entry emphasis with the existing 3->4 emphasis using max.

    The total multiplier stays <=1.5 rather than stacking both bonuses.
    """
    smooth = lambda x: x.clamp(0., 1.).square()*(3.-2.*x.clamp(0., 1.))
    entry = 1.+.5*smooth((progress-.8)/.8)*(1.-smooth(progress-4.5))
    transition = 1.+.5*((progress >= 2.) & (progress <= 4.5))
    return torch.maximum(entry, transition)/transition
