"""Sampled optimization diagnostics; never modify parameter gradients."""
import torch


def gradient_comparison(ppo_loss, anchor_loss, parameters):
    parameters = tuple(p for p in parameters if p.requires_grad)
    if not parameters:
        return {}
    ppo = torch.autograd.grad(ppo_loss,parameters,retain_graph=True,allow_unused=True)
    anchor = torch.autograd.grad(anchor_loss,parameters,retain_graph=True,allow_unused=True)
    zero = ppo_loss.new_zeros(())
    p2 = sum((g.square().sum() for g in ppo if g is not None),zero)
    a2 = sum((g.square().sum() for g in anchor if g is not None),zero)
    dot = sum(((p*a).sum() for p,a in zip(ppo,anchor) if p is not None and a is not None),zero)
    pn,an = p2.sqrt(),a2.sqrt()
    return dict(ppo_grad_norm=pn.detach(),weighted_anchor_grad_norm=an.detach(),
                anchor_to_ppo_grad_ratio=(an/pn.clamp_min(1.e-12)).detach(),
                ppo_anchor_grad_cosine=(dot/(pn*an).clamp_min(1.e-12)).detach(),
                gradient_cosine_valid=((pn > 1.e-12) & (an > 1.e-12)).float().detach())


def parameter_snapshot(parameters):
    return [(p,p.detach().clone()) for p in parameters if p.requires_grad]


def parameter_change(snapshot):
    if not snapshot:
        return {}
    zero = snapshot[0][0].new_zeros(())
    change = sum(((p.detach()-old).square().sum() for p,old in snapshot),zero).sqrt()
    norm = sum((old.square().sum() for _,old in snapshot),zero).sqrt()
    return dict(actor_parameter_update_l2=change,actor_parameter_relative_update=change/norm.clamp_min(1.e-12))
