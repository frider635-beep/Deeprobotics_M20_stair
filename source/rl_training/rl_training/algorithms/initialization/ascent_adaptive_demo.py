"""Contact-verified entry demonstrations, decayed inside the PPO optimizer."""
from pathlib import Path
from types import MethodType
import copy
import hashlib
import numpy as np
import torch
from tensordict import TensorDict


@torch.no_grad()
def validate_demo_authority(algorithm):
    """Reject demonstration targets outside the actual frozen-source envelope."""
    data = algorithm._contact_demo
    if data is None: return dict(samples=0, maximum_normalized_correction=0.)
    from ...models.ascent_landing_actor import AscentLandingMLPModel
    from ...models.ascent_sequence_actor import sequence_gate
    # Demonstrations were collected with CPU float32 inference. Training enables
    # CUDA TF32; checking tight action bounds with that reduced precision can
    # reject the same weights and data. Validate a CPU copy without changing the
    # live actor, its device, training state or the correction limits.
    actor = copy.deepcopy(algorithm.actor).cpu().eval()
    maximum = 0.
    for start in range(0, len(data['policy']), 128):
        obs = TensorDict({k: v[start:start+128].detach().cpu() for k, v in data.items() if k != 'action'},
                         batch_size=[len(data['policy'][start:start+128])])
        inherited = AscentLandingMLPModel.forward(actor, obs)
        cap = actor.sequence_limits*sequence_gate(obs['policy'], obs['motion_control'])[:, None]
        delta = (data['action'][start:start+128].detach().cpu()-inherited).abs()
        if (delta > cap+2.e-5).any():
            raise ValueError('Contact demonstration exceeds this initialization source/action envelope; collect matching demonstrations')
        maximum = max(maximum, float((delta/cap.clamp_min(1.e-5)).max()))
    return dict(samples=len(data['policy']), maximum_normalized_correction=maximum)


def attach_demonstrations(algorithm, paths=None, coefficient=.5, decay_updates=800):
    if coefficient < 0 or not np.isfinite(coefficient) or decay_updates <= 0:
        raise ValueError('Invalid contact demonstration schedule')
    if hasattr(algorithm, 'policy_auxiliary_loss'):
        raise ValueError('Contact demonstration hook already installed')
    algorithm._contact_demo = None
    algorithm._contact_demo_manifest = []
    algorithm._contact_demo_losses = []
    if paths:
        collected = {k: [] for k in ('policy', 'motion_control', 'action')}
        for filename in paths:
            path = Path(filename).expanduser().resolve()
            with np.load(path, allow_pickle=False) as data:
                if 'contact_verified' not in data or not bool(data['contact_verified']):
                    raise ValueError('Only physically contact-verified demonstrations are accepted')
                n = len(data['policy'])
                for key, dim in [('policy', 244), ('motion_control', 3), ('action', 16)]:
                    value = data[key]
                    if value.shape != (n, dim) or n < 2 or not np.isfinite(value).all():
                        raise ValueError('Invalid contact demonstration '+key)
                    collected[key].append(torch.tensor(value, dtype=torch.float32, device=algorithm.device))
                algorithm._contact_demo_manifest.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    samples=n, teacher_checkpoint_sha256=str(data['checkpoint_sha256']),
                    terrain_height_depth=data['terrain_height_depth'].tolist()))
        algorithm._contact_demo = {k: torch.cat(values) for k, values in collected.items()}
    original_update, original_save, original_load = algorithm.update, algorithm.save, algorithm.load

    def auxiliary(self):
        elapsed = max(0, self.guard_updates-self.critic_warmup_iterations)
        factor = max(0., 1.-elapsed/decay_updates)
        if self._contact_demo is None or factor == 0. or coefficient == 0.:
            return next(self.actor.parameters()).new_zeros(())
        data = self._contact_demo
        index = torch.randint(len(data['policy']), (min(128, len(data['policy'])),), device=self.device)
        obs = TensorDict({k: data[k][index] for k in ('policy', 'motion_control')}, batch_size=[len(index)])
        predicted = self.actor(obs)
        error = ((predicted-data['action'][index])/self.actor.sequence_limits.clamp_min(.12)).square()
        cost = error[:, :12].mean()+.1*error[:, 12:].mean()
        self._contact_demo_losses.append(float(cost.detach()))
        return coefficient*factor*cost

    def update(self):
        self._contact_demo_losses = []
        result = original_update()
        result['contact_demonstration'] = sum(self._contact_demo_losses)/max(len(self._contact_demo_losses), 1)
        result['contact_demonstration_scale'] = coefficient*max(0., 1.-max(0, self.guard_updates-self.critic_warmup_iterations)/decay_updates) if self._contact_demo is not None else 0.
        return result

    def save(self):
        state = original_save()
        state['ascent_contact_demonstrations'] = (None if self._contact_demo is None else
            {k: v.detach().cpu().clone() for k, v in self._contact_demo.items()})
        state['ascent_contact_demonstration_manifest'] = self._contact_demo_manifest
        state['ascent_contact_demonstration_schedule'] = dict(coefficient=coefficient, decay_updates=decay_updates)
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if load_cfg is None or load_cfg.get('iteration', False):
            schedule = loaded_dict.get('ascent_contact_demonstration_schedule')
            if schedule is None or schedule != dict(coefficient=coefficient, decay_updates=decay_updates):
                raise ValueError('Stage5 resume requires the same contact demonstration schedule')
            demo = loaded_dict.get('ascent_contact_demonstrations')
            self._contact_demo = None if demo is None else {k: v.to(self.device) for k, v in demo.items()}
            self._contact_demo_manifest = loaded_dict.get('ascent_contact_demonstration_manifest', [])
        return original_load(loaded_dict, load_cfg, strict)

    algorithm.policy_auxiliary_loss = MethodType(auxiliary, algorithm)
    algorithm.update = MethodType(update, algorithm)
    algorithm.save = MethodType(save, algorithm)
    algorithm.load = MethodType(load, algorithm)
