#!/usr/bin/env python3
"""CPU finite-horizon search for physically verified missing entry contacts.

Search bounded four-leg preparation around a recorded pre-entry state. Saves
demonstration observations/actions only when sequential contacts are confirmed;
an FK solution or reduced height error alone is never labelled a demonstration.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
import torch
from collect_m20_gait_reference import ROOT, d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--population', type=int, default=24)
    parser.add_argument('--generations', type=int, default=4)
    parser.add_argument('--start-after-command', type=float, default=.70)
    parser.add_argument('--horizon', type=float, default=2.4)
    parser.add_argument('--parameters-file', type=Path, help='Replay stored parameters instead of searching again')
    parser.add_argument('--timing-search', action='store_true', help='Also optimize the two front preparation onsets')
    parser.add_argument('--warm-parameters', type=Path, help='Initialize search around a previous candidate')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): parser.error('Use an empty output directory')
    if args.population < 4 or args.generations < 1 or args.horizon <= 0: parser.error('Invalid search budget')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); rng = np.random.default_rng(42)
    actor = d.load_actor(args.checkpoint)
    module = d.ascent_sequence_module(); core = d.training_module('ascent_sequence')
    meta = json.loads((args.case_dir/'case.json').read_text())
    h = meta['terrain']['terrain_parameters']['step_height'][0]
    depth = meta['terrain']['terrain_parameters']['tread_depth'][0]
    first = meta['terrain']['terrain_parameters']['start_x'][0]
    count = int(meta['terrain']['terrain_parameters']['step_count'][0])
    cfg = SimpleNamespace(terrain=None, terrain_xml=ROOT/'deploy/deploy_mujoco/terrains/stairs_ascent.xml',
        stair_height=h, tread_depth=depth, stair_count=count, stair_start=first)
    model = d.load_model(ROOT/'deep_robotics_model/M20/mjcf/M20.xml', cfg)
    contract = d.M20Contract(model, cfg.terrain_config)
    with np.load(args.case_dir/'physics.npz') as z: raw = {k: z[k] for k in z.files}
    sample = int(np.searchsorted(raw['time'], meta['settle_s']+args.start_after_command))
    speed = meta['speed_m_s']; command = np.array([speed, 0., 0.], np.float32)
    qpos, qvel = raw['qpos'][sample], raw['qvel'][sample]
    previous = raw['action'][max(sample-1, 0)].astype(np.float32)
    limits = np.asarray(module.SEQUENCE_LIMITS)
    wheel_index = {int(b): i for i, b in enumerate(contract.wheel_body_ids)}

    def rollout(parameters, record=False):
        data = mujoco.MjData(model); data.qpos[:] = qpos; data.qvel[:] = qvel
        data.time = float(raw['time'][sample]); mujoco.mj_forward(model, data)
        hold = d.HeadingHoldController(contract, enabled=True, legacy=True)
        prev = previous.copy(); ledger = core.SequenceLedger(1, 'cpu', depth=depth, targets=count)
        min_error = np.ones(2)*2.; early = 0.; lateral = 0.; bad_contacts = 0; fell = False
        records = []; bias = np.zeros(16)
        bias[[1, 2, 4, 5, 7, 8, 10, 11]] = parameters[:8]*limits[[1, 2, 4, 5, 7, 8, 10, 11]]
        bias[12:14] = parameters[8]*limits[12:14]; bias[14:] = parameters[9]*limits[14:]
        for _ in range(round(args.horizon/d.POLICY_DT)):
            effective, _ = hold.update(data, command)
            obs = d.actor_inputs(contract, data, effective, prev, actor, command)
            with torch.inference_mode():
                action = actor(obs)[0].numpy().copy()
                gate = module.sequence_gate(obs['policy'], obs['motion_control']).item()
            # Smooth spatial preparation before each known missing target.
            # Rear legs prepare with the follower, permitting a body shift.
            x = data.xpos[contract.wheel_body_ids, 0]
            onsets = [first+depth-.27, first-.27, first-.27, first-.27]
            if len(parameters) == 12:
                onsets[:2] = np.asarray(onsets[:2])+.20*parameters[10:12]
            envelope = np.clip((x-np.asarray(onsets))/.20, 0., 1.)
            envelope = envelope*envelope*(3.-2.*envelope)
            # Rear position is behind the fronts; coordinate preparation by
            # the front follower's opportunity, not the rear's future edge.
            envelope[2:] = envelope[1]
            correction = bias.copy()
            correction[:12] *= np.repeat(envelope, 3)
            correction *= gate
            action += correction
            for _ in range(round(d.POLICY_DT/d.PHYSICS_DT)):
                contract.apply_pd(data, action); mujoco.mj_step(model, data)
                support_force = np.zeros(4); levels = np.zeros(4, dtype=int)-1
                for ci in range(data.ncon):
                    contact = data.contact[ci]
                    wrench = np.zeros(6); mujoco.mj_contactForce(model, data, ci, wrench)
                    active_contact = np.linalg.norm(wrench[:3]) > 1.
                    b1, b2 = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
                    if b1 and b2:
                        if active_contact: bad_contacts += 1
                        continue
                    body = b2 if b1 == 0 else b1
                    if body not in wheel_index:
                        if body and active_contact: bad_contacts += 1
                        continue
                    normal = (1. if b2 == body else -1.)*contact.frame[:3]
                    if normal[2] < .7: continue
                    if wrench[0] > 5.:
                        wheel = wheel_index[body]; support_force[wheel] += wrench[0]
                        levels[wheel] = int(round(contact.pos[2]/h))
                support = support_force > 5.
                positions = data.xpos[contract.wheel_body_ids].copy()
                event = ledger.update(torch.tensor(levels[None]), torch.tensor(support[None]),
                    torch.tensor(positions[None, :, 0], dtype=torch.float32), torch.tensor([first]), torch.tensor([True]), d.PHYSICS_DT)
                early += event['early_advance'].item()
                for j, (wheel, tread) in enumerate([(0, 2), (1, 1)]):
                    edge = first+(tread-1)*depth
                    point = positions[wheel]
                    dx = max(edge+.06-point[0], 0., point[0]-(edge+depth-.06))
                    dz = point[2]-.09-tread*h
                    min_error[j] = min(min_error[j], np.sqrt(dx*dx+dz*dz))
                lateral = max(lateral, abs(data.xpos[contract.base_id, 1]))
                rotation = data.xmat[contract.base_id].reshape(3, 3)
                if rotation[2, 2] < .65: fell = True; break
            if record:
                records.append(dict(observation=obs['policy'][0].numpy(), command=command.copy(),
                                    action=action.copy(), time=float(data.time), wheel_pos=positions,
                                    support=support, support_tread=levels))
            prev = action.astype(np.float32)
            if fell: break
            if ledger.covered[:, :2, :2].all() and ledger.pair_paid[:, :2].all(): break
        coverage = int(ledger.covered[:, :2, :2].sum())
        pairs = int(ledger.pair_paid[:, :2].sum())
        missing = int(ledger.missed[:, :2, :2].sum())
        score = 40*coverage+80*pairs-100*missing-50*early-40*min_error.sum()-30*lateral-2*bad_contacts-300*fell
        successful = coverage == 4 and pairs == 2 and missing == 0 and early == 0 and not fell and bad_contacts == 0 and lateral < .15
        return float(score), dict(coverage=coverage, pairs=pairs, missing=missing, early=early,
            minimum_missing_target_distance_m=min_error.tolist(), lateral_m=lateral, fell=fell,
            bad_contact_samples=bad_contacts, front_touchdown_local_s=ledger.landing_time[0, :2, :2].tolist(),
            successful=bool(successful)), records

    dimension = 12 if args.timing_search else 10
    mean = np.zeros(dimension); sigma = np.ones(dimension)*.5
    if args.warm_parameters:
        previous_parameters = np.asarray(json.loads(args.warm_parameters.read_text())['parameters'])
        mean[:min(len(mean), len(previous_parameters))] = previous_parameters[:len(mean)]
    best = None; trace = []
    if args.parameters_file:
        parameters = np.asarray(json.loads(args.parameters_file.read_text())['parameters'])
        if parameters.shape not in [(10,), (12,)] or not np.isfinite(parameters).all() or (np.abs(parameters) > 1.).any():
            parser.error('Invalid bounded preparation parameters')
        score, metrics, _ = rollout(parameters)
        best = (score, parameters, metrics)
    for generation in range(0 if args.parameters_file else args.generations):
        population = np.clip(rng.normal(mean, sigma, size=(args.population, dimension)), -1., 1.)
        population[0] = mean if best is None else best[1]
        samples = []
        for parameters in population:
            score, result, _ = rollout(parameters)
            samples.append((score, parameters.copy(), result))
            if best is None or score > best[0]: best = samples[-1]
        elites = sorted(samples, key=lambda x: x[0], reverse=True)[:max(3, args.population//4)]
        elite_parameters = np.stack([s[1] for s in elites]); mean = elite_parameters.mean(0)
        sigma = np.maximum(.08, elite_parameters.std(0))
        trace.append(dict(generation=generation, best_score=best[0], best_metrics=best[2]))
        print(json.dumps(trace[-1]), flush=True)
        (args.output_dir/'progress.json').write_text(json.dumps(trace, indent=2)+'\n')
        if best[2]['successful']: break
    score, metrics, records = rollout(best[1], record=True)
    np.savez_compressed(args.output_dir/'best_segment.npz', **{k: np.asarray([r[k] for r in records]) for k in records[0]})
    result = dict(checkpoint=str(args.checkpoint), source_case=str(args.case_dir), start_sample=sample,
        start_after_command_s=args.start_after_command, horizon_s=args.horizon, parameters=best[1].tolist(),
        score=score, metrics=metrics, trace=trace, gpu_test_run=False,
        scope='Dynamic MuJoCo local segment from recorded valid pre-entry state; not a complete standing-start ascent or a trained policy.')
    if metrics['successful']:
        np.savez_compressed(args.output_dir/'verified_demonstration.npz',
            policy=np.stack([r['observation'] for r in records]), motion_control=np.stack([r['command'] for r in records]),
            action=np.stack([r['action'] for r in records]), contact_verified=np.asarray(True),
            checkpoint_sha256=np.asarray(hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()),
            terrain_height_depth=np.asarray([h, depth]))
    (args.output_dir/'search.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
