#!/usr/bin/env python3
"""CPU-only warm-start comparison across stair height/depth, retaining raw data."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from collect_m20_gait_reference import ROOT, d, run_case
from analyze_m20_gait_reference import stable_events, LEGS
from evaluate_ascent_entry_v1 import sequence_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True, help='label=checkpoint')
    parser.add_argument('--extended', action='store_true', help='Also screen max training height and unseen height/depth combinations')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows = []
    for spec in args.checkpoints:
        label, filename = spec.split('=', 1)
        path = Path(filename); actor = d.load_actor(path)
        geometries = [(.15, .30), (.10, .30), (.15, .25), (.15, .35), (.20, .30)]
        if args.extended: geometries += [(.25, .30), (.14, .28), (.17, .32)]
        for height, depth in geometries:
            folder = args.output_dir/label/f'h{height:.2f}_d{depth:.2f}'
            cfg = SimpleNamespace(output_dir=folder, model=ROOT/'deep_robotics_model/M20/mjcf/M20.xml',
                terrain_xml=ROOT/'deploy/deploy_mujoco/terrains/stairs_ascent.xml', stair_height=height,
                tread_depth=depth, stair_count=20, settle=6., duration=32.)
            for speed in [.5, .7]:
                meta = run_case(actor, cfg, speed, 0.)
                with np.load(folder/meta['case']/'physics.npz') as z: raw = {k: z[k] for k in z.files}
                events, _ = stable_events(raw, meta['case'])
                treads = {leg: sorted(set(e['tread_id'] for e in events if e['wheel'] == leg and 1 <= e['tread_id'] <= 4)) for leg in LEGS}
                forward = raw['time'] >= 6.
                front = raw['wheel_pos'][:, :2, 0].max(-1)
                entry = forward & (front >= .25) & (front < .25+4*depth)
                distance = np.ptp(raw['root_pos'][entry, 0]) if entry.any() else 0.
                slip = float((raw['slip'][entry]*raw['support'][entry]).sum()*.005/max(distance, .05))
                row = dict(model=label, checkpoint=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    height_m=height, depth_m=depth, speed_m_s=speed, status=meta['status'],
                    first_four_treads=treads, front_first_four_coverage=sum(len(treads[k]) for k in ['FL','FR'])/8.,
                    entry_slip_per_forward_m=slip,
                    max_abs_lateral_m=float(np.abs(raw['root_pos'][forward, 1]).max()),
                    peak_abs_roll_deg=float(np.rad2deg(np.abs(raw['rpy'][forward, 0])).max()),
                    nonwheel_contact_frames=int((raw['nonwheel_contact_peak_force'][forward] > 1.).sum()),
                    self_contact_frames=int((raw['self_contact_peak_force'][forward] > 1.).sum()),
                    command_to_finish_s=meta['duration_s']-meta['settle_s'],
                    sequence=sequence_summary(raw, events), raw_dir=str((folder/meta['case']).resolve()))
                rows.append(row)
                (args.output_dir/'comparison.json').write_text(json.dumps({'rows': rows,
                    'scope': 'CPU deterministic MuJoCo height/depth screening, zero spawn offset, no retraining; not hardware or multi-seed validation.'}, indent=2)+'\n')
                print(json.dumps({k: row[k] for k in ['model','height_m','depth_m','speed_m_s','status','front_first_four_coverage','max_abs_lateral_m']}) ,flush=True)


if __name__ == '__main__': main()
