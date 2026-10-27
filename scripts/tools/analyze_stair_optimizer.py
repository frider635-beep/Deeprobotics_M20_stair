#!/usr/bin/env python3
"""Read TensorBoard optimizer diagnostics; never loads policy checkpoints."""
import argparse
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
import numpy as np

NAMES=('learning_rate','approx_kl','clip_fraction','actor_grad_norm_before_clip',
       'critic_grad_norm_before_clip','ppo_grad_norm','weighted_anchor_grad_norm',
       'anchor_to_ppo_grad_ratio','ppo_anchor_grad_cosine','gradient_cosine_valid',
       'actor_parameter_update_l2','actor_parameter_relative_update','explained_variance_batch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--last',type=int,default=600)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.last<1:parser.error('--last must be positive')
    series={name:{} for name in NAMES};latest=-1
    for path in sorted(args.run.glob('events.out*')):
        for raw in RawEventFileLoader(str(path)).Load():
            event=event_pb2.Event.FromString(raw)
            if not event.HasField('summary'):continue
            for value in event.summary.value:
                if not value.HasField('simple_value'):continue
                if value.tag=='Perf/collection_time':latest=max(latest,event.step)
                if value.tag.startswith('Loss/') and value.tag[5:] in series:
                    series[value.tag[5:]][int(event.step)]=float(value.simple_value)
    if latest<0:parser.error('No complete training iteration metrics found')
    start=latest-args.last+1
    report=dict(run=str(args.run),window=[max(start,0),latest],metrics={},missing=[],
                decision='insufficient_diagnostics_keep_current_learning_rate',
                notes=['Gradient samples are one minibatch every100 updates; parameter delta spans an entire update.',
                       'KL is a sample estimate, explained variance is minibatch-based.',
                       'Anchor/PPO gradient comparison excludes motion regularization and other auxiliary objectives.',
                       'Aggregated gradients cannot alone rule out geometry-specific conflicts.'])
    for name,points in series.items():
        values=np.array([v for s,v in points.items() if s>=start])
        if not len(values):report['missing'].append(name);continue
        report['metrics'][name]=dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
                                    p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))
    metrics=report['metrics']
    if not report['missing'] and metrics['ppo_grad_norm']['n']>=5:
        kl=metrics['approx_kl']['median'];clip=metrics['clip_fraction']['median']
        ratio=metrics['anchor_to_ppo_grad_ratio']['median'];cos=metrics['ppo_anchor_grad_cosine']['median']
        valid=metrics['gradient_cosine_valid']['mean']>.9
        if valid and ratio>1 and cos<-.2:
            report['decision']='inspect_anchor_conflict_before_any_learning_rate_increase'
        elif kl>.02 or clip>.2:
            report['decision']='update_may_be_too_large_inspect_windows_before_reducing_learning_rate'
        elif kl<.001 and clip<.02:
            report['decision']='small_updates_observed_controlled_learning_rate_trial_may_be_considered'
        else:report['decision']='no_clear_optimizer_evidence_for_learning_rate_change'
        report['notes'].append('Thresholds are diagnostic heuristics, not automatic tuning or evidence of task mastery.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
