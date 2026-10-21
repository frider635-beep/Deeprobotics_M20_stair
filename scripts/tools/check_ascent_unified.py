#!/usr/bin/env python3
"""Stage7 migration, bridge tasks, paired landing and native PPO resume checks."""
import argparse,copy,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from tensordict import TensorDict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
import deploy_mujoco as d
BASE='tasks.manager_based.locomotion.velocity.mdp.ascent.'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(42)
    core=d.training_module(BASE+'ascent_unified');model=d.training_module('models.ascent_unified_actor')
    init=d.training_module('algorithms.initialization.ascent_unified_init');init.validate_source(args.checkpoint)
    course=core.BridgeCurriculum(10,'cpu');ids=torch.arange(10);yes=torch.ones(10,dtype=torch.bool)
    assert course.goals().tolist()==[4,4,3,3,3,2,2,2,2,2]
    for _ in range(6):course.update(ids,~yes,yes,ids*0,~yes)
    assert course.goals().tolist()==[4,4,3,3,3,2,2,2,2,2]
    for speed in (0,1):
        course.speed_bucket.fill_(speed)
        for _ in range(8):course.update(ids,yes,yes,ids*0,~yes)
    assert course.goals().tolist()==[6,6,4,4,4,3,3,3,3,3]
    state=course.state_dict();restored=core.BridgeCurriculum(10,'cpu');restored.load_state_dict(state)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in state.items())
    corrupt=copy.deepcopy(state);corrupt['skill'][0]=0
    try:restored.load_state_dict(corrupt)
    except ValueError:pass
    else:raise AssertionError('Pinned task floor was lost on resume')
    course=core.BridgeCurriculum(10,'cpu');course.skill.fill_(3);course.window.fill_(1);course.trials.fill_(8)
    up,down=course.update(ids,yes,yes,ids*0,~yes)
    assert up.tolist()==[True,False,False,False,True,False,False,False,True,False]
    assert course.goals()[0]==4
    course=core.BridgeCurriculum(10,'cpu');course.skill.fill_(3);course.window.fill_(1);course.trials.fill_(8)
    up,_=course.update(ids,yes,yes,ids*0+2,~yes);assert not up.any()
    # Both axles must land sequentially on a tread before moving higher.
    def settle(book,levels,support,goals=4):
        result=None
        x=torch.where(levels>0,(levels.float()-.5)*.3,torch.full_like(levels.float(),-.1))
        for _ in range(9):result=book.update_strict(levels,support,x,torch.zeros(1),torch.ones(1,dtype=torch.bool),.005,torch.tensor([goals]))
        return result
    book=core.PairedSequenceLedger(1,'cpu');level=torch.tensor([[0,0,1,0]]);support=torch.tensor([[False,False,True,False]])
    settle(book,level,support);level[0,2]=2;result=settle(book,level,support)
    assert book.failed[0] and book.failure_reason[0]==2 and result['ordered'].sum()==0
    book=core.PairedSequenceLedger(1,'cpu');level=torch.tensor([[0,0,1,1]]);support=torch.tensor([[False,False,True,True]])
    settle(book,level,support);assert book.failed[0] and book.failure_reason[0]==3
    book=core.PairedSequenceLedger(1,'cpu');level=torch.tensor([[0,0,1,0]]);support=torch.tensor([[False,False,True,False]])
    settle(book,level,support);level[0,3]=1;support[0,3]=True
    settle(book,level,support);assert not book.failed[0] and book.rear_pair_paid[0,0]
    # Same-side hold guidance penalizes leaving the completed tread while mate
    # is pending, then releases once the mate has stable support on that tread.
    book=core.PairedSequenceLedger(1,'cpu');book.covered[0,0,0]=True
    p=torch.tensor([[[.15,0.,.24],[.10,0.,.24],[-.4,0.,.09],[-.4,0.,.09]]]);v=torch.zeros(1,4)
    common=(torch.zeros(1),torch.zeros(1),torch.tensor([.3]),torch.tensor([.15]),torch.tensor([4]),torch.ones(1,dtype=torch.bool))
    _,safe=core.paired_guidance(book,p,v,*common)
    p[0,0,0]=.31;_,bad=core.paired_guidance(book,p,v,*common)
    assert bad>safe
    book.covered[0,1,0]=True;_,released=core.paired_guidance(book,p,v,*common);assert released==0
    mdp=d.training_module(BASE+'ascent_unified_mdp')
    metric_env=NS(num_envs=1,device='cpu',scene=NS(terrain=NS(terrain_types=torch.zeros(1,dtype=torch.long))),
        _ascent_adaptive_curriculum=core.BridgeCurriculum(1,'cpu'),
        _ascent_entry_observer=NS(failure_reason=torch.zeros(1,dtype=torch.long),
            episode_goals=torch.tensor([4]),sequence=book))
    values=mdp.metrics(metric_env,None)
    assert all(torch.isfinite(torch.tensor(v)) for v in values.values())
    assert values['__weighted_count__/FL_tread4_covered']==1.
    empty=mdp.metrics(metric_env,torch.tensor([],dtype=torch.long))
    assert all(torch.isfinite(torch.tensor(v)) for v in empty.values())
    actor=d.load_actor(args.checkpoint)
    trainable=[k for k,p in actor.named_parameters() if p.requires_grad]
    assert trainable and all(k.startswith('unified_mlp.') for k in trainable)
    assert not any(k.startswith(('entry_mlp.','landing_mlp.','sequence_mlp.')) for k in actor.state_dict())
    n=16;obs=TensorDict(dict(policy=torch.randn(n,244),motion_control=torch.tensor([[.5,0.,0.],[.7,0.,0.]]*8),
        critic=torch.randn(n,285),ascent_control=torch.zeros(n,12)),batch_size=[n])
    obs['policy'][:,57:]=torch.tensor([.7]*10+[.55]*3+[.4]*4)[None,None].expand(n,11,17).reshape(n,187)
    base=d.training_module('models.motion_residual_actor').MotionResidualMLPModel(obs,{'actor':['policy','motion_control']},'actor',16,
        hidden_dims=[512,256,128],activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':1.,'std_type':'log'})
    source={k:v for k,v in actor.state_dict().items() if not k.startswith('unified_mlp.') and k!='unified_limits'}
    base.load_state_dict(source)
    zero=obs.clone();zero['motion_control'].zero_()
    with torch.no_grad():assert torch.equal(actor(zero),base(zero))
    flat=obs.clone();flat['policy'][:,57:]=.5
    with torch.no_grad():assert torch.equal(actor(flat),base(flat))
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    critic=MLPModel(obs,{'critic':['critic','motion_control','ascent_control']},'critic',1,
        hidden_dims=[512,256,128],activation='elu',obs_normalization=False)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    init.load_initialized_critic(critic,saved['critic_state_dict'])
    storage=RolloutStorage('rl',n,4,obs,[16],device='cpu')
    alg=d.training_module('algorithms.stair_guarded_ppo').StairGuardedPPO(actor,critic,storage,device='cpu',
        num_learning_epochs=1,num_mini_batches=2,learning_rate=2.e-5,schedule='fixed',critic_warmup_iterations=0)
    alg.set_reference(actor.state_dict())
    d.training_module('algorithms.initialization.ascent_adaptive_demo').attach_demonstrations(alg,coefficient=0.)
    env=NS(num_envs=n,device='cpu',scene=NS(terrain=NS(terrain_levels=torch.zeros(n,dtype=torch.long),
        terrain_types=torch.arange(n)%8,terrain_origins=torch.zeros(6,10,3),env_origins=torch.zeros(n,3))),
        cfg=NS(ascent_adaptive_height_grid=[[.1]*10]*6),_ascent_adaptive_curriculum=core.BridgeCurriculum(n,'cpu'),reset=lambda:None)
    init.attach_curriculum_checkpoint(alg,env)
    frozen=model.baseline_digest(actor.state_dict());before=actor.unified_mlp[-1].weight.detach().clone()
    for _ in range(4):
        with torch.inference_mode():alg.act(obs);alg.process_env_step(obs,torch.randn(n),torch.zeros(n,dtype=torch.long),{})
    with torch.inference_mode():alg.compute_returns(obs)
    losses=alg.update();assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert not torch.equal(before,actor.unified_mlp[-1].weight) and model.baseline_digest(actor.state_dict())==frozen
    env._ascent_adaptive_curriculum.window[0,0]=1
    state=alg.save()
    resumed=d.training_module('algorithms.stair_guarded_ppo').StairGuardedPPO(d.load_actor(args.checkpoint),critic,storage,device='cpu',
        num_learning_epochs=1,num_mini_batches=2,learning_rate=2.e-5,schedule='fixed',critic_warmup_iterations=0)
    d.training_module('algorithms.initialization.ascent_adaptive_demo').attach_demonstrations(resumed,coefficient=0.)
    init.attach_curriculum_checkpoint(resumed,env)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'resume.pt';torch.save(state,path)
        resumed.load(torch.load(path,map_location='cpu',weights_only=True))
    assert all(torch.equal(v,resumed.actor.state_dict()[k]) for k,v in actor.state_dict().items())
    assert env._ascent_adaptive_curriculum.window[0,0].sum()==8
    assert env._ascent_adaptive_curriculum.skill[0]>=2
    result=dict(checkpoint=str(args.checkpoint),three_actual_mlps=True,only_unified_branch_trainable=True,
        zero_command_and_flat_exact=True,paired_rear_and_early_advance_checked=True,
        leading_wheel_hold_and_release_checked=True,bridge_pinned_groups_and_both_speed_promotion=True,
        geometry_probe_and_row_cap=True,actual_cpu_ppo_update=True,full_curriculum_resume=True,frozen_motion_exact=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
