"""Independent weighted-loss/bootstrap replay. No model inference or promotion."""
from pathlib import Path
import argparse,ast,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE_RUNNER=ROOT.parent/'d01-20261110/source/run.py'

def replay(losses,days,weights):
    assert losses.ndim==3 and losses.shape[2]==2
    assert losses.shape[1]==len(days) and losses.shape[0]==len(weights)
    assert np.isfinite(losses).all() and (losses>=0).all() and (days>0).all() and (weights>0).all()
    total=np.einsum('b,bti,t->i',weights,losses,days)/(weights.sum()*days.sum())
    reduction=np.average(losses[:,:,0]-losses[:,:,1],axis=0,weights=weights)
    rng=np.random.default_rng(20261110)
    starts=rng.integers(0,len(days),size=(10000,(len(days)+9)//10))
    ids=((starts[:,:,None]+np.arange(10))%len(days)).reshape(10000,-1)[:,:len(days)]
    bootstrap=(reduction[ids]*days[ids]).sum(1)/days[ids].sum(1)
    ci=np.quantile(bootstrap,[.025,.975])
    gain=total[0]-total[1];relative=gain/total[0]
    return {'control':float(total[0]),'Q2':float(total[1]),'reduction_K':float(gain),'relative_reduction':float(relative),
            'paired_95pct_block_CI_K':ci.tolist(),'gate_relative_at_least_0025':bool(relative>=.0025),
            'gate_CI_low_positive':bool(ci[0]>0),'screen_pass':bool(relative>=.0025 and ci[0]>0)}

def source_summary():
    tree=ast.parse(SOURCE_RUNNER.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summary')
    scope={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'isolated-frozen-summary','exec'),scope)
    return scope['summary']

def compare(expected,actual):
    for key in ('control','Q2','reduction_K','relative_reduction','paired_95pct_block_CI_K'):
        np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=0,err_msg=key)
    for key in ('gate_relative_at_least_0025','gate_CI_low_positive','screen_pass'):assert actual[key]==expected[key],key

def fixtures():
    days=np.full(86,365);days[-1]=364;weights=np.array([1.,2.,7.])
    fn=source_summary();passed=[]
    for name,gain,gate in (('identity_control',0.,False),('positive_below_quarter_percent',.002,False),('positive_above_quarter_percent',.005,True)):
        loss=np.ones((3,86,2));loss[:,:,1]-=gain
        actual=replay(loss,days,weights);compare(fn(loss,days,weights),actual)
        np.testing.assert_allclose(actual['paired_95pct_block_CI_K'],[gain,gain],atol=1e-12)
        assert actual['screen_pass']==gate;passed.append(name)
    rng=np.random.default_rng(71);loss=rng.uniform(1.,3.,(3,86,2))
    compare(fn(loss,days,weights),replay(loss,days,weights));passed.append('unequal_spatial_weights_and_variable_days')
    return passed

parser=argparse.ArgumentParser();parser.add_argument('--result-dir',type=Path);parser.add_argument('--source-runner',type=Path);args=parser.parse_args()
if args.source_runner:SOURCE_RUNNER=args.source_runner.resolve()
passed=fixtures()
report={'independent_fixtures_passed':passed,'GPU_used':False}
if args.result_dir:
    folder=args.result_dir
    result=json.loads((folder/'RESULT.json').read_text(encoding='utf-8'));assert result['mode']=='screen' and result['passed']
    with np.load(folder/'PAIRED_DAILY_LOSSES.npz',allow_pickle=False) as z:
        assert z['bands'].tolist()==[3,9,15,21,27,33,39,45] and z['years'].tolist()==list(range(2015,2101))
        actual=replay(z['losses'],z['days'],z['weights'])
    compare(result['summary'],actual)
    report.update({'raw_loss_sha256':hashlib.sha256((folder/'PAIRED_DAILY_LOSSES.npz').read_bytes()).hexdigest(),'replayed_summary':actual,'official_score':None,'new_entry_permitted':False})
out=ROOT/('STATISTICS_REPLAY.json' if args.result_dir else 'STATISTICS_FIXTURES.json')
out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
