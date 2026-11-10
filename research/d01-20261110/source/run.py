"""Preregistered daily-quantile screen. Never writes a competition submission."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time

ENTRY_TIME=time.perf_counter()

sys.dont_write_bytecode=True
# -B prevents writes, not reads. A fresh empty prefix also excludes old caches.
sys.pycache_prefix=tempfile.mkdtemp(prefix='no-bytecode-',dir='/mnt/datos/proyectos/climx-d01-20261110')
ROOT=Path(__file__).resolve().parent
STORE=Path('/home/ai/proyectos/climx/store_full')
OUT=Path('/mnt/datos/proyectos/climx-d01-20261110/results')
BANDS=(3,9,15,21,27,33,39,45)
TRAIN=('historical','ssp126','ssp585')
VALID='ssp370'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def source_guard(expected):
    manifest=ROOT/'SOURCE_MANIFEST.json'
    assert sha(manifest)==expected, 'manifest mismatch'
    pins=json.loads(manifest.read_text(encoding='utf-8'))['files']
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    assert actual==set(pins)|{'SOURCE_MANIFEST.json'}, 'unexpected source files'
    for name,digest in pins.items():
        assert sha(ROOT/name)==digest, 'source hash mismatch: '+name

parser=argparse.ArgumentParser()
parser.add_argument('mode',choices=('preflight','fixtures','verify','pilot','screen'))
parser.add_argument('--expected-source',required=True)
parser.add_argument('--out',type=Path)
args=parser.parse_args()
source_guard(args.expected_source)  # before any local/scientific imports

import importlib.metadata as md
import importlib
import importlib.util
import resource
import numpy as np
import torch
from climx_sota import qmap
from climx_sota.data import Store
from climx_sota.pipeline import Config,build_scenarios,fit_predictors,ridge_vector
from daily_model import PairedFit,window_quantiles,daily_loss,fixtures

torch.set_num_threads(4)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def versions():
    return {name:md.version(name) for name in ('torch','numpy','pandas','xarray','scipy')}

def runtime_files():
    # Pin executable/package bytes, including GPU BLAS/solver/runtime libraries.
    # Old bytecode reads are redirected to a fresh empty prefix; writing is disabled.
    paths={Path(sys.executable).resolve()}
    roots=[]
    for name in ('torch','numpy','pandas','xarray','scipy'):
        module=importlib.import_module(name)
        roots.append(Path(module.__file__).resolve().parent)
    for name in ('nvidia.cublas','nvidia.cusolver','nvidia.cusparse','nvidia.cuda_runtime','nvidia.cuda_nvrtc','nvidia.nvjitlink'):
        spec=importlib.util.find_spec(name)
        if spec is not None:
            roots.extend(Path(p).resolve() for p in spec.submodule_search_locations or [])
    for root in roots:
        paths.update(p.resolve() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ('.py','.so') or '.so.' in p.name))
    return {str(p):sha(p) for p in sorted(paths)}

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def input_identity():
    meta=json.loads((STORE/'meta.json').read_text(encoding='utf-8'))
    assert (meta['nlat'],meta['nlon'],meta['band_rows'],meta['dtype'])==(192,288,4,'int16')
    pin={'store_metadata_sha256':sha(STORE/'meta.json'),'small_files':{},'raw_band_sha256':{},'bands':list(BANDS),'train':list(TRAIN),'validation':VALID}
    for name in ('forcings.pkl',)+tuple(f'{r}__{suffix}' for r in TRAIN+(VALID,) for suffix in ('gmt.npy','time.npz')):
        pin['small_files'][name]=sha(STORE/name)
    for run in TRAIN+(VALID,):
        for part,days in enumerate(meta['runs'][run]['parts']):
            p=STORE/f'{run}__tasmax__p{part}.dat'
            assert p.stat().st_size==48*days*1152*2
            mm=np.memmap(p,dtype=np.int16,mode='r',shape=(48,days,1152))
            for band in BANDS:
                pin['raw_band_sha256'][f'{p.name}:band{band}']=hashlib.sha256(mm[band].tobytes()).hexdigest()
            del mm
    return pin

def verify_inputs():
    pins=json.loads((ROOT/'INPUTS.json').read_text(encoding='utf-8'))
    assert input_identity()==pins['input_identity'], 'input identity changed'
    assert versions()==pins['runtime'], 'runtime drift'
    assert runtime_files()==pins['runtime_files_sha256'], 'runtime bytes changed'
    return sha(ROOT/'INPUTS.json')

def build_design(store):
    cfg=Config(members=16,workers=1,gen_procs=1)
    pm=fit_predictors(store,list(TRAIN),cfg,log=lambda *a: print(*a,flush=True))
    assert pm.names==['ghg_fast','ghg_slow','aer_pc1','aer_pc2']
    scen=build_scenarios(store,list(TRAIN)+[VALID])
    designs={r:pm.transform(scen[r]) for r in TRAIN+(VALID,)}
    years={r:np.unique(store.time(r)[0]) for r in TRAIN+(VALID,)}
    assert all(len(designs[r])==len(years[r]) for r in TRAIN+(VALID,))
    assert VALID not in TRAIN and 'ssp245' not in designs
    return designs,years,ridge_vector(pm,cfg)

def train_band(store,band,designs,years,penalty,device,cells=None):
    raw={r:store.band(r,'tasmax',band) for r in TRAIN}
    if cells is not None:
        raw={r:a[:,cells] for r,a in raw.items()}
    times={r:store.time(r) for r in TRAIN}
    split={r:[np.flatnonzero(times[r][0]==y) for y in years[r]] for r in TRAIN}
    historical=[qmap.to_365(raw['historical'][ix],times['historical'][1][ix]) for ix in split['historical']]
    transform=qmap.VarTransform('tasmax').fit_clim(historical)
    del historical
    x=np.concatenate([designs[r] for r in TRAIN])
    model=PairedFit(x,penalty,device)
    i=0
    for r in TRAIN:
        for ix in split[r]:
            z=transform.fwd(qmap.to_365(raw[r][ix],times[r][1][ix]))
            model.add(i,window_quantiles(z,device));i+=1
    assert i==len(x)
    model.finish()
    return model,transform

def evaluate(store,band,model,transform,design,years,device,cells=None,pilot=False):
    truth=store.band(VALID,'tasmax',band)
    if cells is not None:truth=truth[:,cells]
    y,doy=store.time(VALID)
    lat=np.repeat(store.lat[band*4:band*4+4],store.nlon)
    if cells is not None:lat=lat[cells]
    w=torch.as_tensor(np.cos(np.deg2rad(lat)),dtype=torch.float64,device=device)
    losses=np.empty((len(years),2));days=[]
    for i,year in enumerate(years):
        ix=np.flatnonzero(y==year);d=doy[ix]
        z=truth[ix]-transform.clim[d]
        days.append(len(ix))
        for candidate in (False,True):
            knots=model.daily_knots(design[i],candidate)
            loss=daily_loss(knots,z,d)
            assert torch.isfinite(loss).all() and (loss>=0).all()
            if not pilot:losses[i,int(candidate)]=(loss@w/w.sum()).item()
        if pilot: return {'finite_daily_quantiles':True,'cells':len(lat),'days':len(ix)}
    return losses,np.asarray(days),float(w.sum())

def summary(losses,days,weights):
    # Complete spatial field stays together in each paired circular time-block draw.
    by_year=(losses*weights[:,None,None]).sum(0)/weights.sum()
    base,candidate=(by_year*days[:,None]).sum(0)/days.sum()
    rng=np.random.default_rng(20261110);gains=[]
    for _ in range(10000):
        idx=np.concatenate([(start+np.arange(10))%len(days) for start in rng.integers(0,len(days),size=9)])[:len(days)]
        b,c=(by_year[idx]*days[idx,None]).sum(0)/days[idx].sum()
        gains.append(float(b-c))
    ci=np.quantile(gains,[.025,.975]);relative=(base-candidate)/base
    return {'metric':'truncated CRPS over daily tasmax quantiles p=.01..99; K; lower is better',
            'control':float(base),'Q2':float(candidate),'reduction_K':float(base-candidate),
            'relative_reduction':float(relative),'paired_95pct_block_CI_K':ci.tolist(),
            'gate_relative_at_least_0025':bool(relative>=.0025),'gate_CI_low_positive':bool(ci[0]>0),
            'screen_pass':bool(relative>=.0025 and ci[0]>0),'official_index_score':None,'submission_permitted':False}

start=ENTRY_TIME
if args.mode in ('preflight','fixtures','verify'):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU modes require empty CUDA visibility'
    report={'fixtures':fixtures('cpu'),'runtime':versions(),'source_manifest_sha256':args.expected_source,'passed':True}
    if args.mode=='preflight':
        report['input_identity']=input_identity()
        report['runtime_files_sha256']=runtime_files()
    if args.mode=='verify':report['input_manifest_sha256']=verify_inputs()
    report['seconds']=time.perf_counter()-start
    source_guard(args.expected_source)
    assert not any(Path(sys.pycache_prefix).rglob('*')), 'unexpected bytecode cache'
    assert args.out is not None
    save(args.out/'PREFLIGHT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_identity','runtime_files_sha256')}|{'runtime_files_pinned':len(report.get('runtime_files_sha256',{}))}))
    sys.exit(0)

assert torch.cuda.is_available(),'CUDA unavailable'
input_sha=verify_inputs()
out=OUT/args.mode;out.mkdir(parents=True,exist_ok=True)
attempt=out/'ATTEMPT.json'
with attempt.open('x',encoding='utf-8') as f:  # atomic O_EXCL: a second process fails closed
    json.dump({'source_manifest_sha256':args.expected_source,'input_manifest_sha256':input_sha,'started_unix':time.time()},f,allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
if args.mode=='screen':
    pilot=json.loads((OUT/'pilot/RESULT.json').read_text(encoding='utf-8'))
    assert pilot['passed'] and pilot['source_manifest_sha256']==args.expected_source and pilot['input_manifest_sha256']==input_sha
device=torch.device('cuda:0')
torch.cuda.reset_peak_memory_stats()
checks=fixtures(device)
store=Store(STORE)
designs,years,penalty=build_design(store)
records=[];all_losses=[];weights=[]
for band in (BANDS[:1] if args.mode=='pilot' else BANDS):
    cells=np.linspace(0,1151,32,dtype=int) if args.mode=='pilot' else None
    model,transform=train_band(store,band,designs,years,penalty,device,cells)
    val=evaluate(store,band,model,transform,designs[VALID],years[VALID],device,cells,pilot=args.mode=='pilot')
    if args.mode=='pilot':records.append(val)
    else:
        loss,days,weight=val
        all_losses.append(loss);weights.append(weight)
        p=out/f'coefficients_band{band:02d}.npz'
        np.savez_compressed(p,control=model.b0.cpu().numpy(),Q2=model.b1.cpu().numpy(),climatology=transform.clim,
                            projection=model.curve.projection,mean=model.curve.mean,scale=model.curve.scale)
        records.append({'band':band,'coefficient_sha256':sha(p)})
    print(json.dumps({'band':band,'completed':True,'seconds':time.perf_counter()-start}),flush=True)
    del model,transform
    torch.cuda.empty_cache()
verify_inputs();source_guard(args.expected_source)
assert not any(Path(sys.pycache_prefix).rglob('*')), 'unexpected bytecode cache'
elapsed=time.perf_counter()-start
peak=torch.cuda.max_memory_allocated()/1024**3
report={'mode':args.mode,'source_manifest_sha256':args.expected_source,'input_manifest_sha256':input_sha,'runtime':versions(),
        'fixtures':checks,'seconds':elapsed,'peak_allocated_VRAM_GiB':peak,'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'records':records,
        'test_accessed':False,'official_submission_created':False}
if args.mode=='pilot':report['passed']=bool(elapsed<=300 and peak<=6)
else:
    loss=np.stack(all_losses);w=np.asarray(weights)
    np.savez_compressed(out/'PAIRED_DAILY_LOSSES.npz',losses=loss,days=days,weights=w,years=years[VALID],bands=BANDS)
    report['summary']=summary(loss,days,w)
    report['passed']=True
save(out/'RESULT.json',report)
print(json.dumps(report),flush=True)
