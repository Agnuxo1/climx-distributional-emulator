"""Apache-2.0. Francisco Angulo de Lafuente, ORCID0009-0001-1634-7063.

Create a path-adapted COPY and replay CPU checks. The archived source is immutable.
No GPU mode, downloads, submissions or installations are run by this helper.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze(source):
    files={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file() and p.name!='SOURCE_MANIFEST.json'}
    path=source/'SOURCE_MANIFEST.json'
    path.write_text(json.dumps({'files':files},indent=2)+'\n',encoding='utf-8')
    return sha(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('fixtures','preflight'))
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--store',type=Path)
    args=parser.parse_args()
    original=Path(__file__).resolve().parents[1]/'source'
    work=args.work.resolve();store=(args.store or work/'store_full').resolve()
    assert work!=original and original not in work.parents and work not in original.parents, 'work must be outside the archive'
    work.mkdir(parents=True,exist_ok=True)
    source=work/'source'
    identity={'original_manifest':sha(original/'SOURCE_MANIFEST.json'),'store':str(store),'work':str(work)}
    if not source.exists():
        shutil.copytree(original,source)
        runner=source/'run.py';text=runner.read_text(encoding='utf-8')
        replacements={"'/home/ai/proyectos/climx/store_full'":repr(str(store)),
                      "'/mnt/datos/proyectos/climx-d01-20261110/results'":repr(str(work/'results')),
                      "'/mnt/datos/proyectos/climx-d01-20261110'":repr(str(work))}
        for old,new in replacements.items():
            assert text.count(old)==1, 'unexpected runner layout'
            text=text.replace(old,new)
        runner.write_text(text,encoding='utf-8')
        (work/'REPRODUCTION_IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
        freeze(source)
    else:
        assert json.loads((work/'REPRODUCTION_IDENTITY.json').read_text(encoding='utf-8'))==identity, 'use a new work directory for another setup'
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
    def run(mode,out):
        manifest=sha(source/'SOURCE_MANIFEST.json')
        subprocess.run([sys.executable,str(source/'run.py'),mode,'--expected-source',manifest,'--out',str(out)],check=True,env=env,cwd=work)
    if args.action=='fixtures':run('fixtures',work/'cpu-fixtures')
    else:
        assert (store/'meta.json').is_file(), 'build the private competition store first'
        run('preflight',work/'input-preflight')
        report=json.loads((work/'input-preflight/PREFLIGHT.json').read_text(encoding='utf-8'))
        assert report['passed']
        (source/'INPUTS.json').write_text(json.dumps({k:report[k] for k in ('input_identity','runtime','runtime_files_sha256')},indent=2)+'\n',encoding='utf-8')
        freeze(source)
        run('verify',work/'final-cpu-verify')
    print(json.dumps({'work':str(work),'new_source_manifest':sha(source/'SOURCE_MANIFEST.json'),'original_archive_source_unchanged':sha(original/'SOURCE_MANIFEST.json')==identity['original_manifest'],'GPU_used':False}))

if __name__=='__main__':main()
