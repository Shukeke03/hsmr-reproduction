"""Download official public assets on Linux; licenses are handled separately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from huggingface_hub import HfApi, hf_hub_download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(os.environ.get('HSMR_INPUTS_ROOT', '~/workspace/datasets/HSMR')).expanduser())
    p.add_argument('--assets', action='store_true', help='Released model and three regressors')
    p.add_argument('--assets-only', action='store_true', help='Download released weights and regressors; no training data')
    p.add_argument('--body-models', action='store_true', help='Official public body model archive including SKEL v1.1.1')
    p.add_argument('--no-training-data', action='store_true', help='Skip all training shards')
    p.add_argument('--backbone', action='store_true', help='Backbone initialization for training from scratch')
    p.add_argument('--evaluation-labels', action='store_true', help='Labels only; images require separate preparation')
    p.add_argument('--dataset', default='coco14', choices=['coco14','coco14vit','h36m','mpii','mpi_inf_3dhp','aic','ava','insta','all'])
    p.add_argument('--shards', type=int, default=1, help='Per dataset; 0 means all')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if sys.platform != 'linux' and not a.dry_run:
        p.error('Data downloads run on legion/Linux only.')
    if a.shards < 0:
        p.error('--shards must be >= 0')
    a.root = a.root.expanduser().resolve()
    lock = json.loads((Path(__file__).resolve().parents[1]/'docs/reproduction/download-lock.json').read_text())
    api = HfApi()
    selected = []
    model_paths = []
    if a.assets or a.assets_only:
        model_paths += ['released_models/HSMR-ViTH-r1d1.tar.gz',
                        'body_models/SMPL_to_J19.pkl',
                        'body_models/J_regressor_SKEL_mix_MALE.pkl',
                        'body_models/J_regressor_SMPL_MALE.pkl']
    if a.backbone:
        model_paths += ['backbone/vitpose_backbone.pth']
    if a.body_models:
        model_paths += ['body_models.tar.gz']
    if a.evaluation_labels:
        model_paths += ['hmr2_evaluation_data.tar.gz','hsmr_evaluation_data.tar.gz']
    for f in api.list_repo_tree('IsshikiHugh/HSMR-data_inputs', recursive=True, revision=lock['model']['revision']):
        if f.path in model_paths:
            selected.append(('model','IsshikiHugh/HSMR-data_inputs',f,a.root))
    names = ['coco14','coco14vit','h36m','mpii','mpi_inf_3dhp','aic','ava','insta'] if a.dataset == 'all' else [a.dataset]
    if a.assets_only or a.no_training_data:
        names = []
    for name in names:
        files = sorted(api.list_repo_tree('IsshikiHugh/HSMR-TrainingData', repo_type='dataset', path_in_repo=name+'-tars', revision=lock['dataset']['revision']), key=lambda f:f.path)
        files = [f for f in files if f.path.endswith('.tar')]
        if a.shards:
            files = files[:a.shards]
        selected += [('dataset','IsshikiHugh/HSMR-TrainingData',f,a.root/'hsmr_training_data') for f in files]
    print(json.dumps({'files':len(selected), 'bytes':sum(f.size for _,_,f,_ in selected),
                      'root':str(a.root), 'paths':[f.path for _,_,f,_ in selected]},indent=2), flush=True)
    if a.dry_run:
        return
    a.root.mkdir(parents=True,exist_ok=True)
    for kind, repo, f, dst in selected:
        print('Downloading '+f.path,flush=True)
        local = Path(hf_hub_download(repo, f.path, repo_type=kind, local_dir=dst,
                                   revision=lock[kind]['revision'], endpoint=api.endpoint))
        sha = hashlib.sha256()
        with local.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                sha.update(chunk)
        digest = sha.hexdigest()
        expected = lock['files'].get(f.path, {})
        expected_sha = expected.get('sha256') or (f.lfs.sha256 if f.lfs else None)
        if expected_sha and digest != expected_sha:
            raise RuntimeError('SHA256 mismatch: '+f.path)
        if local.stat().st_size != expected.get('size', f.size):
            raise RuntimeError('File size mismatch: '+f.path)
        if local.name.endswith('.tar.gz'):
            target = a.root/'released_models' if f.path.startswith('released_models/') else a.root
            target.mkdir(parents=True,exist_ok=True)
            with tarfile.open(local) as archive:
                for member in archive.getmembers():
                    location=(target/member.name).resolve()
                    if not location.is_relative_to(target.resolve()) or member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                        raise ValueError('Unsafe archive member: '+member.name)
                archive.extractall(target)
        record={'repo':repo,'revision':lock[kind]['revision'],'endpoint':api.endpoint,'path':f.path,'bytes':local.stat().st_size,'sha256':digest,'verified_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        with (a.root/'download-manifest.jsonl').open('a') as log:
            log.write(json.dumps(record)+'\n')
        print('Verified '+f.path,flush=True)

if __name__ == '__main__':
    main()
