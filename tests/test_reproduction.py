"""Real Hydra composition and reduced-network checks; no licensed body assets."""
from pathlib import Path
import os
import subprocess
import sys
import hydra
from omegaconf import OmegaConf
import torch
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[1]


def config():
    with hydra.initialize_config_dir(config_dir=str(ROOT/'configs'), version_base=None):
        cfg = hydra.compose(config_name='base', overrides=['exp=hsmr/legion_finetune',
            f'_pm_.root={ROOT}', f'_pm_.inputs={ROOT}/data_inputs', f'_pm_.outputs={ROOT}/data_outputs'])
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def test_legion_config_composes():
    cfg = config()
    assert cfg.pl_trainer.devices == 1
    assert cfg.data.cfg.train.dataloader.batch_size == 1
    assert cfg.pipeline.cfg.SKEL.make_dense is False
    assert cfg.init_backbone is False
    assert cfg.data.cfg.train.datasets[0].item.urls.endswith('/coco14-tars/000000.tar')
    assert list(cfg.callbacks) == ['ckpt']


def test_frozen_backbone_and_head_backward():
    cfg = config()
    # Small instances of the real modules keep this check cheap on the Mac.
    cfg.pipeline.cfg.SKEL = {'_target_':'torch.nn.Identity'}
    cfg.pipeline.cfg.backbone.embed_dim = 64
    cfg.pipeline.cfg.backbone.depth = 2
    cfg.pipeline.cfg.backbone.num_heads = 4
    head = cfg.pipeline.cfg.head.cfg.transformer_decoder
    head.depth = 2
    head.heads = 4
    head.dim_head = 16
    head.mlp_dim = 128
    head.dim = 64
    head.context_dim = 64
    pipeline = instantiate(cfg.pipeline, init_backbone=False, _recursive_=False)
    pipeline.train()
    assert not pipeline.backbone.training
    assert all(not p.requires_grad for p in pipeline.backbone.parameters())
    params, cam = pipeline.head(pipeline.backbone(torch.randn(1,3,256,192)))
    assert params['poses'].shape == (1,46)
    assert params['betas'].shape == (1,10)
    assert cam.shape == (1,3)
    loss = params['poses'].square().mean()+params['betas'].square().mean()+cam.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in pipeline.head.parameters())
    assert all(p.grad is None for p in pipeline.backbone.parameters())


def test_external_paths_are_used(tmp_path):
    env = dict(os.environ, HSMR_INPUTS_ROOT=str(tmp_path/'inputs'), HSMR_OUTPUTS_ROOT=str(tmp_path/'outputs'))
    result = subprocess.run([sys.executable,'-c','from lib.platform import PM; print(PM.inputs); print(PM.outputs)'],
                            cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert str(tmp_path/'inputs') in result.stdout
    assert str(tmp_path/'outputs') in result.stdout
    assert (tmp_path/'outputs').is_dir()


def test_training_cli_config_dump():
    # The upstream wrapper duplicated CLI flags, which broke Hydra --cfg.
    result = subprocess.run([sys.executable, 'exp/run_train.py',
        'exp=hsmr/legion_finetune', '--cfg', 'job'], cwd=ROOT,
        text=True, capture_output=True, check=True)
    assert 'HSMR-legion-head-finetune' in result.stdout
    assert 'resume_path: null' in result.stdout
