"""Check training dependencies and CUDA without loading licensed assets."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import cv2
import hydra
import pytorch_lightning
import webdataset
import skel.skel_model
from lib.modeling.pipelines import HSMRPipeline
from lib.platform import PM
print(json.dumps({
    'python': sys.version.split()[0], 'torch': torch.__version__,
    'lightning': pytorch_lightning.__version__,
    'cuda': torch.cuda.is_available(), 'torch_cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'inputs': str(PM.inputs), 'outputs': str(PM.outputs),
}, indent=2))
if sys.platform == 'linux':
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable on Linux training host')
    x = torch.randn(256, 256, device='cuda', requires_grad=True)
    (x @ x).square().mean().backward()
    torch.cuda.synchronize()
    print('CUDA forward/backward: PASS')
