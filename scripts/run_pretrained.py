"""Run official neural weights on a demo image without training datasets.

Outputs SKEL parameters. Licensed body assets are needed for mesh reconstruction.
An optional synthetic head-update benchmark includes the original HSMR main
losses with --with-body, while excluding dataset IO and pseudo-label refinement.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from lib.platform import PM


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--image', type=Path, default=PM.root/'data_inputs/demo/example_imgs/yoga1.jpg')
    p.add_argument('--output',type=Path,default=PM.outputs/'pretrained-smoke.json')
    p.add_argument('--benchmark-steps',type=int,default=0)
    p.add_argument('--with-body',action='store_true',help='Reconstruct and export a mesh using installed SKEL assets')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if a.benchmark_steps < 0:
        p.error('--benchmark-steps must be >= 0')
    ckpt=a.checkpoint or next((PM.inputs/'released_models').rglob('hsmr.ckpt'),None)
    if ckpt is None or not ckpt.is_file():
        p.error('Missing checkpoint: download scripts/prepare_data.py --assets-only on legion')
    torch.manual_seed(42)
    device=torch.device(a.device)
    with initialize_config_dir(config_dir=str(PM.configs),version_base=None):
        cfg=compose(config_name='base',overrides=['exp=hsmr/legion_finetune',
            f'_pm_.root={PM.root}',f'_pm_.inputs={PM.inputs}',f'_pm_.outputs={PM.outputs}'])
    OmegaConf.resolve(cfg)
    print('Loading official backbone and head weights...',flush=True)
    backbone=instantiate(cfg.pipeline.cfg.backbone)
    head=instantiate(cfg.pipeline.cfg.head)
    try:
        state=torch.load(ckpt,map_location='cpu',mmap=True)['state_dict']
    except RuntimeError:
        state=torch.load(ckpt,map_location='cpu')['state_dict']
    for name,module in [('backbone',backbone),('head',head)]:
        substate={k[len(name)+1:]:v for k,v in state.items() if k.startswith(name+'.')}
        module.load_state_dict(substate,strict=True,assign=True)
        module.to(device)
    del state,substate
    backbone.requires_grad_(False).eval()
    head.eval()
    img=cv2.imread(str(a.image))
    if img is None:
        p.error('Image cannot be read: '+str(a.image))
    h,w=img.shape[:2]; side=max(h,w)
    src=np.float32([[w/2-side/2,h/2-side/2],[w/2+side/2,h/2-side/2],[w/2-side/2,h/2+side/2]])
    dst=np.float32([[0,0],[256,0],[0,256]])
    rgb=cv2.cvtColor(cv2.warpAffine(img,cv2.getAffineTransform(src,dst),(256,256)),cv2.COLOR_BGR2RGB)
    x=torch.from_numpy(rgb.transpose(2,0,1).copy()).float().unsqueeze(0).to(device)/255
    mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1)
    std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
    x=((x-mean)/std)[:,:,:,32:-32]
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
    autocast=lambda:torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda')
    with torch.no_grad(),autocast():
        for _ in range(3):
            params,cam=head(backbone(x))
        sync(device);start=time.perf_counter()
        for _ in range(10):
            params,cam=head(backbone(x))
        sync(device)
        inference_ms=(time.perf_counter()-start)*100
    if not all(torch.isfinite(v).all() for v in [*params.values(),cam]):
        raise RuntimeError('Non-finite predictions')
    result={'checkpoint':str(ckpt),'image':str(a.image),'crop':'whole image padded to square; no detector',
        'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        'backbone_parameters':sum(p.numel() for p in backbone.parameters()),
        'head_parameters':sum(p.numel() for p in head.parameters()),
        'inference_ms_per_image':inference_ms,
        'poses':params['poses'].float().cpu().tolist(),'betas':params['betas'].float().cpu().tolist(),
        'camera':cam.float().cpu().tolist(),'body_mesh_computed':False}
    if a.with_body:
        import trimesh
        body=instantiate(cfg.pipeline.cfg.SKEL).to(device)
        with torch.no_grad():
            mesh=body(poses=params['poses'].float(),betas=params['betas'].float(),skelmesh=True)
        if not torch.isfinite(mesh.skin_verts).all():
            raise RuntimeError('Non-finite SKEL mesh')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        mesh_path=a.output.with_name(a.output.stem+'-skin.obj')
        trimesh.Trimesh(vertices=mesh.skin_verts[0].cpu().numpy(),faces=body.skin_f.cpu().numpy(),process=False).export(mesh_path)
        skeleton_path=a.output.with_name(a.output.stem+'-skeleton.obj')
        trimesh.Trimesh(vertices=mesh.skel_verts[0].cpu().numpy(),faces=body.skel_f.cpu().numpy(),process=False).export(skeleton_path)
        result.update(body_mesh_computed=True,mesh_vertices=list(mesh.skin_verts.shape),joints_shape=list(mesh.joints.shape),mesh_path=str(mesh_path))
        result['skeleton_mesh_path']=str(skeleton_path)
        result['skeleton_vertices']=list(mesh.skel_verts.shape)
        # Project the first 25 OpenPose joints back into the full-image crop.
        from lib.body_models.abstract_skeletons import Skeleton_OpenPose25
        camera=cam[0].float()
        translation=torch.stack([camera[1],camera[2],10000/(256*camera[0]+1e-9)])
        joints=mesh.joints[0]+translation
        uv=(joints[:,:2]/joints[:,2:])*5000+128
        uv=uv.cpu().numpy()
        uv=(uv-128)*side/256+np.array([w/2,h/2])
        preview=img.copy()
        for left,right in Skeleton_OpenPose25.bones:
            pts=uv[[left,right]]
            if np.isfinite(pts).all() and np.abs(pts).max()<100000:
                cv2.line(preview,tuple(pts[0].astype(int)),tuple(pts[1].astype(int)),(50,220,80),2)
        preview_path=a.output.with_name(a.output.stem+'-preview.jpg')
        cv2.imwrite(str(preview_path),preview)
        result['preview_path']=str(preview_path)
    if a.benchmark_steps:
        # Synthetic targets check optimizer/gradient flow. These are not data labels.
        target=params['poses'].detach().clone()+.01
        head.train()
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-5)
        scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
        loss_pipeline=None
        if a.with_body:
            import pytorch_lightning as pl
            from lib.modeling.pipelines.hsmr import HSMRPipeline
            from lib.modeling.losses import Keypoint2DLoss,Keypoint3DLoss,ParameterLoss
            from lib.utils.camera import perspective_projection
            # Reuse loaded modules to call the upstream complete main-loss path.
            loss_pipeline=HSMRPipeline.__new__(HSMRPipeline)
            pl.LightningModule.__init__(loss_pipeline)
            loss_pipeline.cfg=cfg.pipeline.cfg
            loss_pipeline.skel_model=body
            loss_pipeline.backbone=backbone
            loss_pipeline.head=head
            loss_pipeline.kp_2d_loss=Keypoint2DLoss(loss_type='l1')
            loss_pipeline.kp_3d_loss=Keypoint3DLoss(loss_type='l1')
            loss_pipeline.params_loss=ParameterLoss()
            loss_pipeline.to(device)
            gt_params={k:v.detach().float().clone() for k,v in params.items()}
            gt_params['poses']=gt_params['poses']+.01
            gt_params['poses_orient']=gt_params['poses'][:,:3]
            gt_params['poses_body']=gt_params['poses'][:,3:]
            with torch.no_grad():
                gt_mesh=body(poses=gt_params['poses'],betas=gt_params['betas'],skelmesh=False)
                gt_projection=perspective_projection(gt_mesh.joints,translation.unsqueeze(0),focal_length=x.new_full((1,2),5000/256))
                conf=x.new_ones(1,44,1)
                gt3d=torch.cat([gt_mesh.joints,conf],dim=-1)
                gt2d=torch.cat([gt_projection,conf],dim=-1)
                has_params={k:x.new_ones(1) for k in ['poses_orient','poses_body','betas']}
        sync(device);start=time.perf_counter()
        for _ in range(a.benchmark_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(),autocast():
                feats=backbone(x)
            with autocast():
                prediction,pred_cam=head(feats)
            if loss_pipeline is None:
                loss=(prediction['poses'].float()-target.float()).square().mean()
            else:
                prediction={k:v.float() for k,v in prediction.items()}
                pred_mesh=body(poses=prediction['poses'],betas=prediction['betas'],skelmesh=False)
                pred_cam=pred_cam.float()
                pred_translation=torch.stack([pred_cam[:,1],pred_cam[:,2],10000/(256*pred_cam[:,0]+1e-9)],dim=-1)
                projection=perspective_projection(pred_mesh.joints,pred_translation,focal_length=x.new_full((1,2),5000/256))
                loss,_=loss_pipeline._compute_losses_main(cfg.pipeline.cfg.loss_weights,pred_mesh.joints,gt3d,projection,gt2d,prediction,gt_params,has_params)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite benchmark loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer);scaler.update()
        sync(device)
        result['synthetic_head_benchmark']={'steps':a.benchmark_steps,'seconds_per_step':(time.perf_counter()-start)/a.benchmark_steps,
            'includes':'frozen backbone forward + head forward/backward + AdamW' + (' + SKEL mesh/joints + original HSMR main losses' if a.with_body else ''),
            'excludes':'SKELify, data loading, checkpoints' + ('' if a.with_body else ', SKEL mesh/keypoint losses'),
            'final_loss':loss.detach().item()}
    if device.type=='cuda':
        result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['poses','betas','camera']},indent=2),flush=True)

if __name__=='__main__':
    main()
