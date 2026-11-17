"""One fresh-process speed/memory benchmark, including projection and planning."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from romo_gs.tiled import render_tiled,render_dense_camera
from romo_gs.measurement import MemoryMonitor,sync
from rgbd_experiment import load_scene,initial_colors,kwargs


def main(args):
    device=args.device
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    torch.set_num_threads(1)
    data=load_scene('freiburg1_xyz',device)
    n=args.gaussians
    means=data['init_means'][:n].clone().requires_grad_()
    covariance=torch.diag_embed(data['init_scale'][:n,None].expand(-1,3).square()).clone().requires_grad_()
    color=initial_colors(data,'controlled')[:n].clone().requires_grad_()
    opacity=torch.full((n,),.8,device=device,requires_grad=True)
    kw=kwargs(data,0,means,data['init_scale'][:n,None].expand(-1,3),color,opacity,.08)
    kw['covariance']=covariance
    fn=render_dense_camera if args.backend=='dense' else render_tiled
    extra={} if args.backend=='dense' else {'tile_size':args.tile,'gaussian_chunk':256,'recompute':args.backend=='checkpoint','q_cutoff':24.}
    params=[means,covariance,color,opacity]
    # Separate retained-forward-storage audit. Count unique actual storages held
    # by autograd, not a hand-estimated N*H*W memory formula.
    saved={}
    def pack(t):
        storage=t.untyped_storage();key=(str(t.device),storage.data_ptr(),storage.nbytes())
        saved[key]=storage.nbytes();return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        image=fn(**kw,method=args.method,samples=args.samples,**extra)
    saved_bytes=sum(saved.values())
    torch.autograd.grad(image.square().mean(),params)
    del image;gc.collect();sync(device)
    if device=='mps':torch.mps.empty_cache()
    timings=[]
    with MemoryMonitor(device) as monitor:
        for i in range(args.repeats+1):
            sync(device);start=time.perf_counter()
            image=fn(**kw,method=args.method,samples=args.samples,**extra)
            sync(device);middle=time.perf_counter()
            grads=torch.autograd.grad(image.square().mean(),params)
            sync(device);end=time.perf_counter()
            if i:timings.append({'forward_seconds':middle-start,'backward_seconds':end-middle,'total_seconds':end-start})
            del image,grads
    result=vars(args)|{'scene':'freiburg1_xyz/frame0','size':[96,72],'cpu_threads':1,'warmups':1,
                       'median_forward_seconds':float(np.median([r['forward_seconds'] for r in timings])),
                       'median_backward_seconds':float(np.median([r['backward_seconds'] for r in timings])),
                       'median_total_seconds':float(np.median([r['total_seconds'] for r in timings])),
                       'retained_forward_unique_storage_bytes':saved_bytes,'memory':monitor.result(),'raw_timings':timings,
                       'comparison_note':'Fresh process per case; full position/covariance/color/opacity gradients. CPU planning+GPU computation included. No training process runs concurrently.'}
    out=Path('results/phase2/profiles');out.mkdir(exist_ok=True)
    path=out/f'{device}_{args.backend}_{args.method}_n{n}_tile{args.tile}.json'
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps'],required=True)
    p.add_argument('--backend',choices=['dense','tiled','checkpoint'],required=True)
    p.add_argument('--method',choices=['analytic','midpoint'],required=True)
    p.add_argument('--gaussians',type=int,default=2400);p.add_argument('--samples',type=int,default=10)
    p.add_argument('--tile',type=int,default=32);p.add_argument('--repeats',type=int,default=3)
    main(p.parse_args())
