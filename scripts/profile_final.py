"""Isolated cost of a saved final model, separate from initializer benchmarks."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from romo_gs.tiled import render_tiled
from romo_gs.measurement import MemoryMonitor,sync
from rgbd_experiment import load_scene,kwargs


def run(folder,device):
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    torch.set_num_threads(1)
    config=json.loads((folder/'config.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    data=load_scene(config['sequence'],device)
    with np.load(folder/'parameters.npz') as p:
        means=torch.tensor(p['means'],device=device,requires_grad=True)
        covariance=torch.diag_embed(torch.tensor(p['scales'],device=device).square()).requires_grad_()
        colors=torch.tensor(p['colors'],device=device,requires_grad=True)
        opacity=torch.tensor(p['opacity'],device=device,requires_grad=True)
    templates={}
    for i in config['train']+config['test']:
        templates[i]=kwargs(data,i,means,torch.ones_like(means),colors,opacity,config['exposure_seconds'])
        templates[i]['covariance']=covariance
    def render(i,recompute):
        return render_tiled(**templates[i],method=config['method'],samples=10,tile_size=32,gaussian_chunk=256,recompute=recompute)
    gc.collect();sync(device)
    if device=='mps':torch.mps.empty_cache()
    inference=[]
    # Warm each held-out shape once before recording per-view repeats.
    with torch.no_grad():
        for i in config['test']:render(i,False)
    sync(device)
    with MemoryMonitor(device) as inference_memory,torch.no_grad():
        for repeat in range(3):
            for i in config['test']:
                sync(device);start=time.perf_counter();image=render(i,False);sync(device)
                inference.append({'repeat':repeat,'view':i,'seconds':time.perf_counter()-start});del image
    inference_stats=inference_memory.result()
    params=[means,covariance,colors,opacity]
    training=[]
    image=render(config['train'][0],True);torch.autograd.grad(image.square().mean(),params);del image
    gc.collect();sync(device)
    if device=='mps':torch.mps.empty_cache()
    with MemoryMonitor(device) as backward_memory:
        for repeat in range(3):
            sync(device);start=time.perf_counter();image=render(config['train'][0],True);sync(device);middle=time.perf_counter()
            grads=torch.autograd.grad(image.square().mean(),params);sync(device);end=time.perf_counter()
            training.append({'repeat':repeat,'forward_seconds':middle-start,'backward_seconds':end-middle,'total_seconds':end-start})
            del image,grads
    output={'run':folder.name,'device':device,'seed':config['seed'],'sequence':config['sequence'],'track':config['track'],'method':config['method'],
            'gaussians':2400,'resolution':[96,72],'tile':32,'chunk':256,'samples':10,
            'inference_median_seconds':float(np.median([x['seconds'] for x in inference])),
            'forward_backward_median_seconds':float(np.median([x['total_seconds'] for x in training])),
            'inference_memory':inference_stats,'forward_backward_memory':backward_memory.result(),
            'inference_raw':inference,'forward_backward_raw':training,
            'quality':summary['final'],
            'parameter_sha256':hashlib.sha256((folder/'parameters.npz').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Saved150-step seed0 model; inference15timings across5held-out views; forward/back3timings atfirst training view; fresh process per case, no concurrent training; renderer loss is mean square of all5output channels, not a full optimizer step'}
    out=Path('results/phase2/final_profiles');out.mkdir(exist_ok=True)
    (out/(folder.name+'.json')).write_text(json.dumps(output,indent=2));print(json.dumps(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--device',choices=['cpu','mps'],required=True);a=p.parse_args()
    run(Path(a.run),a.device)
