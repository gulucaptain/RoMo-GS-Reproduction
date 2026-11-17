"""One isolated real RGB-D reconstruction experiment, paired by seed."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from romo_gs.tiled import render_tiled,render_dense_camera
from romo_gs.measurement import MemoryMonitor,sync


def load_scene(sequence,device='cpu'):
    with np.load(f'data/prepared/{sequence}.npz') as src:
        return {k:torch.as_tensor(src[k],device=device) for k in src.files}


def initial_colors(data,track):
    observations=data['rgb'] if track=='native' else data['blur']
    valid=data['native_mask'] if track=='native' else data['controlled_mask']
    group=data['fusion_groups'].long();images=data['fusion_images'].long();pixels=data['fusion_pixels'].long()
    color=observations.flatten(1,2)[images,pixels]
    ok=valid.flatten(1,2)[images,pixels]
    n=len(data['init_means']);den=torch.zeros(n,device=group.device)
    den.index_add_(0,group,ok.float())
    result=torch.zeros(n,3,device=group.device)
    result.index_add_(0,group,color*ok[:,None])
    return torch.where(den[:,None]>0,result/den[:,None].clamp_min(1),torch.full_like(result,.5)).clamp(.001,.999)


def kwargs(data,i,means,scales,color,opacity,exposure):
    h,w=data['rgb'].shape[1:3]
    return dict(means=means,covariance=torch.diag_embed(scales.square()),opacity=opacity,colors=color,
                center=data['centers'][i],rotation=data['rotations'][i],intrinsics=data['intrinsics'],height=h,width=w,
                linear_velocity=data['velocities'][i],angular_velocity=data['omegas'][i],exposure=exposure)


def masked_ssim(x,y,mask):
    a=x.permute(2,0,1)[None];b=y.permute(2,0,1)[None]
    pool=lambda q:F.avg_pool2d(q,7,1)
    ma,mb=pool(a),pool(b);va,vb=pool(a*a)-ma*ma,pool(b*b)-mb*mb;cov=pool(a*b)-ma*mb
    score=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    valid=pool(mask[None,None].float())>.999
    return (score*valid).sum()/(valid.sum().clamp_min(1)*3)


def evaluate(data,means,scales,color,opacity,exposure,method,track,tile_size=32,chunk=256,samples=10):
    ids=data['test'].cpu().tolist();allpred=[];blurs=[];perview=[];sq=0.;count=0;depthsq=0.;alphas=[];ssims=[]
    center_sq=0.;observation_sq=0.;full_sq=0.;full_count=0
    mask_key='native_mask' if track=='native' else 'controlled_mask'
    with torch.no_grad():
        for i in ids:
            kw=kwargs(data,i,means,scales,color,opacity,exposure)
            pred=render_tiled(**kw,method='static',tile_size=tile_size,gaussian_chunk=chunk,recompute=False)
            blurred=render_tiled(**kw,method=method,samples=samples,tile_size=tile_size,gaussian_chunk=chunk,recompute=False)
            valid=data[mask_key][i];target=data['rgb'][i]
            measured=blurred[...,:3] if track=='native' else pred[...,:3]
            s=((measured-target).square()*valid[...,None]).sum();n=valid.sum()*3
            c_s=((pred[...,:3]-target).square()*valid[...,None]).sum()
            observed=data['rgb'][i] if track=='native' else data['blur'][i]
            o_s=((blurred[...,:3]-observed).square()*valid[...,None]).sum()
            full_valid=data['native_mask'][i]
            f_s=((pred[...,:3]-target).square()*full_valid[...,None]).sum()
            f_n=full_valid.sum()*3
            d=pred[...,3]/pred[...,4].clamp_min(1e-6)
            ds=((d-data['depth'][i]).square()*valid).sum()
            ssim=masked_ssim(measured,target,valid)
            sq+=float(s.cpu());count+=int(n.cpu());depthsq+=float(ds.cpu())
            center_sq+=float(c_s.cpu());observation_sq+=float(o_s.cpu())
            full_sq+=float(f_s.cpu());full_count+=int(f_n.cpu())
            alphas.append(float((pred[...,4][valid]>.9).float().mean().cpu()));ssims.append(float(ssim.cpu()))
            perview.append({'view':i,'masked_reference_psnr':float((-10*torch.log10(s/n)).cpu()),'masked_ssim':ssims[-1],
                            'depth_rmse_m':float(torch.sqrt(ds/valid.sum()).cpu()),'alpha90_coverage':alphas[-1],
                            'evaluation_pixels':int(valid.sum().cpu()),
                            'full_native_valid_center_psnr':float((-10*torch.log10(f_s/f_n)).cpu()),
                            'full_native_valid_pixels':int(full_valid.sum().cpu())})
            allpred.append(pred.cpu().numpy());blurs.append(blurred[...,:3].cpu().numpy())
    summary={'reference_psnr':float(-10*np.log10(sq/count)),'reference_ssim_mean':float(np.mean(ssims)),
             'center_vs_original_rgb_psnr':float(-10*np.log10(center_sq/count)),
             'observation_psnr':float(-10*np.log10(observation_sq/count)),
             'full_native_valid_center_psnr':float(-10*np.log10(full_sq/full_count)),
             'full_native_valid_rgb_values':full_count,
             'depth_rmse_m':float(np.sqrt(depthsq/(count/3))),'alpha90_coverage':float(np.mean(alphas)),
             'evaluation_rgb_values':count,'per_view':perview}
    return summary,np.array(allpred),np.array(blurs)


def run(args):
    device=args.device
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no fallback')
    torch.set_num_threads(1)
    data=load_scene(args.sequence,device)
    stem=f'{args.sequence}_{args.track}_{args.method}_{device}_seed{args.seed}'
    out=Path(args.output or 'results/phase2/runs')/stem;out.mkdir(parents=True,exist_ok=True)
    # CPU RNG for paired initialization across devices and methods.
    torch.manual_seed(args.seed)
    n=len(data['init_means'])
    basecolor=initial_colors(data,args.track)
    offset=torch.nn.Parameter(torch.zeros(n,3,device=device))
    logscale=torch.nn.Parameter(torch.zeros(n,3,device=device))
    color=torch.nn.Parameter(torch.logit(basecolor)+(.03*torch.randn(n,3)).to(device))
    op=torch.nn.Parameter(torch.full((n,),float(np.log(.8/.2)),device=device))
    optim=torch.optim.Adam([{'params':[offset],'lr':.015},{'params':[logscale],'lr':.015},
                            {'params':[color],'lr':.04},{'params':[op],'lr':.02}])
    def values():
        means=data['init_means']+.03*offset.tanh()
        scales=data['init_scale'][:,None]*logscale.clamp(-.7,.7).exp()
        return means,scales,color.sigmoid(),op.sigmoid()
    exposure=1/60 if args.track=='native' else .08
    target=data['rgb'] if args.track=='native' else data['blur']
    masks=data['native_mask'] if args.track=='native' else data['controlled_mask']
    config=vars(args)|{'exposure_seconds':exposure,'native_exposure_is_assumed':args.track=='native','gaussians':n,
                      'train':data['train'].cpu().tolist(),'test':data['test'].cpu().tolist(),'resolution':[96,72],
                      'loss':'0.8 masked L1+0.2 valid-window SSIM+1e-4 mean squared offset parameter',
                      'position_bound_m':.03,'log_scale_bound':[-.7,.7],'geometry_source':'training RGB-D only',
                      'color_source':'matching training observation track; controlled colors from blurred images only',
                      'q_cutoff':24.,'camera_states':'mocap fixed, no online pose optimization'}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    initial,_,_=evaluate(data,*values(),exposure,args.method,args.track,args.tile,args.chunk,args.samples)
    trace=[];timings=[];sync(device);start=time.perf_counter()
    with MemoryMonitor(device) as monitor:
        for step in range(args.steps):
            i=config['train'][step%len(config['train'])]
            kw=kwargs(data,i,*values(),exposure)
            sync(device);begin=time.perf_counter()
            output=render_tiled(**kw,method=args.method,samples=args.samples,tile_size=args.tile,
                                gaussian_chunk=args.chunk,recompute=True)
            pred=output[...,:3];valid=masks[i]
            l1=((pred-target[i]).abs()*valid[...,None]).sum()/(valid.sum()*3)
            loss=.8*l1+.2*(1-masked_ssim(pred,target[i],valid))+1e-4*offset.square().mean()
            optim.zero_grad();loss.backward();optim.step();sync(device)
            elapsed=time.perf_counter()-begin;timings.append(elapsed)
            if not all(torch.isfinite(p).all() for p in [offset,logscale,color,op]):raise RuntimeError('nonfinite optimization')
            trace.append({'step':step,'view':i,'loss':float(loss.detach().cpu()),'l1':float(l1.detach().cpu()),'seconds':elapsed})
            if step%50==0:print(stem,step,trace[-1],flush=True)
    sync(device);training_seconds=time.perf_counter()-start
    final,images,blurs=evaluate(data,*values(),exposure,args.method,args.track,args.tile,args.chunk,args.samples)
    summary={'sequence':args.sequence,'track':args.track,'method':args.method,'device':device,'seed':args.seed,
             'steps':args.steps,'gaussians':n,'initial':initial,'final':final,'training_seconds':training_seconds,
             'median_step_seconds':float(np.median(timings)),'memory':monitor.result(),
             'interpretation':'Held-out original RGB reference; controlled track adds generated blur; native sensor clean GT unavailable.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'trace.json').write_text(json.dumps(trace,indent=2))
    checkpoint={k:v.detach().cpu().numpy() for k,v in zip(['means','scales','colors','opacity'],values())}
    np.savez_compressed(out/'parameters.npz',**checkpoint)
    local=Path('results/phase2/local_data');local.mkdir(exist_ok=True)
    np.savez_compressed(local/f'{stem}_predictions.npz',heldout=images,blur=blurs)
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',choices=['freiburg1_xyz','freiburg1_desk'],required=True)
    p.add_argument('--track',choices=['native','controlled'],default='controlled');p.add_argument('--method',choices=['analytic','midpoint','static'],required=True)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=300);p.add_argument('--samples',type=int,default=10)
    p.add_argument('--tile',type=int,default=32);p.add_argument('--chunk',type=int,default=256);p.add_argument('--output',default=None)
    run(p.parse_args())
