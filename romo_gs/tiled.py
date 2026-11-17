"""Rectangular calibrated cameras, conservative swept bounds and tiled splatting.

This is a portable PyTorch backend, not the author's AccuTile/CUDA kernel.
Projection is shared; tile inputs are checkpointed during training. Backward
recomputes one tile at a time instead of retaining all pixel footprints.
"""
from functools import partial
import math
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from .core import integrated_opacity, inverse2


def project_camera(means, covariance, center, rotation, intrinsics, velocity, omega, near=.05):
    xyz = (means-center) @ rotation.T
    x,y,z = xyz.unbind(-1)
    zz=z.clamp_min(near)
    fx,fy,cx,cy = intrinsics.unbind(-1)
    uv=torch.stack([fx*x/zz+cx,fy*y/zz+cy],-1)
    zero=torch.zeros_like(zz)
    J=torch.stack([fx/zz,zero,-fx*x/zz.square(),zero,fy/zz,-fy*y/zz.square()],-1).reshape(-1,2,3)
    screen=J@(rotation@covariance@rotation.T)@J.transpose(-1,-2)
    screen=screen+.3*torch.eye(2,dtype=means.dtype,device=means.device)
    pixel_speed=-(J@(torch.linalg.cross(omega.expand_as(xyz),xyz)+velocity)[...,None]).squeeze(-1)
    return uv,inverse2(screen),pixel_speed,z,screen


def tile_pixels(y0,y1,x0,x1,like):
    yy,xx=torch.meshgrid(torch.arange(y0,y1,device=like.device,dtype=like.dtype),
                         torch.arange(x0,x1,device=like.device,dtype=like.dtype),indexing='ij')
    return torch.stack([xx.flatten(),yy.flatten()],-1)


def midpoint_batched(uv,Q,v,o,colors,z,pixels,exposure,samples,chunk):
    """Five time samples per tensor batch; exact same midpoint quadrature.

    Gaussian compositing remains sequential in depth. This changes kernel
    batching, not the exposure model or opacity-compositing order.
    """
    total=torch.zeros((len(pixels),5),device=uv.device,dtype=uv.dtype)
    for temporal_start in range(0,samples,5):
        s=min(5,samples-temporal_start)
        times=(torch.arange(temporal_start,temporal_start+s,device=uv.device,dtype=uv.dtype)+.5)/samples-.5
        trans=torch.ones((s,len(pixels)),device=uv.device,dtype=uv.dtype)
        image=torch.zeros((s,len(pixels),5),device=uv.device,dtype=uv.dtype)
        for start in range(0,len(uv),chunk):
            sl=slice(start,start+chunk)
            d=uv[None,sl,None]-pixels[None,None]+times[:,None,None,None]*exposure*v[None,sl,None]
            q=torch.einsum('snpi,nij,snpj->snp',d,Q[sl],d)
            alpha=(o[None,sl,None]*(-q/2).exp()).clamp(0,.999)
            cumulative=torch.cumprod(torch.cat([torch.ones_like(alpha[:,:1]),1-alpha],1),1)
            weights=trans[:,None]*cumulative[:,:-1]*alpha
            channels=torch.cat([colors[sl],z[sl,None],torch.ones_like(z[sl,None])],-1)
            image=image+torch.matmul(weights.transpose(-1,-2),channels)
            trans=trans*cumulative[:,-1]
        total=total+image.sum(0)/samples
    return total


def footprint_block(uv,Q,v,o,colors,z,pixels,exposure,method,samples,chunk):
    """Ordered Gaussian streaming; emits RGB, weighted depth, accumulated alpha."""
    if method=='midpoint':return midpoint_batched(uv,Q,v,o,colors,z,pixels,exposure,samples,chunk)
    if method == 'analytic': times=[None]
    elif method == 'static': times=[0.]
    elif method == 'midpoint_loop': times=[(i+.5)/samples-.5 for i in range(samples)]
    else: raise ValueError(method)
    total=torch.zeros((len(pixels),5),device=uv.device,dtype=uv.dtype)
    for tau in times:
        trans=torch.ones(len(pixels),device=uv.device,dtype=uv.dtype)
        image=torch.zeros_like(total)
        for start in range(0,len(uv),chunk):
            sl=slice(start,start+chunk)
            delta=uv[sl,None]-pixels
            if tau is None:
                alpha=integrated_opacity(delta,Q[sl,None],v[sl,None],o[sl,None],exposure)
            else:
                d=delta+tau*exposure*v[sl,None]
                q=torch.einsum('npi,nij,npj->np',d,Q[sl],d)
                alpha=o[sl,None]*(-q/2).exp()
            alpha=alpha.clamp(0,.999)
            cumulative=torch.cumprod(torch.cat([torch.ones_like(alpha[:1]),1-alpha],0),0)
            weights=trans[None]*cumulative[:-1]*alpha
            channels=torch.cat([colors[sl],z[sl,None],torch.ones_like(z[sl,None])],-1)
            image=image+weights.T@channels
            trans=trans*cumulative[-1]
        total=total+image/len(times)
    return total


def build_plan(uv,screen,v,z,height,width,tile_size,exposure,q_cutoff=24.):
    """CPU metadata only. Bounds cover the entire LINEAR screen-space sweep.

    q_cutoff=None disables culling for exact value/gradient comparisons.
    Sorting and support membership are discrete; no gradients through them.
    """
    with torch.no_grad():
        u=uv.detach().cpu().numpy();cov=screen.detach().cpu().numpy()
        vel=v.detach().cpu().numpy();depth=z.detach().cpu().numpy()
    order=np.argsort(depth,kind='stable');order=order[np.isfinite(depth[order]) & (depth[order]>.05)]
    if q_cutoff is not None:
        radius=np.sqrt(q_cutoff*np.maximum(np.diagonal(cov,axis1=-2,axis2=-1),0))+abs(vel)*exposure/2+1.
        lower=u-radius;upper=u+radius
    plan=[]
    for y in range(0,height,tile_size):
        for x in range(0,width,tile_size):
            y1=min(y+tile_size,height);x1=min(x+tile_size,width)
            ids=order
            if q_cutoff is not None:
                keep=(upper[ids,0]>=x)&(lower[ids,0]<=x1-1)&(upper[ids,1]>=y)&(lower[ids,1]<=y1-1)
                ids=ids[keep]
            plan.append((y,y1,x,x1,ids))
    return plan


def render_tiled(means,covariance,opacity,colors,center,rotation,intrinsics,height,width,
                 linear_velocity,angular_velocity,exposure,method='analytic',samples=10,
                 tile_size=16,gaussian_chunk=128,q_cutoff=24.,recompute=True,return_stats=False):
    uv,Q,v,z,screen=project_camera(means,covariance,center,rotation,intrinsics,linear_velocity,angular_velocity)
    if method=='static': exposure=0.
    plan=build_plan(uv,screen,v,z,height,width,tile_size,exposure,q_cutoff)
    fn=partial(footprint_block,exposure=exposure,method=method,samples=samples,chunk=gaussian_chunk)
    rows=[];current=[];previous_y=0;active=[]
    for y,y1,x,x1,ids in plan:
        if y!=previous_y:
            rows.append(torch.cat(current,1));current=[];previous_y=y
        active.append(len(ids))
        if not len(ids):
            # Keep a zero derivative path for empty images during gradient checks.
            result=(uv.sum()*0+torch.zeros((y1-y,x1-x,5),device=means.device,dtype=means.dtype))
        else:
            idx=torch.as_tensor(ids.copy(),device=means.device,dtype=torch.long)
            pixels=tile_pixels(y,y1,x,x1,uv)
            inputs=(uv[idx],Q[idx],v[idx],opacity[idx],colors[idx],z[idx],pixels)
            if recompute and torch.is_grad_enabled() and any(t.requires_grad for t in inputs):
                out=checkpoint(fn,*inputs,use_reentrant=False,preserve_rng_state=False)
            else: out=fn(*inputs)
            result=out.reshape(y1-y,x1-x,5)
        current.append(result)
    rows.append(torch.cat(current,1))
    output=torch.cat(rows,0)
    stats={'gaussians':len(means),'tiles':len(plan),'max_active_gaussians':max(active),
           'mean_active_gaussians':float(np.mean(active)),'gaussian_tile_pairs':sum(active),
           'largest_footprint_matrix_elements':min(max(active),gaussian_chunk)*min(tile_size,height)*min(tile_size,width),
           'q_cutoff':q_cutoff,'recompute':recompute,'tile_size':tile_size,'gaussian_chunk':gaussian_chunk}
    return (output,stats) if return_stats else output


def render_dense_camera(means,covariance,opacity,colors,center,rotation,intrinsics,height,width,
                        linear_velocity,angular_velocity,exposure,method='analytic',samples=10):
    uv,Q,v,z,_=project_camera(means,covariance,center,rotation,intrinsics,linear_velocity,angular_velocity)
    ids=z.argsort();ids=ids[z[ids]>.05]
    pixels=tile_pixels(0,height,0,width,uv)
    return footprint_block(uv[ids],Q[ids],v[ids],opacity[ids],colors[ids],z[ids],pixels,
                           exposure,method,samples,max(len(ids),1)).reshape(height,width,5)
