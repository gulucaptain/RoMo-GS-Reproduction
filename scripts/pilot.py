"""Controlled, known-camera 3D reconstruction pilot; NOT online SLAM.

Targets use 64-node Gauss-Legendre integration of full 3D camera translations,
not our analytic renderer. No author data or pretrained model is needed.
"""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from romo_gs.core import render


def sync(device):
    if str(device) == "mps": torch.mps.synchronize()


def psnr(x, y):
    return float(-10*torch.log10((x-y).square().mean().clamp_min(1e-12)))


def scene():
    rng=np.random.default_rng(20261117)
    means=rng.uniform([-1.05,-.85,2.7],[1.05,.85,4.2],(24,3)).astype("float32")
    scales=rng.uniform(.055,.14,(24,3)).astype("float32")
    colors=rng.uniform(.15,.95,(24,3)).astype("float32")
    opacity=rng.uniform(.55,.85,24).astype("float32")
    centers=np.stack([np.linspace(-.6,.6,9),.13*np.sin(np.arange(9)),np.zeros(9)],-1).astype("float32")
    return {k:torch.tensor(v) for k,v in dict(means=means,scales=scales,colors=colors,opacity=opacity,centers=centers).items()}


def args_render(data, i, means=None, scales=None, colors=None, opacity=None):
    return dict(means=data["means"] if means is None else means,
                covariance=torch.diag_embed((data["scales"] if scales is None else scales).square()),
                colors=data["colors"] if colors is None else colors,
                opacity=data["opacity"] if opacity is None else opacity,
                center=data["centers"][i], rotation=data["rotation"],focal=36.,size=32,
                linear_velocity=data["velocity"],angular_velocity=data["omega"],exposure=.45)


def generate(path):
    torch.set_num_threads(1)
    data=scene(); data.update(rotation=torch.eye(3),velocity=torch.tensor([.8,.12,0.]),omega=torch.zeros(3))
    nodes, weights=np.polynomial.legendre.leggauss(64)
    blurry=[]; sharp=[]
    with torch.no_grad():
        for i in range(9):
            kwargs=args_render(data,i)
            sharp.append(render(**kwargs,method="static"))
            total=torch.zeros(32,32,3)
            for t,w in zip(nodes,weights):
                shifted=dict(kwargs,center=kwargs["center"]+data["velocity"]*float(t*.45/2))
                total+=render(**shifted,method="static")*float(w/2)
            blurry.append(total)
    data.update(blurry=torch.stack(blurry),sharp=torch.stack(sharp))
    np.savez_compressed(path,**{k:v.numpy() for k,v in data.items()})


def ssim(x, y):
    # Local SSIM, 7x7 uniform window, RGB channelwise; disclosed independent loss.
    import torch.nn.functional as F
    a=x.permute(2,0,1)[None]; b=y.permute(2,0,1)[None]
    pool=lambda q:F.avg_pool2d(q,7,1)
    ma,mb=pool(a),pool(b)
    va,vb=pool(a*a)-ma*ma,pool(b*b)-mb*mb
    cov=pool(a*b)-ma*mb
    return (((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean()


def load(path, device):
    with np.load(path) as d: return {k:torch.tensor(d[k],device=device) for k in d.files}


def run(device, seeds, steps, out, methods):
    out.mkdir(parents=True,exist_ok=True)
    dataset=Path("results/pilot_scene.npz")
    if not dataset.exists(): generate(dataset)
    if device=="mps" and not torch.backends.mps.is_available(): raise RuntimeError("Mac GPU unavailable; no silent CPU fallback")
    torch.set_num_threads(1)
    data=load(dataset,device)
    config={"device":device,"seeds":seeds,"steps":steps,"methods":methods,"train_views":[0,1,3,4,6,7],"test_views":[2,5,8],
            "size":32,"gaussians":24,"focal":36,"exposure":.45,"velocity":[.8,.12,0.],
            "lr":.015,"loss":"0.8 L1 + 0.2 (1 - SSIM), 7x7 uniform valid window",
            "observation":"64-node Gauss-Legendre average of full static 3D renders at translated camera centers",
            "initialization":"known Gaussian correspondences, GT means + N(0,0.05), log scales + N(0,0.1), color logits + N(0,0.2), opacity logits + N(0,0.1)",
            "scope":"synthetic known-camera offline fixed-count diagonal-world-covariance pilot; camera and velocities frozen"}
    (out/"config.json").write_text(json.dumps(config,indent=2))
    results=[]
    for seed in seeds:
        for method in methods:
            torch.manual_seed(seed)
            # Draw the SAME CPU initialization for both devices and all methods.
            initial=load(dataset,"cpu")
            means=torch.nn.Parameter((initial["means"]+.05*torch.randn(24,3)).to(device))
            logs=torch.nn.Parameter((initial["scales"].log()+.1*torch.randn(24,3)).to(device))
            color=torch.nn.Parameter((torch.logit(initial["colors"])+.2*torch.randn(24,3)).to(device))
            op=torch.nn.Parameter((torch.logit(initial["opacity"])+.1*torch.randn(24)).to(device))
            optim=torch.optim.Adam([means,logs,color,op],lr=.015)
            trace=[];sync(device);start=time.perf_counter()
            for step in range(steps):
                i=config["train_views"][step%6]
                pred=render(**args_render(data,i,means,logs.exp(),color.sigmoid(),op.sigmoid()),method=method,samples=10)
                loss=.8*(pred-data["blurry"][i]).abs().mean()+.2*(1-ssim(pred,data["blurry"][i]))
                optim.zero_grad();loss.backward();optim.step()
                if not all(torch.isfinite(p).all() for p in [means,logs,color,op]): raise RuntimeError("nonfinite optimization")
                trace.append({"step":step,"view":i,"loss":float(loss.detach().cpu())})
            sync(device); elapsed=time.perf_counter()-start
            with torch.no_grad():
                sharp=[];blur=[]
                for i in config["test_views"]:
                    kwargs=args_render(data,i,means,logs.exp(),color.sigmoid(),op.sigmoid())
                    sharp.append(render(**kwargs,method="static"))
                    blur.append(render(**kwargs,method=method,samples=10))
                sharp=torch.stack(sharp);blur=torch.stack(blur)
                record={"seed":seed,"method":method,"steps":steps,"seconds":elapsed,
                        "sharp_heldout_psnr":psnr(sharp,data["sharp"][config["test_views"]]),
                        "blur_heldout_psnr":psnr(blur,data["blurry"][config["test_views"]]),
                        "matched_mean_rmse":float((means-data["means"]).square().mean().sqrt().cpu()),
                        "final_training_loss":trace[-1]["loss"]}
            stem=f"{method}_seed{seed}"
            np.savez_compressed(out/f"{stem}.npz",means=means.detach().cpu().numpy(),scales=logs.exp().detach().cpu().numpy(),
                                colors=color.sigmoid().detach().cpu().numpy(),opacity=op.sigmoid().detach().cpu().numpy(),
                                heldout_sharp=sharp.cpu().numpy(),heldout_blur=blur.cpu().numpy())
            (out/f"{stem}_trace.json").write_text(json.dumps(trace,indent=2))
            results.append(record);(out/"summary.json").write_text(json.dumps(results,indent=2))
            print(json.dumps(record),flush=True)
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--device",choices=["cpu","mps"],default="cpu")
    p.add_argument("--seeds",nargs="+",type=int,default=[0,1,2]);p.add_argument("--steps",type=int,default=240)
    p.add_argument("--methods",nargs="+",choices=["static","analytic","sampled","sampled_midpoint"],default=["static","analytic","sampled","sampled_midpoint"])
    p.add_argument("--output",default=None);a=p.parse_args()
    run(a.device,a.seeds,a.steps,Path(a.output or f"results/pilot_{a.device}"),a.methods)
