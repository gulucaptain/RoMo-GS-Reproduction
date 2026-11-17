"""Synchronized CPU/MPS value, gradient and renderer timing comparison."""
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from romo_gs.core import integrated_opacity, render
from pilot import load, args_render, sync


def main():
    if not torch.backends.mps.is_available(): raise RuntimeError("MPS required")
    torch.set_num_threads(1)
    torch.manual_seed(88)
    delta=torch.randn(128,2)*2
    Q=torch.tensor([[1.2,.2],[.2,.8]]).expand(128,2,2).clone()
    velocity=torch.randn(128,2)*3
    cases={};timings=[]
    for device in ["cpu","mps"]:
        params=[x.to(device).detach().requires_grad_() for x in [delta,Q,velocity,torch.full((128,),.8),torch.full((128,),.4)]]
        output=integrated_opacity(*params)
        grads=torch.autograd.grad(output.sum(),params)
        cases[device]={"value":output.detach().cpu().numpy(),"grads":[g.cpu().numpy() for g in grads]}
        data=load(Path("results/pilot_scene.npz"),device)
        kwargs=args_render(data,0)
        kwargs["means"]=kwargs["means"].clone().requires_grad_()
        for method in ["static","analytic","sampled"]:
            for n in ([5,10,20] if method=="sampled" else [1]):
                forward=[];backward=[]
                for j in range(25):
                    sync(device);start=time.perf_counter()
                    image=render(**kwargs,method=method,samples=n)
                    sync(device);middle=time.perf_counter()
                    torch.autograd.grad(image.sum(),kwargs["means"])
                    sync(device);end=time.perf_counter()
                    if j>=5:forward.append(middle-start);backward.append(end-middle)
                timings.append({"device":device,"method":method,"samples":n,
                                "median_forward_ms":float(np.median(forward)*1000),
                                "median_backward_ms":float(np.median(backward)*1000),
                                "forward_seconds":np.array(forward).tolist(),"backward_seconds":np.array(backward).tolist()})
    out={"platform":platform.platform(),"torch":torch.__version__,"mps_built":torch.backends.mps.is_built(),
         "mps_available":torch.backends.mps.is_available(),"mps_fallback_environment":os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"),
         "size":32,"gaussians":24,"cpu_threads":1,"warmups":5,"repeats":20,
         "value_max_abs_cpu_mps":float(abs(cases["cpu"]["value"]-cases["mps"]["value"]).max()),
         "gradient_max_abs_cpu_mps":[float(abs(a-b).max()) for a,b in zip(cases["cpu"]["grads"],cases["mps"]["grads"])],
         "timings":timings}
    Path("results/device_audit.json").write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!="timings"},indent=2))
    assert out["value_max_abs_cpu_mps"]<2e-5
    assert max(out["gradient_max_abs_cpu_mps"])<2e-4


if __name__=="__main__": main()
