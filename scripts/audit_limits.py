"""Near-static branch sweep and independent observation-convergence check."""
import json
from pathlib import Path
import numpy as np
import torch
from romo_gs.core import integrated_opacity, render
from pilot import load, args_render


def main():
    torch.set_num_threads(1)
    rows=[]
    for exposure in [.01,1.]:
        for speed in np.concatenate([[0.], np.logspace(-9,2,111)]):
            d=torch.tensor([.6,-.3],dtype=torch.float64)
            Q=torch.tensor([[1.3,.2],[.2,.9]],dtype=torch.float64)
            entry={"speed":float(speed),"exposure":exposure}
            for mode in ["paper","stable"]:
                v=torch.tensor([speed,0.],dtype=torch.float64,requires_grad=True)
                f=integrated_opacity(d,Q,v,.8,exposure,mode)
                g=torch.autograd.grad(f,v)[0]
                entry[mode]={"value":f.item(),"velocity_gradient":g.tolist()}
            rows.append(entry)
    data=load(Path("results/pilot_scene.npz"),"cpu")
    audits=[]
    with torch.no_grad():
        for i in range(9):
            kwargs=args_render(data,i)
            nodes,weights=np.polynomial.legendre.leggauss(128)
            image=torch.zeros(32,32,3)
            for t,w in zip(nodes,weights):
                shifted=dict(kwargs,center=kwargs["center"]+data["velocity"]*float(t*.45/2))
                image+=render(**shifted,method="static")*float(w/2)
            analytic=render(**kwargs,method="analytic")
            audits.append({"view":i,"quad64_vs128_max_abs":float((image-data["blurry"][i]).abs().max()),
                           "quad64_vs128_mae":float((image-data["blurry"][i]).abs().mean()),
                           "oracle_analytic_vs_full3d_mae":float((analytic-image).abs().mean()),
                           "oracle_analytic_vs_full3d_max_abs":float((analytic-image).abs().max())})
    result={"near_static":rows,"observation_convergence":audits,
            "max_quad64_vs128":max(x["quad64_vs128_max_abs"] for x in audits),
            "near_static_all_finite":all(np.isfinite([r[m]["value"],*r[m]["velocity_gradient"]]).all() for r in rows for m in ["paper","stable"])}
    Path("results/limits_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="near_static"},indent=2))
    assert result["near_static_all_finite"] and result["max_quad64_vs128"]<2e-6


if __name__=="__main__":main()
