"""Independent adaptive-quadrature, finite-difference and compositing audits."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import torch
from romo_gs.core import integrated_opacity, sampled_opacity


def unpack(x):
    d, v = x[:2], x[5:7]
    L = np.array([[np.exp(x[2]), 0], [x[3], np.exp(x[4])]])
    return d, L@L.T, v, 1/(1+np.exp(-x[7])), np.exp(x[8])


def reference(x):
    d, Q, v, o, t = unpack(x)
    a, b = v@Q@v, v@Q@d
    peak = np.clip(-b/(a*t), -.5, .5) if a else 0
    f = lambda s: o*np.exp(-.5*((d+v*t*s)@Q@(d+v*t*s)))
    # Split at closest approach so narrow high-speed footprints cannot be missed.
    return quad(f, -.5, .5, points=[peak], epsabs=1e-13, epsrel=1e-11, limit=300)[0]


def tensor_value(x, mode="stable"):
    L = torch.stack([x[2].exp(), x[3]*0, x[3], x[4].exp()]).reshape(2, 2)
    return integrated_opacity(x[:2], L@L.T, x[5:7], x[7].sigmoid(), x[8].exp(), mode)


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(seed)
        for i in range(240):
            x = rng.normal(size=9)
            x[:2] *= 3 if i % 3 else 10
            x[2:5] *= .7
            x[5:7] *= 10**rng.uniform(-7, 3)
            x[8] = rng.uniform(-4, 1)
            ref = reference(x)
            row = {"seed": seed, "case": i, "parameters": x.tolist(), "quad": ref}
            for mode in ["paper", "stable"]:
                for dtype in [torch.float64, torch.float32]:
                    value = tensor_value(torch.tensor(x, dtype=dtype), mode).item()
                    row[f"{mode}_{str(dtype).split('.')[-1]}"] = value
            records.append(row)
    gradients = []
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(seed+100)
        for i in range(8):
            x = rng.normal(0, .4, 9)
            if i == 0:
                x[5:7] = 0
            if i == 1:
                x[5:7] *= 1e-5
            tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            grad = torch.autograd.grad(tensor_value(tx), tx)[0].detach().numpy()
            fd = []
            for j in range(9):
                e = np.zeros(9); e[j] = 1e-5
                fd.append((reference(x+e)-reference(x-e))/2e-5)
            gradients.append({"seed": seed, "case": i, "parameters": x.tolist(),
                              "autograd": grad.tolist(), "quadrature_finite_difference": fd,
                              "max_absolute_error": float(np.max(np.abs(grad-fd)))})
    # 1D single-footprint comparison, endpoint rule exactly matches paper Eq.2.
    d = torch.stack([torch.linspace(-12, 12, 1201), torch.zeros(1201)], -1).double()
    Q = torch.tensor([[1., .2], [.2, .7]], dtype=torch.float64)
    v = torch.tensor([12., 2.], dtype=torch.float64)
    truth = integrated_opacity(d, Q, v, .85, 1.)
    sampling = []
    for rule in ["endpoint", "midpoint"]:
        for n in [5, 10, 20, 50, 100, 500, 2000]:
            pred = sampled_opacity(d, Q, v, .85, 1., n, rule)
            sampling.append({"rule": rule, "samples": n,
                             "mae": (pred-truth).abs().mean().item(),
                             "max_absolute_error": (pred-truth).abs().max().item()})
    # Two occluding colored footprints: compare average-alpha composite against
    # time-averaged composite. Vectorized NumPy midpoint integration is independent.
    xx = np.linspace(-10, 10, 401)
    tau = (np.arange(8192)+.5)/8192-.5
    occlusion = []
    for name, speed, opacity in [("stationary", 0., .9), ("low_opacity", 10., .1), ("occluding", 10., .9)]:
        alpha = opacity*np.exp(-.5*(xx[None,None,:]+np.array([speed,-speed])[:,None,None]*tau[None,:,None])**2)
        # Red foreground, blue background, black canvas.
        reference_image = np.stack([alpha[0].mean(0), np.zeros_like(xx),
                                    ((1-alpha[0])*alpha[1]).mean(0)], -1)
        dd = torch.tensor(np.stack([xx, np.zeros_like(xx)], -1))
        integ = integrated_opacity(dd, torch.eye(2,dtype=torch.float64),
                                  torch.tensor([speed,0.],dtype=torch.float64), opacity, 1.).numpy()
        approximation = np.stack([integ, np.zeros_like(xx), (1-integ)*integ], -1)
        occlusion.append({"name": name, "speed": speed, "opacity": opacity,
                          "mae": float(abs(approximation-reference_image).mean()),
                          "max_absolute_error": float(abs(approximation-reference_image).max()),
                          "analytic": approximation.tolist(), "reference": reference_image.tolist()})
    summary = {"integral_cases": len(records), "gradient_cases": len(gradients),
               "gradient_max_absolute_error": max(g["max_absolute_error"] for g in gradients),
               "sampling": sampling,
               "occlusion": [{k:v for k,v in r.items() if k not in {"analytic","reference"}} for r in occlusion]}
    for mode in ["paper", "stable"]:
        for dtype in ["float32", "float64"]:
            key = f"{mode}_{dtype}"
            errors = np.array([abs(r[key]-r["quad"]) for r in records])
            summary[key] = {"max_absolute_error": float(errors.max()), "mean_absolute_error": float(errors.mean()),
                            "nonfinite": int(sum(not np.isfinite(r[key]) for r in records)),
                            "worst_case": int(errors.argmax())}
    summary["acceptance"] = {
        "stable_float64_integral": summary["stable_float64"]["max_absolute_error"] < 1e-9,
        "stable_float32_integral": summary["stable_float32"]["max_absolute_error"] < 2e-5,
        "gradients_vs_independent_quad_fd": summary["gradient_max_absolute_error"] < 2e-7,
        "endpoint_converges": sampling[6]["mae"] < sampling[0]["mae"],
        "occlusion_gap_detected": occlusion[2]["max_absolute_error"] > .01}
    payload = {"summary": summary, "integral_records": records, "gradients": gradients,
               "occlusion": occlusion, "opacity_profile": {"x": d[:,0].tolist(), "analytic": truth.tolist()}}
    (out/"core_raw.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    (out/"core_summary.json").write_text(json.dumps(summary, indent=2))
    env = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "mps_available_in_validation": torch.backends.mps.is_available(),
           "paper_sha256": hashlib.sha256(Path("paper/paper.html").read_bytes()).hexdigest() if Path("paper/paper.html").exists() else None,
           "source_url": "https://arxiv.org/html/2610.24605v1", "seeds": [0,1,2]}
    (out/"environment.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(summary, indent=2))
    assert all(summary["acceptance"].values()), summary["acceptance"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="results/core")
    run(Path(parser.parse_args().output))
