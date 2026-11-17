"""Replay every saved checkpoint on CPU; validate saved images and metrics."""
import json
from pathlib import Path
import numpy as np
import torch
from romo_gs.core import render
from pilot import load, args_render, psnr


def main():
    torch.set_num_threads(1)
    data=load(Path("results/pilot_scene.npz"),"cpu")
    rows=json.loads(Path("results/aggregate.json").read_text())["runs"]
    results=[]
    for row in rows:
        path=Path("results")/row["folder"]/f"{row['method']}_seed{row['seed']}.npz"
        saved=load(path,"cpu")
        sharp=[];blur=[]
        with torch.no_grad():
            for i in [2,5,8]:
                kwargs=args_render(data,i,saved["means"],saved["scales"],saved["colors"],saved["opacity"])
                sharp.append(render(**kwargs,method="static"))
                blur.append(render(**kwargs,method=row["method"],samples=10))
        sharp=torch.stack(sharp);blur=torch.stack(blur)
        error=max(float((sharp-saved["heldout_sharp"]).abs().max()),float((blur-saved["heldout_blur"]).abs().max()))
        metric=psnr(sharp,data["sharp"][[2,5,8]])
        assert error<2e-5,(path,error)
        assert abs(metric-row["sharp_heldout_psnr"])<.001,(path,metric)
        results.append({"checkpoint":str(path),"image_max_abs_error":error,"sharp_psnr_replayed":metric})
    Path("results/replay.json").write_text(json.dumps(results,indent=2))
    print(f"Replayed {len(results)} checkpoints; max image error {max(r['image_max_abs_error'] for r in results):.3e}")


if __name__=="__main__":main()
