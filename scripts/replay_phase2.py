"""Replay all final real-data parameters on CPU and check stored metrics."""
import json
from pathlib import Path
import numpy as np
import torch
from rgbd_experiment import load_scene,evaluate


def main():
    torch.set_num_threads(1);records=[]
    scenes={};gpu_scenes={}
    for folder in sorted(Path('results/phase2/runs').iterdir()):
        if not (folder/'summary.json').exists():continue
        summary=json.loads((folder/'summary.json').read_text());config=json.loads((folder/'config.json').read_text())
        seq=summary['sequence']
        if seq not in scenes:scenes[seq]=load_scene(seq)
        with np.load(folder/'parameters.npz') as p:
            values=[torch.tensor(p[k]) for k in ['means','scales','colors','opacity']]
        metric,images,blur=evaluate(scenes[seq],*values,config['exposure_seconds'],summary['method'],summary['track'],config['tile'],config['chunk'],config['samples'])
        fields=['reference_psnr','observation_psnr','center_vs_original_rgb_psnr','full_native_valid_center_psnr','reference_ssim_mean','depth_rmse_m']
        errors={k:abs(metric[k]-summary['final'][k]) for k in fields}
        assert max(errors[k] for k in fields[:4])<.002,(folder,errors)
        assert errors['depth_rmse_m']<1e-4,(folder,errors)
        assert errors['reference_ssim_mean']<1e-4,(folder,errors)
        path=Path('results/phase2/predictions')/(folder.name+'_predictions.npz')
        if not path.exists():path=Path('results/phase2/local_data')/(folder.name+'_predictions.npz')
        source_device_replay=None
        if path.exists():
            with np.load(path) as previous:
                maximum=max(float(abs(images-previous['heldout']).max()),float(abs(blur-previous['blur']).max()))
            if maximum>=2e-4 and summary['device']=='mps':
                # Keep the strict cross-device gate visible. Near-depth order
                # swaps can violate it despite metric agreement. Verify the
                # actual training-device render instead of widening the gate.
                if torch.backends.mps.is_available():
                    if seq not in gpu_scenes:gpu_scenes[seq]=load_scene(seq,'mps')
                    gm,gi,gb=evaluate(gpu_scenes[seq],*(v.to('mps') for v in values),config['exposure_seconds'],summary['method'],summary['track'],config['tile'],config['chunk'],config['samples'])
                    with np.load(path) as previous:
                        same_device=max(float(abs(gi-previous['heldout']).max()),float(abs(gb-previous['blur']).max()))
                    assert same_device<2e-5,(folder,same_device)
                    source_device_replay={'device':'mps','max_image_error':same_device,'verified':True}
                else:source_device_replay={'device':'mps','verified':False,'reason':'MPS unavailable; CPU metrics verified, strict cross-device image gate exceeded'}
            elif maximum>=2e-4:raise AssertionError((folder,maximum))
        else:
            maximum=None
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,heldout=images,blur=blur)
        row={'run':folder.name,'metric_absolute_errors':errors,'max_prediction_absolute_error':maximum,
             'strict_cpu_image_gate_passed':None if maximum is None else maximum<2e-4,
             'source_device_replay':source_device_replay,
             'prediction_cache_recreated':maximum is None,'replayed':metric}
        records.append(row);print(folder.name,errors,flush=True)
    assert len(records)==28,f'Expected28 final runs, found{len(records)}'
    Path('results/phase2/replay.json').write_text(json.dumps(records,indent=2))
    print(f'Replayed {len(records)} real-data checkpoints',flush=True)


if __name__=='__main__':main()
