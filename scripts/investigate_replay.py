"""Diagnose a recorded CPU/MPS replay discrepancy without changing rendering."""
import json
from pathlib import Path
import numpy as np
import torch
from romo_gs.tiled import project_camera,build_plan,tile_pixels,footprint_block
from rgbd_experiment import load_scene,kwargs,evaluate


def raster(projected,base,order_z=None,method='midpoint'):
    uv,Q,v,z,screen=projected
    h,w=72,96
    plan=build_plan(uv,screen,v,z if order_z is None else order_z,h,w,32,0. if method=='static' else base['exposure'])
    rows=[];current=[];oldy=0
    for y,y1,x,x1,ids in plan:
        if y!=oldy:rows.append(torch.cat(current,1));current=[];oldy=y
        idx=torch.tensor(ids.copy(),device=uv.device)
        if not len(ids):image=torch.zeros(y1-y,x1-x,5,device=uv.device)
        else:
            image=footprint_block(uv[idx],Q[idx],v[idx],base['opacity'][idx],base['colors'][idx],z[idx],
                                  tile_pixels(y,y1,x,x1,uv),base['exposure'],method,10,256).reshape(y1-y,x1-x,5)
        current.append(image)
    rows.append(torch.cat(current,1));return torch.cat(rows,0)


def investigate(name):
    torch.set_num_threads(1)
    folder=Path('results/phase2/runs')/name
    cfg=json.loads((folder/'config.json').read_text())
    cpu=load_scene(cfg['sequence'],'cpu');gpu=load_scene(cfg['sequence'],'mps')
    with np.load(folder/'parameters.npz') as p:values=[torch.tensor(p[k]) for k in ['means','scales','colors','opacity']]
    path=Path('results/phase2/predictions')/(name+'_predictions.npz')
    if not path.exists():path=Path('results/phase2/local_data')/(name+'_predictions.npz')
    with np.load(path) as z:
        stored={k:z[k] for k in z.files}
    result={'run':name,'views':[]}
    with torch.no_grad():
        for j,i in enumerate(cfg['test']):
            c=kwargs(cpu,i,*values,cfg['exposure_seconds']);g=kwargs(gpu,i,*(v.to('mps') for v in values),cfg['exposure_seconds'])
            projected=[]
            for q in [c,g]:projected.append(project_camera(q['means'],q['covariance'],q['center'],q['rotation'],q['intrinsics'],q['linear_velocity'],q['angular_velocity']))
            pc,pg=projected;pg_cpu=tuple(x.cpu() for x in pg)
            cpu_image=raster(pc,c,method=cfg['method']);gpu_image=raster(pg,g,method=cfg['method']).cpu()
            same_projected_cpu=raster(pg_cpu,c,method=cfg['method'])
            gpu_order_cpu=raster(pc,c,pg_cpu[3],method=cfg['method'])
            static_error=abs(cpu_image[...,:3].numpy()-stored['blur'][j])
            # Match the ACTUAL NumPy stable sort in build_plan, including ties.
            order_cpu=np.argsort(pc[3].numpy(),kind='stable');order_gpu=np.argsort(pg_cpu[3].numpy(),kind='stable')
            changed=np.flatnonzero(order_cpu!=order_gpu)
            static_cpu=raster(pc,c,method='static');static_gpu=raster(pg,g,method='static').cpu()
            static_fixed_order=raster(pc,c,pg_cpu[3],method='static')
            row={'view':i,'cpu_vs_stored_blur_max':float(static_error.max()),
                 'gpu_replay_vs_stored_blur_max':float(abs(gpu_image[...,:3].numpy()-stored['blur'][j]).max()),
                 'cpu_raster_of_gpu_projection_vs_gpu_rgb_max':float((same_projected_cpu[...,:3]-gpu_image[...,:3]).abs().max()),
                 'cpu_projection_with_gpu_depth_order_vs_gpu_rgb_max':float((gpu_order_cpu[...,:3]-gpu_image[...,:3]).abs().max()),
                 'depth_order_positions_different':len(changed),
                 'changed_order_examples':[{'position':int(k),'cpu_id':int(order_cpu[k]),'gpu_id':int(order_gpu[k]),
                     'cpu_depths':[float(pc[3][order_cpu[k]]),float(pc[3][order_gpu[k]])],
                     'gpu_depths':[float(pg_cpu[3][order_cpu[k]]),float(pg_cpu[3][order_gpu[k]])]} for k in changed[:4]],
                 'static_cpu_vs_stored_per_channel_max':[float(abs(static_cpu[...,k].numpy()-stored['heldout'][j,...,k]).max()) for k in range(5)],
                 'static_gpu_replay_vs_stored_max':float(abs(static_gpu.numpy()-stored['heldout'][j]).max()),
                 'static_cpu_gpu_order_vs_gpu_max':float((static_fixed_order-static_gpu).abs().max()),
                 'projected_max_abs':{k:float((a-b).abs().max()) for k,a,b in zip(['uv','precision','velocity','depth','screen_covariance'],pc,pg_cpu)},
                 'cpu_vs_gpu_per_channel_max':[float((cpu_image[...,k]-gpu_image[...,k]).abs().max()) for k in range(5)]}
            result['views'].append(row)
    return result


def main():
    replay_path=Path('results/phase2/replay.json')
    names=['freiburg1_desk_native_midpoint_mps_seed1']
    if replay_path.exists():
        names=[r['run'] for r in json.loads(replay_path.read_text()) if r.get('strict_cpu_image_gate_passed') is False]
    if not names:
        result={'run':None,'views':[],'all_cases':[],'exception_cases':0,
                'all_source_device_replays_exact':True,'max_cpu_rgb_error_with_gpu_depth_order':0.,
                'max_cpu_rgb_error_with_identical_gpu_projection':0.}
        Path('results/phase2/replay_discrepancy_audit.json').write_text(json.dumps(result,indent=2))
        print('No CPU pixel exceptions require diagnosis');return
    cases=[investigate(name) for name in names]
    result=dict(cases[0],all_cases=cases,exception_cases=len(cases))
    result['all_source_device_replays_exact']=all(v['gpu_replay_vs_stored_blur_max']==0 and v['static_gpu_replay_vs_stored_max']==0 for c in cases for v in c['views'])
    result['max_cpu_rgb_error_with_gpu_depth_order']=max(v['cpu_projection_with_gpu_depth_order_vs_gpu_rgb_max'] for c in cases for v in c['views'])
    result['max_cpu_rgb_error_with_identical_gpu_projection']=max(v['cpu_raster_of_gpu_projection_vs_gpu_rgb_max'] for c in cases for v in c['views'])
    assert result['all_source_device_replays_exact']
    assert result['max_cpu_rgb_error_with_gpu_depth_order']<2e-4
    Path('results/phase2/replay_discrepancy_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
