"""Real-data preprocessing, tiled value/gradient and label-integration audits."""
import hashlib
import io
import json
import tarfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from romo_gs.tiled import render_tiled,render_dense_camera,project_camera
from rgbd_experiment import load_scene,initial_colors,kwargs
from prepare_rgbd import reproject,resize_float


def main():
    torch.set_num_threads(1);result={'renderer':[],'labels':[],'sources':[]}
    for seq in ['freiburg1_xyz','freiburg1_desk']:
        d=load_scene(seq)
        n=384
        means=d['init_means'][:n].double().requires_grad_()
        scales=d['init_scale'][:n,None].expand(-1,3).double()
        cov=torch.diag_embed(scales.square()).requires_grad_()
        color=initial_colors(d,'controlled')[:n].double().requires_grad_()
        opacity=torch.full((n,),.8,dtype=torch.float64,requires_grad=True)
        kw=kwargs(d,0,means,scales,color,opacity,.08)
        for key in ['center','rotation','intrinsics','linear_velocity','angular_velocity']:kw[key]=kw[key].double()
        kw['covariance']=cov
        for method in ['analytic','midpoint']:
            reference=render_dense_camera(**kw,method=method)
            variants={}
            for name,cutoff in [('exact',None),('culled',24.)]:
                pred=render_tiled(**kw,method=method,q_cutoff=cutoff)
                parameters=[means,cov,color,opacity]
                a=torch.autograd.grad(reference.square().mean(),parameters,retain_graph=True)
                b=torch.autograd.grad(pred.square().mean(),parameters)
                variants[name]={'image_max_absolute_error':float((pred-reference).abs().max().detach()),
                                'gradient_max_absolute_error':max(float((x-y).abs().max()) for x,y in zip(a,b)),
                                'gradient_relative_l2':max(float((x-y).norm()/x.norm().clamp_min(1e-14)) for x,y in zip(a,b))}
            result['renderer'].append({'sequence':seq,'method':method,'gaussians':n,'variants':variants})
            assert variants['exact']['image_max_absolute_error']<1e-10
            assert variants['exact']['gradient_max_absolute_error']<1e-9
            assert variants['culled']['image_max_absolute_error']<2e-4
            assert variants['culled']['gradient_relative_l2']<2e-3
        source=json.loads(Path(f'results/phase2/{seq}_source.json').read_text())
        prepared=Path(f'data/prepared/{seq}.npz')
        assert hashlib.sha256(prepared.read_bytes()).hexdigest()==source['prepared_sha256']
        assert set(source['train']).isdisjoint(source['test'])
        # Training initializer provenance is explicit and mechanically checked.
        assert set(d['fusion_images'].tolist())<=set(source['train'])
        result['sources'].append({'sequence':seq,'prepared_hash_verified':True,'train_test_disjoint':True,
                                  'fusion_contains_no_test_frames':True,'native_valid_fraction_range':[min(x['native_valid_fraction'] for x in source['frames']),max(x['native_valid_fraction'] for x in source['frames'])],
                                  'controlled_valid_fraction_range':[min(x['controlled_valid_fraction'] for x in source['frames']),max(x['controlled_valid_fraction'] for x in source['frames'])]})
        prefix='rgbd_dataset_'+seq+'/'
        with tarfile.open(f'data/downloads/{seq}.tgz','r:gz') as tf:
            for i in [0,2]:
                frame=source['frames'][i]
                rgb=Image.open(io.BytesIO(tf.extractfile(prefix+frame['rgb']).read())).convert('RGB')
                depth=Image.open(io.BytesIO(tf.extractfile(prefix+frame['depth']).read()))
                rgb=np.array(rgb.resize((320,240),Image.Resampling.BOX),dtype=float)/255
                depth=np.array(depth.resize((320,240),Image.Resampling.NEAREST),dtype=float)/5000
                K=np.array([262.5,262.5,159.5,119.5])
                identity,valid=reproject(rgb,depth,K,np.zeros(3),np.eye(3))
                identity_error=float(abs(identity[valid]-rgb[valid]).max())
                assert identity_error<1e-10
                images={}
                v=np.array(frame['local_linear_velocity']);w=np.array(frame['local_angular_velocity'])
                for samples in [16,32,64]:
                    image=np.zeros_like(rgb)
                    for j in range(samples):
                        tau=((j+.5)/samples-.5)*.08
                        warp,_=reproject(rgb,depth,K,v*tau,Rotation.from_rotvec(w*tau).as_matrix())
                        image+=warp/samples
                    images[samples]=resize_float(image,(96,72))
                mask=d['controlled_mask'][i].numpy()
                cached=d['blur'][i].numpy();error=abs(cached-images[32])
                assert error.max()<2e-6
                row={'sequence':seq,'view':i,'identity_max_error':identity_error,'cached32_replay_max_error':float(error.max())}
                for samples in [16,32]:
                    diff=images[samples][mask]-images[64][mask]
                    row[f'{samples}_vs64_mae']=float(abs(diff).mean())
                    row[f'{samples}_vs64_psnr']=float(-10*np.log10((diff**2).mean()))
                    row[f'{samples}_vs64_max_abs']=float(abs(diff).max())
                result['labels'].append(row)
    path=Path('results/phase2/data_renderer_audit.json');path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
