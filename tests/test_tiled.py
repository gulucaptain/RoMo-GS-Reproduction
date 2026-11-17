import pytest
import torch
from romo_gs.tiled import render_tiled,render_dense_camera


def scene(n=11):
    torch.manual_seed(101)
    means=torch.randn(n,3,dtype=torch.float64)*.2;means[:,2]+=2.
    L=torch.randn(n,3,3,dtype=torch.float64)*.03
    return dict(means=means.requires_grad_(),covariance=(L@L.transpose(-1,-2)+.01*torch.eye(3)).requires_grad_(),
                opacity=torch.full((n,),.6,dtype=torch.float64,requires_grad=True),
                colors=torch.rand(n,3,dtype=torch.float64,requires_grad=True),
                center=torch.zeros(3,dtype=torch.float64,requires_grad=True),
                rotation=torch.eye(3,dtype=torch.float64),intrinsics=torch.tensor([17.,18.,7.,5.],dtype=torch.float64),
                height=11,width=15,linear_velocity=torch.tensor([.4,.1,-.2],dtype=torch.float64,requires_grad=True),
                angular_velocity=torch.tensor([.2,-.3,.1],dtype=torch.float64,requires_grad=True),exposure=.2)


@pytest.mark.parametrize('method',['static','analytic','midpoint'])
@pytest.mark.parametrize('chunk',[2,16])
def test_exact_tiling_value_and_gradient(method,chunk):
    kwargs=scene()
    dense=render_dense_camera(**kwargs,method=method,samples=5)
    tiled=render_tiled(**kwargs,method=method,samples=5,tile_size=4,gaussian_chunk=chunk,q_cutoff=None)
    torch.testing.assert_close(dense,tiled,rtol=1e-11,atol=1e-12)
    params=[kwargs[k] for k in ['means','covariance','opacity','colors','center','linear_velocity','angular_velocity']]
    # Avoid depth division: compare all five composited channels directly.
    g1=torch.autograd.grad(dense.square().sum(),params)
    g2=torch.autograd.grad(tiled.square().sum(),params)
    for a,b in zip(g1,g2):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)


def test_culled_tiles_against_dense():
    kwargs=scene(31);kwargs['means'].data[:10,:2]*=20
    for method in ['analytic','midpoint']:
        a=render_dense_camera(**kwargs,method=method,samples=5)
        b=render_tiled(**kwargs,method=method,samples=5,tile_size=4,q_cutoff=24.)
        torch.testing.assert_close(a,b,rtol=0,atol=2e-5)


def test_behind_camera_and_empty_tiles():
    kwargs=scene();kwargs['means'].data[:,2]=-1
    result=render_tiled(**kwargs,tile_size=4)
    assert result.abs().max()==0
    result.sum().backward()
    assert torch.isfinite(kwargs['means'].grad).all()


def test_midpoint_against_original_independent_time_loop():
    from romo_gs.core import render
    k=scene(17);k['height']=k['width']=13
    k['intrinsics']=torch.tensor([18.,18.,6.,6.],dtype=torch.float64)
    old=dict(means=k['means'],covariance=k['covariance'],opacity=k['opacity'],colors=k['colors'],
             center=k['center'],rotation=k['rotation'],focal=18.,size=13,
             linear_velocity=k['linear_velocity'],angular_velocity=k['angular_velocity'],exposure=k['exposure'])
    reference=render(**old,method='sampled_midpoint',samples=10)
    image=render_tiled(**k,method='midpoint',samples=10,q_cutoff=None,tile_size=5,gaussian_chunk=4)[...,:3]
    torch.testing.assert_close(reference,image,rtol=1e-11,atol=1e-12)
    params=[k['means'],k['covariance'],k['colors'],k['opacity']]
    g1=torch.autograd.grad(reference.square().sum(),params)
    g2=torch.autograd.grad(image.square().sum(),params)
    for a,b in zip(g1,g2):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
