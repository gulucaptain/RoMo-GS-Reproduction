import torch
from romo_gs.core import integrated_opacity, composite, project


def test_stationary_integral_and_gradients():
    d = torch.tensor([.4,-.8],dtype=torch.float64,requires_grad=True)
    v = torch.zeros(2,dtype=torch.float64,requires_grad=True)
    Q = torch.eye(2,dtype=torch.float64)
    got = integrated_opacity(d,Q,v,.8,.3)
    want = .8*torch.exp(-d.square().sum()/2)
    torch.testing.assert_close(got,want)
    gd,gv = torch.autograd.grad(got,(d,v))
    torch.testing.assert_close(gd,-d*want)
    torch.testing.assert_close(gv,torch.zeros_like(v))


def test_gradcheck_nonstationary():
    d=torch.tensor([.7,-.3],dtype=torch.float64,requires_grad=True)
    Q=torch.tensor([[1.2,.2],[.2,.8]],dtype=torch.float64,requires_grad=True)
    v=torch.tensor([2.,-.6],dtype=torch.float64,requires_grad=True)
    o=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    t=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(integrated_opacity,(d,Q,v,o,t))


def test_exposure_velocity_scaling_and_reversal():
    torch.manual_seed(5)
    d=torch.randn(50,2,dtype=torch.float64)
    Q=torch.tensor([[2.,.3],[.3,1.]],dtype=torch.float64)
    v=torch.tensor([4.,-2.],dtype=torch.float64)
    a=integrated_opacity(d,Q,v,.9,.1)
    torch.testing.assert_close(a,integrated_opacity(d,Q,-v,.9,.1))
    torch.testing.assert_close(a,integrated_opacity(d,Q,v*10,.9,.01))


def test_projection_velocity_against_finite_difference():
    means=torch.tensor([[.2,.3,3.],[-.3,.1,4.]],dtype=torch.float64)
    cov=torch.eye(3,dtype=torch.float64).repeat(2,1,1)*.01
    center=torch.zeros(3,dtype=torch.float64); R=torch.eye(3,dtype=torch.float64)
    v=torch.tensor([.4,-.1,.2],dtype=torch.float64); w=torch.tensor([.2,-.3,.1],dtype=torch.float64)
    uv,_,speed,_=project(means,cov,center,R,30,32,v,w)
    skew=torch.tensor([[0.,-.1,-.3],[.1,0.,-.2],[.3,.2,0.]],dtype=torch.float64)
    dt=1e-6
    def at(t):
        points=(means-v*t)@torch.matrix_exp(-skew*t).T
        return 30*points[:,:2]/points[:,2:]+15.5
    torch.testing.assert_close(speed,(at(dt)-at(-dt))/(2*dt),atol=1e-7,rtol=1e-7)


def test_composite_order():
    alpha=torch.tensor([[.5],[.5]])
    color=torch.tensor([[1.,0.,0.],[0.,0.,1.]])
    torch.testing.assert_close(composite(alpha,color),torch.tensor([[.5,0.,.25]]))
