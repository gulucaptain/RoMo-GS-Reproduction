"""Paper Eq. (3)-(4), reference integration and differentiable dense 3DGS.

Pixel units; exposure in seconds; projected velocity in pixels/second.
No visibility/tile approximation, early termination or custom CUDA kernels.
"""
import math
import torch


def coefficients(delta, precision, velocity):
    a = torch.einsum("...i,...ij,...j->...", velocity, precision, velocity)
    b = torch.einsum("...i,...ij,...j->...", velocity, precision, delta)
    c = torch.einsum("...i,...ij,...j->...", delta, precision, delta)
    return a, b, c


def integrated_opacity(delta, precision, velocity, opacity, exposure, mode="stable"):
    """Average individual opacity; this does NOT integrate alpha compositing.

    'paper': Eq.4 direct erf difference + App.B static branch a<=1e-6.
    'stable': same integral with erfc for same-sign tails and fourth-order
    small-sweep expansion (an explicitly documented numerical extension).
    Every unselected branch is evaluated at safe values to avoid NaN VJPs.
    Inputs must have SPD precision and positive exposure.
    """
    if mode not in {"paper", "stable"}:
        raise ValueError(mode)
    a, b, c = coefficients(delta, precision, velocity)
    t = torch.as_tensor(exposure, dtype=delta.dtype, device=delta.device)
    if mode == "paper":
        small = a <= 1e-6
    else:
        small = (a * t.square() < 1e-4) & (b.abs() * t < 0.1)
    aa = torch.where(small, torch.ones_like(a), a)
    bb = torch.where(small, torch.zeros_like(b), b)
    u0 = bb / (2 * aa).sqrt()
    h = t * (aa / 2).sqrt() / 2
    lo, hi = u0 - h, u0 + h
    if mode == "paper":
        mass = torch.erf(hi) - torch.erf(lo)
    else:
        # Evaluate the tail branch at positive arguments, including inactive entries.
        left, right = torch.minimum(lo.abs(), hi.abs()), torch.maximum(lo.abs(), hi.abs())
        tail = torch.erfc(left) - torch.erfc(right)
        mass = torch.where(lo * hi > 0, tail, torch.erf(hi) - torch.erf(lo))
    perpendicular = (c - bb.square() / aa).clamp_min(0)
    closed = (math.pi / (2 * aa)).sqrt() / t * (-perpendicular / 2).exp() * mass
    static = (-c / 2).exp()
    if mode == "stable":
        series = 1 + (b.square() - a) * t.square() / 24
        series = series + (b**4 - 6*a*b.square() + 3*a.square()) * t**4 / 1920
        static = static * series
    return opacity * torch.where(small, static, closed)


def sampled_opacity(delta, precision, velocity, opacity, exposure, samples=10, rule="endpoint"):
    if rule == "endpoint":
        tau = torch.linspace(-0.5, 0.5, samples, dtype=delta.dtype, device=delta.device)
    elif rule == "midpoint":
        tau = (torch.arange(samples, dtype=delta.dtype, device=delta.device)+0.5)/samples-0.5
    else:
        raise ValueError(rule)
    d = delta.unsqueeze(0) + tau.reshape((-1,) + (1,)*(delta.ndim-1) + (1,)) * exposure * velocity
    q = torch.einsum("n...i,...ij,n...j->n...", d, precision, d)
    return opacity * (-q/2).exp().mean(0)


def composite(alpha, colors, background=0.0):
    """N x P opacity and N x 3 color, already front-to-back depth sorted."""
    alpha = alpha.clamp(0, 0.999)
    factors = 1-alpha
    trans = torch.cumprod(torch.cat([torch.ones_like(factors[:1]), factors], 0), 0)
    return (trans[:-1]*alpha).T @ colors + trans[-1, :, None]*background


def inverse2(cov):
    a, b, d = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    det = a*d-b*b
    return torch.stack([d, -b, -b, a], -1).reshape(cov.shape)/det[..., None, None]


def project(means, covariance, camera_center, rotation, focal, size, linear_velocity, angular_velocity):
    """rotation is world-to-camera; velocities are in camera coordinates.

    World covariance is full SPD, not restricted to isotropic Gaussians.
    Eq.3.1 pixel velocity = -J(omega cross camera_point + v).
    """
    xyz = (means-camera_center) @ rotation.T
    x, y, z = xyz.unbind(-1)
    uv = focal * xyz[:, :2]/z[:, None] + (size-1)/2
    zero = torch.zeros_like(z)
    J = torch.stack([focal/z, zero, -focal*x/z.square(), zero, focal/z, -focal*y/z.square()], -1).reshape(-1, 2, 3)
    cc = rotation @ covariance @ rotation.T
    cov = J @ cc @ J.transpose(-1, -2)
    # Standard 0.3 pixel^2 screen-space low-pass variance, disclosed in protocol.
    cov = cov + 0.3*torch.eye(2, dtype=means.dtype, device=means.device)
    speed3 = torch.linalg.cross(angular_velocity.expand_as(xyz), xyz) + linear_velocity
    velocity = -(J @ speed3[..., None]).squeeze(-1)
    return uv, inverse2(cov), velocity, z


def render(means, covariance, opacity, colors, center, rotation, focal, size,
           linear_velocity, angular_velocity, exposure, method="analytic", samples=10,
           mode="stable"):
    uv, precision, velocity, z = project(means, covariance, center, rotation, focal, size,
                                         linear_velocity, angular_velocity)
    grid = torch.arange(size, dtype=means.dtype, device=means.device)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    pixels = torch.stack([xx.flatten(), yy.flatten()], -1)
    delta = uv[:, None]-pixels
    Q, v, o = precision[:, None], velocity[:, None], opacity[:, None]
    order = z.argsort()
    if method == "analytic":
        alpha = integrated_opacity(delta, Q, v, o, exposure, mode)
        image = composite(alpha[order], colors[order])
    elif method == "static":
        _, _, c = coefficients(delta, Q, v)
        image = composite((o*(-c/2).exp())[order], colors[order])
    elif method in {"sampled", "sampled_midpoint"}:
        # Paper Eq.2: equal-weight endpoint samples; composite BEFORE averaging.
        images = []
        if method == "sampled":
            times = torch.linspace(-0.5, 0.5, samples, device=means.device, dtype=means.dtype)
        else:
            times = (torch.arange(samples,device=means.device,dtype=means.dtype)+.5)/samples-.5
        for tau in times:
            d = delta + tau*exposure*v
            c = torch.einsum("npi,npij,npj->np", d, Q, d)
            images.append(composite((o*(-c/2).exp())[order], colors[order]))
        image = torch.stack(images).mean(0)
    else:
        raise ValueError(method)
    return image.reshape(size, size, 3)
