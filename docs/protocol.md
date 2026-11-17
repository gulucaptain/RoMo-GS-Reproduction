# Phase-one protocol

All outputs are under `results/`; raw tests and unfavorable comparisons are retained.

## Integral and gradient verification

- Seeds 0,1,2, 240 cases each. Random SPD precision via LLᵀ, anisotropic scales, pixel offsets including tails, velocities spanning 10⁻⁷–10³ and exposures exp(U[−4,1]). Full input vectors are saved.
- Independent reference: SciPy QUADPACK adaptive scalar quadrature on normalized time [−0.5,0.5], split at the Mahalanobis closest-approach time. epsabs=1e−13, epsrel=1e−11, limit=300. This reference does not use erf.
- 24 gradient cases, 9 parameters: δ(2), log-Cholesky diagonal/shear(3), v′(2), opacity logit(1), log exposure(1). Central finite differences of the independent quadrature, h=1e−5. Includes exactly stationary and near-stationary inputs. Gradients are in unconstrained parameter coordinates, not arbitrary nonsymmetric precision entries.
- Acceptance thresholds are set in `validate_core.py`: float64 integral <1e−9; float32 integral <2e−5; gradient error <2e−7. These are local engineering checks, not author-specified tolerances.
- Near-static audit: 112 speeds (zero plus 10⁻⁹ through 10² px/s), two exposures, both modes, all values and velocity gradients saved. The hard paper branch and stable series behavior are separately visible.
- Single-footprint sampling: 1201 x locations, speed (12,2) px/s, Q=[[1,.2],[.2,.7]], o=.85, T=1s. N=5,10,20,50,100,500,2000; endpoint and midpoint errors are measured separately.
- Compositing: two red/blue Gaussians, opposite x velocities, identical width, black background, opacity .1/.9. Reference uses 8192 midpoint samples; static case is a control. This is a constructed stress test, not a claim about average real-image error.

## Reconstruction

- Scene seed 20261117, 24 Gaussians, anisotropic diagonal covariance, SH0 colors. Nine known camera centers, no rotation, camera speed (.8,.12,0) scene units/s, exposure .45 s. Resolution32×32 and focal36 px. Scene units are arbitrary.
- Train views [0,1,3,4,6,7]; held-out views [2,5,8]. Full original GT parameters, camera inputs and observations saved in `pilot_scene.npz`. Colors in [0,1], black background.
- Data generated on CPU with 64-node Gauss–Legendre quadrature over physical camera translations, reprojecting covariance and compositing at each instant. All nine images checked against128 nodes. This avoids training on the analytic model's own observations.
- Initialization uses known correspondence and near-GT positions; identical CPU random draws for all four methods and devices at each seed. Means += N(0,.05), log scales += N(0,.1), color logits += N(0,.2), opacity logits += N(0,.1). This is a controlled optimization pilot, not unknown-geometry SLAM.
- Adam lr=.015, 240 single-view steps, deterministic cycling of six training views, 0.8 mean L1 + 0.2(1−SSIM). SSIM: valid7×7 uniform RGB windows, C1=.01²,C2=.03². Camera, speed and Gaussian count frozen. Optimize means, diagonal scales, RGB and opacity.
- Four methods: static, analytic stable,10 endpoint samples,10 midpoint samples. Seeds0/1/2, CPU1 thread and MPS,24 total runs. The first9+9 runs used three methods; the6 midpoint runs are explicitly added numerical controls, saved separately. All use the same dataset and paired initialization.
- PSNR: −10log10 pooled RGB MSE over all three held-out sharp images, peak1. Report per-seed values, mean and sample SD (n=3), not uncertainty intervals. Position RMSE assumes the known Gaussian correspondence; it is not surface reconstruction accuracy.
- All training losses, final parameters, held-out images and timing retained. Checkpoint replay on CPU verifies stored images and recomputes sharp PSNR. Small CPU/MPS nonconvex optimization differences are retained, not replaced by favorable runs.

## Timing and Mac feasibility

MPS must be available; scripts fail rather than silently choosing CPU. `device_audit.py` compares float32 values and gradients on128 random footprints. Timings use one32×32 view/24 Gaussians, five warmups and20 repetitions, device synchronization around forward and backward, gradients with respect to means only. The sum of median forward and median backward is a descriptive statistic, not a per-frame system FPS or median end-to-end latency. Training timings include the optimizer, SSIM and step logging; they are separate from microbenchmarks. No global CPU fallback variable is set.

No throughput extrapolation to100k Gaussians is made. Dense memory scales with Gaussians×pixels, and MPS needs fused/tiled implementation before claiming practical online operation. Real RGB-D data, camera tracking, motion/occlusion coverage and online density control remain next-stage work.
