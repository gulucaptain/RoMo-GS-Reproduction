# Method and implementation decisions

Source: Jung et al., *RoMo-GS: Efficient Online 3D Gaussian Splatting for Robot Mapping via Exposure-Integrated Motion Blur*, arXiv:2610.24605v1, §3.1, App.A–B. This is an independent implementation, not author code.

For pixel displacement δ = μ′ − x, screen precision Q = Σ′⁻¹, projected velocity v′ and exposure T > 0:

\[
\bar\alpha=\frac{o}{T}\int_{-T/2}^{T/2}\exp[-(\delta+v'\tau)^T Q(\delta+v'\tau)/2]d\tau.
\]

Let a = v′ᵀQv′, b = v′ᵀQδ, c = δᵀQδ. Completing the square gives paper Eq.4:

\[
\bar\alpha=\frac{o}{T}\sqrt{\frac{\pi}{2a}}e^{-(c-b^2/a)/2}
\left[\operatorname{erf}\left(\frac{b}{\sqrt{2a}}+\frac{T\sqrt{a/2}}2\right)-
\operatorname{erf}\left(\frac{b}{\sqrt{2a}}-\frac{T\sqrt{a/2}}2\right)\right].
\]

`mode="paper"` uses the direct erf difference and App.B static branch at a ≤ 1e−6. It explicitly has zero velocity gradient on the static branch. The paper's dimensional label of that tolerance is retained only as a numeric convention; internally a has inverse-time-squared units when velocity is px/s and Q is px⁻². We do not reinterpret its threshold as a displacement threshold.

`mode="stable"` is a documented numerical extension for CPU/MPS float32. Same-sign erf tails use erfc(|u_min|)−erfc(|u_max|). For aT² < 1e−4 and |b|T < 0.1 use the small-sweep fourth-order expansion:

\[
\bar\alpha=o e^{-c/2}\left[1+\frac{(b^2-a)T^2}{24}
+\frac{(b^4-6ab^2+3a^2)T^4}{1920}\right].
\]

This follows by expanding exp(−bτ−aτ²/2) and integrating even powers. It is not an exact formula for all nonzero velocities; its approximation error is independently checked. Both branches stay finite even when inactive, preventing invalid reverse-mode products. The perpendicular squared distance is clamped at zero against float32 roundoff. The opacity integral is not clamped internally, so validation measures the integral itself; compositing clamps opacity to [0,0.999].

The 3D projector takes full SPD world covariance, a world-to-camera rotation and a camera center. It computes Σ′ = J RΣRᵀ Jᵀ + 0.3 I and v′ = −J(ω × p_camera + v). The sign of the velocity does not affect the symmetric scalar exposure integral, so a separate signed finite-difference projection test is necessary and provided. All Gaussians are positive-depth in the synthetic pilot; general clipping, behind-camera rejection and near-plane behavior are not yet implemented.

The dense renderer sorts once by center depth and performs standard front-to-back alpha compositing. It does not reproduce AccuTile, swept-hull assignment, CUDA kernels or depth-order changes during exposure. Colors are SH0 (constant). The projector supports full covariance, while the pilot trains diagonal world covariance only.

`method="sampled"` renders N equally weighted endpoint samples in [−T/2,T/2], as paper Eq.2. `method="sampled_midpoint"` uses N midpoint samples, an independent numerical control. Each sample composites first. `method="analytic"` integrates individual opacity first and then composites. These operations do not commute when multiple partially transparent layers overlap. The analytic model remains approximate even with perfect arithmetic. The experiment separately measures this gap, rather than calling it an integration bug.

Camera state/velocity is differentiable in the projector, but no learned camera-state update, online SLAM, LM, pruning or densification is included in this phase. No claim is made about reproducing §3.2–3.3 or paper tables.
