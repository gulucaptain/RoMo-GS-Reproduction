# Paper coverage

| Paper component | Implemented | Executed verification | Remaining |
|---|---|---|---|
| Eq.3–4 exposure integral | Yes, original + stabilized variant |720 cases vs adaptive quadrature, float32/64 |Extreme tails/scales beyond recorded range |
| App.B stationary treatment |Original hard branch + small-sweep extension |224 speed/exposure configurations; gradients |Author kernel identity unavailable |
| Projected pixel velocity |Full rotation/translation formula |Signed SE(3) finite differences |Full optimization with real tracking errors |
| Single-pass alpha compositing |Dense and checkpointed tiled PyTorch |CPU/MPS, controlled occlusion stress, real-data value/gradient audit |Fused GPU/AccuTile implementation |
| Eq.2 multi-sample baseline |Endpoint rule and separate midpoint control |5–2000 single-footprint samples;10-sample reconstruction |Original ETH3D ablation |
| Gaussian optimization |Adam on position, diagonal scale, RGB, opacity |24 synthetic runs;28 TUM RGB-D pilot fits at2400Gaussians/96×72 |SH, orientation training, unknown count, convergence |
| Tiled rendering |Conservative swept bounds, Gaussian streaming,5-time-sample batches, tile recomputation |16tests;20isolated initializer profiles;12final-map profiles |Not the paper's AccuTile or a fused Metal/CUDA kernel |
| Pose/velocity joint update |Differentiable projection only |Projection derivative unit test |Online trained camera-state corrections |
| LM |No |No |Entire component |
| Hessian-based pruning |No |No |Per-pixel squared sensitivities before reduction; must not substitute squared summed gradient |
| RGB-D initialization |Independent training-depth voxel fusion, observation-only colors |Two15-frame TUM subsets; no test/clean-color leakage |Not the paper's additional-point sampling rule |
| Densification |No |No |Adaptive online Gaussian growth |
| MODP/ORB-SLAM3 front end |No |No |Integration, loop closure, tracking failures |
| Deblur-NeRF/ETH3D/TUM/custom data tables |No original table reproduced |TUM small known-pose RGB-D pilot is separate evidence |All original table results unmatched |

Current conclusion: exposure integration and gradients are validated; the tiled backend reduces retained memory and supports real RGB-D small-scale fitting on CPU/MPS. Controlled-blur midpoint quality is slightly higher on the selected ROIs, while broader-area coverage remains weak. Native exposure is assumed, camera poses are supplied, and all paper tables remain unmatched. The analytic compositing model has a measured approximation error. Full RoMo-GS reproduction remains incomplete.
