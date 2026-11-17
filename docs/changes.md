# Experiment and implementation record

2026-11-17, phase one:

- Implemented original Eq.4/App.B mode separately from the stabilized erfc/small-sweep mode. Reconstruction uses the explicitly named stabilized mode.
- Initial comparison had static, analytic and10 endpoint samples. After confirming endpoint integration bias, added10 midpoint samples as a separate diagnostic, paired by the same initial seeds. The midpoint results are better; both comparisons are retained.
- CPU seed0 was an initial feasibility run, then repeated unchanged with the full3-seed CPU cohort. The finalized saved CPU results come from that complete run; no setting or numeric conclusion was selected based on a favorable seed. First-run timing is not used.
- Fixed raw timing-array field names from `forward_ms`/`backward_ms` to `forward_seconds`/`backward_seconds` to match the original recorded SI values. Median fields were already correctly converted to milliseconds. No timing values changed.
- Re-rendered every saved checkpoint on CPU and compared all held-out images and sharp PSNR. Paper-scale experiments and online components remain unimplemented.

2026-11-17, phase two:

- Added calibrated rectangular cameras, conservative linear-sweep support, depth-ordered Gaussian chunks, checkpointed tiles and batched midpoint rendering. The phase-one core and its original results remain available.
- Added officialTUM fr1/xyz and fr1/desk fixed15-frame subsets. Intrinsics followTUM's recommendedROS default for pre-registeredRGB-D; depth5000units/m and existing scale correction are explicit. Test frames do not enter geometry/color fusion.
- A5-step exploratory pilot exposed inadequate pixel-only sigma coverage. Before the formal cohort, added a training-geometry nearest-neighbor spacing term. Original cache/source metadata,5-step results and a60-step spacing trial are retained separately. No formal result was replaced with the favorable altered initializer.
- Retained20temporal-loop profiles with their implementation snapshot, then batched5time samples at once. Independent time-loop value/gradient checks pass;20fresh profiles provide the final initializer comparison. The change alters batching, not the midpoint quadrature rule.
- ChoseMPS for the3-seed primary cohort from isolated resource measurements before formal fitting. CPU adds4paired seed0controlled cases. Source hashes and the28-run/150-step plan were fixed before training; operational training concurrency is distinct from isolated latency measurements.
- Preserved narrow controlledROI and broader valid-depth PSNR together. Native sensor exposure remains assumed; originalRGB is only a pre-additional-blur reference. Geometry error can worsen despite RGB improvement, and those cases remain in the tables.
- Added12isolated final-map profiles,8colored Gaussian-centerPLY exports and28retained model-output archives. Original/processed inputRGB-D remains an ignored local cache.
- The first strictCPU pixel replay failed for desk/native/midpoint seed1 (max4.9075e−4 versus2e−4 gate). A2.384e−7m projected-depth difference swaps Gaussian653/751 under stable sorting. CPU withMPS order reduces blurRGB error to4.023e−6; CPU synthesis ofMPS projection differs by2.384e−7. MPS self-replay is exact. The original gate and fitted renderer are unchanged: theCPU exception and source-device verification are both recorded.
