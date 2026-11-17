# Rebuild phase two

Run from the `romo_gs` project directory with the environment in the rootREADME. No new dependencies, API credentials, pretrained networks or author code are required.

## Inspect the saved result

The full local working directory contains originalTUM archives under `data/downloads/`, prepared arrays under `data/prepared/`,28final parameter archives under `results/phase2/runs/`, source hashes,150step losses,20isolated profiles and local qualitative images. The deliveryZIP omits original/processed RGB-D observation caches and the virtual environment. It includes all code, configurations, parameter archives,28model-rendered output archives, logs, scalar results and final figures.

Use the result report for scope and numbers. Merely running the scripts does not establish reproduction of the paper's full online system.

## Re-download data and replay existing parameters

```sh
python scripts/download_rgbd.py
python scripts/prepare_rgbd.py
python scripts/replay_phase2.py
python scripts/investigate_replay.py
python scripts/report_phase2.py
```

The downloader verifies the exact reviewed archiveSHA-256. The prepare script reads selected tar members directly and does not extract arbitrary paths. Approximately755MiB of compressed archives are cached. Preprocessing reconstructs the same15selected frames per sequence, poses, masks, generated blur and training-only fused initialization.

`replay_phase2.py` re-renders saved final parameters on CPU, compares stored metrics, and reconstructs missing local prediction caches. The retained model outputs under `results/phase2/predictions/` permit pixel checks without distributing original input images. Near-depth ordering can violate the strict CPU/GPU pixel gate: those cases remain marked as exceptions and require an actualMPS self-replay for full pixel verification. ACPU-only process reports the unverified exception explicitly; the final audit will require source-device verification. This does not change the original pixel threshold or fitted model.

## Re-run every experiment

```sh
python -m pytest -q --junitxml=results/phase2/tests.xml
python scripts/audit_rgbd.py
python scripts/run_phase2.py profiles
python scripts/run_phase2.py train
python scripts/collect_predictions.py
python scripts/run_final_profiles.py
python scripts/replay_phase2.py
python scripts/investigate_replay.py
python scripts/report_phase2.py
python scripts/export_maps.py
python scripts/audit_final.py
python scripts/manifest.py
python scripts/manifest.py --check
```

Run profiles on an otherwise idle machine **before** launching the training queue. Each of the20benchmark cases uses a fresh process,1warmup and3timed repetitions. MPS commands must have actualMac GPU access; there is no CPU fallback. All methods differentiate the same parameter classes. The temporal-loop baseline under `results/phase2/exploration/temporal_loop/` is historical evidence, not a second formal cohort.

The training queue uses twoCPU and twoMPS workers for28runs,150steps each. CPU runs are seed0 controlled cases; MPS is the three-seed primary cohort for both observation tracks. Training operational times include shared-resource contention and are not used as the isolated speed claim. Expect tens of minutes on the recordedM4Pro; actual time varies. Each job saves its own log/configuration/summary/parameters.

After the training queue stops, `run_final_profiles.py` benchmarks all12seed0final models in separate sequential processes. It records15no-grad inference timings across5held-out views and3forward/backward timings at the first training view, with independent memory records. This distinguishes final trained-map cost from the20initializer backend cases.

Completed scheduled jobs are reused. For a full retraining without discarding the original evidence, move `results/phase2/runs/` and `results/phase2/training_logs/` to a named archive directory before invoking the queue. The scripts do not select the best held-out checkpoint. Alternative manual runs should use a separate `--output` directory.

## One manual experiment

```sh
python scripts/rgbd_experiment.py --sequence freiburg1_xyz --track controlled \
  --method analytic --device mps --seed 0 --steps 150 \
  --output results/phase2/manual
```

Use `--method midpoint` for10-point midpoint compositing. The native track assumes1/60s exposure, not measured camera metadata; the controlled track adds0.08s blur on realRGB-D. `--tile` and `--chunk` choose spatial and Gaussian batching. The primary shared settings are32and256. See `docs/phase2_protocol.md` for all calibration, initialization, masking, metric and memory definitions.

## Evidence and hash changes

Original observation arrays stay local and ignored. Their source and prepared-data digests are recorded, not distributed. `results/manifest.json` checks the current source and committed result files; the originalphase-one manifest is recoverable from local commit423a8cb and the phase-oneZIP. Re-running scripts changes timing measurements, environment XML timestamps and some figures; regenerate the manifest after intentionally rebuilding results instead of claiming that changed outputs match the original digest.
