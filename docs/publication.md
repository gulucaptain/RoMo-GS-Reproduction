# Public release scope

Reviewed: 2026-11-17. Repository: https://github.com/gulucaptain/RoMo-GS-Reproduction.

This is an independent partial reproduction, not the authors' implementation. Publication does not change any experimental claim or resolve the limitations in README and COVERAGE. The public snapshot originates from local research commit `7b2c83cf932966397ff325ecd0b89a11ff6d66f4`. The original research checkout, all experiments and omitted files remain preserved locally.

All independently written code, configurations, scalar logs, reports and aggregate plots from the tracked snapshot are retained. External raw datasets, downloaded pretrained weights, environments and source-paper copies are not distributed. `PUBLICATION.json` records any additional omitted tracked artifacts with SHA-256 and reason. Omitted outputs can be regenerated through the existing installation, data preparation and training instructions; exact device-dependent numerical replay is not promised.

Original experiment manifests/test XML are historical research evidence and may mention omitted local files or older research source hashes. They are not a manifest of this publication. `PUBLICATION_FILES.json` separately hashes every public file except itself and Git metadata. Verify with `python scripts/verify_publication.py`; this verifies file integrity, not numerical reproduction.

Third-party data and models retain their own terms. Preserve vendor licenses and obtain external inputs from the documented sources.

Release-directory CPU validation: `16 passed` on 2026-11-17 using the original project environment. Existing results were preserved rather than rerunning expensive training. The public CI repeats the CPU unit tests.
