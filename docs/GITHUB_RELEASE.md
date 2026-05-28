# SharpSV GitHub Release Guide

This repository is prepared to be published as a user-facing SharpSV release rather than a loose research workspace.

## What Users Receive

The GitHub repository includes:

- the full `SharpSV.py` end-to-end pipeline
- packaged default stage-1 and stage-2 model assets
- the packaged native backend used by feature extraction
- the packaged `fermikit` runtime required by stage-3 local assembly
- `environment.yml`, `requirements.txt`, and Python packaging metadata for reproducible installation

End users can run SharpSV directly after environment setup without manually locating pretrained checkpoints.

## Bundled Model Strategy

To keep the repository and wheel compatible with GitHub file-size limits:

- the stage-1 bundled model is stored directly as `sharpsv/_bundle/models/stage1.model.bin`
- the larger stage-2 bundled model is stored as split package assets:
  - `stage2.model.bin.part-000`
  - `stage2.model.bin.part-001`
  - `stage2.model.bin.part-002`
  - `stage2.model.bin.part-003`
  - `stage2.model.bin.part-004`

At runtime SharpSV reconstructs the stage-2 checkpoint into a temporary cache automatically. Users do not need to perform any manual merge step.

## Installation Paths

Recommended:

```bash
conda env update -n SharpSV -f environment.yml --prune
conda activate SharpSV
pip install .
```

Alternative wheel build:

```bash
python setup.py sdist bdist_wheel
pip install dist/sharpsv-0.1.0-cp38-cp38-linux_x86_64.whl
```

## User Command

```bash
python SharpSV.py \
  -bamfilepath /path/to/sample.sorted.bam \
  -fastapath /path/to/reference.fa \
  -workdir ./workdir \
  -processes 32 \
  -output ./SharpSV.vcf
```

Optional advanced overrides:

```bash
python SharpSV.py ... \
  --stage1-model /path/to/custom_stage1.ckpt \
  --stage2-model /path/to/custom_stage2.ckpt
```

## Resume Behavior

SharpSV reuses staged outputs when they already exist:

- if `workdir/stage2_predictions.csv` exists, stage-1 and stage-2 are skipped
- if `final_adaptive_validated.csv` exists, stage-1 through stage-3 are skipped
- if later-stage VCF intermediates already exist, stage-4 resumes from the newest valid artifact

This allows long production runs to recover from interruption without recomputing the entire pipeline.

## Distribution Notes

- Wheels are intentionally platform-specific because the packaged native backend `.so` is included.
- `setup.py` contains a small packaging guard to prevent stale reconstructed model files from being accidentally re-bundled during repeated local builds.
- `conda.recipe/meta.yaml` reuses `pip install .` so wheel and conda distribution paths stay aligned.
