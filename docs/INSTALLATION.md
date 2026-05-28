# SharpSV Installation And Usage

## Overview

SharpSV is distributed as a self-contained structural variant discovery tool:

- bundled stage-1 and stage-2 pretrained models are shipped inside `sharpsv/_bundle/models/`
- the native backend is shipped inside `sharpsv/_bundle/native/`
- the stage-3 assembly runtime is shipped inside `fermikit/fermi.kit/`
- the stage-2 bundled model is split into package parts and reconstructed automatically at runtime

End users do not need to train models or manually provide checkpoint paths for routine use.

## Recommended Environment

For GPU deployments, the recommended installation path is the bundled conda environment:

```bash
conda env update -n SharpSV -f environment.yml --prune
conda activate SharpSV
```

This environment pins:

- Python 3.8
- CUDA-capable PyTorch
- PyTorch Lightning
- `pysam`, `biopython`, `ray[tune]`, and other runtime dependencies

## Install From Source

After creating the environment, install SharpSV itself:

```bash
pip install .
```

This installs:

- the `sharpsv` Python package
- bundled model assets
- bundled native backend
- bundled `fermikit` runtime
- command-line entry points such as `SharpSV`

## Quick Start

Run the full pipeline:

```bash
SharpSV \
  -bamfilepath /path/to/sample.sorted.bam \
  -fastapath /path/to/reference.fa \
  -workdir ./workdir \
  -processes 32 \
  -output ./SharpSV.vcf
```

The default bundled models are used automatically.

You can also use the installed console entrypoint:

```bash
SharpSV \
  -bamfilepath /path/to/sample.sorted.bam \
  -fastapath /path/to/reference.fa \
  -workdir ./workdir \
  -processes 32 \
  -output ./SharpSV.vcf
```

## Optional Model Override

Advanced users can override the bundled models with their own checkpoints:

```bash
SharpSV \
  -bamfilepath /path/to/sample.sorted.bam \
  -fastapath /path/to/reference.fa \
  --stage1-model /path/to/custom_stage1.ckpt \
  --stage2-model /path/to/custom_stage2.ckpt \
  -output ./SharpSV.vcf
```

## Intermediate Outputs

During execution SharpSV produces staged artifacts:

- `workdir/stage1_candidates.csv`
- `workdir/stage2_predictions.csv`
- `workdir/stage3_refined_sv_results.csv`
- `workdir/stage3_assembled_regions/`
- `workdir/stage4_final_adaptive_validated.vcf`
- final `SharpSV.vcf`

Resume logic is built into the pipeline, so rerunning the same command reuses completed stages when possible.

## GitHub Distribution Notes

The repository is prepared for direct GitHub distribution:

- bundled models are included as package data
- the native backend `.so` is included as package data
- `fermikit` runtime binaries are included as package data
- packaging metadata is present for both wheel and conda-based installs

Additional release notes are documented in [GITHUB_RELEASE.md](GITHUB_RELEASE.md).

## Packaging Notes

Python packaging files included in the repository:

- `pyproject.toml`
- `setup.py`
- `MANIFEST.in`
- `conda.recipe/meta.yaml`

These ensure that wheels and future conda packages include:

- `sharpsv/_bundle/models/*.bin`
- `sharpsv/_bundle/native/*.so`
- `fermikit/fermi.kit/*`
