# SharpSV Installation Guide

## Overview

SharpSV is distributed as a self-contained structural variant discovery tool:

- bundled stage-1 and stage-2 pretrained models are published as GitHub Release assets
- the repository only ships the model manifest inside `sharpsv/_bundle/models/`
- the native backend is shipped inside `sharpsv/_bundle/native/`
- the stage-3 assembly runtime is shipped inside `fermikit/fermi.kit/`
- the first SharpSV run downloads the pretrained checkpoints into the local cache and verifies them with SHA256

End users do not need to train models or manually provide checkpoint paths for routine use.

## Recommended Environment

For GPU deployments, the recommended installation path is the bundled conda environment:

```bash
conda env create -n SharpSV -f environment.yml
conda activate SharpSV
```

If you already have a `SharpSV` environment and want to refresh it:

```bash
conda env update -n SharpSV -f environment.yml --prune
conda activate SharpSV
```

This environment pins:

- Python 3.8
- CUDA-capable PyTorch
- PyTorch Lightning
- `pysam`, `biopython`, `ray-tune`, and other runtime dependencies
- conda as the primary package manager for the runtime stack to reduce pip downloads and solver conflicts

## Install From Source

After creating or updating the environment, install SharpSV itself without re-resolving dependencies:

```bash
pip install --no-deps .
```

This installs:

- the `sharpsv` Python package
- bundled model manifest and release metadata
- bundled native backend
- bundled `fermikit` runtime
- command-line entry points such as `SharpSV`

## Verify The Install

After installation, confirm that the main CLI is available:

```bash
SharpSV --help
```

The default bundled models are used automatically at runtime.
If the local cache is empty, SharpSV downloads:

- `stage1.model.bin`
- `stage2.model.bin`

from the configured SharpSV GitHub Release and stores them under the model cache directory before stage-1/stage-2 inference.

Default cache location:

- `XDG_CACHE_HOME/bundled-models` when `XDG_CACHE_HOME` is set
- otherwise `/tmp/sharpsv-cache/bundled-models`

Optional maintainer override:

- set `SHARPSV_BUNDLE_BASE_URL` to point SharpSV at another release or mirror root

## Next Step

After installation, continue with the full usage guide:

- [TUTORIAL.md](TUTORIAL.md): end-to-end run command, outputs, resume behavior, and model overrides

## GitHub Distribution Notes

The repository is prepared for direct GitHub distribution:

- bundled model manifest metadata is included as package data
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

- `sharpsv/_bundle/models/*.json`
- `sharpsv/_bundle/native/*.so`
- `fermikit/fermi.kit/*`
