# SharpSV

SharpSV is a structural variant discovery pipeline for sorted, indexed BAM files. It packages feature extraction, neural screening, image-based refinement, local assembly, adaptive validation, and final VCF generation into a single user-facing workflow.

## Highlights

- full end-to-end pipeline from BAM to final VCF
- bundled pretrained stage-1 and stage-2 models
- packaged native backend and packaged `fermikit` runtime
- GPU-aware stage-1 and stage-2 inference
- resume-aware execution for interrupted long runs
- installable as a Python package with wheel and conda metadata

## Recommended Environment

- Python `3.8`
- CUDA-capable PyTorch `2.4.1`
- PyTorch Lightning `2.4.0`
- packaged native backend loaded from `sharpsv/_bundle/native/`

Create the recommended environment:

```bash
conda env update -n SharpSV -f environment.yml --prune
conda activate SharpSV
```

Then install SharpSV:

```bash
pip install .
```

If you prefer a plain dependency list, `requirements.txt` mirrors the runtime stack. For a user-facing install guide, see [docs/INSTALLATION.md](docs/INSTALLATION.md). For repository and distribution notes, see [docs/GITHUB_RELEASE.md](docs/GITHUB_RELEASE.md).

## Quick Start

SharpSV expects a sorted and indexed BAM file with `MD` tags available.

Run the complete pipeline:

```bash
python SharpSV.py \
  -bamfilepath /path/to/sample.sorted.bam \
  -fastapath /path/to/reference.fa \
  -workdir ./workdir \
  -processes 32 \
  -output ./SharpSV.vcf
```

The default bundled models are used automatically. Advanced users can override them:

```bash
python SharpSV.py ... \
  --stage1-model /path/to/custom_stage1.ckpt \
  --stage2-model /path/to/custom_stage2.ckpt
```

## Pipeline Outputs

- `stage-1`: BAM -> NPZ feature corpus -> `workdir/stage1_candidates.csv`
- `stage-2`: image refinement -> `workdir/stage2_predictions.csv`
- `stage-3`: interval decoding -> `workdir/stage3_refined_sv_results.csv`
- `stage-3`: local assembly -> `workdir/stage3_assembled_regions/`
- `stage-3`: merged assembly BAM -> `workdir/stage3_assembled_regions.sorted.bam`
- `stage-3`: adaptive validation -> `final_adaptive_validated.csv`
- `stage-4`: CSV to VCF export -> `workdir/stage4_final_adaptive_validated.vcf`
- `stage-4`: DEL realignment while preserving all variant types -> final `-output`

## Resume Logic

- if `workdir/stage2_predictions.csv` already exists, SharpSV skips stage-1 and stage-2
- if `final_adaptive_validated.csv` already exists, SharpSV skips stage-1 through stage-3
- legacy `predictions.csv` and older stage-3 artifact names are still recognized for compatibility

This allows production runs to continue after interruption without recomputing finished stages.

## Packaging Notes

- default models are shipped as package data under `sharpsv/_bundle/models/`
- the large stage-2 bundled model is stored as split package assets and reconstructed into a temporary cache at runtime
- the native backend `.so` is shipped under `sharpsv/_bundle/native/`
- the `fermikit/fermi.kit/` runtime is shipped as package data for stage-3 assembly
- `setup.py`, `pyproject.toml`, `MANIFEST.in`, and `conda.recipe/meta.yaml` are aligned so wheel and conda distribution include the same runtime assets

## Stage Entry Scripts

Individual stage wrappers are kept at the repository root:

```bash
python sharpsv_stage1_extract_features.py ...
python sharpsv_stage1_score_candidates.py ...
python sharpsv_stage1_train.py ...
python sharpsv_stage2_refine_predictions.py ...
python sharpsv_stage3_sort_predictions.py ...
python sharpsv_stage3_validate_assembly.py ...
python sharpsv_stage4_export_vcf.py ...
python sharpsv_stage4_realign_vcf.py ...
```

## Layout

The codebase is organized by stage under `sharpsv/`.

- `sharpsv/stage1/`: feature synthesis, prediction, training
- `sharpsv/stage2/`: refinement pipeline, image helpers, refinement model
- `sharpsv/stage3/`: sorting, local assembly, adaptive validation
- `sharpsv/stage4/`: VCF export, DEL realignment, final VCF orchestration
- `sharpsv/_bundle/models/`: packaged model assets
- `sharpsv/_bundle/native/`: packaged native backend
- `sharpsv/utils/`: console and shared helpers
- `legacy/`: archived compatibility modules

See [docs/CODEBASE_LAYOUT.md](docs/CODEBASE_LAYOUT.md) for the detailed layout map.
