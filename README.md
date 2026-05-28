# SharpSV

SharpSV is a site-centered hierarchical analysis and refinement pipeline for structural variant discovery from short-read sequencing data that scans whole-genome alignments to recover structurally abnormal regions, converts candidate loci into Vertical-Site Profile (VSP) image tensors, decodes them with a spatial-sequential neural architecture, and refines final breakpoints through local assembly and adaptive validation; relative to conventional heuristic SV callers, SharpSV is designed to preserve fragmented breakpoint evidence instead of discarding it early, with a particular emphasis on improving insertion sensitivity while maintaining competitive performance across common SV classes.

<p align="center">
  <img src="docs/assets/fig1-workflow.svg" alt="SharpSV workflow overview" width="1120">
</p>

<p align="center">
  <em>Fig. 1. SharpSV workflow overview. A stage-by-stage walkthrough is available in <a href="docs/PIPELINE_OVERVIEW.md">docs/PIPELINE_OVERVIEW.md</a>.</em>
</p>

<p align="center"><strong>License:</strong> MIT</p>

## Concept

SharpSV follows the same four-stage logic described in the manuscript:

- coarse screening: 1,000-bp windows, nine site-wise features, and attentive MIL pruning of normal genomic background
- sequence-to-image encoding: twenty consecutive 50-bp subregions converted into VSP image tensors
- spatial-sequential recognition: CNN and transformer modeling across ordered VSP segments
- breakpoint refinement: local assembly, contig realignment, and adaptive validation to base-pair resolution

The production codebase then appends a practical finalization stage that exports validated calls to VCF and performs DEL-focused realignment refinement for the final deliverable file.

## Highlights

- full end-to-end pipeline from BAM to final VCF
- release-backed pretrained stage-1 and stage-2 models
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

The default bundled models are used automatically. On the first run, SharpSV downloads the stage checkpoints from the SharpSV GitHub Release assets into the local cache and verifies them with SHA256 before inference starts. Advanced users can override them:

```bash
python SharpSV.py ... \
  --stage1-model /path/to/custom_stage1.ckpt \
  --stage2-model /path/to/custom_stage2.ckpt
```

By default the model cache lives under `XDG_CACHE_HOME` or `/tmp/sharpsv-cache/bundled-models`. Maintainers can override the release download root with `SHARPSV_BUNDLE_BASE_URL`.

## Manuscript To Runtime Mapping

- manuscript coarse screening corresponds to repository `stage-1`
- manuscript sequence-to-image encoding plus SSR-Net recognition corresponds to repository `stage-2`
- manuscript local assembly and breakpoint refinement corresponds to repository `stage-3`
- repository `stage-4` is a production-facing output layer for VCF export and DEL realignment

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

- the repository ships a model manifest under `sharpsv/_bundle/models/manifest.json`
- pretrained checkpoints are fetched from GitHub Release assets on first use and cached locally
- each downloaded checkpoint is verified against the manifest SHA256 before SharpSV uses it
- the native backend `.so` is shipped under `sharpsv/_bundle/native/`
- the `fermikit/fermi.kit/` runtime is shipped as package data for stage-3 assembly
- `setup.py`, `pyproject.toml`, `MANIFEST.in`, and `conda.recipe/meta.yaml` are aligned so wheel and conda distribution include the same runtime assets

## Stage Commands

Stage-specific entrypoints are now exposed from the package rather than cluttering the repository root:

```bash
sharpsv-stage1-extract-features ...
sharpsv-stage1-score-candidates ...
sharpsv-stage1-train ...
sharpsv-stage2-refine-predictions ...
sharpsv-stage3-sort-predictions ...
sharpsv-stage3-validate-assembly ...
sharpsv-stage4-export-vcf ...
sharpsv-stage4-realign-vcf ...
```

If you are working directly from a source checkout, the equivalent module form is:

```bash
python -m sharpsv.stage1.features ...
python -m sharpsv.stage1.predict ...
python -m sharpsv.stage1.train ...
python -m sharpsv.stage2.refine ...
python -m sharpsv.stage3.sort_predictions ...
python -m sharpsv.stage3.assembly_validator ...
python -m sharpsv.stage4.export_vcf_cli ...
python -m sharpsv.stage4.realign_vcf ...
```

## Layout

The codebase is organized by stage under `sharpsv/`.

- `sharpsv/stage1/`: feature synthesis, prediction, training
- `sharpsv/stage2/`: refinement pipeline, image helpers, refinement model
- `sharpsv/stage3/`: sorting, local assembly, adaptive validation
- `sharpsv/stage4/`: VCF export, DEL realignment, final VCF orchestration
- `sharpsv/_bundle/models/`: bundled model manifest for release-backed checkpoint downloads
- `sharpsv/_bundle/native/`: packaged native backend
- `sharpsv/utils/`: console and shared helpers
- `legacy/`: archived compatibility modules

See [docs/CODEBASE_LAYOUT.md](docs/CODEBASE_LAYOUT.md) for the detailed layout map.
