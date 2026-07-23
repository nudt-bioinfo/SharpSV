# SharpSV

SharpSV is an end-to-end short-read structural variant discovery tool. It scans whole-genome alignments in 1,000-bp windows, filters normal background with attentive MIL, converts candidate regions into ordered VSP image sequences, classifies them with CNN- and transformer-based recognition, and refines breakpoints through local assembly and realignment before producing the final VCF.

<p align="center">
  <img src="docs/assets/fig1-workflow.svg" alt="SharpSV workflow overview" width="1120">
</p>

<p align="center">
  <em>Fig. 1. SharpSV workflow overview. A stage-by-stage walkthrough is available in <a href="docs/PIPELINE_OVERVIEW.md">docs/PIPELINE_OVERVIEW.md</a>.</em>
</p>

## Highlights

- full end-to-end pipeline from BAM to final VCF
- release-backed pretrained stage-1 and stage-2 models
- packaged `--demo` bundle with bundled BAM/FASTA inputs and stage-by-stage outputs
- packaged native backend and packaged `fermikit` runtime
- GPU-aware stage-1 and stage-2 inference
- resume-aware execution for interrupted long runs
- installable as a Python package with wheel and conda metadata

## Installation

The recommended installation path is the bundled conda environment:

- Python `3.8`
- CUDA-capable PyTorch `2.4.1`
- PyTorch Lightning `2.4.0`
- conda-managed Ray Tune and scientific runtime dependencies
- packaged native backend loaded from `sharpsv/_bundle/native/`

Create the environment from scratch:

```bash
conda env create -n SharpSV -f environment.yml
conda activate SharpSV
```

If you already have a `SharpSV` environment and want to refresh it:

```bash
conda env update -n SharpSV -f environment.yml --prune
conda activate SharpSV
```

Then install SharpSV itself without asking `pip` to resolve the runtime stack again:

```bash
pip install --no-deps .
```

If you prefer a plain dependency list, `requirements.txt` mirrors the runtime stack. For a step-by-step install guide, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Quick Start

SharpSV expects a sorted and indexed BAM file with `MD` tags available.

Try the self-contained demo first:

```bash
python SharpSV.py \
  --demo \
  -workdir ./demo-workdir \
  -processes 4 \
  -output ./demo-workdir/SharpSV.demo.vcf
```

The bundled demo ships with a tiny HG002-derived BAM/FASTA pair, so no external input files are needed. A successful demo run keeps the final VCF plus reusable stage outputs such as `stage1_candidates.csv`, `stage2_predictions.csv`, `stage2_images/`, `stage3_assembled_regions/`, and `final_adaptive_validated.csv` under the chosen workdir.

Run the complete pipeline with the installed console entrypoint:

```bash
SharpSV \
  -bamfilepath /path/to/sample.sorted.bam \
  -fastapath /path/to/reference.fa \
  -workdir ./workdir \
  -processes 32 \
  -output ./SharpSV.vcf
```

If you are working from a source checkout, replace `SharpSV` with `python SharpSV.py`.

On the first run, SharpSV downloads the bundled stage checkpoints from the GitHub Release assets into the local cache and verifies them with SHA256 before inference starts. The model downloader now resumes partial downloads automatically, so rerunning the same command continues from the cached `.part` file instead of starting over. For output files, resume behavior, and advanced usage, see [docs/TUTORIAL.md](docs/TUTORIAL.md).

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Usage Tutorial](docs/TUTORIAL.md)
- [Pipeline Overview](docs/PIPELINE_OVERVIEW.md)
- [Codebase Layout](docs/CODEBASE_LAYOUT.md)
- [GitHub Release Notes](docs/GITHUB_RELEASE.md)

## Pipeline Stages

- `stage-1`: coarse screening over 1,000-bp windows using nine site-wise alignment features and MIL-based background pruning
- `stage-2`: sequence-to-image encoding plus spatial-sequential recognition over ordered 50-bp subregions
- `stage-3`: local assembly and adaptive validation for breakpoint confirmation and refinement
- `stage-4`: production-facing VCF export plus DEL realignment refinement for the final deliverable file

## Project Layout

- `sharpsv/`: main implementation package, organized by stage
- `sharpsv/stage1/` to `sharpsv/stage4/`: feature synthesis, candidate refinement, assembly validation, and final VCF generation
- `fermikit/`: packaged local-assembly runtime used in stage-3
- `sharpsv/_bundle/`: bundled model manifest and native backend assets
- `docs/`: installation, tutorial, pipeline, layout, and release documentation

For developer-oriented command entrypoints and a more detailed map of the repository, see [docs/CODEBASE_LAYOUT.md](docs/CODEBASE_LAYOUT.md).
