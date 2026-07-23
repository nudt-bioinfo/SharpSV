# SharpSV Usage Tutorial

## Overview

This guide covers the normal end-to-end workflow for SharpSV:

1. create the runtime environment
2. run the full BAM-to-VCF pipeline
3. understand what gets written into the workdir
4. rerun safely with built-in resume behavior

For environment setup, see [INSTALLATION.md](INSTALLATION.md). For stage internals, see [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md).

## Before You Start

Prepare the following inputs:

- a sorted and indexed BAM file
- a reference FASTA file
- BAM records with `MD` tags available

Install SharpSV first:

```bash
conda env create -n SharpSV -f environment.yml
conda activate SharpSV
pip install --no-deps .
```

## Run The Bundled Demo

Use the packaged demo bundle when you want a reviewer-friendly sanity check without preparing any local BAM or FASTA files:

```bash
python SharpSV.py \
  --demo \
  -workdir ./demo-workdir \
  -processes 4 \
  -output ./demo-workdir/SharpSV.demo.vcf
```

The demo bundle contains a small HG002-derived BAM/FASTA pair plus the required `.bai`, `.fai`, and BWA index files. The demo reference is synthetic and only 100 kb long, so output coordinates are relative to the packaged demo FASTA rather than full-genome GRCh37 coordinates. The source-region mapping is recorded in `sharpsv/_bundle/demo/demo_region.json`.

`--demo` also enables stage-2 image export automatically, so the run keeps:

- `workdir/stage1_candidates.csv`
- `workdir/stage2_predictions.csv`
- `workdir/stage2_images/` (three grayscale PNGs per candidate window, one per channel)
- `workdir/stage3_refined_sv_results.csv`
- `workdir/stage3_assembled_regions/`
- `workdir/final_adaptive_validated.csv`
- the final VCF at `-output`

## Run The Full Pipeline

Use the installed CLI:

```bash
SharpSV \
  -bamfilepath /path/to/sample.sorted.bam \
  -fastapath /path/to/reference.fa \
  -workdir ./workdir \
  -processes 32 \
  -output ./SharpSV.vcf
```

If you are running directly from a source checkout, replace `SharpSV` with `python SharpSV.py`.

On the first run, SharpSV downloads the bundled stage checkpoints from the GitHub Release assets, stores them in the local cache, and verifies them with SHA256 before inference.
If the network is flaky, SharpSV keeps a partial `.part` file and resumes it automatically on the next run. You can increase retry patience with `SHARPSV_BUNDLE_DOWNLOAD_RETRIES` and `SHARPSV_BUNDLE_DOWNLOAD_RETRY_DELAY_SEC`.

## Common Arguments

- `--demo`: use the packaged demo BAM/FASTA bundle instead of external inputs
- `-bamfilepath`: input sorted BAM file
- `-fastapath`: reference FASTA file
- `-output`: final VCF path
- `-workdir`: intermediate directory; if omitted, SharpSV uses `<output>.workdir`
- `-processes`: worker process count for feature extraction and stage orchestration
- `--stage1-model`: optional override for the bundled stage-1 checkpoint
- `--stage2-model`: optional override for the bundled stage-2 checkpoint
- `--stage2-save-images`: save one contact-sheet PNG per stage-2 candidate window under `workdir/stage2_images/`
- `--force_regenerate_npz`: rebuild stage-1 NPZ features even if reusable outputs already exist

## What SharpSV Writes

SharpSV writes both reusable intermediates and final outputs.

### Stage 1

- NPZ feature blocks are written directly under `workdir/`
- candidate CSV: `workdir/stage1_candidates.csv`

### Stage 2

- refined prediction CSV: `workdir/stage2_predictions.csv`
- optional candidate image contact sheets: `workdir/stage2_images/`

### Stage 3

- sorted candidate intervals: `workdir/stage3_refined_sv_results.csv`
- per-region local assembly outputs: `workdir/stage3_assembled_regions/`
- merged assembly BAM: `workdir/stage3_assembled_regions.sorted.bam`
- adaptive validation CSV: `workdir/final_adaptive_validated.csv`

### Stage 4

- exported VCF from validated calls: `workdir/stage4_final_adaptive_validated.vcf`
- final deliverable VCF at the path given by `-output`

## Resume Behavior

SharpSV is designed to resume from completed stages.

- if `workdir/stage2_predictions.csv` already exists, SharpSV skips stage-1 and stage-2
- if `workdir/final_adaptive_validated.csv` already exists, SharpSV skips stage-1 through stage-3
- legacy `predictions.csv` and older stage-3 artifact names are still recognized for compatibility
- if you need to rebuild the stage-1 feature corpus, add `--force_regenerate_npz`

This means you can usually rerun the same command after interruption without recomputing finished work.

## Custom Checkpoints

Advanced users can replace the bundled stage checkpoints:

```bash
SharpSV \
  -bamfilepath /path/to/sample.sorted.bam \
  -fastapath /path/to/reference.fa \
  --stage1-model /path/to/custom_stage1.ckpt \
  --stage2-model /path/to/custom_stage2.ckpt \
  -output ./SharpSV.vcf
```

## Model Cache

By default the model cache lives under:

- `XDG_CACHE_HOME/bundled-models` when `XDG_CACHE_HOME` is set
- otherwise `/tmp/sharpsv-cache/bundled-models`

Maintainers can point SharpSV at another release or mirror root with `SHARPSV_BUNDLE_BASE_URL`.

## Next References

- installation and packaging: [INSTALLATION.md](INSTALLATION.md)
- stage-by-stage algorithm view: [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md)
- repository structure and stage entrypoints: [CODEBASE_LAYOUT.md](CODEBASE_LAYOUT.md)
