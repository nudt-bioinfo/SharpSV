# SharpSV Codebase Layout

## Top Level

- `SharpSV.py`: main user-facing pipeline entrypoint.
- `sharpsv/`: organized implementation package.
- `fermikit/`: local-assembly toolchain plus Python wrappers used in stage-3.
- `legacy/`: archived compatibility modules and older utilities.
- `docs/`: installation notes, usage tutorial, pipeline overview, codebase layout, release notes, and figure assets for the GitHub presentation layer.
- `workdir/`: intermediate runtime artifacts such as NPZ blocks, stage-1 candidates, stage-2 `stage2_predictions.csv`, and stage-3 products such as `stage3_refined_sv_results.csv`, `stage3_assembled_regions/`, `stage3_merge_temp/`, and `stage3_assembled_regions.sorted.bam`.

## `sharpsv/`

### `sharpsv/cli/`

- `pipeline.py`: the full BAM-to-final-VCF orchestration logic.

### `sharpsv/utils/`

- `console.py`: unified terminal rendering, progress display, ANSI color, and compact layout logic.

### `sharpsv/backend.py` and `sharpsv/native.py`

- package-internal loaders for the bundled native backend, replacing the old root-level compatibility modules.

### `sharpsv/stage1/`

- `features.py`: NPZ feature synthesis from BAM.
- `predict.py`: stage-1 candidate scoring over NPZ features.
- `model.py`: stage-1 neural model and dataset classes.
- `train.py`: stage-1 training entrypoint.

### `sharpsv/stage2/`

- `refine.py`: CPU image generation + GPU refinement inference.
- `features.py`: stage-2 feature helpers.
- `image.py`: pileup/image generation helpers.
- `model.py`: stage-2 refinement model definition.

### `sharpsv/stage3/`

- `pipeline.py`: stage-3 orchestration and resume logic.
- `sort_predictions.py`: converts stage-2 `stage2_predictions.csv` into `stage3_refined_sv_results.csv`.
- `assembly_validator.py`: adaptive validation against raw and locally assembled BAMs.

### `sharpsv/stage4/`

- `pipeline.py`: stage-4 orchestration from validated CSV to final SharpSV VCF.
- `realign_vcf.py`: DEL realignment engine with multiprocessing support.

### `sharpsv/_bundle/models/`

- `manifest.json`: release-backed model manifest with GitHub Release location, asset names, sizes, and SHA256 checksums.

### `sharpsv/_bundle/native/`

- `sharpsv.cpython-38-x86_64-linux-gnu.so`: packaged native backend asset loaded by `sharpsv/backend.py`.

## `fermikit/`

- `local_assemble.py`: builds `workdir/stage3_assembled_regions/` from `stage3_refined_sv_results.csv` and writes `stage3_assembly_results.json`, `stage3_assembled_regions_manifest.tsv`, and `stage3_assembled_regions_archive.tar.gz`.
- `merge_bams.py`: merges per-region assembly BAMs into `workdir/stage3_assembled_regions.sorted.bam` using `workdir/stage3_merge_temp/`.

## Entry Points

Most users should start with the full pipeline entrypoint:

- `SharpSV`
- `python SharpSV.py`

Developer-oriented stage entrypoints are also available and follow the repository stage boundaries:

- stage-1: feature extraction, candidate scoring, and training
- stage-2: refinement inference
- stage-3: prediction sorting and assembly validation
- stage-4: VCF export and DEL realignment

Direct module execution remains available under `python -m sharpsv.<stage>...` when working from a source checkout.
