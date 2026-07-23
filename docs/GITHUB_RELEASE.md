# SharpSV GitHub Release Guide

This repository is prepared to be published as a user-facing SharpSV release rather than a loose research workspace.

## What Users Receive

The GitHub repository includes:

- the full `SharpSV.py` end-to-end pipeline
- release-backed metadata for the default stage-1 and stage-2 model assets
- a packaged `--demo` BAM/FASTA bundle with the required BAM/FASTA indexes for a zero-setup smoke test
- the packaged native backend used by feature extraction
- the packaged `fermikit` runtime required by stage-3 local assembly
- `environment.yml`, `requirements.txt`, and Python packaging metadata for reproducible installation

End users can run SharpSV directly after environment setup without manually locating pretrained checkpoints.

## Bundled Model Strategy

To keep the repository and wheel lightweight and GitHub-friendly:

- the repository stores only `sharpsv/_bundle/models/manifest.json`
- the actual pretrained checkpoints are uploaded to GitHub Release assets
- SharpSV downloads the assets to the local cache on first use
- each download is verified against the manifest SHA256 before it is accepted

The current manifest expects these release asset names:

- `stage1.model.bin`
- `stage2.model.bin`

and the default release tag:

- `bundled-models-v0.1.0`

You can override the download base with `SHARPSV_BUNDLE_BASE_URL` if you host the assets elsewhere.

## Creating The Model Release

Before publishing the lightweight git repository, create a GitHub Release in `nudt-bioinfo/SharpSV` with tag:

- `bundled-models-v0.1.0`

and upload these two assets:

- `stage1.model.bin`
- `stage2.model.bin`

Their manifest checksums are:

- `stage1.model.bin`: `0d84a34ce821bd72bfe9a023942526d1d6d4971d1f7456915ba41876cc3c7128`
- `stage2.model.bin`: `6d70260bad5a5083097795dad4c3f4c524b52c02b8a81b387b87c317115a512c`

Suggested maintainer flow:

1. Open the GitHub repository.
2. Go to `Releases`.
3. Create a new release with tag `bundled-models-v0.1.0`.
4. Upload `stage1.model.bin` and `stage2.model.bin` as release assets.
5. Publish the release.

Once the release is live, a clean SharpSV install can fetch the models automatically on first run.

## Installation Paths

Recommended:

```bash
conda env create -n SharpSV -f environment.yml
conda activate SharpSV
pip install --no-deps .
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

Self-contained demo command:

```bash
python SharpSV.py \
  --demo \
  -workdir ./demo-workdir \
  -processes 4 \
  -output ./demo-workdir/SharpSV.demo.vcf
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
- if `workdir/final_adaptive_validated.csv` exists, stage-1 through stage-3 are skipped
- if later-stage VCF intermediates already exist, stage-4 resumes from the newest valid artifact

This allows long production runs to recover from interruption without recomputing the entire pipeline.

## Distribution Notes

- Wheels are intentionally platform-specific because the packaged native backend `.so` is included.
- the model repository payload is now limited to a JSON manifest, so release assets no longer inflate git history or wheel contents
- `conda.recipe/meta.yaml` reuses `pip install .` so wheel and conda distribution paths stay aligned.
