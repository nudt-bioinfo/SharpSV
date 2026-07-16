# SV5 Direct Inference Checkpoints

This directory is branch-specific and is only used by the `--pipeline-profile sv5-direct` workflow.

Files:

- `stage1_sv5_direct_compact.pt`
  - compact inference export of the branch-specific stage-1 anomaly model
  - contains the trained `best_threshold` used for window filtering
- `stage2_sv5_direct_compact_fp16.pt`
  - compact `fp16` inference export of the branch-specific five-class stage-2 model
  - predicts `DEL`, `INS`, `INV`, `TRA`, and `DUP`

These files were derived from the larger training checkpoints to keep this Git branch compatible with GitHub file-size limits while preserving end-to-end inference behavior.
