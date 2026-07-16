# Simulated Five-Class Inference Checkpoints

This directory is branch-specific and is only used by the `--pipeline-profile simulated-5class` workflow.

Files:

- `stage1_simulated_5class_compact.pt`
  - compact inference export of the simulated-data stage-1 anomaly model
  - contains the trained `best_threshold` used for window filtering
- `stage2_simulated_5class_compact_fp16.pt`
  - compact `fp16` inference export of the simulated-data five-class stage-2 model
  - predicts `DEL`, `INS`, `INV`, `TRA`, and `DUP`

These files were derived from the larger training checkpoints to keep this Git branch compatible with GitHub file-size limits while preserving end-to-end inference behavior.
