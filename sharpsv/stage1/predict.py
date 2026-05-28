# -*- coding:utf-8 -*-
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sharpsv-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sharpsv-cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import numpy as np
import pandas as pd
import torch

from ..bundled_models import resolve_runtime_checkpoint_path
from ..utils.console import emit, emit_banner, emit_progress, format_count


torch.multiprocessing.set_sharing_strategy("file_system")


def load_trained_model(checkpoint_path, data_dirs, output_path, batch_size=32):
    from .model import SharpSVLightningModel

    config = {
        "lr": 1e-5,
        "batch_size": batch_size,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.001,
    }

    runtime_checkpoint_path = resolve_runtime_checkpoint_path(checkpoint_path)
    emit("stage-1", f"restoring candidate scorer from {checkpoint_path}")
    checkpoint = torch.load(runtime_checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = SharpSVLightningModel(
        path=data_dirs,
        config=config,
        predict_mode=True,
        prediction_output_csv=output_path,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        emit("stage-1", f"checkpoint compatibility note: missing={missing}, unexpected={unexpected}")
    model.eval()
    model.freeze()
    return model


def _select_inference_device_ids(max_gpu_workers=2):
    if not torch.backends.cuda.is_built():
        emit(
            "stage-1",
            "CUDA unavailable because this PyTorch build has no CUDA support "
            f"(torch {torch.__version__}, torch.version.cuda={torch.version.cuda}). "
            "Falling back to CPU inference.",
        )
        return []

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        emit(
            "stage-1",
            "CUDA runtime is unavailable to PyTorch "
            f"(torch {torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            f"device_count={torch.cuda.device_count()}). Falling back to CPU inference.",
        )
        return []

    device_ids = list(range(min(torch.cuda.device_count(), max_gpu_workers)))
    gpu_names = [torch.cuda.get_device_name(device_id) for device_id in device_ids]
    emit("stage-1", f"using GPU devices {device_ids} for inference: {gpu_names}")
    return device_ids


def _normalize_batch_indices(batch_indices):
    if (
        isinstance(batch_indices, (list, tuple))
        and len(batch_indices) == 2
        and isinstance(batch_indices[0], (list, tuple))
        and batch_indices[0]
        and all(isinstance(chrom, str) for chrom in batch_indices[0])
    ):
        chroms, positions = batch_indices
        if isinstance(chroms, tuple):
            chroms = list(chroms)
        elif not isinstance(chroms, list):
            chroms = [chroms]

        if isinstance(positions, torch.Tensor):
            positions = positions.tolist()
        elif isinstance(positions, np.ndarray):
            positions = positions.tolist()
        elif isinstance(positions, tuple):
            positions = list(positions)
        elif not isinstance(positions, list):
            positions = [positions]

        if len(chroms) != len(positions):
            raise ValueError(f"Index batch is malformed: {batch_indices}")
        return [(str(chrom), int(pos)) for chrom, pos in zip(chroms, positions)]

    if isinstance(batch_indices, list) and batch_indices and isinstance(batch_indices[0], tuple):
        normalized = []
        for item in batch_indices:
            if len(item) != 2:
                raise ValueError(f"Unsupported tuple-shaped index item: {item}")
            chrom, pos = item
            normalized.append((str(chrom), int(pos)))
        return normalized

    if isinstance(batch_indices, list):
        normalized = []
        for item in batch_indices:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"Unsupported list-shaped batch indices: {batch_indices}")
            chrom, pos = item
            normalized.append((str(chrom), int(pos)))
        return normalized

    if isinstance(batch_indices, tuple) and len(batch_indices) == 2:
        chrom, pos = batch_indices
        return [(str(chrom), int(pos))]

    raise TypeError(f"Unsupported batch index format: {type(batch_indices)!r}")


def _write_stage1_predictions(output_path, all_indices, all_probs):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probs_np = np.asarray(all_probs, dtype=float).reshape(-1)

    if probs_np.size == 0:
        pd.DataFrame(columns=["chr", "position", "score"]).to_csv(output_path, index=False)
        emit("stage-1", "no prediction scores were produced; wrote an empty candidate CSV.")
        return str(output_path)

    if len(all_indices) != probs_np.size:
        raise RuntimeError(
            f"Prediction/index length mismatch: {len(all_indices)} indices vs {probs_np.size} scores."
        )
    threshold = float(np.percentile(probs_np, 90))
    emit("stage-1", f"dynamic top-decile threshold selected at {threshold:.4f}")

    skipped_invalid = 0
    positive_records = [
        {"chr": chrom, "position": int(position), "score": float(score)}
        for (chrom, position), score in zip(all_indices, probs_np)
        if score >= threshold and int(position) > 0
    ]
    skipped_invalid = sum(
        1 for (_, position), score in zip(all_indices, probs_np) if score >= threshold and int(position) <= 0
    )
    if skipped_invalid:
        emit("stage-1", f"dropped {format_count(skipped_invalid)} non-positive candidate positions before writing CSV")
    pd.DataFrame(positive_records, columns=["chr", "position", "score"]).to_csv(output_path, index=False)
    emit(
        "stage-1",
        f"candidate CSV ready: {format_count(len(positive_records))} sites written to {output_path}",
    )
    return str(output_path)


def _default_stage1_batch_size(batch_size, device_ids):
    if not device_ids:
        return batch_size
    if batch_size != 32:
        return batch_size
    # The legacy default (32) underuses A6000-class GPUs badly for this 1D CNN.
    return 512 if len(device_ids) > 1 else 256


def run_inference(model, batch_size=32, max_gpu_workers=2):
    from .model import SharpSVDataset

    npz_file_count = sum(len(list(Path(data_dir).glob("*.npz"))) for data_dir in model.path)
    emit(
        "stage-1",
        f"loading stage-1 feature corpus from {format_count(npz_file_count)} NPZ blocks",
    )
    test_dataset = SharpSVDataset(model.path, mode="test", verbose=False)
    emit(
        "stage-1",
        f"feature corpus ready: {format_count(test_dataset.loaded_file_count)} files, "
        f"{format_count(len(test_dataset))} windows",
    )
    emit_banner(
        "Stage-1 Candidate Scoring",
        details=[
            ("feature blocks", format_count(test_dataset.loaded_file_count)),
            ("windows", format_count(len(test_dataset))),
            ("requested batch", batch_size),
            ("output", model.prediction_output_csv),
        ],
    )

    device_ids = _select_inference_device_ids(max_gpu_workers=max_gpu_workers)
    use_cuda = bool(device_ids)
    effective_batch_size = _default_stage1_batch_size(batch_size, device_ids)
    if effective_batch_size != batch_size:
        emit(
            "stage-1",
            f"raising batch size from {batch_size} to {effective_batch_size} to improve GPU throughput",
        )

    base_model = model.model
    if use_cuda:
        primary_device = torch.device(f"cuda:{device_ids[0]}")
        base_model = base_model.to(primary_device)
        torch.backends.cudnn.benchmark = True
        if len(device_ids) > 1:
            inference_model = torch.nn.DataParallel(base_model, device_ids=device_ids)
        else:
            inference_model = base_model
    else:
        primary_device = torch.device("cpu")
        inference_model = base_model.to(primary_device)

    inference_model.eval()

    total_windows = len(test_dataset)
    total_batches = (total_windows + effective_batch_size - 1) // effective_batch_size
    emit(
        "stage-1",
        f"starting candidate scoring over {format_count(total_windows)} windows "
        f"in {format_count(total_batches)} batches (batch={effective_batch_size})",
    )

    all_indices = test_dataset.indices
    all_probs = np.empty(total_windows, dtype=np.float32)
    data_tensor = test_dataset.data
    use_autocast = use_cuda
    start_time = time.time()
    progress_interval = max(1, min(200, total_batches // 100 if total_batches > 0 else 1))
    cursor = 0

    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, total_windows, effective_batch_size), start=1):
            end = min(start + effective_batch_size, total_windows)
            batch = data_tensor[start:end].view(-1, 1000, 9).permute(0, 2, 1).contiguous()
            batch = batch.to(primary_device, non_blocking=use_cuda)

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    batch_probs, _, _ = inference_model(batch)
            else:
                batch_probs, _, _ = inference_model(batch)

            batch_probs_np = batch_probs.detach().float().cpu().view(-1).numpy()
            all_probs[cursor : cursor + len(batch_probs_np)] = batch_probs_np
            cursor += len(batch_probs_np)

            if batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == total_batches:
                elapsed = time.time() - start_time
                windows_done = end
                windows_per_sec = windows_done / elapsed if elapsed > 0 else 0.0
                remaining_windows = total_windows - windows_done
                eta_sec = remaining_windows / windows_per_sec if windows_per_sec > 0 else float("inf")
                emit_progress(
                    "stage-1",
                    windows_done,
                    total_windows,
                    speed=windows_per_sec,
                    eta_seconds=eta_sec,
                    extra=f"batch {batch_idx}/{format_count(total_batches)}",
                )

    if cursor != total_windows:
        raise RuntimeError(f"Stage-1 inference produced {cursor} scores for {total_windows} windows.")

    return _write_stage1_predictions(model.prediction_output_csv, all_indices, all_probs)


def predict_workdir(workdir, checkpoint_path, output_path, batch_size=32, max_gpu_workers=2):
    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"Workdir not found: {workdir}")
    runtime_checkpoint_path = resolve_runtime_checkpoint_path(checkpoint_path)
    if not os.path.exists(runtime_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_trained_model(checkpoint_path, [workdir], output_path, batch_size)
    return run_inference(model, batch_size=batch_size, max_gpu_workers=max_gpu_workers)


def build_parser():
    parser = argparse.ArgumentParser(description="Predict SharpSV NPZ features with a pretrained checkpoint")
    parser.add_argument("-workdir", "--workdir", required=True, help="Directory containing SharpSV-generated NPZ files")
    parser.add_argument("-checkpointpath", "--checkpointpath", required=True, help="Path to the pretrained .ckpt model")
    parser.add_argument("-output", "--output", required=True, help="Output CSV path for predicted positive sites")
    parser.add_argument("-batchsize", "--batchsize", type=int, default=32, help="Prediction batch size")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    predict_workdir(args.workdir, args.checkpointpath, args.output, args.batchsize)
    emit("stage-1", f"prediction CSV saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
