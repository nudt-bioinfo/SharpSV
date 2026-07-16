#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sharpsv-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sharpsv-cache")

import numpy as np
import pandas as pd
import pysam
import torch
import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sharpsv.stage1.model import AdvancedSVModel, SharpSVDataset
from sharpsv.stage2.features import get_clip_num
from sharpsv.stage2.image import _adapt_contig_id, estimate_insert_sizes, pipeup_column
from sharpsv.stage2.model import SharpSVStage2WindowClassifier
from sharpsv.stage2.refine import SEQ_LEN, SLICE_SIZE, WINDOW_SIZE, draw_pic_tensor


LABEL_NAMES = ["DEL", "INS", "INV", "TRA", "DUP"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the trained stage-1 + five-class stage-2 models over chr1-22,X,Y and export a window-level VCF."
    )
    parser.add_argument(
        "--stage1-workdirs",
        nargs="+",
        required=True,
        help="One or more stage-1 NPZ directories covering the genome.",
    )
    parser.add_argument("--stage1-checkpoint", required=True, help="Path to stage-1 best_model.pt")
    parser.add_argument("--stage2-checkpoint", required=True, help="Path to stage-2 best_model.pt")
    parser.add_argument("--bam", required=True, help="Sorted/indexed BAM")
    parser.add_argument("--fasta", required=True, help="Reference FASTA")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--sample-name", default="SharpSV", help="VCF sample name")
    parser.add_argument("--stage1-batch-size", type=int, default=512, help="Stage-1 GPU inference batch size")
    parser.add_argument("--stage2-batch-size", type=int, default=16, help="Stage-2 GPU inference batch size")
    parser.add_argument("--cpu-workers", type=int, default=24, help="CPU workers used for stage-2 image generation")
    parser.add_argument("--device-ids", nargs="+", type=int, default=[0, 1], help="CUDA device ids to use")
    return parser.parse_args()


def available_process_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_chrom(chrom):
    chrom = str(chrom).upper().replace("CHR", "")
    if chrom == "X":
        return 23
    if chrom == "Y":
        return 24
    if chrom in {"M", "MT"}:
        return 25
    try:
        return int(chrom)
    except ValueError:
        return 999


def sort_records(records):
    return sorted(records, key=lambda row: (parse_chrom(row["chrom"]), int(row["window_start"])))


def load_csv_records(path):
    df = pd.read_csv(path)
    return df.to_dict("records")


def select_device_ids(requested_device_ids):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return []
    visible_ids = [device_id for device_id in requested_device_ids if device_id < torch.cuda.device_count()]
    return visible_ids or [0]


def load_stage1_model(checkpoint_path, device_ids):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    threshold = float(checkpoint["val_metrics"]["best_threshold"])

    model = AdvancedSVModel(in_channels=9, base_filters=64)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    if device_ids:
        primary_device = torch.device(f"cuda:{device_ids[0]}")
    else:
        primary_device = torch.device("cpu")
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(primary_device)
    return checkpoint, threshold, model, primary_device


def score_stage1_candidates(workdirs, checkpoint_path, batch_size, device_ids, output_csv):
    checkpoint, threshold, model, device = load_stage1_model(checkpoint_path, device_ids)
    dataset = SharpSVDataset(workdirs, mode="test", verbose=False)
    data = dataset.data
    indices = dataset.indices

    total = len(data)
    scores = np.empty(total, dtype=np.float32)
    started_at = time.time()
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
            end = min(start + batch_size, total)
            batch = data[start:end].view(-1, 1000, 9).permute(0, 2, 1).contiguous()
            batch = batch.to(device=device, non_blocking=use_cuda)
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    probs, _, _ = model(batch)
            else:
                probs, _, _ = model(batch)
            scores[start:end] = probs.view(-1).detach().float().cpu().numpy()

            if batch_idx == 1 or batch_idx % 100 == 0 or end == total:
                elapsed = time.time() - started_at
                rate = end / elapsed if elapsed > 0 else 0.0
                print(
                    f"[stage1-full] scored {end}/{total} windows ({rate:.1f} windows/s, threshold={threshold:.6f})"
                )

    candidate_idx = np.where(scores >= threshold)[0]
    rows = []
    for idx in candidate_idx:
        chrom, window_start = indices[int(idx)]
        window_start = int(window_start)
        rows.append(
            {
                "chrom": str(chrom),
                "window_start": window_start,
                "window_end": window_start + WINDOW_SIZE,
                "stage1_score": float(scores[int(idx)]),
                "stage1_threshold": threshold,
            }
        )

    rows = sort_records(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["chrom", "window_start", "window_end", "stage1_score", "stage1_threshold"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "stage1_threshold": threshold,
        "total_windows": int(total),
        "predicted_abnormal_windows": int(len(rows)),
        "loaded_stage1_npz_files": int(dataset.loaded_file_count),
        "workdirs": [str(Path(path).resolve()) for path in workdirs],
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    return rows, summary


_WORKER_BAM = None
_WORKER_FASTA = None
_WORKER_MEAN_INSERT = None
_WORKER_SD_INSERT = None
_WORKER_MAPQ_MEAN = None


def init_stage2_worker(bam_path, fasta_path, mean_insert_size, sd_insert_size, mapping_qualities_mean):
    global _WORKER_BAM, _WORKER_FASTA
    global _WORKER_MEAN_INSERT, _WORKER_SD_INSERT, _WORKER_MAPQ_MEAN

    _WORKER_BAM = pysam.AlignmentFile(bam_path, "rb")
    _WORKER_FASTA = pysam.FastaFile(fasta_path)
    _WORKER_MEAN_INSERT = mean_insert_size
    _WORKER_SD_INSERT = sd_insert_size
    _WORKER_MAPQ_MEAN = mapping_qualities_mean


def build_stage2_tensor(task):
    chrom = task["chrom"]
    window_start = int(task["window_start"])
    window_end = int(task["window_end"])

    bam_chr = _adapt_contig_id(_WORKER_BAM, chrom)
    fasta_chr = _adapt_contig_id(_WORKER_FASTA, chrom, prefer_numeric=True)
    if bam_chr is None or fasta_chr is None:
        return {"status": "error", "message": f"contig resolution failed for {chrom}", "task": task}

    try:
        clip_record = get_clip_num(_WORKER_BAM, bam_chr, window_start, window_end)
        clip_dict_record = dict(clip_record)

        seq_tensors = []
        for slice_idx in range(SEQ_LEN):
            slice_start = window_start + slice_idx * SLICE_SIZE
            slice_end = slice_start + SLICE_SIZE
            pile_record = pipeup_column(
                _WORKER_BAM,
                bam_chr,
                slice_start,
                slice_end,
                _WORKER_MEAN_INSERT,
                _WORKER_SD_INSERT,
                _WORKER_FASTA,
                clip_dict_record,
            )
            tensor = draw_pic_tensor(clip_dict_record, pile_record, slice_start, _WORKER_MAPQ_MEAN)
            seq_tensors.append(tensor.numpy())

        images = np.stack(seq_tensors, axis=0)
        return {"status": "ok", "task": task, "images": images}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "task": task}


def load_stage2_model(checkpoint_path, device_ids):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    label_names = checkpoint.get("label_names", LABEL_NAMES)
    model = SharpSVStage2WindowClassifier(
        input_channels=3,
        num_classes=len(label_names),
        d_model=512,
        nhead=8,
        num_layers=4,
        dropout=0.1,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    if device_ids:
        primary_device = torch.device(f"cuda:{device_ids[0]}")
    else:
        primary_device = torch.device("cpu")
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(primary_device)
    return checkpoint, label_names, model, primary_device


def write_stage2_predictions_csv(records, output_csv, label_names):
    fieldnames = [
        "chrom",
        "window_start",
        "window_end",
        "stage1_score",
        "pred_label",
        "pred_label_id",
        "pred_confidence",
    ] + [f"prob_{label}" for label in label_names]
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def predict_stage2_candidates(candidates, bam_path, fasta_path, checkpoint_path, batch_size, cpu_workers, device_ids, output_csv):
    mean_insert_size, sd_insert_size, mapping_qualities_mean = estimate_insert_sizes(bam_path)
    print(
        "[stage2-full] insert-size stats:",
        json.dumps(
            {
                "mean_insert_size": mean_insert_size,
                "sd_insert_size": sd_insert_size,
                "mapping_qualities_mean": mapping_qualities_mean,
            },
            indent=2,
            sort_keys=True,
        ),
    )

    checkpoint, label_names, model, device = load_stage2_model(checkpoint_path, device_ids)
    use_cuda = device.type == "cuda"

    ctx = mp.get_context("spawn")
    tasks = list(candidates)
    started_at = time.time()
    records = []
    failed = []

    with ctx.Pool(
        processes=max(1, int(cpu_workers)),
        initializer=init_stage2_worker,
        initargs=(bam_path, fasta_path, mean_insert_size, sd_insert_size, mapping_qualities_mean),
    ) as pool:
        batch_images = []
        batch_tasks = []
        processed = 0

        def flush_batch():
            nonlocal batch_images, batch_tasks, records
            if not batch_images:
                return

            image_tensor = torch.from_numpy(np.stack(batch_images, axis=0)).to(device=device, non_blocking=use_cuda).float().div_(255.0)
            with torch.no_grad():
                if use_cuda:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(image_tensor, None)
                else:
                    logits = model(image_tensor, None)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred_ids = probs.argmax(axis=1)
            confidences = probs.max(axis=1)

            for task, pred_id, confidence, prob_vec in zip(batch_tasks, pred_ids, confidences, probs):
                row = {
                    "chrom": task["chrom"],
                    "window_start": int(task["window_start"]),
                    "window_end": int(task["window_end"]),
                    "stage1_score": float(task["stage1_score"]),
                    "pred_label": label_names[int(pred_id)],
                    "pred_label_id": int(pred_id),
                    "pred_confidence": float(confidence),
                }
                for cls_idx, cls_name in enumerate(label_names):
                    row[f"prob_{cls_name}"] = float(prob_vec[cls_idx])
                records.append(row)

            batch_images = []
            batch_tasks = []

        for result in pool.imap_unordered(build_stage2_tensor, tasks, chunksize=1):
            processed += 1
            if result["status"] == "error":
                failed.append(
                    {
                        "chrom": result["task"]["chrom"],
                        "window_start": int(result["task"]["window_start"]),
                        "message": result["message"],
                    }
                )
            else:
                batch_images.append(result["images"])
                batch_tasks.append(result["task"])
                if len(batch_images) >= batch_size:
                    flush_batch()

            if processed == 1 or processed % 100 == 0 or processed == len(tasks):
                flush_batch()
                elapsed = time.time() - started_at
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[stage2-full] processed {processed}/{len(tasks)} abnormal windows "
                    f"({rate:.2f} windows/s, emitted={len(records)}, failed={len(failed)})"
                )

        flush_batch()

    records = sort_records(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_stage2_predictions_csv(records, output_csv, label_names)

    summary = {
        "predicted_windows": int(len(records)),
        "failed_windows": int(len(failed)),
        "label_names": list(label_names),
        "label_counts": dict(Counter(row["pred_label"] for row in records)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "failures_preview": failed[:20],
    }
    return records, summary


def get_ref_base(fasta, chrom, pos_0b):
    try:
        ref = fasta.fetch(chrom, pos_0b, pos_0b + 1)
        return ref.upper() if ref else "N"
    except Exception:
        return "N"


def export_window_predictions_to_vcf(records, fasta_path, output_vcf, sample_name):
    fasta = pysam.FastaFile(str(Path(fasta_path).expanduser().resolve()))

    header = pysam.VariantHeader()
    header.add_line("##fileformat=VCFv4.2")
    header.add_line(f"##fileDate={date.today().strftime('%Y%m%d')}")
    header.add_line("##source=SharpSV-stage1-stage2-window-classifier")
    header.add_line("##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"Predicted structural variant class\">")
    header.add_line("##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position of the 1kb candidate window\">")
    header.add_line("##INFO=<ID=SVLEN,Number=1,Type=Integer,Description=\"Length of the 1kb candidate window\">")
    header.add_line("##INFO=<ID=STAGE1SC,Number=1,Type=Float,Description=\"Stage-1 anomaly score\">")
    header.add_line("##INFO=<ID=STAGE2SC,Number=1,Type=Float,Description=\"Stage-2 predicted class confidence\">")
    header.add_line("##INFO=<ID=WINDOW,Number=1,Type=String,Description=\"Original stage-1 candidate window as start-end in 0-based half-open coordinates\">")
    for label in LABEL_NAMES:
        header.add_line(
            f"##INFO=<ID=P_{label},Number=1,Type=Float,Description=\"Stage-2 class probability for {label}\">"
        )
    header.add_line("##INFO=<ID=IMPRECISE,Number=0,Type=Flag,Description=\"Imprecise window-level structural variant prediction\">")
    header.add_meta("FORMAT", items=[("ID", "GT"), ("Number", 1), ("Type", "String"), ("Description", "Genotype")])

    for contig, length in zip(fasta.references, fasta.lengths):
        header.add_line(f"##contig=<ID={contig},length={length}>")
    header.add_sample(sample_name)

    output_vcf.parent.mkdir(parents=True, exist_ok=True)
    with pysam.VariantFile(str(output_vcf), "w", header=header) as vcf_out:
        for idx, row in enumerate(records, start=1):
            chrom = str(row["chrom"])
            start_0b = int(row["window_start"])
            end_0b = int(row["window_end"])
            label = str(row["pred_label"])
            stop_0b = max(end_0b, start_0b + 1)

            rec = vcf_out.new_record(
                contig=chrom,
                start=start_0b,
                stop=stop_0b,
                alleles=(get_ref_base(fasta, chrom, start_0b), f"<{label}>"),
            )
            rec.id = f"SharpSV.{idx}"
            rec.qual = round(float(row["pred_confidence"]) * 100.0, 4)
            rec.filter.add("PASS")
            rec.info["SVTYPE"] = label
            if label == "DEL":
                rec.info["SVLEN"] = -(end_0b - start_0b)
            else:
                rec.info["SVLEN"] = end_0b - start_0b
            rec.info["STAGE1SC"] = float(row["stage1_score"])
            rec.info["STAGE2SC"] = float(row["pred_confidence"])
            rec.info["WINDOW"] = f"{start_0b}-{end_0b}"
            rec.info["IMPRECISE"] = True
            for label_name in LABEL_NAMES:
                rec.info[f"P_{label_name}"] = float(row[f"prob_{label_name}"])
            rec.samples[sample_name]["GT"] = (0, 1)
            vcf_out.write(rec)

    fasta.close()


def main():
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    visible_ids = select_device_ids(args.device_ids)
    if visible_ids:
        print(f"[runtime] using CUDA devices: {visible_ids}")
    else:
        print("[runtime] CUDA unavailable; falling back to CPU inference for stage-1/stage-2 scoring.")

    stage1_candidates_csv = outdir / "stage1_abnormal_windows.csv"
    stage2_predictions_csv = outdir / "stage2_window_predictions.csv"
    output_vcf = outdir / "stage2_window_predictions.vcf"

    full_start = time.time()
    if stage1_candidates_csv.exists():
        stage1_rows = sort_records(load_csv_records(stage1_candidates_csv))
        stage1_summary = {
            "reused_existing_csv": True,
            "predicted_abnormal_windows": int(len(stage1_rows)),
            "stage1_threshold": float(stage1_rows[0]["stage1_threshold"]) if stage1_rows else None,
            "workdirs": [str(Path(path).resolve()) for path in args.stage1_workdirs],
        }
        print(f"[stage1-full] reusing existing abnormal-window CSV: {stage1_candidates_csv}")
    else:
        stage1_rows, stage1_summary = score_stage1_candidates(
            workdirs=args.stage1_workdirs,
            checkpoint_path=args.stage1_checkpoint,
            batch_size=args.stage1_batch_size,
            device_ids=visible_ids,
            output_csv=stage1_candidates_csv,
        )

    if stage2_predictions_csv.exists():
        stage2_rows = sort_records(load_csv_records(stage2_predictions_csv))
        stage2_summary = {
            "reused_existing_csv": True,
            "predicted_windows": int(len(stage2_rows)),
            "failed_windows": 0,
            "label_names": LABEL_NAMES,
            "label_counts": dict(Counter(row["pred_label"] for row in stage2_rows)),
        }
        print(f"[stage2-full] reusing existing stage-2 prediction CSV: {stage2_predictions_csv}")
    else:
        stage2_rows, stage2_summary = predict_stage2_candidates(
            candidates=stage1_rows,
            bam_path=str(Path(args.bam).expanduser().resolve()),
            fasta_path=str(Path(args.fasta).expanduser().resolve()),
            checkpoint_path=args.stage2_checkpoint,
            batch_size=args.stage2_batch_size,
            cpu_workers=min(max(1, int(args.cpu_workers)), available_process_count()),
            device_ids=visible_ids,
            output_csv=stage2_predictions_csv,
        )

    export_window_predictions_to_vcf(
        records=stage2_rows,
        fasta_path=args.fasta,
        output_vcf=output_vcf,
        sample_name=args.sample_name,
    )

    summary = {
        "stage1_summary": stage1_summary,
        "stage2_summary": stage2_summary,
        "gpu_ids": visible_ids,
        "gpu_names": {device_id: torch.cuda.get_device_name(device_id) for device_id in visible_ids},
        "stage1_candidates_csv": str(stage1_candidates_csv),
        "stage2_predictions_csv": str(stage2_predictions_csv),
        "vcf": str(output_vcf),
        "elapsed_seconds": time.time() - full_start,
        "elapsed_hours": (time.time() - full_start) / 3600.0,
    }
    (outdir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
