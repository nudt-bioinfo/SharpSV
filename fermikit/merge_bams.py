import argparse
import math
import os
import subprocess
import time
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_REGION_DIR = MODULE_DIR.parent / "workdir" / "stage3_assembled_regions"
DEFAULT_TEMP_DIR = MODULE_DIR.parent / "workdir" / "stage3_merge_temp"
DEFAULT_FINAL_BAM = MODULE_DIR.parent / "workdir" / "stage3_assembled_regions.bam"
DEFAULT_BATCH_SIZE = 300


def _log(message, logger):
    if logger:
        logger(message)


def _render_bar(current, total, width=16):
    width = max(int(width), 8)
    try:
        current = float(current)
        total = float(total)
    except (TypeError, ValueError):
        return "░" * width
    if total <= 0:
        return "░" * width
    ratio = max(0.0, min(current / total, 1.0))
    filled = min(width, int(round(ratio * width)))
    return ("█" * filled) + ("░" * (width - filled))


def _emit_progress(progress_callback, logger, current, total, speed=None, eta_seconds=None, extra=None):
    if progress_callback is not None:
        progress_callback(current, total, speed=speed, eta_seconds=eta_seconds, extra=extra)
        return

    try:
        current_value = float(current)
        total_value = float(total)
    except (TypeError, ValueError):
        current_value = 0.0
        total_value = 0.0

    ratio = 0.0 if total_value <= 0 else max(0.0, min(current_value / total_value, 1.0))
    bar = _render_bar(current_value, total_value, width=16)
    message = f"[INFO] progress {ratio * 100:5.1f}% |{bar}| {int(current_value):,}/{int(total_value):,}"
    if speed is not None:
        message += f" | {float(speed):.1f}/s"
    if eta_seconds is not None and math.isfinite(float(eta_seconds)):
        message += f" | eta {max(int(round(float(eta_seconds))), 0)}s"
    if extra:
        message += f" | {extra}"
    _log(message, logger)


def run_cmd(cmd, logger=print):
    _log(f"[CMD] {cmd}", logger)
    subprocess.run(cmd, shell=True, check=True)


def merge_region_bams(
    region_dir=DEFAULT_REGION_DIR,
    temp_dir=DEFAULT_TEMP_DIR,
    final_bam=DEFAULT_FINAL_BAM,
    sorted_bam=None,
    batch_size=DEFAULT_BATCH_SIZE,
    samtools_threads=1,
    logger=print,
    progress_callback=None,
):
    region_dir = Path(region_dir).expanduser().resolve()
    temp_dir = Path(temp_dir).expanduser().resolve()
    final_bam = Path(final_bam).expanduser().resolve()
    if sorted_bam is None:
        sorted_bam = final_bam.with_name(final_bam.stem + ".sorted.bam")
    sorted_bam = Path(sorted_bam).expanduser().resolve()

    bam_files = sorted(region_dir.glob("*/*.srt.bam"))
    if not bam_files:
        raise FileNotFoundError(f"No assembled region BAMs were found under {region_dir}")

    _log(f"[INFO] total BAM files discovered for merge: {len(bam_files)}", logger)
    temp_dir.mkdir(parents=True, exist_ok=True)
    final_bam.parent.mkdir(parents=True, exist_ok=True)
    sorted_bam.parent.mkdir(parents=True, exist_ok=True)

    merge_threads = max(1, int(samtools_threads))
    temp_bams = []
    total_batches = max(1, math.ceil(len(bam_files) / batch_size))
    started_at = time.time()
    for index in range(0, len(bam_files), batch_size):
        batch = bam_files[index : index + batch_size]
        batch_id = index // batch_size
        batch_list = temp_dir / f"batch_{batch_id}.txt"
        batch_out = temp_dir / f"batch_{batch_id}.bam"

        with open(batch_list, "w") as handle:
            for bam_file in batch:
                handle.write(str(bam_file) + "\n")

        _log(f"[INFO] merging batch {batch_id} with {len(batch)} region BAMs", logger)
        run_cmd(f"samtools merge -@ {merge_threads} -b '{batch_list}' '{batch_out}'", logger=logger)
        temp_bams.append(batch_out)
        elapsed = time.time() - started_at
        batches_done = len(temp_bams)
        batch_rate = batches_done / elapsed if elapsed > 0 else 0.0
        eta_seconds = (total_batches - batches_done) / batch_rate if batch_rate > 0 else float("inf")
        _emit_progress(
            progress_callback,
            logger,
            batches_done,
            total_batches,
            speed=batch_rate,
            eta_seconds=eta_seconds,
            extra=f"phase batch-merge  ·  inputs {len(bam_files):,}",
        )

    all_batches_list = temp_dir / "all_batches.txt"
    with open(all_batches_list, "w") as handle:
        for bam_file in temp_bams:
            handle.write(str(bam_file) + "\n")

    _log(f"[INFO] merging {len(temp_bams)} batch BAMs into {final_bam}", logger)
    run_cmd(f"samtools merge -@ {merge_threads} -b '{all_batches_list}' '{final_bam}'", logger=logger)
    _log(f"[INFO] sorting merged BAM into {sorted_bam.name}", logger)
    run_cmd(f"samtools sort -@ {merge_threads} '{final_bam}' -o '{sorted_bam}'", logger=logger)
    _log(f"[INFO] indexing merged BAM {sorted_bam.name}", logger)
    run_cmd(f"samtools index '{sorted_bam}'", logger=logger)

    _log(f"[DONE] merged BAM ready: {sorted_bam}", logger)
    return str(sorted_bam)


def build_parser():
    parser = argparse.ArgumentParser(description="Merge per-region assembly BAMs into one indexed BAM")
    parser.add_argument("--region_dir", default=str(DEFAULT_REGION_DIR), help="Stage-3 assembled-region directory containing per-region .srt.bam files")
    parser.add_argument("--temp_dir", default=str(DEFAULT_TEMP_DIR), help="Stage-3 temporary directory for batch merge lists and BAMs")
    parser.add_argument("--final_bam", default=str(DEFAULT_FINAL_BAM), help="Stage-3 intermediate merged BAM path before sorting")
    parser.add_argument("--sorted_bam", default=None, help="Stage-3 final sorted merged BAM path")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="How many BAMs to merge per batch")
    parser.add_argument("--samtools_threads", type=int, default=1, help="Threads passed to samtools merge/sort")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    merge_region_bams(
        region_dir=args.region_dir,
        temp_dir=args.temp_dir,
        final_bam=args.final_bam,
        sorted_bam=args.sorted_bam,
        batch_size=args.batch_size,
        samtools_threads=args.samtools_threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
