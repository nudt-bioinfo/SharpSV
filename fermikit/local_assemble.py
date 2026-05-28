import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import signal
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pysam


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_FERMIKIT_PATH = MODULE_DIR / "fermi.kit"
DEFAULT_MAG2BAM = MODULE_DIR / "fermi.kit" / "mag2bam"
DEFAULT_INPUT_CSV = MODULE_DIR.parent / "workdir" / "stage3_refined_sv_results.csv"
DEFAULT_OUTDIR = MODULE_DIR.parent / "workdir" / "stage3_assembled_regions"
ASSEMBLY_RESULTS_BASENAME = "stage3_assembly_results.json"
LEGACY_ASSEMBLY_RESULTS_BASENAME = "assemble_results.json"
ASSEMBLY_MANIFEST_BASENAME = "stage3_assembled_regions_manifest.tsv"
LEGACY_ASSEMBLY_MANIFEST_BASENAME = "assembled_regions.tsv"
ASSEMBLY_ARCHIVE_STEM = "stage3_assembled_regions_archive"
LEGACY_ASSEMBLY_ARCHIVE_BASENAME = "assembled_regions.tar.gz"
DEFAULT_THREADS = 32
DEFAULT_POOL_PROCS = 4
DEFAULT_WINDOW_EXTEND = 50
PROGRESS_HEARTBEAT_SECONDS = 30
MONITOR_POLL_SECONDS = 1
DEFAULT_MAX_FQ_MB = 256
DEFAULT_MAX_PRE_MB = 1024
TOO_LARGE_SKIP_MARKER_BASENAME = "stage3_too_large_skip.json"


def _log(message, logger):
    if logger:
        logger(message)


def _format_status_counts(status_counts):
    if not status_counts:
        return "no finished regions yet"
    return ", ".join(f"{key}={value:,}" for key, value in sorted(status_counts.items()))


def _format_bytes(num_bytes):
    try:
        value = float(num_bytes)
    except (TypeError, ValueError):
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.1f}{units[unit_idx]}"


def _render_bar(current, total, width=18):
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
    bar = _render_bar(current_value, total_value, width=18)
    message = f"[INFO] progress {ratio * 100:5.1f}% |{bar}| {int(current_value):,}/{int(total_value):,}"
    if speed is not None:
        message += f" | {float(speed):.1f}/s"
    if eta_seconds is not None and math.isfinite(float(eta_seconds)):
        eta_seconds = max(int(round(float(eta_seconds))), 0)
        message += f" | eta {eta_seconds}s"
    if extra:
        message += f" | {extra}"
    _log(message, logger)


def _limit_mb_to_bytes(limit_mb):
    try:
        limit_mb = float(limit_mb)
    except (TypeError, ValueError):
        return 0
    if limit_mb <= 0:
        return 0
    return int(limit_mb * 1024 * 1024)


def _is_valid_bam(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with pysam.AlignmentFile(str(path), "rb") as bam_handle:
            return len(bam_handle.references) > 0
    except Exception:
        return False


def _skip_marker_path(region_dir):
    return Path(region_dir) / TOO_LARGE_SKIP_MARKER_BASENAME


def _load_skip_marker(region_dir):
    marker_path = _skip_marker_path(region_dir)
    if not marker_path.exists():
        return None
    try:
        return json.loads(marker_path.read_text())
    except Exception:
        return {"reason": "invalid-skip-marker", "path": str(marker_path)}


def _write_skip_marker(region_dir, payload):
    marker_path = _skip_marker_path(region_dir)
    marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return marker_path


def _cleanup_region_heavy_artifacts(region_dir, logger=print):
    region_dir = Path(region_dir)
    patterns = [
        "*.bam",
        "*.bam.bai",
        "*.fq",
        "*.fq.gz",
        "*.fmd",
        "*.pre.gz",
        "*.mag.gz",
        "*.fa",
        "*.fasta",
        "*.unitig.fa",
        "*.ctg.fa",
    ]
    removed = 0
    for pattern in patterns:
        for artifact in region_dir.glob(pattern):
            if artifact.name == TOO_LARGE_SKIP_MARKER_BASENAME:
                continue
            try:
                artifact.unlink(missing_ok=True)
                removed += 1
            except Exception as exc:
                _log(f"[WARN] failed to remove oversized artifact {artifact}: {exc}", logger)
    if removed:
        _log(f"[WARN] removed {removed} oversized intermediate artifact(s) in {region_dir}", logger)


def _run_cmd_monitored(cmd, workdir=None, logger=print, file_size_limits=None, poll_seconds=MONITOR_POLL_SECONDS):
    _log(f"[CMD] {cmd}", logger)
    with tempfile.TemporaryFile(mode="w+") as stdout_handle, tempfile.TemporaryFile(mode="w+") as stderr_handle:
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=workdir,
            stdout=stdout_handle,
            stderr=stderr_handle,
            universal_newlines=True,
            preexec_fn=os.setsid,
        )
        try:
            while True:
                try:
                    return_code = process.wait(timeout=max(float(poll_seconds), 0.5))
                    stdout_handle.seek(0)
                    stderr_handle.seek(0)
                    return {
                        "ok": return_code == 0,
                        "returncode": return_code,
                        "stdout": stdout_handle.read(),
                        "stderr": stderr_handle.read(),
                    }
                except subprocess.TimeoutExpired:
                    pass

                if file_size_limits:
                    for watched_path, limit_bytes, label in file_size_limits:
                        watched_path = Path(watched_path)
                        if limit_bytes <= 0 or not watched_path.exists():
                            continue
                        current_size = watched_path.stat().st_size
                        if current_size > limit_bytes:
                            _log(
                                f"[WARN] stopping region build because {label} exceeded limit: "
                                f"{_format_bytes(current_size)} > {_format_bytes(limit_bytes)} ({watched_path})",
                                logger,
                            )
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                                process.wait()
                            stdout_handle.seek(0)
                            stderr_handle.seek(0)
                            return {
                                "ok": False,
                                "returncode": process.returncode,
                                "stdout": stdout_handle.read(),
                                "stderr": stderr_handle.read(),
                                "terminated_by_limit": True,
                                "watched_path": str(watched_path),
                                "limit_bytes": int(limit_bytes),
                                "current_size": int(current_size),
                                "label": label,
                            }
        finally:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception:
                    pass


def _task_region_tag(task):
    row, _valid_chroms, config = task
    raw_chrom, start, end, _pred = row[:4]
    window_extend = int(config["window_extend"])
    return f"{raw_chrom}_{max(0, int(start) - window_extend)}_{int(end) + window_extend}"


def _region_runtime_hint(region_dir):
    region_dir = Path(region_dir)
    if not region_dir.exists():
        return None

    candidates = []
    for pattern in ("*.pre.gz", "*.mag.gz", "*.flt.fq.gz", "*.ec.fq.gz", "*.fq", "*.bam"):
        candidates.extend(region_dir.glob(pattern))

    if not candidates:
        return None

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    stat = latest.stat()
    return f"active {region_dir.name}  ·  {latest.name} { _format_bytes(stat.st_size) }"


def _cleanup_invalid_bam_artifacts(region_dir, logger=print):
    region_dir = Path(region_dir)
    removed = 0
    for bam_file in region_dir.glob("*.srt.bam"):
        if _is_valid_bam(bam_file):
            continue
        try:
            bam_file.unlink(missing_ok=True)
            Path(str(bam_file) + ".bai").unlink(missing_ok=True)
            removed += 1
        except Exception as exc:
            _log(f"[WARN] failed to remove invalid BAM artifact {bam_file}: {exc}", logger)
    if removed:
        _log(f"[WARN] removed {removed} invalid local-assembly BAM artifact(s) in {region_dir}", logger)


def _region_cached_status(task):
    row, valid_chroms, config = task
    try:
        raw_chrom, start, end, pred = row[:4]
    except Exception:
        return None

    if str(pred) not in {"1", "2"}:
        return None

    window_extend = int(config["window_extend"])
    outdir = Path(config["outdir"])
    start = max(0, int(start) - window_extend)
    end = int(end) + window_extend
    region_tag = f"{raw_chrom}_{start}_{end}"
    region_dir = outdir / region_tag

    skip_marker = _load_skip_marker(region_dir)
    if skip_marker is not None:
        return "too_large_skip"

    ready_bams = [bam_file for bam_file in region_dir.glob("*.srt.bam") if _is_valid_bam(bam_file)]
    if ready_bams:
        return "bam_ready"

    mag_files = [mag_file for mag_file in region_dir.glob("*.mag.gz") if mag_file.stat().st_size > 0]
    if mag_files:
        return "assembly_cached"

    return None


def get_bam_chromosomes(bam_path, logger=print):
    _log(f"[INFO] reading BAM header from {bam_path}", logger)
    try:
        cmd = f"samtools view -H '{bam_path}'"
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
        chroms = set()
        for line in output.splitlines():
            if line.startswith("@SQ"):
                for part in line.split("\t"):
                    if part.startswith("SN:"):
                        chroms.add(part[3:])
        sample = list(chroms)[:5] if chroms else []
        _log(f"[INFO] discovered {len(chroms)} contigs in BAM header; sample={sample}", logger)
        return chroms
    except Exception as exc:
        _log(f"[ERROR] failed to read BAM header from {bam_path}: {exc}", logger)
        return set()


def run_cmd(cmd, workdir=None, logger=print):
    _log(f"[CMD] {cmd}", logger)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if "unknown reference name" in result.stderr:
            _log(f"[ERROR] samtools reported an unknown reference name:\n{result.stderr}", logger)
            return False
        return True
    except subprocess.CalledProcessError as exc:
        _log(f"[ERROR] command failed: {cmd}\n{exc.stderr}", logger)
        return False


def find_assembly_outputs(region_dir):
    patterns = ["*.unitig.fa", "*.ctg.fa", "*.mag.gz", "*.fa", "*.fasta", "*.pre.gz"]
    found = []
    for pattern in patterns:
        found.extend(region_dir.glob(pattern))
    return sorted(set(found))


def process_region(task):
    row, valid_chroms, config = task
    try:
        raw_chrom, start, end, pred = row[:4]
    except Exception:
        return {"status": "bad_input", "row": row}

    if str(pred) not in {"1", "2"}:
        return {"status": "skip_pred", "row": row}

    logger = None
    window_extend = int(config["window_extend"])
    bam_path = config["bam_path"]
    outdir = Path(config["outdir"])
    threads = int(config["threads"])
    fermikit_path = config["fermikit_path"]
    max_fq_bytes = int(config.get("max_fq_bytes", 0))
    max_pre_bytes = int(config.get("max_pre_bytes", 0))

    start = max(0, int(start) - window_extend)
    end = int(end) + window_extend

    region_tag = f"{raw_chrom}_{start}_{end}"
    region_dir = outdir / region_tag
    region_dir.mkdir(parents=True, exist_ok=True)
    skip_marker = _load_skip_marker(region_dir)
    if skip_marker is not None:
        return {
            "status": "too_large_skip",
            "region": region_tag,
            "region_dir": str(region_dir),
            "skip": skip_marker,
        }
    ready_bams = [bam_file for bam_file in region_dir.glob("*.srt.bam") if _is_valid_bam(bam_file)]
    if ready_bams:
        return {
            "status": "bam_ready",
            "region": region_tag,
            "region_dir": str(region_dir),
            "assembled": [str(path) for path in ready_bams],
        }
    if list(region_dir.glob("*.srt.bam")):
        _cleanup_invalid_bam_artifacts(region_dir, logger=logger)

    mag_files = [mag_file for mag_file in region_dir.glob("*.mag.gz") if mag_file.stat().st_size > 0]
    if mag_files:
        return {
            "status": "assembly_cached",
            "region": region_tag,
            "region_dir": str(region_dir),
            "assembled": [str(path) for path in mag_files],
        }

    bam_out = region_dir / f"{region_tag}.bam"
    fq_out = region_dir / f"{region_tag}.fq"
    mak = region_dir / f"{region_tag}.mak"

    chrom = str(raw_chrom)
    if valid_chroms and chrom not in valid_chroms:
        if f"chr{chrom}" in valid_chroms:
            chrom = f"chr{chrom}"
        elif chrom.startswith("chr") and chrom[3:] in valid_chroms:
            chrom = chrom[3:]
        elif chrom == "M" and "MT" in valid_chroms:
            chrom = "MT"
        elif chrom == "MT" and "chrM" in valid_chroms:
            chrom = "chrM"

    cmd_extract = f"samtools view -b '{bam_path}' '{chrom}:{start}-{end}' -o '{bam_out}'"
    if not run_cmd(cmd_extract, logger=logger):
        return {"status": "extract_failed", "region": region_tag}

    cmd_fq = f"samtools fastq -s /dev/stdout '{bam_out}' > '{fq_out}'"
    if not run_cmd(cmd_fq, workdir=region_dir, logger=logger):
        return {"status": "fastq_failed", "region": region_tag}

    if not fq_out.exists() or fq_out.stat().st_size < 10000:
        return {"status": "empty_reads", "region": region_tag}
    if max_fq_bytes > 0 and fq_out.stat().st_size > max_fq_bytes:
        marker_path = _write_skip_marker(
            region_dir,
            {
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "region": region_tag,
                "reason": "too_large_skip",
                "phase": "fastq",
                "file": str(fq_out),
                "size_bytes": int(fq_out.stat().st_size),
                "limit_bytes": int(max_fq_bytes),
            },
        )
        _cleanup_region_heavy_artifacts(region_dir, logger=logger)
        return {
            "status": "too_large_skip",
            "region": region_tag,
            "region_dir": str(region_dir),
            "skip_marker": str(marker_path),
            "phase": "fastq",
        }

    cmd_assemble = (
        f"'{fermikit_path}/fermi2.pl' unitig -s1000k -t{threads} -l150 "
        f"-p '{region_tag}' '{fq_out.name}' > '{mak.name}'"
    )
    if not run_cmd(cmd_assemble, workdir=region_dir, logger=logger):
        return {"status": "unitig_failed", "region": region_tag}

    monitored_make = _run_cmd_monitored(
        f"make -f '{mak.name}'",
        workdir=region_dir,
        logger=logger,
        file_size_limits=[(region_dir / f"{region_tag}.pre.gz", max_pre_bytes, "pre.gz")] if max_pre_bytes > 0 else None,
    )
    if not monitored_make["ok"]:
        if monitored_make.get("terminated_by_limit"):
            marker_path = _write_skip_marker(
                region_dir,
                {
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "region": region_tag,
                    "reason": "too_large_skip",
                    "phase": monitored_make.get("label", "assembly"),
                    "file": monitored_make.get("watched_path"),
                    "size_bytes": monitored_make.get("current_size"),
                    "limit_bytes": monitored_make.get("limit_bytes"),
                },
            )
            _cleanup_region_heavy_artifacts(region_dir, logger=logger)
            return {
                "status": "too_large_skip",
                "region": region_tag,
                "region_dir": str(region_dir),
                "skip_marker": str(marker_path),
                "phase": monitored_make.get("label", "assembly"),
            }
        assembled = [str(path) for path in find_assembly_outputs(region_dir)]
        return {"status": "make_failed", "region": region_tag, "assembled": assembled}

    assembled = [str(path) for path in find_assembly_outputs(region_dir)]
    return {
        "status": "assembled",
        "region": region_tag,
        "region_dir": str(region_dir),
        "assembled": assembled,
    }


def run_mag2bam_for_all(outdir, reference_fa, mag2bam_path=None, logger=print, progress_callback=None):
    outdir = Path(outdir)
    mag2bam = str(Path(mag2bam_path or DEFAULT_MAG2BAM).expanduser().resolve())
    mag_jobs = []

    for region_dir in sorted(outdir.iterdir()):
        if not region_dir.is_dir():
            continue
        if _skip_marker_path(region_dir).exists():
            continue
        ready_bams = [bam_file for bam_file in region_dir.glob("*.srt.bam") if _is_valid_bam(bam_file)]
        if ready_bams:
            for bam_file in ready_bams:
                bai_path = Path(str(bam_file) + ".bai")
                if not bai_path.exists():
                    run_cmd(f"samtools index '{bam_file.name}'", workdir=region_dir, logger=logger)
            continue
        if list(region_dir.glob("*.srt.bam")):
            _cleanup_invalid_bam_artifacts(region_dir, logger=logger)
        mag_files = list(region_dir.glob("*.mag.gz"))
        if not mag_files:
            continue
        mag_jobs.append((region_dir, mag_files[0]))

    if not mag_jobs:
        _log("[INFO] no MAG assembly outputs required BAM conversion", logger)
        return

    _log(f"[INFO] mag2bam conversion plan: {len(mag_jobs):,} assembled regions", logger)
    started_at = time.time()
    failures = 0
    progress_step = max(1, min(200, max(1, len(mag_jobs) // 50)))

    for done, (region_dir, mag_file) in enumerate(mag_jobs, start=1):
        _log(f"[INFO] running mag2bam for {region_dir.name}", logger)
        cmd = f"'{mag2bam}' '{reference_fa}' '{mag_file.name}' | sh"
        if not run_cmd(cmd, workdir=region_dir, logger=logger):
            _log(f"[ERROR] mag2bam failed in {region_dir}", logger)
            _cleanup_invalid_bam_artifacts(region_dir, logger=logger)
            failures += 1
        else:
            ready_bams = [bam_file for bam_file in region_dir.glob("*.srt.bam") if _is_valid_bam(bam_file)]
            if not ready_bams:
                _log(f"[ERROR] mag2bam finished but did not produce a valid BAM in {region_dir}", logger)
                _cleanup_invalid_bam_artifacts(region_dir, logger=logger)
                failures += 1
            else:
                for bam_file in ready_bams:
                    bai_path = Path(str(bam_file) + ".bai")
                    if not bai_path.exists():
                        run_cmd(f"samtools index '{bam_file.name}'", workdir=region_dir, logger=logger)

        if done == 1 or done == len(mag_jobs) or done % progress_step == 0:
            elapsed = time.time() - started_at
            rate = done / elapsed if elapsed > 0 else 0.0
            eta_seconds = (len(mag_jobs) - done) / rate if rate > 0 else float("inf")
            _emit_progress(
                progress_callback,
                logger,
                done,
                len(mag_jobs),
                speed=rate,
                eta_seconds=eta_seconds,
                extra=f"phase mag2bam  ·  failures {failures:,}",
            )

    _log("[INFO] mag2bam stage completed", logger)


def package_results(outdir, logger=print):
    outdir = Path(outdir)
    tsv_path = outdir / ASSEMBLY_MANIFEST_BASENAME
    archive_path = Path(str(outdir / ASSEMBLY_ARCHIVE_STEM) + ".tar.gz")
    with open(tsv_path, "w") as handle:
        handle.write("region\tregion_dir\tassembled_files\n")
        for region_dir in sorted(outdir.iterdir()):
            if not region_dir.is_dir():
                continue
            files = [str(path) for path in find_assembly_outputs(region_dir)]
            handle.write(f"{region_dir.name}\t{region_dir}\t{';'.join(files)}\n")

    shutil.make_archive(str(outdir / ASSEMBLY_ARCHIVE_STEM), "gztar", root_dir=str(outdir))
    _log(f"[INFO] wrote {tsv_path}", logger)
    _log(f"[INFO] wrote {archive_path}", logger)


def assemble_regions(
    input_csv=DEFAULT_INPUT_CSV,
    bam_path=None,
    reference_fa=None,
    outdir=DEFAULT_OUTDIR,
    threads=DEFAULT_THREADS,
    pool_procs=DEFAULT_POOL_PROCS,
    window_extend=DEFAULT_WINDOW_EXTEND,
    fermikit_path=DEFAULT_FERMIKIT_PATH,
    mag2bam_path=DEFAULT_MAG2BAM,
    max_fq_mb=DEFAULT_MAX_FQ_MB,
    max_pre_mb=DEFAULT_MAX_PRE_MB,
    logger=print,
    progress_callback=None,
):
    if bam_path is None:
        raise ValueError("bam_path is required")
    if reference_fa is None:
        raise ValueError("reference_fa is required")

    input_csv = str(Path(input_csv).expanduser().resolve())
    bam_path = str(Path(bam_path).expanduser().resolve())
    reference_fa = str(Path(reference_fa).expanduser().resolve())
    outdir = Path(outdir).expanduser().resolve()
    fermikit_path = str(Path(fermikit_path).expanduser().resolve())
    mag2bam_path = str(Path(mag2bam_path).expanduser().resolve())
    max_fq_bytes = _limit_mb_to_bytes(max_fq_mb)
    max_pre_bytes = _limit_mb_to_bytes(max_pre_mb)

    outdir.mkdir(parents=True, exist_ok=True)
    valid_chroms = get_bam_chromosomes(bam_path, logger=logger)
    if not valid_chroms:
        raise RuntimeError(f"Unable to obtain chromosome names from BAM header: {bam_path}")

    _log(f"[INFO] reading refined candidate intervals from {input_csv}", logger)
    with open(input_csv) as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        raise RuntimeError(f"Input CSV is empty: {input_csv}")

    delimiter = "\t" if "\t" in lines[0] else ","
    reader = csv.reader(lines, delimiter=delimiter)
    header = next(reader)
    if not header[0].lower().startswith("chrom"):
        reader = csv.reader(lines, delimiter=delimiter)

    config = {
        "bam_path": bam_path,
        "reference_fa": reference_fa,
        "outdir": str(outdir),
        "threads": int(max(1, threads)),
        "window_extend": int(max(0, window_extend)),
        "fermikit_path": fermikit_path,
        "mag2bam_path": mag2bam_path,
        "max_fq_bytes": max_fq_bytes,
        "max_pre_bytes": max_pre_bytes,
    }
    tasks = [(row, valid_chroms, config) for row in reader if len(row) >= 4]
    _log(f"[INFO] candidate regions: {len(tasks)}", logger)
    _log(
        f"[INFO] assembly execution plan: region_workers={max(1, int(pool_procs))}, "
        f"threads_per_region={int(max(1, threads))}, heartbeat={PROGRESS_HEARTBEAT_SECONDS}s, "
        f"fq_limit={_format_bytes(max_fq_bytes) if max_fq_bytes > 0 else 'off'}, "
        f"pre_limit={_format_bytes(max_pre_bytes) if max_pre_bytes > 0 else 'off'}",
        logger,
    )

    started_at = time.time()
    status_counts = {}
    if tasks:
        for task in tasks:
            cached_status = _region_cached_status(task)
            if cached_status is not None:
                status_counts[cached_status] = status_counts.get(cached_status, 0) + 1

        worker_count = max(1, int(pool_procs))
        progress_step = max(1, min(500, max(1, len(tasks) // 100)))
        with mp.Pool(worker_count) as pool:
            iterator = pool.imap_unordered(process_region, tasks, chunksize=1)
            results = []
            completed = sum(status_counts.values())
            pending_tasks = len(tasks) - completed
            while len(results) < pending_tasks:
                try:
                    item = iterator.next(timeout=PROGRESS_HEARTBEAT_SECONDS)
                except mp.TimeoutError:
                    elapsed = time.time() - started_at
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta_seconds = (len(tasks) - completed) / rate if rate > 0 else float("inf")
                    active_extra = None
                    if completed < len(tasks):
                        active_region = outdir / _task_region_tag(tasks[completed])
                        active_extra = _region_runtime_hint(active_region)
                    _emit_progress(
                        progress_callback,
                        logger,
                        completed,
                        len(tasks),
                        speed=rate,
                        eta_seconds=eta_seconds,
                        extra="  ·  ".join(
                            value
                            for value in (
                                "phase local-asm",
                                _format_status_counts(status_counts),
                                active_extra,
                            )
                            if value
                        ),
                    )
                    continue

                results.append(item)
                status = item.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                completed += 1

                if completed == 1 or completed == len(tasks) or completed % progress_step == 0:
                    elapsed = time.time() - started_at
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta_seconds = (len(tasks) - completed) / rate if rate > 0 else float("inf")
                    _emit_progress(
                        progress_callback,
                        logger,
                        completed,
                        len(tasks),
                        speed=rate,
                        eta_seconds=eta_seconds,
                        extra=f"phase local-asm  ·  {_format_status_counts(status_counts)}",
                    )
    else:
        results = []

    _log(f"[INFO] local assembly finished in {time.time() - started_at:.1f}s", logger)
    if status_counts:
        _log(f"[INFO] assembly status summary: {_format_status_counts(status_counts)}", logger)
    results_path = outdir / ASSEMBLY_RESULTS_BASENAME
    with open(results_path, "w") as handle:
        json.dump(results, handle, indent=2)
    _log(f"[INFO] wrote {results_path}", logger)

    run_mag2bam_for_all(
        outdir,
        reference_fa,
        mag2bam_path=mag2bam_path,
        logger=logger,
        progress_callback=progress_callback,
    )
    package_results(outdir, logger=logger)

    _log("[DONE] local assembly workflow finished", logger)
    return str(outdir)


def build_parser():
    parser = argparse.ArgumentParser(description="Local assembly over SharpSV refined candidate intervals")
    parser.add_argument("--input_csv", default=str(DEFAULT_INPUT_CSV), help="Stage-3 refined SV interval CSV")
    parser.add_argument("--bam", required=True, help="Raw BAM used for local extraction")
    parser.add_argument("--reference_fa", required=True, help="Reference FASTA for mag2bam")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory for stage-3 assembled regions")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="Threads per fermi2 assembly job")
    parser.add_argument("--pool_procs", type=int, default=DEFAULT_POOL_PROCS, help="Concurrent region workers")
    parser.add_argument("--window_extend", type=int, default=DEFAULT_WINDOW_EXTEND, help="Padding around each region")
    parser.add_argument("--fermikit_path", default=str(DEFAULT_FERMIKIT_PATH), help="Path to the fermi.kit directory")
    parser.add_argument("--mag2bam_path", default=str(DEFAULT_MAG2BAM), help="Path to the mag2bam executable")
    parser.add_argument("--max_fq_mb", type=float, default=DEFAULT_MAX_FQ_MB, help="Skip regions whose extracted .fq exceeds this size in MB; <=0 disables")
    parser.add_argument("--max_pre_mb", type=float, default=DEFAULT_MAX_PRE_MB, help="Skip regions whose .pre.gz exceeds this size in MB during assembly; <=0 disables")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    assemble_regions(
        input_csv=args.input_csv,
        bam_path=args.bam,
        reference_fa=args.reference_fa,
        outdir=args.outdir,
        threads=args.threads,
        pool_procs=args.pool_procs,
        window_extend=args.window_extend,
        fermikit_path=args.fermikit_path,
        mag2bam_path=args.mag2bam_path,
        max_fq_mb=args.max_fq_mb,
        max_pre_mb=args.max_pre_mb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
