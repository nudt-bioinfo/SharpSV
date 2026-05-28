import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import pysam

from .assembly_validator import validate_assembly_candidates
from fermikit.local_assemble import (
    ASSEMBLY_ARCHIVE_STEM,
    ASSEMBLY_MANIFEST_BASENAME,
    ASSEMBLY_RESULTS_BASENAME,
    LEGACY_ASSEMBLY_ARCHIVE_BASENAME,
    LEGACY_ASSEMBLY_MANIFEST_BASENAME,
    LEGACY_ASSEMBLY_RESULTS_BASENAME,
    assemble_regions,
    run_mag2bam_for_all,
)
from fermikit.merge_bams import merge_region_bams
from ..utils.console import emit, emit_banner, emit_progress, format_duration
from .sort_predictions import refine_predictions_csv


STAGE2_PREDICTIONS_BASENAME = "stage2_predictions.csv"
LEGACY_STAGE2_PREDICTIONS_BASENAME = "predictions.csv"
REFINED_RESULTS_BASENAME = "stage3_refined_sv_results.csv"
LEGACY_REFINED_RESULTS_BASENAME = "refined_sv_results.csv"
ASSEMBLY_DIRNAME = "stage3_assembled_regions"
LEGACY_ASSEMBLY_DIRNAME = "fermikit_assembled_regions"
MERGE_TEMP_DIRNAME = "stage3_merge_temp"
LEGACY_MERGE_TEMP_DIRNAME = "merge_temp"
MERGED_BAM_BASENAME = "stage3_assembled_regions.bam"
LEGACY_MERGED_BAM_BASENAME = "merged_all_regions.bam"
MERGED_SORTED_BAM_BASENAME = "stage3_assembled_regions.sorted.bam"
LEGACY_MERGED_SORTED_BAM_BASENAME = "merged_all_regions.sorted.bam"
STAGE3_SORT_MARKER = "stage3_sort.complete.json"
STAGE3_ASSEMBLY_MARKER = "stage3_assemble.complete.json"
STAGE3_MERGE_MARKER = "stage3_merge.complete.json"
STAGE3_VALIDATE_MARKER = "stage3_validate.complete.json"


@dataclass(frozen=True)
class Stage3Paths:
    workdir: Path
    predictions_csv: Path
    refined_csv: Path
    assembly_outdir: Path
    merge_temp_dir: Path
    merged_bam: Path
    merged_sorted_bam: Path
    final_output: Path
    sort_marker: Path
    assembly_marker: Path
    merge_marker: Path
    validate_marker: Path

    @property
    def merged_sorted_bai(self):
        return Path(str(self.merged_sorted_bam) + ".bai")


def build_stage3_paths(workdir, final_output):
    workdir_path = Path(workdir).expanduser().resolve()
    final_output_path = Path(final_output).expanduser().resolve()
    return Stage3Paths(
        workdir=workdir_path,
        predictions_csv=workdir_path / STAGE2_PREDICTIONS_BASENAME,
        refined_csv=workdir_path / REFINED_RESULTS_BASENAME,
        assembly_outdir=workdir_path / ASSEMBLY_DIRNAME,
        merge_temp_dir=workdir_path / MERGE_TEMP_DIRNAME,
        merged_bam=workdir_path / MERGED_BAM_BASENAME,
        merged_sorted_bam=workdir_path / MERGED_SORTED_BAM_BASENAME,
        final_output=final_output_path,
        sort_marker=workdir_path / STAGE3_SORT_MARKER,
        assembly_marker=workdir_path / STAGE3_ASSEMBLY_MARKER,
        merge_marker=workdir_path / STAGE3_MERGE_MARKER,
        validate_marker=workdir_path / STAGE3_VALIDATE_MARKER,
    )


def _write_marker(marker_path, payload):
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(marker_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return marker_path


def _marker_is_complete(marker_path, required_paths):
    return marker_path.exists() and all(Path(path).exists() for path in required_paths)


def _is_valid_bam(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with pysam.AlignmentFile(str(path), "rb") as bam_handle:
            return len(bam_handle.references) > 0
    except Exception:
        return False


def _has_assembled_region_outputs(assembly_outdir):
    assembly_outdir = Path(assembly_outdir)
    if not assembly_outdir.exists() or not assembly_outdir.is_dir():
        return False
    for region_dir in assembly_outdir.iterdir():
        if not region_dir.is_dir():
            continue
        if any(region_dir.glob("*.mag.gz")):
            return True
        if any(_is_valid_bam(bam_path) for bam_path in region_dir.glob("*.srt.bam")):
            return True
    return False


def _assembly_state_ready(paths):
    return _marker_is_complete(paths.assembly_marker, [paths.assembly_outdir])


def _merge_state_ready(paths):
    return _is_valid_bam(paths.merged_sorted_bam) and paths.merged_sorted_bai.exists()


def _maybe_move_legacy_path(legacy_path, canonical_path):
    legacy_path = Path(legacy_path)
    canonical_path = Path(canonical_path)
    if canonical_path.exists() or not legacy_path.exists():
        return False
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy_path), str(canonical_path))
    return True


def _migrate_legacy_stage3_artifacts(paths):
    migrated = []
    mappings = [
        (paths.workdir / LEGACY_REFINED_RESULTS_BASENAME, paths.refined_csv),
        (paths.workdir / LEGACY_ASSEMBLY_DIRNAME, paths.assembly_outdir),
        (paths.workdir / LEGACY_MERGE_TEMP_DIRNAME, paths.merge_temp_dir),
        (paths.workdir / LEGACY_MERGED_BAM_BASENAME, paths.merged_bam),
        (paths.workdir / LEGACY_MERGED_SORTED_BAM_BASENAME, paths.merged_sorted_bam),
        (
            Path(str(paths.workdir / LEGACY_MERGED_SORTED_BAM_BASENAME) + ".bai"),
            paths.merged_sorted_bai,
        ),
    ]
    for legacy_path, canonical_path in mappings:
        if _maybe_move_legacy_path(legacy_path, canonical_path):
            migrated.append((legacy_path, canonical_path))

    assembly_internal_mappings = [
        (paths.assembly_outdir / LEGACY_ASSEMBLY_RESULTS_BASENAME, paths.assembly_outdir / ASSEMBLY_RESULTS_BASENAME),
        (paths.assembly_outdir / LEGACY_ASSEMBLY_MANIFEST_BASENAME, paths.assembly_outdir / ASSEMBLY_MANIFEST_BASENAME),
        (
            paths.assembly_outdir / LEGACY_ASSEMBLY_ARCHIVE_BASENAME,
            paths.assembly_outdir / f"{ASSEMBLY_ARCHIVE_STEM}.tar.gz",
        ),
    ]
    for legacy_path, canonical_path in assembly_internal_mappings:
        if _maybe_move_legacy_path(legacy_path, canonical_path):
            migrated.append((legacy_path, canonical_path))
    return migrated


def _csv_header(path):
    try:
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            return next(reader, [])
    except Exception:
        return []


def _csv_has_rows(path):
    try:
        with open(path, newline="") as handle:
            next(handle, None)
            return next(handle, None) is not None
    except Exception:
        return False


def _is_stage2_predictions_csv(path):
    if not Path(path).exists():
        return False
    header = {column.strip() for column in _csv_header(path)}
    return {"chrom", "start", "end", "pred_sequence"}.issubset(header)


def locate_existing_predictions_csv(paths, explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    candidates.extend(
        [
            paths.predictions_csv,
            paths.workdir / LEGACY_STAGE2_PREDICTIONS_BASENAME,
            Path.cwd() / STAGE2_PREDICTIONS_BASENAME,
            Path.cwd() / LEGACY_STAGE2_PREDICTIONS_BASENAME,
            paths.final_output,
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate).expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_stage2_predictions_csv(candidate):
            return candidate
    return None


def ensure_predictions_artifact(paths, explicit_path=None):
    source_path = locate_existing_predictions_csv(paths, explicit_path=explicit_path)
    if source_path is None:
        return None

    paths.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    if source_path != paths.predictions_csv:
        shutil.copy2(source_path, paths.predictions_csv)
        emit("stage-3", f"copied reusable stage-2 predictions from {source_path} to {paths.predictions_csv}")
    return paths.predictions_csv


def _derive_assembly_parallelism(processes, requested_workers=None, requested_threads=None):
    processes = max(int(processes or 1), 1)

    if requested_workers is not None and requested_workers > 0:
        workers = max(1, int(requested_workers))
        threads = max(1, int(requested_threads)) if requested_threads is not None and requested_threads > 0 else max(1, processes // workers)
        return workers, threads

    if requested_threads is not None and requested_threads > 0:
        threads = max(1, int(requested_threads))
        workers = max(1, processes // threads)
        return workers, threads

    if processes >= 32:
        workers = 4
    elif processes >= 24:
        workers = 3
    elif processes >= 12:
        workers = 2
    else:
        workers = 1

    threads = max(1, processes // workers)
    return workers, threads


def _write_empty_final_output(refined_csv, final_output):
    refined_csv = Path(refined_csv).expanduser().resolve()
    final_output = Path(final_output).expanduser().resolve()
    final_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(refined_csv, final_output)
    return str(final_output)


def inspect_stage3_state(workdir, final_output, explicit_predictions_path=None):
    paths = build_stage3_paths(workdir, final_output)
    _migrate_legacy_stage3_artifacts(paths)
    predictions_source = locate_existing_predictions_csv(paths, explicit_path=explicit_predictions_path)
    return {
        "paths": paths,
        "predictions_source": predictions_source,
        "predictions_ready": predictions_source is not None,
        "sort_complete": _marker_is_complete(paths.sort_marker, [paths.refined_csv]),
        "assembly_complete": _assembly_state_ready(paths),
        "assembly_partial": _has_assembled_region_outputs(paths.assembly_outdir),
        "merge_complete": _merge_state_ready(paths),
        "validate_complete": _marker_is_complete(paths.validate_marker, [paths.final_output]),
    }


def run_stage3(
    predictions_csv,
    bam_path,
    fasta_path,
    workdir,
    output_path,
    processes=None,
    max_fq_mb=None,
    max_pre_mb=None,
    region_workers=None,
    threads_per_region=None,
):
    started_at = time.time()
    processes = max(int(processes or 1), 1)
    paths = build_stage3_paths(workdir, output_path)
    migrated = _migrate_legacy_stage3_artifacts(paths)
    for legacy_path, canonical_path in migrated:
        emit("stage-3", f"migrated legacy stage-3 artifact {legacy_path.name} -> {canonical_path.name}")
    predictions_path = ensure_predictions_artifact(paths, explicit_path=predictions_csv)
    if predictions_path is None:
        raise FileNotFoundError(
            "No reusable stage-2 prediction CSV was found for stage-3 "
            f"under {paths.workdir} or the current directory. "
            f"Expected {STAGE2_PREDICTIONS_BASENAME} "
            f"(legacy {LEGACY_STAGE2_PREDICTIONS_BASENAME} is also supported)."
        )

    assembly_pool_procs, assembly_threads_per_region = _derive_assembly_parallelism(
        processes,
        requested_workers=region_workers,
        requested_threads=threads_per_region,
    )
    emit_banner(
        "Stage-3 Assembly Validation",
        details=[
            ("predictions", predictions_path),
            ("refined intervals", paths.refined_csv),
            ("assembled regions", paths.assembly_outdir),
            ("merge cache", paths.merge_temp_dir),
            ("merged asm bam", paths.merged_sorted_bam),
            ("final output", paths.final_output),
            ("threads/region", assembly_threads_per_region),
            ("region workers", assembly_pool_procs),
            ("fq limit", f"{max_fq_mb} MB" if max_fq_mb and max_fq_mb > 0 else "off"),
            ("pre limit", f"{max_pre_mb} MB" if max_pre_mb and max_pre_mb > 0 else "off"),
        ],
    )

    if _marker_is_complete(paths.validate_marker, [paths.final_output]):
        emit("stage-3", f"reusing existing final stage-3 output: {paths.final_output}")
        return str(paths.final_output)

    def _emit_stage3_assemble_progress(current, total, speed=None, eta_seconds=None, extra=None):
        emit_progress(
            "stage-3/assemble",
            current,
            total,
            speed=speed,
            eta_seconds=eta_seconds,
            extra=extra,
            width=18,
        )

    def _emit_stage3_merge_progress(current, total, speed=None, eta_seconds=None, extra=None):
        emit_progress(
            "stage-3/merge",
            current,
            total,
            speed=speed,
            eta_seconds=eta_seconds,
            extra=extra,
            width=16,
        )

    def _emit_stage3_validate_progress(current, total, speed=None, eta_seconds=None, extra=None):
        emit_progress(
            "stage-3/validate",
            current,
            total,
            speed=speed,
            eta_seconds=eta_seconds,
            extra=extra,
            width=18,
        )

    if _marker_is_complete(paths.sort_marker, [paths.refined_csv]):
        emit("stage-3/sort", f"reusing decoded interval CSV: {paths.refined_csv}")
    else:
        emit("stage-3/sort", "decoding stage-2 predictions into refined SV intervals")
        refine_predictions_csv(
            input_csv=predictions_path,
            output_csv=paths.refined_csv,
            logger=lambda message: emit("stage-3/sort", message),
        )
        _write_marker(
            paths.sort_marker,
            {
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_csv": str(predictions_path),
                "output_csv": str(paths.refined_csv),
            },
        )

    if not _csv_has_rows(paths.refined_csv):
        emit("stage-3", "no refined SV intervals remained after stage-3 sorting; writing an empty final output")
        _write_empty_final_output(paths.refined_csv, paths.final_output)
        _write_marker(
            paths.validate_marker,
            {
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "empty-refined-input",
                "output_csv": str(paths.final_output),
            },
        )
        return str(paths.final_output)

    if _assembly_state_ready(paths):
        emit("stage-3/assemble", f"reusing existing local-assembly directory: {paths.assembly_outdir}")
    else:
        if _has_assembled_region_outputs(paths.assembly_outdir):
            emit("stage-3/assemble", f"resuming partial local assembly in {paths.assembly_outdir}")
        else:
            emit("stage-3/assemble", "launching local assembly over refined candidate intervals")
        assemble_regions(
            input_csv=paths.refined_csv,
            bam_path=bam_path,
            reference_fa=fasta_path,
            outdir=paths.assembly_outdir,
            threads=assembly_threads_per_region,
            pool_procs=assembly_pool_procs,
            max_fq_mb=max_fq_mb,
            max_pre_mb=max_pre_mb,
            logger=lambda message: emit("stage-3/assemble", message),
            progress_callback=_emit_stage3_assemble_progress,
        )
        _write_marker(
            paths.assembly_marker,
            {
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_csv": str(paths.refined_csv),
                "assembly_outdir": str(paths.assembly_outdir),
                "threads_per_region": assembly_threads_per_region,
                "pool_procs": assembly_pool_procs,
            },
        )

    if _merge_state_ready(paths):
        emit("stage-3/merge", f"reusing merged assembly BAM: {paths.merged_sorted_bam}")
    else:
        emit("stage-3/merge", "merging per-region assembly BAMs into one indexed BAM")
        merge_region_bams(
            region_dir=paths.assembly_outdir,
            temp_dir=paths.merge_temp_dir,
            final_bam=paths.merged_bam,
            sorted_bam=paths.merged_sorted_bam,
            samtools_threads=min(processes, 8),
            logger=lambda message: emit("stage-3/merge", message),
            progress_callback=_emit_stage3_merge_progress,
        )
        _write_marker(
            paths.merge_marker,
            {
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "assembly_outdir": str(paths.assembly_outdir),
                "merged_sorted_bam": str(paths.merged_sorted_bam),
            },
        )

    if not _is_valid_bam(paths.merged_sorted_bam):
        raise RuntimeError(
            f"Stage-3 merged assembly BAM is invalid or missing sequence dictionary: {paths.merged_sorted_bam}"
        )

    emit("stage-3/validate", "running adaptive assembly-backed validation against the raw BAM")
    validate_assembly_candidates(
        csv_path=paths.refined_csv,
        raw_bam_path=bam_path,
        asm_bam_path=paths.merged_sorted_bam,
        output_path=paths.final_output,
        logger=lambda message: emit("stage-3/validate", message),
        progress_callback=_emit_stage3_validate_progress,
    )
    _write_marker(
        paths.validate_marker,
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_csv": str(paths.refined_csv),
            "raw_bam": str(Path(bam_path).expanduser().resolve()),
            "asm_bam": str(paths.merged_sorted_bam),
            "output_csv": str(paths.final_output),
        },
    )
    emit("stage-3", f"stage-3 completed in {format_duration(time.time() - started_at)}")
    return str(paths.final_output)
