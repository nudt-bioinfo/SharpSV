import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..bundled_models import resolve_bundled_model_ref, validate_bundled_model
from ..stage1.features import available_worker_count, baseinfo_main, inspect_stage1_workdir, write_stage1_completion_marker
from ..stage1.predict import predict_workdir
from ..stage2.refine import refine_intermediate_csv
from ..stage3.pipeline import inspect_stage3_state, run_stage3
from ..stage4.pipeline import inspect_stage4_state, run_stage4
from ..utils.console import emit, emit_banner, format_duration


def build_parser():
    parser = argparse.ArgumentParser(description="Run the full SharpSV pipeline from BAM to the final SharpSV VCF")
    parser.add_argument(
        "-bamfilepath",
        "--bamfilepath",
        "--bam_path",
        dest="bamfilepath",
        required=True,
        help="Input sorted and indexed BAM file",
    )
    parser.add_argument(
        "-fastapath",
        "--fastapath",
        dest="fasta_path",
        required=True,
        help="Reference genome FASTA path",
    )
    parser.add_argument(
        "-fasta_path",
        "--fasta_path",
        dest="fasta_path",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stage1-model",
        dest="stage1_model_path",
        default=None,
        help="Optional override for the bundled stage-1 checkpoint.",
    )
    parser.add_argument(
        "--stage2-model",
        dest="stage2_model_path",
        default=None,
        help="Optional override for the bundled stage-2 checkpoint.",
    )
    parser.add_argument(
        "--pipeline-profile",
        "--profile",
        choices=["release", "simulated-5class"],
        default="release",
        help=(
            "Pipeline profile to run. 'release' keeps the original real-data SharpSV flow. "
            "'simulated-5class' adds the simulated-data five-class flow with direct breakpoint refinement."
        ),
    )
    parser.add_argument(
        "-checkpointpath1",
        "--checkpointpath1",
        dest="stage1_model_path",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-checkpointpath2",
        "--checkpointpath2",
        dest="stage2_model_path",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("-output", "--output", default="SharpSV.vcf", help="Final SharpSV VCF path")
    parser.add_argument(
        "-workdir",
        "--workdir",
        default=None,
        help="Optional intermediate workdir for NPZ files and stage outputs. Defaults to <output>.workdir",
    )
    parser.add_argument(
        "-processes",
        "--processes",
        type=int,
        default=None,
        help="Worker process count for the pipeline. Defaults to all available CPUs.",
    )
    parser.add_argument(
        "--stage1_batchsize",
        type=int,
        default=32,
        help="Stage-1 inference batch size. If left at 32 on GPU, SharpSV auto-increases it for better throughput.",
    )
    parser.add_argument(
        "--force_regenerate_npz",
        action="store_true",
        help="Ignore existing stage-1 NPZ outputs in workdir and regenerate them from BAM.",
    )
    parser.add_argument(
        "--stage3_max_fq_mb",
        type=float,
        default=256,
        help="Skip stage-3 regions whose extracted local-assembly .fq exceeds this size in MB. Use <=0 to disable.",
    )
    parser.add_argument(
        "--stage3_max_pre_mb",
        type=float,
        default=1024,
        help="Skip stage-3 regions whose assembly .pre.gz exceeds this size in MB. Use <=0 to disable.",
    )
    parser.add_argument(
        "--stage3_region_workers",
        type=int,
        default=None,
        help="Concurrent stage-3 local-assembly region workers. Defaults to an auto-tuned value based on -processes.",
    )
    parser.add_argument(
        "--stage3_threads_per_region",
        type=int,
        default=None,
        help="CPU threads per stage-3 local-assembly region. Defaults to an auto-tuned value based on -processes.",
    )
    return parser


def resolve_workdir(output_path, explicit_workdir=None):
    if explicit_workdir:
        return str(Path(explicit_workdir).expanduser().resolve())
    output_path = Path(output_path).expanduser().resolve()
    return str(output_path.parent / f"{output_path.stem}.workdir")


def resolve_bundled_checkpoint(explicit_path, stage_label):
    if explicit_path:
        checkpoint_path = Path(explicit_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{stage_label} checkpoint override not found: {checkpoint_path}")
        return str(checkpoint_path), "override"

    try:
        validate_bundled_model(stage_label)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Bundled {stage_label} checkpoint not found: {exc}. "
            f"Upload the GitHub Release asset or pass an override with "
            f"{'--stage1-model' if stage_label == 'stage-1' else '--stage2-model'}."
        )
    return resolve_bundled_model_ref(stage_label), "bundled"


def resolve_local_checkpoint(explicit_path, default_path, stage_label):
    if explicit_path:
        checkpoint_path = Path(explicit_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{stage_label} checkpoint override not found: {checkpoint_path}")
        return str(checkpoint_path), "override"

    default_path = Path(default_path).expanduser().resolve()
    if not default_path.exists():
        raise FileNotFoundError(
            f"Default {stage_label} checkpoint not found: {default_path}. "
            "Pass an explicit override with --stage1-model or --stage2-model."
        )
    return str(default_path), "local-default"


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _repo_script_path(script_name):
    script_path = _repo_root() / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(
            f"Required helper script not found: {script_path}. "
            "The simulated-5class profile currently requires a full source checkout."
        )
    return script_path


def run_repo_script(script_name, script_args, stage_label):
    script_path = _repo_script_path(script_name)
    command = [sys.executable, str(script_path), *[str(arg) for arg in script_args]]
    emit(stage_label, f"launching helper: {' '.join(command)}")
    subprocess.run(command, check=True)


def run_simulated_fiveclass_pipeline(args, process_count, workdir, output_path, started_at):
    repo_root = _repo_root()
    stage1_checkpoint, stage1_checkpoint_source = resolve_local_checkpoint(
        args.stage1_model_path,
        repo_root / "simulated_models" / "stage1_simulated_5class_compact.pt",
        "simulated stage-1",
    )
    stage2_checkpoint, stage2_checkpoint_source = resolve_local_checkpoint(
        args.stage2_model_path,
        repo_root / "simulated_models" / "stage2_simulated_5class_compact_fp16.pt",
        "simulated stage-2",
    )

    workdir_path = Path(workdir).expanduser().resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_path).expanduser().resolve())
    stage2_export_dir = workdir_path / "sim5_stage2_export"
    refine_dir = workdir_path / "sim5_breakpoint_refined"
    stage1_output = stage2_export_dir / "stage1_abnormal_windows.csv"
    stage2_output = stage2_export_dir / "stage2_window_predictions.csv"
    refined_all_vcf = refine_dir / "refined_events_all.vcf"
    refined_pass_vcf = refine_dir / "refined_events_pass.vcf"
    refined_csv = refine_dir / "refined_events.csv"
    summary_path = workdir_path / "sim5_pipeline_summary.json"

    emit_banner(
        "Structural Variant Discovery Pipeline",
        details=[
            ("started", time.strftime("%Y-%m-%d %H:%M:%S")),
            ("profile", "simulated-5class"),
            ("workdir", str(workdir_path)),
            ("stage-1 windows", str(stage1_output)),
            ("stage-2 output", str(stage2_output)),
            ("refined csv", str(refined_csv)),
            ("refined all vcf", str(refined_all_vcf)),
            ("final output", output_path),
            (
                "stage-1 model",
                stage1_checkpoint if stage1_checkpoint_source == "override" else "local://simulated_models/stage1_simulated_5class_compact.pt",
            ),
            (
                "stage-2 model",
                stage2_checkpoint if stage2_checkpoint_source == "override" else "local://simulated_models/stage2_simulated_5class_compact_fp16.pt",
            ),
            ("cpu workers", process_count),
        ],
    )
    emit("pipeline", "simulated-5class profile selected; the original release profile and real-data three-class workflow remain unchanged")
    emit("pipeline", "the simulated profile skips the original stage-3 local assembly and stage-4 DEL realignment")
    emit("pipeline", f"custom stage-1 candidate CSV will be written to {stage1_output}")
    emit("pipeline", f"custom stage-2 five-class CSV will be written to {stage2_output}")
    emit("pipeline", f"direct breakpoint refinement CSV will be written to {refined_csv}")

    stage1_status = inspect_stage1_workdir(workdir)
    if args.force_regenerate_npz:
        emit("stage-1/features", "forced regeneration enabled; existing NPZ outputs will be replaced")
        reuse_stage1_npz = False
    else:
        reuse_stage1_npz = stage1_status["reusable"]

    if reuse_stage1_npz:
        emit(
            "stage-1/features",
            "reusing existing NPZ feature corpus: "
            f"{stage1_status['root_npz_count']} files ({stage1_status['reason']})",
        )
        if not stage1_status["marker_exists"]:
            marker_path = write_stage1_completion_marker(workdir)
            emit("stage-1/features", f"wrote missing completion marker: {marker_path}")
    else:
        if stage1_status["root_npz_count"] > 0:
            emit(
                "stage-1/features",
                "existing NPZ outputs will be regenerated because "
                f"{stage1_status['reason']}",
            )
        else:
            emit("stage-1/features", "no reusable NPZ outputs found; generating feature corpus from BAM")
        baseinfo_main(
            bamfilepath=args.bamfilepath,
            workdir=workdir,
            max_worker=process_count,
        )

    emit("pipeline", "custom stage-1 + five-class stage-2 full-window inference begins")
    run_repo_script(
        "export_stage2_genomewide_vcf.py",
        [
            "--stage1-workdirs",
            str(workdir_path),
            "--stage1-checkpoint",
            stage1_checkpoint,
            "--stage2-checkpoint",
            stage2_checkpoint,
            "--bam",
            args.bamfilepath,
            "--fasta",
            args.fasta_path,
            "--outdir",
            str(stage2_export_dir),
            "--stage1-batch-size",
            str(args.stage1_batchsize),
            "--cpu-workers",
            str(process_count),
        ],
        stage_label="stage-2",
    )

    emit("pipeline", "direct breakpoint refinement begins")
    run_repo_script(
        "refine_stage2_window_predictions_to_precise_vcf.py",
        [
            "--predictions",
            str(stage2_output),
            "--bam",
            args.bamfilepath,
            "--fasta",
            args.fasta_path,
            "--outdir",
            str(refine_dir),
            "--workers",
            str(process_count),
            "--sample-name",
            "SharpSV",
        ],
        stage_label="stage-4",
    )

    if not refined_pass_vcf.exists():
        raise FileNotFoundError(f"Expected refined PASS VCF was not produced: {refined_pass_vcf}")

    final_output_path = Path(output_path).expanduser().resolve()
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    if refined_pass_vcf.resolve() != final_output_path:
        shutil.copy2(refined_pass_vcf, final_output_path)
        emit("pipeline", f"copied final PASS VCF to {final_output_path}")
    else:
        emit("pipeline", f"final PASS VCF already at requested output path: {final_output_path}")

    summary_payload = {
        "profile": "simulated-5class",
        "bam": str(Path(args.bamfilepath).expanduser().resolve()),
        "fasta": str(Path(args.fasta_path).expanduser().resolve()),
        "workdir": str(workdir_path),
        "stage1_checkpoint": stage1_checkpoint,
        "stage2_checkpoint": stage2_checkpoint,
        "stage2_export_dir": str(stage2_export_dir),
        "refine_dir": str(refine_dir),
        "stage1_candidates_csv": str(stage1_output),
        "stage2_predictions_csv": str(stage2_output),
        "refined_csv": str(refined_csv),
        "refined_all_vcf": str(refined_all_vcf),
        "refined_pass_vcf": str(refined_pass_vcf),
        "final_output": str(final_output_path),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    emit("pipeline", f"wrote simulated-5class pipeline summary to {summary_path}")
    emit("pipeline", f"pipeline completed in {format_duration(time.time() - started_at)}")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)

    started_at = time.time()
    process_count = args.processes or available_worker_count()
    workdir = resolve_workdir(args.output, args.workdir)
    output_path = str(Path(args.output).expanduser().resolve())

    if args.pipeline_profile == "simulated-5class":
        return run_simulated_fiveclass_pipeline(args, process_count, workdir, output_path, started_at)

    stage1_checkpoint, stage1_checkpoint_source = resolve_bundled_checkpoint(
        args.stage1_model_path,
        "stage-1",
    )
    stage2_checkpoint, stage2_checkpoint_source = resolve_bundled_checkpoint(
        args.stage2_model_path,
        "stage-2",
    )
    stage3_final_csv = str(Path.cwd() / "final_adaptive_validated.csv")
    stage1_output = str(Path(workdir) / "stage1_candidates.csv")
    stage2_output = str(Path(workdir) / "stage2_predictions.csv")

    emit_banner(
        "Structural Variant Discovery Pipeline",
        details=[
            ("started", time.strftime("%Y-%m-%d %H:%M:%S")),
            ("workdir", workdir),
            ("stage-2 output", stage2_output),
            ("stage-3 csv", stage3_final_csv),
            ("final output", output_path),
            ("stage-1 model", stage1_checkpoint if stage1_checkpoint_source == "override" else "bundled://stage1"),
            ("stage-2 model", stage2_checkpoint if stage2_checkpoint_source == "override" else "bundled://stage2"),
            ("cpu workers", process_count),
        ],
    )
    emit("pipeline", f"stage-1 candidate CSV will be written to {stage1_output}")
    emit("pipeline", f"stage-2 prediction artifact will be written to {stage2_output}")

    stage4_status = inspect_stage4_state(workdir, output_path)
    if stage4_status["stage5_complete"]:
        emit("pipeline", f"existing final SharpSV VCF detected; pipeline reuses {output_path}")
        emit("pipeline", f"pipeline completed in {format_duration(time.time() - started_at)}")
        return 0

    stage3_status = inspect_stage3_state(workdir, stage3_final_csv)

    if stage4_status["stage3_csv_ready"]:
        emit(
            "pipeline",
            f"reusable stage-3 validated CSV detected at {stage4_status['stage3_csv']}; "
            "stage-1 through stage-3 will be skipped",
        )
    elif stage3_status["validate_complete"]:
        emit("pipeline", f"existing stage-3 validated CSV detected; proceeding to VCF finalization")

    if not stage4_status["stage3_csv_ready"] and stage3_status["predictions_ready"]:
        emit(
            "pipeline",
            f"reusable stage-2 prediction artifact detected at {stage3_status['predictions_source']}; "
            "stage-1 and stage-2 will be skipped",
        )
    elif not stage4_status["stage3_csv_ready"]:
        stage1_status = inspect_stage1_workdir(workdir)
        if args.force_regenerate_npz:
            emit("stage-1/features", "forced regeneration enabled; existing NPZ outputs will be replaced")
            reuse_stage1_npz = False
        else:
            reuse_stage1_npz = stage1_status["reusable"]

        if reuse_stage1_npz:
            emit(
                "stage-1/features",
                "reusing existing NPZ feature corpus: "
                f"{stage1_status['root_npz_count']} files ({stage1_status['reason']})",
            )
            if not stage1_status["marker_exists"]:
                marker_path = write_stage1_completion_marker(workdir)
                emit("stage-1/features", f"wrote missing completion marker: {marker_path}")
        else:
            if stage1_status["root_npz_count"] > 0:
                emit(
                    "stage-1/features",
                    "existing NPZ outputs will be regenerated because "
                    f"{stage1_status['reason']}",
                )
            else:
                emit("stage-1/features", "no reusable NPZ outputs found; generating feature corpus from BAM")

            baseinfo_main(
                bamfilepath=args.bamfilepath,
                workdir=workdir,
                max_worker=process_count,
            )

        emit("pipeline", "stage-1 scoring begins")
        predict_workdir(
            workdir=workdir,
            checkpoint_path=stage1_checkpoint,
            output_path=stage1_output,
            batch_size=args.stage1_batchsize,
        )
        emit("pipeline", "stage-2 refinement begins")
        refine_intermediate_csv(
            fasta_path=args.fasta_path,
            bam_path=args.bamfilepath,
            input_csv_path=stage1_output,
            checkpoint_path=stage2_checkpoint,
            output_path=stage2_output,
            processes=process_count,
        )

    if not stage4_status["stage3_csv_ready"]:
        emit("pipeline", "stage-3 assembly-backed validation begins")
        run_stage3(
            predictions_csv=stage3_status["predictions_source"] or stage2_output,
            bam_path=args.bamfilepath,
            fasta_path=args.fasta_path,
            workdir=workdir,
            output_path=stage3_final_csv,
            processes=process_count,
            max_fq_mb=args.stage3_max_fq_mb,
            max_pre_mb=args.stage3_max_pre_mb,
            region_workers=args.stage3_region_workers,
            threads_per_region=args.stage3_threads_per_region,
        )

    emit("pipeline", "stage-4 VCF export and DEL realignment begins")
    run_stage4(
        workdir=workdir,
        input_csv=stage3_final_csv,
        bam_path=args.bamfilepath,
        fasta_path=args.fasta_path,
        output_vcf=output_path,
        processes=process_count,
    )
    emit("pipeline", f"pipeline completed in {format_duration(time.time() - started_at)}")
    return 0
