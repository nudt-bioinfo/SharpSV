import argparse
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


def main(argv=None):
    args = build_parser().parse_args(argv)

    started_at = time.time()
    process_count = args.processes or available_worker_count()
    stage1_checkpoint, stage1_checkpoint_source = resolve_bundled_checkpoint(
        args.stage1_model_path,
        "stage-1",
    )
    stage2_checkpoint, stage2_checkpoint_source = resolve_bundled_checkpoint(
        args.stage2_model_path,
        "stage-2",
    )
    workdir = resolve_workdir(args.output, args.workdir)
    output_path = str(Path(args.output).expanduser().resolve())
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
