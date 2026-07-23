import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pysam

from ..utils.console import emit, emit_banner, format_duration


STAGE3_FINAL_CSV_BASENAME = "final_adaptive_validated.csv"
STAGE4_VCF_BASENAME = "stage4_final_adaptive_validated.vcf"
LEGACY_STAGE4_VCFGZ_BASENAME = "final_adaptive_validated.vcf.gz"
STAGE4_MARKER = "stage4_vcf_export.complete.json"
STAGE5_MARKER = "stage5_realignment.complete.json"


@dataclass(frozen=True)
class Stage4Paths:
    workdir: Path
    stage3_final_csv: Path
    stage4_vcf: Path
    final_vcf: Path
    stage4_marker: Path
    stage5_marker: Path


def build_stage4_paths(workdir, output_vcf):
    workdir_path = Path(workdir).expanduser().resolve()
    final_vcf_path = Path(output_vcf).expanduser().resolve()
    return Stage4Paths(
        workdir=workdir_path,
        stage3_final_csv=workdir_path / STAGE3_FINAL_CSV_BASENAME,
        stage4_vcf=workdir_path / STAGE4_VCF_BASENAME,
        final_vcf=final_vcf_path,
        stage4_marker=workdir_path / STAGE4_MARKER,
        stage5_marker=workdir_path / STAGE5_MARKER,
    )


def _write_marker(marker_path, payload):
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(marker_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return marker_path


def _marker_is_complete(marker_path, required_paths):
    return marker_path.exists() and all(Path(path).exists() for path in required_paths)


def _read_marker_output_csv(marker_path):
    marker_path = Path(marker_path)
    if not marker_path.exists():
        return None
    try:
        with open(marker_path) as handle:
            payload = json.load(handle)
    except Exception:
        return None

    output_csv = payload.get("output_csv")
    if not output_csv:
        return None
    output_path = Path(output_csv).expanduser().resolve()
    if output_path.exists():
        return output_path
    return None


def _normalize_chrom(chrom):
    return str(chrom).replace("chr", "")


def _get_ref_base(fasta, chrom, pos_0b):
    if not fasta:
        return "N"
    try:
        ref = fasta.fetch(chrom, pos_0b, pos_0b + 1)
        return ref.upper() if ref else "N"
    except Exception:
        return "N"


def _sort_dataframe(df):
    try:
        df["chrom_int"] = df["chrom"].astype(str).str.replace("X", "23").str.replace("Y", "24").astype(int)
        return df.sort_values(by=["chrom_int", "start"])
    except Exception:
        return df.sort_values(by=["chrom", "start"])


def export_stage3_csv_to_vcf(input_csv, output_vcf, ref_fasta_path, sample_name="SharpSV"):
    input_csv = Path(input_csv).expanduser().resolve()
    output_vcf = Path(output_vcf).expanduser().resolve()
    output_vcf.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()
    df = _sort_dataframe(df)

    header = pysam.VariantHeader()
    header.add_line("##fileformat=VCFv4.2")
    header.add_line(f"##fileDate={date.today().strftime('%Y%m%d')}")
    header.add_line("##source=SharpSV")
    header.add_meta("INFO", items=[("ID", "SVTYPE"), ("Number", 1), ("Type", "String"), ("Description", "Type of structural variant")])
    header.add_meta("INFO", items=[("ID", "SVLEN"), ("Number", 1), ("Type", "Integer"), ("Description", "Difference in length between REF and ALT alleles")])
    header.add_meta("INFO", items=[("ID", "END"), ("Number", 1), ("Type", "Integer"), ("Description", "End position")])
    header.add_meta("FORMAT", items=[("ID", "GT"), ("Number", 1), ("Type", "String"), ("Description", "Genotype")])

    for chrom in df["chrom"].unique():
        header.add_line(f"##contig=<ID={_normalize_chrom(chrom)}>")

    header.add_sample(sample_name)

    fasta = pysam.FastaFile(str(Path(ref_fasta_path).expanduser().resolve())) if ref_fasta_path and os.path.exists(ref_fasta_path) else None
    written = 0
    with pysam.VariantFile(str(output_vcf), "w", header=header) as vcf_out:
        for _, row in df.iterrows():
            try:
                label = int(row["label"])
            except Exception:
                continue
            if label not in {1, 2}:
                continue

            chrom = _normalize_chrom(row["chrom"])
            start_0b = int(row["start"])
            end_0b = int(row["end"])

            rec = vcf_out.new_record()
            rec.chrom = chrom
            rec.pos = start_0b
            rec.id = "."
            rec.qual = 999
            rec.filter.add("PASS")
            rec.ref = _get_ref_base(fasta, chrom, start_0b)

            if label == 1:
                rec.stop = end_0b
                rec.alts = ("<DEL>",)
                rec.info["SVTYPE"] = "DEL"
                rec.info["SVLEN"] = -(end_0b - start_0b)
            else:
                rec.stop = start_0b + 1
                ins_len = abs(end_0b - start_0b) or 50
                rec.alts = ("<INS>",)
                rec.info["SVTYPE"] = "INS"
                rec.info["SVLEN"] = ins_len

            rec.samples[sample_name]["GT"] = (0, 1)
            vcf_out.write(rec)
            written += 1

    if fasta is not None:
        fasta.close()
    return written


def refine_vcf_preserving_all_variants(input_vcf, bam_path, ref_path, output_vcf, processes=1, logger=None):
    if logger is None:
        logger = lambda message: None

    try:
        from .realign_vcf import BamStatCalculator, refine_del_records_parallel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Stage-4 DEL realignment requires the realignment dependencies to be installed, "
            f"but an import failed: {exc}"
        ) from exc

    logger(f"analyzing BAM stats from {bam_path}")
    stats = BamStatCalculator(bam_path)

    with pysam.VariantFile(str(input_vcf)) as vcf_scan:
        records = list(vcf_scan)
        header = vcf_scan.header.copy()

    del_tasks = []
    for idx, record in enumerate(records):
        is_del = (record.info.get("SVTYPE") == "DEL") or (record.alts and "<DEL>" in str(record.alts[0]))
        if is_del:
            del_tasks.append((idx, record.chrom, record.pos, record.stop))

    logger(
        f"dispatching DEL realignment with {max(1, int(processes or 1))} worker(s) "
        f"over {len(del_tasks):,} DEL record(s)"
    )
    refine_results = refine_del_records_parallel(
        del_tasks,
        bam_path=bam_path,
        ref_path=ref_path,
        insert_mean=stats.mean,
        insert_std=stats.std,
        processes=processes,
    )

    try:
        header.info.add("SPRITES_LEN", 1, "Integer", "Refined length by SpritesModel")
        header.info.add("EVIDENCE", 1, "String", "Evidence type: SR (SplitRead) or DP (DiscordantPair)")
        header.info.add("IMPRECISE", 0, "Flag", "Imprecise structural variation")
    except Exception:
        pass

    counts = {
        "total": 0,
        "written": 0,
        "non_del_passthrough": 0,
        "del_refined": 0,
        "del_no_evidence_kept": 0,
        "rescued_dp": 0,
    }

    with pysam.VariantFile(str(output_vcf), "w", header=header) as vcf_out:
        for idx, record in enumerate(records):
            counts["total"] += 1
            is_del = (record.info.get("SVTYPE") == "DEL") or (record.alts and "<DEL>" in str(record.alts[0]))

            if not is_del:
                vcf_out.write(record)
                counts["written"] += 1
                counts["non_del_passthrough"] += 1
                continue

            refined_len, method = refine_results.get(idx, (None, None))

            if refined_len is None:
                vcf_out.write(record)
                counts["written"] += 1
                counts["del_no_evidence_kept"] += 1
                continue

            record.info["SVLEN"] = -refined_len
            record.info["SPRITES_LEN"] = -refined_len
            record.stop = record.pos + refined_len
            record.info["EVIDENCE"] = method

            if method == "SR":
                if "IMPRECISE" in record.info:
                    del record.info["IMPRECISE"]
            elif method == "DP":
                record.info["IMPRECISE"] = True
                counts["rescued_dp"] += 1

            vcf_out.write(record)
            counts["written"] += 1
            counts["del_refined"] += 1

    return counts


def inspect_stage4_state(workdir, output_vcf):
    paths = build_stage4_paths(workdir, output_vcf)
    csv_candidates = [paths.stage3_final_csv]
    marker_output_csv = _read_marker_output_csv(paths.workdir / "stage3_validate.complete.json")
    if marker_output_csv is not None and marker_output_csv not in csv_candidates:
        csv_candidates.append(marker_output_csv)
    stage3_csv = next((path.resolve() for path in csv_candidates if path.exists()), None)
    return {
        "paths": paths,
        "stage3_csv": stage3_csv,
        "stage3_csv_ready": stage3_csv is not None,
        "stage4_complete": _marker_is_complete(paths.stage4_marker, [paths.stage4_vcf]),
        "stage5_complete": _marker_is_complete(paths.stage5_marker, [paths.final_vcf]),
    }


def run_stage4(workdir, input_csv, bam_path, fasta_path, output_vcf, processes=1):
    started_at = time.time()
    paths = build_stage4_paths(workdir, output_vcf)
    input_csv_path = Path(input_csv).expanduser().resolve()

    emit_banner(
        "Stage-4 VCF Finalization",
        details=[
            ("stage-3 csv", input_csv_path),
            ("stage-4 vcf", paths.stage4_vcf),
            ("final vcf", paths.final_vcf),
            ("realign workers", max(1, int(processes or 1))),
        ],
    )

    if _marker_is_complete(paths.stage5_marker, [paths.final_vcf]):
        emit("stage-4", f"reusing existing final SharpSV VCF: {paths.final_vcf}")
        return str(paths.final_vcf)

    if _marker_is_complete(paths.stage4_marker, [paths.stage4_vcf]):
        emit("stage-4/export", f"reusing intermediate VCF: {paths.stage4_vcf}")
    else:
        emit("stage-4/export", "converting final adaptive validation CSV into VCF")
        written = export_stage3_csv_to_vcf(
            input_csv=input_csv_path,
            output_vcf=paths.stage4_vcf,
            ref_fasta_path=fasta_path,
        )
        emit("stage-4/export", f"wrote {written:,} VCF records to {paths.stage4_vcf}")
        _write_marker(
            paths.stage4_marker,
            {
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_csv": str(input_csv_path),
                "output_vcf": str(paths.stage4_vcf),
            },
        )

    emit("stage-4/realign", "refining DEL breakpoints while preserving all variant types in the final VCF")
    counts = refine_vcf_preserving_all_variants(
        input_vcf=paths.stage4_vcf,
        bam_path=bam_path,
        ref_path=fasta_path,
        output_vcf=paths.final_vcf,
        processes=processes,
        logger=lambda message: emit("stage-4/realign", message),
    )
    emit(
        "stage-4/realign",
        "final VCF summary: "
        f"total={counts['total']:,}, written={counts['written']:,}, "
        f"passthrough_non_del={counts['non_del_passthrough']:,}, "
        f"del_refined={counts['del_refined']:,}, "
        f"del_kept_without_new_evidence={counts['del_no_evidence_kept']:,}, "
        f"rescued_dp={counts['rescued_dp']:,}",
    )
    _write_marker(
        paths.stage5_marker,
        {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_vcf": str(paths.stage4_vcf),
            "output_vcf": str(paths.final_vcf),
            "bam_path": str(Path(bam_path).expanduser().resolve()),
            "fasta_path": str(Path(fasta_path).expanduser().resolve()),
            "counts": counts,
        },
    )
    emit("stage-4", f"stage-4 completed in {format_duration(time.time() - started_at)}")
    return str(paths.final_vcf)
